"""Train one adapter from one config and seed.

Two methods share an identical LoRA placement, optimiser, data and step count, so the
only difference is the loss:

    task      : next-token cross-entropy on the wrapped text (the standard baseline)
    geometry  : the geometry objective of ``losses.py`` on the layer-5 count readout,
                with no output-level loss at all (``ce_weight = 0``)

Everything else (rank-1 LoRA on all projections of blocks 0-5, AdamW at 1e-4, 400
steps of batch 4 over the FineWeb train pool wrapped to random widths in [20, 150]) is
held fixed between them, so a difference in behaviour is a difference in objective.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths

MAX_SEQ_LEN = 1024
REQUIRED_KEYS = ("config_id", "method", "seeds")


# --------------------------------------------------------------------------- #
#  Batching
# --------------------------------------------------------------------------- #
def make_batch(docs, tok, wmin, wmax, rng, max_count, split_long_words, max_seq_len=MAX_SEQ_LEN):
    """Wrap each doc to a random width in ``[wmin, wmax]`` and label tokens by count."""
    import torch

    from . import wrapping

    ids_list, cnt_list = [], []
    for d in docs:
        width = int(rng.integers(wmin, wmax + 1))
        wrapped = wrapping.wrap(d, width, split_long_words)
        prev_nl = wrapping.previous_newline_index(wrapped)
        enc = tok(wrapped, add_special_tokens=True, truncation=True,
                  max_length=max_seq_len, return_offsets_mapping=True)
        ids_list.append(enc["input_ids"])
        cnt_list.append(wrapping.token_character_counts(enc["offset_mapping"], prev_nl, max_count))
    width_max = max(len(x) for x in ids_list)
    input_ids = torch.full((len(ids_list), width_max), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), width_max), dtype=torch.long)
    count = torch.zeros((len(ids_list), width_max), dtype=torch.long)
    for i, (ids, cnt) in enumerate(zip(ids_list, cnt_list)):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attn[i, :len(ids)] = 1
        count[i, :len(cnt)] = torch.tensor(cnt)
    return input_ids, attn, count


# --------------------------------------------------------------------------- #
#  Config handling
# --------------------------------------------------------------------------- #
def check_config(cfg: dict) -> None:
    """Validate a config dict without loading a model (used by the tests and before training)."""
    for k in REQUIRED_KEYS:
        if k not in cfg:
            raise ValueError(f"config {cfg.get('config_id', '?')} missing key {k!r}")
    method = cfg["method"]
    if method not in ("base", "task", "geometry"):
        raise ValueError(f"{cfg['config_id']}: unknown method {method!r}")
    if method == "geometry":
        from .targets import MODES
        if cfg.get("target", "helix") not in MODES[1:]:
            raise ValueError(f"{cfg['config_id']}: geometry target must be one of {MODES[1:]}")
        if float(cfg.get("alpha_r", 0)) <= 0:
            raise ValueError(f"{cfg['config_id']}: geometry needs alpha_r > 0")
    lo, hi = cfg.get("edit_layers", [0, 5])
    if not (0 <= lo <= hi):
        raise ValueError(f"{cfg['config_id']}: bad edit_layers {cfg.get('edit_layers')}")


def _defaults(cfg: dict) -> dict:
    """Fill a config with the defaults so the trainer reads a complete spec.

    ``spread_weight`` and ``spread_tau`` are the paper's lambda and tau; ``geometry_weight``
    and ``ce_weight`` switch the geometry and cross-entropy objectives on or off.
    """
    geometry = cfg["method"] == "geometry"
    d = dict(
        target="helix" if geometry else "none",
        edit_layers=[0, 5], rank=1, scaling=2.0, target_modules="all",
        lr=1e-4, steps=400, batch_size=4, wmin=20, wmax=150,
        max_count=150, min_count=1, num_train_docs=3000, max_seq_len=MAX_SEQ_LEN,
        split_long_words=False,  # never break a word across lines
        geom_layer=paths.COUNT_LAYER,
        alpha_r=1.2, spread_tau=0.5, spread_weight=1.0,
        geometry_weight=1.0 if geometry else 0.0,
        ce_weight=1.0 if cfg["method"] == "task" else 0.0,
    )
    d.update(cfg)
    return d


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def run(cfg: dict, seed: int, out_dir: Path) -> dict:
    """Train one adapter and save it to ``out_dir``. Returns a small result dict."""
    import torch
    import torch.nn.functional as F

    from common.lora import ATTENTION_MODULES, MLP_MODULES, attached, init_lora, save_adapter, trainable_tensors
    from common.model import enable_determinism

    from . import corpus, harvest, losses, targets
    from .probes import HelixFrame

    cfg = _defaults(cfg)
    check_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    enable_determinism()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model, tok = harvest.load_model()
    lo, hi = cfg["edit_layers"]
    modules = {"all": ATTENTION_MODULES + MLP_MODULES, "attn": ATTENTION_MODULES,
               "mlp": MLP_MODULES}[cfg["target_modules"]]
    params = init_lora(model, list(range(lo, hi + 1)), modules, int(cfg["rank"]), seed)
    scaling = float(cfg["scaling"])
    tensors = trainable_tensors(params)
    print(f"  {cfg['config_id']} seed {seed}: {sum(t.numel() for t in tensors):,} LoRA params", flush=True)

    device = next(model.parameters()).device
    max_count, min_count = cfg["max_count"], cfg["min_count"]
    geometry_weight, ce_weight = float(cfg["geometry_weight"]), float(cfg["ce_weight"])
    slot = cfg["geom_layer"] + 1  # hidden_states[l + 1] is the output of block l

    docs = corpus.load_docs(paths.corpus_dir() / "train_docs.txt")[:cfg["num_train_docs"]]

    # Fix the targets on the base model, from the readout of a few batches of tokens.
    frozen = None
    if geometry_weight > 0:
        P, b, lookup = targets.build(cfg["target"], HelixFrame.load(), max_count, device)
        points, counts = [], []
        with torch.no_grad():
            for batch_docs in _prepass_batches(docs, cfg["batch_size"]):
                ids, attn, count = make_batch(batch_docs, tok, cfg["wmin"], cfg["wmax"], rng, max_count,
                                              cfg["split_long_words"], cfg["max_seq_len"])
                hidden = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                               output_hidden_states=True, use_cache=False).hidden_states[slot]
                mask = attn.bool() & (count >= min_count)
                points.append((hidden.float() @ P.T + b)[mask.to(device)].cpu())
                counts.append(count[mask])
        frozen = losses.freeze_targets(torch.cat(points).double().numpy(), torch.cat(counts).numpy(),
                                       lookup, float(cfg["alpha_r"]),
                                       literal=cfg["target"] == "count_regression")
        print(f"  froze targets on {int(frozen['has_target'].sum())} counts | "
              f"r0={float(frozen['r0']):.3f} sigma0={float(frozen['sigma0']):.3f}", flush=True)

    opt = torch.optim.AdamW(tensors, lr=cfg["lr"])
    log, ptr = [], 0
    with attached(model, params, scaling):
        for step in range(cfg["steps"]):
            if ptr + cfg["batch_size"] > len(docs):
                ptr = 0
                rng.shuffle(docs)
            batch = docs[ptr:ptr + cfg["batch_size"]]
            ptr += cfg["batch_size"]
            input_ids, attn, count = make_batch(batch, tok, cfg["wmin"], cfg["wmax"], rng,
                                                max_count, cfg["split_long_words"], cfg["max_seq_len"])
            input_ids, attn, count = input_ids.to(device), attn.to(device), count.to(device)
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=frozen is not None, use_cache=False)

            logits = out.logits[:, :-1].float()
            labels = input_ids[:, 1:]
            valid = attn[:, 1:].bool() & attn[:, :-1].bool()
            lm = F.cross_entropy(logits[valid], labels[valid])

            gloss = torch.zeros((), device=device)
            if frozen is not None:
                proj = out.hidden_states[slot].float() @ P.T + b
                mask = attn.bool() & (count >= min_count)
                gloss = losses.geometry_loss(proj[mask], count[mask], frozen,
                                             float(cfg["spread_tau"]), float(cfg["spread_weight"]))

            loss = ce_weight * lm + geometry_weight * gloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == cfg["steps"] - 1:
                log.append((step, float(lm), float(gloss), float(loss)))
                print(f"  step {step:4d} | lm {float(lm):.4f} | geom {float(gloss):.4f} | "
                      f"total {float(loss):.4f}", flush=True)

    save_adapter(out_dir / "adapter.safetensors", params, scaling)
    np.save(out_dir / "loss_log.npy", np.array(log))
    (out_dir / "config.json").write_text(json.dumps({**cfg, "seed": seed}, indent=2))
    print(f"  saved adapter -> {out_dir}", flush=True)
    return dict(config_id=cfg["config_id"], seed=seed, final_loss=log[-1][3] if log else None)


def _prepass_batches(docs, batch_size, n_batches=8):
    for bi in range(0, min(n_batches * batch_size, len(docs)), batch_size):
        yield docs[bi:bi + batch_size]
