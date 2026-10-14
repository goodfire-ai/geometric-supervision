"""Loading the model and collecting layer activations by character-count.

Everything that needs hidden states goes through here: fitting the probes, the
per-count means, the point clouds for the helix-fit figures, and the per-layer probe
sweep. Character-count labels come from ``wrapping`` (the same definition training and
evaluation use), so a token is counted identically everywhere.

HF's ``hidden_states`` has one slot per layer plus the embeddings: slot 0 is the
embeddings and slot ``l + 1`` is the output of block ``l``. "Layer ``l``" everywhere in
this repo means the output of block ``l``, read at ``hidden_states[l + 1]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import corpus, paths, wrapping


def load_model(adapter_dir: str | None = None):
    """The frozen bf16 base model and a right-padding tokenizer, optionally with a trained
    adapter attached (by forward hooks; the adapter is not merged into the weights)."""
    import torch

    from common.lora import attach_lora, load_adapter
    from common.model import load_base_model, load_tokenizer

    tok = load_tokenizer(padding_side="right")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_base_model(device)
    model.config.use_cache = False
    if adapter_dir:
        params, scaling = load_adapter(Path(adapter_dir) / "adapter.safetensors", device=device)
        attach_lora(model, params, scaling)
        print(f"  attached adapter {adapter_dir}", flush=True)
    return model, tok


def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def collect(model, tok, docs, layers, width, max_count, min_count=1,
            batch_size=8, max_seq_len=1536, split_long_words=False, cap_per_count=None):
    """Collect per-token activations at ``layers`` for docs wrapped to ``width``.

    Returns ``(hiddens, counts)`` where ``hiddens[layer]`` is ``[N, 4096]`` float32 and
    ``counts`` is ``[N]`` int. With ``cap_per_count`` set, at most that many tokens are
    kept per count value (keeps the point clouds small without changing the geometry).
    """
    import torch

    device = next(model.parameters()).device
    per_layer = {L: [] for L in layers}
    all_counts = []
    kept_per_count: dict[int, int] = {}
    for chunk in _batches(docs, batch_size):
        wrapped = [wrapping.wrap(d, width, split_long_words) for d in chunk]
        encs = [tok(w, add_special_tokens=True, truncation=True, max_length=max_seq_len,
                    return_offsets_mapping=True) for w in wrapped]
        width_max = max(len(e["input_ids"]) for e in encs)
        input_ids = torch.full((len(encs), width_max), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(encs), width_max), dtype=torch.long)
        for j, e in enumerate(encs):
            n = len(e["input_ids"])
            input_ids[j, :n] = torch.tensor(e["input_ids"])
            attn[j, :n] = 1
        with torch.no_grad():
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                        output_hidden_states=True, use_cache=False)
        for j, e in enumerate(encs):
            n = len(e["input_ids"])
            prev_nl = wrapping.previous_newline_index(wrapped[j])
            counts = wrapping.token_character_counts(e["offset_mapping"], prev_nl, max_count)
            valid = counts >= min_count
            if cap_per_count is not None:
                for k in np.nonzero(valid)[0]:
                    c = int(counts[k])
                    if kept_per_count.get(c, 0) >= cap_per_count:
                        valid[k] = False
                    else:
                        kept_per_count[c] = kept_per_count.get(c, 0) + 1
            idx = np.nonzero(valid)[0]
            if idx.size == 0:
                continue
            all_counts.append(counts[idx])
            for L in layers:
                h = out.hidden_states[L + 1][j, :n].float().cpu().numpy()
                per_layer[L].append(h[idx])
    counts = np.concatenate(all_counts) if all_counts else np.zeros(0, np.int64)
    hiddens = {L: (np.concatenate(v) if v else np.zeros((0, 4096), np.float32)) for L, v in per_layer.items()}
    return hiddens, counts


def layer_counts(model, tok, docs, layer, width, seed=1, **kw):
    """Convenience: per-token ``(H, counts)`` at a single ``layer`` (used to fit the probes)."""
    import torch
    torch.manual_seed(seed)
    hiddens, counts = collect(model, tok, docs, [layer], width, max_count=width, **kw)
    return hiddens[layer], counts


def per_count_means(hiddens, counts, max_count):
    """Mean activation for each count in ``[1, max_count]`` (NaN row where a count is absent)."""
    means = np.full((max_count + 1, hiddens.shape[1]), np.nan, np.float32)
    for c in range(1, max_count + 1):
        sel = counts == c
        if sel.any():
            means[c] = hiddens[sel].mean(0)
    return means


def run_means(num_docs: int, max_count: int) -> None:
    layer = paths.COUNT_LAYER
    docs = corpus.load_docs(paths.corpus_dir() / "diag_docs.txt")[:num_docs]
    model, tok = load_model()
    # width-150 for the in-range means; wider widths to reach high counts for the to-300 means.
    width = 150 if max_count <= 150 else max_count
    hiddens, counts = collect(model, tok, docs, [layer], width=width, max_count=max_count)
    means = per_count_means(hiddens[layer], counts, max_count)
    tag = "" if max_count <= 150 else "_to300"
    out = paths.manifold_dir() / f"mean_hiddens_layer{layer}{tag}.npy"
    np.save(out, means)
    print(f"  saved {out} ({np.isfinite(means[:, 0]).sum()} counts present)", flush=True)


def run_cloud(adapter_dir: str | None, tag: str, num_docs: int, max_count: int, cap: int) -> None:
    layer = paths.COUNT_LAYER
    docs = corpus.load_docs(paths.corpus_dir() / "eval_docs.txt")[:num_docs]
    model, tok = load_model(adapter_dir)
    width = 150 if max_count <= 150 else max_count
    hiddens, counts = collect(model, tok, docs, [layer], width=width, max_count=max_count, cap_per_count=cap)
    out = paths.probes_dir() / f"cloud_layer{layer}_{tag}.npz"
    np.savez(out, H=hiddens[layer].astype(np.float16), count=counts.astype(np.int16))
    print(f"  saved {out} ({hiddens[layer].shape[0]} tokens)", flush=True)


def run_probe_r2_by_layer(num_docs: int, cmin: int = 20, cmax: int = 150,
                          max_tokens: int = 25000) -> None:
    """Fig. 5a: character-count probe R^2 at every layer.

    Rows are written by layer (-1 = the embeddings, 0..31 = block outputs); the count
    representation peaks at layer 5, where the geometry LoRA is supervised. The ridge
    penalty is tuned per layer on a validation split, using many tokens with no
    per-count cap, so the curve reaches its plateau rather than an under-fit dip.
    """
    import pandas as pd
    import torch

    from .probes import ridge_fit

    docs = corpus.load_docs(paths.corpus_dir() / "diag_docs.txt")[:num_docs]
    model, tok = load_model()
    device = next(model.parameters()).device
    # hidden_states has num_layers+1 slots: slot 0 = embeddings, slot i+1 = output of layer i.
    # Collect all of them so every one of the 32 layers (0..31) appears in the sweep.
    n_slots = model.config.num_hidden_layers + 1

    per_slot: dict[int, list] = {s: [] for s in range(n_slots)}
    all_counts: list = []
    for chunk in _batches(docs, 8):
        wrapped = [wrapping.wrap(d, 150) for d in chunk]  # no-split, as everywhere
        encs = [tok(w, add_special_tokens=True, truncation=True, max_length=1536,
                    return_offsets_mapping=True) for w in wrapped]
        width_max = max(len(e["input_ids"]) for e in encs)
        input_ids = torch.full((len(encs), width_max), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(encs), width_max), dtype=torch.long)
        for j, e in enumerate(encs):
            n = len(e["input_ids"])
            input_ids[j, :n] = torch.tensor(e["input_ids"])
            attn[j, :n] = 1
        with torch.no_grad():
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                        output_hidden_states=True, use_cache=False)
        for j, e in enumerate(encs):
            n = len(e["input_ids"])
            prev_nl = wrapping.previous_newline_index(wrapped[j])
            counts = wrapping.token_character_counts(e["offset_mapping"], prev_nl, cmax)
            idx = np.nonzero((counts >= cmin) & (counts <= cmax))[0]
            if idx.size == 0:
                continue
            all_counts.append(counts[idx])
            for s in range(n_slots):
                per_slot[s].append(out.hidden_states[s][j, :n].half().cpu().numpy()[idx])
        if sum(len(c) for c in all_counts) >= max_tokens:
            break

    counts = np.concatenate(all_counts)
    sel = np.arange(len(counts))
    if len(counts) > max_tokens:
        sel = np.random.default_rng(1).choice(len(counts), max_tokens, replace=False)
    counts = counts[sel]
    ct = torch.as_tensor(counts, dtype=torch.float32, device=device)
    perm = np.random.default_rng(0).permutation(len(ct))
    ntr, nva = int(0.7 * len(perm)), int(0.15 * len(perm))
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
    print(f"  {len(ct)} tokens (counts {cmin}-{cmax}) at {n_slots} hidden-state slots", flush=True)

    def r2(y, yhat):
        return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    rows = []
    for s in range(n_slots):
        X = torch.as_tensor(np.concatenate(per_slot[s])[sel], dtype=torch.float32, device=device)
        best = None
        for lam in (1.0, 10.0, 100.0, 1e3, 1e4):
            beta = ridge_fit(X[tr], ct[tr, None], lam)  # raw (unstandardised) ridge
            pv = (torch.cat([X[va], torch.ones(len(va), 1, device=device)], 1) @ beta).squeeze(1)
            r2v = r2(ct[va], pv)
            if best is None or r2v > best[0]:
                best = (r2v, lam, beta)
        _, lam, beta = best
        pte = (torch.cat([X[te], torch.ones(len(te), 1, device=device)], 1) @ beta).squeeze(1)
        rows.append(dict(layer=s - 1, r2=r2(ct[te], pte), lam=lam))
        print(f"  layer {s - 1:2d}: R^2={rows[-1]['r2']:.3f} (lam={lam:g})", flush=True)
    out = paths.probes_dir() / "probe_r2_by_layer.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    peak = max(rows, key=lambda r: r["r2"])
    print(f"  saved {out} | peak at layer {peak['layer']} R^2={peak['r2']:.3f}", flush=True)
