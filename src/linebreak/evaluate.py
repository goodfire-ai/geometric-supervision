"""Linebreaking error: how far the model's newline lands from the target width.

For each line (teacher-forced on the correctly wrapped document) we find the first
position where the model's argmax is a newline. Let ``L`` be the line length there and
``next_word`` the actual next word in the text. The *linebreaking error* is

    over_word = L + len(next_word) - width

so a value >= 0 is a valid break (the next word would have overflowed the width) and a
value < 0 is premature (that word would still have fit, wasting ``|over_word|`` chars).
We record one row per line; the mean of ``|over_word|`` is the headline L1 error, and
its sign gives the too-early rate.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import corpus, paths, wrapping

# By default Fig. 4 is scored on seed 1 of each LoRA at every 5th width from 20 to 300,
# which takes about a tenth of the time. The paper scored all three seeds at every width
# from 1 to 300 (run_linebreak.sh --all); the headline errors differ by about 0.05.
WIDTHS = list(range(20, 301, 5))
EVAL_SEEDS = [1]
PAPER_WIDTHS = list(range(1, 301))
PAPER_SEEDS = [1, 2, 3]
MAX_SEQ_LEN = 1536


def collect_breaks(model, tok, docs, width, newline_ids, max_seq_len, device):
    """First model newline per line: ``(line_index, L, word_len, tok_len, on_gold)``."""
    import torch

    rows = []
    for d in docs:
        text = wrapping.wrap(d, width)
        enc = tok(text, add_special_tokens=True, truncation=True,
                  max_length=max_seq_len, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if len(ids) < 4:
            continue
        prev_nl = wrapping.previous_newline_index(text)
        spans = list(re.finditer(r"\S+", text))
        word_start = np.array([m.start() for m in spans])
        word_len = np.array([m.end() - m.start() for m in spans])
        tok_start = np.array([o[0] for o in offs])
        with torch.inference_mode():
            preds = model(input_ids=torch.tensor([ids], device=device),
                          use_cache=False).logits[0, :-1].argmax(-1).cpu().tolist()
        pred_nl = [p in newline_ids for p in preds]
        gold_nl = [ids[t + 1] in newline_ids for t in range(len(ids) - 1)]
        line_index, found = 0, False
        for t in range(len(ids) - 1):
            if not found and pred_nl[t]:
                s, e = offs[t]
                if e > s:
                    L = (e - 1) - prev_nl[e - 1]
                    k = int(np.searchsorted(word_start, e, side="left"))  # the next word
                    if k < len(word_len):
                        wl = int(word_len[k])
                        j = int(np.searchsorted(tok_start, word_start[k], side="left"))
                        tl = int(offs[j][1] - offs[j][0]) if j < len(offs) else wl
                        rows.append((line_index, int(L), wl, tl, bool(gold_nl[t])))
                        found = True
            if gold_nl[t]:
                line_index += 1
                found = False
    return rows


def evaluate_model(adapter_dir, tag, docs, widths, max_seq_len):
    """Run the eval over ``widths`` -> ``eval/errors_<tag>.csv``.

    Resumable: widths already present in the CSV are skipped (and the model is not even
    loaded if there is nothing to do), and results are saved after every width, so a
    re-run only computes the missing widths.
    """
    from . import harvest

    out = paths.eval_dir() / f"errors_{tag}.csv"
    rows, done = [], set()
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {int(w) for w in prev.width.unique()}
    todo = [w for w in widths if w not in done]
    if not todo:
        print(f"  {tag}: all {len(widths)} widths already present -> skip", flush=True)
        return pd.DataFrame(rows)
    print(f"  {tag}: {len(done)} widths done, {len(todo)} to eval", flush=True)

    model, tok = harvest.load_model(adapter_dir)
    device = next(model.parameters()).device
    newline_ids = wrapping.newline_token_ids(tok)
    for w in todo:
        for (li, L, wl, tl, on_gold) in collect_breaks(model, tok, docs, w, newline_ids, max_seq_len, device):
            rows.append(dict(width=w, line_index=li, L=L, word_len=wl, tok_len=tl,
                             over_word=L + wl - w, over_tok=L + tl - w, on_gold=int(on_gold)))
        pd.DataFrame(rows).to_csv(out, index=False)  # incremental save (resumable)
    print(f"  saved {out} ({len(rows)} breaks total)", flush=True)
    return pd.DataFrame(rows)


def _model_units(configs_path, adapters_dir, seeds=None):
    """List the models to evaluate: ``(tag, adapter_dir_or_None)`` for base + each seed
    (only those in ``seeds`` when given)."""
    units = [("base", None)]
    for cfg in (json.loads(l) for l in configs_path.read_text().splitlines() if l.strip()):
        if cfg["method"] != "base":
            for seed in (s for s in cfg["seeds"] if seeds is None or s in seeds):
                units.append((f"{cfg['config_id']}_seed{seed}",
                              str(adapters_dir / cfg["config_id"] / f"seed-{seed}")))
    return units


def evaluate_all(configs_path, adapters_dir, num_docs, widths, max_seq_len, seeds=None):
    """Evaluate base + every ``(config, seed)`` adapter, one CSV per model (single process)."""
    docs = corpus.load_docs(paths.corpus_dir() / "eval_docs.txt")[:num_docs]
    for tag, adapter in _model_units(configs_path, adapters_dir, seeds):
        print(f"[eval] {tag}", flush=True)
        evaluate_model(adapter, tag, docs, widths, max_seq_len)


def _done_widths(tag):
    out = paths.eval_dir() / f"errors_{tag}.csv"
    if out.exists():
        return {int(w) for w in pd.read_csv(out, usecols=["width"]).width.unique()}
    return set()


def run_shard(configs_path, adapters_dir, num_docs, widths, max_seq_len, shard, num_shards, seeds=None):
    """One shard of a parallel eval: takes every ``num_shards``-th missing (model, width)
    unit and writes its rows to ``eval/_parts/part_<shard>.csv``. Merge afterwards."""
    import torch

    from . import harvest
    docs = corpus.load_docs(paths.corpus_dir() / "eval_docs.txt")[:num_docs]
    part_dir = paths.eval_dir() / "_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    part = part_dir / f"part_{shard:03d}.csv"
    cols = ["tag", "width", "line_index", "L", "word_len", "tok_len", "over_word", "over_tok", "on_gold"]
    rows = []
    for tag, adapter in _model_units(configs_path, adapters_dir, seeds):
        done = _done_widths(tag)
        todo = [w for w in widths if w not in done]
        mine = [w for j, w in enumerate(todo) if j % num_shards == shard]  # balanced per model
        if not mine:
            continue
        model, tok = harvest.load_model(adapter)
        device = next(model.parameters()).device
        newline_ids = wrapping.newline_token_ids(tok)
        for w in mine:
            for (li, L, wl, tl, on) in collect_breaks(model, tok, docs, w, newline_ids, max_seq_len, device):
                rows.append(dict(tag=tag, width=w, line_index=li, L=L, word_len=wl, tok_len=tl,
                                 over_word=L + wl - w, over_tok=L + tl - w, on_gold=int(on)))
            pd.DataFrame(rows).to_csv(part, index=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not rows:
        pd.DataFrame(columns=cols).to_csv(part, index=False)
    print(f"shard {shard}/{num_shards}: {len(rows)} breaks -> {part}", flush=True)


def merge_shards(configs_path, adapters_dir):
    """Fold all ``_parts/part_*.csv`` into the per-model ``errors_<tag>.csv`` (widths are
    disjoint across shards and vs. what was already there), then remove the parts."""
    part_dir = paths.eval_dir() / "_parts"
    parts = sorted(part_dir.glob("part_*.csv"))
    frames = [f for f in (pd.read_csv(p) for p in parts) if len(f)]
    allp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["tag"])
    for tag in sorted(set(allp["tag"])) if len(allp) else []:
        out = paths.eval_dir() / f"errors_{tag}.csv"
        new = allp[allp.tag == tag].drop(columns="tag")
        if out.exists():
            existing = pd.read_csv(out)
            new = new[~new.width.isin(set(existing.width))]  # widths are disjoint; keep existing's
            new = pd.concat([existing, new], ignore_index=True)
        new = new.sort_values(["width", "line_index"]).reset_index(drop=True)
        new.to_csv(out, index=False)
        print(f"  merged {tag}: {new.width.nunique()} widths, {len(new)} breaks", flush=True)
    for p in parts:
        p.unlink()


def summarize(df, li_min=1):
    """Headline scalars from a error CSV: L1 error, too-early rate, short-context L1."""
    d = df[df.line_index >= li_min]
    short = d[d.line_index <= 3]
    return dict(
        n=int(len(d)),
        l1=float(d.over_word.abs().mean()),
        pct_early=float(100 * (d.over_word < 0).mean()),
        mean_early=float(-d.over_word[d.over_word < 0].mean()) if (d.over_word < 0).any() else 0.0,
        l1_short_context=float(short.over_word.abs().mean()) if len(short) else float("nan"),
    )
