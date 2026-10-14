"""The months task: prompts, variables, and accuracy bands.

A prompt asks "What month is {offset} months after {input}?". The three task
variables are the input month (1..12), the offset (a positive integer written in
words), and their sum. The correct answer is the month whose 1-based index is
`sum mod 12`, so it depends only on the sum.

The training set is every prompt with sum <= 24 (210 prompts). The evaluation
grid is every prompt with sum <= 96 (1074 prompts), sorted by prompt text so
that saved predictions line up across runs.
"""
from __future__ import annotations

from typing import Any

import numpy as np

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

TEMPLATE = "Q: What month is {offset} months after {input}?\nA:"

TRAIN_MAX_SUM = 24
EVAL_MAX_SUM = 96

# Accuracy bands reported everywhere: in range (the training range), near out of
# range, far out of range, and two pooled bands. Bands past 96 only apply when a
# wider evaluation grid is used (the training-range ladder).
BANDS = {
    "2-24": (2, 24),
    "25-48": (25, 48),
    "49-96": (49, 96),
    "25-96": (25, 96),
    "2-96": (2, 96),
    "97-132": (97, 132),
    "2-132": (2, 132),
}

CARRIERS = ("input-month", "offset", "sum")


# ---------------------------------------------------------------------------
# number words
# ---------------------------------------------------------------------------

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """American English number words without "and": 105 -> "one hundred five".

    Valid for 1..999. The offsets in the prompts are written this way.
    """
    if not 1 <= n <= 999:
        raise ValueError(f"number_to_words handles 1..999, got {n}")
    if n < 100:
        return _under_one_hundred(n)
    hundreds, remainder = divmod(n, 100)
    words = _ONES[hundreds] + " hundred"
    if remainder:
        words += " " + _under_one_hundred(remainder)
    return words


def _under_one_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return _TENS[tens] + "-" + _ONES[ones]


# ---------------------------------------------------------------------------
# prompt grids
# ---------------------------------------------------------------------------

def build_grid(min_sum: int, max_sum: int) -> list[dict]:
    """Every (input month, offset) pair whose sum lies in [min_sum, max_sum].

    Rows are ordered by sum, then by input month. Each row records:
        sum           input_index + offset
        input_index   1..12
        input_month   month name
        offset        positive integer
        offset_words  the offset written in words, as it appears in the prompt
        answer_index  0-based index of the correct month, (sum - 1) mod 12
        answer_month  the correct month name
    """
    rows = []
    for total in range(min_sum, max_sum + 1):
        for input_index in range(1, 13):
            offset = total - input_index
            if offset < 1:
                continue
            answer_index = (total - 1) % 12
            rows.append({
                "sum": total,
                "input_index": input_index,
                "input_month": MONTHS[input_index - 1],
                "offset": offset,
                "offset_words": number_to_words(offset),
                "answer_index": answer_index,
                "answer_month": MONTHS[answer_index],
            })
    return rows


def render_prompt(row: dict) -> str:
    return TEMPLATE.format(offset=row["offset_words"], input=row["input_month"])


def training_grid(max_sum: int = TRAIN_MAX_SUM) -> tuple[list[dict], list[str]]:
    """The training prompts (sum 2..max_sum), in grid order."""
    rows = build_grid(2, max_sum)
    if max_sum == TRAIN_MAX_SUM and len(rows) != 210:
        raise RuntimeError(f"training grid has {len(rows)} prompts, expected 210")
    return rows, [render_prompt(row) for row in rows]


def evaluation_grid(max_sum: int = EVAL_MAX_SUM) -> tuple[list[dict], list[str]]:
    """The evaluation prompts (sum 2..max_sum), sorted by prompt text.

    Sorting fixes the row order of every saved predictions file, so predictions
    from different runs can be compared position by position.
    """
    rows = build_grid(2, max_sum)
    if max_sum == EVAL_MAX_SUM and len(rows) != 1074:
        raise RuntimeError(f"evaluation grid has {len(rows)} prompts, expected 1074")
    rows.sort(key=render_prompt)
    return rows, [render_prompt(row) for row in rows]


def carrier_values(carrier: str, rows: list[dict]) -> np.ndarray:
    """The value of a task variable for each row, as the probes expect it.

    The input month is 0-based here (January = 0) because that is how the
    probe planes were labeled; the offset and the sum are the literal integers.
    """
    if carrier == "input-month":
        return np.asarray([row["input_index"] - 1 for row in rows], dtype=np.int64)
    if carrier == "offset":
        return np.asarray([row["offset"] for row in rows], dtype=np.int64)
    if carrier == "sum":
        return np.asarray([row["sum"] for row in rows], dtype=np.int64)
    raise ValueError(f"unknown carrier {carrier!r}; expected one of {CARRIERS}")


def answers(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["answer_index"] for row in rows], dtype=np.int64)


def sums(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["sum"] for row in rows], dtype=np.int64)


# ---------------------------------------------------------------------------
# accuracy bands
# ---------------------------------------------------------------------------

def band_summary(predictions: np.ndarray, correct: np.ndarray,
                 sum_values: np.ndarray) -> dict:
    """Accuracy per band and per sum value.

    Returns {"bands": {name: {n, correct, accuracy}}, "per_sum": {sum: {n, correct}}}.
    Bands with no prompts in this grid are left out.
    """
    hit = predictions == correct
    bands = {}
    for name, (low, high) in BANDS.items():
        selected = (sum_values >= low) & (sum_values <= high)
        n = int(selected.sum())
        if n == 0:
            continue
        n_correct = int(hit[selected].sum())
        bands[name] = {"n": n, "correct": n_correct, "accuracy": n_correct / n}
    per_sum = {}
    for value in sorted(set(sum_values.tolist())):
        selected = sum_values == value
        per_sum[str(int(value))] = {"n": int(selected.sum()),
                                    "correct": int(hit[selected].sum())}
    return {"bands": bands, "per_sum": per_sum}


def accuracy_in_range(per_sum: dict, low: int, high: int) -> float:
    """Accuracy over sums in [low, high] from a per_sum table (pooled counts)."""
    n = sum(entry["n"] for value, entry in per_sum.items() if low <= int(value) <= high)
    n_correct = sum(entry["correct"] for value, entry in per_sum.items()
                    if low <= int(value) <= high)
    return n_correct / n if n else float("nan")


def month_token_ids(tokenizer: Any) -> list[int]:
    """Token ids of " January" ... " December".

    Scoring restricts the argmax to these twelve tokens, which assumes each
    month name with a leading space is a single token. That is checked here.
    """
    ids = []
    for name in MONTHS:
        encoded = tokenizer.encode(f" {name}", add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"' {name}' is {len(encoded)} tokens; expected one")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 12:
        raise RuntimeError("month token ids are not distinct")
    return ids
