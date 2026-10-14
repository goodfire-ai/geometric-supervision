"""Wrapping a document to a fixed width and labelling each token by character-count.

The task: take newline-free prose, insert a newline at the last word boundary that
keeps every line to at most ``width`` characters, and ask the model to predict those
newlines. The *character-count* of a token is how many characters have elapsed since
the previous newline: the variable whose helical representation we study and supervise.

Training, evaluation and harvesting all go through these functions, so they count
characters the same way.
"""
from __future__ import annotations

import textwrap

import numpy as np


def wrap(text: str, width: int, split_long_words: bool = False) -> str:
    """Insert newlines so every line is at most ``width`` characters.

    Uses the standard-library ``textwrap`` on each existing line (the documents we use
    have none). By default a word longer than ``width`` is kept whole and allowed to
    overflow, and hyphenated words are not split; this is the wrapping all results use.
    ``split_long_words=True`` hard-wraps long words instead.
    """
    kwargs = {} if split_long_words else dict(break_long_words=False, break_on_hyphens=False)
    wrapper = textwrap.TextWrapper(width=width, **kwargs)
    lines = []
    for line in text.splitlines():
        lines.extend([line] if line.strip() == "" else wrapper.wrap(line))
    return "\n".join(lines)


def previous_newline_index(text: str) -> np.ndarray:
    """For each character position, the index of the most recent ``\\n`` before it (-1 if none).

    The character-count at position ``i`` is then ``i - previous_newline_index[i]``.
    """
    prev_nl = np.full(len(text), -1, np.int64)
    last = -1
    for i, char in enumerate(text):
        prev_nl[i] = last
        if char == "\n":
            last = i
    return prev_nl


def token_character_counts(offsets: list[tuple[int, int]], prev_nl: np.ndarray,
                           max_count: int) -> np.ndarray:
    """Character-count label for each token, from its offset span and ``prev_nl``.

    A token occupying characters ``[s, e)`` is labelled by the count at its last
    character, ``(e - 1) - prev_nl[e - 1]``, clipped to ``max_count``. Zero-width
    tokens (special tokens) get 0.
    """
    counts = np.zeros(len(offsets), np.int64)
    for j, (s, e) in enumerate(offsets):
        if e > s:
            counts[j] = min((e - 1) - prev_nl[e - 1], max_count)
    return counts


def newline_token_ids(tokenizer) -> set[int]:
    """Ids of every vocabulary item whose decoded string contains a newline."""
    ids = list(range(len(tokenizer)))
    decoded = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
    return {i for i, s in zip(ids, decoded) if "\n" in s}
