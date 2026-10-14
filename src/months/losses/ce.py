"""Task loss: cross-entropy on the correct month token at the answer position."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

MICROBATCH = 64


def cross_entropy_on_answer(model: Any, tokenizer: Any, prompts: list[str],
                            answer_indices: np.ndarray, month_ids: list[int]) -> float:
    """Full-vocabulary cross-entropy of the correct month, averaged over prompts.

    Runs the prompts in microbatches and calls backward on each, weighted so
    the accumulated gradient equals that of the mean loss over all prompts.
    Returns the mean loss as a float (the gradients are already in .grad).
    """
    device = model.device
    n_prompts = len(prompts)
    month_ids_tensor = torch.tensor(month_ids, device=device)
    total = 0.0
    for start in range(0, n_prompts, MICROBATCH):
        chunk = prompts[start:start + MICROBATCH]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        logits = model(input_ids=encoded.input_ids,
                       attention_mask=encoded.attention_mask).logits[:, -1, :].float()
        answers = torch.as_tensor(answer_indices[start:start + len(chunk)], device=device)
        targets = month_ids_tensor[answers]
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum") / n_prompts
        loss.backward()
        total += float(loss.detach())
    return total
