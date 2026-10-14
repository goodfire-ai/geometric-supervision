"""Loading the frozen base model.

Every experiment uses the same bytes: Llama-3.1-8B at a pinned revision, in
bfloat16, with all parameters frozen.
"""
from __future__ import annotations

import os
from typing import Any

import torch

MODEL_NAME = "meta-llama/Llama-3.1-8B"
MODEL_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


def enable_determinism(strict: bool = False) -> None:
    """Use deterministic GPU kernels, so that rerunning a configuration on the same
    hardware gives the same result. With `strict=True`, an operation without a
    deterministic kernel raises an error instead of a warning.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def load_tokenizer(padding_side: str = "left") -> Any:
    """The tokenizer, padding with EOS. Left padding puts every prompt's last
    token at position -1, which is where the months task reads its answer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_base_model(device: str = "cuda") -> Any:
    """The frozen bfloat16 base model at the pinned revision."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16)
    if model.dtype != torch.bfloat16:
        raise RuntimeError(f"model loaded as {model.dtype}, expected bfloat16")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def decoder_layers(model: Any) -> Any:
    """The list of transformer blocks, whatever the wrapper class."""
    inner = model
    while not hasattr(inner, "embed_tokens"):
        inner = inner.model
    return inner.layers


def runtime_fingerprint() -> dict[str, Any]:
    """Versions and settings recorded with every result."""
    import transformers
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
    }
