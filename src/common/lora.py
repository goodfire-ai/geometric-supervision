"""Low-rank adapters attached to the frozen model with forward hooks.

A LoRA on a linear layer adds `scaling * (x @ A.T) @ B.T` to that layer's
output. A is initialized like a PyTorch linear layer (Kaiming uniform) and B
starts at zero, so the adapter has no effect before training. The
adapter parameters are kept in float32 and the update is added to the
bfloat16 output; the adapter is never merged into the weights, so a model is
evaluated the same way it was trained.

Hooks leave the model itself untouched, so one loaded model can be switched
between the base model and any number of adapters:

    params = init_lora(model, layers=[4], module_names=MLP_MODULES, rank=1, seed=42)
    with attached(model, params, scaling=2.0):
        ...                                 # forward passes see the adapter
    save_adapter("adapter.safetensors", params, scaling=2.0)
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from common.model import decoder_layers

MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _linear_module(model: Any, layer: int, module_name: str) -> nn.Linear:
    block = decoder_layers(model)[layer]
    if module_name in MLP_MODULES:
        return getattr(block.mlp, module_name)
    if module_name in ATTENTION_MODULES:
        return getattr(block.self_attn, module_name)
    raise ValueError(f"unknown module {module_name!r}")


def init_lora(model: Any, layers: list[int], module_names: list[str], rank: int,
              seed: int) -> dict[tuple[int, str], tuple[nn.Parameter, nn.Parameter]]:
    """Fresh adapter parameters {(layer, module): (A, B)} on the model's device.

    The random draws come from one CPU generator seeded with `seed`, taken in
    (layer, module) order, so the initialization is identical across machines.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(model.parameters()).device
    params = {}
    for layer in layers:
        for module_name in module_names:
            linear = _linear_module(model, layer, module_name)
            a = torch.empty(rank, linear.in_features, dtype=torch.float32)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
            b = torch.zeros(linear.out_features, rank, dtype=torch.float32)
            params[(layer, module_name)] = (nn.Parameter(a.to(device)),
                                            nn.Parameter(b.to(device)))
    return params


def attach_lora(model: Any, params: dict, scaling: float):
    """Register the adapter hooks. Returns a function that removes them."""
    handles = []
    for (layer, module_name), (a, b) in params.items():
        linear = _linear_module(model, layer, module_name)

        def hook(module, inputs, output, a=a, b=b):
            x = inputs[0].to(a.dtype)
            delta = scaling * ((x @ a.t()) @ b.t())
            return output + delta.to(output.dtype)

        handles.append(linear.register_forward_hook(hook))

    def remove():
        for handle in handles:
            handle.remove()

    return remove


@contextmanager
def attached(model: Any, params: dict | None, scaling: float):
    """Context manager form of `attach_lora`; `params=None` means the base model."""
    remove = attach_lora(model, params, scaling) if params else None
    try:
        yield
    finally:
        if remove is not None:
            remove()


def trainable_tensors(params: dict) -> list[nn.Parameter]:
    return [tensor for pair in params.values() for tensor in pair]


def parameter_count(params: dict) -> int:
    return sum(tensor.numel() for tensor in trainable_tensors(params))


def save_adapter(path, params: dict, scaling: float) -> None:
    """Write {"L{layer}.{module}.A": A, "L{layer}.{module}.B": B} to a safetensors file."""
    from safetensors.torch import save_file
    tensors = {}
    for (layer, module_name), (a, b) in params.items():
        tensors[f"L{layer}.{module_name}.A"] = a.detach().float().cpu().contiguous()
        tensors[f"L{layer}.{module_name}.B"] = b.detach().float().cpu().contiguous()
    save_file(tensors, str(path), metadata={"scaling": repr(float(scaling))})


def load_adapter(path, device: str = "cuda") -> tuple[dict, float]:
    """Load an adapter saved by `save_adapter` as ({(layer, module): (A, B)}, scaling)."""
    from safetensors import safe_open
    params = {}
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        scaling = float(archive.metadata()["scaling"])
        for key in archive.keys():
            if not key.endswith(".A"):
                continue
            prefix = key[:-2]
            layer_text, module_name = prefix.split(".", 1)
            a = archive.get_tensor(key).to(device)
            b = archive.get_tensor(prefix + ".B").to(device)
            params[(int(layer_text[1:]), module_name)] = (a, b)
    return params, scaling
