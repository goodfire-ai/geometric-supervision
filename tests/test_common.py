import numpy as np
import pytest
import torch

from common import geometry
from common.lora import MLP_MODULES, attached, init_lora, load_adapter, save_adapter


@pytest.fixture(scope="module")
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2)
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


def test_lora_is_identity_at_init_and_detaches(tiny_llama, tmp_path):
    ids = torch.tensor([[1, 2, 3, 4]])
    base = tiny_llama(ids).logits
    params = init_lora(tiny_llama, [0, 1], MLP_MODULES, rank=2, seed=0)
    with attached(tiny_llama, params, scaling=2.0):
        assert torch.equal(tiny_llama(ids).logits, base)          # B = 0
        for _, b in params.values():
            torch.nn.init.normal_(b.data)
        assert not torch.allclose(tiny_llama(ids).logits, base)   # the update is applied
        trained = tiny_llama(ids).logits
    assert torch.equal(tiny_llama(ids).logits, base)              # hooks removed on exit

    save_adapter(tmp_path / "adapter.safetensors", params, scaling=2.0)
    loaded, scaling = load_adapter(tmp_path / "adapter.safetensors", device="cpu")
    assert scaling == 2.0 and loaded.keys() == params.keys()
    with attached(tiny_llama, loaded, scaling):
        assert torch.allclose(tiny_llama(ids).logits, trained)


def test_lora_init_is_seeded():
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    first = init_lora(model, [0], MLP_MODULES, rank=1, seed=3)
    second = init_lora(model, [0], MLP_MODULES, rank=1, seed=3)
    assert all(torch.equal(first[k][0], second[k][0]) for k in first)


def test_pooled_and_per_value_spread_differ_as_described():
    # two values with spreads 0.2 and 0.8 (in units of sigma0 = 1), tau = 0.5
    offsets = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    points = torch.cat([0.2 * offsets, 0.8 * offsets + torch.tensor([10.0, 0.0])])
    class_index = torch.tensor([0, 0, 1, 1])
    centroids = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    sigma0 = torch.tensor(1.0)
    pooled, _ = geometry.spread_term(points, class_index, centroids, sigma0, 0.5, "pooled")
    per_value, _ = geometry.spread_term(points, class_index, centroids, sigma0, 0.5, "per_value")
    assert abs(pooled.item() - (np.sqrt((0.04 + 0.64) / 2) - 0.5) ** 2) < 1e-6
    assert abs(per_value.item() - 0.09) < 1e-6


def test_procrustes_r2_is_invariant_to_similarity_transforms():
    rng = np.random.default_rng(0)
    ideal = rng.normal(size=(50, 3))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert geometry.procrustes_r2(3.0 * ideal @ rotation + 5.0, ideal) > 1 - 1e-9
    assert geometry.procrustes_r2(rng.permutation(ideal), ideal) < 0.2
