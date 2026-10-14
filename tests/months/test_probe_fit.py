import math

import numpy as np
import pytest
import torch

from months.probes import fit


def test_addition_prompts():
    prompts, first, second = fit.addition_prompts()
    assert len(prompts) == 199 * 199 == 39601
    assert prompts[0] == "1+1=" and prompts[-1] == "199+199="
    values = fit.variable_values(first, second)
    slots = fit.probe_slots(values)
    # a and b: periods 2..99 (98 each) x sin/cos; a+b: periods 2..150 (149) x sin/cos
    assert len(slots) == 2 * 98 * 2 + 149 * 2 == 690
    assert (slots[0].carrier, slots[0].period, slots[0].function) == ("input-month", 2, "sin")
    assert slots[196].carrier == "offset" and slots[392].carrier == "sum"


def test_split_matches_scikit_learn():
    sklearn_split = pytest.importorskip("sklearn.model_selection")
    n = 39601
    train_index, test_index = fit.train_test_indices(n)
    assert len(test_index) == math.ceil(0.2 * n) == 7921 and len(train_index) == 31680
    _, _, ref_train, ref_test = sklearn_split.train_test_split(
        np.zeros((n, 1)), np.arange(n), test_size=0.2, random_state=42)
    assert np.array_equal(train_index, ref_train)
    assert np.array_equal(test_index, ref_test)


def test_batched_fit_matches_one_at_a_time():
    """The stacked fit must reproduce the per-probe loop (same init, same Adam trajectory)."""
    rng = np.random.default_rng(0)
    n, hidden = 300, 12
    first = rng.integers(1, 40, size=n)
    second = rng.integers(1, 40, size=n)
    values = fit.variable_values(first, second)
    slots = fit.probe_slots(values, max_period=4)          # small: a,b periods 2..19 capped -> 2..4
    activations = torch.tensor(rng.normal(size=(n, hidden)), dtype=torch.float32)
    activations[:, 0] += torch.tensor(np.sin(2 * np.pi * values["sum"] / 3), dtype=torch.float32)
    batched = fit.fit_probes_batched(activations, slots, values, device="cpu", epochs=60)
    loop = fit.fit_probes_one_at_a_time(activations, slots, values, device="cpu", epochs=60)
    assert torch.equal(batched["trained"], loop["trained"])
    assert torch.allclose(batched["weight"], loop["weight"], atol=1e-6)
    assert torch.allclose(batched["bias"], loop["bias"], atol=1e-6)
    finite = ~torch.isnan(loop["r2"])
    assert torch.allclose(batched["r2"][finite], loop["r2"][finite], atol=1e-5)
    # the period-2 sine target is constant, so that probe is never trained
    untrained = [i for i, slot in enumerate(slots) if slot.period == 2 and slot.function == "sin"]
    assert untrained and not batched["trained"][untrained].any()


def test_planes_from_fit_pairs_cos_and_sin():
    rng = np.random.default_rng(0)
    first = rng.integers(1, 40, size=50)
    values = fit.variable_values(first, first)
    slots = fit.probe_slots(values, max_period=3)
    result = {"weight": torch.arange(len(slots) * 4, dtype=torch.float32).reshape(len(slots), 4),
              "bias": torch.arange(len(slots), dtype=torch.float32),
              "r2": torch.zeros(len(slots)), "trained": torch.ones(len(slots), dtype=torch.bool)}
    planes = fit.planes_from_fit(slots, result)
    entry = planes["sum__p3"]
    cos_index = slots.index(fit.ProbeSlot("sum", 3, "cos"))
    sin_index = slots.index(fit.ProbeSlot("sum", 3, "sin"))
    assert np.array_equal(entry["weight"][0], result["weight"][cos_index].numpy())
    assert np.array_equal(entry["weight"][1], result["weight"][sin_index].numpy())
