"""Fitting Fourier probes the way Feucht et al. did.

The probes are trained on a plain addition task. Prompts are "{a}+{b}=" for
every a, b in 1..199 (39,601 prompts, each exactly five tokens including the
beginning-of-sequence token, so no padding is involved). For each layer and
capture site, the activation at the last token ("=") is the input, and one
linear probe (a `nn.Linear(hidden, 1)`) is fit per target:

    variable     a (the first operand), b (the second), or a+b
    period T     2 .. min(max(variable) // 2, 150), i.e. 2..99 for a and b and
                 2..150 for a+b
    function     sin(2*pi*variable/T) and cos(2*pi*variable/T)

Each probe is trained with full-batch Adam (lr 1e-3, 500 epochs, mean squared
error) on an 80% split of the prompts and scored by R^2 on the other 20%. A
probe whose target is constant on the training split (the period-2 sine, which
is zero at every integer) is left at its initialization.

Two details make the fit match the probes of Feucht et al.:

  * The initialization is deterministic. Feucht et al. seeded torch once per
    (layer, site) and created probes in a fixed order (variables a, b, a+b;
    periods ascending; sin before cos), so each probe's initial weights are a
    specific draw from that stream. This module reproduces that stream
    on CPU and then trains all probes of a layer at once.

  * Training all probes together is equivalent to training them one at a time.
    A stacked `Linear(hidden, K)` whose loss is the SUM over probes of each
    probe's mean squared error has the same per-probe gradients, and Adam
    updates every parameter independently, so the trajectories are the same up
    to floating-point reduction order. This turns 690 separate 500-step fits
    into one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

OPERAND_MIN, OPERAND_MAX = 1, 199
PROMPT_TEMPLATE = "{a}+{b}="

LEARNING_RATE = 1e-3
EPOCHS = 500
TEST_FRACTION = 0.2
SPLIT_SEED = 42
INIT_SEED = 42
MAX_PERIOD = 150
CONSTANT_TARGET_TOLERANCE = 1e-6

# The order Feucht et al. create probes in, which fixes the random initialization.
VARIABLE_ORDER = ["input-month", "offset", "sum"]   # a, b, a+b in probe-file terms


def addition_prompts() -> tuple[list[str], np.ndarray, np.ndarray]:
    """All "{a}+{b}=" prompts for a, b in 1..199, a in the outer loop."""
    prompts, first, second = [], [], []
    for a in range(OPERAND_MIN, OPERAND_MAX + 1):
        for b in range(OPERAND_MIN, OPERAND_MAX + 1):
            prompts.append(PROMPT_TEMPLATE.format(a=a, b=b))
            first.append(a)
            second.append(b)
    return prompts, np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


def variable_values(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    return {"input-month": first, "offset": second, "sum": first + second}


def train_test_indices(n: int, test_fraction: float = TEST_FRACTION,
                       seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """The same split as scikit-learn's train_test_split(test_size, random_state).

    scikit-learn draws one permutation from RandomState(seed); the first
    ceil(test_fraction * n) indices are the test set and the rest the training
    set. Reimplemented so the split is defined without the dependency (the test
    suite checks it against scikit-learn when that is installed).
    """
    n_test = int(math.ceil(test_fraction * n))
    permutation = np.random.RandomState(seed).permutation(n)
    return permutation[n_test:], permutation[:n_test]


@dataclass(frozen=True)
class ProbeSlot:
    carrier: str
    period: int
    function: str      # "sin" or "cos"


def probe_slots(values: dict[str, np.ndarray], max_period: int = MAX_PERIOD) -> list[ProbeSlot]:
    """Every probe of one (layer, site), in initialization order."""
    slots = []
    for carrier in VARIABLE_ORDER:
        top_period = min(int(values[carrier].max()) // 2, max_period)
        for period in range(2, top_period + 1):
            for function in ("sin", "cos"):
                slots.append(ProbeSlot(carrier, period, function))
    return slots


def initial_parameters(n_slots: int, hidden: int, seed: int = INIT_SEED
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Initial weights [K, hidden] and biases [K], one nn.Linear draw per slot in order."""
    torch.manual_seed(seed)
    weights, biases = [], []
    for _ in range(n_slots):
        linear = nn.Linear(hidden, 1)
        weights.append(linear.weight.detach().clone().reshape(-1))
        biases.append(linear.bias.detach().clone().reshape(()))
    return torch.stack(weights), torch.stack(biases)


def targets_for(slots: list[ProbeSlot], values: dict[str, np.ndarray],
                indices: np.ndarray) -> np.ndarray:
    """Target matrix [len(indices), K] of sin/cos(2*pi*value/period), float32."""
    columns = []
    for slot in slots:
        angle = 2 * np.pi * values[slot.carrier][indices] / slot.period
        columns.append(np.sin(angle) if slot.function == "sin" else np.cos(angle))
    return np.stack(columns, axis=1).astype(np.float32)


def r_squared(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-column R^2; NaN where the target has no variance."""
    residual = ((target - prediction) ** 2).sum(dim=0)
    total = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = 1 - residual / total
    return torch.where(total > 1e-8, r2, torch.full_like(r2, float("nan")))


def fit_probes_batched(activations: torch.Tensor, slots: list[ProbeSlot],
                       values: dict[str, np.ndarray], device: str = "cuda",
                       epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE,
                       init_seed: int = INIT_SEED) -> dict:
    """Fit every probe of one (layer, site) at once.

    activations   [n_prompts, hidden] float32 (CPU or GPU)
    Returns {"weight": [K, hidden], "bias": [K], "r2": [K], "trained": [K] bool}
    as CPU tensors, in slot order.
    """
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("TF32 must be off for the probe fit to be reproducible")
    train_index, test_index = train_test_indices(activations.shape[0])
    x_train = activations[train_index].to(device, torch.float32)
    x_test = activations[test_index].to(device, torch.float32)
    y_train = torch.as_tensor(targets_for(slots, values, train_index), device=device)
    y_test = torch.as_tensor(targets_for(slots, values, test_index), device=device)

    # Probes with a constant training target are not trained (Feucht et al. skip
    # them), so they keep their initial weights. Their columns are excluded
    # from the loss, which leaves them untouched by Adam.
    trained = y_train.std(dim=0) >= CONSTANT_TARGET_TOLERANCE

    weight0, bias0 = initial_parameters(len(slots), activations.shape[1], seed=init_seed)
    weight = nn.Parameter(weight0.to(device))
    bias = nn.Parameter(bias0.to(device))
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate)
    n_train = x_train.shape[0]
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = x_train @ weight.T + bias
        squared_error = (prediction - y_train) ** 2
        # sum over probes of each probe's mean squared error
        loss = squared_error[:, trained].sum() / n_train
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        r2 = r_squared(x_test @ weight.T + bias, y_test)
        r2[~trained] = float("nan")
    return {"weight": weight.detach().cpu(), "bias": bias.detach().cpu(),
            "r2": r2.cpu(), "trained": trained.cpu()}


def fit_probes_one_at_a_time(activations: torch.Tensor, slots: list[ProbeSlot],
                             values: dict[str, np.ndarray], device: str = "cuda",
                             epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE,
                             init_seed: int = INIT_SEED) -> dict:
    """The reference loop, one nn.Linear and one Adam per probe. Used for testing."""
    train_index, test_index = train_test_indices(activations.shape[0])
    x_train = activations[train_index].to(device, torch.float32)
    x_test = activations[test_index].to(device, torch.float32)
    y_train_all = torch.as_tensor(targets_for(slots, values, train_index), device=device)
    y_test_all = torch.as_tensor(targets_for(slots, values, test_index), device=device)
    torch.manual_seed(init_seed)
    weights, biases, r2s, trained = [], [], [], []
    for column in range(len(slots)):
        probe = nn.Linear(activations.shape[1], 1)
        y_train = y_train_all[:, column]
        y_test = y_test_all[:, column]
        is_trained = bool(y_train.std() >= CONSTANT_TARGET_TOLERANCE)
        probe = probe.to(device)
        if is_trained:
            optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
            loss_function = nn.MSELoss()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = loss_function(probe(x_train).squeeze(-1), y_train)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            prediction = probe(x_test).squeeze(-1)
            r2 = r_squared(prediction[:, None], y_test[:, None])[0] if is_trained else torch.tensor(float("nan"))
        weights.append(probe.weight.detach().cpu().reshape(-1))
        biases.append(probe.bias.detach().cpu().reshape(()))
        r2s.append(r2.cpu())
        trained.append(is_trained)
    return {"weight": torch.stack(weights), "bias": torch.stack(biases),
            "r2": torch.stack(r2s), "trained": torch.tensor(trained)}


def planes_from_fit(slots: list[ProbeSlot], fit: dict) -> dict:
    """Pair sin and cos probes into planes: {"<carrier>__p<period>": {weight [2, hidden], bias [2], r2}}."""
    by_slot = {slot: index for index, slot in enumerate(slots)}
    planes = {}
    for slot in slots:
        if slot.function != "cos":
            continue
        cos_index = by_slot[slot]
        sin_index = by_slot[ProbeSlot(slot.carrier, slot.period, "sin")]
        planes[f"{slot.carrier}__p{slot.period}"] = {
            "weight": torch.stack([fit["weight"][cos_index], fit["weight"][sin_index]]).numpy(),
            "bias": torch.stack([fit["bias"][cos_index], fit["bias"][sin_index]]).numpy(),
            "r2": {"cos": float(fit["r2"][cos_index]), "sin": float(fit["r2"][sin_index])},
        }
    return planes
