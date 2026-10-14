import numpy as np
import pytest

from months import data


def test_grid_sizes():
    assert len(data.training_grid()[0]) == 210
    assert len(data.evaluation_grid()[0]) == 1074
    assert len(data.evaluation_grid(132)[0]) == 1506


def test_first_training_prompt():
    rows, prompts = data.training_grid()
    assert prompts[0] == "Q: What month is one months after January?\nA:"
    assert rows[0]["answer_month"] == "February"


def test_evaluation_grid_is_sorted_by_prompt():
    _, prompts = data.evaluation_grid()
    assert prompts == sorted(prompts)


def test_number_words():
    assert data.number_to_words(1) == "one"
    assert data.number_to_words(21) == "twenty-one"
    assert data.number_to_words(40) == "forty"
    assert data.number_to_words(105) == "one hundred five"
    num2words = pytest.importorskip("num2words")
    for n in range(1, 100):
        assert data.number_to_words(n) == num2words.num2words(n)


def test_carrier_values_ranges():
    rows, _ = data.training_grid()
    assert set(data.carrier_values("input-month", rows)) == set(range(12))
    assert set(data.carrier_values("offset", rows)) == set(range(1, 24))
    assert set(data.carrier_values("sum", rows)) == set(range(2, 25))


def test_band_counts():
    rows, _ = data.evaluation_grid()
    sums = data.sums(rows)
    summary = data.band_summary(np.zeros(len(rows), dtype=int), data.answers(rows), sums)
    assert summary["bands"]["2-24"]["n"] == 210
    assert summary["bands"]["25-48"]["n"] == 288
    assert summary["bands"]["49-96"]["n"] == 576
    assert "97-132" not in summary["bands"]
