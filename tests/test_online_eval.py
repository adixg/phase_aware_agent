"""The gate arithmetic of the online evaluation, on rows where the answer is known."""

import pytest

from analysis.dag import online_gates as G
from analysis.dag.online_eval import GRID, gate, passes
from prediction.predictor.online_baseline import OnlineForecaster
from tests.test_online_baseline import row, turns


def sessions(n_sessions, make):
    rows = []
    for i in range(n_sessions):
        for r in make(i):
            r["session"] = f"s{i}"
            rows.append(r)
    return rows


def informative(i):                      # the category tells how long the turn is
    cat, n = ("Edit", 4) if i % 2 else ("Read", 14)
    return turns([n], cat=cat)


def test_the_grid_fits_the_budget():
    assert len(GRID) == len(set(GRID)) <= G.TUNING_BUDGET


def test_a_model_equal_to_the_baseline_gains_nothing():
    rows = sessions(40, informative)
    m = OnlineForecaster(rows, "A", ("cat",), 20)
    res = gate(m, m, rows, draws=50)
    assert res["reduction"] == pytest.approx(0.0) and not passes(res)


def test_an_informative_level_beats_the_step_only_baseline_and_passes():
    rows = sessions(60, informative)
    base, model = OnlineForecaster(rows, "A", (), 1), OnlineForecaster(rows, "A", ("cat",), 1)
    res = gate(model, base, rows, steps=(1, 2), draws=200)
    assert res["reduction"] > 0.5 and res["ci"][0] > 0 and passes(res)
    assert set(res["steps"]) == {1, 2} and res["steps"][1]["n"] == 60


def test_the_gate_needs_both_the_size_and_a_positive_interval():
    assert not passes({"reduction": 0.04, "ci": (0.01, 0.07)})        # under the 5% bar
    assert not passes({"reduction": 0.08, "ci": (-0.01, 0.17)})       # the interval reaches below zero
    assert passes({"reduction": 0.08, "ci": (0.02, 0.14)})
