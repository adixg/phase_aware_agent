"""Next-tool predictor: classes, smoothing, the A-to-R label join, and the scores."""

import csv
import gzip

import numpy as np
import pytest

from analysis.dag import tool_eval
from analysis.dag.tool_eval import scores
from analysis.dag.dataset import CATS
from prediction.predictor.tool_baseline import CLASSES, CODE, END, ToolForecaster


def row(step=1, prev1="Read", nxt="Edit", prev2=None, harness="h"):
    return {"step": step, "prev1_cat": prev1, "prev2_cat": prev2, "prev3_cat": None, "prev1_errors": 0.0,
            "prev1_finish": "tool_calls", "prev1_width": 1.0, "prev1_result_tokens": 100.0, "harness": harness,
            "model": "m", "next_cat": nxt}


def alternating(n=60):
    """After Read comes Edit, after Edit comes Read, whatever the harness."""
    return [row(1, "Read", "Edit", harness="x") if i % 2 else row(1, "Edit", "Read", harness="y") for i in range(n)]


def near(x, y, tol=1e-3):
    return abs(x - y) < tol


def test_there_are_seventeen_classes_with_end_last():
    assert len(CLASSES) == 17 and CLASSES[-1] == END == "__none__" and CODE[END] == 16


def test_probabilities_sum_to_one_over_the_classes():
    p = ToolForecaster(alternating(), ("cat",), 20).proba(row())
    assert p.shape == (17,) and near(p.sum(), 1.0, 1e-9) and (p >= 0).all()


def test_the_previous_tool_table_learns_what_follows_each_tool():
    model = ToolForecaster(alternating(), ("cat",), 1e-6)
    assert near(model.proba(row(prev1="Read"))[CODE["Edit"]], 1.0)
    assert near(model.proba(row(prev1="Edit"))[CODE["Read"]], 1.0)


def test_a_previous_tool_never_seen_falls_back_to_the_step():
    p = ToolForecaster(alternating(), ("cat",), 20).proba(row(prev1="Browser"))
    assert near(p[CODE["Edit"]], 0.5) and near(p[CODE["Read"]], 0.5)


def test_a_chain_with_no_extra_level_equals_the_baseline_and_a_deeper_one_separates_more():
    rows = [row(1, "Read", "Edit", prev2="Edit") if i % 2 else row(1, "Read", "Write", prev2="Read") for i in range(80)]
    base = ToolForecaster(rows, ("cat",), 1e-6).proba(row(prev1="Read", prev2="Edit"))
    deep = ToolForecaster(rows, ("cat", "cat2"), 1e-6).proba(row(prev1="Read", prev2="Edit"))
    assert near(base[CODE["Edit"]], 0.5) and near(deep[CODE["Edit"]], 1.0)


def test_every_model_chain_starts_with_the_baseline_levels_and_the_grid_fits_the_budget():
    assert all(c[0] == "cat" for c in tool_eval.MODEL_CHAINS.values())
    assert len(tool_eval.GRID) <= tool_eval.G.TUNING_BUDGET


class Uniform:
    def proba(self, r):
        return np.full(17, 1 / 17)


class Certain:
    def proba(self, r):
        p = np.zeros(17)
        p[CODE[r["next_cat"]]] = 1.0
        return p


def test_scores_of_a_perfect_and_a_uniform_forecast():
    rows = [row(nxt="Edit"), row(nxt="Read")]
    perfect, uniform = scores(Certain(), rows), scores(Uniform(), rows)
    assert near(perfect["brier"].sum(), 0, 1e-9) and perfect["top1"].all() and near(perfect["logloss"].max(), 0, 1e-9)
    assert near(uniform["brier"][0], (256 + 16) / 289, 1e-9) and near(uniform["logloss"][0], np.log(17), 1e-9)


def write_rows(path, rows):
    cols = ["session", "day", "turn", "step", "event", "harness", "model", "prev1_cat", "prev2_cat", "prev3_cat",
            "prev1_errors", "prev1_width", "prev1_result_tokens", "prev1_finish"] + [f"cat_{c}" for c in CATS]
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def test_the_label_of_an_a_row_comes_from_the_r_row_of_the_same_request(tmp_path):
    base = {"session": "s", "day": "2026-08-23", "turn": 0, "harness": "h", "model": "m"}
    rows = [dict(base, step=0, event="A"), dict(base, step=0, event="R", prev1_cat="*Search", prev1_width=1),
            dict(base, step=1, event="A", prev1_cat="*Search", prev1_width=1, prev1_errors=0, prev1_result_tokens=5, prev1_finish="tool_calls",
                 **{"cat_*Search": 2, "cat_Read": 1}),
            dict(base, step=1, event="R", prev1_cat="__none__", prev1_width=0),
            dict(base, step=2, event="A", prev1_cat="__none__"), dict(base, step=2, event="R", prev1_cat="Strange", prev1_width=1),
            dict(base, step=3, event="A", day="2026-09-01"), dict(base, step=3, event="R", day="2026-09-01", prev1_cat="Read")]
    path = tmp_path / "online_rows.csv.gz"
    write_rows(path, rows)
    got = tool_eval.load(str(path), ["2026-08-23"])
    assert [(r["step"], r["next_cat"]) for r in got] == [(0, "*Search"), (1, END), (2, "Other")]    # the Sep 1 request is filtered out
    assert got[0]["prev1_cat"] is None and got[1]["prev1_cat"] == "*Search" and got[1]["prev1_result_tokens"] == 5.0
    assert got[0]["dom"] is None and got[1]["dom"] == "*Search"        # the turn's most used tool so far, None before any


def test_the_dominant_tool_level_separates_turns_by_what_they_have_used_most():
    rows = []
    for i in range(80):
        r = row(1, "Read", "Edit" if i % 2 else "Write")
        r["dom"] = "Edit" if i % 2 else "Write"
        rows.append(r)
    model = ToolForecaster(rows, ("cat", "dom"), 1e-6)
    assert near(model.proba(dict(row(prev1="Read"), dom="Edit"))[CODE["Edit"]], 1.0)
    assert near(model.proba(dict(row(prev1="Read"), dom="Write"))[CODE["Write"]], 1.0)
    assert near(model.proba(dict(row(prev1="Read"), dom=None))[CODE["Edit"]], 0.5)       # no tool yet: the parent
