"""Tool-duration predictor: the round-time rows, the feature built from earlier rounds only, and the forecaster."""

import csv
import gzip

import pytest

from analysis.dag import duration_eval_16w as D
from prediction.predictor.duration_baseline import CAP_MS, DurationForecaster

COLS = ["session", "day", "step", "event", "harness", "model", "prev1_cat", "prev1_width", "prev2_cat", "prev2_round_s", "stall_s"]


def write(path, rows):
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def r_row(session, step, cat, stall, prev2="", day="2026-08-01", width=1):
    return {"session": session, "day": day, "step": step, "event": "R", "harness": "h", "model": "m",
            "prev1_cat": cat, "prev1_width": width, "prev2_cat": "", "prev2_round_s": prev2, "stall_s": stall}


def test_a_rows_the_last_response_and_other_days_are_not_duration_rows(tmp_path):
    rows = [dict(r_row("s", 0, "Read", 2.0), event="A"), r_row("s", 0, "Read", 2.0), r_row("s", 1, "__none__", ""),
            r_row("s", 2, "Edit", 1.0, day="2026-09-01")]
    write(tmp_path / "x.csv.gz", rows)
    got = D.load(str(tmp_path / "x.csv.gz"), ["2026-08-01"])
    assert [(g["step"], g["cat"], g["dur_ms"]) for g in got] == [(0, "Read", 2000)]


def test_the_same_category_feature_uses_only_earlier_rounds_and_resets_with_the_session(tmp_path):
    rows = [r_row("a", 0, "Read", 2.0), r_row("a", 1, "Read", 5.0, prev2=2.0), r_row("a", 2, "Edit", 1.0, prev2=5.0),
            r_row("a", 3, "Read", 9.0, prev2=1.0), r_row("b", 0, "Read", 4.0)]
    write(tmp_path / "x.csv.gz", rows)
    got = D.load(str(tmp_path / "x.csv.gz"), ["2026-08-01"])
    assert [g["samecat"] for g in got] == [None, 2.0, None, 5.0, None]     # never the round's own time, never another session's
    assert [g["prevdur"] for g in got] == [None, 2.0, 5.0, 1.0, None]


def test_a_round_longer_than_the_cap_is_capped_in_the_label_and_in_the_features(tmp_path):
    rows = [r_row("a", 0, "Read", 99999.0), r_row("a", 1, "Read", 1.0, prev2=99999.0)]
    write(tmp_path / "x.csv.gz", rows)
    got = D.load(str(tmp_path / "x.csv.gz"), ["2026-08-01"])
    assert got[0]["dur_ms"] == CAP_MS and got[1]["samecat"] == CAP_MS / 1000 and got[1]["prevdur"] == CAP_MS / 1000


def rows_of(n=80):
    out = []
    for i in range(n):
        cat, dur = ("Read", 1000) if i % 2 else ("Edit", 9000)
        out.append({"cat": cat, "cat2": None, "width": 1.0, "prevdur": 1.0 + i, "samecat": None, "harness": "h",
                    "model": "m", "dur_ms": dur, "step": 1})
    return out


def test_the_category_table_learns_the_scale_of_each_tool():
    model = DurationForecaster(rows_of(), ("cat",), 1e-6)
    assert model.predict(rows_of()[1])[0] == pytest.approx(1.0, abs=0.01)
    assert model.predict(rows_of()[0])[0] == pytest.approx(9.0, abs=0.01)


def test_a_category_never_seen_falls_back_to_the_pooled_distribution():
    model = DurationForecaster(rows_of(), ("cat",), 20)
    assert model.predict(dict(rows_of()[0], cat="Browser"))[0] == pytest.approx(5.0, abs=0.01)


def test_crps_is_in_seconds_and_zero_for_a_certain_outcome():
    rows = [dict(rows_of()[0], dur_ms=2000) for _ in range(30)]
    model = DurationForecaster(rows, ("cat",), 20)
    assert D.per_row(model, rows[:1])[0] == pytest.approx(0, abs=1e-9)
    assert D.per_row(model, [dict(rows[0], dur_ms=5000)])[0] == pytest.approx(3.0, abs=1e-6)   # |5 s - 2 s|


def test_the_grid_fits_the_budget_every_chain_contains_the_baseline_and_every_step_can_see_the_previous_round():
    assert len(D.GRID) == D.G.TUNING_BUDGET == 12
    assert all(c[0] == "cat" for c in D.MODEL_CHAINS.values()) and min(D.G.GATE_STEPS) >= 1
