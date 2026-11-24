"""Session dataset and forecaster: features known at a user turn's start, labels from the turns to come."""

import pytest

from analysis.dag.dataset import annotate_session
from analysis.dag.session import FEATURES, SessionHistory, audit, features, labels, rows_of, scramble, session_view
from prediction.predictor.session_baseline import SessionForecaster


def rnd(start, prompt=1000, completion=100):
    return {"req": 0, "start": start, "end": start + 1000, "ttft": 100, "prompt": prompt, "completion": completion,
            "cached": 0, "model": "m", "finish": "tool_calls", "trigger": "user",
            "tools": [{"i": 0, "name": "x", "cat": "Read", "outcome": "ok", "result_tokens": 10, "latency_ms": 1000}],
            "round_ms": 1000, "subs": []}


def turn(sid, i, start, n, idle=None, user="u1", prompt=1000):
    rounds = [rnd(start + 2000 * k, prompt if k == 0 else prompt + 50 * k) for k in range(n)]
    return {"session": sid, "user": user, "harness": "h", "day": "2026-08-23", "turn": i, "start": start,
            "end": start + 2000 * n, "idle_before_ms": idle, "rounds": rounds}


def session(sid="a", sizes=(3, 1, 2), user="u1", t0=0):
    turns, t = [], t0
    for i, n in enumerate(sizes):
        turns.append(turn(sid, i, t, n, None if i == 0 else 5000, user, 1000 + 100 * i))
        t = turns[-1]["end"] + 5000
    annotate_session(turns)
    return turns


def test_labels_count_what_comes_after_each_turn():
    s = session()
    got = [labels(s, u) for u in range(3)]
    assert [(g["cur_calls"], g["fut_turns"], g["fut_calls"]) for g in got] == [(3, 2, 3), (1, 1, 2), (2, 0, 0)]
    assert got[0]["fut_work_s"] > got[1]["fut_work_s"] > got[2]["fut_work_s"] == 0


def test_features_hold_the_earlier_turns_and_this_turns_arrival():
    s = session()
    f = features(session_view(s, 1), {"n": 0, "turns": None, "calls": None})
    assert f["turn_idx"] == 1 and f["prior_mean_calls"] == 3 and f["prior_last_calls"] == 3
    assert f["first_prompt"] == 1100 and f["idle_before_s"] == 5.0
    assert f["prior_last_wall_s"] == pytest.approx(6.0)
    first = features(session_view(s, 0), {"n": 0, "turns": None, "calls": None})
    assert first["prior_mean_calls"] is None and first["idle_before_s"] is None and first["elapsed_s"] == 0


def test_the_future_cannot_reach_a_feature():
    s = session(sizes=(3, 4, 2))
    h = {"n": 0, "turns": None, "calls": None}
    for u in range(3):
        assert features(session_view(s, u), h) == features(session_view(scramble(s, u), u), h)
    assert audit([s, session("b", (2, 2))], SessionHistory([s]), 10) == 0


def test_a_feature_that_read_the_future_would_be_caught():
    s = session(sizes=(3, 4, 2))
    h = {"n": 0, "turns": None, "calls": None}
    leaky = lambda turns, u: len(turns[u]["rounds"])       # this turn's own size: known only at its end
    assert leaky(s, 1) != leaky(scramble(s, 1), 1)


def test_account_history_counts_only_sessions_that_ended_before():
    a, b = session("a", (3, 1), t0=0), session("b", (2,), t0=10**9)
    hist = SessionHistory([a, b])
    end_a = a[-1]["end"]
    assert hist.before("u1", end_a)["n"] == 0                  # ended strictly before: not at that very moment
    got = hist.before("u1", end_a + 1)
    assert got["n"] == 1 and got["turns"] == 2 and got["calls"] == 4
    assert hist.before("u1", b[0]["start"])["n"] == 1 and hist.before("nobody", 10**12)["n"] == 0


def test_rows_of_gives_one_row_per_turn_with_every_column():
    rows = list(rows_of(session(), SessionHistory([session()])))
    assert [r["turn"] for r in rows] == [0, 1, 2] and all(c in rows[0] for c in FEATURES)


def forecast_rows(n=60):
    rows = []
    for i in range(n):
        harness, calls = ("x", 10) if i % 2 else ("y", 100)
        rows.append({"turn_idx": 0, "harness": harness, "model": "m", "fut_calls": calls, "fut_turns": calls // 10})
    return rows


def near(x, y, tol=1e-3):
    return abs(x - y) < tol


def test_the_turn_index_alone_forecasts_the_pooled_distribution():
    calls, turns = SessionForecaster(forecast_rows(), (), 20).predict({"turn_idx": 0, "harness": "x", "model": "m"})
    assert near(calls, 55) and near(turns, 5.5)


def test_a_deeper_level_separates_what_it_explains():
    model = SessionForecaster(forecast_rows(), ("harness",), 1e-6)
    assert near(model.predict({"turn_idx": 0, "harness": "x", "model": "m"})[0], 10)
    assert near(model.predict({"turn_idx": 0, "harness": "y", "model": "m"})[0], 100)


def test_heavy_smoothing_falls_back_to_the_turn_index():
    model = SessionForecaster(forecast_rows(), ("harness",), 1e9)
    assert near(model.predict({"turn_idx": 0, "harness": "x", "model": "m"})[0], 55, 0.1)


def test_a_group_never_seen_falls_back_to_its_parent():
    model = SessionForecaster(forecast_rows(), ("harness",), 20)
    assert near(model.predict({"turn_idx": 0, "harness": "never", "model": "m"})[0], 55)


def test_crps_is_zero_for_a_certain_outcome_and_rises_with_the_miss():
    rows = [{"turn_idx": 0, "harness": "x", "model": "m", "fut_calls": 7, "fut_turns": 1} for _ in range(20)]
    calls, _, _ = SessionForecaster(rows, (), 20).marginals(rows[0])
    assert near(calls.crps(7), 0) and calls.crps(10) > calls.crps(8) > 0
