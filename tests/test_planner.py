"""Rollout planner over predicted futures, with the stand-in predictor in place of the DAG predictor."""

import math

import pytest

from analysis.replay.planner import Forecast, FutureSample, NoisyOracle, RolloutPlanner, aged, true_future
from analysis.replay.sim import Config, Profile, Simulator, build_sessions, reply_time

PROFILE = Profile(decode_ms=10.0, prefill_ms=1.0)


def req(prompt, completion, user=False, gap=None, cat="__none__", lcp=0):
    return {"prompt": prompt, "lcp": lcp, "completion": completion, "user": user, "append": True, "cat": cat, "gap": gap}


def session(sid, start_s, reqs):
    return {"sid": sid, "start": int(start_s * 1000), "requests": reqs}


def two_turn_session():
    # turn 1: two calls with a 3 s tool gap; the model answers, the user replies; turn 2: one call
    return [session("s", 0, [req(1000, 100, user=True, gap=3000, cat="Read"), req(1100, 100, lcp=1000, gap=9000),
                             req(1200, 100, user=True, lcp=1100)])]


def test_true_future_splits_turns_where_the_model_answers():
    [s] = build_sessions(two_turn_session(), PROFILE, reply=reply_time("fixed:5"), unit="session")
    f = true_future(s, 0)
    assert [len(t) for t in f.turns] == [2, 1]
    assert f.turn_times == pytest.approx([2.0 + 3 + 1.0, 1.0])          # 1 s prefill + 1 s decode, then cached
    assert f.total == pytest.approx(2 + 3 + 1 + 5 + 1)                  # the user's reply is fixed at 5 s
    assert f.total == pytest.approx(s.z[0])


def test_exact_stand_in_predictor_returns_the_truth():
    [s] = build_sessions(two_turn_session(), PROFILE, reply=reply_time("fixed:5"), unit="session")
    fc = NoisyOracle(error=0, samples=3).forecast(s, 1, now=0.0)
    assert fc.first == 1 and all(x == true_future(s, 1) for x in fc.samples)


def test_noisy_samples_keep_their_shape_and_repeat():
    [s] = build_sessions(two_turn_session(), PROFILE, reply=reply_time("fixed:5"), unit="session")
    a = NoisyOracle(error=0.5, samples=20, seed=1).forecast(s, 0, 0.0)
    b = NoisyOracle(error=0.5, samples=20, seed=1).forecast(s, 0, 0.0)
    assert a.samples == b.samples and len({len(x.calls) for x in a.samples}) > 1
    assert all(x.calls[-1][1] == 0.0 and all(d > 0 for d, _ in x.calls) for x in a.samples)


def test_forecast_drops_samples_the_session_has_outrun():
    fc = Forecast(0.0, 2, [FutureSample((((1.0, 1.0), (1.0, 0.0)),)), FutureSample((((2.0, 1.0),) * 4,))])
    assert [len(c) for c in aged(fc, 2)] == [2, 4]
    assert [len(c) for c in aged(fc, 4)] == [2]           # 2 calls done: the first sample is over
    assert aged(fc, 9) == [[(pytest.approx(10 / 6), 0.0)]]  # every sample over: one more call, of the mean length


def test_turn_totals_adapter_keeps_each_turns_gpu_and_wall_time():
    f = FutureSample.from_turn_totals([6.0, 2.0], [20.0, 2.0], [3, 1])
    assert f.turn_times == pytest.approx([20.0, 2.0])
    assert [sum(d for d, _ in t) for t in f.turns] == pytest.approx([6.0, 2.0])
    assert f.total == pytest.approx(20 + 5 + 2)


def trap():
    """One slot, busy with H until 1 s. A (one 5 s call) and B (2 s call, 5 s tool, 2 s call) are
    queued at 0.5 s; deadlines are 1.5 x time alone: A by 8, B by 14. EDF runs A first and B's
    second call ends at 15. Running B's first call first gets all three on time."""
    raw = [session("H", 0, [req(0, 100, user=True)]),
           session("A", 0.5, [req(0, 500, user=True)]),
           session("B", 0.5, [req(0, 200, user=True, gap=5000, cat="Read"), req(200, 200)])]
    return build_sessions(raw, PROFILE, reply=reply_time("fixed:5"), unit="session")


def run(sessions, planner=None):
    cfg = Config(slots=1, kv_tokens=math.inf, profile=PROFILE, policy="edf", deadline_mode="proportional", slack=1.5)
    return Simulator(sessions, cfg, planner=planner).run()


def test_planner_with_exact_futures_beats_edf_on_the_trap():
    assert sum(not t.missed for t in run(trap()).turns) == 2
    planner = RolloutPlanner(NoisyOracle(error=0, samples=1), samples=1, horizon=100)
    sim = run(trap(), planner)
    assert sum(not t.missed for t in sim.turns) == 3
    assert planner.decisions >= 1 and planner.changed >= 1


@pytest.mark.parametrize("error", [0.0, 0.3, 1.0])
def test_planner_runs_every_request_whatever_its_forecast(error):
    raw = [session(f"s{i}", i * 0.7, [req(500, 100, user=True, gap=2000, cat="Read"), req(700, 80, lcp=500, gap=4000),
                                      req(900, 120, user=True, lcp=700)]) for i in range(6)]
    sessions = build_sessions(raw, PROFILE, reply=reply_time("fixed:5"), unit="session")
    cfg = Config(slots=2, kv_tokens=math.inf, profile=PROFILE, policy="edf", deadline_mode="proportional", slack=1.2)
    sim = Simulator(sessions, cfg, planner=RolloutPlanner(NoisyOracle(error=error, samples=4), samples=4)).run()
    assert len(sim.turns) == 6 and sim.running == 0 and not sim.active and not sim.ready


def test_confidence_keeps_edf_unless_the_override_wins_in_most_samples():
    # exact futures: B-first wins in every sample, so any confidence still overrides EDF
    planner = RolloutPlanner(NoisyOracle(error=0, samples=4), samples=4, horizon=100, confidence=0.75, trace=[])
    assert sum(not t.missed for t in run(trap(), planner).turns) == 3
    d = next(d for d in planner.trace if d["choice"] != d["candidates"][0]["session"])
    edf_scores, alt = d["candidates"][0]["scores"], next(c for c in d["candidates"] if c["session"] == d["choice"])
    assert all(a > b for a, b in zip(alt["scores"], edf_scores))
    # a confidence above 1 can never be met: EDF's choice always stands
    never = RolloutPlanner(NoisyOracle(error=0, samples=4), samples=4, horizon=100, confidence=1.01)
    assert sum(not t.missed for t in run(trap(), never).turns) == 2 and never.changed == 0


def test_rollout_charges_a_session_only_what_is_left_at_the_horizon():
    # one session alone, mid-way through a 10 s call that started 4 s ago, then a 3 s gap and a 2 s call:
    # from now it needs 6 + 3 + 2 = 11 s whether or not the horizon cuts it
    fc = Forecast(0.0, 0, [FutureSample((((10.0, 3.0), (2.0, 0.0)),))])
    planner = RolloutPlanner(None, samples=1, horizon=5)
    sim = type("S", (), {"cfg": type("C", (), {"slots": 2})()})()
    for deadline, ok in ((100 + 11, 1), (100 + 10.9, 0)):
        snap = {0: ("running", 4.0, aged(fc, 0), deadline), 1: ("ready", 100.0, [[(1.0, 0.0)]], 1e9)}
        assert planner._rollout(sim, snap, 0, 1, 100.0) == ok + 1
