"""The admission forecaster as the simulator's remaining-calls estimate, on hand-made turns."""

import pytest

from analysis.replay.forecast import Forecaster
from analysis.replay.sim import POLICIES, Cheap, Profile, build_sessions, estimate_remaining

PROFILE = Profile(decode_ms=10.0, prefill_ms=1.0)


def raw_req(user, end, gap, prompt=1000, tools=1):
    return {"prompt": prompt, "provider_prompt": prompt, "lcp": 0, "completion": 100, "user": user, "append": True,
            "cat": "Read" if gap is not None else "__none__", "gap": gap, "model": "m", "n_tools": tools,
            "start": end - 1000, "end": end}


def chain(sid, turns, harness="h"):
    """A session whose turns have the given numbers of requests, 2 s apart."""
    total, reqs = sum(turns), []
    for n in turns:
        for k in range(n):
            end = 2000 * (len(reqs) + 1)
            reqs.append(raw_req(k == 0, end, None if len(reqs) == total - 1 else 1000))
    return {"sid": sid, "start": 0, "harness": harness, "requests": reqs}


def sessions(*lengths_per_session):
    return build_sessions([chain(f"s{i}", ts) for i, ts in enumerate(lengths_per_session)], PROFILE)


def test_turn_metadata_is_read_from_the_raw_records():
    s = sessions([3, 2])[0]
    assert [(t.calls, t.tools, t.model, t.prompt) for t in s.turns] == [(3, 3, "m", 1000), (2, 2, "m", 1000)]
    assert s.harness == "h"
    assert [t.duration_ms for t in s.turns] == [5000, 3000]   # first start 1000 to last end 6000; then 7000 to 10000


def test_forecast_is_conditioned_on_the_turn_having_survived_so_far():
    f = Forecaster(sessions([2], [10]), first_only=True)
    key = ("h", "m", 1000)
    assert f.remaining_calls(key, 0) == pytest.approx(6.0)     # equal weight on 2 and 10 requests
    assert f.remaining_calls(key, 1) == pytest.approx(5.0)     # the 2-request turn is still possible: (1 + 9) / 2
    assert f.remaining_calls(key, 2) == pytest.approx(8.0)     # past it: only the 10-request turn is left
    assert f.remaining_calls(key, 9) == pytest.approx(1.0)
    assert f.remaining_calls(key, 50) == 1.0                   # beyond everything seen in training


def test_first_only_forecaster_leaves_later_turns_to_the_cheap_table():
    train = sessions([2, 4], [6, 3])
    first, every = Forecaster(train, True), Forecaster(train, False)
    test = sessions([5, 5])
    first.annotate(test)
    every.annotate(test)
    reqs = test[0].reqs
    assert all(r.pred_first > 0 for r in reqs if r.turn == 0) and all(r.pred_first == 0 for r in reqs if r.turn == 1)
    assert all(r.pred_all > 0 for r in reqs)


def test_first_only_fit_ignores_later_turns():
    f = Forecaster(sessions([4, 20]), first_only=True)          # the 20-request turn is not a first turn
    assert f.remaining_calls(("h", "m", 1000), 0) == pytest.approx(4.0)
    assert Forecaster(sessions([4, 20]), first_only=False).remaining_calls(("h", "m", 1000), 0) == pytest.approx(12.0)


def test_policy_uses_the_forecast_where_there_is_one_and_cheap_elsewhere():
    train = sessions([2, 6])
    cheap = Cheap(train, PROFILE)
    f = Forecaster(train, first_only=True)
    test = sessions([5, 5])
    f.annotate(test)
    first, later = test[0].reqs[0], test[0].reqs[5]
    p = POLICIES["srpt_predfirst"]
    per_call = cheap.service + cheap.gap
    assert estimate_remaining(first, p, cheap, PROFILE) - estimate_remaining(first, POLICIES["srpt_cheap"], cheap, PROFILE) \
        == pytest.approx((first.pred_first - cheap.remaining_calls(first)) * per_call)
    assert estimate_remaining(later, p, cheap, PROFILE) == pytest.approx(estimate_remaining(later, POLICIES["srpt_cheap"], cheap, PROFILE))


def test_forecast_policies_exist_for_each_family_and_the_pure_ones_have_no_deadline_logic():
    for fam in ("llf", "srpt", "psrpt"):
        for v in ("predfirst", "predall"):
            assert POLICIES[f"{fam}_{v}"].forecast == v[4:]
    assert POLICIES["psrpt_predall"].pure and not POLICIES["srpt_predall"].pure
