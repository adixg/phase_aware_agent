"""Time-left predictor: the target, the chains, and the pre-registered grid."""

import pytest

from analysis.dag import session_eval_16w as S
from prediction.predictor.session_baseline import SessionTimeForecaster


def rows(n=80):
    out = []
    for i in range(n):
        heavy = i % 2
        out.append({"turn_idx": 0, "user_mean_path_s": 5000.0 if heavy else 50.0, "user_mean_turns": 8.0, "prior_last_calls": None,
                    "harness": "h", "model": "m", "first_prompt": 1000, "rem_path_s": 4000.4 if heavy else 40.6,
                    "rem_calls": 400 if heavy else 4})
    return out


def test_the_target_is_whole_seconds_and_the_requests_left():
    assert SessionTimeForecaster.TARGETS == ("rem_path_s", "rem_calls")
    assert SessionTimeForecaster.value({"rem_path_s": 40.6}, "rem_path_s") == 41
    assert SessionTimeForecaster.value({"rem_calls": 7}, "rem_calls") == 7


def test_the_turn_index_alone_forecasts_the_pooled_time():
    secs, calls = SessionTimeForecaster(rows(), (), 20).predict(rows()[0])
    assert secs == pytest.approx((4000 + 41) / 2, abs=1) and calls == pytest.approx(202, abs=0.5)


def test_the_users_session_time_level_separates_light_and_heavy_users():
    model = SessionTimeForecaster(rows(), ("histpath",), 1e-6)
    assert model.predict(rows()[0])[0] == pytest.approx(41, abs=1) and model.predict(rows()[1])[0] == pytest.approx(4000, abs=1)


def test_a_user_with_no_history_falls_back_to_the_turn_index():
    model = SessionTimeForecaster(rows(), ("histpath",), 20)
    new = dict(rows()[0], user_mean_path_s=None)
    assert model.predict(new)[0] == pytest.approx((4000 + 41) / 2, abs=1)


def test_crps_is_zero_for_a_certain_outcome_and_the_gate_averages_over_the_gated_turns():
    certain = [dict(r, rem_path_s=100.0, rem_calls=5) for r in rows()]
    model = SessionTimeForecaster(certain, (), 20)
    assert S.per_row(model, certain[:3], "rem_path_s").max() == pytest.approx(0, abs=1e-9)
    assert S.G.GATE_TURNS == (0, 1, 2, 5)


def test_the_grid_fits_the_budget_and_every_chain_starts_at_a_level_that_exists_at_turn_zero():
    assert len(S.GRID) == S.G.TUNING_BUDGET == 12
    assert all(c[0] in ("histpath", "hist") for c in S.MODEL_CHAINS.values())     # user history, known at arrival
