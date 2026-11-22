"""The online forecaster on small synthetic turns whose answers are known."""

import pytest

from prediction.predictor.online_baseline import OnlineForecaster, step_bucket


def near(x):
    """Strength 1e-6 still lets a parent contribute about 1e-7 of its mass: compare to 1e-4."""
    return pytest.approx(x, abs=1e-4)


def row(step, after, event="A", cat="Read", width=1, harness="h", stall=2.0, last=False):
    return {"event": event, "step": step, "harness": harness, "model": "m", "prev1_cat": cat, "prev1_width": width,
            "prev1_errors": 0, "prev1_round_s": stall, "prev1_result_tokens": 10, "user_prior_mean_rounds": 5.0,
            "sess_prior_mean_rounds": 5.0, "rem_after": after, "stall_s": None if last else stall}


def turns(lengths, **kw):
    """Rows of A events for turns of the given numbers of requests."""
    return [row(k, n - k - 1, last=k == n - 1, **kw) for n in lengths for k in range(n)]


def test_step_bucket_is_exact_then_coarse():
    assert [step_bucket(k) for k in (0, 5, 15)] == [0, 5, 15]
    assert step_bucket(16) == step_bucket(19) < step_bucket(20) < step_bucket(30) < step_bucket(120) == step_bucket(500)


def test_conditioning_on_the_step_is_conditioning_on_survival():
    # one-request turns and ten-request turns, equally many: at step 0 half are done after this request, at step 1 none
    f = OnlineForecaster(turns([1, 10]), "A", levels=(), strength=1e-6)
    assert f.predict(row(0, 0)).mean_after == near((0 + 9) / 2)
    assert f.predict(row(0, 0)).p_last == near(0.5)
    p1 = f.predict(row(1, 0))                                  # the 1-request turn is gone
    assert p1.mean_after == near(8.0) and p1.p_last == near(0.0)


def test_a_response_without_tools_ends_the_turn_at_r():
    rows = []
    for n in (3, 3, 3, 6):
        for k in range(n):
            last = k == n - 1
            rows.append(row(k, n - k - 1, event="R", width=0 if last else 1, last=last))
    f = OnlineForecaster(rows, "R", levels=("tools",), strength=1e-6)
    done = f.predict(row(2, 0, event="R", width=0))
    assert done.mean_after == near(0.0) and done.p_last == near(1.0)
    assert f.predict(row(2, 0, event="R", width=1)).p_last == near(0.0)     # called tools: the turn goes on


def test_deeper_levels_sharpen_the_forecast_and_sparse_cells_fall_back():
    short = turns([2] * 30, cat="Edit")                        # after Edit the turn is short
    long_ = turns([12] * 30, cat="Read")
    f = OnlineForecaster(short + long_, "A", levels=("cat",), strength=1e-6)
    assert f.predict(row(1, 0, cat="Edit")).mean_after == near(0.0)
    assert f.predict(row(1, 0, cat="Read")).mean_after == near(10.0)
    # an unseen category falls back to the step cell: 30 short and 30 long turns, so (0 + 10) / 2
    unseen = f.predict(row(1, 0, cat="Browser"))
    assert unseen.mean_after == near(5.0) and unseen.support["group_depth"] == 1


def test_smoothing_blends_a_small_cell_with_its_parent():
    f = OnlineForecaster(turns([2] * 2, cat="Edit") + turns([12] * 40, cat="Read"), "A", levels=("cat",), strength=20)
    mean = f.predict(row(1, 0, cat="Edit")).mean_after
    assert 0.0 < mean < 10.0                                   # 2 rows against a parent worth 20: mostly the parent


def test_stall_forecast_uses_only_rounds_that_have_a_stall():
    f = OnlineForecaster(turns([3, 3], stall=4.0), "A", levels=("cat",), strength=1e-6)
    p = f.predict(row(0, 2))
    assert p.mean_stall_s == pytest.approx(4.0) and p.quantiles_stall_s[0.5] == pytest.approx(4.0)


def test_quantiles_are_ordered():
    f = OnlineForecaster(turns([1, 3, 6, 12, 25]), "A", levels=(), strength=20)
    q = f.predict(row(0, 0)).quantiles_after
    assert q[0.1] <= q[0.5] <= q[0.9]
