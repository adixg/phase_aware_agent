"""The predicted-versus-actual export: naming, the distribution columns, and the windows."""

import pytest

from analysis.dag import predictions_vs_actual as P
from analysis.dag import splits
from prediction.predictor.empirical_baseline import Marginal
from prediction.predictor.tool_baseline import CLASSES, END


def test_classes_are_named_as_people_read_them_and_columns_are_safe_identifiers():
    assert P.label("*Search") == "Search" and P.label(END) == "turn_ends" and P.label("Read") == "Read"
    cols = [P.column(c) for c in CLASSES]
    assert len(set(cols)) == len(CLASSES) == 17                       # no two classes share a column name
    assert all(c.replace("_", "").isalnum() for c in cols) and "Wait_Poll" in cols and "File_Ops" in cols and "turn_ends" in cols


def marginal(samples):
    return Marginal.from_atoms([((v,), 1 / len(samples)) for v in samples], 0)


def test_distribution_columns_match_a_hand_calculation():
    s = P.dist_stats(marginal((1000, 2000, 2000, 5000)), 2000)
    assert s["mean"] == pytest.approx(2500)
    assert (s["p10"], s["p50"], s["p90"]) == (1000, 2000, 5000)
    assert s["cdf_below_actual"] == pytest.approx(0.25) and s["cdf_at_actual"] == pytest.approx(0.75)
    assert s["crps"] == pytest.approx(250)                            # E|X-2000| - half E|X-X'| = 1000 - 750


def test_an_actual_value_beyond_the_forecast_sits_at_the_top_of_its_distribution():
    s = P.dist_stats(marginal((1000, 2000, 2000, 5000)), 9000)
    assert s["cdf_below_actual"] == pytest.approx(1.0) and s["cdf_at_actual"] == pytest.approx(1.0)
    assert P.dist_stats(marginal((1000, 2000)), 10)["cdf_at_actual"] == pytest.approx(0.0)


def test_each_week_trains_on_days_strictly_before_the_days_it_scores():
    for week in ("tune", "validation"):
        train, test = P.windows(week)
        assert max(train) < min(test)
    assert P.windows("tune")[1] == splits.TUNE_DAYS and P.windows("validation")[1] == splits.VALIDATION_DAYS
    assert set(P.windows("validation")[0]) == set(splits.FIT_DAYS + splits.TUNE_DAYS)   # as the validation refit did


def test_the_readme_template_fills_without_a_missing_key():
    text = P.README.format(week="tune", first="a", last="b", train="c to d", note="n", steps="s")
    assert "next_tool.csv" in text and "tool_duration.csv" in text and "time_left.csv" in text and "{" not in text
