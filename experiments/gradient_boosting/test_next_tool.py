import csv
import gzip

import numpy as np
import pytest

pytest.importorskip("xgboost")

from analysis.dag import online, splits, tool_eval
from analysis.dag.dataset import UserHistory
from prediction.predictor.tool_baseline import CLASSES, CODE, END
from tests.test_dataset import three_rounds

from .next_tool import check_splits, load_rows
from .predictor import NUMERIC, PARAMS, Encoder, XGBoostToolPredictor, training_matrix


def write_rows(path, rows):
    with gzip.open(path, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=online.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_probability_columns_match_evaluator_with_an_absent_training_class():
    rows = [{"step": i, "harness": "h", "next_cat": category}
            for i, category in enumerate(CLASSES) if category != "Other"] * 10
    encoder = Encoder.fit(rows)
    params = PARAMS | {"nthread": 1, "max_depth": 6, "min_child_weight": 0, "eta": 0.5}
    model = XGBoostToolPredictor.fit(encoder, training_matrix(rows, encoder, params), params, 30)
    probabilities = model.predict(rows)
    assert probabilities.shape == (len(rows), 17)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert (probabilities >= 0).all()
    assert probabilities.argmax(axis=1).tolist() == [CODE[r["next_cat"]] for r in rows]
    np.testing.assert_allclose(model.proba(rows[-1]), probabilities[-1])
    assert tool_eval.scores(model, [rows[-1]])["top1"][0] == 1


def test_encoding_is_consistent_and_has_distinct_missing_and_unknown_columns():
    rows = [{"step": 0, "harness": "known", "prev1_cat": "Read", "prev1_width": 1},
            {"step": 0, "harness": None, "prev1_cat": None},
            {"step": 0, "harness": "unseen", "prev1_cat": "new category"}]
    encoder = Encoder.fit(rows[:2])
    matrix = encoder.transform(rows)
    np.testing.assert_array_equal(matrix, np.vstack([encoder.transform([row]) for row in rows]))
    np.testing.assert_array_equal(matrix[:, -3:], np.eye(3))
    np.testing.assert_array_equal(matrix[:, len(NUMERIC) + len(CLASSES):][:, :2],
                                  [[0, 0], [1, 0], [0, 1]])
    assert encoder.harnesses == ("known",)
    assert np.isnan(matrix[1:, NUMERIC.index("prev1_width")]).all()


def test_future_changes_label_but_not_encoded_event_a(tmp_path):
    turns = three_rounds()
    history = UserHistory([turns])
    day = turns[0]["day"]
    good_file = tmp_path / "good.csv.gz"
    changed_file = tmp_path / "changed.csv.gz"
    write_rows(good_file, online.rows_of(turns, history))
    write_rows(changed_file, online.rows_of(online.scramble(turns, 0, 1, "A"), history))
    good = load_rows(good_file, (day,))
    changed = load_rows(changed_file, (day,))
    assert [row["next_cat"] for row in good] == ["Read", "Edit", END]
    assert changed[1]["next_cat"] == "Other"
    encoder = Encoder.fit(good)
    np.testing.assert_array_equal(encoder.transform(good[:2]), encoder.transform(changed))


def test_window_loader_stops_before_parsing_later_outcomes(tmp_path):
    turns = three_rounds()
    day = splits.FIT_DAYS[-1]
    turns[0]["day"] = day
    path = tmp_path / "rows.csv.gz"
    write_rows(path, online.rows_of(turns, UserHistory([turns])))
    with gzip.open(path, "at") as target:
        target.write(f"s,u,{splits.VALIDATION_DAYS[0]},invalid outcome row\n")
        target.write("unparseable line\n")
    assert len(load_rows(path, (day,))) == 3


def test_duplicate_sessions_across_days_are_rejected():
    fit = [{"day": splits.FIT_DAYS[-1], "session": "same"}]
    with pytest.raises(ValueError, match="sessions overlap"):
        check_splits(fit, [{"day": splits.TUNE_DAYS[0], "session": "same"}])


def test_final_command_requires_explicit_approval(monkeypatch):
    from . import next_tool

    monkeypatch.setattr("sys.argv", ["next_tool", "--final-evaluation"])
    with pytest.raises(SystemExit) as error:
        next_tool.main()
    assert error.value.code == 2
