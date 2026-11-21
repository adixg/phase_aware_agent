"""Regression checks for the empirical admission forecast."""

import unittest
from pathlib import Path

from prediction.predictor.empirical_baseline import HORIZON, EmpiricalPredictor
from prediction.predictor.evaluate_empirical_baseline import load_admissions


def admission(**changes):
    row = dict(
        horizon=HORIZON, event="admission", cutoff_ms=42,
        features=dict(harness="h", model="m", initial_prompt_estimate=30),
        feature_available_ms=dict(harness=42, model=42, initial_prompt_estimate=42),
    )
    return row | changes


class EmpiricalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            (("h", "m", 20), (1, 0, 10)),
            (("h", "m", 30), (4, 9, 40)),
            (("h", "n", 30), (5, 10, 50)),
            (("k", "m", 20), (6, 11, 60)),
        ]
        self.model = EmpiricalPredictor(self.rows, prompt_edges=(30,), strength=2)

    def test_joint_atoms_and_backoff(self):
        forecast = self.model.predict(admission(), HORIZON)
        self.assertEqual(forecast.support, {"group_depth": 3, "group_rows": 1})
        self.assertAlmostEqual(sum(weight for _, weight in forecast.distribution), 1)
        weights = dict(forecast.distribution)
        self.assertGreater(weights[(4, 9, 40)], weights[(1, 0, 10)])
        self.assertTrue(set(forecast.sample(100, seed=10)) <= {row for _, row in self.rows})
        for inputs, depth in [
            (("h", "n", 20), 2), (("h", "new", 30), 1), (("other", "m", 30), 0),
        ]:
            self.assertEqual(self.model.distribution(inputs)[2]["group_depth"], depth)

    def test_weighted_crps_matches_definition(self):
        atoms, marginals, _ = self.model.distribution(("h", "m", 30))
        for index, outcome in enumerate((2, 8, 45)):
            error = sum(weight * abs(row[index] - outcome) for row, weight in atoms)
            dispersion = sum(
                left_weight * right_weight * abs(left[index] - right[index])
                for left, left_weight in atoms for right, right_weight in atoms
            )
            self.assertAlmostEqual(marginals[index].crps(outcome), error - dispersion / 2)

    def test_reject_future_features_and_unsupported_updates(self):
        row = admission()
        row["feature_available_ms"]["initial_prompt_estimate"] = 43
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.model.predict(row)
        with self.assertRaisesRegex(ValueError, "admission"):
            self.model.predict(admission(event="root_response_completed"))

    def test_evaluator_refuses_test_before_file_access(self):
        with self.assertRaisesRegex(ValueError, "Only train and validation"):
            load_admissions(Path("unused"), "test")


if __name__ == "__main__":
    unittest.main()
