"""An event-A next-tool classifier for the repository's evaluator rows."""

from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from prediction.predictor.online_baseline import step_bucket
from prediction.predictor.tool_baseline import CLASSES, CODE

TOOL_FEATURES = ("prev1_cat", "prev2_cat", "prev3_cat", "dom")
NUMERIC = ("step", "step_bucket", "prev1_width", "prev1_errors", "prev1_result_tokens")
CATEGORICAL = (*TOOL_FEATURES, "harness")
SEED = 42
PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(CLASSES),
    "tree_method": "hist",
    "device": "cpu",
    "seed": SEED,
    "nthread": 4,
}


@dataclass(frozen=True)
class Encoder:
    """Fixed tool vocabulary and a harness vocabulary learned from fit rows."""

    harnesses: tuple[str, ...]

    @classmethod
    def fit(cls, rows):
        return cls(tuple(sorted({r["harness"] for r in rows if r.get("harness")})))

    def transform(self, rows) -> np.ndarray:
        # Each one-hot block ends with separate missing and unknown columns.
        blocks = [CLASSES] * len(TOOL_FEATURES) + [self.harnesses]
        widths = [len(values) + 2 for values in blocks]
        matrix = np.zeros((len(rows), len(NUMERIC) + sum(widths)), dtype=np.float32)
        for col, name in enumerate(NUMERIC):
            if name == "step_bucket":
                matrix[:, col] = [step_bucket(r["step"]) for r in rows]
            else:
                matrix[:, col] = [r.get(name) for r in rows]  # None becomes NaN
        offset = len(NUMERIC)
        row_indices = np.arange(len(rows))
        for name, values, width in zip(CATEGORICAL, blocks, widths):
            codes = {value: offset + i for i, value in enumerate(values)}
            codes.update({None: offset + width - 2, "": offset + width - 2})
            columns = [codes.get(r.get(name), offset + width - 1) for r in rows]
            matrix[row_indices, np.asarray(columns, dtype=int)] = 1
            offset += width
        return matrix

    def specification(self):
        return {
            "numeric": list(NUMERIC),
            "categorical": list(CATEGORICAL),
            "classes": list(CLASSES),
            "harnesses": list(self.harnesses),
            "categorical_encoding": "one-hot with separate missing and unknown columns",
            "numeric_missing": "NaN",
        }


def training_matrix(rows, encoder, params):
    labels = np.fromiter((CODE[row["next_cat"]] for row in rows), dtype=np.int32, count=len(rows))
    return xgb.QuantileDMatrix(
        encoder.transform(rows), label=labels, nthread=params["nthread"]
    )


@dataclass
class XGBoostToolPredictor:
    encoder: Encoder
    booster: xgb.Booster

    @classmethod
    def fit(cls, encoder, matrix, params, rounds):
        return cls(encoder, xgb.train(params, matrix, num_boost_round=rounds))

    def predict(self, rows) -> np.ndarray:
        probabilities = self.booster.inplace_predict(
            self.encoder.transform(rows), strict_shape=True
        ).astype(np.float64)
        if (probabilities.shape != (len(rows), len(CLASSES))
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("XGBoost returned invalid next-tool probabilities")
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def proba(self, row) -> np.ndarray:
        return self.predict([row])[0]
