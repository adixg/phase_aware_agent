"""Which tool a request will call: the online hierarchy, with a probability over the tool categories.

At event A of a request (analysis/dag/online.py) the response has not been generated. The forecaster gives a
probability for each of 17 classes: the 16 tool categories plus END (the response calls no tool, so the
turn ends). A round with several tools is labelled by its first tool's category.

It is the same smoothed hierarchy as the online and session forecasters: groups are the prefixes of an
ordered list of levels, starting from the step bucket, and each level contributes n / (n + strength) of
the mass with the rest coming from its parent, so a sparse cell falls back to a coarser one. The class is
coded as an integer so the hierarchy's marginal over one target gives the distribution directly.

The previous-tool table (the baseline) is the chain ("cat",): step bucket, then the previous round's first
tool category. Every model chain starts with "cat", so it contains the baseline.

The level "dom" is the tool category the turn has used most so far (analysis/dag/tool_eval.py works it out from the
turn's tool-category counts), None before the first tool.

Rows are the dicts analysis/dag/tool_eval.py builds from online_rows.csv.gz, numbers parsed (None for missing).
"""

from __future__ import annotations

import numpy as np

from analysis.dag.dataset import CATS

from .online_baseline import Hierarchy, bucket, quantile_edges, step_bucket

END = "__none__"
CLASSES = tuple(CATS) + (END,)
CODE = {c: i for i, c in enumerate(CLASSES)}
LEVELS = ("cat", "cat2", "cat3", "err", "finish", "width", "result", "harness", "model", "dom")


class ToolForecaster:
    """Fit on A rows; `levels` is the ordered chain after the step, e.g. ("cat", "cat2")."""

    def __init__(self, rows: list[dict], levels: tuple[str, ...] = ("cat",), strength: float = 20):
        assert all(lv in LEVELS for lv in levels), levels
        self.levels, self.strength = tuple(levels), strength
        self.edges = {"result": quantile_edges([r["prev1_result_tokens"] for r in rows])} if "result" in levels else {}
        self.h = Hierarchy([(self.inputs(r), (CODE[r["next_cat"]],)) for r in rows], strength=strength)

    def label(self, row: dict, level: str):
        if level == "cat":
            return row["prev1_cat"]
        if level == "cat2":
            return row["prev2_cat"]
        if level == "cat3":
            return row["prev3_cat"]
        if level == "err":
            return None if row["prev1_errors"] is None else row["prev1_errors"] > 0
        if level == "finish":
            return row["prev1_finish"]
        if level == "width":
            return None if row["prev1_width"] is None else min(row["prev1_width"], 4)
        if level == "result":
            return bucket(row["prev1_result_tokens"], self.edges["result"])
        return row[level]                       # harness, model, dom (the turn's most used tool so far, None before any)

    def inputs(self, row: dict) -> tuple:
        return (step_bucket(row["step"]),) + tuple(self.label(row, lv) for lv in self.levels)

    def proba(self, row: dict) -> np.ndarray:
        """Probability of each class in CLASSES order, from the deepest populated group."""
        _, marginals, _ = self.h.distribution(self.inputs(row))
        m = marginals[0]
        p = np.zeros(len(CLASSES))
        for i, v in enumerate(m.values):
            p[int(v)] = m.mass[i + 1] - m.mass[i]
        return p / p.sum()
