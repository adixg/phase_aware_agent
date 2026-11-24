"""How long the tool round of a request will take: the hierarchy, on the tool that was called.

At event R of a request (analysis/dag/online.py) the response has completed and named its tools, which have not
returned. The round's time, from the response ending to the next request starting, is what the tools (and
whatever the harness does around them) took; every tool of a round carries that same time, so for a round with
several tools it is the round's, not a single tool's. The forecaster gives the distribution of that time in
milliseconds, capped at 3,600 s (the simulator's cap on a long idle gap; 0.04% of rounds hold half of all
round-seconds, so uncapped scores would measure those rounds alone).

Same smoothed hierarchy as the others (each level contributes n / (n + strength) of the mass, the rest from its
parent), but with no step bucket first: the chain starts from the tool category. The baseline is the
category-only table, the chain ("cat",); every model chain starts with "cat", so it contains the baseline.

Levels: cat (the first tool's category), cat2 (the previous round's), width (tools in the round, 4+ pooled),
prevdur (the previous round's time), samecat (the last time a round with the same first category took in this
session; None before the first), harness, model. `prevdur` and `samecat` are quartile buckets.

Rows are the dicts analysis/dag/duration_eval_16w.py builds from online_rows.csv.gz.
"""

from __future__ import annotations

from .online_baseline import Hierarchy, bucket, mean_of, quantile_edges

CAP_MS = 3_600_000
NUMERIC_LEVELS = {"prevdur": "prevdur", "samecat": "samecat"}
LEVELS = ("cat", "cat2", "width", "prevdur", "samecat", "harness", "model")


class DurationForecaster:
    """Fit on R rows that have a round time; `levels` is the ordered chain, e.g. ("cat", "samecat")."""

    def __init__(self, rows: list[dict], levels: tuple[str, ...] = ("cat",), strength: float = 20):
        assert all(lv in LEVELS for lv in levels), levels
        self.levels, self.strength = tuple(levels), strength
        self.edges = {lv: quantile_edges([r[col] for r in rows], 4) for lv, col in NUMERIC_LEVELS.items() if lv in levels}
        self.h = Hierarchy([(self.inputs(r), (r["dur_ms"],)) for r in rows], strength=strength)

    def label(self, row: dict, level: str):
        if level == "width":
            return None if row["width"] is None else min(int(row["width"]), 4)
        if level in NUMERIC_LEVELS:
            return bucket(row[NUMERIC_LEVELS[level]], self.edges[level])
        return row[level]                        # cat, cat2, harness, model

    def inputs(self, row: dict) -> tuple:
        return tuple(self.label(row, lv) for lv in self.levels)

    def marginal(self, row: dict):
        """The distribution of the round's time in milliseconds, from the deepest populated group."""
        _, marginals, _ = self.h.distribution(self.inputs(row))
        return marginals[0]

    def predict(self, row: dict) -> tuple[float, float]:
        """Expected seconds and the median in seconds."""
        m = self.marginal(row)
        return mean_of(m) / 1000, m.quantile(0.5) / 1000
