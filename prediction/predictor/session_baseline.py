"""Forecasting what a session still holds when a user turn starts: the online hierarchy, on the turn index.

At the start of user turn u (analysis/dag/session.py) two targets are forecast, each its own distribution:
  fut_calls   requests in the user turns after this one
  fut_turns   how many user turns come after this one
A training row exists at turn u only for a session that reached turn u, so conditioning on the turn index is
conditioning on survival, as the step bucket is in online_baseline.py. Deeper levels add what the session and
the user have shown so far (the last turn's size, the pause, the user's earlier sessions), each contributing
n / (n + strength) of the mass, the rest coming from its parent, so a sparse cell falls back to a coarser one.

Rows are the dicts analysis/dag/session.py writes, numbers parsed (None for missing).
"""

from __future__ import annotations

from .online_baseline import Hierarchy, bucket, mean_of, quantile_edges, step_bucket

NUMERIC_LEVELS = {"hist": "user_mean_turns", "histcalls": "user_mean_calls", "last": "prior_last_calls",
                  "prior": "prior_mean_calls", "idle": "idle_before_s", "prompt": "first_prompt",
                  "wall": "prior_last_wall_s", "elapsed": "elapsed_s"}
CATEGORICAL_LEVELS = {"harness": "harness", "model": "model"}
LEVELS = (*NUMERIC_LEVELS, *CATEGORICAL_LEVELS)


class SessionForecaster:
    """Fit on rows; `levels` is the ordered chain after the turn index, e.g. ("hist", "last")."""

    def __init__(self, rows: list[dict], levels: tuple[str, ...] = (), strength: float = 20):
        assert all(lv in LEVELS for lv in levels), levels
        self.levels, self.strength = tuple(levels), strength
        self.edges = {lv: quantile_edges([r[col] for r in rows]) for lv, col in NUMERIC_LEVELS.items() if lv in levels}
        self.calls = Hierarchy([(self.inputs(r), (r["fut_calls"],)) for r in rows], strength=strength)
        self.turns = Hierarchy([(self.inputs(r), (r["fut_turns"],)) for r in rows], strength=strength)

    def label(self, row: dict, level: str):
        if level in CATEGORICAL_LEVELS:
            return row[CATEGORICAL_LEVELS[level]]
        return bucket(row[NUMERIC_LEVELS[level]], self.edges[level])

    def inputs(self, row: dict) -> tuple:
        return (step_bucket(row["turn_idx"]),) + tuple(self.label(row, lv) for lv in self.levels)

    def marginals(self, row: dict):
        """(distribution of fut_calls, distribution of fut_turns, support), from the deepest populated group."""
        x = self.inputs(row)
        _, calls, support = self.calls.distribution(x)
        _, turns, _ = self.turns.distribution(x)
        return calls[0], turns[0], support

    def predict(self, row: dict) -> tuple[float, float]:
        """Expected requests and expected turns to come."""
        calls, turns, _ = self.marginals(row)
        return mean_of(calls), mean_of(turns)
