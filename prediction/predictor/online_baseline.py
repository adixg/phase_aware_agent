"""Online forecasting of a running turn: the offline predictor's empirical hierarchy, conditioned on progress.

The admission predictor (empirical_baseline.py) mixes whole outcome tuples along harness > model > prompt
bucket. Here the hierarchy starts from where the turn is. A training row exists at step k only for a turn
that reached step k, so conditioning on the step bucket is conditioning on survival; deeper levels add
what the running turn has shown (last tool category, width, harness, history). Each level contributes
n / (n + strength) of the mass, as in the admission predictor, and the rest comes from its parent, so a
sparse cell falls back to a coarser one.

Two events (analysis/dag/online.py), fitted separately: A, request admitted, and R, response completed.
Two targets, each its own distribution:
  rem_after   requests still to start after the current one (0 = the turn ends with it)
  stall_s     wall time of the current request's round, for requests that are not last

Rows are the dicts analysis/dag/online.py writes, numbers parsed (None for missing).
"""

from __future__ import annotations

from dataclasses import dataclass

from .empirical_baseline import EmpiricalPredictor, Marginal

QUANTILES = (0.1, 0.5, 0.9)
STEP_EDGES = [16, 20, 30, 50, 80, 120]          # exact up to 15, then these bins
NUMERIC_LEVELS = {"stall": "prev1_round_s", "result": "prev1_result_tokens", "hist": "user_prior_mean_rounds",
                  "sess": "sess_prior_mean_rounds"}
LEVELS = ("tools", "step", "harness", "model", "cat", "width", "err", *NUMERIC_LEVELS)


def step_bucket(step: int) -> int:
    return step if step < STEP_EDGES[0] else STEP_EDGES[0] + sum(step >= e for e in STEP_EDGES[1:])


def quantile_edges(values: list[float], parts: int = 3) -> tuple[float, ...]:
    v = sorted(x for x in values if x is not None)
    return tuple(sorted({v[len(v) * i // parts] for i in range(1, parts)})) if v else ()


def bucket(x: float | None, edges: tuple[float, ...]):
    return None if x is None else sum(x >= e for e in edges)


def label(row: dict, level: str, edges: dict) -> object:
    """The value of one hierarchy level for a row. `tools` is whether the current response called any tool
    (known at R, where it decides whether the turn ended; a constant at A)."""
    if level == "tools":
        return None if row["event"] == "A" else (row["prev1_width"] or 0) > 0
    if level == "step":
        return step_bucket(row["step"])
    if level == "harness":
        return row["harness"]
    if level == "model":
        return row["model"]
    if level == "cat":
        return row["prev1_cat"]
    if level == "width":
        return None if row["prev1_width"] is None else min(row["prev1_width"], 4)
    if level == "err":
        return None if row["prev1_errors"] is None else row["prev1_errors"] > 0
    return bucket(row[NUMERIC_LEVELS[level]], edges[level])


PAD = (0, 0)     # EmpiricalPredictor builds one marginal per entry of its three TARGETS: pad ours to three


class Hierarchy(EmpiricalPredictor):
    """EmpiricalPredictor whose groups are the prefixes of an ordered list of levels, on one target.
    Its smoothing and marginals are reused as they are (rows are (inputs, (value,)), padded here)."""

    def __init__(self, rows, *, strength):
        super().__init__([(x, (v, *PAD)) for x, (v,) in rows], prompt_edges=(), strength=strength)

    def group_keys(self, inputs):
        yield ()
        for i in range(1, len(inputs) + 1):
            yield tuple(inputs[:i])


@dataclass(frozen=True)
class Forecast:
    mean_after: float
    quantiles_after: dict
    p_last: float
    mean_stall_s: float | None
    quantiles_stall_s: dict | None
    support: dict


def mean_of(m: Marginal) -> float:
    return m.weighted_sum[-1] / m.mass[-1]


class OnlineForecaster:
    """Fit on rows of one event; `levels` is the ordered chain after the step, e.g. ("tools", "harness", "cat")."""

    def __init__(self, rows: list[dict], event: str, levels: tuple[str, ...], strength: float = 20):
        assert all(r["event"] == event for r in rows)
        self.event, self.levels, self.strength = event, tuple(levels), strength
        self.edges = {lv: quantile_edges([r[col] for r in rows]) for lv, col in NUMERIC_LEVELS.items() if lv in levels}
        self.after = Hierarchy([(self.inputs(r), (r["rem_after"],)) for r in rows], strength=strength)
        moving = [r for r in rows if r["stall_s"] is not None]
        self.stall = Hierarchy([(self.inputs(r), (round(r["stall_s"] * 1000),)) for r in moving], strength=strength)

    def inputs(self, row: dict) -> tuple:
        return (step_bucket(row["step"]),) + tuple(label(row, lv, self.edges) for lv in self.levels)

    def marginals(self, row: dict):
        """(distribution of rem_after, support) and the stall distribution, from the deepest populated group."""
        x = self.inputs(row)
        _, after, support = self.after.distribution(x)
        _, stall, _ = self.stall.distribution(x)
        return after[0], stall[0], support

    def predict(self, row: dict) -> Forecast:
        after, stall, support = self.marginals(row)
        p_last = after.mass[1] if after.values[0] == 0 else 0.0
        return Forecast(mean_of(after), {p: after.quantile(p) for p in QUANTILES}, p_last,
                        mean_of(stall) / 1000, {p: stall.quantile(p) / 1000 for p in QUANTILES}, support)
