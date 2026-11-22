"""Andrew's admission forecaster (prediction/predictor/empirical_baseline.py) as the simulator's
remaining-calls estimate.

The predictor maps (harness, model, prompt size) at a turn's admission to an empirical joint
distribution over (LLM requests, tool calls, duration). Here it is fitted on training-week turns and
read through one number: the expected number of requests still to come. As a turn progresses the
distribution is conditioned on having survived: at step k the turn has more than k requests, so the
estimate is E[N - k | N > k] (and 1 once k is past everything seen in training).

Two fits, because the predictor was built for the first user turn of a session only (about 7% of
turns, but the long ones):
  first  fitted on and applied to first turns; every later turn keeps the Cheap table's estimate
  all    the same algorithm fitted on every turn, features read at that turn's first request

Differences from the predictor's own pipeline, all from the replay's compact files: the prompt size is
the provider's prompt-token count of the turn's first request (the pipeline sums five role token
counts, which the compact files do not keep), and request counts exclude subagent requests.
"""

from __future__ import annotations

import numpy as np

from analysis.replay.sim import Session
from prediction.predictor.empirical_baseline import EmpiricalPredictor
from prediction.predictor.evaluate_empirical_baseline import prompt_edges

BUCKETS, STRENGTH = 2, 20     # the configuration the predictor's validation run selected


def fit_rows(sessions: list[Session], first_only: bool) -> list[tuple]:
    return [((s.harness, t.model, t.prompt), (t.calls, t.tools, t.duration_ms))
            for s in sessions for i, t in enumerate(s.turns) if i == 0 or not first_only]


class Forecaster:
    def __init__(self, train: list[Session], first_only: bool, buckets: int = BUCKETS, strength: int = STRENGTH):
        rows = fit_rows(train, first_only)
        self.first_only = first_only
        self.model = EmpiricalPredictor(rows, prompt_edges=prompt_edges(rows, buckets), strength=strength)
        self._survival: dict[tuple, tuple] = {}

    def _arrays(self, inputs: tuple):
        """Sorted request counts with, for each, the mass and the mass-weighted count at or above it."""
        if inputs not in self._survival:
            atoms, _, _ = self.model.distribution(inputs)
            n = np.array([row[0] for row, _ in atoms], dtype=float)
            w = np.array([weight for _, weight in atoms])
            order = np.argsort(n)
            n, w = n[order], w[order]
            self._survival[inputs] = (n, w[::-1].cumsum()[::-1], (w * n)[::-1].cumsum()[::-1])
        return self._survival[inputs]

    def remaining_calls(self, inputs: tuple, step: int) -> float:
        """Expected requests left, the current one included, given the turn is already past `step`."""
        n, mass, weighted = self._arrays(inputs)
        i = int(np.searchsorted(n, step, side="right"))     # atoms with N > step
        if i == len(n):
            return 1.0
        return max(1.0, weighted[i] / mass[i] - step)

    def annotate(self, sessions: list[Session]) -> None:
        """Set each request's forecast. A first-only forecaster leaves later turns at 0 (= use Cheap)."""
        field = "pred_first" if self.first_only else "pred_all"
        for s in sessions:
            for r in s.reqs:
                if self.first_only and r.turn > 0:
                    continue
                t = s.turns[r.turn]
                setattr(r, field, self.remaining_calls((s.harness, t.model, t.prompt), r.step))
