"""Decision rules for the tool-duration predictor (predictor 2) on the 16-week split, fixed before any model was scored.

Setting. At event R of every request that called a tool (analysis/dag/online.py: the response has completed and
named its tools, which have not returned), predict how long the round takes: from the response ending to the
next request starting, in milliseconds. The trace gives every tool of a round the same time, the round's, so
for a round with several tools this is the round's time, not one tool's (per-tool durations cannot be told
apart in this data). The tool category is known at this point; it is what the prediction conditions on.

Features are what R knows (the same view as every online row): the response's first tool category and the number
of tools, the step, the previous round's category and time, harness and model, plus one computed from earlier
rounds only: the last time a round with the same first category took in this session. A round's time becomes
known when the next request starts, so at R of request k only rounds before k count (tests/test_duration.py).

Cap. Round times are extremely heavy-tailed: on the fit days (May 17 - Aug 8, 507,593 rounds) median 1.45 s, mean
22.2 s, 90th percentile 15.7 s, 99th percentile 184 s, maximum 428,491 s (five days). 221 rounds (0.044%) last
over 3,600 s and hold half of all round-seconds. Times are capped at 3,600 s, the simulator's own cap on a long
idle gap, for fitting and for scoring. By category the median runs from 0.73 s (Search) to 2.32 s (Interpreter)
and the 99th percentile from 40 s (Edit) to 311 s (Interpreter), so the category alone separates the scale.

Earlier result. A stall forecast from the online forecaster, on the two-week data, with the chain tuned for
requests remaining rather than for time, scored -0.9% (interval -1.8 to -0.4) against a category-only table.
This version is tuned for time, on its own, with 16 weeks.

Data protocol. The 16-week split (analysis/dag/splits.py): fit May 17 - Aug 8, tuning Aug 9-15, validation Aug 16-22
read ONCE, then the model is refitted on fit + tuning; Aug 23 - Sep 1 unused; Sep 2-5 held out. At most
TUNING_BUDGET configurations are tried on the tuning days (analysis/dag/duration_eval_16w.py logs them); the grid
is written there before any scoring.

Offline gate.
  Baseline  the category-only table: the empirical distribution of the round's time given the response's first tool
            category. Its smoothing strength is tuned on its own small grid, which does not use up the model's
            budget and can only make it harder to beat. Every chain starts from the same level, so a model with no
            extra level equals the baseline.
  Metric    CRPS of the predicted distribution of the capped round time, in seconds, each turn counted once per
            decision step (a request that called a tool), the relative reduction over the baseline averaged over
            the decision steps in GATE_STEPS. Steps start at 1 because the previous round's time does not exist
            at step 0, so every chain can act at every gated step.
  Pass      relative CRPS reduction >= OFFLINE_MIN and the lower end of a 95% paired bootstrap interval over
            sessions (BOOTSTRAP draws) above 0.
  Fail      report it as a negative result.

No simulator gate in this step. The duration distribution sets whether a request's cache is worth keeping while its
tool runs, and feeds the remaining-time rollout; plugging it into the simulator is a separate, later test.
"""

from analysis.dag.splits import FIT_DAYS, HELDOUT_DAYS, TUNE_DAYS, VALIDATION_DAYS  # noqa: F401

OFFLINE_MIN = 0.05              # relative CRPS reduction over the category-only table
GATE_STEPS = (1, 2, 5, 10, 20)  # decision steps (requests already started in the turn) the gate averages over
BOOTSTRAP = 1000
TUNING_BUDGET = 12              # configurations tried on the tuning days
