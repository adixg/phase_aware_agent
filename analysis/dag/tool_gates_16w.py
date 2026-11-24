"""Decision rules for the next-tool predictor on the 16-week split, fixed before any model was scored on it.

This replaces the first tool gate (tool_gates.py, Aug 23 - Sep 1) for the rebuilt predictor. What it keeps: the
setting, the 17 classes, the previous-tool baseline, the multiclass Brier metric, the 5% bar, the 12-configuration
budget. What it changes, and why:

  Data. The 16-week split (analysis/dag/splits.py): fit May 17 - Aug 8, tuning Aug 9-15, validation Aug 16-22,
        read once; the days the first models used (Aug 23 - Sep 1) are not used; Sep 2-5 stays held out. The
        first tool model was tuned on Aug 28-29 (best: the previous three tools, +2.0% over the baseline,
        under the bar) and its validation days were never read; its files stay in data/freeinference/tools.
        Learning curve on fit days, fitted on 1 to 4 days: +3.2%, +3.5%, +3.9%, +4.1% (steps 2, 5, 10),
        rising, with only 40% of the last-three-tools contexts of the test days seen 20 times in training.
        More data is the reason for this version.
  Steps. GATE_STEPS = (3, 5, 10, 20), not (0, 1, 2, 5, 10). A chain of up to three previous tools cannot differ
        from the baseline before step 3 (the tools before that do not exist yet), so steps 0 to 2 only dilute
        the average. This choice was made after seeing the first model's tuning-days result by step (0.0% at
        steps 0 and 1, +2.4% at 2, +4.6% at 5, +2.8% at 10), so it is not blind; the reason is structural and
        every model below is judged only where it can act.
  Model. One level is new: `dom`, the tool the turn has used most so far (from counts the rows already hold).
        The grid is written below, before any scoring.

Setting. At event A of every request (analysis/dag/online.py: the request is admitted, its response not yet
generated), predict which tool the response will call, as a probability over 17 classes: the 16 tool categories
plus END, meaning the response calls no tool and the turn ends. A round with several tools is labelled by its first
tool's category. The label comes from the R row of the same request.

Offline gate.
  Baseline  the previous-tool table: the empirical distribution of the next class given the step bucket and the
            previous round's first tool category. Its smoothing strength is tuned on its own small grid, which
            does not use up the model's budget and can only make it harder to beat. Every chain starts from the
            same two levels, so a model with no extra level equals the baseline.
  Metric    multiclass Brier score of the predicted distribution, each turn counted once per decision step, the
            relative reduction over the baseline averaged over GATE_STEPS. Log loss, top-1 and top-3 accuracy
            are reported beside it, not gated.
  Pass      relative Brier reduction >= OFFLINE_MIN and the lower end of a 95% paired bootstrap interval over
            sessions (BOOTSTRAP draws) above 0.
  No simulator gate: next-tool probabilities feed the tool-duration predictor and the remaining-time rollout and
  are judged there.
"""

from analysis.dag.splits import FIT_DAYS, HELDOUT_DAYS, TUNE_DAYS, VALIDATION_DAYS  # noqa: F401

OFFLINE_MIN = 0.05              # relative Brier reduction over the previous-tool table
GATE_STEPS = (3, 5, 10, 20)     # decision steps every chain can act at
BOOTSTRAP = 1000
TUNING_BUDGET = 12              # configurations tried on the tuning days
