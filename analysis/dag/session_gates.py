"""Decision rules for the session forecaster, fixed before any session model was fitted or scored.

Why. Under deadlines that cover a whole session, the oracle's gain comes mostly from knowing how many
turns the user will still send, not from the current turn (the oracle decomposition in the status report:
at 1M tokens of KV the true count of later turns closes 64% of the gap from history to oracle, the
current turn 15%). The online forecaster predicts the current turn only. This tests whether the turns to
come can be predicted well enough to matter.

Setting. At the start of each user turn u of a session (the user's message has arrived, its first request
is about to be admitted), predict what the session still holds after this turn:
  fut_calls   requests in the user turns after this one      (the gated target)
  fut_turns   how many user turns come after this one         (secondary, reported, not gated)
Turn 0 is the session's arrival, where a session-level deadline is set: the case that matters most.
Features are what is known at that moment (analysis/dag/session.py): the turn index, the session's earlier
turns (their sizes, durations and pauses), the user's pause before this turn, this turn's first prompt
size, harness and model, and the user's earlier sessions that had ended by then.

Data protocol (the same days as the online forecaster; a session belongs to the day it starts).
  fit        Aug 23-27     tuning        Aug 28-29
  validation Aug 30-Sep 1  read ONCE, after the model and every setting are frozen, then refit on Aug 23-29
  held out   Sep 2-5       never read here (the offline predictor's frozen split)
  At most TUNING_BUDGET configurations are tried on the tuning days (analysis/dag/session_eval.py logs them).
  The grid below was written before any fit, tuning or validation day was scored.
  Sizes: 1,049 sessions on the fit days, 384 on the tuning days, 783 on the validation days. Few sessions
  reach later turns (277 of the validation sessions pass turn 5), so the intervals are wide by construction.

Offline gate.
  Baseline  the turn-index-only table: the empirical distribution of fut_calls among training sessions
            that reached turn u (survival-conditioned). Its mean is what the simulator's history policy
            already uses (Cheap.future_calls). Its smoothing strength is tuned on its own small grid,
            which does not use up the model's budget and can only make it harder to beat.
  Metric    CRPS of the predicted distribution of fut_calls, each session counted once at a fixed turn,
            the relative reduction over the baseline averaged over the decision turns in GATE_TURNS.
  Pass      relative CRPS reduction >= OFFLINE_MIN and the lower end of a 95% paired bootstrap interval
            over sessions (BOOTSTRAP draws) above 0.
  Fail      report it as a negative result: no simulator run.

Simulator gate (if it passes offline), on the validation days, deadlines over whole sessions with a fixed
5 s reply, as in run_gate.py --unit session.
  The forecast's mean replaces the future-calls term of the history policy: the calls left in the current
  user turn (true, as in the `ucur` variant) plus the predicted calls of the turns to come (`upred`).
  Share = (upred - ucur) / (ucount - ucur), where ucount knows the true number of later turns, averaged
  over the realistic-KV cells (1M, 2M, 4M tokens; all loads and tightnesses).
  Metric    mean latency, percent faster than attained service, for the pure shortest-remaining-time policy
            (the metric where the exact count helps); attainment is reported beside it.
  Pass      share >= SIM_MIN and p95 latency no worse than the `ucur` policy.
"""

OFFLINE_MIN = 0.05          # relative CRPS reduction over the turn-index-only table
GATE_TURNS = (0, 1, 2, 5)   # decision turns the gate averages over
BOOTSTRAP = 1000
SIM_MIN = 0.25              # share of the gain from knowing the true number of later turns
TUNING_BUDGET = 12          # configurations tried on the tuning days
FIT_DAYS = ("2026-08-23", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27")
TUNE_DAYS = ("2026-08-28", "2026-08-29")
VALIDATION_DAYS = ("2026-08-30", "2026-08-31", "2026-09-01")
HELDOUT_DAYS = ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05")
