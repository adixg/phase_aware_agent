"""Decision rules for the time-left predictor (predictor 3) on the 16-week split, fixed before any model was scored.

Setting. At the start of each user turn u of a session (analysis/dag/session.py), predict the time the system
itself has left in the session, the current turn included:

  rem_path_s   modelled GPU work (the replay's L40S cost model) plus the round times between requests, for every
               request from this turn's first to the session's last, in whole seconds. The user's pauses are
               in neither: they are the user's time, not the system's, and the simulator fixes them.
  rem_calls    requests left, the current turn included (secondary, reported, not gated)

Turn 0 is the session's arrival, where a session-level deadline is set. Features are what is known when the
turn starts (analysis/dag/session.py): the turn index, the session's earlier turns (sizes, durations, pauses),
the pause before this turn, this turn's first prompt size, harness and model, and the user's earlier sessions
(turns, requests and, new in this version, time per session) that had ended by then. A leakage audit rebuilt
3,000 turns from a scrambled future with 0 changed features.

What was looked at beforehand, all on the fit days (May 17 - Aug 8) and none of it a model outcome: at a session's
arrival 6,106 sessions, rem_path_s median 564 s, mean 2,624 s, 90th percentile 4,053 s, 99th percentile
47,674 s, maximum 449,322 s. The target is extremely heavy-tailed, so the CRPS is dominated by the longest
sessions. Earlier, on the two-week data, the same kind of forecaster for the requests and turns still to come
gained 2.3% (validation) against a 5% bar (analysis/dag/session_gates.py); that result stands and this one
replaces nothing, it asks the same question of a time target, with user history by time and 16 weeks.

Data protocol. The 16-week split (analysis/dag/splits.py): fit May 17 - Aug 8, tuning Aug 9-15, validation Aug 16-22
read ONCE, then the model is refitted on fit + tuning; Aug 23 - Sep 1 unused; Sep 2-5 held out. A session belongs
to the day it starts. At most TUNING_BUDGET configurations are tried on the tuning days
(analysis/dag/session_eval_16w.py logs them); the grid is written there before any scoring.

Offline gate.
  Baseline  the turn-index-only table: the empirical distribution of rem_path_s among training sessions that
            reached turn u (survival-conditioned). Its smoothing strength is tuned on its own small grid, which
            does not use up the model's budget and can only make it harder to beat.
  Metric    CRPS of the predicted distribution of rem_path_s, in seconds, each session counted once at a fixed
            turn, the relative reduction over the baseline averaged over the decision turns in GATE_TURNS. Every
            chain starts from a level that exists at turn 0 (user history), so every chain can act at every
            decision turn.
  Pass      relative CRPS reduction >= OFFLINE_MIN and the lower end of a 95% paired bootstrap interval over
            sessions (BOOTSTRAP draws) above 0.
  Fail      report it as a negative result.

No simulator gate in this step. The forecast of the whole session's time sets a session-level deadline's slack at
arrival and the priority of its requests; plugging it into the simulator is a separate, later test with its own
rule, as the online forecaster's was.
"""

from analysis.dag.splits import FIT_DAYS, HELDOUT_DAYS, TUNE_DAYS, VALIDATION_DAYS  # noqa: F401

OFFLINE_MIN = 0.05          # relative CRPS reduction over the turn-index-only table
GATE_TURNS = (0, 1, 2, 5)   # decision turns the gate averages over
BOOTSTRAP = 1000
TUNING_BUDGET = 12          # configurations tried on the tuning days
