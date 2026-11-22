"""Decision rules for the online predictor, fixed before any online model was fitted or scored.

Setting. Two events per request (analysis/dag/online.py): A = request admitted (sets priority),
R = response completed (retains or evicts KV). Each is judged on its own.

Data protocol.
  fit        Aug 23-27     tuning        Aug 28-29
  validation Aug 30-Sep 1  read ONCE, after the model and every setting are frozen, then refit on Aug 23-29
  held out   Sep 2-5       never read here (the offline predictor's frozen split)
  At most TUNING_BUDGET configurations are tried on the tuning days.

Offline gate (per event).
  Baseline  the step-only table: the empirical distribution of requests remaining given the step and
            that the turn survived to it. At R the baseline also knows whether the response called any
            tool, because that alone decides whether the turn ended (a response without tools ends the
            turn in 12,569 of 12,572 cases in the training week; one with tools in 0.3%): free information
            the model must not be credited for.
  Metric    CRPS of the predicted distribution of `rem_after`, each turn weighted once (`w_turn`),
            averaged over decision steps t in GATE_STEPS.
  Pass      relative CRPS reduction over the baseline >= OFFLINE_MIN and the lower end of a 95% paired
            bootstrap interval over sessions (BOOTSTRAP draws) above 0.
  Fail      stop that event: report it as a negative result, no simulator run for it.

Simulator gate (events that pass offline), on the validation days, eviction/admission as in run_gate.py.
  Attainment points the online policy gains over the same policy with history, as a share of the gain of
  the same policy with the true remaining-call count (admission: llf/srpt_calls; eviction: the oracle
  eviction), averaged over the realistic-KV cells (1M, 2M, 4M tokens; all loads and tightnesses).
  Pass      share >= SIM_MIN and no worse than history on p95 turn latency.
"""

OFFLINE_MIN = 0.05          # relative CRPS reduction over the step-only table
GATE_STEPS = (1, 2, 5, 10)  # decision steps (requests already started in the turn) the gate averages over
BOOTSTRAP = 1000
SIM_MIN = 0.25              # share of the gap to the true remaining-call count
TUNING_BUDGET = 12          # configurations tried on the tuning days
FIT_DAYS = ("2026-08-23", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27")
TUNE_DAYS = ("2026-08-28", "2026-08-29")
VALIDATION_DAYS = ("2026-08-30", "2026-08-31", "2026-09-01")
HELDOUT_DAYS = ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05")
