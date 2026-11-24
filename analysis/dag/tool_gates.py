"""Decision rules for the next-tool predictor, fixed before any tool model was fitted or scored.

Setting. At event A of every request (analysis/dag/online.py: the request is admitted, its response not yet
generated), predict which tool the response will call, as a probability over 17 classes: the 16 tool
categories of the trace plus END, meaning the response calls no tool and the turn ends. A round with several
tools is labelled by its first tool's category (the same convention as the `prev1_cat` feature). The label of
the A row of request k is read from the R row of the same request, which is when the answer is known.

Features are what A knows (the same view as every online row): the step, the previous rounds' tool
categories, widths, errors, result sizes and finish reasons, the turn's tool-category counts, harness, model
and account history. Nothing of request k's own response can reach them (the online audit, 0 of 6,000).

Data protocol (the same days as the online and session forecasters).
  fit        Aug 23-27     tuning        Aug 28-29
  validation Aug 30-Sep 1  read ONCE, after the model and every setting are frozen, then refit on Aug 23-29
  held out   Sep 2-5       never read here (the offline predictor's frozen split)
  At most TUNING_BUDGET configurations are tried on the tuning days (analysis/dag/tool_eval.py logs them).
  The grid and the rules below were written before any tool model was scored. Only label counts on the fit
  days (Aug 23-27) were looked at beforehand: 17 classes, END 8.6%, no other class above 14.0%, and a
  previous-category rule reaching 40% top-1 against 12% for the most common class overall.

Offline gate.
  Baseline  the previous-tool table: the empirical distribution of the next class given the step bucket and the
            previous round's first tool category (the first-order Markov table the simulator's history policy
            is built from). Its smoothing strength is tuned on its own small grid, which does not use up the
            model's budget and can only make it harder to beat. Every model chain starts from the same two
            levels, so a model with no extra level equals the baseline.
  Metric    multiclass Brier score of the predicted distribution (sum over the 17 classes of the squared
            error to the one-hot outcome), each turn counted once per decision step, the relative reduction
            over the baseline averaged over the decision steps in GATE_STEPS. Log loss, top-1 and top-3
            accuracy are reported beside it, not gated.
  Pass      relative Brier reduction >= OFFLINE_MIN and the lower end of a 95% paired bootstrap interval over
            sessions (BOOTSTRAP draws) above 0.
  No simulator gate. Next-tool probabilities do not drive a scheduler decision by themselves: they feed the
  tool-duration predictor (which tool is running) and the remaining-time rollout. They are judged there.
"""

OFFLINE_MIN = 0.05          # relative Brier reduction over the previous-tool table
GATE_STEPS = (0, 1, 2, 5, 10)   # decision steps (requests already started in the turn) the gate averages over
BOOTSTRAP = 1000
TUNING_BUDGET = 12          # configurations tried on the tuning days
FIT_DAYS = ("2026-08-23", "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27")
TUNE_DAYS = ("2026-08-28", "2026-08-29")
VALIDATION_DAYS = ("2026-08-30", "2026-08-31", "2026-09-01")
HELDOUT_DAYS = ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05")
