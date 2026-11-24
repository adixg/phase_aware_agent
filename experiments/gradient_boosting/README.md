# XGBoost next-tool prediction

The model implements `proba(row)` for rows returned by `analysis.dag.tool_eval.load`.
Its 17 output columns follow `prediction.predictor.tool_baseline.CLASSES` exactly.
The canonical loader labels each A row from the first tool category in the
matching R row, or `__none__` for turn end. Only A features enter the model.

| Features | Why they are available at admission |
|---|---|
| Request step and the baseline's step bucket | The request's position in the current turn is known. |
| Previous three tool categories | The earlier request rounds have already finished. |
| Harness | The session already identifies its harness. |
| Dominant tool category | Counts reflect only tool calls from completed rounds in the current turn. |
| Previous round's width, error count and result token count | The previous round returned before this request was admitted. |

Tool categories use a fixed one-hot vocabulary. Harness categories use a sorted
vocabulary learned on fit only. Each categorical feature has separate missing
and unknown columns. Missing numeric values use XGBoost's native NaN handling.
The encoder reads only the fields above. Session IDs, dates, R-event feature
values and labels are excluded from the feature matrix.

## Tuning protocol

Fit May 17–Aug 8, 2026, and tune Aug 9–15. Fit uses all A requests. Score
steps 3, 5, 10 and 20 with the repository evaluator. Steps are weighted
equally for reported macro metrics. The selection objective is the mean
of the four per-step relative Brier reductions. Top-k and log loss follow
the repository's tie convention and probability floor. The smoothed
previous-tool baseline selects its smoothing strength on the same tuning
rows using the repository's grid.

The tuning artifacts live under
`data/freeinference/experiments/gradient_boosting/next_tool/`. The grid is
written before scoring. The run also writes per-step results, the May 17–Aug 8
fitted booster, and the frozen configuration. Process peak RSS includes
loading, baseline selection, matrix creation and training allocations. Candidate
times measure booster fitting with a shared prebuilt matrix.

From the repository root, with XGBoost 3.1.3, NumPy, orjson, zstandard and
pytest installed:

```sh
/tmp/phase-aware-xgb-runtime/bin/python -m pytest -q experiments/gradient_boosting/test_next_tool.py tests/test_online.py tests/test_tool.py tests/test_splits.py
/tmp/phase-aware-xgb-runtime/bin/python -m experiments.gradient_boosting.next_tool --tune
```

## Final evaluation after approval

Aug 23–Sep 1, 2026 is the declared final window. Some days were previously used
in other team experiments. Aug 16–22 was already viewed by the team. Sep 2–5
remains held out.

After approval, run from the repository root:

```sh
/tmp/phase-aware-xgb-runtime/bin/python -m experiments.gradient_boosting.next_tool --final-evaluation --approve-final-evaluation
```

This refits XGBoost and the previous-tool baseline on May 17–Aug 15 with the
frozen configuration, then scores Aug 23–Sep 1 once. It creates an exclusive
marker before loading final outcomes. It writes the per-step metrics and the
repository's paired session-bootstrap interval to `final_evaluation.json`.

For the future Softmax comparison, use the same `load_rows` request rows,
`FIT_DAYS + TUNE_DAYS` for refitting, `FINAL_DAYS` for scoring and
`analysis.dag.tool_eval_16w.run_gate` for metrics and bootstrap. All models
must receive the same A requests at the same four steps in the canonical
class order. Coordinate their frozen configurations and scoring before the
final read.

## Recorded results

The [tuning grid](results/grid.json), [tuning metrics](results/tuning_results.json),
[frozen configuration](results/frozen.json), and [final metrics](results/final_evaluation.json)
are snapshots of the completed run. The input was the FreeInference online
dataset built from Hugging Face revision
`bb26cf72c9c20f2b05f765d88548156a74d3c253` with the repository's DAG pipeline.
The run used Python 3.10.22, XGBoost 3.1.3, and NumPy 2.2.6.

The default candidate improved tuning Brier by 8.71% over the smoothed
previous-tool baseline. On the previously exposed Aug 23–Sep 1 window, it
improved Brier by 8.23% (95% paired session-bootstrap CI 7.67%–8.80%),
passing the 5% criterion. The evaluation used 24,864 requests at steps
3, 5, 10, and 20. The final evaluation has already been run for this freeze.
