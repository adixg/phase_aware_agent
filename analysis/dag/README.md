# Building the predictor datasets and your own models

This folder turns the public FreeInference agentic trace into model-ready rows, and holds the three predictors
built on them, with their pre-registered rules and evaluation:

| Predictor | Question | Rows | Gate and evaluation |
|---|---|---|---|
| 1 Next tool | which tool will this response call (16 categories, or the turn ends)? | request events (A) | `tool_gates_16w.py`, `tool_eval_16w.py` |
| 2 Tool duration | how long will the tools of this response take? | request events (R) | `duration_gates_16w.py`, `duration_eval_16w.py` |
| 3 Time left | how much of its own time will the session still need? | user turns | `session_gates_16w.py`, `session_eval_16w.py` |

The models are in `prediction/predictor/` (`tool_baseline.py`, `duration_baseline.py`, `session_baseline.py`). Results and
the reasoning behind them are in the "Agent Session Predictors" artifact and the status report.

## 1. Get the rows

Nothing is generated in git: `data/` is ignored, so build it. Run everything from the repo root.

```sh
pip install numpy orjson zstandard huggingface_hub        # or: pip install -e '.[predictors]' (which also pulls langgraph, unused here)

# 1. Download all 16 weekly releases, about 420 MB, from a pinned revision (the dataset is public, no login).
python -c "from huggingface_hub import snapshot_download as d; d('harvardMadsys/freeinference_agentic_trace', repo_type='dataset', revision='bb26cf72c9c20f2b05f765d88548156a74d3c253', local_dir='data/freeinference', allow_patterns=['release/*/manifest.json','release/*/traces/*'])"

# 2. One record per user turn (the agent DAG), about 70 s. Sep 2-5 is skipped: it is the held-out split.
python -m analysis.dag.build --week all --out data/freeinference/dag_all

# 3. The model-ready rows, a few minutes. Both stream one day at a time, so memory stays small.
python -m analysis.dag.online  data/freeinference/dag_all      # request events: 2,171,252 rows, 78 columns
python -m analysis.dag.session data/freeinference/dag_all      # user turns: 77,955 rows
```

About 0.9 GB on disk in all (traces 417 MB, DAG 355 MB, rows 136 MB). Seven days at a week boundary appear in two weekly
releases with different sessions; the builder merges both, so a day's file is complete.

## 2. What the rows are

Every row holds only what is known at one moment, and the labels are the future from that moment. A leak audit rebuilds
rows from a scrambled future and checks no feature changes (`python -m analysis.dag.online data/freeinference/dag_all
--days 2026-08-03,2026-08-12 --audit 4000`, and the same for `session`).

* **`online_rows.csv.gz`**: two rows per request. **A** is the request being admitted (its response does not exist yet).
  **R** is the response completed (its tools are named but have not returned). Features, labels and columns are
  documented at the top of `online.py` and listed in `online_rows.meta.json`. Rows of one session are contiguous and in
  time order.
* **`session_rows.csv.gz`**: one row per user turn, at the turn's start. Documented at the top of `session.py`.
* Account history (the user's earlier turns or sessions) counts only what had ended before the moment, over the whole
  release, so it needs no carry-over step.

Per predictor, the loaders return ready dicts:

```python
from analysis.dag import splits, tool_eval, duration_eval_16w, session_eval
rows = tool_eval.load("data/freeinference/dag_all/online_rows.csv.gz", splits.FIT_DAYS)        # next tool: A rows + label
rows = duration_eval_16w.load("data/freeinference/dag_all/online_rows.csv.gz", splits.FIT_DAYS)  # tool duration: R rows with a round time
rows = session_eval.load("data/freeinference/dag_all/session_rows.csv.gz", splits.FIT_DAYS)    # time left: user turns
```

The loaders hold a few hundred thousand rows as Python dicts, roughly 1 GB for the fit weeks. To use another tool,
read the gzipped CSV directly (pandas works); join each A row to the R row of the same `(session, turn, step)` for the
next-tool label.

## 3. The split, and rules for a fair comparison

`splits.py` is the one source for the days, in time order: fit May 17 - Aug 8, tuning Aug 9-15, validation Aug 16-22,
unused Aug 23 - Sep 1, held out Sep 2-5.

* Tune on the tuning week. Read the validation week once, after the model is frozen and refitted on fit + tuning.
* **The validation week has already been read once for each of our three models.** A new model should tune on Aug 9-15
  and use Aug 23 - Sep 1 as its fresh validation, and say so.
* **Never touch Sep 2-5.** `build.py` refuses it unless given `--include-heldout`.
* Write the rule down before scoring: the baseline, the metric, the bar, the number of settings you will try. Ours are in
  the `*_gates_16w.py` files. The bar is a relative error reduction of at least 5% over a simple table, with a 95%
  bootstrap interval over sessions whose lower end is above zero.

## 4. Plug in your own model

Give your model the method the evaluation calls, then score it against the same baseline with the existing `gate()`.

| Predictor | Method your model needs | Scored by | Steps |
|---|---|---|---|
| 1 Next tool | `proba(row) -> np.ndarray` of 17 probabilities in `CLASSES` order | `tool_eval.gate` (Brier) | 3, 5, 10, 20 |
| 2 Tool duration | `marginal(row)` -> an object with `.crps(ms)`; round time in milliseconds, capped at 3,600,000 | `duration_eval_16w.gate` (CRPS) | 1, 2, 5, 10, 20 |
| 3 Time left | `marginal(row, target)` and `value(row, target)`; whole seconds | `session_eval_16w.gate` (CRPS) | turns 0, 1, 2, 5 |

```python
import collections, numpy as np
from analysis.dag import tool_eval, tool_gates_16w as G, splits
from prediction.predictor.tool_baseline import ToolForecaster, CLASSES, CODE

class PrevToolOnly:                                     # a toy: replace with your model
    def __init__(self, rows):
        self.counts = collections.defaultdict(lambda: np.ones(len(CLASSES)))
        for r in rows:
            self.counts[r["prev1_cat"]][CODE[r["next_cat"]]] += 1
    def proba(self, row):
        c = self.counts[row["prev1_cat"]]
        return c / c.sum()

rows = tool_eval.load("data/freeinference/dag_all/online_rows.csv.gz", splits.FIT_DAYS + splits.TUNE_DAYS)
fit  = [r for r in rows if r["day"] in set(splits.FIT_DAYS)]
tune = [r for r in rows if r["day"] in set(splits.TUNE_DAYS)]
baseline = ToolForecaster(fit, ("cat",), 80)            # the previous-tool table the gate uses
res = tool_eval.gate(PrevToolOnly(fit), baseline, tune, steps=G.GATE_STEPS)
print(res["reduction"], res["ci"])                      # relative Brier reduction and its 95% interval
```

A distribution for predictors 2 and 3 can come from samples, as long as the weights sum to 1:

```python
from prediction.predictor.empirical_baseline import Marginal
m = Marginal.from_atoms([((v,), 1 / len(samples)) for v in samples], 0)
m.crps(actual)
```

## 5. Reproduce our results

```sh
python -m analysis.dag.tool_eval_16w --tune          # then --validate: reads the validation week once
python -m analysis.dag.duration_eval_16w --tune
python -m analysis.dag.session_eval_16w --tune
python -m pytest -q                                  # needs numpy and pytest
```

`--tune` logs every setting it tries and writes `frozen.json`; `--validate` refuses to run twice in a checkout. Results
land in `data/freeinference/tools16w/`, `duration16w/` and `session16w/`.

## Notes

* The tool-duration round time is the round's, not one tool's: every tool in a round shares one time in this trace.
* The trace has no prompt text. Features are timing, token counts, tool categories, harness and history.
* `ModuleNotFoundError: orjson` or `zstandard` means the `pip install` line was skipped.
