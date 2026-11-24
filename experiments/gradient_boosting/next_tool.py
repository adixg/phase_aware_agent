"""Tune and freeze an XGBoost next-tool model using the repository's 16-week gate."""

import argparse
import gzip
import hashlib
import json
import resource
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import xgboost as xgb

from analysis.dag import splits, tool_eval, tool_eval_16w as evaluation, tool_gates_16w as gate
from prediction.predictor.tool_baseline import CLASSES, ToolForecaster

from . import predictor
from .predictor import PARAMS, Encoder, XGBoostToolPredictor, training_matrix

OUT = Path("data/freeinference/experiments/gradient_boosting/next_tool")
FIT_DAYS = splits.FIT_DAYS
TUNE_DAYS = splits.TUNE_DAYS
FINAL_DAYS = splits.EXPOSED_DAYS
STEPS = gate.GATE_STEPS
GRID = (
    ("default", 100, {"max_depth": 6, "eta": 0.3}),
    ("shallow", 100, {"max_depth": 3, "eta": 0.3}),
    ("slow", 160, {"max_depth": 6, "eta": 0.1}),
    ("regularized", 120, {"max_depth": 5, "eta": 0.15, "min_child_weight": 5,
                          "subsample": 0.85, "colsample_bytree": 0.9, "reg_lambda": 5}),
)
assert len(GRID) <= gate.TUNING_BUDGET


def load_rows(path, days):
    """Feed only the requested dates to the canonical A/R loader.

    The online writer orders days chronologically and writes session, user and
    day as its first fields. Inspect that prefix before handing off a row.
    """
    days = set(days)
    last = max(days)
    with tempfile.TemporaryDirectory() as folder:
        selected = Path(folder) / "selected.csv.gz"
        with gzip.open(path, "rt", newline="") as source, gzip.open(
            selected, "wt", newline="", compresslevel=1
        ) as target:
            header = next(source)
            if not header.startswith("session,user,day,"):
                raise ValueError("Unexpected online CSV prefix")
            target.write(header)
            previous = ""
            for line in source:
                session, user, day, _ = line.split(",", 3)
                if '"' in session or '"' in user or len(day) != 10 or day < previous:
                    raise ValueError("Expected unquoted IDs and chronological day order")
                if day > last:
                    break
                if day in days:
                    target.write(line)
                previous = day
        rows = tool_eval.load(selected, days)
    if {row["day"] for row in rows} != days:
        raise ValueError("CSV does not cover every requested date")
    return rows


def check_splits(fit, score):
    if max(r["day"] for r in fit) >= min(r["day"] for r in score):
        raise ValueError("Training must precede scoring")
    if {r["session"] for r in fit} & {r["session"] for r in score}:
        raise ValueError("Training and scoring sessions overlap")


class BatchPredictions:
    """Expose batch-computed probabilities through the shared evaluator interface."""

    def __init__(self, model, rows):
        self.by_row = dict(zip(map(id, rows), model.predict(rows)))

    def proba(self, row):
        return self.by_row[id(row)]


def macro_average(result, side):
    return {k: float(np.mean([step[side][k] for step in result["steps"].values()]))
            for k in ("brier", "top1", "top3", "logloss")}


def evaluate(model, base, rows, draws=0):
    gate_rows = [r for r in rows if r["step"] in STEPS]
    result = evaluation.run_gate(BatchPredictions(model, gate_rows), base, rows, draws=draws)
    if tuple(result["steps"]) != STEPS:
        raise ValueError("Missing decision steps")
    if draws == 0:
        del result["ci"]
    return result


def memory_mib():
    """Process peak RSS, including Python, dataset and training allocations."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** (
        2 if sys.platform == "darwin" else 1
    ))


def write_json(path, value, mode="x"):
    with path.open(mode) as target:
        print(json.dumps(value, indent=2, allow_nan=False), file=target)


def source_digest():
    return hashlib.sha256(Path(predictor.__file__).read_bytes()).hexdigest()


def tuning_windows():
    return {"fit": list(FIT_DAYS), "tune": list(TUNE_DAYS), "final": list(FINAL_DAYS)}


def tune(path, out):
    out.mkdir(parents=True, exist_ok=True)
    # Reserve the entire predeclared grid before loading outcomes.
    write_json(out / "grid.json", {"shared_params": PARAMS, "grid": GRID})
    rows = load_rows(path, FIT_DAYS + TUNE_DAYS)
    fit_days = set(FIT_DAYS)
    fit = [r for r in rows if r["day"] in fit_days]
    tune_rows = [r for r in rows if r["day"] not in fit_days]
    del rows
    check_splits(fit, tune_rows)
    scoring_rows = [r for r in tune_rows if r["step"] in STEPS]
    print(f"Fit: {len(fit):,} requests. Tuning: {len(tune_rows):,} requests, "
          f"{len(scoring_rows):,} at gate steps.", flush=True)
    baseline_strength = tool_eval.best_baseline(fit, scoring_rows, STEPS)
    base_started = perf_counter()
    baseline = ToolForecaster(fit, tool_eval.BASELINE, baseline_strength)
    baseline_seconds = perf_counter() - base_started
    encoder = Encoder.fit(fit)
    matrix_started = perf_counter()
    matrix = training_matrix(fit, encoder, PARAMS)
    matrix_seconds = perf_counter() - matrix_started
    results = []
    winner = None
    for name, rounds, options in GRID:
        params = PARAMS | options
        started = perf_counter()
        model = XGBoostToolPredictor.fit(encoder, matrix, params, rounds)
        seconds = perf_counter() - started
        peak = memory_mib()
        result = evaluate(model, baseline, tune_rows)
        summary = {"name": name, "rounds": rounds, "params": params, "fit_seconds": seconds,
                   "process_peak_rss_mib": peak, "macro": macro_average(result, "model"),
                   "gate": result}
        results.append(summary)
        if winner is None or result["reduction"] > winner[0]["gate"]["reduction"]:
            winner = summary, model
        print(json.dumps({k: v for k, v in summary.items() if k != "gate"} |
                         {"brier_reduction": result["reduction"]}), flush=True)
    selected, model = winner
    model.booster.save_model(out / "tuned_model.ubj")
    write_json(out / "tuning_results.json", {
        "fit_requests": len(fit), "tune_requests": len(tune_rows),
        "scoring_requests": len(scoring_rows), "matrix_seconds": matrix_seconds,
        "baseline": {"strength": baseline_strength, "fit_seconds": baseline_seconds,
                     "macro": macro_average(selected["gate"], "base")},
        "candidates": results, "selected": selected["name"], "process_peak_rss_mib": memory_mib(),
    })
    write_json(out / "frozen.json", {
        "name": selected["name"], "params": selected["params"], "rounds": selected["rounds"],
        "encoder": encoder.specification(), "predictor_sha256": source_digest(),
        "baseline_strength": baseline_strength, "classes": list(CLASSES),
        "steps": list(STEPS), "windows": tuning_windows(), "xgboost_version": xgb.__version__,
        "bootstrap_draws": gate.BOOTSTRAP, "minimum_brier_reduction": gate.OFFLINE_MIN,
    })
    print(f"Frozen {selected['name']} in {out}", flush=True)


def final_evaluation(path, out):
    frozen = json.loads((out / "frozen.json").read_text())
    encoder = Encoder(tuple(frozen["encoder"]["harnesses"]))
    if (frozen["encoder"] != encoder.specification() or frozen["predictor_sha256"] != source_digest()
            or frozen["windows"] != tuning_windows() or frozen["steps"] != list(STEPS)
            or frozen["classes"] != list(CLASSES) or frozen["xgboost_version"] != xgb.__version__
            or frozen["bootstrap_draws"] != gate.BOOTSTRAP
            or frozen["minimum_brier_reduction"] != gate.OFFLINE_MIN):
        raise ValueError("Runtime differs from the frozen evaluation protocol")
    attempted = out / "final_evaluation_started.json"
    if attempted.exists():
        raise FileExistsError("Final evaluation has already started for this freeze")
    fit = load_rows(path, FIT_DAYS + TUNE_DAYS)
    baseline = ToolForecaster(fit, tool_eval.BASELINE, frozen["baseline_strength"])
    model = XGBoostToolPredictor.fit(
        encoder, training_matrix(fit, encoder, frozen["params"]), frozen["params"], frozen["rounds"]
    )
    # Exclusive creation precedes any access to the final window's outcomes.
    write_json(attempted, {"days": list(FINAL_DAYS)})
    final = load_rows(path, FINAL_DAYS)
    check_splits(fit, final)
    result = evaluate(model, baseline, final, draws=gate.BOOTSTRAP)
    write_json(out / "final_evaluation.json", {
        "gate": result, "pass": evaluation.passes(result),
        "model_macro": macro_average(result, "model"), "baseline_macro": macro_average(result, "base"),
    })
    tool_eval.show(frozen["name"], result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--final-evaluation", action="store_true")
    parser.add_argument("--approve-final-evaluation", action="store_true")
    parser.add_argument("--rows", type=Path, default=Path(evaluation.ROWS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.final_evaluation != args.approve_final_evaluation:
        parser.error("Final evaluation requires both --final-evaluation and --approve-final-evaluation")
    if args.tune:
        tune(args.rows, args.out)
    else:
        final_evaluation(args.rows, args.out)


if __name__ == "__main__":
    main()
