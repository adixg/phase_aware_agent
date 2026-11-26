"""Reconstruct and describe the frozen final-evaluation refit on previously exposed Aug 23–Sep 1 requests."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.dag import tool_eval, tool_eval_16w as evaluation, tool_gates_16w as gate
from experiments.gradient_boosting.next_tool import (
    FINAL_DAYS, FIT_DAYS, OUT, STEPS, TUNE_DAYS, check_splits, load_rows,
    macro_average, source_digest, tuning_windows,
)
from experiments.gradient_boosting.predictor import Encoder, XGBoostToolPredictor, training_matrix
from prediction.predictor.tool_baseline import CLASSES, CODE, END, ToolForecaster


HERE = Path(__file__).resolve().parent
ONLINE = Path("data/freeinference/dag_all/online_rows.csv.gz")
LABELS = ["Turn ends" if name == END else name.lstrip("*") for name in CLASSES]


def confusion_plot(counts):
    totals = counts.sum(axis=1)
    rates = np.divide(counts, totals[:, None], out=np.zeros_like(counts, dtype=float),
                      where=totals[:, None] > 0)
    fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
    heat = ax.imshow(np.ma.masked_where(np.broadcast_to(totals[:, None] == 0, counts.shape), rates),
                     cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(heat, ax=ax, label="Share within actual class", format=matplotlib.ticker.PercentFormatter(1))
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=55, ha="right")
    ax.set_yticks(range(len(LABELS)), [f"{name}  ({n:,})" for name, n in zip(LABELS, totals)])
    ax.set(xlabel="Predicted top-1 class", ylabel="Actual class (requests)",
           title="Next-tool confusion on previously exposed requests\nReconstructed final refit · Aug 23–Sep 1 · steps 3, 5, 10, 20")
    for i, j in zip(*np.nonzero(counts)):
        ax.text(j, i, f"{rates[i, j]:.1%}".rstrip("%"), ha="center", va="center",
                fontsize=7, color="white" if rates[i, j] > 0.52 else "#20354a")
    fig.savefig(HERE / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def feature_importance(booster, encoder):
    spec = encoder.specification()
    groups = list(spec["numeric"])
    for feature in spec["categorical"]:
        values = spec["harnesses"] if feature == "harness" else spec["classes"]
        groups.extend([feature] * (len(values) + 2))
    if booster.num_features() != len(groups):
        raise ValueError("Booster does not match the frozen encoder width")
    gains = {name: 0.0 for name in dict.fromkeys(groups)}
    for column, gain in booster.get_score(importance_type="total_gain").items():
        gains[groups[int(column.removeprefix("f"))]] += gain
    total = sum(gains.values())
    shares = sorted(((name, gain / total) for name, gain in gains.items()), key=lambda item: item[1])
    labels = {
        "prev1_cat": "Previous tool category", "prev2_cat": "Second previous tool category",
        "prev3_cat": "Third previous tool category", "prev1_width": "Previous round tool count",
        "prev1_errors": "Previous round error count", "prev1_result_tokens": "Previous round result tokens",
        "dom": "Dominant tool so far", "harness": "Harness", "step": "Request step", "step_bucket": "Step bucket",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    bars = ax.barh([labels[name] for name, _ in shares], [share for _, share in shares], color="#206b78")
    ax.bar_label(bars, labels=[f"{share:.1%}" for _, share in shares], padding=5)
    ax.set_xlim(0, max(gains.values()) / total + 0.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set(xlabel="Share of total split gain",
           title="Grouped feature importance · reconstructed final refit\nFit May 17–Aug 15, 2026")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(HERE / "feature_importance.png", dpi=160)
    plt.close(fig)
    return dict(reversed(shares))


def compare_results(actual, expected, path="evaluation"):
    if type(actual) is not type(expected):
        raise ValueError(f"{path} has a different type")
    if isinstance(expected, dict) and actual.keys() == expected.keys():
        return max(compare_results(actual[key], value, f"{path}.{key}") for key, value in expected.items())
    if isinstance(expected, list) and len(actual) == len(expected):
        return max(compare_results(a, e, f"{path}[{i}]") for i, (a, e) in enumerate(zip(actual, expected)))
    if isinstance(expected, float) and np.isclose(actual, expected, rtol=0, atol=1e-12):
        return abs(actual - expected)
    if actual != expected:
        raise ValueError(f"{path} differs: {actual!r} vs {expected!r}")
    return 0.0


def main():
    frozen = json.loads((OUT / "frozen.json").read_text())
    committed = json.loads((HERE.parent / "results/frozen.json").read_text())
    encoder = Encoder(tuple(frozen["encoder"]["harnesses"]))
    if (frozen != committed or frozen["encoder"] != encoder.specification()
            or frozen["predictor_sha256"] != source_digest() or xgb.__version__ != frozen["xgboost_version"]
            or frozen["windows"] != tuning_windows() or frozen["steps"] != list(STEPS)
            or frozen["classes"] != list(CLASSES) or frozen["bootstrap_draws"] != gate.BOOTSTRAP
            or frozen["minimum_brier_reduction"] != gate.OFFLINE_MIN):
        raise ValueError("Runtime or local freeze differs from the committed protocol")
    fit = load_rows(ONLINE, FIT_DAYS + TUNE_DAYS)
    baseline = ToolForecaster(fit, tool_eval.BASELINE, frozen["baseline_strength"])
    model = XGBoostToolPredictor.fit(
        encoder, training_matrix(fit, encoder, frozen["params"]), frozen["params"], frozen["rounds"]
    )
    final = load_rows(ONLINE, FINAL_DAYS)
    check_splits(fit, final)
    fit_count = len(fit)
    del fit
    rows = [r for r in final if r["step"] in STEPS]
    p = model.predict(rows)
    cached = _CachedProbabilities(rows, p)
    result = evaluation.run_gate(cached, baseline, final, draws=frozen["bootstrap_draws"])
    reproduced = {
        "gate": result, "pass": evaluation.passes(result),
        "model_macro": macro_average(result, "model"), "baseline_macro": macro_average(result, "base"),
    }
    # Normalize JSON object keys and compare all recorded metrics before writing plots.
    reproduced = json.loads(json.dumps(reproduced))
    original = json.loads((HERE.parent / "results/final_evaluation.json").read_text())
    max_delta = compare_results(reproduced, original)
    actual = np.array([CODE[r["next_cat"]] for r in rows])
    predicted = p.argmax(axis=1)
    counts = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(counts, (actual, predicted), 1)
    metrics = tool_eval.scores(cached, rows)
    steps = np.array([r["step"] for r in rows])
    harnesses = np.array([r["harness"] or "(missing)" for r in rows])

    def summarize(mask):
        return {"n": int(mask.sum()), **{name: float(values[mask].mean()) for name, values in metrics.items()}}

    by_step = {t: summarize(steps == t) for t in STEPS}
    print(json.dumps({
        "booster_sha256": hashlib.sha256(model.booster.save_raw(raw_format="ubj")).hexdigest(),
        "fit_requests": fit_count, "evaluation": reproduced, "max_score_delta": max_delta,
        "n": len(rows), "argmax_ties": int(((p == p.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum()),
        "labels": LABELS, "counts": counts.tolist(), "pooled": summarize(np.ones(len(rows), dtype=bool)),
        "step": by_step, "harness": {h: summarize(harnesses == h) for h in sorted(set(harnesses))},
        "importance": feature_importance(model.booster, encoder),
    }, indent=2))
    confusion_plot(counts)


class _CachedProbabilities:
    def __init__(self, rows, probabilities):
        self.by_row = dict(zip(map(id, rows), probabilities))

    def proba(self, row):
        return self.by_row[id(row)]


if __name__ == "__main__":
    main()
