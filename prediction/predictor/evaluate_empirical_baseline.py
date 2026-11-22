"""Tune and evaluate the admission baseline using only train and validation."""

import argparse
import gzip
import json
import math
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from random import Random
from statistics import fmean

from .empirical_baseline import (
    FEATURES, HORIZON, QUANTILES, TARGETS, EmpiricalPredictor, admission_features,
)
from .preprocess_freeinference import ROOT, SPLITS, midnight

FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.0)
SEEDS = tuple(range(20261119, 20261129))
BUCKET_COUNTS = (2, 4, 8)
STRENGTHS = (5, 20, 80)


def load_admissions(source, split):
    if split not in ("train", "validation"):
        raise ValueError("Only train and validation are available to this evaluator")
    _, lower, upper = next(row for row in SPLITS if row[0] == split)
    directory = source / split
    snapshots = sorted(directory.glob("*.snapshots.jsonl.gz"))
    targets = sorted(directory.glob("*.targets.jsonl.gz"))
    if not snapshots or [p.name.replace(".snapshots.", ".targets.") for p in snapshots] != [
        p.name for p in targets
    ]:
        raise ValueError(f"Unpaired files in {split}")
    rows = []
    keys = set()
    for x, y in zip(snapshots, targets):
        with gzip.open(x, "rt") as features, gzip.open(y, "rt") as labels:
            for left, right in zip_longest(features, labels):
                if left is None or right is None:
                    raise ValueError(f"Unpaired row in {split}")
                snapshot, target = json.loads(left), json.loads(right)
                key = snapshot["key"]
                if (key, snapshot["ordinal"]) != (target["key"], target["ordinal"]):
                    raise ValueError(f"Join mismatch in {split}")
                if snapshot["ordinal"] != 0:
                    continue
                if key in keys:
                    raise ValueError("Duplicate admission")
                cutoff, endpoint = snapshot["cutoff_ms"], target["observed_endpoint_ms"]
                if not midnight(lower) <= cutoff <= endpoint < midnight(upper):
                    raise ValueError("Admission or target outside its temporal split")
                values = tuple(target[name] for name in TARGETS)
                if any(type(value) is not int or value < 0 for value in values):
                    raise ValueError("Invalid admission label")
                if values[-1] != endpoint - cutoff:
                    raise ValueError("Duration label disagrees with gateway endpoint")
                rows.append((admission_features(snapshot, HORIZON), values))
                keys.add(key)
    if not rows:
        raise ValueError(f"No admissions in {split}")
    return rows, keys


def prompt_edges(rows, buckets):
    prompts = sorted(inputs[2] for inputs, _ in rows if inputs[2] is not None)
    return tuple(sorted({
        prompts[len(prompts) * i // buckets] for i in range(1, buckets)
    })) if prompts else ()


def fit(rows, buckets, strength, conditional=True):
    return EmpiricalPredictor(rows, prompt_edges=prompt_edges(rows, buckets),
                              strength=strength, conditional=conditional)


def score(model, validation):
    metrics = defaultdict_lists()
    depths = Counter()
    for inputs, outcomes in validation:
        _, marginals, support = model.distribution(inputs)
        depths[support["group_depth"]] += 1
        for name, outcome, marginal in zip(TARGETS, outcomes, marginals):
            low, median, high = (marginal.quantile(q) for q in QUANTILES)
            values = metrics[name]
            values["mae"].append(abs(median - outcome))
            values["crps"].append(marginal.crps(outcome))
            values["coverage_80"].append(low <= outcome <= high)
            values["interval_width_80"].append(high - low)
            for q, estimate in zip(QUANTILES, (low, median, high)):
                residual = outcome - estimate
                values[f"pinball_{q}"].append(max(q * residual, (q - 1) * residual))
    return metrics, dict(depths)


def defaultdict_lists():
    return {
        name: {metric: [] for metric in (
            "mae", "crps", "coverage_80", "interval_width_80",
            *(f"pinball_{q}" for q in QUANTILES),
        )}
        for name in TARGETS
    }


def averages(scores):
    return {name: {metric: fmean(values) for metric, values in target.items()}
            for name, target in scores.items()}


def objective(metrics, reference):
    return fmean(metrics[name]["crps"] / reference[name]["crps"] for name in TARGETS)


def percentile(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * q) - 1)]


def paired_intervals(candidate, reference, size):
    rng = Random(SEEDS[0])
    differences = {
        name: {metric: [x - y for x, y in zip(candidate[name][metric], reference[name][metric])]
               for metric in ("mae", "crps")}
        for name in TARGETS
    }
    draws = defaultdict_lists()
    for _ in range(1000):
        indices = rng.choices(range(size), k=size)
        for name, metrics in differences.items():
            for metric, deltas in metrics.items():
                draws[name][metric].append(fmean(deltas[i] for i in indices))
    return {
        name: {metric: [percentile(values, 0.025), percentile(values, 0.975)]
               for metric, values in metrics.items() if values}
        for name, metrics in draws.items()
    }


def learning_curve(train, validation, buckets, strength, reference):
    results = {fraction: [] for fraction in FRACTIONS}
    for seed in SEEDS:
        order = list(range(len(train)))
        Random(seed).shuffle(order)
        for fraction in FRACTIONS:
            if fraction == 1 and results[fraction]:
                continue
            size = math.ceil(len(train) * fraction)
            subset = [train[index] for index in order[:size]] if fraction < 1 else train
            scores, _ = score(fit(subset, buckets, strength), validation)
            unconditional, _ = score(fit(subset, buckets, strength, conditional=False), validation)
            metrics, global_metrics = averages(scores), averages(unconditional)
            row = {
                "normalized_crps": objective(metrics, reference),
                "unconditional_normalized_crps": objective(global_metrics, reference),
            }
            for name in TARGETS:
                for metric in ("mae", "crps"):
                    row[f"{name}_{metric}"] = metrics[name][metric]
            results[fraction].append(row)
    return [
        dict(fraction=fraction, roots=math.ceil(len(train) * fraction), replicates=len(runs),
             metrics={key: dict(mean=fmean(row[key] for row in runs),
                                p10=percentile([row[key] for row in runs], 0.1),
                                p90=percentile([row[key] for row in runs], 0.9))
                      for key in runs[0]})
        for fraction, runs in results.items()
    ]


def run(source, output):
    train, train_keys = load_admissions(source, "train")
    validation, validation_keys = load_admissions(source, "validation")
    if train_keys & validation_keys:
        raise ValueError("Train and validation contain overlapping root keys")
    global_scores, _ = score(fit(train, 1, 1, conditional=False), validation)
    reference = averages(global_scores)
    candidates = []
    for buckets in BUCKET_COUNTS:
        for strength in STRENGTHS:
            scores, depths = score(fit(train, buckets, strength), validation)
            metrics = averages(scores)
            candidates.append(dict(buckets=buckets, strength=strength, metrics=metrics,
                                   normalized_crps=objective(metrics, reference), group_depths=depths))
    best = min(candidates, key=lambda row: row["normalized_crps"])
    buckets, strength = best["buckets"], best["strength"]
    model = fit(train, buckets, strength)
    best_scores, _ = score(model, validation)
    report = dict(
        train_roots=len(train), validation_roots=len(validation), horizon=HORIZON,
        unconditional=reference, selected=best, candidates=candidates,
        paired_candidate_minus_unconditional_ci95=paired_intervals(best_scores, global_scores, len(validation)),
        learning_curve_seeds=SEEDS,
        learning_curve=learning_curve(train, validation, buckets, strength, reference),
    )
    output.mkdir()
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    artifact = dict(schema_version=1, features=FEATURES, targets=TARGETS, horizon=HORIZON,
                    prompt_edges=model.prompt_edges, strength=strength, rows=train)
    (output / "model.json").write_text(json.dumps(artifact, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="preprocessed")
    parser.add_argument("--output", default="empirical-baseline")
    args = parser.parse_args()
    for value in (args.source, args.output):
        if Path(value).name != value or value in {"", ".", "..", "release"}:
            parser.error("Use simple directory names under data/freeinference")
    output = ROOT / args.output
    if output.exists() or output.is_symlink():
        parser.error(f"Output already exists: {output}")
    run(ROOT / args.source, output)
    print(f"Saved fitted baseline and validation results under {output}")


if __name__ == "__main__":
    main()
