"""Evaluate the online forecaster under the rules in online_gates.py.

    python -m analysis.dag.online_eval --smoke      # plumbing check on fit days only; touches no tuning or validation day
    python -m analysis.dag.online_eval --tune       # fit Aug 23-27, score Aug 28-29; at most TUNING_BUDGET configurations
    python -m analysis.dag.online_eval --validate   # refit Aug 23-29, score Aug 30-Sep 1: once, after --tune froze a config

The metric is CRPS of the predicted distribution of `rem_after` at each decision step in GATE_STEPS. At a
fixed step a turn has one row, so every turn counts once. The gate averages the per-step relative
reductions over the baseline and bootstraps over sessions. The baseline is the step-only table (at R it
also knows whether the response called a tool); it is tuned on its own small grid, which does not use up
the model's budget and can only make it harder to beat. Stall CRPS (against a category-only table) is
reported as a secondary result for the R event, whose use is retention, not priority.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

from analysis.dag import online_gates as G
from prediction.predictor.online_baseline import OnlineForecaster

ROWS = "data/freeinference/dag/online_rows.csv.gz"
VALIDATION_ROWS = "data/freeinference/dag_val/online_rows.csv.gz"
OUT = "data/freeinference/online"
BASELINE_LEVELS = {"A": (), "R": ("tools",)}                 # after the step
BASELINE_STRENGTHS = (5, 20, 80, 320)
# Chains put the tool category first: on a held-out fit day (Aug 26, nothing from the tuning or validation days)
# category gained +2 to +3% CRPS over the step table and harness lost 1.5 to 5% at strengths up to 80, so the
# first grid (strengths 5 and 20, harness before category) was revised before --tune ever ran.
MODEL_CHAINS = {"m1": ("tools", "cat"), "m2": ("tools", "cat", "width"), "m3": ("tools", "cat", "harness"),
                "m4": ("tools", "cat", "hist"), "m5": ("tools", "cat", "width", "err"), "m6": ("tools", "cat", "stall")}
MODEL_STRENGTHS = (320, 1280)
GRID = [(c, s) for c in MODEL_CHAINS for s in MODEL_STRENGTHS]   # 12 configurations = TUNING_BUDGET
STALL_BASELINE = ("cat",)

assert len(GRID) <= G.TUNING_BUDGET

INT = {"step", "turn", "rem_after"}
FLOAT = {"prev1_width", "prev1_round_s", "prev1_errors", "prev1_result_tokens", "user_prior_mean_rounds",
         "sess_prior_mean_rounds", "stall_s"}
STR = {"session", "day", "event", "harness", "model", "prev1_cat"}


def load(path: str, days) -> list[dict]:
    days, rows = set(days), []
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["day"] not in days:
                continue
            out = {k: r[k] for k in STR}
            out.update({k: int(r[k]) for k in INT})
            out.update({k: (float(r[k]) if r[k] != "" else None) for k in FLOAT})
            out["prev1_cat"] = r["prev1_cat"] or None
            rows.append(out)
    return rows


def split(rows, event):
    return [r for r in rows if r["event"] == event]


def per_row(model: OnlineForecaster, rows, stall: bool = False) -> np.ndarray:
    if stall:
        return np.array([model.marginals(r)[1].crps(round(r["stall_s"] * 1000)) / 1000 for r in rows])
    return np.array([model.marginals(r)[0].crps(r["rem_after"]) for r in rows])


def gate(model, base, rows, steps=G.GATE_STEPS, draws=G.BOOTSTRAP, seed=0, stall=False) -> dict:
    """Mean over `steps` of the relative CRPS reduction over `base`, with a paired bootstrap over sessions."""
    sessions = sorted({r["session"] for r in rows})
    index = {s: i for i, s in enumerate(sessions)}
    per_step = {}
    for t in steps:
        rs = [r for r in rows if r["step"] == t and (r["stall_s"] is not None or not stall)]
        if not rs:
            continue
        ids = np.array([index[r["session"]] for r in rs])
        m, b = per_row(model, rs, stall), per_row(base, rs, stall)
        per_step[t] = {"n": len(rs), "model": float(m.mean()), "base": float(b.mean()),
                       "sum_m": np.bincount(ids, m, len(sessions)), "sum_b": np.bincount(ids, b, len(sessions)),
                       "cnt": np.bincount(ids, minlength=len(sessions)).astype(float)}
    if not per_step:
        return {"steps": {}, "reduction": float("nan"), "ci": (float("nan"), float("nan"))}
    point = np.mean([1 - v["model"] / v["base"] for v in per_step.values()])
    if draws == 0:
        return {"steps": {t: {k: v[k] for k in ("n", "model", "base")} for t, v in per_step.items()},
                "reduction": float(point), "ci": (float("nan"), float("nan"))}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        w = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions))).astype(float)
        boot.append(np.mean([1 - (w @ v["sum_m"]) / (w @ v["sum_b"]) for v in per_step.values() if w @ v["cnt"] > 0]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"steps": {t: {k: v[k] for k in ("n", "model", "base")} for t, v in per_step.items()},
            "reduction": float(point), "ci": (float(lo), float(hi))}


def passes(result: dict) -> bool:
    return result["reduction"] >= G.OFFLINE_MIN and result["ci"][0] > 0


def best_baseline(fit, tune, event):
    """The baseline's own tuning: its best smoothing strength on the tuning days (mean CRPS over the gate steps)."""
    scored = []
    for s in BASELINE_STRENGTHS:
        b = OnlineForecaster(fit, event, BASELINE_LEVELS[event], s)
        scored.append((np.mean([per_row(b, [r for r in tune if r["step"] == t]).mean() for t in G.GATE_STEPS]), s))
    return min(scored)[1]


def show(event, name, res, note=""):
    print(f"  {event} {name}: relative CRPS reduction {100 * res['reduction']:+.1f}% "
          f"(95% CI {100 * res['ci'][0]:+.1f}, {100 * res['ci'][1]:+.1f}) {note}")
    for t, v in res["steps"].items():
        print(f"      t={t:<3} n={v['n']:<5} base {v['base']:.3f}  model {v['model']:.3f}  ({100 * (1 - v['model'] / v['base']):+.1f}%)")


def smoke():
    rows = load(ROWS, G.FIT_DAYS)
    fit = [r for r in rows if r["day"] in G.FIT_DAYS[:3]]
    test = [r for r in rows if r["day"] == G.FIT_DAYS[3]]            # Aug 26: a fit day, so no tuning or validation day is read
    print(f"smoke: fit {len(fit):,} rows (Aug 23-25), score {len(test):,} rows (Aug 26)")
    for event in ("A", "R"):
        f, t = split(fit, event), split(test, event)
        base = OnlineForecaster(f, event, BASELINE_LEVELS[event], 320)
        model = OnlineForecaster(f, event, MODEL_CHAINS["m1"], 320)
        show(event, "m1/320 vs step-only/320", gate(model, base, t, draws=200))


def tune():
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "tuning_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    new = [f"{c}/{s}" for c, s in GRID if f"{c}/{s}" not in log]
    if len(log) + len(new) > G.TUNING_BUDGET:
        raise SystemExit(f"tuning budget of {G.TUNING_BUDGET} configurations exhausted ({len(log)} already tried)")
    for c, s_ in GRID:                      # recorded before anything is scored, so a crash cannot hide a try
        log[f"{c}/{s_}"] = True
    json.dump(log, open(log_path, "w"), indent=1)
    rows = load(ROWS, G.FIT_DAYS + G.TUNE_DAYS)
    fit, tune_rows = [r for r in rows if r["day"] in G.FIT_DAYS], [r for r in rows if r["day"] in G.TUNE_DAYS]
    frozen = {}
    for event in ("A", "R"):
        f, t = split(fit, event), split(tune_rows, event)
        s0 = best_baseline(f, t, event)
        base = OnlineForecaster(f, event, BASELINE_LEVELS[event], s0)
        print(f"\nevent {event}: baseline step-only{' + tools' if event == 'R' else ''}, strength {s0}")
        results = []
        for chain, strength in GRID:
            res = gate(OnlineForecaster(f, event, MODEL_CHAINS[chain], strength), base, t, draws=0)
            results.append((res["reduction"], chain, strength))
            print(f"  {chain}/{strength:<3} {MODEL_CHAINS[chain]!s:55} reduction {100 * res['reduction']:+.1f}%")
        r, chain, strength = max(results)
        frozen[event] = {"chain": chain, "levels": MODEL_CHAINS[chain], "strength": strength, "baseline_strength": s0,
                         "tune_reduction": r}
        print(f"  -> {event}: {chain}/{strength}, {100 * r:+.1f}% on the tuning days")
    json.dump(frozen, open(os.path.join(OUT, "frozen.json"), "w"), indent=1)


def validate():
    frozen_path, done = os.path.join(OUT, "frozen.json"), os.path.join(OUT, "validation_done.json")
    if not os.path.exists(frozen_path):
        raise SystemExit("run --tune first: nothing is frozen")
    if os.path.exists(done):
        raise SystemExit("the validation days were read already; they are read once")
    frozen = json.load(open(frozen_path))
    fit = load(ROWS, G.FIT_DAYS + G.TUNE_DAYS)
    val = load(VALIDATION_ROWS, G.VALIDATION_DAYS)
    report = {}
    for event in ("A", "R"):
        f, v = split(fit, event), split(val, event)
        c = frozen[event]
        base = OnlineForecaster(f, event, BASELINE_LEVELS[event], c["baseline_strength"])
        model = OnlineForecaster(f, event, tuple(c["levels"]), c["strength"])
        res = gate(model, base, v)
        show(event, f"{c['chain']}/{c['strength']} vs step-only", res, "-> PASS" if passes(res) else "-> FAIL")
        if event == "R":
            sres = gate(model, OnlineForecaster(f, event, STALL_BASELINE, 20), v, stall=True)
            show(event, "stall vs category-only (secondary)", sres)
            report["R_stall"] = {k: sres[k] for k in ("reduction", "ci")}
        report[event] = {"reduction": res["reduction"], "ci": res["ci"], "pass": passes(res), "steps": res["steps"]}
    json.dump(report, open(done, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("smoke", "tune", "validate"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"smoke": smoke, "tune": tune, "validate": validate}[[m for m in ("smoke", "tune", "validate") if getattr(a, m)][0]]()
