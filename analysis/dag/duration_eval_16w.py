"""Evaluate the tool-duration predictor (predictor 2) on the 16-week split under the rules in duration_gates_16w.py.

    python -m analysis.dag.duration_eval_16w --smoke      # plumbing check on fit days only; touches no tuning or validation day
    python -m analysis.dag.duration_eval_16w --tune       # fit May 17-Aug 8, score Aug 9-15; at most TUNING_BUDGET configurations
    python -m analysis.dag.duration_eval_16w --validate   # refit on May 17-Aug 15, score Aug 16-22: once, after --tune froze a config

Rows are the R events of online_rows.csv.gz whose response called a tool and so has a round time (the `stall_s`
label). The metric is the CRPS, in seconds, of the predicted distribution of the capped round time at each decision
step in GATE_STEPS; the gate averages the per-step relative reductions over the category-only table (tuned on its
own small grid) and bootstraps over sessions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

from analysis.dag import duration_gates_16w as G
from prediction.predictor.duration_baseline import CAP_MS, DurationForecaster

ROWS = "data/freeinference/dag_all/online_rows.csv.gz"
OUT = "data/freeinference/duration16w"
BASELINE = ("cat",)
BASELINE_STRENGTHS = (5, 20, 80, 320)
MODEL_CHAINS = {"d1": ("cat", "width"), "d2": ("cat", "prevdur"), "d3": ("cat", "samecat"),
                "d4": ("cat", "samecat", "prevdur"), "d5": ("cat", "harness"), "d6": ("cat", "cat2")}
MODEL_STRENGTHS = (20, 80)
GRID = [(c, s) for c in MODEL_CHAINS for s in MODEL_STRENGTHS]       # 12 configurations = TUNING_BUDGET

assert len(GRID) <= G.TUNING_BUDGET
assert all(chain[0] == "cat" for chain in MODEL_CHAINS.values())     # every chain contains the baseline


def load(path: str, days) -> list[dict]:
    """R rows of the given days that have a round time. `samecat` is the capped time of the last earlier round of the
    same session with the same first tool category: rounds before this one only, since a round's time is known when
    the next request starts, which is before the next response completes."""
    days, rows, sid, seen = set(days), [], None, {}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["event"] != "R" or r["day"] not in days:
                continue
            if r["session"] != sid:
                sid, seen = r["session"], {}
            if r["stall_s"] == "":                       # the response called no tool: no round
                continue
            cat = sys.intern(r["prev1_cat"])
            dur = min(round(float(r["stall_s"]) * 1000), CAP_MS)
            rows.append({"session": sid, "day": r["day"], "step": int(r["step"]), "cat": cat,
                         "cat2": sys.intern(r["prev2_cat"]) if r["prev2_cat"] else None,
                         "width": float(r["prev1_width"]) if r["prev1_width"] else None,
                         "prevdur": min(float(r["prev2_round_s"]), CAP_MS / 1000) if r["prev2_round_s"] else None,
                         "samecat": seen.get(cat), "harness": sys.intern(r["harness"]),
                         "model": sys.intern(r["model"]) if r["model"] else None, "dur_ms": dur})
            seen[cat] = dur / 1000
    return rows


def per_row(model: DurationForecaster, rows) -> np.ndarray:
    return np.array([model.marginal(r).crps(r["dur_ms"]) / 1000 for r in rows])


def gate(model, base, rows, steps=G.GATE_STEPS, draws=G.BOOTSTRAP, seed=0) -> dict:
    """Mean over `steps` of the relative CRPS reduction over `base`, with a paired bootstrap over sessions."""
    sessions = sorted({r["session"] for r in rows})
    index = {s: i for i, s in enumerate(sessions)}
    per_step = {}
    for t in steps:
        rs = [r for r in rows if r["step"] == t]
        if not rs:
            continue
        ids = np.array([index[r["session"]] for r in rs])
        m, b = per_row(model, rs), per_row(base, rs)
        per_step[t] = {"n": len(rs), "model": float(m.mean()), "base": float(b.mean()),
                       "sum_m": np.bincount(ids, m, len(sessions)), "sum_b": np.bincount(ids, b, len(sessions)),
                       "cnt": np.bincount(ids, minlength=len(sessions)).astype(float)}
    if not per_step:
        return {"steps": {}, "reduction": float("nan"), "ci": (float("nan"), float("nan"))}
    point = np.mean([1 - v["model"] / v["base"] for v in per_step.values()])
    shown = {t: {k: v[k] for k in ("n", "model", "base")} for t, v in per_step.items()}
    if draws == 0:
        return {"steps": shown, "reduction": float(point), "ci": (float("nan"), float("nan"))}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        w = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions))).astype(float)
        boot.append(np.mean([1 - (w @ v["sum_m"]) / (w @ v["sum_b"]) for v in per_step.values() if w @ v["cnt"] > 0]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"steps": shown, "reduction": float(point), "ci": (float(lo), float(hi))}


def passes(result: dict) -> bool:
    return result["reduction"] >= G.OFFLINE_MIN and result["ci"][0] > 0


def best_baseline(fit, tune) -> int:
    """The baseline's own tuning: its best smoothing strength on the tuning days (mean CRPS over the gate steps)."""
    scored = []
    for s in BASELINE_STRENGTHS:
        b = DurationForecaster(fit, BASELINE, s)
        scored.append((np.mean([per_row(b, [r for r in tune if r["step"] == t]).mean() for t in G.GATE_STEPS]), s))
    return min(scored)[1]


def show(name, res, note=""):
    print(f"  {name}: relative CRPS reduction {100 * res['reduction']:+.1f}% "
          f"(95% CI {100 * res['ci'][0]:+.1f}, {100 * res['ci'][1]:+.1f}) {note}")
    for t, v in res["steps"].items():
        print(f"      step {t:<3} n={v['n']:<6} base {v['base']:.3f} s  model {v['model']:.3f} s  ({100 * (1 - v['model'] / v['base']):+.1f}%)")


def split(rows, days):
    days = set(days)
    return [r for r in rows if r["day"] in days]


def smoke():
    rows = load(ROWS, G.FIT_DAYS)
    fit, test = [r for r in rows if r["day"] < "2026-08-01"], [r for r in rows if r["day"] >= "2026-08-01"]   # Aug 1-8: fit days
    print(f"smoke: fit {len(fit):,} rounds (to Jul 31), score {len(test):,} rounds (Aug 1-8)")
    base = DurationForecaster(fit, BASELINE, 20)
    for chain in ("d1", "d3"):
        show(f"{chain}/20 vs category-only/20", gate(DurationForecaster(fit, MODEL_CHAINS[chain], 20), base, test, draws=200))


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
    fit, tune_rows = split(rows, G.FIT_DAYS), split(rows, G.TUNE_DAYS)
    del rows
    print(f"fit {len(fit):,} rounds, tuning {len(tune_rows):,}")
    s0 = best_baseline(fit, tune_rows)
    base = DurationForecaster(fit, BASELINE, s0)
    print(f"baseline: category-only table, strength {s0}")
    results = []
    for chain, strength in GRID:
        res = gate(DurationForecaster(fit, MODEL_CHAINS[chain], strength), base, tune_rows, draws=0)
        results.append((res["reduction"], chain, strength))
        print(f"  {chain}/{strength:<3} {MODEL_CHAINS[chain]!s:36} reduction {100 * res['reduction']:+.1f}%", flush=True)
    r, chain, strength = max(results)
    json.dump({"chain": chain, "levels": MODEL_CHAINS[chain], "strength": strength, "baseline_strength": s0,
               "tune_reduction": r}, open(os.path.join(OUT, "frozen.json"), "w"), indent=1)
    print(f"  -> {chain}/{strength}, {100 * r:+.1f}% on the tuning days")


def validate():
    frozen_path, done = os.path.join(OUT, "frozen.json"), os.path.join(OUT, "validation_done.json")
    if not os.path.exists(frozen_path):
        raise SystemExit("run --tune first: nothing is frozen")
    if os.path.exists(done):
        raise SystemExit("the validation days were read already; they are read once")
    c = json.load(open(frozen_path))
    rows = load(ROWS, G.FIT_DAYS + G.TUNE_DAYS + G.VALIDATION_DAYS)
    train, val = split(rows, G.FIT_DAYS + G.TUNE_DAYS), split(rows, G.VALIDATION_DAYS)
    del rows
    base = DurationForecaster(train, BASELINE, c["baseline_strength"])
    model = DurationForecaster(train, tuple(c["levels"]), c["strength"])
    res = gate(model, base, val)
    show(f"{c['chain']}/{c['strength']} vs category-only", res, "-> PASS" if passes(res) else "-> FAIL")
    json.dump({"reduction": res["reduction"], "ci": res["ci"], "pass": passes(res), "steps": res["steps"]},
              open(done, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("smoke", "tune", "validate"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"smoke": smoke, "tune": tune, "validate": validate}[[m for m in ("smoke", "tune", "validate") if getattr(a, m)][0]]()
