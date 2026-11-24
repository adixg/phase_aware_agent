"""Evaluate the time-left predictor (predictor 3) on the 16-week split under the rules in session_gates_16w.py.

    python -m analysis.dag.session_eval_16w --smoke      # plumbing check on fit days only; touches no tuning or validation day
    python -m analysis.dag.session_eval_16w --tune       # fit May 17-Aug 8, score Aug 9-15; at most TUNING_BUDGET configurations
    python -m analysis.dag.session_eval_16w --validate   # refit on May 17-Aug 15, score Aug 16-22: once, after --tune froze a config

The metric is the CRPS of the predicted distribution of `rem_path_s` (the system's own seconds left in the session,
the current turn included) at each decision turn in GATE_TURNS. At a fixed turn a session has one row, so every
session counts once. The gate averages the per-turn relative reductions over the baseline (the turn-index-only
table, tuned on its own small grid) and bootstraps over sessions. The requests left (`rem_calls`) are scored the
same way and reported as a secondary result.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.dag import session_gates_16w as G
from analysis.dag.session_eval import load
from prediction.predictor.session_baseline import SessionTimeForecaster

ROWS = "data/freeinference/dag_all/session_rows.csv.gz"
OUT = "data/freeinference/session16w"
BASELINE_STRENGTHS = (5, 20, 80, 320)
MODEL_CHAINS = {"s1": ("histpath",), "s2": ("histpath", "last"), "s3": ("histpath", "hist"),
                "s4": ("histpath", "harness"), "s5": ("histpath", "prompt"), "s6": ("hist", "last")}
MODEL_STRENGTHS = (20, 80)
GRID = [(c, s) for c in MODEL_CHAINS for s in MODEL_STRENGTHS]       # 12 configurations = TUNING_BUDGET
TARGETS = SessionTimeForecaster.TARGETS                              # rem_path_s (gated), rem_calls (secondary)

assert len(GRID) <= G.TUNING_BUDGET


def per_row(model: SessionTimeForecaster, rows, target: str) -> np.ndarray:
    return np.array([model.marginal(r, target).crps(model.value(r, target)) for r in rows])


def gate(model, base, rows, target="rem_path_s", turns=G.GATE_TURNS, draws=G.BOOTSTRAP, seed=0) -> dict:
    """Mean over `turns` of the relative CRPS reduction over `base`, with a paired bootstrap over sessions."""
    sessions = sorted({r["session"] for r in rows})
    index = {s: i for i, s in enumerate(sessions)}
    per_turn = {}
    for u in turns:
        rs = [r for r in rows if r["turn_idx"] == u]
        if not rs:
            continue
        ids = np.array([index[r["session"]] for r in rs])
        m, b = per_row(model, rs, target), per_row(base, rs, target)
        per_turn[u] = {"n": len(rs), "model": float(m.mean()), "base": float(b.mean()),
                       "sum_m": np.bincount(ids, m, len(sessions)), "sum_b": np.bincount(ids, b, len(sessions)),
                       "cnt": np.bincount(ids, minlength=len(sessions)).astype(float)}
    if not per_turn:
        return {"turns": {}, "reduction": float("nan"), "ci": (float("nan"), float("nan"))}
    point = np.mean([1 - v["model"] / v["base"] for v in per_turn.values()])
    shown = {u: {k: v[k] for k in ("n", "model", "base")} for u, v in per_turn.items()}
    if draws == 0:
        return {"turns": shown, "reduction": float(point), "ci": (float("nan"), float("nan"))}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        w = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions))).astype(float)
        boot.append(np.mean([1 - (w @ v["sum_m"]) / (w @ v["sum_b"]) for v in per_turn.values() if w @ v["cnt"] > 0]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"turns": shown, "reduction": float(point), "ci": (float(lo), float(hi))}


def passes(result: dict) -> bool:
    return result["reduction"] >= G.OFFLINE_MIN and result["ci"][0] > 0


def best_baseline(fit, tune) -> int:
    """The baseline's own tuning: its best smoothing strength on the tuning days (mean CRPS over the gate turns)."""
    scored = []
    for s in BASELINE_STRENGTHS:
        b = SessionTimeForecaster(fit, (), s)
        scored.append((np.mean([per_row(b, [r for r in tune if r["turn_idx"] == u], "rem_path_s").mean()
                                for u in G.GATE_TURNS]), s))
    return min(scored)[1]


def show(name, res, note=""):
    print(f"  {name}: relative CRPS reduction {100 * res['reduction']:+.1f}% "
          f"(95% CI {100 * res['ci'][0]:+.1f}, {100 * res['ci'][1]:+.1f}) {note}")
    for u, v in res["turns"].items():
        print(f"      turn {u:<3} n={v['n']:<5} base {v['base']:,.1f}  model {v['model']:,.1f}  ({100 * (1 - v['model'] / v['base']):+.1f}%)")


def split(rows, days):
    days = set(days)
    return [r for r in rows if r["day"] in days]


def smoke():
    rows = split(load(ROWS, G.FIT_DAYS), G.FIT_DAYS)
    fit = [r for r in rows if r["day"] < "2026-07-18"]
    test = [r for r in rows if r["day"] >= "2026-07-18"]                # Jul 18 - Aug 8: fit days, so no tuning or validation day is read
    print(f"smoke: fit {len(fit):,} turns (to Jul 17), score {len(test):,} turns (Jul 18 - Aug 8)")
    base = SessionTimeForecaster(fit, (), 20)
    for chain in ("s1", "s2"):
        model = SessionTimeForecaster(fit, MODEL_CHAINS[chain], 20)
        for target in TARGETS:
            show(f"{chain}/20 vs turn-index-only/20, {target}", gate(model, base, test, target, draws=200))


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
    print(f"fit {len(fit):,} turns, tuning {len(tune_rows):,}")
    s0 = best_baseline(fit, tune_rows)
    base = SessionTimeForecaster(fit, (), s0)
    print(f"baseline: turn-index-only table, strength {s0}")
    results = []
    for chain, strength in GRID:
        res = gate(SessionTimeForecaster(fit, MODEL_CHAINS[chain], strength), base, tune_rows, draws=0)
        results.append((res["reduction"], chain, strength))
        print(f"  {chain}/{strength:<3} {MODEL_CHAINS[chain]!s:26} reduction {100 * res['reduction']:+.1f}%", flush=True)
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
    base = SessionTimeForecaster(train, (), c["baseline_strength"])
    model = SessionTimeForecaster(train, tuple(c["levels"]), c["strength"])
    report = {}
    for target in TARGETS:
        res = gate(model, base, val, target)
        show(f"{c['chain']}/{c['strength']} vs turn-index-only, {target}", res,
             ("-> PASS" if passes(res) else "-> FAIL") if target == "rem_path_s" else "(secondary)")
        report[target] = {"reduction": res["reduction"], "ci": res["ci"], "pass": passes(res), "turns": res["turns"]}
    json.dump(report, open(done, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("smoke", "tune", "validate"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"smoke": smoke, "tune": tune, "validate": validate}[[m for m in ("smoke", "tune", "validate") if getattr(a, m)][0]]()
