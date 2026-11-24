"""Evaluate the session forecaster under the rules in session_gates.py.

    python -m analysis.dag.session_eval --smoke      # plumbing check on fit days only; touches no tuning or validation day
    python -m analysis.dag.session_eval --tune       # fit Aug 23-27, score Aug 28-29; at most TUNING_BUDGET configurations
    python -m analysis.dag.session_eval --validate   # refit on Aug 23-29, score Aug 30-Sep 1: once, after --tune froze a config

The metric is the CRPS of the predicted distribution of `fut_calls` (requests in the turns still to come) at
each decision turn in GATE_TURNS. At a fixed turn a session has one row, so every session counts once. The
gate averages the per-turn relative reductions over the baseline (the turn-index-only table, tuned on its own
small grid) and bootstraps over sessions. The distribution of `fut_turns` is scored the same way and reported
as a secondary result.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os

import numpy as np

from analysis.dag import session_gates as G
from prediction.predictor.session_baseline import SessionForecaster

ROWS = "data/freeinference/dag/session_rows.csv.gz"
VALIDATION_ROWS = "data/freeinference/dag_val/session_rows.csv.gz"
OUT = "data/freeinference/session"
BASELINE_STRENGTHS = (5, 20, 80, 320)
MODEL_CHAINS = {"m1": ("hist",), "m2": ("last",), "m3": ("prior",), "m4": ("hist", "last"), "m5": ("prior", "last"),
                "m6": ("harness", "hist")}
MODEL_STRENGTHS = (20, 80)
GRID = [(c, s) for c in MODEL_CHAINS for s in MODEL_STRENGTHS]       # 12 configurations = TUNING_BUDGET
TARGETS = {"fut_calls": 0, "fut_turns": 1}                           # which marginal of SessionForecaster.marginals

assert len(GRID) <= G.TUNING_BUDGET

INT = {"turn", "turn_idx", "cur_calls", "fut_turns", "fut_calls"}
FLOAT = {"first_prompt", "idle_before_s", "elapsed_s", "prior_mean_calls", "prior_last_calls", "prior_last_wall_s",
         "prior_mean_work_s", "prior_mean_idle_s", "user_sessions", "user_mean_turns", "user_mean_calls", "fut_work_s"}
STR = {"session", "day", "harness", "model"}


def load(path: str, days) -> list[dict]:
    days, rows = set(days), []
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["day"] not in days:
                continue
            out = {k: r[k] for k in STR}
            out["model"] = r["model"] or None
            out.update({k: int(r[k]) for k in INT})
            out.update({k: (float(r[k]) if r[k] != "" else None) for k in FLOAT})
            rows.append(out)
    return rows


def per_row(model: SessionForecaster, rows, target: str) -> np.ndarray:
    i = TARGETS[target]
    return np.array([model.marginals(r)[i].crps(r[target]) for r in rows])


def gate(model, base, rows, target="fut_calls", turns=G.GATE_TURNS, draws=G.BOOTSTRAP, seed=0) -> dict:
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
        b = SessionForecaster(fit, (), s)
        scored.append((np.mean([per_row(b, [r for r in tune if r["turn_idx"] == u], "fut_calls").mean()
                                for u in G.GATE_TURNS]), s))
    return min(scored)[1]


def show(name, res, note=""):
    print(f"  {name}: relative CRPS reduction {100 * res['reduction']:+.1f}% "
          f"(95% CI {100 * res['ci'][0]:+.1f}, {100 * res['ci'][1]:+.1f}) {note}")
    for u, v in res["turns"].items():
        print(f"      turn {u:<3} n={v['n']:<5} base {v['base']:.2f}  model {v['model']:.2f}  ({100 * (1 - v['model'] / v['base']):+.1f}%)")


def smoke():
    rows = load(ROWS, G.FIT_DAYS)
    fit = [r for r in rows if r["day"] in G.FIT_DAYS[:3]]
    test = [r for r in rows if r["day"] == G.FIT_DAYS[3]]            # Aug 26: a fit day, so no tuning or validation day is read
    print(f"smoke: fit {len(fit)} turns (Aug 23-25), score {len(test)} turns (Aug 26)")
    base = SessionForecaster(fit, (), 20)
    for chain in ("m1", "m4"):
        model = SessionForecaster(fit, MODEL_CHAINS[chain], 20)
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
    fit, tune_rows = [r for r in rows if r["day"] in G.FIT_DAYS], [r for r in rows if r["day"] in G.TUNE_DAYS]
    s0 = best_baseline(fit, tune_rows)
    base = SessionForecaster(fit, (), s0)
    print(f"baseline: turn-index-only table, strength {s0}")
    results = []
    for chain, strength in GRID:
        res = gate(SessionForecaster(fit, MODEL_CHAINS[chain], strength), base, tune_rows, draws=0)
        results.append((res["reduction"], chain, strength))
        print(f"  {chain}/{strength:<3} {MODEL_CHAINS[chain]!s:22} reduction {100 * res['reduction']:+.1f}%")
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
    fit = load(ROWS, G.FIT_DAYS + G.TUNE_DAYS)
    val = load(VALIDATION_ROWS, G.VALIDATION_DAYS)
    base = SessionForecaster(fit, (), c["baseline_strength"])
    model = SessionForecaster(fit, tuple(c["levels"]), c["strength"])
    report = {}
    for target in TARGETS:
        res = gate(model, base, val, target)
        show(f"{c['chain']}/{c['strength']} vs turn-index-only, {target}", res,
             ("-> PASS" if passes(res) else "-> FAIL") if target == "fut_calls" else "(secondary)")
        report[target] = {"reduction": res["reduction"], "ci": res["ci"], "pass": passes(res), "turns": res["turns"]}
    json.dump(report, open(done, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("smoke", "tune", "validate"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"smoke": smoke, "tune": tune, "validate": validate}[[m for m in ("smoke", "tune", "validate") if getattr(a, m)][0]]()
