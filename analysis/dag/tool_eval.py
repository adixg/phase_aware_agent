"""Evaluate the next-tool predictor under the rules in tool_gates.py.

    python -m analysis.dag.tool_eval --smoke      # plumbing check on fit days only; touches no tuning or validation day
    python -m analysis.dag.tool_eval --tune       # fit Aug 23-27, score Aug 28-29; at most TUNING_BUDGET configurations
    python -m analysis.dag.tool_eval --validate   # refit on Aug 23-29, score Aug 30-Sep 1: once, after --tune froze a config

Rows are the A events of online_rows.csv.gz. Each one's label, the first tool category of its response (or END
when it calls no tool), is read from the R event of the same request. The metric is the multiclass Brier score
of the predicted distribution over the 17 classes at each decision step in GATE_STEPS; the gate averages the
per-step relative reductions over the previous-tool table (tuned on its own small grid) and bootstraps over
sessions. Log loss, top-1 and top-3 accuracy are reported beside it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os

import numpy as np

from analysis.dag import tool_gates as G
from prediction.predictor.tool_baseline import CLASSES, CODE, END, ToolForecaster

ROWS = "data/freeinference/dag/online_rows.csv.gz"
VALIDATION_ROWS = "data/freeinference/dag_val/online_rows.csv.gz"
OUT = "data/freeinference/tools"
BASELINE = ("cat",)
BASELINE_STRENGTHS = (5, 20, 80, 320)
MODEL_CHAINS = {"m1": ("cat", "cat2"), "m2": ("cat", "err"), "m3": ("cat", "cat2", "cat3"),
                "m4": ("cat", "cat2", "err"), "m5": ("cat", "harness"), "m6": ("cat", "result")}
MODEL_STRENGTHS = (20, 80)
GRID = [(c, s) for c in MODEL_CHAINS for s in MODEL_STRENGTHS]       # 12 configurations = TUNING_BUDGET

assert len(GRID) <= G.TUNING_BUDGET
assert all(chain[0] == "cat" for chain in MODEL_CHAINS.values())     # every chain contains the baseline

FLOAT = {"prev1_errors", "prev1_width", "prev1_result_tokens"}
CATEGORY = {"prev1_cat", "prev2_cat", "prev3_cat"}
STR = {"session", "day", "harness", "model", "prev1_finish"}


def load(path: str, days) -> list[dict]:
    """A rows of the given days, each with `next_cat` taken from the R row of the same request."""
    days, answer, rows = set(days), {}, []
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["day"] not in days:
                continue
            key = (r["session"], r["turn"], r["step"])
            if r["event"] == "R":
                c = r["prev1_cat"] or END
                answer[key] = c if c in CODE else "Other"
                continue
            out = {k: r[k] or None for k in STR}
            out.update({k: (float(r[k]) if r[k] != "" else None) for k in FLOAT})
            out.update({k: (r[k] or None) for k in CATEGORY})
            out.update(step=int(r["step"]), turn=int(r["turn"]), _key=key)
            rows.append(out)
    for r in rows:
        r["next_cat"] = answer[r.pop("_key")]
    return rows


def scores(model: ToolForecaster, rows) -> dict:
    """Per-row Brier, log loss and top-1 / top-3 hits."""
    P = np.array([model.proba(r) for r in rows])
    y = np.array([CODE[r["next_cat"]] for r in rows])
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1
    rank = (P > P[np.arange(len(y)), y][:, None]).sum(axis=1)       # classes ranked above the true one
    return {"brier": ((P - onehot) ** 2).sum(axis=1), "logloss": -np.log(np.maximum(P[np.arange(len(y)), y], 1e-6)),
            "top1": (rank == 0).astype(float), "top3": (rank < 3).astype(float)}


def gate(model, base, rows, steps=G.GATE_STEPS, draws=G.BOOTSTRAP, seed=0) -> dict:
    """Mean over `steps` of the relative Brier reduction over `base`, with a paired bootstrap over sessions."""
    sessions = sorted({r["session"] for r in rows})
    index = {s: i for i, s in enumerate(sessions)}
    per_step = {}
    for t in steps:
        rs = [r for r in rows if r["step"] == t]
        if not rs:
            continue
        ids = np.array([index[r["session"]] for r in rs])
        m, b = scores(model, rs), scores(base, rs)
        per_step[t] = {"n": len(rs), "sum_m": np.bincount(ids, m["brier"], len(sessions)),
                       "sum_b": np.bincount(ids, b["brier"], len(sessions)),
                       "cnt": np.bincount(ids, minlength=len(sessions)).astype(float),
                       "model": {k: float(v.mean()) for k, v in m.items()}, "base": {k: float(v.mean()) for k, v in b.items()}}
    if not per_step:
        return {"steps": {}, "reduction": float("nan"), "ci": (float("nan"), float("nan"))}
    point = np.mean([1 - v["model"]["brier"] / v["base"]["brier"] for v in per_step.values()])
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
    """The baseline's own tuning: its best smoothing strength on the tuning days (mean Brier over the gate steps)."""
    scored = []
    for s in BASELINE_STRENGTHS:
        b = ToolForecaster(fit, BASELINE, s)
        scored.append((np.mean([scores(b, [r for r in tune if r["step"] == t])["brier"].mean() for t in G.GATE_STEPS]), s))
    return min(scored)[1]


def show(name, res, note=""):
    print(f"  {name}: relative Brier reduction {100 * res['reduction']:+.1f}% "
          f"(95% CI {100 * res['ci'][0]:+.1f}, {100 * res['ci'][1]:+.1f}) {note}")
    for t, v in res["steps"].items():
        b, m = v["base"], v["model"]
        print(f"      step {t:<3} n={v['n']:<6} Brier {b['brier']:.4f} -> {m['brier']:.4f} ({100 * (1 - m['brier'] / b['brier']):+.1f}%)"
              f"   log loss {b['logloss']:.3f} -> {m['logloss']:.3f}   top-1 {100 * b['top1']:.1f}% -> {100 * m['top1']:.1f}%"
              f"   top-3 {100 * b['top3']:.1f}% -> {100 * m['top3']:.1f}%")


def smoke():
    rows = load(ROWS, G.FIT_DAYS)
    fit = [r for r in rows if r["day"] in G.FIT_DAYS[:3]]
    test = [r for r in rows if r["day"] == G.FIT_DAYS[3]]            # Aug 26: a fit day, so no tuning or validation day is read
    print(f"smoke: fit {len(fit):,} requests (Aug 23-25), score {len(test):,} requests (Aug 26); {len(CLASSES)} classes")
    base = ToolForecaster(fit, BASELINE, 20)
    for chain in ("m1", "m2"):
        show(f"{chain}/20 vs previous-tool table/20", gate(ToolForecaster(fit, MODEL_CHAINS[chain], 20), base, test, draws=200))


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
    base = ToolForecaster(fit, BASELINE, s0)
    print(f"baseline: previous-tool table, strength {s0}")
    results = []
    for chain, strength in GRID:
        res = gate(ToolForecaster(fit, MODEL_CHAINS[chain], strength), base, tune_rows, draws=0)
        results.append((res["reduction"], chain, strength))
        print(f"  {chain}/{strength:<3} {MODEL_CHAINS[chain]!s:34} reduction {100 * res['reduction']:+.1f}%")
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
    base = ToolForecaster(fit, BASELINE, c["baseline_strength"])
    model = ToolForecaster(fit, tuple(c["levels"]), c["strength"])
    res = gate(model, base, val)
    show(f"{c['chain']}/{c['strength']} vs previous-tool table", res, "-> PASS" if passes(res) else "-> FAIL")
    json.dump({"reduction": res["reduction"], "ci": res["ci"], "pass": passes(res), "steps": res["steps"]},
              open(done, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("smoke", "tune", "validate"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    {"smoke": smoke, "tune": tune, "validate": validate}[[m for m in ("smoke", "tune", "validate") if getattr(a, m)][0]]()
