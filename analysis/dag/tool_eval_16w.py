"""Evaluate the next-tool predictor on the 16-week split under the rules in tool_gates_16w.py.

    python -m analysis.dag.tool_eval_16w --smoke      # plumbing check on fit days only; touches no tuning or validation day
    python -m analysis.dag.tool_eval_16w --tune       # fit May 17-Aug 8, score Aug 9-15; at most TUNING_BUDGET configurations
    python -m analysis.dag.tool_eval_16w --validate   # refit on May 17-Aug 15, score Aug 16-22: once, after --tune froze a config

Same rows, labels and metric as tool_eval.py; the data are the A events of the one continuous folder of all
weeks (data/freeinference/dag_all/online_rows.csv.gz), the decision steps are those of tool_gates_16w.py, and the
grid adds the `dom` level.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis.dag import tool_gates_16w as G
from analysis.dag.tool_eval import BASELINE, BASELINE_STRENGTHS, best_baseline, gate, load, scores, show  # noqa: F401
from prediction.predictor.tool_baseline import CLASSES, ToolForecaster

ROWS = "data/freeinference/dag_all/online_rows.csv.gz"
OUT = "data/freeinference/tools16w"
BASE3 = ("cat", "cat2", "cat3")
MODEL_CHAINS = {"n1": BASE3, "n2": BASE3 + ("harness",), "n3": BASE3 + ("dom",), "n4": BASE3 + ("err",),
                "n5": BASE3 + ("harness", "dom"), "n6": ("cat", "cat2", "dom")}
MODEL_STRENGTHS = (20, 80)
GRID = [(c, s) for c in MODEL_CHAINS for s in MODEL_STRENGTHS]       # 12 configurations = TUNING_BUDGET

assert len(GRID) <= G.TUNING_BUDGET
assert all(chain[0] == "cat" for chain in MODEL_CHAINS.values())     # every chain contains the baseline


def passes(result: dict) -> bool:
    return result["reduction"] >= G.OFFLINE_MIN and result["ci"][0] > 0


def run_gate(model, base, rows, draws=G.BOOTSTRAP):
    return gate(model, base, rows, steps=G.GATE_STEPS, draws=draws)


def smoke():
    rows = load(ROWS, G.FIT_DAYS)
    fit = [r for r in rows if r["day"] < "2026-08-01"]
    test = [r for r in rows if r["day"] >= "2026-08-01"]              # Aug 1-8: fit days, so no tuning or validation day is read
    print(f"smoke: fit {len(fit):,} requests (May 17-Jul 31), score {len(test):,} requests (Aug 1-8); {len(CLASSES)} classes")
    base = ToolForecaster(fit, BASELINE, 20)
    for chain in ("n1", "n3"):
        show(f"{chain}/20 vs previous-tool table/20", run_gate(ToolForecaster(fit, MODEL_CHAINS[chain], 20), base, test, draws=200))


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
    fit, tune_rows = [r for r in rows if r["day"] in set(G.FIT_DAYS)], [r for r in rows if r["day"] in set(G.TUNE_DAYS)]
    del rows
    print(f"fit {len(fit):,} requests, tuning {len(tune_rows):,}")
    s0 = best_baseline(fit, tune_rows, G.GATE_STEPS)
    base = ToolForecaster(fit, BASELINE, s0)
    print(f"baseline: previous-tool table, strength {s0}")
    results = []
    for chain, strength in GRID:
        res = run_gate(ToolForecaster(fit, MODEL_CHAINS[chain], strength), base, tune_rows, draws=0)
        results.append((res["reduction"], chain, strength))
        print(f"  {chain}/{strength:<3} {MODEL_CHAINS[chain]!s:46} reduction {100 * res['reduction']:+.1f}%", flush=True)
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
    train, val = [r for r in rows if r["day"] not in set(G.VALIDATION_DAYS)], [r for r in rows if r["day"] in set(G.VALIDATION_DAYS)]
    del rows
    base = ToolForecaster(train, BASELINE, c["baseline_strength"])
    model = ToolForecaster(train, tuple(c["levels"]), c["strength"])
    res = run_gate(model, base, val)
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
