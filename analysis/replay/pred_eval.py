"""Simple rules on a predicted remaining time, under memory pressure: how much of the oracle's gain
over EDF survives a predictor with error, and whether trusting it only when its samples agree helps.

    python -m analysis.replay.pred_eval --days 2026-08-16 --kv 2000000 --out pred.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp

from analysis.replay.run_gate import PROFILES, TRAIN_DAYS, find_compress, load
from analysis.replay.sim import Cheap, Config, Simulator, summarize

_G: dict = {}


def run_one(task: dict) -> dict:
    sessions, compress, horizon = _G[task["day"]]
    cfg = Config(task["slots"], task["kv"] or math.inf, PROFILES["l40s"], policy=task["policy"],
                 deadline_mode="proportional", slack=task["scale"], compress=compress, preempt=task["preempt"],
                 pred_error=task["error"], pred_calibrated=task["calibrated"], pred_gate=task["gate"])
    return {**task, **summarize(Simulator(sessions, cfg, _G["cheap"]).run(), horizon)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", default="data/freeinference/compact")
    ap.add_argument("--days", required=True)
    ap.add_argument("--scales", default="1.1,1.2,1.5")
    ap.add_argument("--kv", default="2000000,1000000")
    ap.add_argument("--preempt", default="none,priority")
    ap.add_argument("--errors", default="0.1,0.3,0.6")
    ap.add_argument("--gate", type=float, default=0.15, help="largest sample CV the gated rule trusts")
    ap.add_argument("--load", type=float, default=0.9)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = [s for d in TRAIN_DAYS for s in load(args.compact, d, PROFILES["l40s"], reply="fixed:5", unit="session")]
    _G["cheap"] = Cheap(train, PROFILES["l40s"])
    del train
    for day in args.days.split(","):
        sessions = load(args.compact, day, PROFILES["l40s"], reply="fixed:5", unit="session")
        _G[day] = (sessions, *find_compress(sessions, args.slots, args.load))
    tasks = []
    for day in args.days.split(","):
        for scale in map(float, args.scales.split(",")):
            for kv in map(float, args.kv.split(",")):
                for pre in args.preempt.split(","):
                    cell = dict(day=day, scale=scale, kv=kv, preempt=pre, slots=args.slots, load=args.load)
                    for p in ("edf", "llf_oracle", "srpt_oracle"):
                        tasks.append(dict(cell, policy=p, error=0.0, calibrated=True, gate=None))
                    for e in map(float, args.errors.split(",")):
                        for p in ("llf_pred", "srpt_pred"):
                            tasks.append(dict(cell, policy=p, error=e, calibrated=True, gate=None))
                        for cal in (True, False):
                            tasks.append(dict(cell, policy="llf_pred", error=e, calibrated=cal, gate=args.gate))
    print(len(tasks), "runs", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool, open(args.out, "w") as f:
        for row in pool.imap_unordered(run_one, tasks, chunksize=4):
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
