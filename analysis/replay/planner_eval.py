"""The rollout planner against EDF and the oracle policies, with the stand-in predictor at several
error levels and refresh rates: how good and how fresh a DAG predictor must be for planning to pay.

    python -m analysis.replay.planner_eval --days 2026-08-16 --scales 1.1,1.2 --kv 0,2000000 --out planner.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time

from analysis.replay.planner import NoisyOracle, RolloutPlanner
from analysis.replay.run_gate import PROFILES, TRAIN_DAYS, find_compress, load
from analysis.replay.sim import Cheap, Config, Simulator, summarize

_G: dict = {}


def run_one(task: dict) -> dict:
    sessions, compress, horizon = _G[task["day"]]
    cfg = Config(task["slots"], task["kv"] or math.inf, PROFILES["l40s"], policy=task["policy"],
                 deadline_mode="proportional", slack=task["scale"], compress=compress)
    planner = None
    if task["planner"]:
        planner = RolloutPlanner(NoisyOracle(error=task["error"], samples=task["samples"]), refresh_turns=task["refresh"],
                                 candidates=task["candidates"], horizon=task["rollout_s"], samples=task["samples"],
                                 confidence=task.get("confidence", 0.0))
    t0 = time.time()
    sim = Simulator(sessions, cfg, _G["cheap"], planner=planner).run()
    out = {**task, **summarize(sim, horizon), "run_s": round(time.time() - t0, 1)}
    if planner:
        out.update(decisions=planner.decisions, changed=planner.changed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", default="data/freeinference/compact")
    ap.add_argument("--days", required=True)
    ap.add_argument("--scales", default="1.1,1.2,1.5")
    ap.add_argument("--kv", default="0", help="pool sizes in tokens; 0 = unlimited")
    ap.add_argument("--errors", default="0,0.3,0.6", help="stand-in predictor error (sigma of log ratio)")
    ap.add_argument("--refresh", default="1,3", help="re-forecast every this many user turns")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--confidence", default="0", help="share of samples an override must win, comma separated")
    ap.add_argument("--baselines", default="edf,llf_oracle,srpt_oracle", help="policies to run without the planner")
    ap.add_argument("--rollout-s", type=float, default=120.0)
    ap.add_argument("--load", type=float, default=0.9)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # history tables from the training weeks, as run_gate builds them (the oracle policies fall back
    # on them at a session's last call, and history-based eviction would use them)
    train = [s for d in TRAIN_DAYS for s in load(args.compact, d, PROFILES["l40s"], reply="fixed:5", unit="session")]
    _G["cheap"] = Cheap(train, PROFILES["l40s"])
    del train
    for day in args.days.split(","):
        sessions = load(args.compact, day, PROFILES["l40s"], reply="fixed:5", unit="session")
        _G[day] = (sessions, *find_compress(sessions, args.slots, args.load))
    base = dict(slots=args.slots, load=args.load, samples=args.samples, candidates=args.candidates, rollout_s=args.rollout_s)
    tasks = []
    for day in args.days.split(","):
        for scale in map(float, args.scales.split(",")):
            for kv in map(float, args.kv.split(",")):
                cell = dict(base, day=day, scale=scale, kv=kv)
                for p in filter(None, args.baselines.split(",")):
                    tasks.append(dict(cell, policy=p, planner=False, error=None, refresh=None))
                for e in map(float, args.errors.split(",")):
                    for r in map(int, args.refresh.split(",")):
                        for q in map(float, args.confidence.split(",")):
                            tasks.append(dict(cell, policy="edf", planner=True, error=e, refresh=r, confidence=q))
    print(len(tasks), "runs", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool, open(args.out, "w") as f:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks)):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 10 == 0:
                print(i, "done", flush=True)


if __name__ == "__main__":
    main()
