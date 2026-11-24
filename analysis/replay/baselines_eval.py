"""Deadline attainment and mean session time for EDF, Continuum, CacheWise and predictor-driven rules,
under memory pressure and CPU tiers from optimistic to realistic. Same sessions, arrivals and
deadlines (scale x time alone) for every row.

    python -m analysis.replay.baselines_eval --days 2026-08-16 --out baselines.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp

from analysis.replay.run_gate import PROFILES, TRAIN_DAYS, find_compress, load
from analysis.replay.sim import Cheap, Config, Simulator, summarize

L40S = PROFILES["l40s"]
TIERS = {   # name: (profile, cpu_tokens as a multiple of the GPU pool, shared link)
    "recompute (no CPU tier)": (L40S, None, False),
    "CPU tier: unlimited, private links": (PROFILES["l40s_offload"], math.inf, False),
    "CPU tier: 3.2x GPU, shared link": (PROFILES["l40s_offload"], 3.2, True),
    "CPU tier: 1x GPU, shared link": (PROFILES["l40s_offload"], 1.0, True),
    "CPU tier: 3.2x GPU, shared link 10x slower": (dataclasses.replace(PROFILES["l40s_offload"], reload_ms=0.022), 3.2, True),
}
SCHEDULERS = {   # name: Config fields
    "EDF + LRU": dict(policy="edf"),
    "Continuum (TTL pins, program FCFS)": dict(policy="continuum", pin="ttl"),
    "CacheWise (fewest tokens, reuse-aware eviction)": dict(policy="fewest", eviction="cachewise"),
    "EDF + Continuum TTL pins": dict(policy="edf", pin="ttl"),
    "EDF + predictive offload (true tool times)": dict(policy="edf", offload_after_s=10.0),
    "oracle shortest-remaining": dict(policy="srpt_oracle"),
    "predicted shortest-remaining, 10% error": dict(policy="srpt_pred", pred_error=0.1),
}
_G: dict = {}


def run_one(task: dict) -> dict:
    profile, cpu_mult, link = TIERS[task["tier"]]
    sessions, compress, horizon = _G[task["day"]]
    extra = dict(SCHEDULERS[task["scheduler"]])
    if cpu_mult is not None:
        extra.update(cpu_tokens=cpu_mult * task["kv"] if cpu_mult < math.inf else math.inf, shared_link=link)
    cfg = Config(task["slots"], task["kv"], profile, deadline_mode="proportional", slack=task["scale"],
                 compress=compress, **extra)
    return {**task, **summarize(Simulator(sessions, cfg, _G["cheap"]).run(), horizon)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", default="data/freeinference/compact")
    ap.add_argument("--days", required=True)
    ap.add_argument("--scales", default="1.1,1.2,1.5")
    ap.add_argument("--kv", default="2000000,1000000")
    ap.add_argument("--load", type=float, default=0.9)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # one session build serves every tier: service times are worked out at dispatch from the profile,
    # and the history tables only feed eviction scores and Continuum's tool-gap distributions
    train = [s for d in TRAIN_DAYS for s in load(args.compact, d, L40S, reply="fixed:5", unit="session")]
    _G["cheap"] = Cheap(train, L40S)
    del train
    for day in args.days.split(","):
        sessions = load(args.compact, day, L40S, reply="fixed:5", unit="session")
        _G[day] = (sessions, *find_compress(sessions, args.slots, args.load))
    tasks = [dict(day=d, scale=s, kv=k, tier=t, scheduler=n, slots=args.slots, load=args.load)
             for d in args.days.split(",") for s in map(float, args.scales.split(","))
             for k in map(float, args.kv.split(",")) for t in TIERS for n in SCHEDULERS
             if not (TIERS[t][1] is None and "offload" in n)]
    print(len(tasks), "runs", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool, open(args.out, "w") as f:
        for row in pool.imap_unordered(run_one, tasks, chunksize=2):
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
