"""Predictive KV offload under memory pressure: offloading a session's cache when its tool gap is
predicted to be long and prefetching it before the predicted return, against recompute and
against an on-demand CPU tier with LRU or oracle eviction. Every variant schedules with EDF.

    python -m analysis.replay.offload_eval --days 2026-08-16 --out offload.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp

from analysis.replay.run_gate import PROFILES, TRAIN_DAYS, find_compress, load
from analysis.replay.sim import Cheap, Config, Simulator, summarize

VARIANTS = {   # name: (profile, eviction, offload_after_s, gap_error)
    "recompute, LRU": ("l40s", "lru", None, 0.0),
    "CPU tier on demand, LRU": ("l40s_offload", "lru", None, 0.0),
    "CPU tier on demand, oracle eviction": ("l40s_offload", "oracle", None, 0.0),
    "predictive offload >10 s, true gaps": ("l40s_offload", "lru", 10.0, 0.0),
    "predictive offload >30 s, true gaps": ("l40s_offload", "lru", 30.0, 0.0),
    "predictive offload >10 s, gaps +-30%": ("l40s_offload", "lru", 10.0, 0.3),
    "predictive offload >10 s, gaps +-60%": ("l40s_offload", "lru", 10.0, 0.6),
}
_G: dict = {}


def run_one(task: dict) -> dict:
    profile, eviction, after, err = VARIANTS[task["variant"]]
    sessions, compress, horizon = _G[(task["day"], profile)]
    cfg = Config(task["slots"], task["kv"], PROFILES[profile], policy=task["policy"], eviction=eviction,
                 deadline_mode="proportional", slack=task["scale"], compress=compress,
                 offload_after_s=after, gap_error=err)
    return {**task, **summarize(Simulator(sessions, cfg, _G[("cheap", profile)]).run(), horizon)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", default="data/freeinference/compact")
    ap.add_argument("--days", required=True)
    ap.add_argument("--scales", default="1.1,1.2,1.5")
    ap.add_argument("--kv", default="2000000,1000000")
    ap.add_argument("--policies", default="edf")
    ap.add_argument("--load", type=float, default=0.9)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for profile in ("l40s", "l40s_offload"):
        train = [s for d in TRAIN_DAYS for s in load(args.compact, d, PROFILES[profile], reply="fixed:5", unit="session")]
        _G[("cheap", profile)] = Cheap(train, PROFILES[profile])
        del train
        for day in args.days.split(","):
            sessions = load(args.compact, day, PROFILES[profile], reply="fixed:5", unit="session")
            # the same arrivals for every variant: the load is set from the recompute profile's work
            _G[(day, profile)] = (sessions, *(_G[(day, "l40s")][1:] if profile != "l40s" else find_compress(sessions, args.slots, args.load)))
    tasks = [dict(day=d, scale=s, kv=k, policy=p, variant=v, slots=args.slots, load=args.load)
             for d in args.days.split(",") for s in map(float, args.scales.split(","))
             for k in map(float, args.kv.split(",")) for p in args.policies.split(",") for v in VARIANTS]
    print(len(tasks), "runs", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool, open(args.out, "w") as f:
        for row in pool.imap_unordered(run_one, tasks, chunksize=2):
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
