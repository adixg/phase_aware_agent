"""The oracle gate: how much deadline attainment does knowing the future add over what history
alone gives? Cheap tables and deadline budgets come from the training days, results from the test days.

    python analysis/replay/run_gate.py --compact data/freeinference/compact --out runs.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import multiprocessing as mp
import os
import time

import orjson

from analysis.replay.sim import Cheap, Config, Profile, Simulator, build_sessions, reply_time, summarize

def days(first: str, last: str) -> list[str]:
    a, b = datetime.date.fromisoformat(first), datetime.date.fromisoformat(last)
    return [str(a + datetime.timedelta(n)) for n in range((b - a).days + 1)]


# in time order: history tables and deadline budgets from the first 8 weeks, results from the last 8
TRAIN_DAYS = days("2026-05-17", "2026-07-11")
TEST_DAYS = days("2026-07-12", "2026-09-05")

PROFILES = {
    "l40s": Profile(decode_ms=22.4, prefill_ms=0.046, overhead_ms=140.0),  # measured, validate.py
    "l40s_offload": Profile(decode_ms=22.4, prefill_ms=0.046, overhead_ms=140.0, reload_ms=0.0022),
    "trace": Profile(decode_ms=4.0, prefill_ms=0.134),                     # FreeInference medians
}
BUDGET_QUANTILES = (0.5, 0.75, 0.95)  # classes: a turn draws one at random, whatever its length

# (admission, eviction)
ADMISSION_CHEAP = ["fcfs", "returning", "las", "edf", "llf_cheap", "srpt_cheap"]
ADMISSION_ORACLE = ["llf_oracle", "srpt_oracle"]
ADMISSION_SIGNALS = ["llf_calls", "llf_gaps", "llf_tokens", "srpt_calls", "srpt_gaps", "srpt_tokens"]
PAIRS = [(a, "lru") for a in ADMISSION_CHEAP + ADMISSION_ORACLE + ADMISSION_SIGNALS] + \
    [(a, e) for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap") for e in ("cheap", "oracle")] + \
    [(a, "oracle") for a in ADMISSION_ORACLE]

_G: dict = {}


def load(compact: str, day: str, profile: Profile, reply: str = "recorded", unit: str = "turn"):
    """One day's sessions. A day at a week boundary is in both weeks' releases, with different
    sessions, so compact files are named WEEK__DAY.json and a day is all of them."""
    raw = [s for path in sorted(glob.glob(os.path.join(compact, f"*__{day}.json")))
           for s in orjson.loads(open(path, "rb").read())["sessions"]]
    return build_sessions(raw, profile, reply=reply_time(reply), unit=unit) if raw else []


def find_compress(sessions, slots: int, target: float) -> tuple[float, float]:
    """Compression of session start times that spreads the day's GPU work over a horizon where
    work / (slots * horizon) = target. Returns (compress, horizon)."""
    work = sum(r.ideal for s in sessions for r in s.reqs)
    horizon = work / (slots * target)
    span = max(s.arrival for s in sessions) - min(s.arrival for s in sessions)
    return span / horizon, horizon


def compress_task(a: tuple) -> tuple:
    pn, day, target, slots = a
    return a, find_compress(_G["test"][(pn, day)], slots, target)


def run_one(task: dict) -> dict:
    sessions = _G["test"][(task["profile"], task["day"])]
    profile = PROFILES[task["profile"]]
    cfg = Config(task["slots"], task["kv"] if task["kv"] else math.inf, profile, policy=task["policy"],
                 eviction=task["eviction"], deadline_mode=task["mode"], slack=task["tight"],
                 budgets=tuple(q * task["tight"] for q in _G["quantiles"][task["profile"]]),
                 compress=task["compress"], preempt=task.get("preempt", "none"))
    s = summarize(Simulator(sessions, cfg, _G["cheap"][task["profile"]]).run(), task["horizon"])
    return {**task, **s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", default="data/freeinference/compact")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiles", default="l40s")
    ap.add_argument("--days", default=",".join(TEST_DAYS))
    ap.add_argument("--loads", default="0.5,0.7,0.9")
    ap.add_argument("--kv", default="0", help="pool sizes in tokens, comma separated; 0 = unlimited")
    ap.add_argument("--tight", default="1,2,4", help="multiplier on class budgets, or the slack if proportional")
    ap.add_argument("--mode", default="class", choices=["class", "proportional"])
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--reply", default="recorded",
                    help="user pause between turns: recorded, scale:K, fixed:S, exp:MEAN or lognormal:MEDIAN,SIGMA")
    ap.add_argument("--unit", default="turn", choices=["turn", "session"], help="what each deadline covers")
    ap.add_argument("--preempt", default="none", choices=["none", "priority"],
                    help="priority: a waiting request that does not fit may preempt running ones ranked below it")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    profiles, days = args.profiles.split(","), args.days.split(",")
    _G.update(test={}, cheap={}, quantiles={})
    for pn in profiles:
        p = PROFILES[pn]
        train = [s for d in TRAIN_DAYS for s in load(args.compact, d, p, args.reply, args.unit)]
        _G["cheap"][pn] = Cheap(train, p)
        z = sorted(x for s in train for x in s.z)
        _G["quantiles"][pn] = [z[int(q * len(z))] for q in BUDGET_QUANTILES]
        print(pn, "budgets (s) at tightness 1:", [round(x, 1) for x in _G["quantiles"][pn]], flush=True)
        del train
        for d in days:
            _G["test"][(pn, d)] = load(args.compact, d, p, args.reply, args.unit)

    loads = [float(x) for x in args.loads.split(",")]
    with mp.get_context("fork").Pool(args.workers) as pool:
        compress = dict(pool.map(compress_task, [(pn, d, l, args.slots) for pn in profiles for d in days for l in loads]))
        for k, v in compress.items():
            print(k, "compress %.2f horizon %.0f s" % v, flush=True)
        tasks = [dict(profile=pn, day=d, load=l, compress=compress[(pn, d, l, args.slots)][0], horizon=compress[(pn, d, l, args.slots)][1], slots=args.slots, reply=args.reply, unit=args.unit, preempt=args.preempt, kv=float(kv),
                      mode=args.mode, tight=float(t), policy=a, eviction=e)
                 for pn in profiles for d in days for l in loads for kv in args.kv.split(",")
                 for t in args.tight.split(",") for a, e in PAIRS if not (e != "lru" and float(kv) == 0)]
        print(len(tasks), "runs", flush=True)
        with open(args.out, "w") as f:
            for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4)):
                f.write(json.dumps(r) + "\n")
                if i % 500 == 0:
                    print(i, round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
