"""Follow-up checks on the gate results that report.py's summary hides:

- eviction gain for each admission policy separately (the report takes the best over all four,
  which lets oracle eviction rescue a weak admission policy),
- the proportional-deadline sweep by slack,
- mean and p95 latency, an objective the deadline policies were not built for.

    python -m analysis.replay.breakdown data/freeinference/gate
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict

from analysis.replay.report import REALISTIC_KV, best, load, mean

SWEEPS = ("class_l40s", "class_trace", "prop_l40s", "class_offload", "class_l40s_slots16")
CHEAP = [(a, "lru") for a in ("fcfs", "returning", "las", "edf", "llf_cheap", "srpt_cheap")] + \
    [(a, "cheap") for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap")]
ORACLE = [(a, e) for a in ("llf_oracle", "srpt_oracle") for e in ("lru", "oracle")] + \
    [(a, "oracle") for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap")]


def evict_gap(cell: dict, admission: str) -> float | None:
    o = mean(cell, (admission, "oracle"))
    cheap = [x for x in (mean(cell, (admission, "lru")), mean(cell, (admission, "cheap"))) if x is not None]
    return None if o is None or not cheap else 100 * (o - max(cheap))


def eviction_by_admission(gate: str) -> None:
    for sweep in SWEEPS:
        path = os.path.join(gate, f"{sweep}.jsonl")
        if not os.path.exists(path):
            continue
        cells = load(path)
        print("==", sweep)
        for a in ("edf", "fcfs", "llf_cheap", "srpt_cheap"):
            v = [(evict_gap(c, a), k) for k, c in cells.items() if k[2] in REALISTIC_KV]
            v = [x for x in v if x[0] is not None]
            top = max(v, key=lambda x: x[0])
            print(f"  eviction gap with {a} admission: median {st.median(x[0] for x in v):+.1f} "
                  f"max {top[0]:+.1f} at load/tight/kv {top[1]}, cells>=3: {sum(x[0] >= 3 for x in v)}/{len(v)}")


def proportional_by_slack(gate: str) -> None:
    cells = load(os.path.join(gate, "prop_l40s.jsonl"))
    print("== proportional: best oracle pair - best non-oracle pair, by slack")
    by = defaultdict(list)
    for k, c in cells.items():
        if k[2] in REALISTIC_KV:
            pc, cv = best(c, CHEAP)
            po, ov = best(c, ORACLE)
            by[k[1]].append((100 * (ov - cv), 100 * cv, k, pc, po))
    for slack in sorted(by):
        v = by[slack]
        top = max(v)
        print(f"  slack {slack}: median gap {st.median(x[0] for x in v):+.1f} max {top[0]:+.1f} "
              f"(best cheap attainment there {top[1]:.1f}%, cell {top[2]}, {top[3]} vs {top[4]}); "
              f"mean cheap attainment {st.mean(x[1] for x in v):.1f}%")


def latency(gate: str, tight: float = 1.0) -> None:
    print(f"== mean / p95 latency, class l40s, tightness {tight}: best non-oracle vs best oracle, mean over days")
    runs = defaultdict(lambda: defaultdict(dict))
    for line in open(os.path.join(gate, "class_l40s.jsonl")):
        r = json.loads(line)
        if r["tight"] == tight:
            runs[(r["load"], r["kv"])][(r["policy"], r["eviction"])][r["day"]] = (r["mean_latency"], r["p95_latency"])
    print("  load  KV   fcfs  | best cheap | best oracle | oracle vs cheap: mean, p95")
    for k in sorted(runs):
        if k[1] not in (0, *REALISTIC_KV):
            continue
        d = runs[k]
        m = lambda p, i: st.mean(x[i] for x in d[p].values())
        cs = [p for p in d if p in CHEAP and d[p]]
        os_ = [p for p in d if p in ORACLE and d[p]]
        bc, bo = min(cs, key=lambda p: m(p, 0)), min(os_, key=lambda p: m(p, 0))
        bc95, bo95 = min(m(p, 1) for p in cs), min(m(p, 1) for p in os_)
        print(f"  {k[0]} {k[1] / 1e6:4.0f}M {m(('fcfs', 'lru'), 0):7.0f} | {m(bc, 0):7.0f} {'/'.join(bc):16s} | "
              f"{m(bo, 0):7.0f} {'/'.join(bo):18s} | {100 * (m(bo, 0) / m(bc, 0) - 1):+6.1f}% {100 * (bo95 / bc95 - 1):+6.1f}%")


if __name__ == "__main__":
    gate = sys.argv[1] if len(sys.argv) > 1 else "data/freeinference/gate"
    eviction_by_admission(gate)
    proportional_by_slack(gate)
    latency(gate)
