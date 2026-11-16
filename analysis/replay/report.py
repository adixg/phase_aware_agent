"""Tables and the stop-rule verdict from run_gate.py output.

    python analysis/replay/report.py data/freeinference/gate/class_l40s.jsonl
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict

ORACLE_ADMISSION = ("llf_oracle", "srpt_oracle")
CHEAP_ADMISSION = ("fcfs", "returning", "las", "edf", "llf_cheap", "srpt_cheap")
SIGNALS = ("calls", "gaps", "tokens")
REALISTIC_KV = (1e6, 2e6, 4e6)   # roughly one to two large GPUs at ~56 KB per token
SMALL, LARGE = 3.0, 5.0          # points of attainment


def load(path: str):
    cells = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r["attainment"] is not None:
            cells[(r["load"], r["tight"], r["kv"])][(r["policy"], r["eviction"], r["day"])] = r["attainment"]
    return cells


def per_day(cell: dict, pair: tuple[str, str]) -> dict[str, float]:
    return {d: v for (p, e, d), v in cell.items() if (p, e) == pair}


def mean(cell: dict, pair: tuple[str, str]) -> float | None:
    v = per_day(cell, pair)
    return st.mean(v.values()) if v else None


def best(cell: dict, pairs) -> tuple[tuple[str, str] | None, float]:
    scored = [(mean(cell, p), p) for p in pairs if mean(cell, p) is not None]
    return (None, float("nan")) if not scored else max(scored, key=lambda x: x[0])[::-1]


def gap(cell: dict, oracle_pairs, cheap_pairs):
    """Best oracle minus best cheap, in points, with the across-day standard error of that difference."""
    po, o = best(cell, oracle_pairs)
    pc, c = best(cell, cheap_pairs)
    if po is None or pc is None:
        return None
    a, b = per_day(cell, po), per_day(cell, pc)
    diffs = [100 * (a[d] - b[d]) for d in a if d in b]
    se = st.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    return 100 * (o - c), se, po, pc, 100 * c


def label(pair) -> str:
    return "/".join(pair).replace("_cheap", "~").replace("_oracle", "*")


def main(path: str) -> None:
    cells = load(path)
    print(f"# {path}\n")
    verdicts = defaultdict(list)
    adm_cheap = [(a, "lru") for a in CHEAP_ADMISSION]
    adm_oracle = [(a, "lru") for a in ORACLE_ADMISSION]
    evict_cheap = [(a, e) for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap") for e in ("lru", "cheap")]
    joint_cheap = adm_cheap + evict_cheap
    joint_oracle = [(a, e) for a in ORACLE_ADMISSION for e in ("lru", "oracle")] + \
        [(a, "oracle") for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap")]

    for (load_t, tight) in sorted({(k[0], k[1]) for k in cells}):
        print(f"\nload {load_t}, tightness {tight}")
        print("%8s %9s %9s %10s | %8s | %8s | %8s | %s" % ("KV", "FCFS/LRU", "EDF/LRU", "bestCheap",
                                                          "adm gap", "evict", "joint", "best cheap / oracle"))
        for kv in sorted({k[2] for k in cells if k[0] == load_t and k[1] == tight}):
            c = cells[(load_t, tight, kv)]
            kvs = "inf" if kv == 0 else "%dM" % (kv / 1e6)
            fcfs, edf = mean(c, ("fcfs", "lru")), mean(c, ("edf", "lru"))
            ga = gap(c, adm_oracle, adm_cheap)
            line = "%8s %8.1f%% %8.1f%%" % (kvs, 100 * fcfs, 100 * edf)
            if kv == 0:                  # unlimited memory: nothing to evict
                print(line, "%9.1f%% | %+5.1f±%.1f |        - |        - | %s / %s" % (
                    ga[4], ga[0], ga[1], label(ga[3]), label(ga[2])))
                verdicts["admission"].append((load_t, tight, kv, ga[0]))
                continue
            ge = max((gap(c, [(a, "oracle")], [(a, "lru"), (a, "cheap")])
                      for a in ("fcfs", "edf", "llf_cheap", "srpt_cheap")), key=lambda g: g[0])
            gj = gap(c, joint_oracle, joint_cheap)
            print(line, "%9.1f%% | %+5.1f±%.1f | %+5.1f±%.1f | %+5.1f±%.1f | %s / %s" % (
                gj[4], ga[0], ga[1], ge[0], ge[1], gj[0], gj[1], label(gj[3]), label(gj[2])))
            if kv in REALISTIC_KV:
                verdicts["admission"].append((load_t, tight, kv, ga[0]))
                verdicts["eviction"].append((load_t, tight, kv, ge[0]))
                verdicts["joint"].append((load_t, tight, kv, gj[0]))

    print("\n\nWhich signal is worth having? Attainment points over the same policy family with no true signals")
    print("(eviction lru, averaged over the realistic KV sizes and all loads and tightnesses)")
    for fam in ("llf", "srpt"):
        rows = {s: [] for s in SIGNALS + ("all",)}
        for k, c in cells.items():
            if k[2] not in REALISTIC_KV:
                continue
            base = mean(c, (f"{fam}_cheap", "lru"))
            for s in SIGNALS:
                v = mean(c, (f"{fam}_{s}", "lru"))
                if v is not None:
                    rows[s].append(100 * (v - base))
            rows["all"].append(100 * (mean(c, (f"{fam}_oracle", "lru")) - base))
        print(f"  {fam}: " + "  ".join(f"{s} {st.mean(v):+.1f}" for s, v in rows.items() if v))

    print("\n\nVerdict (realistic KV: 1, 2, 4M tokens; unlimited memory also counted for admission)")
    for ch, v in verdicts.items():
        gaps = [x[3] for x in v]
        share_large = sum(g >= LARGE for g in gaps) / len(gaps)
        n_small = sum(g < SMALL for g in gaps)
        peak = max(v, key=lambda x: x[3])
        decision = "drop the predictor" if max(gaps) < SMALL else \
            ("build" if max(gaps) >= LARGE else "noise-test the oracle")
        print(f"  {ch}: gap max {max(gaps):+.1f} at load {peak[0]} tight {peak[1]} KV {peak[2] / 1e6:.0f}M, "
              f"median {st.median(gaps):+.1f}; {n_small}/{len(gaps)} cells under {SMALL:.0f} points, "
              f"{100 * share_large:.0f}% at {LARGE:.0f}+ -> {decision}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
