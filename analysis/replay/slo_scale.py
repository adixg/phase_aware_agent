"""The smallest deadline scale each scheduler needs, from a proportional-deadline run of run_gate.py
(deadline = arrival + scale x the session's time alone on the simulated server). This is the
measure agent-serving papers report: the scale at which 95% (or 99%) of workflows finish on time,
and how much a scheduler lowers it.

    python -m analysis.replay.slo_scale data/freeinference/gate_session_reply5/scale_l40s.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

TARGETS = (0.95, 0.99)
NAMED = [("fcfs", "lru"), ("las", "lru"), ("edf", "lru"), ("edf", "oracle"), ("llf_oracle", "lru"),
         ("llf_oracle", "oracle"), ("srpt_oracle", "oracle")]


def is_oracle(pair: tuple[str, str]) -> bool:
    return pair[0] in ("llf_oracle", "srpt_oracle") or pair[1] == "oracle"


def is_signal(pair: tuple[str, str]) -> bool:
    return pair[0].split("_")[-1] in ("calls", "gaps", "tokens")


def pooled(runs: list[dict]) -> float:
    """Attainment over all days' sessions together, not a mean of daily rates."""
    n = sum(r["feasible_turns"] for r in runs)
    return sum(r["attainment"] * r["feasible_turns"] for r in runs) / n if n else float("nan")


def min_scale(curve: dict[float, float], target: float) -> float | None:
    """Smallest scale reaching the target, interpolated between grid points; None if none does."""
    pts = sorted(curve.items())
    if pts[0][1] >= target:
        return pts[0][0]
    for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
        if a1 >= target:
            return s0 + (s1 - s0) * (target - a0) / (a1 - a0)
    return None


def curves(runs: list[dict], days: list[str]) -> dict:
    """(kv, pair) -> {scale: pooled attainment}, using the given days (repeats allowed)."""
    by = defaultdict(list)
    for r in runs:
        by[(r["kv"], (r["policy"], r["eviction"]), r["tight"], r["day"])].append(r)
    out = defaultdict(dict)
    for kv, pair, scale in {(k[0], k[1], k[2]) for k in by}:
        sel = [r for d in days for r in by.get((kv, pair, scale, d), [])]
        if sel:
            out[(kv, pair)][scale] = pooled(sel)
    return out


def best(c: dict, kv: float, pairs, target: float) -> tuple[tuple[str, str] | None, float | None]:
    found = [(min_scale(c[(kv, p)], target), p) for p in pairs if (kv, p) in c]
    found = [f for f in found if f[0] is not None]
    return (min(found)[1], min(found)[0]) if found else (None, None)


def main(path: str, boot: int = 200) -> None:
    runs = [json.loads(line) for line in open(path)]
    runs = [r for r in runs if r["attainment"] is not None]
    days = sorted({r["day"] for r in runs})
    c = curves(runs, days)
    pairs = sorted({k[1] for k in c})
    cheap = [p for p in pairs if not is_oracle(p) and not is_signal(p)]
    oracle = [p for p in pairs if is_oracle(p) and not is_signal(p)]
    scales = sorted({s for v in c.values() for s in v})
    rng = random.Random(0)
    samples = [[rng.choice(days) for _ in days] for _ in range(boot)]
    boot_c = [curves(runs, s) for s in samples]
    fmt = lambda s: "  >%.1f" % scales[-1] if s is None else "%6.2f" % s

    for kv in sorted({k[0] for k in c}):
        print(f"\n== KV {'unlimited' if kv == 0 else '%dM tokens' % (kv / 1e6)} ({len(days)} days)")
        print("%-22s" % "" + "".join("%7.1f" % s for s in scales) + "   95% at   99% at")
        for p in NAMED:
            if (kv, p) in c:
                cur = c[(kv, p)]
                print("%-22s" % "/".join(p) + "".join("%6.1f%%" % (100 * cur[s]) for s in scales)
                      + "   " + fmt(min_scale(cur, .95)) + "   " + fmt(min_scale(cur, .99)))
        for t in TARGETS:
            pc, sc = best(c, kv, cheap, t)
            po, so = best(c, kv, oracle, t)
            if sc is None or so is None:
                print(f"  {int(100 * t)}%: not reached by the best {'non-oracle' if sc is None else 'oracle'} scheduler")
                continue
            diffs = []
            for bc in boot_c:
                a, b = best(bc, kv, cheap, t)[1], best(bc, kv, oracle, t)[1]
                if a is not None and b is not None:
                    diffs.append(a - b)
            diffs.sort()
            interval = (f"95% interval {diffs[int(.025 * len(diffs))]:.2f} to {diffs[int(.975 * len(diffs)) - 1]:.2f}"
                        if len(diffs) >= 40 else "too few resamples for an interval")
            print(f"  {int(100 * t)}% on time: best non-oracle {'/'.join(pc)} needs {sc:.2f}x, best oracle {'/'.join(po)} "
                  f"{so:.2f}x: {sc - so:.2f} lower ({100 * (sc - so) / sc:.1f}%), {interval}")


if __name__ == "__main__":
    main(sys.argv[1])
