"""Check the built DAGs against the raw counts and measure their structure.

    python -m analysis.dag.census data/freeinference/dag

Validation: every request became a node, parallel tool calls share one latency that equals the
round's wall time, and no round runs backwards in time. Census: how wide the rounds are, how long
turns are, how many turns hold a subagent, and how often LLM requests of one turn overlap.
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from collections import Counter

import orjson


def quantiles(v: list[float], qs=(0.5, 0.9, 0.99)) -> list[float]:
    v = sorted(v)
    return [v[min(len(v) - 1, int(q * len(v)))] for q in qs] if v else []


def walk(rounds: list[dict]):
    """Every round of a chain and of the subagents under it."""
    for r in rounds:
        yield r
        for sub in r["subs"]:
            yield from walk(sub["rounds"])


def max_overlap(intervals: list[tuple[int, int]]) -> int:
    ev = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals], key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def main(folder: str) -> None:
    raw = json.load(open(os.path.join(folder, "stats.json")))
    c = Counter()
    rounds_per_turn, widths, round_ms = [], Counter(), {"1 tool": [], "2+ tools": []}
    sub_flags, overlap_turns = Counter(), []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        for line in open(path, "rb"):
            t = orjson.loads(line)
            c["turns"] += 1
            top = t["rounds"]
            rounds_per_turn.append(len(top))
            c["requests_top"] += len(top)
            has_sub, fan = False, False
            for r in walk(top):
                c["requests_all"] += 1
            for r in top:
                w = len(r["tools"])
                widths[min(w, 5)] += 1
                fan |= w > 1
                if w > 1:
                    lats = {x["latency_ms"] for x in r["tools"]}
                    c["parallel_rounds"] += 1
                    c["parallel_latency_identical"] += len(lats) == 1
                    c["parallel_latency_eq_round"] += lats == {r["round_ms"]}
                if r["round_ms"] is not None and w >= 1:
                    round_ms["1 tool" if w == 1 else "2+ tools"].append(r["round_ms"] / 1000)
                if w == 1 and r["round_ms"] is not None:
                    c["single_latency_eq_round"] += r["tools"][0]["latency_ms"] == r["round_ms"]
                    c["single_rounds"] += 1
                for sub in r["subs"]:
                    has_sub = True
                    sub_flags[sub["outlives_next"]] += 1
            c["max_width"] = max(c["max_width"], max((len(r["tools"]) for r in top), default=0))
            c["turns_with_fanout"] += fan
            if has_sub:
                c["turns_with_subagent"] += 1
                iv = [(r["start"], r["end"]) for r in walk(top)]
                overlap_turns.append(max_overlap(iv))
    n = c["turns"]
    print(f"== validation ({raw['week']})")
    print(f"  requests: DAG {c['requests_all']:,} vs raw {raw['requests_top'] + raw.get('subagent_requests', 0):,} "
          f"(top {c['requests_top']:,} vs {raw['requests_top']:,})")
    print(f"  turns: {n:,} (raw count {raw['turns']:,}); turn_index disagreements {raw.get('turn_index_mismatch', 0)}; "
          f"negative round times {raw.get('negative_round_ms', 0)}")
    pr = c["parallel_rounds"]
    print(f"  parallel rounds with one shared latency: {c['parallel_latency_identical']}/{pr}; "
          f"equal to the round's wall time: {c['parallel_latency_eq_round']}/{pr}")
    print(f"  single-tool rounds whose latency equals the round's wall time: "
          f"{c['single_latency_eq_round']}/{c['single_rounds']}")
    print("\n== structure")
    print(f"  turns {n:,}, rounds {c['requests_top']:,}; rounds per turn: median/p90/p99 = "
          f"{'/'.join(str(x) for x in quantiles(rounds_per_turn))}, mean {st.mean(rounds_per_turn):.1f}")
    total = sum(widths.values())
    print("  tool calls per round: " + ", ".join(f"{'5+' if k == 5 else k}: {100 * widths[k] / total:.1f}%" for k in sorted(widths)))
    print(f"  fan-out (2+ tools): {100 * pr / total:.1f}% of rounds, {100 * c['turns_with_fanout'] / n:.1f}% of turns; widest {c['max_width']}")
    for k, v in round_ms.items():
        if v:
            print(f"  round wall time, {k}: median/p90/p99 = {'/'.join(f'{x:.1f}s' for x in quantiles(v))} (n={len(v):,})")
    sub_calls, sub_req = raw.get("subagent_calls", 0), raw.get("subagent_requests", 0)
    print(f"  subagents: {sub_calls} calls, {sub_req:,} requests = {100 * sub_req / (raw['requests_top'] + sub_req):.1f}% of all requests; "
          f"{c['turns_with_subagent']} turns ({100 * c['turns_with_subagent'] / n:.2f}%) hold one")
    known = sub_flags[True] + sub_flags[False]
    if known:
        print(f"  subagent outlives the parent's next request: {sub_flags[True]}/{known} "
              f"({100 * sub_flags[True] / known:.0f}%) (undefined for {sub_flags[None]} in a turn's last round)")
    if overlap_turns:
        multi = sum(x >= 2 for x in overlap_turns)
        print(f"  in turns with a subagent, simultaneous LLM requests: max {max(overlap_turns)}, "
              f">=2 in {multi}/{len(overlap_turns)} turns")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/freeinference/dag")
