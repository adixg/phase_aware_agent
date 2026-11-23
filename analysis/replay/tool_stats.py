"""Tool-call statistics from FreeInference traces, as small JSON tables for charts: which tool category
follows which, how long each takes, how much text it returns, how long turns run, and how the
prompt grows. Top-level sessions only (subagents are skipped, as in the replay).

    python analysis/replay/tool_stats.py data/freeinference/release/*/traces/*.jsonl.zst OUT_DIR
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict

import orjson
import zstandard

from analysis.replay.extract import _cut_blocks

START, STOP = "(turn start)", "(model answers)"
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
INPUTS: list[str] = []   # trace files, named in each table's meta


def scan(path: str) -> dict:
    trans, mix, err, gaps, results = Counter(), Counter(), Counter(), defaultdict(list), defaultdict(list)
    turn_calls, parallel, think, prompt_by_step = Counter(), Counter(), [], defaultdict(list)
    with open(path, "rb") as fh:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh)):
            s = orjson.loads(_cut_blocks(line)[0])
            reqs = s["requests"]
            prev, calls_in_turn = START, 0
            for i, r in enumerate(reqs):
                if i and r["step_trigger"] == "user":
                    turn_calls[calls_in_turn] += 1
                    prev, calls_in_turn = START, 0
                    gap = (r["start_ms"] - reqs[i - 1]["end_ms"]) / 1000
                    think.append(max(0.0, gap))
                calls_in_turn += 1
                prompt_by_step[min(i, 400) // 10 * 10].append((r["provider_tokens"] or {}).get("prompt") or 0)
                calls = sorted(r["tool_calls"], key=lambda c: c["call_index"])
                parallel[len(calls)] += 1
                if not calls:
                    trans[(prev, STOP)] += 1
                    prev = STOP
                    continue
                for c in calls:
                    cat = c.get("category") or "Other"
                    trans[(prev, cat)] += 1
                    prev = cat
                    mix[cat] += 1
                    err[cat] += c.get("outcome") in ("error", "timeout")
                    if c.get("result_tokens") is not None:
                        results[cat].append(c["result_tokens"])
                if len(calls) == 1 and calls[0].get("latency_ms") is not None and i + 1 < len(reqs) \
                        and reqs[i + 1]["step_trigger"] == "tool":   # one call: the gap is its own
                    gaps[calls[0].get("category") or "Other"].append(calls[0]["latency_ms"] / 1000)
            turn_calls[calls_in_turn] += 1
    return dict(trans=trans, mix=mix, err=err, gaps=gaps, results=results, turn_calls=turn_calls,
                parallel=parallel, think=think, prompt_by_step=prompt_by_step)


def q(v: list[float], p: float) -> float:
    return v[min(len(v) - 1, int(p * len(v)))]


def write(path: str, header: list[str], rows) -> None:
    """{rows, meta}: the rows, and this script's own text so a reader can see how they were made."""
    meta = [{"label": "tool_stats.py", "value": open(__file__).read(), "type": "code", "lang": "python"},
            {"label": "Input", "value": "FreeInference agentic trace, " + ", ".join(os.path.basename(p) for p in INPUTS)}]
    with open(path, "w") as f:
        json.dump({"rows": [dict(zip(header, r)) for r in rows], "meta": meta}, f)


def main(paths: list[str], out: str) -> None:
    os.makedirs(out, exist_ok=True)
    INPUTS[:] = paths
    with mp.get_context("fork").Pool(3) as pool:
        parts = pool.map(scan, paths)
    agg = {k: (Counter() if isinstance(v, Counter) else defaultdict(list) if isinstance(v, defaultdict) else [])
           for k, v in parts[0].items()}
    for p in parts:
        for k, v in p.items():
            if isinstance(v, Counter):
                agg[k].update(v)
            elif isinstance(v, defaultdict):
                for kk, vv in v.items():
                    agg[k][kk].extend(vv)
            else:
                agg[k].extend(v)

    out_of = Counter()
    for (a, _), n in agg["trans"].items():
        out_of[a] += n
    write(f"{out}/transitions.json", ["from", "to", "count", "share_of_from"],
          [(a, b, n, round(n / out_of[a], 5)) for (a, b), n in sorted(agg["trans"].items())])
    total = sum(agg["mix"].values())
    rows = []
    for cat, n in agg["mix"].most_common():
        g, rt = sorted(agg["gaps"].get(cat, [])), sorted(agg["results"].get(cat, []))
        rows.append([cat, n, round(100 * n / total, 2), round(100 * agg["err"][cat] / n, 2), len(g)]
                    + [round(q(g, p), 3) if g else None for p in QUANTILES]
                    + [round(sum(g) / len(g), 3) if g else None]
                    + [q(rt, 0.5) if rt else None, q(rt, 0.9) if rt else None])
    write(f"{out}/categories.json", ["category", "calls", "share_pct", "error_pct", "timed_calls"]
          + [f"gap_p{int(100 * p)}_s" for p in QUANTILES] + ["gap_mean_s", "result_tokens_p50", "result_tokens_p90"], rows)
    write(f"{out}/turn_calls.json", ["llm_calls", "turns"], sorted(agg["turn_calls"].items()))
    write(f"{out}/parallel_calls.json", ["tool_calls_in_response", "responses"], sorted(agg["parallel"].items()))
    think = sorted(agg["think"])
    edges = [0, 5, 15, 30, 60, 120, 300, 600, 1800, 3600, 4 * 3600, 24 * 3600, float("inf")]
    names = ["<5 s", "5-15 s", "15-30 s", "30-60 s", "1-2 min", "2-5 min", "5-10 min", "10-30 min",
             "30-60 min", "1-4 h", "4-24 h", ">1 day"]
    counts = [sum(lo <= t < hi for t in think) for lo, hi in zip(edges, edges[1:])]
    write(f"{out}/think_time.json", ["bucket", "order", "turns", "share_pct"],
          [(n, i, c, round(100 * c / len(think), 2)) for i, (n, c) in enumerate(zip(names, counts))])
    write(f"{out}/prompt_growth.json", ["request_index", "requests", "prompt_p25", "prompt_p50", "prompt_p75"],
          [(k, len(v), q(sorted(v), .25), q(sorted(v), .5), q(sorted(v), .75))
           for k, v in sorted(agg["prompt_by_step"].items()) if len(v) >= 200])
    print("calls", total, "turns", sum(agg["turn_calls"].values()), "files", len(paths))


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
