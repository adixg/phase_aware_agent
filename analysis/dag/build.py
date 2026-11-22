"""Build the agent DAG of every turn from raw FreeInference sessions.

A turn starts at a user-triggered request and runs until the next one. Inside it, each LLM request
is followed by a round of tool calls (several = a fan-out) and the next request starts when the
round is over:

    LLM -> tool, tool, ... -> LLM -> tool -> LLM ...

What the trace does and does not say:
  * Every tool call in a round carries the same latency_ms, the round's wall time (the gap from the
    request's end to the next request's start). Per-tool durations, and so the straggler, are not
    recoverable; `round_ms` is the one duration a round has.
  * A tool call can hold a subagent session. Its latency_ms is short next to the child's span, so the
    child usually runs on after the parent moved on. No join edge is invented: each subagent records
    whether it outlived the parent's next request.

Train and test are separate weeks. This builds one week, the training week by default.

    python -m analysis.dag.build                      # data/freeinference/dag/<day>.jsonl, training week
    python -m analysis.dag.build --week test          # only when you are done designing
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import Counter

import orjson
import zstandard

WEEKS = {"train": "2026-08-23_2026-08-29", "test": "2026-08-30_2026-09-05"}
KEY = b'"block_ids"'


def strip_blocks(line: bytes) -> bytes:
    """The line with every block_ids list emptied: a day is several GB, nearly all of it block ids."""
    parts, pos = [], 0
    while (i := line.find(KEY, pos)) >= 0:
        start = line.index(b"[", i) + 1
        end = line.index(b"]", start)
        parts += [line[pos:start], b"]"]
        pos = end + 1
    parts.append(line[pos:])
    return b"".join(parts)


def _tool(c: dict) -> dict:
    return {"i": c["call_index"], "name": c["name"], "cat": c.get("category") or "Other",
            "outcome": c.get("outcome"), "result_tokens": c.get("result_tokens"),
            "latency_ms": c.get("latency_ms")}


def rounds_of(requests: list[dict], stats: Counter, base: int = 0) -> list[dict]:
    """One round per request, in order: the LLM request, its tool calls and their subagents.
    `round_ms` is the wall time until the next request in the list (None for the last)."""
    out = []
    for i, r in enumerate(requests):
        nxt = requests[i + 1] if i + 1 < len(requests) else None
        tok = r.get("provider_tokens") or {}
        round_ms = None if nxt is None else nxt["start_ms"] - r["end_ms"]
        if round_ms is not None and round_ms < 0:
            stats["negative_round_ms"] += 1
        subs = []
        for c in r["tool_calls"]:
            sub = c.get("subagent")
            if not sub:
                continue
            stats["subagent_calls"] += 1
            stats["subagent_requests"] += len(sub["requests"])
            child = rounds_of(sub["requests"], stats)
            subs.append({"call": c["call_index"], "sid": sub["session_id"],
                         "start": sub["start_ms"], "end": sub["end_ms"], "rounds": child,
                         "outlives_next": None if nxt is None else sub["end_ms"] > nxt["start_ms"]})
        out.append({"req": base + i, "start": r["start_ms"], "end": r["end_ms"], "ttft": r.get("ttft_ms"),
                    "prompt": tok.get("prompt"), "completion": tok.get("completion"), "cached": tok.get("cached"),
                    "finish": r.get("finish_reason"), "trigger": r.get("step_trigger"),
                    "tools": [_tool(c) for c in r["tool_calls"]], "round_ms": round_ms, "subs": subs})
    return out


def session_turns(s: dict, stats: Counter) -> list[dict]:
    """The session's turns, each one a chain of rounds."""
    reqs = s["requests"]
    stats["sessions"] += 1
    stats["requests_top"] += len(reqs)
    cuts = [i for i, r in enumerate(reqs) if i == 0 or r["step_trigger"] == "user"]
    turns, prev_end = [], None
    for t, a in enumerate(cuts):
        b = cuts[t + 1] if t + 1 < len(cuts) else len(reqs)
        if reqs[a].get("turn_index") is not None and reqs[a]["turn_index"] != t:
            stats["turn_index_mismatch"] += 1
        rounds = rounds_of(reqs[a:b], stats, base=a)
        turns.append({"session": s["session_id"], "user": s["user_id"], "harness": s["harness"],
                      "day": s["day"], "turn": t, "start": reqs[a]["start_ms"], "end": reqs[b - 1]["end_ms"],
                      "idle_before_ms": None if prev_end is None else reqs[a]["start_ms"] - prev_end,
                      "rounds": rounds})
        prev_end = reqs[b - 1]["end_ms"]
        stats["turns"] += 1
    return turns


def to_graph(turn: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    """Explicit nodes and edges of one turn: LLM -> its tool calls -> the next LLM. A subagent hangs
    off its tool call (spawn edge) and has no edge back: see the module docstring."""
    nodes, edges = [], []

    def add(rounds: list[dict], prefix: str) -> None:
        for k, r in enumerate(rounds):
            llm = f"{prefix}L{k}"
            nodes.append({"id": llm, "kind": "llm", "start": r["start"], "end": r["end"],
                          "prompt": r["prompt"], "completion": r["completion"]})
            nxt = f"{prefix}L{k + 1}" if k + 1 < len(rounds) else None
            for t in r["tools"]:
                tid = f"{prefix}L{k}.T{t['i']}"
                nodes.append({"id": tid, "kind": "tool", "cat": t["cat"], "round_ms": r["round_ms"]})
                edges.append((llm, tid))
                if nxt:
                    edges.append((tid, nxt))
            if nxt and not r["tools"]:
                edges.append((llm, nxt))
            for sub in r["subs"]:
                child = f"{prefix}L{k}.T{sub['call']}.S{sub['sid'][:6]}."
                if sub["rounds"]:
                    edges.append((f"{prefix}L{k}.T{sub['call']}", f"{child}L0"))
                    add(sub["rounds"], child)

    add(turn["rounds"], "")
    return nodes, edges


def build_day(path: str, out_path: str, stats: Counter) -> None:
    with open(path, "rb") as fh, open(out_path, "wb") as out:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh)):
            s = orjson.loads(strip_blocks(line))
            del line
            if not s["requests"]:
                continue
            for turn in session_turns(s, stats):
                out.write(orjson.dumps(turn) + b"\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--week", choices=list(WEEKS), default="train",
                    help="train (Aug 23-29, the default) or test (Aug 30-Sep 5)")
    ap.add_argument("--root", default="data/freeinference/release")
    ap.add_argument("--out", default="data/freeinference/dag")
    args = ap.parse_args()
    week = WEEKS[args.week]
    folder = os.path.join(args.root, week, "traces")
    os.makedirs(args.out, exist_ok=True)
    total = Counter()
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".jsonl.zst"):
            continue
        day = name[: -len(".jsonl.zst")]
        if args.week == "test" and day == "2026-08-29":
            continue                      # a 4-session fragment of a day the training week holds in full
        stats = Counter()
        build_day(os.path.join(folder, name), os.path.join(args.out, f"{day}.jsonl"), stats)
        total.update(stats)
        print(day, json.dumps(stats), flush=True)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({"week": week, **total}, f, indent=1)


if __name__ == "__main__":
    main()
