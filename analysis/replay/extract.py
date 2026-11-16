"""Reduce a FreeInference trace day to what the replay needs: one JSON file of sessions whose
requests carry token counts, the prefix shared with the previous request, and the gap after it.

A day decompresses to several GB, nearly all of it block_ids, so those are cut out of each
line and parsed as int arrays before the JSON is decoded.

    python analysis/replay/extract.py data/freeinference/release/WEEK/traces/DAY.jsonl.zst OUT.json
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter

import numpy as np
import orjson
import zstandard

KEY = b'"block_ids"'
BLOCK_TOKENS = 16


def _cut_blocks(line: bytes) -> tuple[bytes, list[np.ndarray]]:
    """The line with every block_ids list emptied, and those lists in document order."""
    parts, arrays, pos = [], [], 0
    while (i := line.find(KEY, pos)) >= 0:
        start = line.index(b"[", i) + 1
        end = line.index(b"]", start)
        parts += [line[pos:start], b"]"]
        body = line[start:end]
        arrays.append(np.fromstring(body, dtype=np.int64, sep=",") if body.strip()
                      else np.empty(0, dtype=np.int64))
        pos = end + 1
    parts.append(line[pos:])
    return b"".join(parts), arrays


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = a[:n] != b[:n]
    return int(diff.argmax()) if diff.any() else n


def _slowest_tool(req: dict) -> tuple[str, int]:
    calls = req["tool_calls"]
    if not calls:
        return "__none__", 0
    slowest = max(calls, key=lambda c: c.get("latency_ms") or 0)
    return slowest.get("category") or "Other", len(calls)


def _walk(requests: list[dict], arrays, own: list | None) -> int:
    """Consume `arrays` in document order (a request's block_ids precede its tool calls);
    top-level arrays go to `own`, subagents' are dropped. Returns the subagent requests seen."""
    nested = 0
    for r in requests:
        a = next(arrays)
        if own is not None:
            own.append(a)
        for c in r["tool_calls"]:
            if c.get("subagent"):
                sub = c["subagent"]["requests"]
                nested += len(sub) + _walk(sub, arrays, None)
    return nested


def session_record(s: dict, arrays: list[np.ndarray], stats: Counter, seen_first: set) -> dict:
    reqs, own = s["requests"], []
    stats["subagent_requests_dropped"] += _walk(reqs, iter(arrays), own)
    out, prev = [], None
    for i, r in enumerate(reqs):
        cur = own[i]
        lcp = _lcp(prev, cur) if prev is not None else 0
        cat, n_tools = _slowest_tool(r)
        next_start = reqs[i + 1]["start_ms"] if i + 1 < len(reqs) else None
        tok = r["provider_tokens"] or {}
        out.append({
            "start": r["start_ms"], "end": r["end_ms"], "ttft": r["ttft_ms"],
            "prompt": len(cur) * BLOCK_TOKENS, "lcp": lcp * BLOCK_TOKENS,
            "provider_prompt": tok.get("prompt"), "completion": tok.get("completion") or 0,
            "cached": tok.get("cached"), "user": r["step_trigger"] == "user",
            "append": (r["mutation"] or {}).get("transition") == "append",
            "cat": cat, "n_tools": n_tools, "model": r["model"],
            "gap": None if next_start is None else max(0, next_start - r["end_ms"]),
        })
        prev = cur
    first = own[0].tolist()
    stats["first_blocks"] += len(first)
    stats["first_blocks_seen_before"] += sum(b in seen_first for b in first)
    seen_first.update(first)
    return {"sid": s["session_id"], "user": s["user_id"], "harness": s["harness"],
            "start": s["start_ms"], "requests": out}


def main(path: str, out_path: str) -> None:
    stats, seen_first, sessions = Counter(), set(), []
    with open(path, "rb") as fh:
        for line in io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh)):
            stripped, arrays = _cut_blocks(line)
            s = orjson.loads(stripped)
            del stripped, line
            if s["requests"]:
                sessions.append(session_record(s, arrays, stats, seen_first))
            del arrays
    stats["sessions"] = len(sessions)
    stats["requests"] = sum(len(s["requests"]) for s in sessions)
    with open(out_path, "wb") as fh:
        fh.write(orjson.dumps({"stats": stats, "sessions": sessions}))
    print(path.rsplit("/", 1)[-1], json.dumps(stats), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
