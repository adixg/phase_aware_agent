"""How long sessions and turns last, as small JSON tables for charts: each session's wall-clock
span, each turn's recorded duration and its duration on an empty simulated server, how long
users take to reply between turns, each session's time without those replies (its turns added
up: only LLM calls and tool calls), and each session on the empty simulated server with 5 s
replies, the time the gate's session deadline budgets are drawn from, with those budgets.

    python analysis/replay/durations.py data/freeinference/compact OUT_DIR [WEEK,WEEK,...]
"""

from __future__ import annotations

import json
import math
import os
import sys

import orjson

from analysis.replay.run_gate import BUDGET_QUANTILES, PROFILES, TRAIN_DAYS
from analysis.replay.sim import build_sessions, reply_time

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
EDGES = [0, 10, 30, 60, 300, 900, 1800, 3600, 3 * 3600, 8 * 3600, 24 * 3600, 3 * 86400, math.inf]
NAMES = ["<10 s", "10-30 s", "30 s-1 min", "1-5 min", "5-15 min", "15-30 min", "30-60 min", "1-3 h",
         "3-8 h", "8-24 h", "1-3 days", ">3 days"]
ACTIVE_EDGES = [0, 60, 300, 900, 1800, 3600, 2 * 3600, 4 * 3600, 8 * 3600, 24 * 3600, math.inf]
ACTIVE_NAMES = ["<1 min", "1-5 min", "5-15 min", "15-30 min", "30-60 min", "1-2 h", "2-4 h", "4-8 h", "8-24 h", ">1 day"]


def q(v: list[float], p: float) -> float:
    return v[min(len(v) - 1, int(p * len(v)))]


def write(path: str, rows: list[dict], inputs: list[str]) -> None:
    meta = [{"label": "durations.py", "value": open(__file__).read(), "type": "code", "lang": "python"},
            {"label": "Input", "value": "FreeInference agentic trace, reduced by extract.py: " + ", ".join(inputs)}]
    with open(path, "w") as f:
        json.dump({"rows": rows, "meta": meta}, f)


def main(compact: str, out: str, weeks: list[str] | None = None) -> None:
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(compact) if f.endswith(".json")   # WEEK__DAY.json
                   and (not weeks or f.split("__")[0] in weeks))
    sessions, recorded, empty, train_empty, replies = [], [], [], [], []
    active, active_empty = [], []   # per session: wall clock less user replies; simulated with 5 s replies
    for name in files:
        day = name[:-5].split("__")[-1]
        raw = orjson.loads(open(os.path.join(compact, name), "rb").read())["sessions"]
        for s in raw:
            rs = s["requests"]
            sessions.append((rs[-1]["end"] - rs[0]["start"]) / 1000)
            first, n_turns = 0, len(recorded)
            for i, r in enumerate(rs):
                if i + 1 == len(rs) or rs[i + 1]["user"]:
                    recorded.append((r["end"] - rs[first]["start"]) / 1000)
                    first = i + 1
                    if i + 1 < len(rs):
                        replies.append(r["gap"] / 1000)
            active.append(sum(recorded[n_turns:]))
        empty += [x for s in build_sessions(raw, PROFILES["l40s"]) for x in s.z]
        # as the gate runs: one deadline per session, users replying after 5 s
        z = [s.z[0] for s in build_sessions(raw, PROFILES["l40s"], reply=reply_time("fixed:5"), unit="session")]
        active_empty += z
        if day in TRAIN_DAYS:
            train_empty += z
    inputs = files

    total = len(sessions)
    write(f"{out}/session_durations.json", [
        {"bucket": n, "order": i, "sessions": c, "share_pct": round(100 * c / total, 2)}
        for i, (n, c) in enumerate(zip(NAMES, [sum(lo <= t < hi for t in sessions) for lo, hi in zip(EDGES, EDGES[1:])]))
    ], inputs)

    assert len(active) == len(sessions) == len(active_empty)
    write(f"{out}/session_active_buckets.json", [
        {"bucket": n, "order": i, "sessions": c, "share_pct": round(100 * c / len(active), 2)}
        for i, (n, c) in enumerate(zip(ACTIVE_NAMES, [sum(lo <= t < hi for t in active) for lo, hi in zip(ACTIVE_EDGES, ACTIVE_EDGES[1:])]))
    ], inputs)
    # the same sessions before sorting: what share of each one's wall clock is user replies
    idle_share = sorted(1 - a / w for a, w in zip(active, sessions) if w > 0)
    recorded.sort(), empty.sort(), sessions.sort(), replies.sort(), active.sort(), active_empty.sort()
    grid_s = [round(10 ** (k / 10), 3) for k in range(0, 59)]   # 1 s to about 7 days
    cdf_s = lambda v, t: 100 * __import__("bisect").bisect_right(v, t) / len(v)
    write(f"{out}/session_active_cdf.json", [
        {"seconds": t, "wall_clock_pct": round(cdf_s(sessions, t), 2), "without_replies_pct": round(cdf_s(active, t), 2),
         "simulated_pct": round(cdf_s(active_empty, t), 2)} for t in grid_s
    ], inputs)
    rows = []
    for what, v in (("session, wall clock", sessions), ("turn, as recorded", recorded), ("turn, empty simulated server", empty),
                    ("user reply, as recorded", replies), ("session without user replies", active),
                    ("session on the empty simulated server, 5 s replies", active_empty),
                    ("share of session that is user replies", idle_share)):
        digits = 3 if what.startswith("share") else 1
        rows.append({"what": what, "count": len(v), **{f"p{int(100 * p)}_s": round(q(v, p), digits) for p in QUANTILES}})
    write(f"{out}/duration_quantiles.json", rows, inputs)

    grid = [round(10 ** (k / 10), 3) for k in range(-3, 59)]      # 0.5 s to about 7 days, log spaced
    cdf = lambda v, t: 100 * (__import__("bisect").bisect_right(v, t)) / len(v)
    write(f"{out}/turn_duration_cdf.json", [
        {"seconds": t, "recorded_pct": round(cdf(recorded, t), 2), "empty_server_pct": round(cdf(empty, t), 2)} for t in grid
    ], inputs)

    marks = {5.0, 60.0, 3600.0}        # the shares the page quotes: under 5 s, 1 min, and the replay's 1 h cap
    reply_grid = sorted(set(grid) | marks)
    write(f"{out}/user_reply_cdf.json", [
        {"seconds": t, "share_pct": round(cdf(replies, t), 2)} for t in reply_grid
    ], inputs)

    if train_empty:   # only when the gate's training weeks are among the inputs
        train_empty.sort()
        write(f"{out}/deadline_budgets.json", [
            {"class": name, "percentile": int(100 * p), "budget_s": round(q(train_empty, p), 1)}
            for name, p in zip(("tight", "medium", "loose"), BUDGET_QUANTILES)
        ], [f for f in files if f[:-5].split("__")[-1] in TRAIN_DAYS])
    print("sessions", total, "turns", len(recorded), len(empty), "replies", len(replies))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(",") if len(sys.argv) > 3 else None)
