"""Session dataset: what is known, and what is still to come, when each user turn of a session starts.

One row per user turn u of every session. The moment is the user's message arriving, with its first request
about to be admitted. Turn 0 is the session's arrival, where a session-level deadline is set.

Features come from `session_view`, a copy of the session that holds nothing else: the earlier turns in
full, and for turn u only what exists at its start (when it began, the user's pause before it, the first
request's prompt size and model). Nothing of turn u's own requests or of later turns can reach a feature
by construction; the audit scrambles them to confirm. Account history uses the user's earlier sessions that
had ended before the turn started.

Labels, counted from the start of turn u:
  cur_calls   requests in this turn
  fut_turns   user turns after this one
  fut_calls   requests in those turns
  fut_work_s  modelled GPU work of those requests (the replay's cost model, as in analysis/dag/dataset.py)
  cur_path_s  the system's own time in this turn: modelled work plus the round times between its requests
  fut_path_s  the same for the turns after this one (user pauses are in neither: they are the user's time)
  rem_path_s  cur_path_s + fut_path_s: the time left in the session from this turn's start (the gated target
              of the time-left predictor, analysis/dag/session_eval_16w.py)
  rem_calls   cur_calls + fut_calls

    python -m analysis.dag.session data/freeinference/dag
    python -m analysis.dag.session data/freeinference/dag --audit 3000
    python -m analysis.dag.session data/freeinference/dag_val --history-from data/freeinference/dag
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import gzip
import json
import os
import random
from collections import defaultdict

from analysis.dag.dataset import NA, day_files, load_sessions, mean, turn_summary

META = ["session", "user", "day", "turn"]
FEATURES = ["harness", "model", "turn_idx", "first_prompt", "idle_before_s", "elapsed_s", "prior_mean_calls",
            "prior_last_calls", "prior_last_wall_s", "prior_mean_work_s", "prior_mean_idle_s", "user_sessions",
            "user_mean_turns", "user_mean_calls", "user_mean_path_s"]
LABELS = ["cur_calls", "fut_turns", "fut_calls", "fut_work_s", "cur_path_s", "fut_path_s", "rem_path_s", "rem_calls"]
COLUMNS = META + FEATURES + LABELS


def turn_path_s(t: dict) -> float:
    """The system's own time in a turn: modelled work plus the round times between its requests (not the user's pause)."""
    return sum(r["_ideal"] + (r["round_ms"] or 0) / 1000 for r in t["rounds"])


def session_view(turns: list[dict], u: int) -> dict:
    """Everything known when turn u of this session starts, and nothing else."""
    t = turns[u]
    first = t["rounds"][0]
    return {"user": t["user"], "session": t["session"], "harness": t["harness"], "day": t["day"], "turn": u,
            "turn_start": t["start"], "session_start": turns[0]["start"], "idle_before_ms": t["idle_before_ms"],
            "first_prompt": first["prompt"], "model": first.get("model"),
            "earlier": [turn_summary(x) for x in turns[:u]],
            "earlier_idle_ms": [x["idle_before_ms"] for x in turns[1:u] if x["idle_before_ms"] is not None]}


class SessionHistory:
    """Per user, the sessions sorted by end time with running sums: 'sessions that ended before T' is a binary search.
    Built from sessions all at once, or a few at a time with `add` and then `finish` (a large folder is streamed)."""

    def __init__(self, sessions: list[list[dict]] | None = None):
        self.rows = defaultdict(list)
        self.ends, self.cum = {}, {}
        if sessions is not None:
            self.add(sessions)
            self.finish()

    def add(self, sessions: list[list[dict]]) -> None:
        for turns in sessions:
            self.rows[turns[0]["user"]].append((turns[-1]["end"], len(turns), sum(len(t["rounds"]) for t in turns),
                                                sum(turn_path_s(t) for t in turns)))

    def finish(self) -> "SessionHistory":
        for user, rows in self.rows.items():
            rows.sort()
            self.ends[user] = [r[0] for r in rows]
            acc, a, b, d = [(0, 0, 0.0)], 0, 0, 0.0
            for _, n, c, p in rows:
                a, b, d = a + n, b + c, d + p
                acc.append((a, b, d))
            self.cum[user] = acc
        self.rows = defaultdict(list)
        return self

    def before(self, user: str, T: int) -> dict:
        n = bisect.bisect_left(self.ends.get(user, []), T)       # sessions that ended strictly before T
        if n == 0:
            return {"n": 0, "turns": NA, "calls": NA, "path": NA}
        a, b, d = self.cum[user][n]
        return {"n": n, "turns": a / n, "calls": b / n, "path": d / n}


def features(v: dict, h: dict) -> dict:
    ear = v["earlier"]
    calls = [e["rounds"] for e in ear]
    idle = v["earlier_idle_ms"]
    return {"harness": v["harness"], "model": v["model"], "turn_idx": v["turn"], "first_prompt": v["first_prompt"],
            "idle_before_s": NA if v["idle_before_ms"] is None else v["idle_before_ms"] / 1000,
            "elapsed_s": (v["turn_start"] - v["session_start"]) / 1000,
            "prior_mean_calls": mean(calls), "prior_last_calls": calls[-1] if calls else NA,
            "prior_last_wall_s": ear[-1]["wall"] if ear else NA, "prior_mean_work_s": mean([e["work"] for e in ear]),
            "prior_mean_idle_s": mean([x / 1000 for x in idle]), "user_sessions": h["n"],
            "user_mean_turns": h["turns"], "user_mean_calls": h["calls"], "user_mean_path_s": h["path"]}


def labels(turns: list[dict], u: int) -> dict:
    later = turns[u + 1:]
    cur_calls, fut_calls = len(turns[u]["rounds"]), sum(len(t["rounds"]) for t in later)
    cur_path, fut_path = turn_path_s(turns[u]), sum(turn_path_s(t) for t in later)
    return {"cur_calls": cur_calls, "fut_turns": len(later), "fut_calls": fut_calls,
            "fut_work_s": sum(turn_summary(t)["work"] for t in later), "cur_path_s": cur_path, "fut_path_s": fut_path,
            "rem_path_s": cur_path + fut_path, "rem_calls": cur_calls + fut_calls}


def rows_of(turns: list[dict], hist: SessionHistory):
    for u, t in enumerate(turns):
        row = {"session": t["session"], "user": t["user"], "day": t["day"], "turn": u}
        row.update(features(session_view(turns, u), hist.before(t["user"], t["start"])))
        row.update(labels(turns, u))
        yield row


def scramble(turns: list[dict], u: int) -> list[dict]:
    """Copy of a session with everything turn u's start cannot know replaced by garbage: later turns gone, and
    turn u's requests (past the fact that the first exists, with its prompt and model) and end time wrecked."""
    turns = copy.deepcopy(turns)[: u + 1]
    t = turns[u]
    bad = {"i": 0, "name": "x", "cat": "GARBAGE", "outcome": "error", "result_tokens": -7, "latency_ms": -7}
    first = {**t["rounds"][0], "end": -1, "ttft": -1, "completion": 99999, "cached": -1, "finish": "garbage",
             "tools": [bad] * 9, "round_ms": -1, "subs": []}
    t["rounds"] = [first] * 5
    t["end"] = -1
    return turns


def audit(sessions: list[list[dict]], hist: SessionHistory, n: int, seed: int = 0) -> int:
    """Rebuild the features of n random turns from a scrambled future: any difference is a leak."""
    rnd = random.Random(seed)
    points = [(si, u) for si, turns in enumerate(sessions) for u in range(len(turns))]
    leaks = 0
    for si, u in rnd.sample(points, min(n, len(points))):
        turns = sessions[si]
        h = hist.before(turns[u]["user"], turns[u]["start"])
        a, b = features(session_view(turns, u), h), features(session_view(scramble(turns, u), u), h)
        bad = [c for c in FEATURES if a[c] != b[c]]
        if bad:
            leaks += 1
            print("LEAK", turns[u]["session"][:8], u, bad[:5])
    print(f"audit: {min(n, len(points)):,} turns rebuilt from a scrambled future, {leaks} with a changed feature")
    return leaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", default="data/freeinference/dag")
    ap.add_argument("--out", default=None, help="gzipped CSV; default <folder>/session_rows.csv.gz")
    ap.add_argument("--audit", type=int, default=0, help="rebuild this many turns from a scrambled future and compare")
    ap.add_argument("--days", default="", help="comma list of days to write rows for (history still counts every day)")
    ap.add_argument("--history-from", default=None,
                    help="folder of earlier days whose sessions count toward account history (rows are written only "
                         "for the main folder)")
    args = ap.parse_args()
    wanted = args.days.split(",") if args.days else None
    files = day_files(args.folder)
    hist = SessionHistory()
    if args.history_from:                       # streamed a day at a time: a large folder does not fit in memory at once
        for f in day_files(args.history_from):
            hist.add(load_sessions(args.history_from, [f]))
    for f in files:
        hist.add(load_sessions(args.folder, [f]))
    hist.finish()
    if args.audit:
        sessions = load_sessions(args.folder, day_files(args.folder, wanted))
        raise SystemExit(1 if audit(sessions, hist, args.audit) else 0)
    out = args.out or os.path.join(args.folder, "session_rows.csv.gz")
    n = sessions_n = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for f in day_files(args.folder, wanted):
            sessions = load_sessions(args.folder, [f])
            sessions_n += len(sessions)
            for turns in sessions:
                for row in rows_of(turns, hist):
                    w.writerow({k: ("" if v is None else v) for k, v in row.items()})
                    n += 1
    with open(out.replace(".csv.gz", "") + ".meta.json", "w") as fh:
        json.dump({"columns": COLUMNS, "features": FEATURES, "labels": LABELS, "rows": n, "sessions": sessions_n}, fh, indent=1)
    print(f"{n:,} rows, {sessions_n:,} sessions -> {out}")


if __name__ == "__main__":
    main()
