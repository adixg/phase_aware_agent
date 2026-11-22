"""Prefix-level dataset: one row per LLM request of a turn, built from what is known when it starts.

A row answers: standing at the start of request k of a turn, what do we know (features) and what
happens next (labels)? Features come only from a `prefix_view`, a restricted copy of the turn that
holds the completed rounds 0..k-1, the current request's start time, its prompt size, and the
earlier turns of the session. Nothing of request k's own result (completion, tools, end time) or of
later rounds can reach a feature: they are not in the view. The audit scrambles them to confirm.

Account history uses the user's turns that had already ended when the request started.

GPU work is the replay's cost model (the measured L40S profile) on the tokens a request has to
prefill, assuming its session's earlier context is resident: `ideal`, as in analysis/replay/sim.py.
The trace carries no GPU times, so work in seconds is modelled; `rem_wall_s` is observed.

Labels at request k of a turn with n requests (all from k to the end of the turn):
  rem_rounds      n - k, the requests left including this one (1 = the last)
  rem_work_s      GPU work of those requests and of the subagent requests they spawn
  rem_parent_work_s   the same without subagents
  rem_path_s      critical path of the parent chain: their work plus the round times between them
                  (subagents run beside the chain and are left out)
  rem_wall_s      observed time from this request's start to the turn's end
  next_stall_s    wall time of this request's round (its tools), None for the last request
  next_width, next_cat   tool calls this request makes and the first one's category

    python -m analysis.dag.dataset data/freeinference/dag --out data/freeinference/dag/prefix_rows.csv.gz
    python -m analysis.dag.dataset data/freeinference/dag --audit 2000
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import glob
import gzip
import json
import os
import random
from collections import defaultdict

import orjson

from analysis.replay.run_gate import PROFILES

PROFILE = PROFILES["l40s"]
CATS = ["Other", "*Search", "*Interpreter", "Read", "Edit", "*File Ops", "Write", "*Network", "*Wait/Poll",
        "*Git/VCS", "*Build/Test", "Browser", "*Packages", "MCP", "Ask User", "Subagent"]
PREV = 3          # how many completed rounds back the per-round features look
NA = None


def service_s(new: int, completion: int) -> float:
    return (PROFILE.overhead_ms + PROFILE.prefill_ms * new + PROFILE.decode_ms * completion) / 1000


def annotate(rounds: list[dict], prev_ctx: int | None = None) -> int | None:
    """Add `_new` (tokens to prefill) and `_ideal` (service time) to every round of a chain, and
    `_sub_work` (work of the subagents it spawns). Returns the context size the next request inherits."""
    for r in rounds:
        r["_new"] = r["prompt"] if prev_ctx is None else max(0, r["prompt"] - prev_ctx)
        r["_ideal"] = service_s(r["_new"], r["completion"] or 0)
        r["_sub_work"] = 0.0
        for sub in r["subs"]:
            annotate(sub["rounds"])
            r["_sub_work"] += sum(c["_ideal"] + c["_sub_work"] for c in sub["rounds"])
        prev_ctx = r["prompt"] + (r["completion"] or 0)
    return prev_ctx


def annotate_session(turns: list[dict]) -> None:
    prev = None
    for t in turns:
        prev = annotate(t["rounds"], prev)


def turn_summary(t: dict) -> dict:
    return {"rounds": len(t["rounds"]), "work": sum(r["_ideal"] for r in t["rounds"]),
            "wall": (t["end"] - t["start"]) / 1000, "end": t["end"],
            "fan": any(len(r["tools"]) > 1 for r in t["rounds"])}


def prefix_view(turns: list[dict], ti: int, k: int) -> dict:
    """Everything a scheduler knows when request k of turn ti starts, and nothing else."""
    t = turns[ti]
    cur = t["rounds"][k]
    done = [{"completion": r["completion"], "finish": r["finish"], "round_ms": r["round_ms"],
             "ideal": r["_ideal"], "n_subs": len(r["subs"]),
             "tools": [{"cat": x["cat"], "outcome": x["outcome"], "result_tokens": x["result_tokens"]}
                       for x in r["tools"]]} for r in t["rounds"][:k]]
    return {"user": t["user"], "session": t["session"], "harness": t["harness"], "day": t["day"],
            "turn": ti, "step": k, "turn_start": t["start"], "idle_before_ms": t["idle_before_ms"],
            "cur_start": cur["start"], "cur_prompt": cur["prompt"], "cur_new": cur["_new"],
            "done": done, "earlier": [turn_summary(u) for u in turns[:ti]]}


class UserHistory:
    """Per user, the turns sorted by end time with running sums, so 'before time T' is a binary search."""

    def __init__(self, sessions: list[list[dict]]):
        by_user = defaultdict(list)
        for turns in sessions:
            for t in turns:
                s = turn_summary(t)
                by_user[t["user"]].append((s["end"], s["rounds"], s["work"], s["fan"]))
        self.ends, self.cum = {}, {}
        for u, rows in by_user.items():
            rows.sort()
            self.ends[u] = [r[0] for r in rows]
            acc, a, b, c = [(0, 0.0, 0)], 0, 0.0, 0
            for _, n, w, f in rows:
                a, b, c = a + n, b + w, c + f
                acc.append((a, b, c))
            self.cum[u] = acc

    def before(self, user: str, T: int) -> dict:
        n = bisect.bisect_left(self.ends.get(user, []), T)       # turns that ended strictly before T
        if n == 0:
            return {"n": 0, "rounds": NA, "work": NA, "fan": NA}
        a, b, c = self.cum[user][n]
        return {"n": n, "rounds": a / n, "work": b / n, "fan": c / n}


def mean(v: list[float]):
    return sum(v) / len(v) if v else NA


FEATURES = (["harness", "turn_idx", "step", "prompt", "new_tokens", "idle_before_s", "active_s", "attained_work_s",
             "prefix_completion", "prefix_result_tokens", "prefix_errors", "prefix_fanout_rounds", "prefix_max_width",
             "prefix_subs", "prefix_stall_s"]
            + [f"cat_{c}" for c in CATS]
            + [f"prev{i}_{f}" for i in range(1, PREV + 1) for f in ("width", "cat", "errors", "result_tokens", "round_s", "completion", "finish")]
            + ["sess_prior_mean_rounds", "sess_prior_mean_work_s", "sess_prior_mean_wall_s",
               "user_prior_turns", "user_prior_mean_rounds", "user_prior_mean_work_s", "user_prior_frac_fanout"])
LABELS = ["rem_rounds", "rem_work_s", "rem_parent_work_s", "rem_path_s", "rem_wall_s", "next_stall_s", "next_width", "next_cat"]
META = ["session", "user", "day", "turn", "step", "w_session", "w_turn"]
COLUMNS = META + FEATURES + LABELS


def features(v: dict, hist: dict) -> dict:
    """Features of one decision point, from a prefix_view and the user history alone."""
    done = v["done"]
    now = v.get("cutoff", v["cur_start"])          # a response-completed view (online.py) decides later than the start
    f = {"harness": v["harness"], "turn_idx": v["turn"], "step": v["step"], "prompt": v["cur_prompt"],
         "new_tokens": v["cur_new"],
         "idle_before_s": NA if v["idle_before_ms"] is None else v["idle_before_ms"] / 1000,
         "active_s": (now - v["turn_start"]) / 1000,
         "attained_work_s": sum(r["ideal"] for r in done),
         "prefix_completion": sum(r["completion"] or 0 for r in done),
         "prefix_result_tokens": sum(x["result_tokens"] or 0 for r in done for x in r["tools"]),
         "prefix_errors": sum(x["outcome"] not in ("ok", None) for r in done for x in r["tools"]),
         "prefix_fanout_rounds": sum(len(r["tools"]) > 1 for r in done),
         "prefix_max_width": max((len(r["tools"]) for r in done), default=0),
         "prefix_subs": sum(r["n_subs"] for r in done),
         "prefix_stall_s": sum((r["round_ms"] or 0) for r in done) / 1000}
    counts = {c: 0 for c in CATS}
    for r in done:
        for x in r["tools"]:
            counts[x["cat"] if x["cat"] in counts else "Other"] += 1
    f.update({f"cat_{c}": n for c, n in counts.items()})
    for i in range(1, PREV + 1):
        r = done[-i] if len(done) >= i else None
        open_ = r is not None and r.get("open")      # emitted but its tools have not returned: outcomes unknown
        f.update({f"prev{i}_width": NA if r is None else len(r["tools"]),
                  f"prev{i}_cat": NA if r is None else (r["tools"][0]["cat"] if r["tools"] else "__none__"),
                  f"prev{i}_errors": NA if r is None or open_ else sum(x["outcome"] not in ("ok", None) for x in r["tools"]),
                  f"prev{i}_result_tokens": NA if r is None or open_ else sum(x["result_tokens"] or 0 for x in r["tools"]),
                  f"prev{i}_round_s": NA if r is None or r["round_ms"] is None else r["round_ms"] / 1000,
                  f"prev{i}_completion": NA if r is None else r["completion"],
                  f"prev{i}_finish": NA if r is None else r["finish"]})
    e = v["earlier"]
    f.update({"sess_prior_mean_rounds": mean([u["rounds"] for u in e]),
              "sess_prior_mean_work_s": mean([u["work"] for u in e]),
              "sess_prior_mean_wall_s": mean([u["wall"] for u in e]),
              "user_prior_turns": hist["n"], "user_prior_mean_rounds": hist["rounds"],
              "user_prior_mean_work_s": hist["work"], "user_prior_frac_fanout": hist["fan"]})
    return f


def labels(t: dict) -> list[dict]:
    """Labels at every request of a turn, from the whole turn (this is the future the features must not see)."""
    rs, n = t["rounds"], len(t["rounds"])
    out, work, sub, gap = [], 0.0, 0.0, 0.0
    suffix = []
    for k in range(n - 1, -1, -1):          # suffix sums from the last request back
        work += rs[k]["_ideal"]
        sub += rs[k]["_sub_work"]
        if k < n - 1:
            gap += (rs[k]["round_ms"] or 0) / 1000
        suffix.append((work, sub, gap))
    suffix.reverse()
    for k, r in enumerate(rs):
        w, s, g = suffix[k]
        out.append({"rem_rounds": n - k, "rem_work_s": w + s, "rem_parent_work_s": w, "rem_path_s": w + g,
                    "rem_wall_s": (t["end"] - r["start"]) / 1000,
                    "next_stall_s": NA if r["round_ms"] is None else r["round_ms"] / 1000,
                    "next_width": len(r["tools"]), "next_cat": r["tools"][0]["cat"] if r["tools"] else "__none__"})
    return out


def load_sessions(folder: str) -> list[list[dict]]:
    """Turns grouped by session, in time order within each (build.py writes a session's turns together)."""
    sessions, cur, sid = [], [], None
    for path in sorted(glob.glob(os.path.join(folder, "2026-*.jsonl"))):
        for line in open(path, "rb"):
            t = orjson.loads(line)
            if t["session"] != sid and cur:
                sessions.append(cur)
                cur = []
            sid = t["session"]
            cur.append(t)
    if cur:
        sessions.append(cur)
    for turns in sessions:
        annotate_session(turns)
    return sessions


def rows_of(turns: list[dict], hist: UserHistory):
    total = sum(len(t["rounds"]) for t in turns)
    for ti, t in enumerate(turns):
        lab = labels(t)
        for k in range(len(t["rounds"])):
            v = prefix_view(turns, ti, k)
            row = {"session": t["session"], "user": t["user"], "day": t["day"], "turn": ti, "step": k,
                   "w_session": 1 / total, "w_turn": 1 / len(t["rounds"])}
            row.update(features(v, hist.before(t["user"], v["cur_start"])))
            row.update(lab[k])
            yield row


def scramble(turns: list[dict], ti: int, k: int) -> list[dict]:
    """Copy of a session with everything the request at (ti, k) cannot know replaced by garbage."""
    turns = copy.deepcopy(turns)[: ti + 1]
    t = turns[ti]
    cur = t["rounds"][k]
    bad = {"i": 0, "name": "x", "cat": "GARBAGE", "outcome": "error", "result_tokens": -7, "latency_ms": -7}
    t["rounds"] = t["rounds"][:k] + [{**cur, "end": -1, "ttft": -1, "completion": 99999, "cached": -1, "finish": "garbage",
                                      "tools": [bad] * 9, "round_ms": -1, "subs": [], "trigger": "garbage"}]
    t["end"] = -1
    annotate_session(turns)
    return turns


def audit(sessions: list[list[dict]], hist: UserHistory, n: int, seed: int = 0) -> int:
    """Rebuild the features of n random rows from a scrambled future: any difference is a leak."""
    rnd = random.Random(seed)
    points = [(si, ti, k) for si, turns in enumerate(sessions) for ti, t in enumerate(turns) for k in range(len(t["rounds"]))]
    leaks = 0
    for si, ti, k in rnd.sample(points, min(n, len(points))):
        turns = sessions[si]
        real = prefix_view(turns, ti, k)
        cut = scramble(turns, ti, k)
        fake = prefix_view(cut, ti, k)
        a, b = features(real, hist.before(real["user"], real["cur_start"])), features(fake, hist.before(real["user"], real["cur_start"]))
        bad = [c for c in FEATURES if a[c] != b[c]]
        if bad:
            leaks += 1
            print("LEAK", turns[ti]["session"][:8], ti, k, bad[:5])
    print(f"audit: {min(n, len(points)):,} rows rebuilt from a scrambled future, {leaks} with a changed feature")
    return leaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", default="data/freeinference/dag")
    ap.add_argument("--out", default=None, help="write the rows here (gzipped CSV); default <folder>/prefix_rows.csv.gz")
    ap.add_argument("--audit", type=int, default=0, help="rebuild this many rows from a scrambled future and compare")
    args = ap.parse_args()
    sessions = load_sessions(args.folder)
    hist = UserHistory(sessions)
    if args.audit:
        raise SystemExit(1 if audit(sessions, hist, args.audit) else 0)
    out = args.out or os.path.join(args.folder, "prefix_rows.csv.gz")
    n = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for turns in sessions:
            for row in rows_of(turns, hist):
                w.writerow({k: ("" if v is None else v) for k, v in row.items()})
                n += 1
    with open(out.replace(".csv.gz", "") + ".meta.json", "w") as fh:
        json.dump({"columns": COLUMNS, "features": FEATURES, "labels": LABELS, "meta": META, "cats": CATS, "prev": PREV,
                   "profile": PROFILE.__dict__, "rows": n, "sessions": len(sessions)}, fh, indent=1)
    print(f"{n:,} rows, {len(sessions):,} sessions -> {out}")


if __name__ == "__main__":
    main()
