"""Online prediction dataset: what is known, and what is still to come, at the two points where a
scheduler acts on a running turn.

    A  request admitted  (a tool returned, or the user spoke): sets the request's priority
    R  response completed: the model has answered, its tools have not returned: retains or evicts the KV

Each request of every turn gives one row per event. Information only grows along the chain
A0 < R0 < A1 < R1 ...:
  A_k knows rounds 0..k-1 in full (tools, outcomes, result sizes, the stall each one took), plus the
      size of request k's prompt.
  R_k adds request k's response: how long it took, how many tokens, why it stopped, and which tools
      it called. Not yet the tools' outcomes or the stall: those arrive with A_{k+1}.
Features come from a view of the turn that holds nothing else (`prefix_view` for A, `response_view` for
R), so they cannot see the future by construction; the audit scrambles the future to confirm. The
account-history feature uses turns that ended strictly before the event.

Labels, counted from the event (`rem_after` is requests still to start after the current one):
  rem_after   n - k - 1            is_last   rem_after == 0
  stall_s     wall time of this request's round: from its response ending to the next request starting
  rem_wall_s  observed time from the event to the turn's end
  rem_work_s  modelled GPU work still to do after the event (A counts the current request, R does not),
              with the subagent work the current and later requests spawn
  rem_path_s  critical path of the parent chain: that work plus the round times between its requests

`to_snapshot` re-expresses a row in the offline predictor's snapshot schema (prediction/predictor): the
same key, ordinal, cutoff, features and feature_available_ms fields, a `current_user_turn` horizon, and
the events `admission` (a turn's first request), `request_admitted`, `root_response_completed`.

    python -m analysis.dag.online data/freeinference/dag
    python -m analysis.dag.online data/freeinference/dag --audit 3000
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import os
import random

from analysis.dag.dataset import (CATS, FEATURES, NA, PREV, UserHistory, annotate_session, day_files, features,
                                  load_sessions, prefix_view)

EVENTS = ("A", "R")
EVENT_NAMES = {"R": "root_response_completed"}       # A is `admission` at a turn's first request, else `request_admitted`
RESP = ["resp_ttft_s", "resp_gen_s"]                 # the response's own timing: known at R, not at A
LABELS = ["rem_after", "is_last", "stall_s", "rem_wall_s", "rem_work_s", "rem_path_s"]
META = ["session", "user", "day", "turn", "step", "event", "cutoff_ms", "turn_start_ms", "w_session", "w_turn"]
COLUMNS = META + ["model"] + FEATURES + RESP + LABELS
STATIC = {"harness", "model", "turn_idx", "step", "idle_before_s", "sess_prior_mean_rounds", "sess_prior_mean_work_s",
          "sess_prior_mean_wall_s", "user_prior_turns", "user_prior_mean_rounds", "user_prior_mean_work_s",
          "user_prior_frac_fanout"}                  # known when the turn is admitted, whatever the event


def response_view(turns: list[dict], ti: int, k: int) -> dict:
    """Everything a scheduler knows when request k's response has just completed, and nothing else."""
    v = prefix_view(turns, ti, k)
    cur = turns[ti]["rounds"][k]
    v["done"].append({"completion": cur["completion"], "finish": cur["finish"], "round_ms": None,
                      "ideal": cur["_ideal"], "n_subs": 0, "open": True,
                      "tools": [{"cat": x["cat"], "outcome": None, "result_tokens": None} for x in cur["tools"]]})
    v["cutoff"] = cur["end"]
    return v


def view(turns: list[dict], ti: int, k: int, event: str) -> dict:
    return prefix_view(turns, ti, k) if event == "A" else response_view(turns, ti, k)


def cutoff_of(turns: list[dict], ti: int, k: int, event: str) -> int:
    r = turns[ti]["rounds"][k]
    return r["start"] if event == "A" else r["end"]


def labels_at(t: dict, k: int, event: str) -> dict:
    rs, n = t["rounds"], len(t["rounds"])
    r = rs[k]
    first = k if event == "A" else k + 1                      # first request whose own work is still to come
    work = sum(x["_ideal"] for x in rs[first:]) + sum(x["_sub_work"] for x in rs[k:])
    gaps = sum((x["round_ms"] or 0) for x in rs[k:n - 1]) / 1000
    now = r["start"] if event == "A" else r["end"]
    return {"rem_after": n - k - 1, "is_last": n - k - 1 == 0,
            "stall_s": NA if r["round_ms"] is None else r["round_ms"] / 1000,
            "rem_wall_s": (t["end"] - now) / 1000, "rem_work_s": work,
            "rem_path_s": sum(x["_ideal"] for x in rs[first:]) + gaps}


def rows_of(turns: list[dict], hist: UserHistory):
    total = sum(len(t["rounds"]) for t in turns)
    for ti, t in enumerate(turns):
        for k, cur in enumerate(t["rounds"]):
            for event in EVENTS:
                v = view(turns, ti, k, event)
                cutoff = cutoff_of(turns, ti, k, event)
                row = {"session": t["session"], "user": t["user"], "day": t["day"], "turn": ti, "step": k,
                       "event": event, "cutoff_ms": cutoff, "turn_start_ms": t["start"],
                       "w_session": 1 / total, "w_turn": 1 / len(t["rounds"]), "model": cur.get("model")}
                row.update(features(v, hist.before(t["user"], cutoff)))
                row.update({"resp_ttft_s": NA, "resp_gen_s": NA} if event == "A" else
                           {"resp_ttft_s": NA if cur["ttft"] is None else cur["ttft"] / 1000,
                            "resp_gen_s": (cur["end"] - cur["start"]) / 1000})
                row.update(labels_at(t, k, event))
                yield row


def to_snapshot(row: dict) -> dict:
    """A row in the offline predictor's snapshot schema. Static fields are available from the turn's
    admission, everything that moves with the turn from the event's cutoff."""
    event = EVENT_NAMES.get(row["event"]) or ("admission" if row["step"] == 0 else "request_admitted")
    feats = {c: row[c] for c in ["model"] + FEATURES + RESP}
    return {"key": f"{row['session']}/turn{row['turn']}", "ordinal": 2 * row["step"] + (row["event"] == "R"),
            "horizon": "current_user_turn", "event": event, "cutoff_ms": row["cutoff_ms"], "features": feats,
            "feature_available_ms": {c: (row["turn_start_ms"] if c in STATIC else row["cutoff_ms"]) for c in feats}}


def scramble(turns: list[dict], ti: int, k: int, event: str) -> list[dict]:
    """Copy of a session with everything the event at request (ti, k) cannot know replaced by garbage."""
    turns = copy.deepcopy(turns)[: ti + 1]
    t = turns[ti]
    cur = t["rounds"][k]
    bad = {"i": 0, "name": "x", "cat": "GARBAGE", "outcome": "error", "result_tokens": -7, "latency_ms": -7}
    if event == "A":
        new = {**cur, "end": -1, "ttft": -1, "completion": 99999, "cached": -1, "finish": "garbage",
               "tools": [bad] * 9, "round_ms": -1, "subs": [], "trigger": "garbage"}
    else:           # the response is known (tokens, finish, tool names and categories); what the tools did is not
        new = {**cur, "tools": [{**x, "outcome": "error", "result_tokens": -7, "latency_ms": -7} for x in cur["tools"]],
               "round_ms": -1, "subs": []}
    t["rounds"] = t["rounds"][:k] + [new]
    t["end"] = -1
    annotate_session(turns)
    return turns


def audit(sessions: list[list[dict]], hist: UserHistory, n: int, seed: int = 0) -> int:
    """Rebuild the features of n random events from a scrambled future: any difference is a leak."""
    rnd = random.Random(seed)
    points = [(si, ti, k, e) for si, turns in enumerate(sessions) for ti, t in enumerate(turns)
              for k in range(len(t["rounds"])) for e in EVENTS]
    leaks = 0
    for si, ti, k, e in rnd.sample(points, min(n, len(points))):
        turns = sessions[si]
        user, cut = turns[ti]["user"], cutoff_of(turns, ti, k, e)
        h = hist.before(user, cut)
        a, b = features(view(turns, ti, k, e), h), features(view(scramble(turns, ti, k, e), ti, k, e), h)
        bad = [c for c in FEATURES if a[c] != b[c]]
        if bad:
            leaks += 1
            print("LEAK", turns[ti]["session"][:8], ti, k, e, bad[:5])
    print(f"audit: {min(n, len(points)):,} events rebuilt from a scrambled future, {leaks} with a changed feature")
    return leaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", nargs="?", default="data/freeinference/dag")
    ap.add_argument("--out", default=None, help="gzipped CSV; default <folder>/online_rows.csv.gz")
    ap.add_argument("--audit", type=int, default=0, help="rebuild this many events from a scrambled future and compare")
    ap.add_argument("--days", default="", help="comma list of days to write rows for (history still counts every day)")
    ap.add_argument("--history-from", default=None,
                    help="folder of earlier days whose turns count toward account history (rows are written only "
                         "for the main folder): a later week's users arrive with the history they already had")
    args = ap.parse_args()
    wanted = args.days.split(",") if args.days else None
    hist = UserHistory()
    for folder in ([args.history_from] if args.history_from else []) + [args.folder]:
        for f in day_files(folder):             # streamed a day at a time: a large folder does not fit in memory at once
            hist.add(load_sessions(folder, [f]))
    hist.finish()
    if args.audit:
        sessions = load_sessions(args.folder, day_files(args.folder, wanted))
        raise SystemExit(1 if audit(sessions, hist, args.audit) else 0)
    out = args.out or os.path.join(args.folder, "online_rows.csv.gz")
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
        json.dump({"columns": COLUMNS, "features": FEATURES, "resp": RESP, "labels": LABELS, "meta": META,
                   "events": EVENTS, "cats": CATS, "prev": PREV, "rows": n, "sessions": sessions_n}, fh, indent=1)
    print(f"{n:,} rows, {sessions_n:,} sessions -> {out}")


if __name__ == "__main__":
    main()
