"""Predicted against actual, one row per scored decision, for the three predictors.

    python -m analysis.dag.predictions_vs_actual                      # the tuning week (Aug 9-15)
    python -m analysis.dag.predictions_vs_actual --week validation    # the validation week (Aug 16-22), see below
    python -m analysis.dag.predictions_vs_actual --all-steps          # every request or turn, not only the gated ones

Each predictor is refitted with the settings it froze (frozen.json under data/freeinference/tools16w, duration16w and
session16w) and scored on the chosen week; the baseline it must beat is refitted the same way. For the tuning week the
models are fitted on the fit weeks (May 17 - Aug 8); for the validation week, on the fit and tuning weeks, exactly as
the validation did. Files are written to data/freeinference/predictions/<week>/:

    next_tool.csv   one row per request, the forecast over 17 classes and what the response called
    tool_duration.csv   one row per round, the forecast distribution of its time and how long it took
    time_left.csv   one row per user turn, the forecast distribution of the seconds left and what was needed
    README.md       every column

The validation week has already been read once for each of these models. These files add no new decision to it, but
nothing should be tuned on them. With --week validation the script also checks that its per-step averages equal the
numbers reported (validation_done.json) and says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from bisect import bisect_left, bisect_right

import numpy as np

from analysis.dag import duration_eval_16w as D
from analysis.dag import duration_gates_16w as G2
from analysis.dag import session_eval_16w as S
from analysis.dag import session_gates_16w as G3
from analysis.dag import splits, tool_eval
from analysis.dag import tool_gates_16w as G1
from analysis.dag.session_eval import load as load_sessions
from prediction.predictor.duration_baseline import DurationForecaster
from prediction.predictor.session_baseline import SessionTimeForecaster
from prediction.predictor.tool_baseline import CLASSES, CODE, END, ToolForecaster

ONLINE = "data/freeinference/dag_all/online_rows.csv.gz"
SESSIONS = "data/freeinference/dag_all/session_rows.csv.gz"
OUT = "data/freeinference/predictions"


def label(c: str) -> str:
    """A class as people read it: the trace's asterisks dropped, END called turn_ends."""
    return "turn_ends" if c == END else c.lstrip("*")


def column(c: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", label(c)).strip("_")


def dist_stats(m, y) -> dict:
    """A forecast distribution (a Marginal) against the actual value y."""
    total = m.mass[-1]
    return {"mean": m.weighted_sum[-1] / total, "p10": m.quantile(0.1), "p50": m.quantile(0.5), "p90": m.quantile(0.9),
            "cdf_below_actual": m.mass[bisect_left(m.values, y)] / total, "cdf_at_actual": m.mass[bisect_right(m.values, y)] / total,
            "crps": m.crps(y)}


def frozen(folder: str) -> dict:
    return json.load(open(os.path.join("data/freeinference", folder, "frozen.json")))


def windows(week: str) -> tuple[tuple, tuple]:
    if week == "tune":
        return splits.FIT_DAYS, splits.TUNE_DAYS
    return splits.FIT_DAYS + splits.TUNE_DAYS, splits.VALIDATION_DAYS


def write(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path}: {len(rows):,} rows, {len(header)} columns")


def next_tool(out: str, week: str, all_steps: bool) -> dict:
    train_days, test_days = windows(week)
    c = frozen("tools16w")
    rows = tool_eval.load(ONLINE, train_days + test_days)
    train, test = [r for r in rows if r["day"] in set(train_days)], [r for r in rows if r["day"] in set(test_days)]
    del rows
    if not all_steps:
        test = [r for r in test if r["step"] in G1.GATE_STEPS]
    base = ToolForecaster(train, tool_eval.BASELINE, c["baseline_strength"])
    model = ToolForecaster(train, tuple(c["levels"]), c["strength"])
    header = ["session", "day", "turn", "step", "harness", "prev1_tool", "prev2_tool", "prev3_tool", "actual",
              "model_top1", "model_top1_p", "model_top2", "model_top2_p", "model_top3", "model_top3_p", "model_p_actual",
              "model_brier", "model_top1_correct", "base_top1", "base_top1_p", "base_p_actual", "base_brier", "base_top1_correct"]
    header += [f"model_p_{column(k)}" for k in CLASSES] + [f"base_p_{column(k)}" for k in CLASSES]
    body, per_step = [], {}
    for r in test:
        pm, pb, y = model.proba(r), base.proba(r), CODE[r["next_cat"]]
        onehot = np.zeros(len(CLASSES))
        onehot[y] = 1
        bm, bb = float(((pm - onehot) ** 2).sum()), float(((pb - onehot) ** 2).sum())
        top = np.argsort(-pm)[:3]
        t1 = int(pb.argmax())
        body.append([r["session"], r["day"], r["turn"], r["step"], r["harness"], *[label(r[k]) if r[k] else "" for k in ("prev1_cat", "prev2_cat", "prev3_cat")],
                     label(r["next_cat"]), *[v for i in top for v in (label(CLASSES[i]), round(float(pm[i]), 5))], round(float(pm[y]), 5),
                     round(bm, 5), int(top[0] == y), label(CLASSES[t1]), round(float(pb[t1]), 5), round(float(pb[y]), 5), round(bb, 5), int(t1 == y),
                     *[round(float(p), 5) for p in pm], *[round(float(p), 5) for p in pb]])
        per_step.setdefault(r["step"], []).append((bm, bb))
    write(os.path.join(out, "next_tool.csv"), header, body)
    return {t: {"model": float(np.mean([a for a, _ in v])), "base": float(np.mean([b for _, b in v])), "n": len(v)} for t, v in per_step.items()}


def tool_duration(out: str, week: str, all_steps: bool) -> dict:
    train_days, test_days = windows(week)
    c = frozen("duration16w")
    rows = D.load(ONLINE, train_days + test_days)
    train, test = [r for r in rows if r["day"] in set(train_days)], [r for r in rows if r["day"] in set(test_days)]
    del rows
    if not all_steps:
        test = [r for r in test if r["step"] in G2.GATE_STEPS]
    base = DurationForecaster(train, D.BASELINE, c["baseline_strength"])
    model = DurationForecaster(train, tuple(c["levels"]), c["strength"])
    names = ["mean_s", "p10_s", "p50_s", "p90_s", "cdf_below_actual", "cdf_at_actual", "crps_s"]
    header = ["session", "day", "step", "tool", "previous_tool", "tools_in_round", "previous_round_s", "same_tool_last_s", "harness", "actual_s"]
    header += [f"model_{n}" for n in names] + [f"base_{n}" for n in names]
    body, per_step = [], {}

    def cells(m, y):
        s = dist_stats(m, y)
        return [round(s[k] / 1000, 4) if k in ("mean", "p10", "p50", "p90", "crps") else round(s[k], 5) for k in ("mean", "p10", "p50", "p90", "cdf_below_actual", "cdf_at_actual", "crps")], s["crps"] / 1000

    for r in test:
        y = r["dur_ms"]
        cm, sm = cells(model.marginal(r), y)
        cb, sb = cells(base.marginal(r), y)
        body.append([r["session"], r["day"], r["step"], label(r["cat"]), label(r["cat2"]) if r["cat2"] else "", int(r["width"]) if r["width"] else "",
                     "" if r["prevdur"] is None else round(r["prevdur"], 3), "" if r["samecat"] is None else round(r["samecat"], 3),
                     r["harness"], round(y / 1000, 3), *cm, *cb])
        per_step.setdefault(r["step"], []).append((sm, sb))
    write(os.path.join(out, "tool_duration.csv"), header, body)
    return {t: {"model": float(np.mean([a for a, _ in v])), "base": float(np.mean([b for _, b in v])), "n": len(v)} for t, v in per_step.items()}


def time_left(out: str, week: str, all_steps: bool) -> dict:
    train_days, test_days = windows(week)
    c = frozen("session16w")
    rows = load_sessions(SESSIONS, train_days + test_days)
    train, test = [r for r in rows if r["day"] in set(train_days)], [r for r in rows if r["day"] in set(test_days)]
    del rows
    if not all_steps:
        test = [r for r in test if r["turn_idx"] in G3.GATE_TURNS]
    base = SessionTimeForecaster(train, (), c["baseline_strength"])
    model = SessionTimeForecaster(train, tuple(c["levels"]), c["strength"])
    names = ["mean", "p10", "p50", "p90", "cdf_below_actual", "cdf_at_actual", "crps"]
    header = ["session", "day", "turn", "harness", "user_mean_session_s", "user_mean_turns", "user_sessions", "previous_turn_requests", "first_prompt_tokens"]
    for target, unit in (("rem_path_s", "s"), ("rem_calls", "requests")):
        header += [f"{target}_actual"] + [f"{target}_model_{n}" for n in names] + [f"{target}_base_{n}" for n in names]
    body, per_turn = [], {}
    for r in test:
        line = [r["session"], r["day"], r["turn_idx"], r["harness"], "" if r["user_mean_path_s"] is None else round(r["user_mean_path_s"], 1),
                "" if r["user_mean_turns"] is None else round(r["user_mean_turns"], 2), int(r["user_sessions"] or 0),
                "" if r["prior_last_calls"] is None else int(r["prior_last_calls"]), int(r["first_prompt"] or 0)]
        for target in ("rem_path_s", "rem_calls"):
            y = model.value(r, target)
            sm, sb = dist_stats(model.marginal(r, target), y), dist_stats(base.marginal(r, target), y)
            line += [y] + [round(sm[k], 5) if "cdf" in k else round(sm[k], 3) for k in ("mean", "p10", "p50", "p90", "cdf_below_actual", "cdf_at_actual", "crps")]
            line += [round(sb[k], 5) if "cdf" in k else round(sb[k], 3) for k in ("mean", "p10", "p50", "p90", "cdf_below_actual", "cdf_at_actual", "crps")]
            if target == "rem_path_s":
                per_turn.setdefault(r["turn_idx"], []).append((sm["crps"], sb["crps"]))
        body.append(line)
    write(os.path.join(out, "time_left.csv"), header, body)
    return {t: {"model": float(np.mean([a for a, _ in v])), "base": float(np.mean([b for _, b in v])), "n": len(v)} for t, v in per_turn.items()}


README = """# Predicted against actual

One row per scored decision for each of the three predictors. Produced by `python -m analysis.dag.predictions_vs_actual`
(week: {week}, test days {first} to {last}). The models are fitted on {train} with the settings each froze; the baseline each
must beat is fitted the same way. {note}

Rows are the decision steps the pass rule scores ({steps}). Lower is better for Brier and CRPS.

## next_tool.csv: one row per request
Question: which tool will this response call? 17 classes: the 16 tool categories plus `turn_ends` (no tool, the turn ends).
* `session, day, turn, step, harness`: which request. `prev1_tool, prev2_tool, prev3_tool`: the previous rounds' first tools.
* `actual`: the first tool the response called, or `turn_ends`.
* `model_top1/2/3` and `model_top1_p/2_p/3_p`: the model's three most likely classes and their probabilities.
* `model_p_actual`: the probability the model gave to what happened. `model_brier`: sum over the 17 classes of the squared gap to the outcome (a uniform guess scores 0.941).
* `model_top1_correct`: 1 if its top class was the actual one.
* `base_*`: the same for the baseline (step, then the previous tool).
* `model_p_<class>` and `base_p_<class>`: the full probability for each of the 17 classes, so any other metric can be computed.

## tool_duration.csv: one row per round (a response that called a tool)
Question: how long will the tools of this response take? The round's time, from the response ending to the next request
starting, capped at 3,600 s. A round with several tools has one time.
* `tool, previous_tool, tools_in_round, previous_round_s, same_tool_last_s, harness`: what the forecast knew.
* `actual_s`: how long the round took.
* `model_mean_s, model_p10_s, model_p50_s, model_p90_s`: the forecast distribution (mean and 10th, 50th, 90th percentiles).
* `model_cdf_below_actual, model_cdf_at_actual`: the share of the forecast distribution strictly below, and at or below, the actual value. If the forecasts are calibrated, a value drawn uniformly between the two is uniform on 0 to 1 across rows.
* `model_crps_s`: the CRPS of this one forecast, in seconds. `base_*`: the same for the baseline (the tool category alone).

## time_left.csv: one row per user turn, at its start
Question: how much of its own time will the session still need (modelled GPU work plus the waits between requests, the
current turn included, user pauses excluded)? Turn 0 is the session's arrival.
* `session, day, turn, harness, user_mean_session_s, user_mean_turns, user_sessions, previous_turn_requests, first_prompt_tokens`: what the forecast knew.
* `rem_path_s_actual`: the seconds actually needed. `rem_path_s_model_*` and `rem_path_s_base_*`: mean, p10, p50, p90, the two cdf columns and the CRPS, in seconds. The baseline is the turn-index-only table.
* `rem_calls_*`: the same for the requests left (a secondary target).

## Reading them
* The model beats its baseline when its score column is lower on average, not on every row.
* The percentile columns let you check calibration: about 10% of actuals should fall below `p10`, 50% below `p50`, 90% below `p90`.
* The trace has no prompt text, so these forecasts use timing, token counts, tool categories, the harness and history only.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--week", choices=["tune", "validation"], default="tune")
    ap.add_argument("--all-steps", action="store_true", help="every request or turn, not only the gated decision steps")
    ap.add_argument("--only", choices=["next_tool", "tool_duration", "time_left"], default=None)
    args = ap.parse_args()
    train_days, test_days = windows(args.week)
    out = os.path.join(OUT, args.week)
    os.makedirs(out, exist_ok=True)
    if args.week == "validation":
        print("WARNING: this is the validation week. Each model has already been scored on it once; do not tune on these rows.")
    print(f"week {args.week}: test {test_days[0]} to {test_days[-1]}, models fitted on {train_days[0]} to {train_days[-1]}")
    runs = {"next_tool": next_tool, "tool_duration": tool_duration, "time_left": time_left}
    got = {name: fn(out, args.week, args.all_steps) for name, fn in runs.items() if args.only in (None, name)}
    steps = "steps 3, 5, 10, 20 / steps 1, 2, 5, 10, 20 / turns 0, 1, 2, 5" if not args.all_steps else "every step and turn"
    note = ("This is the validation week: each model has been scored on it once already (see validation_done.json); please do not tune on it."
            if args.week == "validation" else "This is the tuning week, the one the settings were chosen on, so the model's scores here are slightly flattering.")
    open(os.path.join(out, "README.md"), "w").write(README.format(
        week=args.week, first=test_days[0], last=test_days[-1], train=f"{train_days[0]} to {train_days[-1]}", note=note, steps=steps))
    if args.week == "validation" and not args.all_steps:
        for name, folder, key in (("next_tool", "tools16w", "steps"), ("tool_duration", "duration16w", "steps"), ("time_left", "session16w", "turns")):
            if name not in got:
                continue
            done = os.path.join("data/freeinference", folder, "validation_done.json")
            if not os.path.exists(done):
                print(f"  (no {done} here, so {name} was not checked against reported scores)")
                continue
            stored = json.load(open(done))
            stored = stored["rem_path_s"]["turns"] if name == "time_left" else stored["steps"]
            worst = 0.0
            for t, v in got[name].items():
                s = stored[str(t)]
                sm, sb = (s["model"]["brier"], s["base"]["brier"]) if name == "next_tool" else (s["model"], s["base"])
                worst = max(worst, abs(v["model"] - sm), abs(v["base"] - sb))
                assert v["n"] == s["n"], (name, t, v["n"], s["n"])
            print(f"  check {name}: per-step averages equal the reported validation scores (largest difference {worst:.2e})")


if __name__ == "__main__":
    main()
