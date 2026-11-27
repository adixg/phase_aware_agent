"""How predictable is what is left of the CURRENT TURN, for a scheduler? At each admitted request (A event of
Mallika's online rows) it forecasts, with her online hierarchy (prediction/predictor/online_baseline.py):

    rem_work_s   modelled GPU work left in the turn, the current request included: what the scheduler controls
    rem_path_s   the same plus the tool rounds between requests: the turn's time left
    rem_after    requests still to start after this one

Each target picks its feature chain on a split inside the fit weeks (fit May 17 - Jul 31, choose Aug 1-8), is refit
on May 17 - Aug 8 and scored once on the tuning week Aug 9-15 (the 16-week split of analysis/dag/splits.py; the
validation week and later days are not read). The error is reported on the replay's noisy-oracle scale: sigma of
log(p50 / actual), where sim.py's pred_error 0.1 / 0.3 means sigma 0.1 / 0.3 (rem_after uses requests + 1).

Needs the track 1 code and rows: run from a checkout of origin/track1/proportional-deadline-online whose
data/freeinference/dag_all holds online_rows.csv.gz (scripts/turn_forecast_eval.sbatch sets that up). Memory:
about 10-15 GB for the full run, so not on a laptop; --smoke uses a few days only.

    cd <track1 checkout> && PYTHONPATH=. python <this repo>/scripts/turn_forecast_eval.py [--smoke] [--targets ...]
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math

import numpy as np

from analysis.dag import splits
from prediction.predictor.online_baseline import NUMERIC_LEVELS, Hierarchy, OnlineForecaster, quantile_edges

ROWS = "data/freeinference/dag_all/online_rows.csv.gz"
INT = {"step", "turn", "rem_after"}
FLOAT = {"prev1_width", "prev1_round_s", "prev1_errors", "prev1_result_tokens", "user_prior_mean_rounds",
         "sess_prior_mean_rounds", "stall_s", "rem_path_s", "rem_work_s"}
STR = {"session", "day", "harness", "model", "prev1_cat"}
CHAINS = {"step only (baseline)": (), "cat": ("cat",), "cat+harness": ("cat", "harness"), "cat+width": ("cat", "width"),
          "cat+hist": ("cat", "hist"), "cat+stall": ("cat", "stall"), "cat+result": ("cat", "result"),
          "cat+hist+result": ("cat", "hist", "result")}
TIME = {"rem_work_s", "rem_path_s"}


def load(days) -> list[dict]:
    days, rows = set(days), []
    with gzip.open(ROWS, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["event"] != "A" or r["day"] not in days:
                continue
            o = {k: r[k] for k in STR}
            o.update({k: int(r[k]) for k in INT})
            o.update({k: (float(r[k]) if r[k] != "" else None) for k in FLOAT})
            o["prev1_cat"] = o["prev1_cat"] or None
            rows.append(o)
    return rows


def encode(target: str, y: float) -> int:
    """Times in ms rounded to a 2%-wide log bin (keeps each group's distribution small); counts as they are."""
    if target not in TIME:
        return int(y)
    return int(round(math.exp(round(math.log(max(1.0, y * 1000)) / 0.02) * 0.02)))


def decode(target: str, v: float) -> float:
    return v / 1000 if target in TIME else v


class Turn(OnlineForecaster):
    """The online forecaster's inputs (step bucket, then `levels`), one target."""

    def __init__(self, rows, target: str, levels, strength: float):
        self.event, self.levels, self.strength, self.target = "A", tuple(levels), strength, target
        self.edges = {lv: quantile_edges([r[col] for r in rows]) for lv, col in NUMERIC_LEVELS.items() if lv in levels}
        self.h = Hierarchy([(self.inputs(r), (encode(target, r[target]),)) for r in rows if r[target] is not None],
                           strength=strength)

    def forecast(self, row) -> tuple[float, float, float, float]:
        _, m, _ = self.h.distribution(self.inputs(row))
        m, t = m[0], self.target
        return (decode(t, m.quantile(0.1)), decode(t, m.quantile(0.5)), decode(t, m.quantile(0.9)),
                decode(t, m.crps(encode(t, row[t]))))


def score(model: Turn, rows) -> dict:
    t = model.target
    rows = [r for r in rows if r[t] is not None and (r[t] > 0 or t not in TIME)]
    y = np.array([r[t] for r in rows], dtype=float)
    p = np.array([model.forecast(r) for r in rows])
    shift = 0.0 if t in TIME else 1.0
    err = np.log(np.maximum(p[:, 1] + shift, 1e-3) / (y + shift))
    ranks = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return {"n": len(y), "sigma": float(err.std()), "robust_sigma": float(1.4826 * np.median(np.abs(err))),
            "typical_factor": float(math.exp(np.median(np.abs(err)))), "within_2x": float(np.mean(np.abs(err) < math.log(2))),
            "rank_r": float(np.corrcoef(ranks(p[:, 1]), ranks(y))[0, 1]) if len(y) > 2 else float("nan"),
            "crps": float(p[:, 3].mean()), "median_actual": float(np.median(y)),
            "below_p10": float(np.mean(y < p[:, 0])), "below_p50": float(np.mean(y < p[:, 1])),
            "below_p90": float(np.mean(y < p[:, 2]))}


GROUPS = {"turn start (step 0)": lambda r: r["step"] == 0, "steps 1-2": lambda r: 1 <= r["step"] <= 2,
          "steps 3-9": lambda r: 3 <= r["step"] <= 9, "steps 10+": lambda r: r["step"] >= 10, "all requests": lambda r: True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", default="rem_work_s,rem_path_s,rem_after")
    ap.add_argument("--strength", type=float, default=320)
    ap.add_argument("--smoke", action="store_true", help="a few days only: a plumbing check that fits in little memory")
    ap.add_argument("--out", default=None, help="also write the results as JSON here")
    args = ap.parse_args()
    if args.smoke:
        inner, choose, test = splits.days("2026-07-29", "2026-07-31"), splits.days("2026-08-01", "2026-08-01"), ("2026-08-09",)
    else:
        inner, choose, test = splits.days("2026-05-17", "2026-07-31"), splits.days("2026-08-01", "2026-08-08"), splits.TUNE_DAYS
    fit_days = tuple(inner) + tuple(choose)
    rows = load(fit_days + tuple(test))
    fit = [r for r in rows if r["day"] in set(fit_days)]
    tune = [r for r in rows if r["day"] in set(test)]
    del rows
    f1 = [r for r in fit if r["day"] in set(inner)]
    c1 = [r for r in fit if r["day"] in set(choose)][::3]     # a third of the choosing rows ranks the chains well
    print(f"admitted requests: fit {len(fit):,} (inner {len(f1):,}, choosing {len(c1):,}), scored {len(tune):,} "
          f"({test[0]} to {test[-1]})", flush=True)
    results = {}
    for target in args.targets.split(","):
        print(f"\n== {target}", flush=True)
        tried = {}
        for name, levels in CHAINS.items():
            m = Turn(f1, target, levels, args.strength)
            tried[name] = score(m, c1)
            del m
            gc.collect()
            print(f"  choose  {name:22s} CRPS {tried[name]['crps']:9.2f}  sigma {tried[name]['sigma']:.2f}", flush=True)
        best = min((n for n in CHAINS if CHAINS[n]), key=lambda n: tried[n]["crps"])
        print(f"  chosen: {best}", flush=True)
        model, base = Turn(fit, target, CHAINS[best], args.strength), Turn(fit, target, (), args.strength)
        results[target] = {"chain": best, "choosing": tried, "scored": {}}
        for g, keep in GROUPS.items():
            rs = [r for r in tune if keep(r)]
            for who, m in (("model", model), ("base", base)):
                x = score(m, rs)
                results[target]["scored"].setdefault(g, {})[who] = x
                print(f"  {g:20s} {who:5s} n={x['n']:7,}  sigma {x['sigma']:.2f} (robust {x['robust_sigma']:.2f})  "
                      f"typical x{x['typical_factor']:.2f}  within 2x {100 * x['within_2x']:3.0f}%  rank r {x['rank_r']:.2f}  "
                      f"below p10/50/90 {100 * x['below_p10']:.0f}/{100 * x['below_p50']:.0f}/{100 * x['below_p90']:.0f}%  "
                      f"median actual {x['median_actual']:.2f}", flush=True)
        del model, base
        gc.collect()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print("written", args.out)


if __name__ == "__main__":
    main()
