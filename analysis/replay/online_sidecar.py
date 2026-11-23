"""The online forecaster's predictions as a sidecar file the replay reads, so the replay never recomputes features.

Exploratory use. The frozen configuration (analysis/dag/online_eval.py --tune) is fitted on Aug 23-29 and
applied to the validation days; nothing is retuned. One row per request and event:

    session, turn, step, event, mean_after, p_last, stall_s
    A  request admitted:   mean_after = requests still to start after this one (the replay adds the current one)
    R  response completed: stall_s = forecast wall time of this request's round

    python -m analysis.replay.online_sidecar          # data/freeinference/online/sidecar_val.csv.gz
"""

from __future__ import annotations

import csv
import gzip
import json
import os

from analysis.dag import online_eval as E
from analysis.dag import online_gates as G
from prediction.predictor.online_baseline import OnlineForecaster

SIDECAR = os.path.join(E.OUT, "sidecar_val.csv.gz")
COLUMNS = ["session", "turn", "step", "event", "mean_after", "p_last", "stall_s"]


def build(out: str = SIDECAR) -> int:
    frozen = json.load(open(os.path.join(E.OUT, "frozen.json")))
    fit = E.load(E.ROWS, G.FIT_DAYS + G.TUNE_DAYS)
    val = E.load(E.VALIDATION_ROWS, G.VALIDATION_DAYS)
    n = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for event in ("A", "R"):
            c = frozen[event]
            model = OnlineForecaster(E.split(fit, event), event, tuple(c["levels"]), c["strength"])
            for r in E.split(val, event):
                p = model.predict(r)
                w.writerow([r["session"], r["turn"], r["step"], event, "%.6f" % p.mean_after, "%.6f" % p.p_last,
                            "%.4f" % p.mean_stall_s])
                n += 1
    return n


def annotate(sessions, path: str = SIDECAR) -> int:
    """Set pred_online (remaining calls, this one included) and pred_stall on every request with a prediction."""
    table = {}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            table[(r["session"], int(r["turn"]), int(r["step"]), r["event"])] = r
    hit = 0
    for s in sessions:
        for r in s.reqs:
            a = table.get((s.sid, r.turn, r.step, "A"))
            if a is not None:
                r.pred_online = float(a["mean_after"]) + 1.0
                hit += 1
            b = table.get((s.sid, r.turn, r.step, "R"))
            if b is not None:
                r.pred_stall = float(b["stall_s"])
    return hit


if __name__ == "__main__":
    print(f"{build():,} predictions -> {SIDECAR}")
