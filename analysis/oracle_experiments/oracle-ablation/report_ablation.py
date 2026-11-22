"""Summarize paired policy effects and request-level demotion diagnostics."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
ROWS = [json.loads(line) for line in (ROOT / "runs.jsonl").read_text().splitlines()]
INDEX = {(r["day"], r["kv"], r["family"], r["ordering"], r["demotion"]): r for r in ROWS}
DAYS = sorted({r["day"] for r in ROWS})
assert len(ROWS) == len(INDEX) == 54
assert DAYS == ["2026-08-30", "2026-08-31", "2026-09-01"]


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


performance, diagnosis, effects = [], [], []
for kv in (0, 1_000_000):
    pool = "1M" if kv else "Unlimited"
    edf = [INDEX[day, kv, "edf", "none", "none"] for day in DAYS]
    print(pool, "common turns", sum(r["paired"]["turns"] for r in edf),
          "feasible", sum(r["paired"]["feasible"] for r in edf))
    variants = sorted({key[2:] for key in INDEX})
    for family, ordering, demotion in variants:
        group = [INDEX[day, kv, family, ordering, demotion] for day in DAYS]
        avg = lambda key: mean(r["paired"][key] for r in group)
        delta = [100 * (r["paired"]["attainment"] - b["paired"]["attainment"])
                 for r, b in zip(group, edf)]
        delta_all = 100 * (avg("attainment_all") - mean(r["paired"]["attainment_all"] for r in edf))
        label = "EDF" if family == "edf" else f"{family.upper()} {ordering}/{demotion}"
        performance.append([pool, label, f"{avg('attainment'):.2%} / {avg('attainment_all'):.2%}",
                            f"{mean(delta):+.2f} / {delta_all:+.2f}",
                            f"{min(delta):+.2f} to {max(delta):+.2f}",
                            f"{avg('mean_latency'):.0f} / {avg('p95_latency'):.0f}",
                            f"{mean(r['mean_queue_wait'] for r in group):.1f} / "
                            f"{mean(r['p95_queue_wait'] for r in group):.1f}",
                            f"{mean(r['memory_blocked_share'] for r in group):.1%} / "
                            f"{mean(r['memory_blocked_seconds'] for r in group):.0f}"])
        counts = defaultdict(int)
        for r in group:
            for key, v in r["demotions"].items():
                counts[key] += v
        diagnosis.append([pool, label, *(counts[key] for key in (
            "events", "requests", "optimistic_fit", "optimistic_fit_met", "optimistic_fit_missed",
            "already_overdue", "ideal_cannot_fit_before_deadline"))])
    for family in ("llf", "srpt"):
        paired = []
        for day in DAYS:
            cc, oc, co, oo = [100 * INDEX[day, kv, family, o, d]["paired"]["attainment"]
                              for o, d in (("cheap", "cheap"), ("oracle", "cheap"),
                                           ("cheap", "oracle"), ("oracle", "oracle"))]
            paired.append([oc-cc, co-cc, oo-co, oo-oc, oo-cc, oo-oc-co+cc])
        effects.append([pool, family.upper(), *(f"{mean(vals):+.2f} [{min(vals):+.2f}, {max(vals):+.2f}]"
                        for vals in zip(*paired))])

report = "\n\n".join([
    "# Oracle ordering and demotion ablation",
    "Attainment and latency use turns in the middle 80% of the horizon under every variant of a "
    "given day and KV size. Performance columns average daily metrics. Delta ranges are observed "
    "paired daily ranges across three days. Queue waits use requests ready in that window, and "
    "blocked time is a clipped wall-clock measure. P95 columns average daily percentiles.",
    table(["KV", "Variant (order/demotion)", "Feasible / all", "Δ EDF pp feasible / all",
           "Daily feasible Δ range", "Latency mean / P95 (s)", "Queue mean / P95 (s)",
           "Memory blocked % / seconds"], performance),
    "Demotion counts sum across days for requests ready in the measurement window. Each unique "
    "request is classified at its first demotion. Events can repeat when memory blocking sends "
    "the same request back to the ready heap. Ideal-fit demotions indicate estimator false "
    "positives against the optimistic bound. A subsequent miss in this group reflects time beyond "
    "that bound under the replayed schedule, including any waiting created by demotion itself.",
    table(["KV", "Variant", "Demotion events", "Unique requests", "Ideal fits",
           "Ideal fits, met", "Ideal fits, missed", "Already overdue", "Not overdue, ideal cannot fit"], diagnosis),
    "Feasible attainment effects are paired day means in percentage points, with the observed day "
    "range in brackets. The interaction is full gain minus both single-switch gains.",
    table(["KV", "Policy", "Ordering with cheap demotion", "Demotion with cheap ordering",
           "Ordering with Oracle demotion", "Demotion with Oracle ordering", "Full gain", "Interaction"], effects),
])
(ROOT / "comparison.md").write_text(report + "\n")
print(report)
