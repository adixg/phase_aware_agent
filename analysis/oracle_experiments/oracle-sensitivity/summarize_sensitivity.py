from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / "runs.jsonl").read_text().splitlines()]
assert len(rows) == 72
groups = defaultdict(list)
for row in rows:
    groups[row["load"], row["kv"], row["policy"], row["eviction"]].append(row)

header = [
    "Load", "KV", "Policy / eviction", "Feasible %", "All %", "Delta feasible / all (pp)",
    "Mean / P95 latency (s)", "Slot util. %", "KV hit %", "Evicted (M tokens)",
    "Queue wait mean / P95 (s)", "Queued %", "Memory blocked % / s"
]
lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
for (load, kv, policy, eviction), group in groups.items():
    average = lambda key: mean(row[key] for row in group)
    baseline = None
    if policy.endswith("_oracle"):
        baseline = groups[load, kv, policy.replace("_oracle", "_cheap"), eviction]
    elif eviction == "oracle":
        baseline = groups[load, kv, policy, "cheap"]
    metrics = ("common_attainment", "common_attainment_all")
    delta = "—" if baseline is None else " / ".join(
        f"{100 * (average(metric) - mean(r[metric] for r in baseline)):+.2f}" for metric in metrics
    )
    cells = [f"{load}", "1M" if kv else "∞", f"{policy} / {eviction}",
             *(f"{100*average(metric):.2f}" for metric in metrics), delta,
             f"{average('common_mean_latency'):.0f} / {average('common_p95_latency'):.0f}",
             f"{100*average('clipped_utilization'):.1f}", f"{100*average('hit_rate'):.1f}",
             f"{average('tokens_evicted')/1e6:.0f}",
             f"{average('mean_queue_wait'):.1f} / {average('p95_queue_wait'):.1f}",
             f"{100*average('queued_share'):.1f}",
             f"{100*average('memory_blocked_share'):.1f} / {average('memory_blocked_seconds'):.0f}"]
    lines.append("| " + " | ".join(cells) + " |")

notes = ("All policy metrics are means of three daily metrics. Attainment and latency use the "
         "same turns across policies within each day/load/KV cell, by intersecting turns arriving "
         "in the middle 80% of the horizon. P95 columns average daily percentiles. Utilization "
         "clips service intervals to the window. Queue waits are ready-to-dispatch for requests "
         "ready in the window. Memory blocked is wall time in the window with a selected request "
         "unable to dispatch for lack of KV despite a free slot. KV hit rate and evictions span "
         "the full run. Oracle admission deltas are versus matched cheap admission. Oracle "
         "eviction deltas are versus cheap eviction with EDF.")
(ROOT / "comparison.md").write_text(notes + "\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
