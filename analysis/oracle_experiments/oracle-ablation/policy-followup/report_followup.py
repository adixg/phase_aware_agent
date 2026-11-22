"""Report day-paired policy results and recovery of the original Oracle gain."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
index = {(r["day"], r["kv"], r["family"], r["demotion"]): r for r in rows}
days = ["2026-08-30", "2026-08-31", "2026-09-01"]
assert len(rows) == len(index) == 42
assert {r["day"] for r in rows} == set(days)


def series(kv, family, demotion, metric):
    return [index[day, kv, family, demotion]["paired"][metric] for day in days]


def diff(values, baseline):
    return [100 * (value - reference) for value, reference in zip(values, baseline)]


variants = [("edf", "none")] + [(f, d) for f in ("llf", "srpt")
                                for d in ("cheap", "overdue", "oracle")]
header = ["KV", "Policy", "Feasible / all attainment (%)", "Δ EDF feasible / all (pp)",
          "Daily feasible Δ range (pp)", "Mean / P95 latency (s)", "Δ EDF mean / P95 (s)"]
lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
recovery = []
for kv in (0, 1_000_000):
    pool = "1M" if kv else "Unlimited"
    edf = [series(kv, "edf", "none", m)
           for m in ("attainment", "attainment_all", "mean_latency", "p95_latency")]
    sample = [index[day, kv, "edf", "none"]["paired"] for day in days]
    print(pool, "common turns", sum(r["turns"] for r in sample), "feasible", sum(r["feasible"] for r in sample))
    for family, demotion in variants:
        data = [series(kv, family, demotion, m)
                for m in ("attainment", "attainment_all", "mean_latency", "p95_latency")]
        feasible, all_turns, latency, p95 = map(mean, data)
        feasible_delta, all_delta = [diff(a, b) for a, b in zip(data[:2], edf[:2])]
        label = "EDF" if family == "edf" else f"{family.upper()} ({demotion})"
        cells = [pool, label, f"{100*feasible:.2f} / {100*all_turns:.2f}",
                 f"{mean(feasible_delta):+.2f} / {mean(all_delta):+.2f}",
                 f"{min(feasible_delta):+.2f} to {max(feasible_delta):+.2f}",
                 f"{latency:.0f} / {p95:.0f}",
                 f"{latency-mean(edf[2]):+.0f} / {p95-mean(edf[3]):+.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
        if demotion == "overdue":
            gains = []
            for metric in ("attainment", "attainment_all"):
                baseline = series(kv, family, "cheap", metric)
                fix = mean(diff(series(kv, family, "overdue", metric), baseline))
                oracle = mean(diff(series(kv, family, "oracle", metric), baseline))
                gains.append(f"{fix:+.2f} of {oracle:+.2f} pp ({fix/oracle:.1%})")
            recovery.append(f"- {pool} {family.upper()} recovered {gains[0]} feasible and {gains[1]} all-turn.")

intro = ("Metrics average three days. Attainment and latency use the intersection of turns "
         "arriving in the middle 80% of the horizon across all seven policies within each day/KV cell. "
         "P95 is the mean of daily P95s. Overdue variants use cheap ordering. "
         "Oracle variants use Oracle ordering and demotion. Ranges are observed paired daily ranges.")
report = intro + "\n\n" + "\n".join(lines) + "\n\n" + "\n".join(recovery) + "\n"
(root / "comparison.md").write_text(report)
print(report)
