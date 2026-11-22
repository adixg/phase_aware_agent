"""Report paired policy outcomes and same-state dispatch disagreements."""
import json
from statistics import mean

from prepare import DAYS, ROOT, SLOTS
from replay import POLICIES, SCENARIOS


def read(name):
    rows = [json.loads(line) for line in (ROOT / name).read_text().splitlines()]
    return {(r["day"], r["scenario"], r["policy"]): r for r in rows}


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, r)) + " |"
                     for r in (headers, ["---"] * len(headers), *rows))


def metrics(rows):
    feasible = [r for r in rows if r["feasible"]]
    latencies = sorted(r["latency"] for r in rows)
    return (mean(r["met"] for r in feasible), mean(r["met"] for r in rows),
            mean(latencies), latencies[int(.95 * len(latencies))])


def delta(a, b):
    return f"{a-b:+.1f}"


def main():
    runs = read("runs.jsonl")
    cases = {key: {r["key"]: r for r in row["results"]} for key, row in read("workflows.jsonl").items()}
    assert len(runs) == len(cases) == len(DAYS) * len(POLICIES) * len(SCENARIOS)
    all_turns, windows, dispatch, individual = [], [], [], []
    for scenario in SCENARIOS:
        group = [cases[day, scenario, "edf"] for day in DAYS]
        eligible = [sum(r["line"] is not None for r in day.values()) for day in group]
        assert sum(eligible) == 8
        intersection = {}
        for day in DAYS:
            baseline = cases[day, scenario, "edf"]
            horizon = runs[day, scenario, "edf"]["horizon"]
            comparison = [cases[day, scenario, p] for p in POLICIES]
            assert all(baseline.keys() == c.keys() for c in comparison)
            assert all(abs(r["budget"] - baseline[k]["budget"]) < 1e-8
                       and r["feasible"] == baseline[k]["feasible"] for c in comparison for k, r in c.items())
            intersection[day] = set.intersection(*({k for k, r in c.items()
                                                   if .1*horizon <= r["arrival"] < .9*horizon}
                                                  for c in comparison))
        for policy in POLICIES:
            daily = [cases[day, scenario, policy] for day in DAYS]
            for target, selected in [(all_turns, [set(r) for r in group]),
                                     (windows, [intersection[d] for d in DAYS])]:
                policy_stats = [metrics([actual[k] for k in cohort]) for actual, cohort in zip(daily, selected)]
                edf_stats = [metrics([base[k] for k in cohort]) for base, cohort in zip(group, selected)]
                a, b, avg, p95 = map(mean, zip(*policy_stats))
                daily_deltas = [100*(x[0]-y[0]) for x,y in zip(policy_stats, edf_stats)]
                target.append([scenario, policy, sum(map(len, selected)), f"{100*a:.2f} / {100*b:.2f}",
                               f"{mean(daily_deltas):+.2f} / {100*(b-mean(m[1] for m in edf_stats)):+.2f}",
                               f"{min(daily_deltas):+.2f} to {max(daily_deltas):+.2f}", f"{avg:.1f} / {p95:.1f}"])
            stats = [runs[day, scenario, policy] for day in DAYS]
            dispatch.append([scenario, policy, *(sum(r[k] for r in stats) for k in
                             ("ready_contested", "contested", "disagreements", "child_contested", "child_disagreements",
                              "child_dispatched_overdue")),
                             f"{mean(r['slot_util'] for r in stats):.1%}"])
        for day, baseline in zip(DAYS, group):
            for key, value in sorted(baseline.items(), key=lambda item: item[1]["line"] or 0):
                if value["line"] is None:
                    continue
                others = [cases[day, scenario, p][key] for p in POLICIES[1:]]
                individual.append([scenario, f"{day} line {value['line']}", f"{value['budget']:.1f}",
                                   f"{value['latency']:.1f}",
                                   *(f"{x['latency']:.1f} ({delta(x['latency'], value['latency'])})" for x in others),
                                   "/".join("Y" if r["met"] else "N" for r in [value, *others])])

    headings = ["Scenario", "Policy", "Turns", "Feasible / all (%)", "Δ EDF (pp)",
                "Feasible Δ daily range (pp)", "Mean / P95 (s)"]
    report = "\n\n".join([
        "# Validation DAG replay",
        "All timings and attainment are calculated on paired populations and averaged across days. "
        "Feasibility uses the no-contention critical path for the scenario. The endpoint is the "
        "last visible modeled LLM response in the workflow. P95 is the mean of daily P95s.",
        "## All replayed turns", table(headings, all_turns),
        "## Common middle 80% arrival window", table(headings, windows),
        "## Dispatch diagnostics",
        "Ready contention counts dispatches with multiple ready jobs. Live contention requires "
        "multiple unexpired ready jobs. Disagreements compare the selected job to shadow EDF on "
        "that exact ready set. These are policy-local states. Child disagreements require the "
        "selected job or the shadow EDF job to be a child request.",
        table(["Scenario", "Policy", "Ready contention", "Live contention", "EDF disagreements",
               "Live contention with child", "Disagreements involving child", "Overdue child dispatches",
               "Window slot utilization"], dispatch),
        "## All eligible first turns",
        "Values are completion latency from the workflow's admission. Parentheses are paired "
        "deltas versus EDF. Deadline status is in policy column order.",
        table(["Scenario", "Validation locator", "Budget (s)", "EDF (s)", "DAG (s)",
               "Remaining GPU work (s)", "Flat timing (s)", "Met deadline"], individual),
    ])
    (ROOT / "comparison.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
