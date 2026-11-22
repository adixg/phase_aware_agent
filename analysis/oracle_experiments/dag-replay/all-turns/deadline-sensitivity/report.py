"""Paired deadline sensitivity tables and per-turn outcome changes."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
DAYS = ("2026-08-30", "2026-08-31", "2026-09-01")
SCENARIOS = ("no_join", "plausible_join")
POLICIES = ("edf", "dag", "work", "flat_timing")


def read(name):
    rows = [json.loads(line) for line in (ROOT / name).read_text().splitlines()]
    return {(r["scope"], r["multiplier"], r["scenario"], r["policy"], r["day"]): r for r in rows}


def metrics(turns):
    feasible = [r for r in turns if r["feasible"]]
    latency = sorted(r["latency"] for r in turns)
    return (len(feasible), sum(r["met"] for r in feasible), sum(r["met"] for r in turns),
            mean(latency), latency[int(.95 * len(latency))])


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in (headers, ["---"] * len(headers), *rows))


def percent_pair(feas, all_turns):
    return f"{100*feas:.2f} / {100*all_turns:.2f}"


def flips(turns, reference):
    return (sum(r["met"] and not b["met"] for r, b in zip(turns, reference)),
            sum(b["met"] and not r["met"] for r, b in zip(turns, reference)))


def write_flips(turns):
    with (ROOT / "deadline-flips.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("scope", "multiplier", "scenario", "policy", "day", "root_hash",
                         "nested_shard_line", "turn", "budget_s", "ideal_s", "edf_latency_s",
                         "policy_latency_s", "edf_met", "policy_met"))
        for (scope, multiplier, scenario, policy, day), row in turns.items():
            if policy == "edf":
                continue
            baseline = turns[scope, multiplier, scenario, "edf", day]["results"]
            for actual, reference in zip(row["results"], baseline):
                if actual["met"] == reference["met"]:
                    continue
                sid, turn = actual["key"].rsplit(":", 1)
                writer.writerow((scope, multiplier, scenario, policy, day,
                                 hashlib.sha256(sid.encode()).hexdigest()[:16], actual["line"],
                                 turn, actual["budget"], actual["ideal"], reference["latency"],
                                 actual["latency"], reference["met"], actual["met"]))


def main():
    runs, turns = (read(f"global-{name}.jsonl") | read(f"nested-{name}.jsonl")
                   for name in ("runs", "workflows"))
    assert len(runs) == len(turns) == 168
    write_flips(turns)
    headline, subgroups, background, changed = {}, {}, {}, {}
    with (ROOT / "nested-paired.csv").open("w", newline="") as out:
        csvout = csv.writer(out)
        csvout.writerow(["scope", "multiplier", "scenario", "policy", "date", "shard_line", "turn",
                         "feasible", "observed_overlap", "budget_s", "ideal_s", "latency_s",
                         "edf_latency_s", "delta_edf_s", "met", "edf_met", "live_child_contests",
                         "cross_chain_ready", "child_disagreements"])
        for scope in ("global", "nested"):
            headline[scope], subgroups[scope], background[scope], changed[scope] = [], [], [], []
            for multiplier in ((1, 2, 3, 8) if scope == "global" else (2, 3, 8)):
                for scenario in SCENARIOS:
                    for policy in POLICIES:
                        keys = [(scope, multiplier, scenario, policy, d) for d in DAYS]
                        baseline_keys = [(scope, multiplier, scenario, "edf", d) for d in DAYS]
                        daily = [turns[k]["results"] for k in keys]
                        ref = [turns[k]["results"] for k in baseline_keys]
                        assert [len(x) for x in daily] == [len(x) for x in ref]
                        assert sum(map(len, daily)) == 5136
                        for day, base in zip(daily, ref):
                            assert all(r["key"] == b["key"] and r["feasible"] == b["feasible"]
                                       and abs(r["budget"]-b["budget"]) < 1e-8 for r,b in zip(day,base))
                        da, db, ml, p95 = [], [], [], []
                        for actual, base in zip(daily, ref):
                            f, fm, am, avg, tail = metrics(actual)
                            f0, fm0, am0, _, _ = metrics(base)
                            da.append((fm/f, fm0/f0))
                            db.append((am/len(actual), am0/len(base)))
                            ml.append(avg)
                            p95.append(tail)
                        att_f, edf_f = map(mean, zip(*da))
                        att_all, edf_all = map(mean, zip(*db))
                        sum_metric = lambda name: sum(runs[k][name] for k in keys)
                        a = [r for day in daily for r in day]
                        b = [r for day in ref for r in day]
                        nested = [(r, base) for r, base in zip(a,b) if r["line"] is not None]
                        assert len(nested) == 32
                        n, nb = map(list, zip(*nested))
                        assert all(not r["met"] or r["feasible"] for r in n)
                        nf, nfm, nm, nml, np95 = metrics(n)
                        up, down = flips(a,b)
                        nup, ndown = flips(n,nb)
                        headline[scope].append([
                            multiplier, scenario, policy, percent_pair(att_f, att_all),
                            f"{100*(att_f-edf_f):+.2f} / {100*(att_all-edf_all):+.2f}",
                            f"{nfm}/{nf} / {nm}/32", f"+{up}/−{down}", f"+{nup}/−{ndown}",
                            f"{mean(ml):.1f} / {mean(p95):.1f}", f"{nml:.1f} / {np95:.1f}",
                            sum_metric("contested"), sum_metric("child_contested"),
                            f"{sum_metric('disagreements')} / {sum_metric('child_disagreements')}",
                        ])
                        for label, subset in (
                            ("all nested", nested),
                            ("feasible nested", [(r, base) for r,base in nested if r["feasible"]]),
                            ("feasible + observed overlap", [(r,base) for r,base in nested
                                                              if r["feasible"] and r["overlap"]]),
                        ):
                            x,y = map(list, zip(*subset))
                            _,_,met,avg,tail = metrics(x)
                            improves, worsens = flips(x,y)
                            subgroups[scope].append([
                                multiplier, scenario, policy, label, f"{met}/{len(x)}",
                                f"+{improves}/−{worsens}", f"{avg:.1f} / {tail:.1f}",
                                f"{mean(r['latency']-base['latency'] for r,base in subset):+.1f}",
                                sum(r["live_child"] > 0 for r in x), sum(r["cross_chain_ready"] > 0 for r in x),
                            ])
                        baseline_wait = mean(runs["global",1,scenario,policy,d]["background_mean_wait"] for d in DAYS)
                        curr_wait = mean(runs[k]["background_mean_wait"] for k in keys)
                        bg = [(r, base) for r,base in zip(a,b) if r["line"] is None]
                        bg_up, bg_down = flips(*map(list,zip(*bg)))
                        background[scope].append([
                            multiplier, scenario, policy, f"{curr_wait:.2f}",
                            f"{curr_wait-baseline_wait:+.2f}", sum_metric("background_overdue"),
                            f"+{bg_up}/−{bg_down}",
                        ])
                        for day, actual, base in zip(DAYS, daily, ref):
                            for r, er in zip(actual, base):
                                if r["line"] is None:
                                    continue
                                turn = r["key"].rsplit(":",1)[1]
                                csvout.writerow([scope,multiplier,scenario,policy,day,r["line"],turn,
                                                 r["feasible"],r["overlap"],r["budget"],r["ideal"],
                                                 r["latency"],er["latency"],r["latency"]-er["latency"],
                                                 r["met"],er["met"],r["live_child"],
                                                 r["cross_chain_ready"],r["child_disagreements"]])
                                if r["met"] != er["met"]:
                                    changed[scope].append([multiplier, scenario, policy,
                                                           f"{day} line {r['line']} turn {turn}",
                                                           "gain" if r["met"] else "loss", r["overlap"],
                                                           f"{er['latency']:.1f} / {r['latency']:.1f} / {r['budget']:.1f}"])

    for scope in ("global", "nested"):
        title = "Primary: all class deadlines scaled" if scope == "global" else "Diagnostic: only nested deadlines scaled"
        sections = [
            f"# {title}",
            "Full-workload percentages and latency are unweighted daily means. Nested metrics pool cases. "
            "Disagreements use shadow EDF on the policy's own ready set. Child contention requires at "
            "least two unexpired ready jobs, one of them nested. Gains/losses compare realized turn outcomes to EDF.",
            table(["Multiplier", "Scenario", "Policy", "Feasible / all (%)", "Δ EDF (pp)", "Nested feasible / all met",
                   "All flips", "Nested flips", "All mean / P95 (s)", "Nested mean / P95 (s)",
                   "Live contests", "Child contests", "EDF choices changed / child"], headline[scope]),
            "## Nested case groups",
            table(["Multiplier", "Scenario", "Policy", "Cohort", "Met", "Flips", "Mean / P95 (s)",
                   "Mean Δ EDF (s)", "Cases with child contention", "Cases with cross-chain ready"], subgroups[scope]),
            "## Exact nested deadline flips relative to EDF",
            table(["Multiplier", "Scenario", "Policy", "Locator", "Change", "Gateway overlap",
                   "EDF latency / policy latency / budget (s)"], changed[scope]),
            "## Background scheduling",
            "Background comprises the other 5,104 turns. Wait measures ready to dispatch. Overdue dispatch "
            "counts are relative to the active budgets. The nested-only diagnostic holds background budgets at 1×.",
            table(["Multiplier", "Scenario", "Policy", "Background mean wait (s)", "Wait Δ vs 1× (s)",
                   "Overdue background dispatches", "Background deadline flips vs EDF"], background[scope]),
        ]
        (ROOT / f"{scope}-comparison.md").write_text("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
