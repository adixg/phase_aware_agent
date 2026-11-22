"""Deadline-only sensitivity on the unchanged all-turn replay engine."""
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sys.path.insert(0, str(BASE))
from replay import DAYS, POLICIES, SCENARIOS, SLOTS, Replay, build

MULTIPLIERS = (1, 2, 3, 8)


class ObservedReplay(Replay):
    def __init__(self, jobs, workflows, arrivals, budgets, multiplier, scope):
        super().__init__(jobs, workflows, arrivals, budgets)
        self.multiplier = multiplier
        self.scope = scope
        self.background_waits = []
        self.background_overdue = 0

    def release(self, time, job_id):
        wf = self.jobs[job_id].workflow
        first = wf not in self.arrivals
        super().release(time, job_id)
        if first and self.scope == "nested" and self.workflows[wf].line is not None:
            self.deadlines[wf] = time + (self.deadlines[wf] - time) * self.multiplier

    def choose(self, now, policy):
        chosen = super().choose(now, policy)
        wf = self.jobs[chosen].workflow
        if self.workflows[wf].line is None:
            self.background_waits.append(now - self.ready[chosen][0])
            self.background_overdue += now > self.deadlines[wf]
        return chosen


def percentile(values):
    return sorted(values)[int(.95 * len(values))]


def summarize(day, scenario, policy, sim, horizon, joins, gpu):
    jobs, workflows = sim.jobs, sim.workflows
    results = []
    for i, workflow in enumerate(workflows):
        arrival, end = sim.arrivals[i], sim.completions[i]
        budget = sim.deadlines[i] - arrival
        latency = end - arrival
        assert latency + 1e-7 >= workflow.ideal
        results.append(dict(key=workflow.key, line=workflow.line, arrival=arrival,
                            completion=end, latency=latency, budget=budget,
                            feasible=workflow.ideal <= budget + 1e-8,
                            ideal=workflow.ideal, overlap=workflow.overlap,
                            live_child=sim.live_child_by_workflow[i],
                            cross_chain_ready=sim.cross_chain_by_workflow[i],
                            child_disagreements=sim.child_disagreed_by_workflow[i],
                            met=end <= sim.deadlines[i]))
    lo, hi = .1 * horizon, .9 * horizon
    util = sum(max(0., min(b, hi) - max(a, lo)) for a, b in sim.busy_intervals) / (SLOTS * (hi-lo))
    row = dict(day=day, scenario=scenario, policy=policy, jobs=len(jobs),
               child_jobs=sum(j.child for j in jobs), joins=joins,
               workflows=len(workflows), nested=sum(w.line is not None for w in workflows),
               horizon=horizon, gpu_work=gpu, slot_util=util, ready_contested=sim.ready_contested,
               contested=sim.contested, disagreements=sim.disagreed, child_contested=sim.child_contested,
               child_disagreements=sim.child_disagreed, child_dispatched_overdue=sim.nested_overdue,
               examples=sim.examples, background_mean_wait=mean(sim.background_waits),
               background_p95_wait=percentile(sim.background_waits),
               background_overdue=sim.background_overdue)
    return row, results


def read(path):
    return {(r["day"], r["scenario"], r["policy"]): r
            for r in map(json.loads, path.read_text().splitlines())}


def main():
    inputs = json.loads((BASE / "inputs.json").read_text())
    old_runs, old_cases = (read(BASE / name) for name in ("runs.jsonl", "workflows.jsonl"))
    config = dict(base_setup=inputs["setup"], multipliers=MULTIPLIERS, days=DAYS,
                  primary="multiply all class budgets",
                  diagnostic="multiply the same class budgets only on the 32 eligible nested turns",
                  inputs_sha256=hashlib.sha256((BASE / "inputs.json").read_bytes()).hexdigest(),
                  engine_sha256=hashlib.sha256((BASE / "replay.py").read_bytes()).hexdigest())
    (OUT / "setup.json").write_text(json.dumps(config, indent=2) + "\n")

    for scope in ("global", "nested"):
        multipliers = MULTIPLIERS if scope == "global" else MULTIPLIERS[1:]
        with (OUT / f"{scope}-runs.jsonl").open("w") as runs, (OUT / f"{scope}-workflows.jsonl").open("w") as cases:
            for day in DAYS:
                for scenario in SCENARIOS:
                    jobs, workflows, arrivals, horizon, joins, gpu = build(day, inputs[day], scenario)
                    for multiplier in multipliers:
                        budgets = [b * multiplier if scope == "global" else b
                                   for b in inputs["setup"]["budgets"]]
                        for policy in POLICIES:
                            sim = ObservedReplay(jobs, workflows, arrivals, budgets, multiplier, scope).run(policy)
                            row, results = summarize(day, scenario, policy, sim, horizon, joins, gpu)
                            key = day, scenario, policy
                            if multiplier == 1:
                                assert all(row[k] == v for k, v in old_runs[key].items()), key
                                assert results == old_cases[key]["results"], key
                            for current, old in zip(results, old_cases[key]["results"]):
                                assert current["key"] == old["key"] and current["ideal"] == old["ideal"], key
                                expected = old["budget"] * (multiplier if scope == "global" or
                                                            current["line"] is not None else 1)
                                assert abs(current["budget"] - expected) < 1e-8, key
                            row.update(multiplier=multiplier, scope=scope)
                            runs.write(json.dumps(row) + "\n")
                            cases.write(json.dumps(dict(day=day, scenario=scenario, policy=policy,
                                                       multiplier=multiplier, scope=scope, results=results)) + "\n")
                        print(day, scenario, scope, multiplier, flush=True)


if __name__ == "__main__":
    main()
