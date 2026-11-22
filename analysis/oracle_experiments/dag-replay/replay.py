"""Dependency replay with infinite KV and the upstream L40S ideal-service model."""
from collections import defaultdict
from dataclasses import dataclass, field
import heapq
import itertools
import json
from statistics import mean
import zlib

from prepare import DAYS, LOAD, PROFILE, ROOT, SEED, SLOTS, compact, cutoff
from analysis.replay.sim import build_sessions

POLICIES = ("edf", "dag", "work", "flat_timing")
SCENARIOS = ("no_join", "plausible_join")


@dataclass
class Job:
    name: str
    workflow: int
    service: float
    edges: dict[int, float] = field(default_factory=dict)
    incoming: int = 0
    tail: float = 0
    child: bool = False


@dataclass
class Workflow:
    key: str
    jobs: list[int] = field(default_factory=list)
    line: int | None = None
    ideal: float = 0
    gpu: float = 0
    lag: float = 0


def build(day, inputs, scenario):
    raw = [s for s in compact(day) if all(r["start"] < cutoff("2026-09-02") and
           r["end"] < cutoff("2026-09-02") and r["prompt"] + r["completion"] <= 1_000_000
           for r in s["requests"])]
    sessions = build_sessions(raw, PROFILE)
    assert len(sessions) == inputs["sessions"]
    candidates = {c["sid"]: c for c in inputs["candidates"]}
    jobs, workflows, arrivals = [], [], []
    join_count = 0

    def add(job):
        group = workflows[job.workflow]
        group.jobs.append(len(jobs))
        group.gpu += job.service
        jobs.append(job)
        return len(jobs) - 1

    def connect(source, target, lag):
        assert lag >= 0 and target not in jobs[source].edges
        jobs[source].edges[target] = lag
        jobs[target].incoming += 1
        if jobs[source].workflow == jobs[target].workflow:
            workflows[jobs[source].workflow].lag += lag

    for session in sessions:
        branches = [b for b in candidates.get(session.sid, {}).get("branches", ()) if not b["excluded"]]
        first = len(jobs)
        for step, req in enumerate(session.reqs):
            if req.step == 0:
                workflows.append(Workflow(f"{session.sid}:{req.turn}",
                                          line=candidates[session.sid]["line"]
                                          if req.turn == 0 and branches else None))
            add(Job(f"root/r{step}", len(workflows) - 1, req.ideal))
        by_emitter = {b["emitter"]: b for b in branches if b["join_candidate"] and scenario == "plausible_join"}
        for step, req in enumerate(session.reqs[:-1]):
            if f"root/r{step}" not in by_emitter:
                connect(first + step, first + step + 1, req.gap)
        for branch in branches:
            emitter = first + int(branch["emitter"].split("/r")[-1])
            assert jobs[emitter].workflow == jobs[first].workflow
            previous = emitter
            for idx, request in enumerate(branch["jobs"]):
                job = add(Job(request["id"], jobs[first].workflow, request["service"], child=True))
                lag = branch["launch_ms"] / 1000 if idx == 0 else branch["gaps"][idx - 1]
                connect(previous, job, lag)
                previous = job
            if branch["emitter"] in by_emitter:
                successor = first + int(branch["successor"].split("/r")[-1])
                assert jobs[successor].workflow == jobs[first].workflow
                connect(previous, successor, branch["residual_ms"] / 1000)
                join_count += 1
        arrivals.append((session.arrival, first))

    # Earliest feasible completion and downstream longest paths with unlimited slots.
    remaining = [job.incoming for job in jobs]
    order = [i for _, i in arrivals]
    start = [0.] * len(jobs)
    for i in order:
        job = jobs[i]
        end = start[i] + job.service
        group = workflows[job.workflow]
        group.ideal = max(group.ideal, end)
        for target, lag in job.edges.items():
            remaining[target] -= 1
            if job.workflow == jobs[target].workflow:
                start[target] = max(start[target], end + lag)
            if remaining[target] == 0:
                order.append(target)
    assert len(order) == len(jobs) and len(set(order)) == len(jobs)
    for i in reversed(order):
        job = jobs[i]
        job.tail = job.service + max((lag + jobs[t].tail for t, lag in job.edges.items()
                                      if jobs[t].workflow == job.workflow), default=0.)
    gpu = sum(j.service for j in jobs)
    horizon = gpu / (SLOTS * LOAD)
    compress = (max(t for t, _ in arrivals) - min(t for t, _ in arrivals)) / horizon
    return jobs, workflows, [(t / compress, i) for t, i in arrivals], horizon, join_count, gpu


class Replay:
    def __init__(self, jobs, workflows, arrivals, budgets, slots=SLOTS):
        self.jobs, self.workflows, self.budgets, self.slots = jobs, workflows, budgets, slots
        self.events, self.ready = [], {}
        self.sequence = itertools.count()
        self.running = {}
        self.running_by_workflow = defaultdict(dict)
        self.active_lags = defaultdict(dict)
        self.gpu_left = [w.gpu for w in workflows]
        self.lag_left = [w.lag for w in workflows]
        self.predecessors = [j.incoming for j in jobs]
        self.outstanding = [len(w.jobs) for w in workflows]
        self.arrivals, self.deadlines, self.completions = {}, {}, {}
        self.finished = set()
        self.ready_contested = self.contested = self.disagreed = self.child_disagreed = 0
        self.child_contested = self.nested_overdue = 0
        self.examples = []
        self.busy_intervals = []
        for time, job in arrivals:
            self.push(time, "release", job)

    def push(self, time, kind, payload):
        heapq.heappush(self.events, (time, next(self.sequence), kind, payload))

    def release(self, time, job_id):
        job = self.jobs[job_id]
        workflow = job.workflow
        if workflow not in self.arrivals:
            self.arrivals[workflow] = time
            key = self.workflows[workflow].key
            cls = zlib.crc32(f"{SEED}:{key}".encode()) % len(self.budgets)
            self.deadlines[workflow] = time + self.budgets[cls]
        self.ready[job_id] = (time, next(self.sequence))

    def finish(self, time, uid):
        job = self.jobs[uid]
        wf = job.workflow
        self.running.pop(uid)
        self.running_by_workflow[wf].pop(uid)
        self.finished.add(uid)
        self.gpu_left[wf] -= job.service
        self.outstanding[wf] -= 1
        if self.outstanding[wf] == 0:
            self.completions[wf] = time
        for target, lag in job.edges.items():
            internal = wf == self.jobs[target].workflow
            if internal:
                self.active_lags[wf][uid, target] = (time, lag)
            self.push(time + lag, "edge", (uid, target, internal, lag))

    def process(self, now, kind, payload):
        if kind == "release":
            self.release(now, payload)
        elif kind == "finish":
            self.finish(now, payload)
        else:
            source, target, internal, lag = payload
            if internal:
                wf = self.jobs[source].workflow
                self.active_lags[wf].pop((source, target))
                self.lag_left[wf] -= lag
            self.predecessors[target] -= 1
            if self.predecessors[target] == 0:
                self.release(now, target)

    def work(self, wf, now, include_lag):
        gpu = self.gpu_left[wf] - sum(now - t for t in self.running_by_workflow[wf].values())
        if not include_lag:
            return gpu
        lag = self.lag_left[wf] - sum(now - t for t, _ in self.active_lags[wf].values())
        return gpu + lag

    def choose(self, now, policy):
        ready = self.ready
        self.ready_contested += len(ready) > 1
        on_time = [i for i in ready if now <= self.deadlines[self.jobs[i].workflow]]
        if not on_time:
            return min(ready, key=ready.__getitem__)
        edf = min(on_time, key=lambda i: (self.deadlines[self.jobs[i].workflow], ready[i]))
        work = {}
        def rank(i):
            job = self.jobs[i]
            wf = job.workflow
            if policy == "dag":
                remaining = job.tail
            else:
                if wf not in work:
                    work[wf] = self.work(wf, now, policy == "flat_timing")
                remaining = work[wf]
            return self.deadlines[wf] - remaining, ready[i]
        chosen = edf if policy == "edf" else min(on_time, key=rank)
        if len(on_time) > 1:
            self.contested += 1
            self.child_contested += any(self.jobs[i].child for i in on_time)
        if chosen != edf:
            self.disagreed += 1
            involves_child = self.jobs[chosen].child or self.jobs[edf].child
            self.child_disagreed += involves_child
            if involves_child and len(self.examples) < 30:
                def details(i):
                    j = self.jobs[i]
                    return dict(job=j.name, workflow=self.workflows[j.workflow].key,
                                line=self.workflows[j.workflow].line, child=j.child,
                                slack=self.deadlines[j.workflow] - now, tail=j.tail,
                                ready=ready[i][0])
                flat = min(on_time, key=lambda i: (
                    self.deadlines[self.jobs[i].workflow] - self.work(self.jobs[i].workflow, now, True),
                    ready[i],
                ))
                self.examples.append(dict(time=now, chosen=details(chosen), edf=details(edf),
                                          flat=details(flat), eligible=len(on_time)))
        return chosen

    def run(self, policy):
        while self.events:
            now = self.events[0][0]
            while self.events and self.events[0][0] == now:
                _, _, kind, payload = heapq.heappop(self.events)
                self.process(now, kind, payload)
            while self.ready and len(self.running) < self.slots:
                i = self.choose(now, policy)
                self.ready.pop(i)
                job = self.jobs[i]
                if job.child and now > self.deadlines[job.workflow]:
                    self.nested_overdue += 1
                self.running[i] = now
                self.running_by_workflow[job.workflow][i] = now
                self.busy_intervals.append((now, now + job.service))
                self.push(now + job.service, "finish", i)
        assert len(self.finished) == len(self.jobs) and len(self.completions) == len(self.workflows)
        return self


def main():
    inputs = json.loads((ROOT / "inputs.json").read_text())
    budgets = inputs["setup"]["budgets"]
    with (ROOT / "runs.jsonl").open("w") as out, (ROOT / "workflows.jsonl").open("w") as cases:
        for day in DAYS:
            for scenario in SCENARIOS:
                jobs, workflows, arrivals, horizon, joins, gpu = build(day, inputs[day], scenario)
                for policy in POLICIES:
                    sim = Replay(jobs, workflows, arrivals, budgets).run(policy)
                    results = []
                    for i, workflow in enumerate(workflows):
                        arrival, end = sim.arrivals[i], sim.completions[i]
                        budget = sim.deadlines[i] - arrival
                        latency = end - arrival
                        assert latency + 1e-7 >= workflow.ideal
                        results.append(dict(key=workflow.key, line=workflow.line, arrival=arrival,
                                            completion=end, latency=latency, budget=budget,
                                            feasible=workflow.ideal <= budget + 1e-8,
                                            ideal=workflow.ideal, met=end <= sim.deadlines[i]))
                    lo, hi = .1 * horizon, .9 * horizon
                    util = sum(max(0., min(b, hi) - max(a, lo)) for a, b in sim.busy_intervals) / (SLOTS * (hi-lo))
                    row = dict(day=day, scenario=scenario, policy=policy, jobs=len(jobs),
                               child_jobs=sum(j.child for j in jobs), joins=joins,
                               workflows=len(workflows), nested=sum(w.line is not None for w in workflows),
                               horizon=horizon, gpu_work=gpu, slot_util=util, ready_contested=sim.ready_contested,
                               contested=sim.contested,
                               disagreements=sim.disagreed, child_contested=sim.child_contested,
                               child_disagreements=sim.child_disagreed, child_dispatched_overdue=sim.nested_overdue,
                               examples=sim.examples)
                    out.write(json.dumps(row) + "\n")
                    cases.write(json.dumps(dict(day=day, scenario=scenario, policy=policy, results=results)) + "\n")
                    out.flush()
                    cases.flush()
                    print(day, scenario, policy, row["nested"], joins, sim.contested, sim.child_disagreed, flush=True)


if __name__ == "__main__":
    main()
