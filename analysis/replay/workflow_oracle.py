"""Clairvoyant oracle vs non-preemptive EDF on identical GPUs, one LLM call per GPU at a time.

A workflow is one session of the trace, replayed as the gate does (see sim.build_sessions with
unit="session" and 5 s user replies): a chain of LLM calls, each followed by a tool or user gap
that holds no GPU. Durations are those of an unlimited KV cache, so they do not depend on the
schedule, and every time is an integer number of milliseconds so both schedulers see exactly the
same instance. Each workflow's deadline is arrival + scale x its time alone on the server.

EDF picks, whenever a GPU is free, the ready LLM call whose workflow has the earliest deadline
(ties: arrival, then id). It is handed only a view of the ready call (EdfView), never a workflow,
so it cannot read future calls or durations. The oracle is given the whole instance: every
future call, gap, duration, arrival and deadline. It solves max #(C_i <= D_i), then minimum total
completion time, with OR-Tools CP-SAT, warm-started from EDF and from clairvoyant list schedules,
and reports the best schedule found next to the solver's proven upper bound.

    python -m analysis.replay.workflow_oracle --days 2026-08-16 --scales 1.1,1.2,1.5 --out oracle_vs_edf.csv
    python -m analysis.replay.workflow_oracle --merge part_a.csv part_b.csv --out oracle_vs_edf.csv
(needs orjson and ortools)
"""

from __future__ import annotations

import argparse
import csv
import heapq
import time
from dataclasses import dataclass, field
from typing import Callable

GPUS = 8


@dataclass(frozen=True)
class Workflow:
    wid: str
    arrival: int                 # ms
    deadline: int                # ms, absolute
    llm: tuple[int, ...]         # ms of GPU time per LLM call, in order
    gaps: tuple[int, ...]        # ms after call k before call k+1 is ready (len(llm) - 1)

    def __post_init__(self):
        assert len(self.gaps) == len(self.llm) - 1 and all(d > 0 for d in self.llm)

    @property
    def alone(self) -> int:
        """Completion time with the GPUs to itself, measured from arrival."""
        return sum(self.llm) + sum(self.gaps)


@dataclass(frozen=True)
class EdfView:
    """All EDF may know about a ready LLM call: whose it is, when that workflow arrived, its deadline."""
    wid: str
    arrival: int
    deadline: int


@dataclass(frozen=True)
class Op:
    wid: str
    k: int
    gpu: int
    start: int
    end: int


@dataclass
class Schedule:
    ops: list[Op]
    completion: dict[str, int]
    info: dict = field(default_factory=dict)

    def on_time(self, wfs: list[Workflow]) -> int:
        return sum(self.completion.get(w.wid, 10 ** 18) <= w.deadline for w in wfs)


def workflows_from_sessions(sessions, compress: float, scale: float) -> list[Workflow]:
    """Sessions built with unit="session" (one turn each) and an unlimited cache: call k takes its
    `ideal` time, since its prefix is always resident."""
    out = []
    for s in sessions:
        assert len(s.z) == 1, "build sessions with unit='session'"
        llm = tuple(max(1, round(r.ideal * 1000)) for r in s.reqs)
        gaps = tuple(round(r.gap * 1000) for r in s.reqs[:-1])
        arrival = round(s.arrival / compress * 1000)
        alone = sum(llm) + sum(gaps)
        out.append(Workflow(s.sid, arrival, arrival + round(scale * alone), llm, gaps))
    return out


def list_schedule(wfs: list[Workflow], gpus: int, priority: Callable, view: Callable) -> Schedule:
    """Non-preemptive list scheduling: whenever a GPU is free, start the ready call with the
    smallest priority(view(workflow, call, now), now). Ties fall to arrival, then workflow id.
    GPUs are taken lowest id first."""
    by_id = {w.wid: w for w in wfs}
    events = [(w.arrival, 0, w.wid, 0, -1) for w in wfs]   # (time, 0 = call ready / 1 = call done, wid, k, gpu)
    heapq.heapify(events)
    ready: list = []
    free = list(range(gpus))
    ops, completion = [], {}
    while events:
        now = events[0][0]
        while events and events[0][0] == now:
            _, kind, wid, k, g = heapq.heappop(events)
            if kind == 0:
                ready.append((wid, k))
            else:
                heapq.heappush(free, g)
                w = by_id[wid]
                if k + 1 < len(w.llm):
                    heapq.heappush(events, (now + w.gaps[k], 0, wid, k + 1, -1))
                else:
                    completion[wid] = now
        if free and ready:
            ranked = sorted(ready, key=lambda c: (priority(view(by_id[c[0]], c[1], now), now),
                                                  by_id[c[0]].arrival, c[0]))
            for wid, k in ranked[:len(free)]:
                ready.remove((wid, k))
                g = heapq.heappop(free)
                end = now + by_id[wid].llm[k]
                ops.append(Op(wid, k, g, now, end))
                heapq.heappush(events, (end, 1, wid, k, g))
    return Schedule(ops, completion)


def edf(wfs: list[Workflow], gpus: int = GPUS) -> Schedule:
    """Non-preemptive EDF. Its priority sees an EdfView only: no calls, gaps or durations."""
    s = list_schedule(wfs, gpus, priority=lambda v, now: v.deadline,
                      view=lambda w, k, now: EdfView(w.wid, w.arrival, w.deadline))
    s.info["scheduler"] = "edf"
    return s


def _remaining(w: Workflow, k: int) -> int:
    """True time from the start of call k to the workflow's end, with no waiting."""
    return sum(w.llm[k:]) + sum(w.gaps[k:])


def clairvoyant_list(wfs: list[Workflow], gpus: int, rule: str) -> Schedule:
    """List schedules using the true future: least laxity, or shortest remaining among the
    workflows that can still make it. Workflows that cannot make it go last."""
    def priority(c, now):
        w, k = c
        rem = _remaining(w, k)
        hopeless = now + rem > w.deadline
        return (hopeless, w.deadline - now - rem if rule == "llf" else rem)
    s = list_schedule(wfs, gpus, priority=priority, view=lambda w, k, now: (w, k))
    s.info["scheduler"] = f"clairvoyant_{rule}"
    return s


def assign_gpus(starts: list[tuple[int, int, str, int]], gpus: int) -> list[Op]:
    """Concrete GPUs for calls whose overlap never exceeds `gpus` (interval colouring)."""
    free = list(range(gpus))
    busy: list = []    # (end, gpu)
    ops = []
    for start, end, wid, k in sorted(starts):
        while busy and busy[0][0] <= start:
            heapq.heappush(free, heapq.heappop(busy)[1])
        g = heapq.heappop(free)          # IndexError here would mean the capacity was violated
        heapq.heappush(busy, (end, g))
        ops.append(Op(wid, k, g, start, end))
    return ops


def oracle(wfs: list[Workflow], gpus: int = GPUS, time_limit: float = 60.0, workers: int = 8,
           tie_break: bool = True) -> Schedule:
    """CP-SAT with the whole future known: maximise workflows on time, then minimise total
    completion time. Identical GPUs, one call each, are one cumulative resource of capacity
    `gpus`; concrete GPUs are assigned afterwards. Warm-started from the best of EDF and the
    clairvoyant list schedules, so the result is never below them. tie_break=False skips the
    second objective, which changes no on-time count."""
    from ortools.sat.python import cp_model

    seeds = [edf(wfs, gpus), clairvoyant_list(wfs, gpus, "llf"), clairvoyant_list(wfs, gpus, "srpt")]
    seed = max(seeds, key=lambda s: (s.on_time(wfs), -sum(s.completion.values())))
    seed_start = {(o.wid, o.k): o.start for o in seed.ops}
    horizon = max(w.arrival for w in wfs) + sum(w.alone for w in wfs) + 1

    m = cp_model.CpModel()
    starts, intervals, ends, ok = {}, [], {}, {}
    for w in wfs:
        earliest = w.arrival
        for k, d in enumerate(w.llm):
            s = m.NewIntVar(earliest, horizon, f"s_{w.wid}_{k}")
            intervals.append(m.NewFixedSizeIntervalVar(s, d, f"i_{w.wid}_{k}"))
            starts[(w.wid, k)] = s
            if k:
                m.Add(s >= starts[(w.wid, k - 1)] + w.llm[k - 1] + w.gaps[k - 1])
            earliest += d + (w.gaps[k] if k < len(w.gaps) else 0)
        ends[w.wid] = starts[(w.wid, len(w.llm) - 1)] + w.llm[-1]
        ok[w.wid] = m.NewBoolVar(f"ok_{w.wid}")
        m.Add(ends[w.wid] <= w.deadline).OnlyEnforceIf(ok[w.wid])
    m.AddCumulative(intervals, [1] * len(intervals), gpus)
    for key, v in starts.items():
        m.AddHint(v, seed_start[key])
    for w in wfs:
        m.AddHint(ok[w.wid], seed.completion[w.wid] <= w.deadline)

    def solve(limit):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = limit
        solver.parameters.num_search_workers = workers
        return solver, solver.Solve(m)

    t0 = time.time()
    m.Maximize(sum(ok.values()))
    solver, status = solve(time_limit)
    info = {"scheduler": "oracle", "phase1_status": solver.StatusName(status), "seed": seed.info["scheduler"],
            "seed_on_time": seed.on_time(wfs)}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        seed.info.update(info, oracle_upper_bound=len(wfs), solve_s=time.time() - t0, note="solver found nothing; seed kept")
        return seed
    best = round(solver.ObjectiveValue())
    info["oracle_upper_bound"] = round(solver.BestObjectiveBound())
    phase1 = {key: solver.Value(v) for key, v in starts.items()}
    if best < seed.on_time(wfs):     # the hint was not kept within the time limit: keep the seed
        best, phase1 = seed.on_time(wfs), seed_start

    by_id = {w.wid: w for w in wfs}
    if not tie_break:
        ops = assign_gpus([(st, st + by_id[wid].llm[k], wid, k) for (wid, k), st in phase1.items()], gpus)
        completion = {w.wid: phase1[(w.wid, len(w.llm) - 1)] + w.llm[-1] for w in wfs}
        info.update(phase2_status="skipped", solve_s=time.time() - t0)
        return Schedule(ops, completion, info)

    # phase 2: at least as many on time, least total completion time
    m.ClearObjective()
    m.ClearHints()
    m.Add(sum(ok.values()) >= best)
    for key, v in starts.items():
        m.AddHint(v, phase1[key])
    m.Minimize(sum(ends.values()))
    solver, status = solve(time_limit)
    info["phase2_status"] = solver.StatusName(status)
    chosen = {key: solver.Value(v) for key, v in starts.items()} if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else phase1
    ops = assign_gpus([(st, st + by_id[wid].llm[k], wid, k) for (wid, k), st in chosen.items()], gpus)
    completion = {w.wid: chosen[(w.wid, len(w.llm) - 1)] + w.llm[-1] for w in wfs}
    info["solve_s"] = time.time() - t0
    return Schedule(ops, completion, info)


def check(wfs: list[Workflow], sched: Schedule, gpus: int) -> list[str]:
    """Every call scheduled once, for exactly its duration, after its dependencies, on one GPU
    that is running nothing else; completion is the end of the last call."""
    errors = []
    by_id = {w.wid: w for w in wfs}
    got = {(o.wid, o.k): o for o in sched.ops}
    if len(got) != len(sched.ops):
        errors.append("a call is scheduled twice")
    for w in wfs:
        for k, d in enumerate(w.llm):
            o = got.get((w.wid, k))
            if o is None:
                errors.append(f"{w.wid} call {k} never scheduled")
                continue
            if o.end - o.start != d:
                errors.append(f"{w.wid} call {k} ran {o.end - o.start} ms, not {d}")
            ready = w.arrival if k == 0 else got[(w.wid, k - 1)].end + w.gaps[k - 1] if (w.wid, k - 1) in got else None
            if ready is not None and o.start < ready:
                errors.append(f"{w.wid} call {k} starts at {o.start} before it is ready at {ready}")
            if not 0 <= o.gpu < gpus:
                errors.append(f"{w.wid} call {k} on GPU {o.gpu}")
        last = got.get((w.wid, len(w.llm) - 1))
        if last is not None and sched.completion.get(w.wid) != last.end:
            errors.append(f"{w.wid} completion {sched.completion.get(w.wid)} is not its last call's end {last.end}")
    for g in range(gpus):
        mine = sorted((o.start, o.end, o.wid, o.k) for o in sched.ops if o.gpu == g)
        for a, b in zip(mine, mine[1:]):
            if b[0] < a[1]:
                errors.append(f"GPU {g} runs {a[2]}#{a[3]} and {b[2]}#{b[3]} at once")
    extra = set(got) - {(w.wid, k) for w in wfs for k in range(len(w.llm))}
    if extra:
        errors.append(f"calls that are not in the workload: {sorted(extra)[:3]}")
    return errors


def compare(wfs: list[Workflow], gpus: int = GPUS, time_limit: float = 60.0, tie_break: bool = True,
            workers: int = 8) -> dict:
    """Both schedulers on one instance; both schedules are checked before anything is reported."""
    a, b = edf(wfs, gpus), oracle(wfs, gpus, time_limit, workers=workers, tie_break=tie_break)
    for s in (a, b):
        errs = check(wfs, s, gpus)
        assert not errs, f"{s.info['scheduler']} schedule is infeasible: {errs[:3]}"
    n, ea, ob = len(wfs), a.on_time(wfs), b.on_time(wfs)
    assert ob >= ea, "the oracle can always reproduce EDF's schedule; investigate"
    return {
        "workflows": n, "calls": sum(len(w.llm) for w in wfs),
        "edf_on_time": ea, "edf_attainment_pct": 100 * ea / n,
        "oracle_on_time": ob, "oracle_attainment_pct": 100 * ob / n,
        "abs_improvement_pts": 100 * (ob - ea) / n,
        "rel_improvement_pct": 100 * (ob - ea) / ea if ea else float("nan"),
        "oracle_upper_bound_on_time": b.info.get("oracle_upper_bound"),
        "oracle_upper_bound_pct": 100 * b.info.get("oracle_upper_bound", n) / n,
        "oracle_status": b.info.get("phase1_status"), "oracle_seed": b.info.get("seed"),
        "oracle_seed_on_time": b.info.get("seed_on_time"), "solve_s": round(b.info.get("solve_s", 0), 1),
    }


def totals(rows: list[dict]) -> list[dict]:
    """One row per scale over all days: counts are summed, so each workflow weighs the same."""
    out = []
    for scale in sorted({float(r["scale"]) for r in rows}):
        sel = [r for r in rows if float(r["scale"]) == scale]
        n, ea, ob, ub = (sum(int(r[k]) for r in sel) for k in ("workflows", "edf_on_time", "oracle_on_time",
                                                                  "oracle_upper_bound_on_time"))
        out.append({"day": "all", "scale": scale, "gpus": sel[0]["gpus"], "load": sel[0]["load"], "workflows": n,
                    "calls": sum(int(r["calls"]) for r in sel), "edf_on_time": ea, "edf_attainment_pct": 100 * ea / n,
                    "oracle_on_time": ob, "oracle_attainment_pct": 100 * ob / n, "abs_improvement_pts": 100 * (ob - ea) / n,
                    "rel_improvement_pct": 100 * (ob - ea) / ea if ea else float("nan"),
                    "oracle_upper_bound_on_time": ub, "oracle_upper_bound_pct": 100 * ub / n,
                    "oracle_status": "/".join(sorted({r["oracle_status"] for r in sel}))})
    return out


def write_csv(rows: list[dict], path: str) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    from analysis.replay.run_gate import PROFILES, SPANS, find_compress, load

    TEST_DAYS = SPANS["16weeks"][1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", default="data/freeinference/compact")
    ap.add_argument("--days", default=",".join(TEST_DAYS))
    ap.add_argument("--scales", default="1.1,1.2,1.5")
    ap.add_argument("--load", type=float, default=0.9)
    ap.add_argument("--gpus", type=int, default=GPUS)
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds per CP-SAT phase")
    ap.add_argument("--workers", type=int, default=8, help="CP-SAT search threads")
    ap.add_argument("--no-tie-break", action="store_true",
                    help="skip the second objective (least total completion time); on-time counts are unchanged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", nargs="+", metavar="CSV", help="combine earlier runs' CSVs into --out instead of running")
    args = ap.parse_args()
    if args.merge:
        rows = [r for p in args.merge for r in csv.DictReader(open(p)) if r["day"] != "all"]
        write_csv(rows + totals(rows), args.out)
        return
    rows = []
    for day in args.days.split(","):
        sessions = load(args.compact, day, PROFILES["l40s"], reply="fixed:5", unit="session")
        if not sessions:
            continue
        compress, _ = find_compress(sessions, args.gpus, args.load)
        for scale in (float(x) for x in args.scales.split(",")):
            wfs = workflows_from_sessions(sessions, compress, scale)
            row = {"day": day, "scale": scale, "gpus": args.gpus, "load": args.load, **compare(wfs, args.gpus, args.time_limit, not args.no_tie_break, args.workers)}
            rows.append(row)
            print(f"{day} x{scale}: EDF {row['edf_attainment_pct']:.1f}%  oracle {row['oracle_attainment_pct']:.1f}% "
                  f"(bound {row['oracle_upper_bound_pct']:.1f}%, {row['oracle_status']}, {row['solve_s']} s)", flush=True)
    rows += totals(rows)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
