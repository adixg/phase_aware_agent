"""Clairvoyant oracle vs non-preemptive EDF: what each may know, and that every schedule is feasible."""

import random

import pytest

from analysis.replay.workflow_oracle import (EdfView, Op, Schedule, Workflow, assign_gpus, check, clairvoyant_list,
                                            edf, list_schedule)


def wf(wid, arrival, deadline, llm, gaps=()):
    return Workflow(wid, arrival, deadline, tuple(llm), tuple(gaps))


def edf_trap():
    """One GPU. EDF runs A first (deadline 7 < 9) and B's second call then misses. Running B's first
    call before A gets both on time: B0 0-2, A 2-7, B1 7-9 (ready at 2 + 5)."""
    return [wf("A", 0, 7, [5]), wf("B", 0, 9, [2, 2], [5])]


def random_instance(seed, n=8, max_calls=4):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        calls = [rng.randint(1, 6) for _ in range(rng.randint(1, max_calls))]
        gaps = [rng.randint(0, 6) for _ in calls[1:]]
        arrival = rng.randint(0, 12)
        out.append(wf(f"w{i}", arrival, arrival + int(rng.uniform(1.0, 1.6) * (sum(calls) + sum(gaps))), calls, gaps))
    return out


# --- what each scheduler may know -------------------------------------------------------------

def test_edf_priority_is_given_only_the_ready_calls_workflow_id_arrival_and_deadline():
    seen = []
    list_schedule(edf_trap(), 1, priority=lambda v, now: seen.append(v) or v.deadline,
                  view=lambda w, k, now: EdfView(w.wid, w.arrival, w.deadline))
    assert seen and all(type(v) is EdfView for v in seen)
    assert set(EdfView.__dataclass_fields__) == {"wid", "arrival", "deadline"}


def test_edf_decisions_do_not_depend_on_the_future():
    # same workflows, but B's unseen later calls differ: EDF's first decisions cannot change
    a = [wf("A", 0, 20, [3]), wf("B", 0, 30, [2, 2], [1]), wf("C", 1, 25, [4])]
    b = [wf("A", 0, 20, [3]), wf("B", 0, 30, [2, 50, 50], [1, 1]), wf("C", 1, 25, [4])]
    first = lambda s: sorted((o.start, o.wid, o.k) for o in s.ops if o.start < 4)
    assert first(edf(a, 1)) == first(edf(b, 1))


def test_clairvoyant_schedule_uses_the_future_and_beats_edf_on_the_trap():
    wfs = edf_trap()
    assert edf(wfs, 1).on_time(wfs) == 1
    assert clairvoyant_list(wfs, 1, "llf").on_time(wfs) == 2


# --- feasibility --------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("gpus", [1, 2, 3])
def test_list_schedules_are_feasible(seed, gpus):
    wfs = random_instance(seed)
    for s in (edf(wfs, gpus), clairvoyant_list(wfs, gpus, "llf"), clairvoyant_list(wfs, gpus, "srpt")):
        assert check(wfs, s, gpus) == []
        assert set(s.completion) == {w.wid for w in wfs}


def test_checker_catches_every_kind_of_violation():
    wfs = [wf("A", 0, 10, [2, 2], [3])]
    good = Schedule([Op("A", 0, 0, 0, 2), Op("A", 1, 0, 5, 7)], {"A": 7})
    assert check(wfs, good, 1) == []
    early = Schedule([Op("A", 0, 0, 0, 2), Op("A", 1, 0, 4, 6)], {"A": 6})       # ignores the 3 ms gap
    cut = Schedule([Op("A", 0, 0, 0, 1), Op("A", 1, 0, 5, 7)], {"A": 7})         # call 0 cut short
    missing = Schedule([Op("A", 0, 0, 0, 2)], {"A": 2})
    two = [wf("A", 0, 10, [3]), wf("B", 0, 10, [3])]
    clash = Schedule([Op("A", 0, 0, 0, 3), Op("B", 0, 0, 1, 4)], {"A": 3, "B": 4})
    assert any("before it is ready" in e for e in check(wfs, early, 1))
    assert any("ran 1 ms" in e for e in check(wfs, cut, 1))
    assert any("never scheduled" in e for e in check(wfs, missing, 1))
    assert any("at once" in e for e in check(two, clash, 1))


def test_gpu_assignment_respects_capacity():
    ops = assign_gpus([(0, 4, "a", 0), (1, 3, "b", 0), (3, 6, "b", 1), (4, 5, "c", 0)], 2)
    assert check([wf("a", 0, 9, [4]), wf("b", 0, 9, [2, 3], [0]), wf("c", 0, 9, [1])],
                 Schedule(ops, {"a": 4, "b": 6, "c": 5}), 2) == []
    with pytest.raises(IndexError):
        assign_gpus([(0, 4, "a", 0), (1, 3, "b", 0), (2, 5, "c", 0)], 2)


# --- the oracle ---------------------------------------------------------------------------------

def test_oracle_finds_the_schedule_edf_misses():
    pytest.importorskip("ortools")
    from analysis.replay.workflow_oracle import compare, oracle
    wfs = edf_trap()
    s = oracle(wfs, 1, time_limit=10)
    assert check(wfs, s, 1) == [] and s.on_time(wfs) == 2 and s.info["phase1_status"] == "OPTIMAL"
    row = compare(wfs, 1, time_limit=10)
    assert (row["edf_on_time"], row["oracle_on_time"], row["abs_improvement_pts"]) == (1, 2, 50.0)


@pytest.mark.parametrize("seed", range(15))
def test_optimal_oracle_is_feasible_and_never_below_edf(seed):
    pytest.importorskip("ortools")
    from analysis.replay.workflow_oracle import oracle
    wfs = random_instance(seed, n=7)
    for gpus in (1, 2):
        e, o = edf(wfs, gpus), oracle(wfs, gpus, time_limit=10)
        assert check(wfs, o, gpus) == []
        assert o.info["phase1_status"] == "OPTIMAL"
        assert o.on_time(wfs) >= e.on_time(wfs)
        assert o.on_time(wfs) == o.info["oracle_upper_bound"]


def test_both_schedulers_get_the_same_instance_and_count_every_workflow():
    pytest.importorskip("ortools")
    from analysis.replay.workflow_oracle import compare
    wfs = random_instance(3, n=6)
    frozen = list(wfs)
    row = compare(wfs, 2, time_limit=10)
    assert wfs == frozen                      # neither scheduler changed the workload or deadlines
    assert row["workflows"] == 6 and row["edf_attainment_pct"] == 100 * row["edf_on_time"] / 6
