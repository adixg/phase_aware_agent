"""Replay simulator against small cases whose answers can be worked out by hand."""

import math

import pytest

from analysis.replay.sim import (POLICIES, Cheap, Config, Profile, Req, Session, Simulator, _annotate, build_sessions,
                                 estimate_remaining, estimate_windowed, reply_time, step_bucket, summarize)

# 1 ms per prompt token, 10 ms per output token: easy arithmetic in seconds
PROFILE = Profile(decode_ms=10.0, prefill_ms=1.0)


def raw_req(prompt, completion=100, lcp=0, user=False, gap=None, cat="__none__", append=True):
    return {"prompt": prompt, "lcp": lcp, "completion": completion, "user": user, "append": append,
            "cat": cat, "gap": gap}


def raw_session(sid, start_s, reqs):
    return {"sid": sid, "start": int(start_s * 1000), "requests": reqs}


def run(raw, profile=PROFILE, cheap=None, **kw):
    cfg = Config(slots=kw.pop("slots", 1), kv_tokens=kw.pop("kv_tokens", math.inf), profile=profile, **kw)
    return Simulator(build_sessions(raw, profile), cfg, cheap).run()


def single(sid, start, prompt=1000, completion=100):
    return raw_session(sid, start, [raw_req(prompt, completion, user=True)])


def test_single_request_service_time():
    sim = run([single("a", 0)])
    assert sim.turns[0].latency == pytest.approx(1.0 + 1.0)   # 1000 * 1 ms + 100 * 10 ms


def test_fcfs_queues_on_one_slot():
    sim = run([single("a", 0), single("b", 0.5)])
    assert sorted(t.latency for t in sim.turns) == pytest.approx([2.0, 3.5])


def test_second_slot_removes_the_queue():
    sim = run([single("a", 0), single("b", 0.5)], slots=2)
    assert [t.latency for t in sim.turns] == pytest.approx([2.0, 2.0])


def two_step(sid, start, gap=5.0, prompt2=1100, lcp=1000):
    return raw_session(sid, start, [
        raw_req(1000, 100, user=True, gap=int(gap * 1000), cat="Read"),
        raw_req(prompt2, 100, lcp=lcp, gap=None),
    ])


def test_resident_prefix_covers_previous_prompt_and_output():
    # step 2's base = lcp 1000 + previous output 100 = its whole 1100-token prompt
    sim = run([two_step("a", 0)])
    assert sim.turns[0].latency == pytest.approx((1 + 1) + 5 + (0 + 1))
    assert sim.tokens_hit == 1100


def test_mutation_only_reuses_the_common_prefix():
    raw = [raw_session("a", 0, [
        raw_req(1000, 100, user=True, gap=0, cat="Read"),
        raw_req(1100, 100, lcp=400, append=False),
    ])]
    assert run(raw).turns[0].latency == pytest.approx(2 + (0.7 + 1))


def test_evicted_prefix_is_recomputed():
    # pool holds one session; b's request between a's two steps pushes a's prefix out
    raw = [two_step("a", 0, gap=5), single("b", 3, prompt=1000, completion=100)]
    sim = run(raw, kv_tokens=1100, slots=2)
    a = max(sim.turns, key=lambda t: t.latency)
    assert a.latency == pytest.approx(2 + 5 + (1.1 + 1))
    assert sim.tokens_evicted == 2200   # a's prefix for b, then b's for a's return


def test_cpu_tier_makes_the_reload_cheap():
    raw = [two_step("a", 0, gap=5), single("b", 3)]
    prof = Profile(decode_ms=10.0, prefill_ms=1.0, reload_ms=0.1)
    a = max(run(raw, profile=prof, kv_tokens=1100, slots=2).turns, key=lambda t: t.latency)
    assert a.latency == pytest.approx(2 + 5 + (0.11 + 1))


def test_lru_evicts_the_oldest_and_oracle_the_furthest_next_use():
    # b finishes first but returns soon (t=5); a returns at t=53. c needs 300 more tokens than
    # are free, shaved off the tail of the victim. LRU picks b, the one about to come back.
    raw = [
        raw_session("b", 0, [raw_req(1000, 100, user=True, gap=3_000, cat="Read"), raw_req(1100, 100, lcp=1000)]),
        raw_session("a", 1, [raw_req(1000, 100, user=True, gap=50_000, cat="Read"), raw_req(1100, 100, lcp=1000)]),
        raw_session("c", 3.5, [raw_req(1000, 100, user=True)]),
    ]
    b_turn = lambda sim: next(t for t in sim.turns if t.z == pytest.approx(6.0))
    assert b_turn(run(raw, kv_tokens=3000, slots=3, eviction="oracle")).latency == pytest.approx(6.0)
    assert b_turn(run(raw, kv_tokens=3000, slots=3, eviction="lru")).latency == pytest.approx(6.3)


def test_queued_session_is_scored_by_its_place_in_line():
    raw = [two_step("queued", 0), two_step("gap", 0), two_step("front", 0)]
    cheap = cheap_for(raw)
    cfg = Config(slots=2, kv_tokens=1e9, profile=PROFILE, eviction="oracle")
    sim = Simulator(build_sessions(raw, PROFILE), cfg, cheap)
    queued, in_gap, front = sim.states
    in_gap.next_use = 100.0
    ranks = {queued.idx: 40, front.idx: 0}
    assert sim._score(front, 10.0, ranks) < sim._score(queued, 10.0, ranks)
    assert sim._score(in_gap, 10.0, ranks) == pytest.approx(90.0)


def blocking_job():
    return [single("hog", 0, prompt=1000, completion=100), single("late", 0.1), single("urgent", 0.2)]


def budgets_for(raw, wanted):
    """Budgets indexed by each session's crc32 class, so a test can set per-session deadlines."""
    import zlib
    table = [math.inf] * 97
    for sid, b in wanted.items():
        table[zlib.crc32(f"0:{sid}:0".encode()) % 97] = b
    return tuple(table)


def test_edf_serves_the_nearer_deadline_first():
    raw = blocking_job()
    budgets = budgets_for(raw, {"hog": 100.0, "late": 100.0, "urgent": 3.9})
    assert sum(t.missed for t in run(raw, policy="fcfs", budgets=budgets).turns) == 1
    assert sum(t.missed for t in run(raw, policy="edf", budgets=budgets).turns) == 0


def chain(sid, start, n, gap_s=1.0):
    reqs = [raw_req(100, 100, user=i == 0, gap=None if i == n - 1 else int(gap_s * 1000), cat="Read", lcp=0, append=False)
            for i in range(n)]
    return raw_session(sid, start, reqs)


def cheap_for(raw):
    return Cheap(build_sessions(raw, PROFILE), PROFILE)


def test_oracle_laxity_sees_the_long_turn_that_edf_cannot():
    # "long" has the later deadline (EDF runs short first) but far more left to do, so its
    # laxity (8.5 s) is smaller than short's (14.1 s)
    raw = [single("hog", 0, prompt=0, completion=200), chain("long", 0.1, 6), chain("short", 0.2, 1)]
    budgets = budgets_for(raw, {"long": 20.0, "short": 15.0, "hog": 1000.0})
    edf = run(raw, policy="edf", budgets=budgets)
    oracle = run(raw, policy="llf_oracle", budgets=budgets, cheap=cheap_for(raw))
    long_z = 6 * 1.1 + 5
    assert [t for t in oracle.turns if t.z == pytest.approx(long_z)][0].latency < \
        [t for t in edf.turns if t.z == pytest.approx(long_z)][0].latency


def test_lost_turn_is_served_after_the_others():
    raw = [single("hog", 0), single("doomed", 0.1), single("fine", 0.2)]
    budgets = budgets_for(raw, {"doomed": 0.5, "fine": 50.0, "hog": 1000.0})
    # at t=2 the doomed turn's deadline has passed, so "fine" goes first despite a later deadline
    sim = run(raw, policy="edf", budgets=budgets)
    by_budget = {t.budget: t for t in sim.turns}
    assert by_budget[50.0].latency == pytest.approx(2 + 2 - 0.2)


def test_remaining_work_annotation_is_exact():
    reqs = [Req(1000, 0, 100, 2.0, "Read", "__start__", 0, 0, False),
            Req(1100, 1000, 100, 3.0, "Read", "Read", 1, 0, False),
            Req(1200, 1100, 100, 60.0, "__none__", "Read", 2, 0, True),
            Req(500, 0, 50, None, "__none__", "__none__", 0, 1, True)]
    z = _annotate(reqs, PROFILE)
    assert [r.rem_calls for r in reqs] == [3, 2, 1, 1]
    assert reqs[0].rem_service == pytest.approx(2 + 1.1 + 1.1)
    assert reqs[0].rem_gaps == pytest.approx(5) and reqs[2].rem_gaps == 0
    assert z == pytest.approx([2 + 1.1 + 1.1 + 5, 1.0])


def test_closed_loop_keeps_the_concurrency_limit():
    raw = [single("a", 100), single("b", 200), single("c", 300)]
    sim = run(raw, slots=4, closed=1)
    assert [t.latency for t in sim.turns] == pytest.approx([2.0, 2.0, 2.0])
    assert sim.busy[2][0] == pytest.approx(4.0)     # one after another, arrival times ignored


def test_every_request_is_served_when_memory_is_tight():
    # a prompt bigger than the pool still runs (clamped), and queues behind memory do not deadlock
    raw = [single("big", 0, prompt=5000), single("a", 0), single("b", 0)]
    sim = run(raw, kv_tokens=1500, slots=3)
    assert len(sim.turns) == 3 and sim.used <= 1500


def test_summarize_counts_only_feasible_turns():
    raw = [single("a", 0), single("b", 0)]
    budgets = budgets_for(raw, {"a": 1.0, "b": 10.0})   # a needs 2 s, so it can never make 1 s
    s = summarize(run(raw, budgets=budgets))
    assert s["feasible_turns"] == 1 and s["attainment"] == 1.0


def test_requests_of_a_turn_are_linked_and_the_last_has_no_successor():
    sess = build_sessions([chain("a", 0, 3), single("b", 1)], PROFILE)[0]
    assert sess.reqs[0].nxt is sess.reqs[1] and sess.reqs[1].nxt is sess.reqs[2]
    assert sess.reqs[2].nxt is None


def test_window_equals_the_oracle_once_it_reaches_the_end_of_the_turn():
    raw = [chain("long", 0, 6)]
    cheap, head = cheap_for(raw), build_sessions(raw, PROFILE)[0].reqs[0]
    exact = 6 * 1.1 + 5 * 1.0                       # six calls of 1.1 s and five 1 s gaps
    assert estimate_windowed(head, 6, cheap) == pytest.approx(exact)
    assert estimate_windowed(head, 100, cheap) == pytest.approx(exact)
    assert estimate_remaining(head, POLICIES["llf_oracle"], cheap, PROFILE) == pytest.approx(exact)


def test_window_knows_the_next_k_requests_and_guesses_the_rest():
    raw = [chain("long", 0, 6)]
    cheap, reqs = cheap_for(raw), build_sessions(raw, PROFILE)[0].reqs
    known = 2 * (1.1 + 1.0)                          # two calls and the gap after each
    tail = cheap.service + (cheap.remaining_calls(reqs[2]) - 1) * (cheap.service + cheap.gap)
    assert estimate_windowed(reqs[0], 2, cheap) == pytest.approx(known + tail)
    # one step of lookahead is the one request plus a guess for everything after it
    assert estimate_windowed(reqs[5], 1, cheap) == pytest.approx(1.1)   # last request: nothing to guess


def test_longer_windows_move_the_estimate_toward_the_truth():
    # a long turn the history underestimates: here each deeper window is no further from the truth
    # (not true of every turn: a guess can overshoot, then a deeper window corrects it)
    raw = [chain("long", 0, 12)] + [chain(f"s{i}", i, 2) for i in range(1, 20)]
    cheap, head = cheap_for(raw), build_sessions(raw, PROFILE)[0].reqs[0]
    truth = head.rem_service + head.rem_gaps
    errs = [abs(estimate_windowed(head, k, cheap) - truth) for k in (1, 2, 4, 8, 12)]
    assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])) and errs[-1] == pytest.approx(0, abs=1e-9)


def test_deep_lookahead_schedules_like_the_oracle():
    raw = [single("hog", 0, prompt=0, completion=200), chain("long", 0.1, 6), chain("short", 0.2, 1)]
    budgets = budgets_for(raw, {"long": 20.0, "short": 15.0, "hog": 1000.0})
    lat = lambda pol: sorted(t.latency for t in run(raw, policy=pol, budgets=budgets, cheap=cheap_for(raw)).turns)
    assert lat("llf_la8") == pytest.approx(lat("llf_oracle"))
    assert lat("srpt_la8") == pytest.approx(lat("srpt_oracle"))


def test_summarize_reports_tail_latency_and_slowdown():
    s = summarize(run([single("a", 0), single("b", 0)]))
    assert s["p99_latency"] == pytest.approx(4.0)    # b waits behind a: 2 s + 2 s
    assert s["mean_slowdown"] == pytest.approx((1.0 + 2.0) / 2)   # latency / no-contention duration


def test_pure_srpt_ignores_deadlines_that_the_deadline_policy_gives_up_on():
    # at t=2 "doomed" (0.2 s of work) can no longer make its deadline and "fine" needs 6 s. The
    # deadline-aware SRPT serves fine first; pure SRPT serves the shorter job, doomed, first.
    raw = [single("hog", 0), single("doomed", 0.1, prompt=100, completion=10), single("fine", 0.2, prompt=1000, completion=500)]
    budgets = budgets_for(raw, {"hog": 1000.0, "doomed": 0.5, "fine": 50.0})
    history = cheap_for([chain("past", 0, 3)])      # the history table needs some tool gaps to average
    doomed = lambda pol: next(t for t in run(raw, policy=pol, budgets=budgets, cheap=history).turns if t.budget == 0.5)
    assert doomed("psrpt_oracle").latency == pytest.approx(2 + 0.2 - 0.1)
    assert doomed("srpt_oracle").latency == pytest.approx(2 + 6 + 0.2 - 0.1)


def test_pure_srpt_has_cheap_oracle_and_depth_variants():
    assert {"psrpt_cheap", "psrpt_oracle", "psrpt_la1", "psrpt_la8"} <= set(POLICIES)
    assert all(POLICIES[p].pure for p in ("psrpt_cheap", "psrpt_oracle", "psrpt_la4")) and not POLICIES["srpt_oracle"].pure


def two_turns(gap_s):
    return [raw_session("a", 0, [raw_req(1000, 100, user=True, gap=int(gap_s * 1000)), raw_req(1000, 100, user=True)])]


@pytest.mark.parametrize("spec, pause", [("recorded", 100), ("fixed:5", 5), ("scale:0.5", 50), ("fixed:7200", 3600)])
def test_reply_time_sets_the_pause_before_the_next_turn(spec, pause):
    sessions = build_sessions(two_turns(100), PROFILE, reply=reply_time(spec))
    sim = Simulator(sessions, Config(slots=1, kv_tokens=math.inf, profile=PROFILE)).run()
    assert sim.turns[1].arrival == pytest.approx(2 + pause)   # the 3600 s cap still applies


def test_drawn_reply_times_repeat_and_match_their_distribution():
    exp, logn = reply_time("exp:30"), reply_time("lognormal:60,1.5")
    assert exp("s1", 1, 0) == exp("s1", 1, 0) and exp("s1", 1, 0) != exp("s1", 2, 0)
    draws = sorted(exp(f"s{i}", 1, 0) for i in range(4000))
    assert sum(draws) / len(draws) == pytest.approx(30, rel=0.06)
    assert sorted(logn(f"s{i}", 1, 0) for i in range(4000))[2000] == pytest.approx(60, rel=0.1)


def test_unknown_reply_time_is_refused():
    with pytest.raises(ValueError):
        reply_time("gamma:2")


def test_session_unit_puts_the_whole_session_under_one_deadline():
    # two turns, 5 s apart: as sessions they are one unit whose time includes the pause. The second
    # request reuses the first one's 100 output tokens, so it prefills 900 of its 1000.
    sessions = build_sessions(two_turns(100), PROFILE, reply=reply_time("fixed:5"), unit="session")
    assert [len(s.z) for s in sessions] == [1] and sessions[0].z[0] == pytest.approx(2 + 5 + 1.9)
    assert [r.step for r in sessions[0].reqs] == [0, 1] and sessions[0].reqs[0].rem_calls == 2
    sim = Simulator(sessions, Config(slots=1, kv_tokens=math.inf, profile=PROFILE)).run()
    assert [t.latency for t in sim.turns] == pytest.approx([2 + 5 + 1.9])


def test_turn_unit_still_splits_at_user_messages():
    sessions = build_sessions(two_turns(100), PROFILE, reply=reply_time("fixed:5"))
    assert sessions[0].z == pytest.approx([2, 1.9])


def test_late_steps_share_a_bucket_per_doubling():
    assert [step_bucket(s) for s in (0, 11, 12, 23, 24, 47, 48, 500)] == [0, 11, 12, 12, 13, 13, 14, 17]


def preempt_case():
    # A starts at 0 (1000-token prompt: 1 s prefill, then 1 s decode); B arrives at 0.5 needing as much,
    # in a 2000-token pool that cannot hold both. B's deadline is tight, A's loose.
    raw = [single("A", 0), single("B", 0.5)]
    return raw, budgets_for(raw, {"A": 100.0, "B": 3.0})


def test_without_preemption_the_urgent_request_waits():
    raw, budgets = preempt_case()
    sim = run(raw, slots=2, kv_tokens=2000, policy="edf", budgets=budgets)
    by_budget = {t.budget: t for t in sim.turns}
    assert by_budget[3.0].latency == pytest.approx(2 - 0.5 + 2) and by_budget[3.0].missed
    assert sim.preemptions == 0


def test_priority_preemption_runs_the_urgent_request_and_resumes_the_victim():
    raw, budgets = preempt_case()
    sim = run(raw, slots=2, kv_tokens=2000, policy="edf", budgets=budgets, preempt="priority")
    by_budget = {t.budget: t for t in sim.turns}
    # A is halfway through its prompt at 0.5: it keeps 500 computed tokens, B runs 0.5-2.5, then A
    # resumes with 500 tokens left to read (0.5 s) and its whole output (1 s): done at 4.0
    assert sim.preemptions == 1
    assert by_budget[3.0].latency == pytest.approx(2.0) and not by_budget[3.0].missed
    assert by_budget[100.0].latency == pytest.approx(4.0)
    assert sim.running == 0 and not sim.active


def test_a_less_urgent_request_never_preempts():
    raw = [single("A", 0), single("B", 0.5)]
    budgets = budgets_for(raw, {"A": 3.0, "B": 100.0})     # now the running one is the urgent one
    sim = run(raw, slots=2, kv_tokens=2000, policy="edf", budgets=budgets, preempt="priority")
    assert sim.preemptions == 0
    assert sorted(t.latency for t in sim.turns) == pytest.approx([2.0, 3.5])


def test_preempted_in_decode_keeps_its_output_so_far():
    # A decodes from 1 s; B (urgent) arrives at 1.5, when A has 50 of its 100 output tokens
    raw = [single("A", 0), single("B", 1.5)]
    budgets = budgets_for(raw, {"A": 100.0, "B": 3.0})
    sim = run(raw, slots=2, kv_tokens=2000, policy="edf", budgets=budgets, preempt="priority")
    by_budget = {t.budget: t for t in sim.turns}
    # A keeps 1050 tokens; B needs 1100 of the 950 free, so 150 of A's go. A resumes at 3.5 with
    # 900 of its 1050-token prompt cached: 0.15 s prefill + 0.5 s for its last 50 tokens
    assert by_budget[100.0].latency == pytest.approx(3.5 + 0.15 + 0.5)


# ---- the session oracle split into the user turn, the number of turns to come, and their sizes

def three_turns(sid="a", start=0.0):
    """User turns of 3, 1 and 2 requests."""
    return raw_session(sid, start, [
        raw_req(1000, 100, user=True, gap=1000, cat="Read"), raw_req(1000, 100, gap=1000, cat="Read"),
        raw_req(1000, 100, gap=2000, cat="Read"),
        raw_req(1000, 100, user=True, gap=3000, cat="Read"),
        raw_req(1000, 100, user=True, gap=1000, cat="Read"), raw_req(1000, 100, gap=None),
    ])


@pytest.mark.parametrize("unit", ["turn", "session"])
def test_user_turn_fields_are_the_same_whatever_the_deadline_unit(unit):
    sess = build_sessions([three_turns()], PROFILE, unit=unit)[0]
    got = [(r.uturn, r.rem_uturn, r.fut_turns, r.fut_calls) for r in sess.reqs]
    assert got == [(0, 3, 2, 3), (0, 2, 2, 3), (0, 1, 2, 3), (1, 1, 1, 2), (2, 2, 0, 0), (2, 1, 0, 0)]


def test_a_session_deadline_still_treats_the_session_as_one_turn():
    sess = build_sessions([three_turns()], PROFILE, unit="session")[0]
    assert len(sess.z) == 1 and sess.reqs[0].rem_calls == 6


def history_of_two_turn_sessions(n=25):
    raw = [raw_session(f"s{i}", i, [raw_req(1000, 100, user=True, gap=1000, cat="Read"), raw_req(1000, 100),
                                    raw_req(1000, 100, user=True, gap=None)]) for i in range(n)]
    raw = [dict(s, requests=[dict(r, gap=(1000 if j == 1 else r["gap"])) for j, r in enumerate(s["requests"])]) for s in raw]
    return Cheap(build_sessions(raw, PROFILE, unit="session"), PROFILE)


def test_history_knows_the_turn_index_and_borrows_from_earlier_turns_when_sparse():
    cheap = history_of_two_turn_sessions()
    assert cheap.future_calls(0) == pytest.approx(1.0)         # turns of 2 then 1: one request after the first turn
    assert cheap.future_calls(1) == pytest.approx(0.0)
    assert cheap.future_calls(7) == pytest.approx(0.0)         # nobody reached turn 7: the nearest earlier turn answers
    assert cheap.calls_per_uturn == pytest.approx(1.5)


def test_policies_for_the_decomposition_exist_and_pure_ones_keep_no_deadline_logic():
    for fam in ("llf", "srpt", "psrpt"):
        assert POLICIES[f"{fam}_ucur"].scope == "cur" and POLICIES[f"{fam}_ucount"].scope == "count"
    assert POLICIES["psrpt_ucur"].pure and not POLICIES["llf_ucur"].pure


def test_knowing_the_count_of_later_turns_changes_only_the_later_turns_term():
    cheap = history_of_two_turn_sessions()
    r = build_sessions([three_turns()], PROFILE, unit="session")[0].reqs[0]
    est = lambda name: estimate_remaining(r, POLICIES[name], cheap, PROFILE)
    per_call = cheap.service + cheap.gap
    # current turn true (3 calls) plus 1.0 expected later, against 2 later turns of 1.5 calls each
    assert est("llf_ucount") - est("llf_ucur") == pytest.approx((2 * 1.5 - cheap.future_calls(0)) * per_call)
    # the history estimate and the current-turn estimate differ only in their call count
    assert est("llf_ucur") - est("llf_cheap") == pytest.approx(
        ((3 + cheap.future_calls(0)) - cheap.remaining_calls(r)) * per_call)


def test_with_nothing_left_after_this_turn_the_count_variant_is_exactly_the_turn():
    cheap = history_of_two_turn_sessions()
    last = build_sessions([three_turns()], PROFILE, unit="session")[0].reqs[-2]    # first request of the final turn
    assert last.fut_turns == 0
    prefill = (PROFILE.overhead_ms + PROFILE.prefill_ms * (last.prompt - last.base)) / 1000
    expected = prefill + PROFILE.decode_ms * cheap.completion / 1000 + (last.rem_uturn - 1) * (cheap.service + cheap.gap)
    assert estimate_remaining(last, POLICIES["llf_ucount"], cheap, PROFILE) == pytest.approx(expected)


@pytest.mark.parametrize("policy", ["llf_ucur", "llf_ucount", "srpt_ucur", "psrpt_ucount"])
def test_the_decomposition_policies_run(policy):
    raw = [three_turns("a", 0), three_turns("b", 1)]
    sim = run(raw, slots=1, policy=policy, budgets=budgets_for(raw, {"a": 30.0, "b": 30.0}), cheap=cheap_for(raw))
    assert len(sim.turns) == 6
