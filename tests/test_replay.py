"""Replay simulator against small cases whose answers can be worked out by hand."""

import math

import pytest

from analysis.replay.sim import (Cheap, Config, Profile, Req, Session, Simulator, _annotate, build_sessions,
                                 reply_time, step_bucket, summarize)

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


def pred_case():
    raw = [chain("long", 0.1, 6), chain("short", 0.2, 1), single("hog", 0, prompt=0, completion=200)]
    return raw, budgets_for(raw, {"long": 20.0, "short": 15.0, "hog": 1000.0})


def test_predicted_remaining_with_no_error_matches_the_oracle():
    raw, budgets = pred_case()
    cheap = cheap_for(raw)
    oracle = run(raw, policy="llf_oracle", budgets=budgets, cheap=cheap)
    pred = run(raw, policy="llf_pred", budgets=budgets, cheap=cheap, pred_error=0.0)
    assert [t.latency for t in pred.turns] == pytest.approx([t.latency for t in oracle.turns])


def test_a_gate_that_trusts_nothing_is_exactly_edf():
    raw, budgets = pred_case()
    edf = run(raw, policy="edf", budgets=budgets)
    gated = run(raw, policy="llf_pred", budgets=budgets, pred_error=0.5, pred_gate=0.0)
    assert [t.latency for t in gated.turns] == [t.latency for t in edf.turns]


def test_predicted_remaining_is_repeatable_and_noisy():
    raw, budgets = pred_case()
    a = run(raw, policy="srpt_pred", budgets=budgets, pred_error=0.8, seed=1)
    b = run(raw, policy="srpt_pred", budgets=budgets, pred_error=0.8, seed=1)
    assert [t.latency for t in a.turns] == [t.latency for t in b.turns]


CPU = Profile(decode_ms=10.0, prefill_ms=1.0, reload_ms=0.5)    # 1100 tokens reload in 0.55 s


def gap_session(gap_s=50):
    # 2 s call, a long tool, then a call whose whole prompt is the first one's prompt and output
    return raw_session("X", 0, [raw_req(1000, 100, user=True, gap=int(gap_s * 1000), cat="Read"), raw_req(1100, 100, lcp=1000)])


def test_offload_and_prefetch_in_time_cost_nothing():
    sim = run([gap_session()], profile=CPU, offload_after_s=10)
    # 1100 tokens over the tool gap, then the final 1200 when the session ends (no return predicted)
    assert sim.tokens_offloaded == 1100 + 1200 and sim.prefetches == 1 and sim.transfer_wait == 0
    assert sim.turns[0].latency == pytest.approx(2 + 50 + 1)


def test_a_late_prefetch_makes_the_call_wait_for_the_rest_of_the_copy():
    # a negative margin starts the copy 0.55 - 0.3 s before the tool returns: 0.3 s still to go
    sim = run([gap_session()], profile=CPU, offload_after_s=10, prefetch_margin_s=-0.3)
    assert sim.transfer_wait == pytest.approx(0.3)
    assert sim.turns[0].latency == pytest.approx(2 + 50 + 0.3 + 1)


def test_offload_frees_memory_that_a_newcomer_would_otherwise_evict():
    raw = [gap_session(), single("Y", 10, prompt=1000, completion=100)]
    reactive = run(raw, profile=CPU, kv_tokens=1500, slots=2)
    predictive = run(raw, profile=CPU, kv_tokens=1500, slots=2, offload_after_s=10)
    assert reactive.tokens_evicted > 0 and predictive.tokens_evicted == 0


def test_short_gaps_stay_on_the_gpu():
    sim = run([gap_session(gap_s=5)], profile=CPU, offload_after_s=10)
    assert sim.tokens_offloaded == 1200 and sim.prefetches == 0        # only the cache left at the end
    assert sim.turns[0].latency == pytest.approx(2 + 5 + 1)


def test_offload_needs_a_cpu_tier():
    with pytest.raises(ValueError):
        run([gap_session()], offload_after_s=10)


def two_returning(gap_s=50):
    return [raw_session(sid, 0, [raw_req(1000, 100, user=True, gap=int(gap_s * 1000), cat="Read"),
                                 raw_req(1100, 100, lcp=1000)]) for sid in ("X", "Y")]


def test_shared_link_makes_simultaneous_reloads_queue():
    # both caches go to CPU over the 50 s tool call and are reloaded on demand (the prefetch is set too
    # late to help): each reload is 0.55 s, and on one link the second waits for the first
    kw = dict(profile=CPU, slots=2, offload_after_s=10, prefetch_margin_s=-100)
    apart = run(two_returning(), **kw)
    shared = run(two_returning(), shared_link=True, **kw)
    assert sorted(t.latency for t in apart.turns) == pytest.approx([53.55, 53.55])
    assert sorted(t.latency for t in shared.turns) == pytest.approx([53.55, 54.1])
    assert shared.link_wait == pytest.approx(0.55)


def test_a_full_cpu_tier_keeps_only_part_of_a_prefix():
    # 1500 tokens of CPU for two 1100-token prefixes: the first stored loses 700, recomputed at 1 ms each
    sim = run(two_returning(), profile=CPU, slots=2, offload_after_s=10, prefetch_margin_s=-100, cpu_tokens=1500)
    assert sorted(t.latency for t in sim.turns) == pytest.approx([53.55, 2 + 50 + 0.2 + 0.7 + 1])
    assert sim.cpu_used <= 1500


def pin_case():
    # X pauses for a 1 s tool at t=2; W ends at 2.1; Y arrives at 2.5 needing 1000 more tokens than are free.
    # LRU cuts X (idle longest), which then recomputes 1000 tokens (1 s) when its tool returns at 3.
    return [raw_session("X", 0, [raw_req(1000, 100, user=True, gap=1000, cat="Read"), raw_req(1100, 100, lcp=1000)]),
            raw_session("W", 0.1, [raw_req(1000, 100, user=True)]),
            raw_session("Y", 2.5, [raw_req(1000, 100, user=True)])]


def x_latency(sim):
    return next(t.latency for t in sim.turns if t.z == pytest.approx(4.0))


def test_continuum_pin_keeps_a_cache_through_its_tool_call():
    raw = pin_case()
    lru = run(raw, kv_tokens=2300, slots=2)
    # TTL: one past Read gap of 1 s; restoring 1100 tokens costs 1.1 s > 1 s, so the cache is pinned for 1 s
    pinned = run(raw, kv_tokens=2300, slots=2, pin="ttl", cheap=cheap_for(raw))
    assert x_latency(lru) == pytest.approx(2 + 1 + 1.0 + 1)
    assert x_latency(pinned) == pytest.approx(2 + 1 + 1)
    assert pinned.pinned_evicted == 0


def test_continuum_ttl_is_zero_when_restoring_is_cheaper_than_waiting():
    raw = pin_case()
    raw[0]["requests"][0]["gap"] = 5000                  # a 5 s tool costs more to wait for than 1.1 s of prefill
    sim = run(raw, kv_tokens=2300, slots=2, pin="ttl", cheap=cheap_for(raw))
    assert x_latency_gap(sim, 5) == pytest.approx(2 + 5 + 1.0 + 1)


def x_latency_gap(sim, gap):
    return next(t.latency for t in sim.turns if t.z == pytest.approx(2 + gap + 1))


def test_cachewise_evicts_a_session_waiting_on_its_user_before_one_in_a_tool_call():
    raw = pin_case()
    sim = run(raw, kv_tokens=2300, slots=2, eviction="cachewise", cheap=cheap_for(raw))
    assert x_latency(sim) == pytest.approx(2 + 1 + 1)


def test_fewest_tokens_first():
    # one slot busy until 1 s; A (1000-token prompt) and B (100) wait: B goes first under "fewest"
    raw = [single("hog", 0, prompt=0, completion=100), single("A", 0.2, prompt=1000, completion=100),
           single("B", 0.4, prompt=100, completion=100)]
    fcfs = run(raw, slots=1)
    fewest = run(raw, slots=1, policy="fewest")
    a, b = 1000 * 0.001 + 1.0, 100 * 0.001 + 1.0
    assert sorted(t.latency for t in fcfs.turns) == pytest.approx(sorted([1.0, 1 + a - 0.2, 1 + a + b - 0.4]))
    assert sorted(t.latency for t in fewest.turns) == pytest.approx(sorted([1.0, 1 + b - 0.4, 1 + b + a - 0.2]))


def test_shared_prefill_queues_prompts_on_one_gpu():
    """Two cold 1000-token prompts arriving together, two slots: with shared prefill the second waits
    for the first's prompt (1 s), decode still overlaps; off, both run in parallel as before."""
    from analysis.replay.sim import Config, Profile, Simulator, build_sessions
    prof = Profile(decode_ms=10.0, prefill_ms=1.0)
    raw = [{"sid": s, "start": 0.0, "requests": [{"prompt": 1000, "lcp": 0, "completion": 10, "append": False,
                                                    "user": True, "cat": "__none__", "gap": None}]} for s in "ab"]
    sessions = build_sessions(raw, prof, unit="session")
    ends = {}
    for shared in (False, True):
        cfg = Config(slots=2, kv_tokens=1e6, profile=prof, policy="fcfs", deadline_mode="proportional",
                     slack=100.0, shared_prefill=shared)
        sim = Simulator(sessions, cfg).run()
        ends[shared] = sorted(t.latency for t in sim.turns)
        assert sim.prefill_wait == (1.0 if shared else 0.0)
    assert ends[False] == pytest.approx([1.1, 1.1])
    assert ends[True] == pytest.approx([1.1, 2.1])
