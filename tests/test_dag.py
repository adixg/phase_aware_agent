"""DAG construction from raw FreeInference sessions, on small hand-made cases."""

from collections import Counter

from analysis.dag.build import WEEKS, session_turns, strip_blocks, to_graph


def call(i, name="Read", lat=1000, sub=None):
    return {"call_index": i, "name": name, "category": "Subagent" if sub else "File", "outcome": "ok",
            "result_tokens": 10, "latency_ms": lat, "subagent": sub}


def req(start, end, tools=(), trigger="tool", turn=0, completion=50):
    return {"step_trigger": trigger, "turn_index": turn, "start_ms": start, "end_ms": end, "ttft_ms": 100,
            "provider_tokens": {"prompt": 1000, "completion": completion, "cached": 0},
            "finish_reason": "tool_calls" if tools else "stop", "tool_calls": list(tools)}


def session(requests):
    return {"session_id": "s1", "user_id": "u", "harness": "h", "day": "2026-08-23", "requests": requests}


def test_user_trigger_starts_a_new_turn_and_idle_is_kept():
    s = session([req(0, 1000, [call(0)], "user"), req(2000, 3000), req(60_000, 61_000, [], "user", turn=1)])
    turns = session_turns(s, Counter())
    assert [len(t["rounds"]) for t in turns] == [2, 1]
    assert turns[0]["idle_before_ms"] is None and turns[1]["idle_before_ms"] == 60_000 - 3000


def test_round_wall_time_is_the_gap_to_the_next_request():
    s = session([req(0, 1000, [call(0, lat=1500), call(1, lat=1500)], "user"), req(2500, 3000)])
    r = session_turns(s, Counter())[0]["rounds"]
    assert r[0]["round_ms"] == 1500 and r[1]["round_ms"] is None
    assert len(r[0]["tools"]) == 2


def test_fan_out_joins_before_the_next_request():
    s = session([req(0, 1000, [call(0), call(1), call(2)], "user"), req(2000, 3000)])
    nodes, edges = to_graph(session_turns(s, Counter())[0])
    assert {e for e in edges} == {("L0", "L0.T0"), ("L0", "L0.T1"), ("L0", "L0.T2"),
                                  ("L0.T0", "L1"), ("L0.T1", "L1"), ("L0.T2", "L1")}
    assert sum(n["kind"] == "llm" for n in nodes) == 2 and sum(n["kind"] == "tool" for n in nodes) == 3


def sub_session(start, end, n=2):
    return {"session_id": "child-1234", "user_id": "u", "harness": "h", "start_ms": start, "end_ms": end,
            "requests": [req(start + 100 * i, start + 100 * i + 50, [], "user") for i in range(n)]}


def test_subagent_is_a_branch_with_no_join_and_flags_whether_it_outlives_the_parent():
    long_child = call(0, "delegate_task", 4000, sub_session(500, 90_000))
    s = session([req(0, 1000, [long_child], "user"), req(5000, 6000)])
    stats = Counter()
    turn = session_turns(s, stats)[0]
    sub = turn["rounds"][0]["subs"][0]
    assert sub["outlives_next"] is True and len(sub["rounds"]) == 2
    assert stats["subagent_calls"] == 1 and stats["subagent_requests"] == 2
    nodes, edges = to_graph(turn)
    assert ("L0.T0", "L0.T0.Schild-.L0") in edges                  # spawn edge
    assert not any(b == "L1" and a.startswith("L0.T0.S") for a, b in edges)   # no edge back to the parent
    assert ("L0.T0", "L1") in edges                               # the parent moves on by itself


def test_short_subagent_does_not_outlive_the_parents_next_request():
    s = session([req(0, 1000, [call(0, sub=sub_session(500, 900))], "user"), req(5000, 6000)])
    assert session_turns(s, Counter())[0]["rounds"][0]["subs"][0]["outlives_next"] is False


def test_negative_round_time_is_counted_not_hidden():
    stats = Counter()
    session_turns(session([req(0, 1000, [call(0)], "user"), req(900, 2000)]), stats)
    assert stats["negative_round_ms"] == 1


def test_turn_index_disagreement_is_counted():
    stats = Counter()
    session_turns(session([req(0, 1000, [], "user", turn=0), req(2000, 3000, [], "user", turn=5)]), stats)
    assert stats["turn_index_mismatch"] == 1


def test_block_ids_are_cut_out():
    line = b'{"a": 1, "block_ids": [1, 2, 3], "tool_calls": [], "block_ids": [4]}'
    assert strip_blocks(line) == b'{"a": 1, "block_ids": [], "tool_calls": [], "block_ids": []}'


def test_weeks_do_not_overlap_except_the_one_shared_day():
    assert WEEKS["train"] != WEEKS["test"]


def test_census_overlap_and_quantiles():
    from analysis.dag.census import max_overlap, quantiles
    assert max_overlap([(0, 10), (5, 15), (12, 20)]) == 2
    assert max_overlap([(0, 10), (10, 20)]) == 1          # touching is not overlapping
    assert quantiles(list(range(100))) == [50, 90, 99]
