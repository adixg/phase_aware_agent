"""Prefix-level rows: labels worked out by hand, and the guarantee that features cannot see the future."""

import copy
from collections import Counter

import pytest

from analysis.dag.build import session_turns
from analysis.dag.dataset import (COLUMNS, FEATURES, UserHistory, annotate_session, audit, features, labels, prefix_view,
                                  rows_of, scramble, service_s)


def call(i, cat="File", outcome="ok", result=10, lat=1000):
    return {"call_index": i, "name": "t", "category": cat, "outcome": outcome, "result_tokens": result,
            "latency_ms": lat, "subagent": None}


def req(start, end, prompt, completion=50, tools=(), trigger="tool", turn=0):
    return {"step_trigger": trigger, "turn_index": turn, "start_ms": start, "end_ms": end, "ttft_ms": 10,
            "provider_tokens": {"prompt": prompt, "completion": completion, "cached": 0},
            "finish_reason": "tool_calls" if tools else "stop", "tool_calls": list(tools)}


def make(requests, sid="s1", user="u1"):
    raw = {"session_id": sid, "user_id": user, "harness": "h", "day": "2026-08-23", "requests": requests}
    turns = session_turns(raw, Counter())
    annotate_session(turns)
    return turns


def three_rounds():
    # request 0 -> 1 s of tool, request 1 -> two parallel tools, request 2 answers
    return make([req(0, 1000, 1000, tools=[call(0, lat=2000, cat="Read")], trigger="user"),
                 req(3000, 4000, 1250, tools=[call(0, lat=1000, cat="Edit", outcome="error"), call(1, lat=1000)]),
                 req(5000, 6000, 1400)])


def test_labels_by_hand():
    t = three_rounds()[0]
    lab = labels(t)
    assert [x["rem_rounds"] for x in lab] == [3, 2, 1]
    assert [x["rem_wall_s"] for x in lab] == pytest.approx([6.0, 3.0, 1.0])
    assert [x["next_stall_s"] for x in lab] == [2.0, 1.0, None]
    assert [x["next_width"] for x in lab] == [1, 2, 0]
    assert [x["next_cat"] for x in lab] == ["Read", "Edit", "__none__"]
    ideal = [r["_ideal"] for r in t["rounds"]]
    assert lab[2]["rem_path_s"] == pytest.approx(ideal[2])                      # last request: its own work, no stall
    assert lab[0]["rem_path_s"] - lab[1]["rem_path_s"] == pytest.approx(ideal[0] + 2.0)
    assert lab[0]["rem_work_s"] == pytest.approx(sum(ideal))


def test_new_tokens_are_what_the_previous_context_does_not_cover():
    t = three_rounds()[0]["rounds"]
    assert t[0]["_new"] == 1000                       # first request of the session: the whole prompt
    assert t[1]["_new"] == 1250 - (1000 + 50)         # prompt minus previous prompt and completion
    assert t[0]["_ideal"] == pytest.approx(service_s(1000, 50))


def test_a_later_turn_inherits_the_context_of_the_last_request_before_it():
    turns = make([req(0, 1000, 1000, trigger="user"), req(60_000, 61_000, 1200, trigger="user", turn=1)])
    assert turns[1]["rounds"][0]["_new"] == 1200 - 1050


def test_first_request_has_an_empty_prefix_and_a_later_one_sees_only_the_past():
    turns = three_rounds()
    h = UserHistory([turns])
    f0 = features(prefix_view(turns, 0, 0), h.before("u1", 0))
    assert f0["step"] == 0 and f0["active_s"] == 0 and f0["prev1_width"] is None and f0["attained_work_s"] == 0
    f2 = features(prefix_view(turns, 0, 2), h.before("u1", 5000))
    assert f2["step"] == 2 and f2["active_s"] == 5.0 and f2["prev1_width"] == 2 and f2["prev2_cat"] == "Read"
    assert f2["prev1_errors"] == 1 and f2["prefix_errors"] == 1 and f2["prefix_max_width"] == 2
    assert f2["prefix_stall_s"] == pytest.approx(3.0) and f2["cat_Read"] == 1 and f2["cat_Edit"] == 1


def test_features_do_not_move_when_the_future_is_scrambled():
    turns = three_rounds()
    h = UserHistory([turns])
    for k in range(3):
        real = features(prefix_view(turns, 0, k), h.before("u1", turns[0]["rounds"][k]["start"]))
        fake_turns = scramble(turns, 0, k)
        fake = features(prefix_view(fake_turns, 0, k), h.before("u1", turns[0]["rounds"][k]["start"]))
        assert real == fake
    assert audit([turns], h, 50) == 0


def test_the_audit_can_see_a_leak():
    # a feature that peeked at the current request's own completion would differ once it is scrambled
    turns = three_rounds()
    h = UserHistory([turns])
    leaky = copy.deepcopy(prefix_view(turns, 0, 1))
    leaky["cur_prompt"] = turns[0]["rounds"][1]["completion"]               # not what the real view holds
    scrambled = prefix_view(scramble(turns, 0, 1), 0, 1)
    scrambled["cur_prompt"] = scramble(turns, 0, 1)[0]["rounds"][1]["completion"]
    assert features(leaky, h.before("u1", 3000))["prompt"] != features(scrambled, h.before("u1", 3000))["prompt"]
    # and a change to the past does move the features
    past = copy.deepcopy(turns)
    past[0]["rounds"][0]["tools"][0]["cat"] = "Write"
    annotate_session(past)
    assert features(prefix_view(past, 0, 1), h.before("u1", 3000)) != features(prefix_view(turns, 0, 1), h.before("u1", 3000))


def test_user_history_counts_only_turns_that_ended_before_the_request():
    a = make([req(0, 10_000, 1000, trigger="user")], sid="a")                     # ends at 10 s
    b = make([req(5_000, 6_000, 1000, trigger="user")], sid="b")                  # a concurrent session of the same user
    c = make([req(20_000, 21_000, 1000, trigger="user")], sid="c")
    h = UserHistory([a, b, c])
    assert h.before("u1", 5_000)["n"] == 0            # a is still running
    assert h.before("u1", 10_000)["n"] == 1           # only b (ended at 6 s); a ends exactly now, not before
    assert h.before("u1", 20_000)["n"] == 2
    assert h.before("nobody", 20_000)["n"] == 0


def test_rows_have_every_column_and_weights_sum_to_one_per_session():
    turns = three_rounds() + make([req(0, 1, 500, trigger="user")], sid="s2")
    rows = list(rows_of(three_rounds(), UserHistory([turns])))
    assert all(set(r) == set(COLUMNS) for r in rows) and len(rows) == 3
    assert sum(r["w_session"] for r in rows) == pytest.approx(1.0)
    assert sum(r["w_turn"] for r in rows) == pytest.approx(1.0)


def test_feature_and_label_names_do_not_collide():
    assert not set(FEATURES) & {"rem_rounds", "rem_work_s", "rem_path_s", "rem_wall_s", "next_stall_s"}
