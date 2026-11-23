"""Online events: A (request admitted) and R (response completed), on hand-made turns."""

import pytest

from analysis.dag.dataset import UserHistory, features, prefix_view
from analysis.dag.online import (COLUMNS, EVENTS, FEATURES, audit, cutoff_of, labels_at, response_view, rows_of,
                                 scramble, to_snapshot, view)
from tests.test_dataset import call, make, req, three_rounds


def hist(turns):
    return UserHistory([turns])


def feat(turns, k, e, source=None):
    """Features of event (k, e); `source` supplies the view (a scrambled copy), the history stays the real one."""
    return features(view(source or turns, 0, k, e), hist(turns).before("u1", cutoff_of(turns, 0, k, e)))


def test_a_knows_the_finished_rounds_and_r_adds_the_response_but_not_what_its_tools_did():
    turns = three_rounds()                                   # request 1 calls Edit (an error) and a second tool
    a, r = feat(turns, 1, "A"), feat(turns, 1, "R")
    assert a["prev1_width"] == 1 and a["active_s"] == 3.0 and a["prev1_round_s"] == 2.0   # request 0's round, 2 s
    assert r["prev1_width"] == 2 and r["prev1_cat"] == "Edit" and r["active_s"] == 4.0    # response 1 ended at 4 s
    assert r["prev1_completion"] == 50 and r["prev1_finish"] == "tool_calls"
    assert r["prev1_errors"] is None and r["prev1_result_tokens"] is None and r["prev1_round_s"] is None   # not yet known
    assert r["prev2_round_s"] == 2.0 and r["prev2_errors"] == 0                            # the round before is complete
    assert r["cat_Edit"] == 1 and r["prefix_fanout_rounds"] == 1 and r["prefix_max_width"] == 2
    assert r["prefix_errors"] == 0                                                         # the error is not seen yet


def test_information_grows_along_the_chain():
    turns = three_rounds()
    a1, r1, a2 = (feat(turns, 1, "A"), feat(turns, 1, "R"), feat(turns, 2, "A"))
    assert a1["attained_work_s"] < r1["attained_work_s"] == a2["attained_work_s"]   # R_k and A_{k+1} share the work seen
    assert a2["prev1_errors"] == 1 and r1["prev1_errors"] is None                  # ... and A_{k+1} adds what the tools did


def test_labels_by_hand_for_both_events():
    t = three_rounds()[0]
    ideal = [r["_ideal"] for r in t["rounds"]]
    a1, r1 = labels_at(t, 1, "A"), labels_at(t, 1, "R")
    assert a1["rem_after"] == r1["rem_after"] == 1 and not a1["is_last"]
    assert a1["stall_s"] == r1["stall_s"] == 1.0
    assert a1["rem_wall_s"] == pytest.approx(3.0) and r1["rem_wall_s"] == pytest.approx(2.0)        # from 3 s and from 4 s to the turn's end at 6 s
    assert a1["rem_work_s"] == pytest.approx(ideal[1] + ideal[2]) and r1["rem_work_s"] == pytest.approx(ideal[2])
    assert a1["rem_path_s"] == pytest.approx(ideal[1] + 1.0 + ideal[2]) and r1["rem_path_s"] == pytest.approx(1.0 + ideal[2])
    last = labels_at(t, 2, "R")
    assert last["is_last"] and last["stall_s"] is None and last["rem_work_s"] == 0 and last["rem_wall_s"] == pytest.approx(0.0)


def test_one_row_per_event_and_request_with_a_shared_schema():
    turns = three_rounds()
    rows = list(rows_of(turns, hist(turns)))
    assert len(rows) == 6 and [(r["step"], r["event"]) for r in rows] == [(k, e) for k in range(3) for e in EVENTS]
    assert all(set(r) == set(COLUMNS) for r in rows)
    assert [r["cutoff_ms"] for r in rows] == [0, 1000, 3000, 4000, 5000, 6000]
    assert rows[0]["resp_gen_s"] is None and rows[1]["resp_gen_s"] == 1.0


def test_features_do_not_move_when_the_future_is_scrambled_for_either_event():
    turns = three_rounds()
    for k in range(3):
        for e in EVENTS:
            assert feat(turns, k, e) == feat(turns, k, e, scramble(turns, 0, k, e))
    assert audit([turns], hist(turns), 100) == 0


def test_a_scrambled_response_does_change_the_r_features():
    # the audit must be able to fail: R features do depend on the response itself
    turns = three_rounds()
    cheat = scramble(turns, 0, 1, "A")                      # an A-style scramble garbles the response
    assert feat(turns, 1, "R", cheat) != feat(turns, 1, "R")


def test_snapshot_follows_the_offline_schema_and_nothing_is_available_after_its_cutoff():
    turns = three_rounds()
    snaps = [to_snapshot(r) for r in rows_of(turns, hist(turns))]
    assert [s["event"] for s in snaps] == ["admission", "root_response_completed", "request_admitted", "root_response_completed",
                                           "request_admitted", "root_response_completed"]
    assert [s["ordinal"] for s in snaps] == [0, 1, 2, 3, 4, 5] and len({s["key"] for s in snaps}) == 1
    for s in snaps:
        assert s["horizon"] == "current_user_turn" and set(s["feature_available_ms"]) == set(s["features"])
        assert all(t <= s["cutoff_ms"] for t in s["feature_available_ms"].values())
    assert snaps[0]["features"]["harness"] == "h"


def test_response_view_extends_the_prefix_view_without_touching_it():
    turns = three_rounds()
    before = prefix_view(turns, 0, 1)
    response_view(turns, 0, 1)
    assert prefix_view(turns, 0, 1) == before                 # building R did not mutate A's view


def test_history_carried_over_from_an_earlier_week_counts_only_turns_that_ended_before():
    earlier = make([req(0, 1000, 1000, trigger="user")], sid="old")          # one turn of user u1, ended at 1000
    turns = make([req(5000, 6000, 1000, trigger="user")], sid="new")
    h = UserHistory([earlier, turns])
    assert features(prefix_view(turns, 0, 0), h.before("u1", 5000))["user_prior_turns"] == 1
    assert features(prefix_view(turns, 0, 0), h.before("u1", 500))["user_prior_turns"] == 0      # not yet ended
    assert features(prefix_view(turns, 0, 0), UserHistory([turns]).before("u1", 5000))["user_prior_turns"] == 0
