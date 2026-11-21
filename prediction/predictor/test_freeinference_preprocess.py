"""Regression checks for time cutoffs and trace dependency semantics."""

import copy
import unittest

from prediction.predictor.freeinference_graph import make_examples, reconstruct
from prediction.predictor.preprocess_freeinference import midnight, split_for


def request(step, start, end, tools=()):
    return dict(
        step=step, turn_index=0, step_trigger="tool" if step else "user",
        link_reason="tool_call_id" if step else None, start_ms=start, end_ms=end,
        model="sample", provider_tokens={"prompt": 50, "completion": 4},
        role_tokens={"system": 10, "user": 20, "assistant": 0,
                     "tool": 0, "tool_definitions": 5},
        tool_calls=list(tools),
    )


def trace():
    calls = [dict(call_index=index, category="Read", latency_ms=6)
             for index in range(2)]
    calls[1]["subagent"] = dict(harness="example", start_ms=18, end_ms=30,
                                requests=[request(0, 18, 30)])
    return dict(
        harness="example", start_ms=10, end_ms=28,
        requests=[request(0, 10, 14, calls), request(1, 20, 28)],
    )


class PreprocessingTests(unittest.TestCase):
    def test_batch_gap_and_unknown_sibling_order(self):
        graph = reconstruct(trace())
        self.assertEqual(graph["intervals"][0]["elapsed_ms"], 6)
        self.assertEqual(graph["intervals"][0]["result_dependencies"], "unknown_subset")
        self.assertEqual(graph["intervals"][0]["sibling_execution_order"], "unknown")
        self.assertEqual(graph["quality"]["tool_latency_formula_mismatch"], 0)
        self.assertEqual(graph["nodes"]["root/r0/t0"]["released_latency_ms"], 6)
        self.assertIsNone(graph["nodes"]["root/r0/t0"]["duration_ms"])
        self.assertEqual(
            [(edge["kind"], edge["provenance"]) for edge in graph["edges"]],
            [("emits", "observed"), ("emits", "observed"), ("spawn", "inferred"),
             ("child_entry", "observed"), ("continuation", "observed")],
        )

    def test_future_mutations_do_not_change_admission_or_first_update(self):
        raw = trace()
        changed = copy.deepcopy(raw)
        changed["requests"][0]["provider_tokens"]["prompt"] = 999
        changed["requests"][1]["model"] = "future-model"
        changed["requests"][1]["provider_tokens"]["completion"] = 888
        changed["requests"][1]["tool_calls"] = [dict(call_index=0, category="Search")]
        base = list(make_examples(reconstruct(raw), updates=True))
        updated = list(make_examples(reconstruct(changed), updates=True))
        self.assertEqual(base[0][0], updated[0][0])
        self.assertNotEqual(base[0][1], updated[0][1])
        self.assertEqual(base[0][0]["features"]["initial_prompt_estimate"], 35)
        self.assertEqual(base[0][1]["remaining_llm_count"], 3)
        self.assertEqual(base[0][1]["remaining_observed_duration_ms"], 20)
        self.assertEqual(base[0][1]["future_matched_subagent_count"], 1)
        self.assertIsNone(base[0][1]["actual_subagent_branch_occurs"])

        changed["requests"][0]["provider_tokens"]["prompt"] = 50
        updated = list(make_examples(reconstruct(changed), updates=True))
        self.assertEqual(base[1][0], updated[1][0])
        self.assertNotEqual(base[1][1], updated[1][1])
        self.assertEqual(base[1][0]["features"]["emitted_root_tool_count"], 2)
        self.assertEqual(base[1][0]["features"]["closed_root_tool_phase_elapsed_ms"], [])
        self.assertEqual(base[2][0]["features"]["closed_root_tool_phase_elapsed_ms"], [6])

    def test_completion_snapshot_exists_without_future_work(self):
        raw = trace()
        raw["requests"][0]["tool_calls"][1].pop("subagent")
        examples = list(make_examples(reconstruct(raw), updates=True))
        self.assertEqual(examples[-1][0]["cutoff_ms"], 28)
        self.assertEqual(examples[-1][1]["remaining_llm_count"], 0)
        self.assertEqual(examples[-1][1]["remaining_observed_duration_ms"], 0)

    def test_overlap_is_preserved_but_elapsed_is_unknown(self):
        raw = trace()
        raw["requests"][1]["start_ms"] = 13
        for tool in raw["requests"][0]["tool_calls"]:
            tool["latency_ms"] = -1
        graph = reconstruct(raw)
        self.assertEqual(graph["intervals"][0]["signed_gap_ms"], -1)
        self.assertIsNone(graph["intervals"][0]["elapsed_ms"])

    def test_utc_boundaries_purge_label_crossings(self):
        validation = midnight("2026-08-30")
        test = midnight("2026-09-02")
        self.assertEqual(split_for(validation - 2, validation - 1), ("train", True))
        self.assertEqual(split_for(validation - 1, validation), ("train", False))
        self.assertEqual(split_for(validation, test), ("validation", False))
        self.assertEqual(split_for(validation, validation + 2, validation - 1), ("validation", False))
        self.assertEqual(split_for(test, test + 1), ("test", True))
        self.assertEqual(split_for(test, midnight("2026-09-06")), ("test", False))


if __name__ == "__main__":
    unittest.main()
