"""Regression checks for admission scope and template probability."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from .empirical_baseline import FEATURES, HORIZON, TARGETS
from .freeinference_graph import make_examples, reconstruct
from .future_dag import FutureDAGGenerator, extract_admission
from .test_freeinference_preprocess import request, trace


def fixture():
    raw = trace()
    raw["end_ms"] = 40
    later = request(2, 31, 40)
    later.update(turn_index=1, step_trigger="user", link_reason="assistant_message")
    raw["requests"].append(later)
    graph = reconstruct(raw)
    snapshot, label = next(make_examples(graph))
    snapshot["key"] = "train-key"
    graph.update(nodes=list(graph["nodes"].values()), start_day="2026-08-23", source_line=1)
    return graph, snapshot, label


class FutureDAGTests(TestCase):
    def test_projection_preserves_branch_and_excludes_later_turn(self):
        graph, snapshot, label = fixture()
        row = extract_admission(graph, snapshot, label)
        self.assertEqual(row.targets, (3, 2, 20))
        self.assertEqual(row.branch_count, 1)
        self.assertNotIn("root/r2", {node["id"] for node in row.nodes})
        self.assertEqual(
            [(edge["kind"], edge["provenance"]) for edge in row.edges if edge["provenance"] == "inferred"],
            [("spawn", "inferred")],
        )
        self.assertTrue(all("trace_timing" not in node for node in row.nodes if node["kind"] == "tool"))
        label["supported_edge_indices"].pop()
        with self.assertRaisesRegex(ValueError, "scope differs"):
            extract_admission(graph, snapshot, label)

    def test_probabilities_are_per_template_and_samples_are_isolated(self):
        graph, snapshot, label = fixture()
        left = extract_admission(graph, snapshot, label)
        other = deepcopy(snapshot)
        other["key"] = "other-train-key"
        other["features"]["model"] = "other"
        right = extract_admission(graph, other, label)
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.json"
            artifact.write_text(json.dumps(dict(
                schema_version=1, horizon=HORIZON, features=FEATURES, targets=TARGETS,
                prompt_edges=[], strength=1,
            )))
            with patch("prediction.predictor.future_dag.load_admission_dags",
                       return_value=iter((left, right))) as load:
                generator = FutureDAGGenerator(directory, artifact)
                load.assert_called_once_with(directory, "train")

        query = deepcopy(snapshot)
        query["key"] = "validation-key"
        atoms, _, support = generator.distribution(query)
        self.assertEqual({row[-1]: weight for row, weight in atoms}, {0: 0.875, 1: 0.125})
        self.assertEqual(support["group_depth"], 3)
        draws = generator.predict_dags(query, num_samples=3)
        query["features"]["completed_root_llms"] = 100
        self.assertEqual(draws, generator.predict_dags(query, num_samples=3))
        draws[0]["nodes"][0]["trace_timing"]["relative_end_ms"] = 0
        self.assertEqual(generator.predict_dags(query, num_samples=1)[0]["nodes"][0]["trace_timing"],
                         {"relative_start_ms": 0, "relative_end_ms": 4})
        query["feature_available_ms"]["model"] = query["cutoff_ms"] + 1
        with self.assertRaisesRegex(ValueError, "unavailable"):
            generator.predict_dags(query)
