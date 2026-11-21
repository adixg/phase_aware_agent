"""Sample initial-turn FreeInference DAGs from training admissions."""

import gzip
import json
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from random import Random

from .empirical_baseline import FEATURES, HORIZON, TARGETS, EmpiricalPredictor, admission_features
from .freeinference_graph import validate_graph
from .preprocess_freeinference import SPLITS, midnight


@dataclass(frozen=True)
class AdmissionDAG:
    snapshot: dict
    targets: tuple
    nodes: tuple
    edges: tuple
    branch_count: int
    source_day: str
    source_line: int


def extract_admission(graph, snapshot, label):
    """Project a normalized session to the exact admission label scope."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    root = nodes["root/r0"]
    cutoff = snapshot["cutoff_ms"]
    if len(nodes) != len(graph["nodes"]) or cutoff != root["start_ms"]:
        raise ValueError("Invalid admission root or duplicate node IDs")
    scope = {key: node for key, node in nodes.items() if node["root_turn"] == root["root_turn"]}
    edges = [edge for edge in graph["edges"] if edge["source"] in scope and edge["target"] in scope]
    if set(scope) != set(label["future_node_ids"]) or edges != [
        graph["edges"][index] for index in label["supported_edge_indices"]
    ]:
        raise ValueError("Graph scope differs from admission label")

    validate_graph(dict(nodes=scope, edges=edges, intervals=[]))
    memberships = {(edge["source"], edge["target"], edge["kind"]) for edge in edges}
    kinds = Counter(node["kind"] for node in scope.values())
    categories = Counter(node["category"] for node in scope.values() if node["kind"] == "tool")
    endpoint = max(node["end_ms"] for node in scope.values() if node["kind"] == "llm")
    targets = (kinds["llm"], kinds["tool"], endpoint - cutoff)
    if (targets != tuple(label[name] for name in TARGETS)
            or endpoint != label["observed_endpoint_ms"]
            or kinds["subagent"] != label["future_matched_subagent_count"]
            or bool(kinds["subagent"]) != label["future_matched_subagent_occurs"]
            or categories.pop(None, 0) != label["unknown_future_tool_category_count"]
            or categories != label["future_tool_category_counts"]):
        raise ValueError("Graph counts, categories, or duration differ from admission label")

    projected = []
    for node in scope.values():
        node_id, kind = node["id"], node["kind"]
        item = {field: node[field] for field in ("id", "kind", "chain")}
        if kind == "tool":
            if ((node["emitter"], node_id, "emits") not in memberships
                    or any(node[field] is not None for field in ("start_ms", "end_ms", "duration_ms"))):
                raise ValueError("Missing tool membership or unexpected tool timing")
            item.update(category=node["category"], emitter=node["emitter"])
        elif kind in ("llm", "subagent"):
            item["trace_timing"] = {
                "relative_start_ms": node["start_ms"] - cutoff,
                "relative_end_ms": node["end_ms"] - cutoff,
            }
            if kind == "subagent":
                parent = node["parent_tool"]
                if ((parent, node_id, "spawn") not in memberships
                        or (node_id, f"{node_id}/r0", "child_entry") not in memberships):
                    raise ValueError("Incomplete subagent membership")
                item.update(parent_tool=parent, completion_dependency=node["completion_dependency"])
            elif node["chain"] != "root" and node["chain"] not in scope:
                raise ValueError("Missing subagent chain")
            if kind == "llm":
                item["sibling_tool_execution_order"] = node["sibling_tool_execution_order"]
        else:
            raise ValueError(f"Unknown node type: {kind}")
        projected.append(item)
    return AdmissionDAG(snapshot, targets, tuple(projected), tuple(edges), kinds["subagent"],
                        graph["start_day"], graph["source_line"])


def load_admission_dags(source, split):
    """Stream graph/snapshot/target joins for train or validation only."""
    if split not in ("train", "validation"):
        raise ValueError("Only train and validation DAGs are supported")
    _, lower, upper = next(row for row in SPLITS if row[0] == split)
    lower_ms, upper_ms = midnight(lower), midnight(upper)
    directory = Path(source) / split
    tables = ("graphs", "snapshots", "targets")
    names = [
        {path.name.removesuffix(f".{table}.jsonl.gz")
         for path in directory.glob(f"*.{table}.jsonl.gz")}
        for table in tables
    ]
    if not names[0] or any(stems != names[0] for stems in names[1:]):
        raise ValueError(f"Unpaired shards in {split}")

    def admissions(snapshots, targets):
        for left, right in zip_longest(snapshots, targets):
            if left is None or right is None:
                raise ValueError("Unpaired snapshot and target")
            snapshot, target = json.loads(left), json.loads(right)
            if (snapshot["key"], snapshot["ordinal"]) != (target["key"], target["ordinal"]):
                raise ValueError("Snapshot and target join mismatch")
            if snapshot["ordinal"] == 0:
                yield snapshot, target

    seen = set()
    for stem in sorted(names[0]):
        with ExitStack() as stack:
            graphs, snapshots, targets = [
                stack.enter_context(gzip.open(directory / f"{stem}.{table}.jsonl.gz", "rt"))
                for table in tables
            ]
            for line, pair in zip_longest(graphs, admissions(snapshots, targets)):
                if line is None or pair is None:
                    raise ValueError("Unpaired graph and admission")
                graph = json.loads(line)
                snapshot, target = pair
                key, cutoff = snapshot["key"], snapshot["cutoff_ms"]
                if graph["key"] != key or key in seen:
                    raise ValueError("Graph join mismatch or duplicate key")
                if not lower_ms <= cutoff <= target["observed_endpoint_ms"] < upper_ms:
                    raise ValueError("Admission or target outside its temporal split")
                admission_features(snapshot, HORIZON)
                seen.add(key)
                yield extract_admission(graph, snapshot, target)


def graph_payload(template, metadata_origin):
    """Serialize a fresh copy with gateway times attributed to their trace."""
    return dict(
        nodes=deepcopy(list(template.nodes)),
        edges=[dict(edge) for edge in template.edges],
        matched_subagent_count=template.branch_count,
        trace_metadata=dict(origin=metadata_origin, timing_reference="trace_admission",
                            observed_duration_ms=template.targets[-1]),
    )


class FutureDAGGenerator:
    """Draw whole validated DAGs with the baseline's empirical weighting."""

    def __init__(self, source, baseline_artifact, *, conditional=True):
        config = json.loads(Path(baseline_artifact).read_text())
        if (config["schema_version"] != 1 or config["horizon"] != HORIZON
                or tuple(config["features"]) != FEATURES or tuple(config["targets"]) != TARGETS):
            raise ValueError("Unsupported empirical forecast artifact")
        self.templates = tuple(load_admission_dags(source, "train"))
        self.training_keys = frozenset(row.snapshot["key"] for row in self.templates)
        self.model = EmpiricalPredictor(
            [(admission_features(row.snapshot, HORIZON), (*row.targets, index))
             for index, row in enumerate(self.templates)],
            prompt_edges=config["prompt_edges"], strength=config["strength"],
            conditional=conditional,
        )

    def distribution(self, snapshot):
        """Return atoms in TARGETS order followed by a unique template index."""
        return self.model.distribution(admission_features(snapshot, HORIZON))

    def predict_dags(self, snapshot, num_samples=10, seed=42):
        """Draw with replacement. Probability is mass of the individual template."""
        if type(num_samples) is not int or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        atoms, _, support = self.distribution(snapshot)
        weights = [mass for _, mass in atoms]
        draws = Random(seed).choices(atoms, weights=weights, k=num_samples)
        return [
            dict(
                graph_payload(self.templates[atom[-1]], "training_template"),
                horizon=HORIZON, cutoff_ms=snapshot["cutoff_ms"],
                template_index=atom[-1], template_probability=mass, support=dict(support),
                forecast=dict(zip(TARGETS, atom[:-1])),
            )
            for atom, mass in draws
        ]
