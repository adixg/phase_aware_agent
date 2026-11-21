"""Compare conditional and uniform training-template DAG forecasts on validation."""

import argparse
import json
from collections import Counter, defaultdict
from graphlib import TopologicalSorter
from pathlib import Path
from statistics import fmean

from .empirical_baseline import FEATURES, QUANTILES, TARGETS
from .future_dag import FutureDAGGenerator, graph_payload, load_admission_dags
from .preprocess_freeinference import ROOT


def structure(nodes, edges):
    """A typed, categorized topology in the normalized trace's local ID scheme."""
    return (
        tuple(sorted((node["id"], node["kind"], node.get("category")) for node in nodes)),
        tuple(sorted((edge["source"], edge["target"], edge["kind"], edge["provenance"])
                     for edge in edges)),
    )


def check_sample(sample, template):
    nodes, edges = sample["nodes"], sample["edges"]
    if (sample["forecast"] != dict(zip(TARGETS, template.targets))
            or sample["matched_subagent_count"] != template.branch_count
            or nodes != list(template.nodes) or edges != list(template.edges)):
        raise ValueError("Sample differs from training template")
    ids = {node["id"] for node in nodes}
    if len(ids) != len(nodes) or any(
        edge["source"] not in ids or edge["target"] not in ids for edge in edges
    ):
        raise ValueError("Invalid sampled graph endpoints")
    inferred = Counter()
    for retain_inferred in (True, False):
        sorter = TopologicalSorter({node_id: () for node_id in ids})
        for edge in edges:
            if retain_inferred and edge["provenance"] == "inferred":
                inferred[edge["kind"]] += 1
            if retain_inferred or edge["provenance"] == "observed":
                sorter.add(edge["target"], edge["source"])
        tuple(sorter.static_order())
    counts = Counter(node["kind"] for node in nodes)
    if (counts["llm"], counts["tool"], counts["subagent"]) != (
        *template.targets[:2], template.branch_count
    ) or any("trace_timing" in node for node in nodes if node["kind"] == "tool"):
        raise ValueError("Invalid sampled counts or tool timing")
    return inferred


def evaluate(generator, validation, num_samples, seed):
    shapes = [structure(row.nodes, row.edges) for row in generator.templates]
    metrics = defaultdict(list)
    bins = defaultdict(list)
    inferred = Counter()
    depths = Counter()
    sampled_branch = []
    for index, actual in enumerate(validation):
        atoms, marginals, support = generator.distribution(actual.snapshot)
        depths[support["group_depth"]] += 1
        for name, marginal, observed in zip(TARGETS, marginals, actual.targets):
            low, median, high = (marginal.quantile(q) for q in QUANTILES)
            metrics[f"{name}.crps"].append(marginal.crps(observed))
            metrics[f"{name}.median_mae"].append(abs(median - observed))
            metrics[f"{name}.coverage_80"].append(low <= observed <= high)
            metrics[f"{name}.exact_mean"].append(marginal.weighted_sum[-1])
            metrics[f"{name}.observed_mean"].append(observed)

        probabilities = defaultdict(float)
        branch_probability = 0.0
        for atom, mass in atoms:
            template_index = atom[-1]
            probabilities[shapes[template_index]] += mass
            if generator.templates[template_index].branch_count:
                branch_probability += mass
        branch_actual = bool(actual.branch_count)
        metrics["branch_probability"].append(branch_probability)
        metrics["branch_observed_frequency"].append(branch_actual)
        metrics["branch_brier"].append((branch_probability - branch_actual) ** 2)
        bins[min(9, int(10 * branch_probability))].append((branch_probability, branch_actual))
        metrics["structural_pair_difference_probability"].append(
            1 - sum(mass ** 2 for mass in probabilities.values())
        )
        metrics["exact_observed_structure_mass"].append(
            probabilities.get(structure(actual.nodes, actual.edges), 0)
        )

        draws = generator.predict_dags(actual.snapshot, num_samples, seed + index)
        for draw in draws:
            inferred.update(check_sample(draw, generator.templates[draw["template_index"]]))
            sampled_branch.append(bool(draw["matched_subagent_count"]))
        for name in TARGETS:
            metrics[f"{name}.sample_mean"].append(fmean(draw["forecast"][name] for draw in draws))
        metrics["distinct_sampled_structures"].append(
            len({shapes[draw["template_index"]] for draw in draws})
        )

    return dict(
        metrics={name: fmean(values) for name, values in metrics.items()},
        draws_checked=len(validation) * num_samples,
        sampled_branch_frequency=fmean(sampled_branch),
        inferred_edges_in_draws=dict(inferred),
        support_depths=dict(depths),
        branch_reliability=[
            dict(lower=bucket / 10, upper=(bucket + 1) / 10, roots=len(rows),
                 predicted=fmean(p for p, _ in rows), observed=fmean(y for _, y in rows))
            for bucket, rows in sorted(bins.items())
        ],
    )


def mermaid(graph):
    ids = {node["id"]: f"n{i}" for i, node in enumerate(graph["nodes"])}
    lines = ["flowchart TD"]
    for node in graph["nodes"]:
        suffix = node["id"].rsplit("/", 1)[-1]
        label = {
            "llm": f"LLM {suffix}",
            "tool": f"{suffix} {node.get('category') or 'Unknown category'}",
            "subagent": "Subagent",
        }[node["kind"]]
        # Mermaid entity syntax keeps quoted labels safe.
        label = label.replace("&", "&amp;").replace('"', "#quot;")
        lines.append(f'    {ids[node["id"]]}["{label}"]')
    for edge in graph["edges"]:
        kind = edge["kind"]
        arrow = f"-->|{kind}|" if edge["provenance"] == "observed" else f"-. {kind} .->"
        lines.append(f'    {ids[edge["source"]]} {arrow} {ids[edge["target"]]}')
    return "\n".join(lines)


def example_report(generator, actual, seed):
    features = {name: actual.snapshot["features"][name] for name in FEATURES}
    draws = generator.predict_dags(actual.snapshot, 3, seed)
    observed = dict(graph_payload(actual, "validation_observation"),
                    observed=dict(zip(TARGETS, actual.targets)))
    payload = dict(features_at_admission=features, source_day=actual.source_day,
                   source_line=actual.source_line, seed=seed, draws=draws, observed=observed)
    lines = [
        "# Initial-turn DAG forecast",
        "",
        f"Validation admission on {actual.source_day}, shard line {actual.source_line}.",
        "",
        f"At t=0 the sampler sees harness `{features['harness']}`, model "
        f"`{features['model']}`, and {features['initial_prompt_estimate']:,} estimated input tokens.",
        "The initial LLM request is admitted. Its response, tool calls, and subagents are future outcomes.",
        f"Three draws with replacement use seed {seed}. Solid arrows show released links. "
        "Dotted arrows show inferred spawn or single-candidate feedback links.",
    ]
    for index, graph in enumerate([*draws, observed], 1):
        forecast = graph["forecast"] if index <= 3 else graph["observed"]
        llms, tools, duration = (forecast[name] for name in TARGETS)
        branches = graph["matched_subagent_count"]
        branch_label = "branch" if branches == 1 else "branches"
        lines.extend(["", f"## Sample {index}" if index <= 3 else "## Validation observation", ""])
        if index <= 3:
            lines.append(f"Training template {graph['template_index']} has "
                         f"{graph['template_probability']:.3%} probability per draw.")
        lines.extend([
            f"{llms} LLM requests, {tools} tool calls, {branches} "
            f"matched subagent {branch_label}, and {duration / 1000:.3f} s of observed gateway duration "
            "in the source trace.",
            "", "```mermaid", mermaid(graph), "```",
        ])
    lines.extend([
        "", "Each eligible training root initially has equal mass globally. "
        "Each matching harness, harness/model, and harness/model/prompt group mixes uniform local "
        "mass with its parent in proportion to its size and the baseline smoothing strength. "
        "A template can receive mass from several levels.", "",
    ])
    return payload, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "preprocessed")
    parser.add_argument("--baseline", type=Path, default=ROOT / "empirical-baseline/model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "future-dag")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--example-seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1 or args.output.exists():
        parser.error("Use a positive sample count and a fresh output directory")

    conditional = FutureDAGGenerator(args.source, args.baseline)
    uniform = FutureDAGGenerator(args.source, args.baseline, conditional=False)
    validation = tuple(load_admission_dags(args.source, "validation"))
    if not validation or conditional.training_keys.intersection(row.snapshot["key"] for row in validation):
        raise ValueError("Missing validation data or overlapping training keys")
    report = dict(
        train_roots=len(conditional.templates), validation_roots=len(validation),
        samples_per_root=args.samples, seed=args.seed,
        prompt_edges=conditional.model.prompt_edges, strength=conditional.model.strength,
        conditional=evaluate(conditional, validation, args.samples, args.seed),
        uniform=evaluate(uniform, validation, args.samples, args.seed),
    )
    row = next(row for row in validation if (row.source_day, row.source_line) == ("2026-09-01", 141))
    payload, markdown = example_report(conditional, row, args.example_seed)
    args.output.mkdir(parents=True)
    for name, value in (("validation", report), ("example", payload)):
        (args.output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    (args.output / "example.md").write_text(markdown)
    print(f"Saved validation results and example under {args.output}")


if __name__ == "__main__":
    main()
