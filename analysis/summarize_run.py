"""Save the four existing analyses for one independently selected trace run."""

import argparse
import json
import platform
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np
import scipy

from . import (load_traces, phase_durations, pairwise_ks, transition_entropy,
               cross_job_overlap, non_llm_fraction)


LIMITATIONS = [
    "KS p-values assume independent continuous samples; within-job correlation and ties limit interpretation. No multiple-comparison correction is applied; numeric zero p-values may reflect underflow.",
    "Transitions follow top-level span step within each job; terminal spans have no successor. Entropy describes observed routing, not scheduling uncertainty.",
    "Overlap assumes a common timestamp clock and reports job residency and recorded span activity, not GPU contention. Pairwise seconds can double-count wall time.",
    "Outside-LLM time includes other phases and uninstrumented gaps. LLM spans can include waiting and CPU work, so this is not GPU utilization. The pooled fraction weights job lifetimes, not run wall time.",
    "All recorded jobs, including unsuccessful jobs, are included. These serial workloads do not establish behavior under concurrent load or causal differences between phases.",
]


def summarize(run_dir, output_dir):
    paths = sorted(run_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"No trace files in {run_dir}")
    jobs = load_traces(paths)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if set(jobs) != {j["job_id"] for j in manifest["jobs"]}:
        raise ValueError("Trace jobs do not match manifest jobs")
    samples = phase_durations(jobs)
    durations = {}
    for phase, values in sorted(samples.items()):
        durations[phase] = {
            "count": len(values), "total_s": sum(values), "mean_s": fmean(values),
            "std_population_s": pstdev(values), "min_s": min(values),
            "max_s": max(values),
            **{f"p{p}_s": float(np.percentile(values, p)) for p in (25, 50, 75, 90, 95, 99)},
            "samples_s": values,
        }
    overlaps = {}
    for basis in ("job", "span"):
        pairs = cross_job_overlap(jobs, basis=basis)
        positive = [{"job_a": a, "job_b": b, **value}
                    for (a, b), value in pairs.items() if value["duration_s"] > 0]
        overlaps[basis] = {
            "pair_count": len(pairs), "overlapping_pair_count": len(positive),
            "zero_overlap_pair_count": len(pairs) - len(positive),
            "total_pairwise_overlap_s": sum(v["duration_s"] for v in pairs.values()),
            "max_pairwise_overlap_s": max((v["duration_s"] for v in pairs.values()), default=0),
            "positive_pairs": positive,
            "omitted_pairs": "All unlisted job pairs have zero overlap.",
        }
    outside = non_llm_fraction(jobs)
    outside["total_job_lifetime_s"] = sum(v["duration_s"] for v in outside["per_job"].values())
    outside["total_outside_llm_s"] = sum(v["non_llm_s"] for v in outside["per_job"].values())
    result = {
        "run_id": run_dir.name, "input_directory": str(run_dir),
        "input_files": [str(p) for p in paths],
        "versions": {"python": platform.python_version(), "scipy": scipy.__version__, "numpy": np.__version__},
        "num_jobs": len(jobs), "num_span_records": sum(len(j.spans) for j in jobs.values()),
        "num_summary_records": sum(j.summary is not None for j in jobs.values()),
        "observed_phases": sorted(samples),
        "manifest": {k: manifest.get(k) for k in ("mode", "task_set", "num_tasks", "num_succeeded", "num_correct", "num_errors", "model", "gpu", "git_commit")},
        "phase_durations": durations,
        "pairwise_ks": [{"phase_a": a, "phase_b": b, **v} for (a, b), v in pairwise_ks(jobs).items()],
        "transition_entropy": transition_entropy(jobs),
        "cross_job_overlap": overlaps, "outside_llm": outside,
        "limitations": LIMITATIONS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{run_dir.name}_summary"
    target.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [f"# {run_dir.name}", "", f"Jobs: {len(jobs)}; span records: {result['num_span_records']}; summary records: {result['num_summary_records']}.", "",
             "Duration statistics are seconds; standard deviation is population SD and percentiles use linear interpolation.", "",
             "| Phase | n | Mean | SD | Min | p50 | p95 | p99 | Max |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for phase, d in durations.items():
        lines.append(f"| {phase} | {d['count']} | " + " | ".join(f"{d[k]:.9g}" for k in ("mean_s", "std_population_s", "min_s", "p50_s", "p95_s", "p99_s", "max_s")) + " |")
    lines += ["", "Exact two-sided pairwise KS (uncorrected):", ""]
    for v in result["pairwise_ks"]:
        lines.append(f"- {v['phase_a']} vs {v['phase_b']}: D={v['statistic']:.9g}, p={v['pvalue']:.9g}.")
    lines += ["", "Transition entropy in bits:", ""]
    for phase in sorted(samples):
        entropy = result["transition_entropy"]["entropy_bits"].get(phase)
        lines.append(f"- {phase}: {entropy:.9g}." if entropy is not None else f"- {phase}: undefined (no outgoing transitions).")
    lines += [f"- Weighted conditional entropy: {result['transition_entropy']['conditional_entropy_bits']:.9g}.", ""]
    for basis, v in overlaps.items():
        lines.append(f"Cross-job overlap ({basis}): {v['overlapping_pair_count']} / {v['pair_count']} pairs; total {v['total_pairwise_overlap_s']:.9g} s; maximum {v['max_pairwise_overlap_s']:.9g} s.\n")
    lines += [f"Outside-LLM pooled fraction: {outside['pooled_fraction']:.9g} ({outside['pooled_fraction']:.6%}); {outside['total_outside_llm_s']:.9g} / {outside['total_job_lifetime_s']:.9g} job-seconds.", "",
              "The JSON includes raw duration samples, full transition counts/probabilities, per-job outside-LLM fractions, and positive overlap intervals; omitted overlap pairs are zero.", "", "Limitations:", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    target.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    summarize(args.run_dir, args.output_dir)
