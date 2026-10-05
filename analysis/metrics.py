"""Small descriptive metrics; timestamps are seconds on a shared clock."""

from collections import Counter, defaultdict
from itertools import combinations
from math import log2


def phase_durations(jobs):
    """Return raw duration_s samples per phase, in job and span order."""
    samples = defaultdict(list)
    for job in jobs.values():
        for span in job.spans:
            samples[span["phase"]].append(span["duration_s"])
    return dict(samples)


def pairwise_ks(jobs):
    """Exact two-sided two-sample KS comparisons for all observed phases.

    P-values use the independent, continuous-sample null; ties and correlated
    spans within jobs limit their inferential interpretation.
    """
    from scipy.stats import ks_2samp

    samples = phase_durations(jobs)
    result = {}
    for a, b in combinations(sorted(samples), 2):
        test = ks_2samp(samples[a], samples[b], method="exact")
        result[a, b] = {"statistic": float(test.statistic), "pvalue": float(test.pvalue)}
    return result


def transition_entropy(jobs):
    """Adjacent-span transitions within jobs, with entropy measured in bits.

    Terminal spans contribute no outgoing transition; no cross-job transitions
    or implicit WAIT/terminal phases are introduced.
    """
    counts = defaultdict(Counter)
    for job in jobs.values():
        for current, following in zip(job.spans, job.spans[1:]):
            counts[current["phase"]][following["phase"]] += 1
    probabilities, entropy = {}, {}
    total = sum(sum(row.values()) for row in counts.values())
    weighted = 0.0
    for phase, row in counts.items():
        n = sum(row.values())
        probabilities[phase] = {next_phase: count / n for next_phase, count in row.items()}
        entropy[phase] = -sum(p * log2(p) for p in probabilities[phase].values())
        weighted += n * entropy[phase]
    return {"counts": {p: dict(row) for p, row in counts.items()},
            "probabilities": probabilities, "entropy_bits": entropy,
            "conditional_entropy_bits": weighted / total if total else 0.0}


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("interval end precedes start")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _lifetime(job):
    if job.summary is None:
        raise ValueError("job_summary required for lifetime metrics")
    start, end = job.summary["start_ts"], job.summary["end_ts"]
    if end <= start:
        raise ValueError("job lifetime must have positive duration")
    return start, end


def cross_job_overlap(jobs, *, basis="job"):
    """Pairwise half-open overlap intervals and seconds, including zero pairs.

    basis='job' uses summary lifetimes; basis='span' uses unions of recorded
    spans. Neither measures GPU overlap. Pair totals must not be summed as
    unique wall time when three or more jobs overlap.
    """
    if basis not in ("job", "span"):
        raise ValueError("basis must be 'job' or 'span'")
    intervals = {
        key: [_lifetime(job)] if basis == "job" else
        _union((s["start_ts"], s["end_ts"]) for s in job.spans)
        for key, job in jobs.items()
    }
    result = {}
    for a, b in combinations(sorted(jobs), 2):
        overlaps = _union(
            (max(x, u), min(y, v))
            for x, y in intervals[a] for u, v in intervals[b]
            if max(x, u) < min(y, v)
        )
        result[a, b] = {"intervals": overlaps, "duration_s": sum(y - x for x, y in overlaps)}
    return result


def non_llm_fraction(jobs, *, llm_phases=("llm_generate",)):
    """Complement of unioned LLM time within each summary lifetime.

    Includes gaps and other phases; pooled fraction weights by job lifetime,
    counting concurrent jobs separately. This is a proxy, not GPU service time.
    """
    per_job = {}
    total = outside = 0.0
    for key, job in jobs.items():
        start, end = _lifetime(job)
        llm = _union(
            (max(start, s["start_ts"]), min(end, s["end_ts"]))
            for s in job.spans if s["phase"] in llm_phases
            and max(start, s["start_ts"]) < min(end, s["end_ts"])
        )
        duration = end - start
        non_llm = duration - sum(y - x for x, y in llm)
        per_job[key] = {"duration_s": duration, "non_llm_s": non_llm,
                        "fraction": non_llm / duration}
        total += duration
        outside += non_llm
    return {"per_job": per_job, "pooled_fraction": outside / total if total else None}
