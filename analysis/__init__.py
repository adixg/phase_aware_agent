"""Offline analysis of caller-selected phase traces."""

from .traces import JobTrace, load_traces
from .metrics import (
    cross_job_overlap,
    non_llm_fraction,
    phase_durations,
    pairwise_ks,
    transition_entropy,
)

__all__ = [
    "JobTrace", "load_traces", "phase_durations", "pairwise_ks",
    "transition_entropy", "cross_job_overlap", "non_llm_fraction",
]
