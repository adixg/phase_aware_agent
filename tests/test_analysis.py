import json
from pathlib import Path

import pytest

from analysis import (
    JobTrace, cross_job_overlap, load_traces, non_llm_fraction,
    pairwise_ks, phase_durations, transition_entropy,
)


FIXTURES = Path(__file__).parent / "fixtures/traces/synthetic_run"
ORIGIN = 1700000000


@pytest.fixture
def jobs():
    return load_traces(sorted(FIXTURES.glob("*.jsonl")))


def test_loader_schema_and_top_level_order(jobs, tmp_path):
    assert list(jobs) == ["job_a", "job_b"]
    assert jobs["job_a"].summary["duration_s"] == 10
    assert jobs["job_b"].summary["duration_s"] == 12
    records = []
    for job in jobs.values():
        records.extend(reversed(job.spans))
        records.append(job.summary)
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    loaded = load_traces(path)
    for key in jobs:
        assert loaded[key] == jobs[key]
        assert [s["step"] for s in loaded[key].spans] == list(range(5))
        assert [s["meta"]["step"] for s in loaded[key].spans] == [0, 0, 1, 1, 2]


def test_durations_and_ks(jobs):
    assert phase_durations(jobs) == {
        "llm_generate": [2, 4, 2, 4, 2, 4], "retrieve": [1, 1], "tool_exec": [1, 1]}
    results = pairwise_ks(jobs)
    for phase in ("retrieve", "tool_exec"):
        assert results["llm_generate", phase]["statistic"] == 1
        assert results["llm_generate", phase]["pvalue"] == pytest.approx(1 / 14)
    assert results["retrieve", "tool_exec"] == {"statistic": 0, "pvalue": 1}


def test_transition_entropy(jobs):
    result = transition_entropy(jobs)
    assert result["counts"] == {
        "llm_generate": {"retrieve": 2, "tool_exec": 2},
        "retrieve": {"llm_generate": 2}, "tool_exec": {"llm_generate": 2}}
    assert result["probabilities"]["llm_generate"] == {"retrieve": .5, "tool_exec": .5}
    assert result["entropy_bits"] == {"llm_generate": 1, "retrieve": 0, "tool_exec": 0}
    assert result["conditional_entropy_bits"] == .5


@pytest.mark.parametrize("basis", ["job", "span"])
def test_cross_job_overlap(jobs, basis):
    assert cross_job_overlap(jobs, basis=basis) == {
        ("job_a", "job_b"): {"intervals": [(ORIGIN + 1, ORIGIN + 10)], "duration_s": 9}}


def test_non_llm_fraction(jobs):
    result = non_llm_fraction(jobs)
    assert result["per_job"]["job_a"] == {"duration_s": 10, "non_llm_s": 2, "fraction": .2}
    assert result["per_job"]["job_b"]["fraction"] == pytest.approx(1 / 6)
    assert result["per_job"]["job_b"]["non_llm_s"] == 2
    assert result["pooled_fraction"] == pytest.approx(2 / 11)


def test_gaps_union_clipping_and_touching_intervals():
    def span(start, end):
        return {"phase": "llm_generate", "start_ts": start, "end_ts": end}
    jobs = {
        "a": JobTrace([span(-1, 2), span(1, 3), span(6, 7)], {"start_ts": 0, "end_ts": 10}),
        "b": JobTrace([span(3, 5), span(6, 8)], {"start_ts": 3, "end_ts": 8}),
    }
    assert cross_job_overlap(jobs, basis="span")["a", "b"] == {
        "intervals": [(6, 7)], "duration_s": 1}
    assert cross_job_overlap(jobs)["a", "b"]["duration_s"] == 5
    result = non_llm_fraction(jobs)
    assert result["per_job"]["a"]["fraction"] == .6
    assert result["pooled_fraction"] == pytest.approx(7 / 15)


def test_empty_and_missing_summary():
    assert phase_durations({}) == {}
    assert pairwise_ks({}) == {}
    assert transition_entropy({})["conditional_entropy_bits"] == 0
    assert cross_job_overlap({}) == {}
    assert non_llm_fraction({})["pooled_fraction"] is None
    with pytest.raises(ValueError, match="job_summary"):
        non_llm_fraction({"a": JobTrace()})


@pytest.mark.parametrize("records", [
    [{"record_type": "unknown", "job_id": "a"}],
    [{"record_type": "span", "job_id": "a"}],
    [{"record_type": "job_summary", "job_id": "a", "start_ts": 0, "end_ts": 1,
      "duration_s": 1}] * 2,
])
def test_invalid_records_have_source_location(records, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="bad.jsonl:"):
        load_traces(path)
