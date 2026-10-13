# Offline trace analysis

Install with `pip install -e '.[test,analysis]'`. Run synthetic checks with
`python -m pytest tests/test_analysis.py`.

Save JSON and Markdown reports for an individual real run with:

```sh
python -m analysis.summarize_run results/traces/serial-6157079
python -m analysis.summarize_run results/traces/serial-6157083
```

Each invocation loads only that directory's JSONL files and checks job IDs against
its manifest. Reports go to `results/analysis` (override with `--output-dir`). JSON
includes raw duration samples, summary statistics, KS results, transition counts
and probabilities, per-job outside-LLM fractions, and both overlap bases. Zero
overlap pairs are represented by a count; all positive pairs retain intervals.

```python
from pathlib import Path
from analysis import (load_traces, phase_durations, pairwise_ks,
                      transition_entropy, cross_job_overlap, non_llm_fraction)

jobs = load_traces(sorted(Path("tests/fixtures/traces/synthetic_run").glob("*.jsonl")))
samples = phase_durations(jobs)
ks = pairwise_ks(jobs)
transitions = transition_entropy(jobs)
overlap = cross_job_overlap(jobs)
outside_llm = non_llm_fraction(jobs)
```

The caller selects files from one run with unique job IDs and a common timestamp
clock. There is no run_id in the schema, so the loader cannot identify runs or
correct clock skew. Records are retained as dictionaries and spans are sorted
by top-level step, never meta.step. A job may span multiple input files; duplicate
summaries are rejected. Summary-dependent metrics fail if a summary is missing.

Duration distributions are raw duration_s samples. Pairwise KS uses SciPy's exact
two-sided test, without multiple-comparison correction. The independent,
continuous-sample null is only a reference: repeated/tied durations and correlated
spans limit p-value interpretation.

Transitions are adjacent observed phases within each job. Entropy is in bits;
conditional entropy weights each phase by its number of outgoing transitions.
Terminal spans have no successor. Empty transition sets have entropy zero.

Overlap reports each job pair's half-open intervals and total seconds. The default
uses summary start_ts/end_ts (job residency); basis="span" uses recorded span
unions (instrumented activity). Neither estimates GPU contention. Pairwise seconds
can double-count wall time when three or more jobs overlap.

Outside-LLM time is summary lifetime minus the union of llm_generate intervals,
clipped to that lifetime. This includes other phases and uninstrumented gaps.
Overlapping LLM spans are counted once per job. The pooled fraction divides summed
outside-LLM seconds by summed job lifetimes, not run wall time. Empty input returns
None for this undefined fraction; nonpositive job lifetimes are rejected.
No arrival_ts, react_step, WAIT spans, queue_s, or GPU-service intervals are
assumed. LLM spans may themselves include CPU work or waiting, so the complement
is only a proxy for non-GPU time, not a GPU utilization measurement.

## Synthetic reference

The two files match the tracer's per-job JSONL layout and share origin 1700000000.
Times below are offsets in seconds; meta.step repeats for an agent iteration.

| Job | Ordered phase intervals | Summary lifetime |
| --- | --- | --- |
| a | LLM 0–2, retrieve 2–3, LLM 3–7, tool 7–8, LLM 8–10 | 0–10 |
| b | LLM 1–5, tool 5–6, LLM 6–8, retrieve 8–9, LLM 9–13 | 1–13 |

LLM durations are [2,4,2,4,2,4]; retrieve/tool each have [1,1]. KS statistic
is 1 with p=1/14 for LLM versus either other phase, and 0 with p=1 for
retrieve versus tool. LLM has two transitions to each other phase; each other
phase has two transitions to LLM. Entropies are 1, 0, 0 bits and weighted
conditional entropy is 0.5 bits. Both overlap definitions give [1,10), or 9
seconds. Outside-LLM fractions are 2/10 and 2/12; pooled fraction is 4/22.
