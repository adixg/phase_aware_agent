# TraceLab external trace analysis

## Overview

This directory contains existing empirical tool-latency results and a compact export of observed coding-agent session sequences from TraceLab (SyFI Lab, University of Washington). Source: [TraceLab](https://tracelab.cs.washington.edu) and [uw-syfi/TraceLab](https://github.com/uw-syfi/TraceLab). Released trace data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); source code is separately Apache 2.0. Changes here comprise field selection, identifier pseudonymization, session sampling, and revised documentation. No endorsement is implied.

## Data

```text
tracelab/
  README.md
  tool_distributions/
    README.md
    individual_tool_ecdf.csv
    individual_tool_ecdf.png
    individual_tool_ecdf.pdf
    major_tool_summary.csv
    latency_quality_by_tool_provider.csv
    tool_latency_by_tool.png
    tool_latency_weighted_bins_paper.png
  session_sequences/
    README.md
    sample_sessions.json
    session_summary.csv
```

## Methodology

The seven analysis CSV/figure files were copied byte-for-byte from TraceLab's `artifacts/tool_calls/tool_latency_distribution/individual_tool_audit/` and its parent. Distributions were not regenerated. The original audit README was adapted for this directory, with local figure links and technical documentation wording.

Sequences were extracted from `trace/syfi_coding_trace.duckdb` in read-only mode. The database contains 665,453 rounds and 743,819 tool calls. All child joins use `round_pk`; round, tool, and event ordering retain their source indices. Eight purposively selected session groups cover both providers, 3–120 rounds, diverse tools, multiple calls per round, and overlapping logged intervals. See [session methodology and schema](session_sequences/README.md). The JSON records the exact source database SHA-256. Raw prompts, conversation content, tool inputs/outputs, command text, and sensitive paths are excluded.

## Results

Tool durations differ materially by tool and provider. Among positive effective observations, Read has a median of 27 ms, Bash 329 ms, WebFetch 6.018 s, and WebSearch 17.521 s. Upper tails and discrete steps are visible in the empirical distributions. Across all tool types, calls lasting at least 60 s account for 3.39% of positive calls and 83.57% of summed effective latency. These totals describe attributed call time, including waiting and interaction tools.

The session sample retains observed LLM/tool ordering, input-result links, command continuation references, and overlapping call intervals. It contains both single-call rounds and batches; a single serial alternation is insufficient to describe every session.

## Limitations

Effective latency prefers internal runner-reported duration when present, otherwise wall timestamp difference. Measurement conventions vary across tools/providers. Of 743,819 calls, 600,272 have positive effective latency, 142,778 have zero, and 769 have missing latency; positive-only distributions exclude the latter two groups. Zero does not establish instantaneous execution. All-call quality counts are retained separately.

Wall intervals can include waits, interruptions, polling, and lifecycle effects; extreme values are not validated as continuous active execution. Summed overlapping calls are not elapsed session time. Source duplicates and within-session correlations prevent assuming independent observations. Logged events do not directly measure inference service time, CPU/GPU demand, memory, network utilization, or scheduling dependencies. Selected session examples are not statistically representative.

## Potential use

Tool/provider-conditioned empirical duration distributions can parameterize tool phases while retaining point masses and tails. Missing and zero measurements should remain explicit quality states. Observed sessions can supply phase order, batching, input relationships, and overlap for trace-informed workload generation. Duration resampling independently of sequence is a modeling choice that loses correlations; these exports do not establish that durations are independent. Resource assignments and extrapolation to other workloads require additional measurements. No replay, scheduling, or deadline model is implemented here.

## Reproduction

Distribution source and existing analysis commands are documented in [tool_distributions/README.md](tool_distributions/README.md). Session grouping, deterministic selection, schema, timing semantics, and extraction rules are documented in [session_sequences/README.md](session_sequences/README.md). Reuse the source database fingerprint recorded in the JSON for the same input snapshot. Source database and existing analysis outputs were not modified during export.
