# TraceLab tool latency

Scope: individual tool latency distributions only. No scheduling implementation, deadlines, or workload replay.

## Major tools

Latencies below are **seconds**, conditional on strictly positive effective latency. Valid = every positive observation, not a reservoir sample. Missing and zero calls are excluded; none of these tools has a negative effective latency. Errors remain included. `exec` pools 46,754 Codex calls and 3 Claude calls; other rows belong to the provider shown.

| Tool | Provider | Calls | Valid | Missing | Zero | p50 (s) | p90 (s) | p99 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exec_command | codex | 251,630 | 144,778 | 23 | 106,829 | 0.336 | 1.003 | 30.001 |
| Bash | claude | 158,610 | 158,456 | 139 | 15 | 0.329 | 20.406 | 300.136 |
| write_stdin | codex | 75,975 | 56,384 | 23 | 19,568 | 5.002 | 60.001 | 300.001 |
| Read | claude | 63,984 | 63,663 | 281 | 40 | 0.027 | 0.245 | 3.013 |
| exec | Codex + 3 Claude | 46,757 | 46,756 | 1 | 0 | 1.457 | 10.050 | 10.078 |
| Edit | claude | 36,713 | 36,530 | 180 | 3 | 0.075 | 1.997 | 62.218 |
| apply_patch | codex | 32,332 | 24,715 | 2 | 7,615 | 0.064 | 1.454 | 13.584 |
| WebFetch | claude | 3,636 | 3,634 | 0 | 2 | 6.018 | 19.357 | 201.289 |
| WebSearch | claude | 1,710 | 1,707 | 1 | 2 | 17.521 | 94.903 | 262.707 |

## Figures

1. [Per-tool empirical CDFs](individual_tool_ecdf.png) ([PDF](individual_tool_ecdf.pdf)): the nine major tools, with separate provider curves. Exact cumulative probability at each distinct positive observation; no binning, sampling, smoothing, or fitted distribution. Axes span the full observed range separately per panel; the dotted line is 99%.
2. [Existing full per-tool boxes](tool_latency_by_tool.png): median and IQR with 5th–95th percentile whiskers; useful for comparing typical spreads and showing waiting/user/sub-agent tools alongside the major tools. Outliers hidden; labels count **all** calls, not valid samples.
3. [Existing compact weighted bins](tool_latency_weighted_bins_paper.png): provider-specific call share versus summed-latency share; useful context for why rare slow calls matter. Includes all tool types, not only these nine.

## Five findings

1. **Typical latency varies greatly by tool.** Read/Edit/apply_patch medians are 27/75/64 ms; WebFetch/WebSearch medians are 6.02/17.52 s. Treating all tool phases as one latency population loses this structure.
2. **Upper tails matter even for fast tools.** Bash p99/p50 is 912×; Edit is 830×; Read is 112×. The slowest ceil(1% of positive calls) contributes 68.7% of Bash, 79.4% of Edit, 88.8% of Read, and 86.8% of apply_patch summed effective latency. These are empirical upper tails, not evidence establishing a particular mathematical heavy-tail family.
3. **Several distributions have steps or shoulders.** exec_command has a prominent step near 1 s; write_stdin has repeated steps, with p50/p90/p99 near 5/60/300 s; Codex exec has a large step near 10 s. These are consistent with polling/yield/measurement boundaries, but durations alone do not establish the cause. A smooth single-family fit could erase these point masses.
4. **Web tools are broadly slow with long upper tails.** WebFetch p99 is 201 s (33× median); WebSearch p99 is 263 s (15× median). Respectively 5.1% and 14.3% of positive calls exceed 60 s. Their CDFs have shoulders and jumps rather than proving lognormality.
5. **Call frequency and attributed latency tell different stories.** Existing global bins show 61.3% of positive calls below 1 s contribute only 0.56% of summed latency. Calls at least 60 s are 3.39% of calls and contribute 83.57% of summed latency. This includes waiting, interaction, and sub-agent calls; it is not a CPU utilization result.

## Quality and comparability caveats

- Across the database: 743,819 calls; 600,272 positive, 769 missing, 142,778 zero, and 0 negative effective latencies. Zero does not prove instantaneous execution. The positive-only percentiles describe a selected population.
- Effective latency uses the existing shared rule: internal timing if non-null, otherwise wall timestamp difference. A zero internal value takes precedence over a positive wall value. This affects 104,799 exec_command, 19,351 write_stdin, and 6,142 apply_patch calls. Zero exclusion rates are 42.5%, 25.8%, and 23.6% respectively. We preserve this convention rather than silently substituting wall time.
- Measurement sources differ: exec_command selects internal timing for 250,923/251,630 calls and write_stdin for 75,107/75,975; Bash, Read, Edit, and exec have no internal timing here. WebFetch mixes 1,006 internal with 2,630 wall measurements; WebSearch mixes 494 internal with 1,215 wall measurements. See the audit CSV for every tool/provider. Provider and tool identity are strongly confounded.
- Codex exec has 46,753 positive samples: p50/p90/p99 = 1.457/10.050/10.078 s. Claude exec has only 3 samples: 0.020/0.147/0.176 s; these cannot support a provider-performance conclusion.
- Extreme maxima require caution: Bash reaches 42.8 h, Read 3.44 h, Edit 8.66 h, apply_patch 10.99 h, exec_command 13.55 h, and exec 13.45 h. Timestamp gaps, interruptions, and other lifecycle effects may be included. No outliers were deleted; extreme observations are not validated as continuous active execution.
- A tool interaction can return while a command remains running; exec_command and write_stdin durations are individual interactions, not whole-command completion times. Wall-clock latency is not CPU execution time. Network, waiting, user interaction, and sub-agent tools need separate resource interpretations.
- Summing overlapping tool calls does not yield session elapsed time. Preserved duplicate source rows and repeated calls within sessions also mean these observations are not necessarily independent. This coding-agent workload may differ materially from other workloads.
- Provider-wide positive-call medians are 115 ms for Claude and 921 ms for Codex; summed effective latency is 2,412.70 h and 676.28 h, respectively.

## Implications for phase modeling

Use tool- and provider-conditioned empirical distributions as descriptive starting points, retaining discrete masses and rare tails. Keep missing/zero measurement states separate from positive durations. Distinguish local file/edit interactions, command launch/poll interactions, network access, user waits, and sub-agent activity before assigning resource demand. These traces can motivate modeling hypotheses; CPU service demand and target-RAG distributions still require their own measurements. No scheduling policy or implementation is inferred here.

## Files and reproduction

The three CSVs retain the existing analysis values and units (milliseconds). `individual_tool_ecdf.csv` records distinct positive latencies, multiplicities, cumulative counts, and empirical probabilities. `major_tool_summary.csv` summarizes nine major tools; `latency_quality_by_tool_provider.csv` includes measurement-quality counts and tail statistics by tool and provider. Figures and CSVs were copied without regeneration.

Source: TraceLab `artifacts/tool_calls/tool_latency_distribution/individual_tool_audit/` and its parent directory. Reproduce within TraceLab using its existing environment:

```sh
.venv/bin/python artifacts/tool_calls/tool_latency_distribution/individual_tool_audit/analyze.py --db trace/syfi_coding_trace.duckdb
```

The two parent-directory figures are produced by `artifacts/tool_calls/tool_latency_distribution/plot.py`. Reproduction writes analysis outputs; it was not run for this export. See the parent README for provenance and license.
