# serial-6157079

Jobs: 100; span records: 558; summary records: 100.

Duration statistics are seconds; standard deviation is population SD and percentiles use linear interpolation.

| Phase | n | Mean | SD | Min | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_generate | 337 | 1.06548653 | 0.319498028 | 0.215707064 | 1.04817963 | 1.61024194 | 1.94276462 | 2.52541757 |
| retrieve | 221 | 0.019388283 | 0.0354675387 | 0.0150449276 | 0.0161416531 | 0.0171105862 | 0.0175973415 | 0.485913992 |

Exact two-sided pairwise KS (uncorrected):

- llm_generate vs retrieve: D=0.992507754, p=1.1423797e-156.

Transition entropy in bits:

- llm_generate: 0.371232327.
- retrieve: -0.
- Weighted conditional entropy: 0.192911122.

Cross-job overlap (job): 0 / 4950 pairs; total 0 s; maximum 0 s.

Cross-job overlap (span): 0 / 4950 pairs; total 0 s; maximum 0 s.

Outside-LLM pooled fraction: 0.014032332 (1.403233%); 5.11028409 / 364.179245 job-seconds.

The JSON includes raw duration samples, full transition counts/probabilities, per-job outside-LLM fractions, and positive overlap intervals; omitted overlap pairs are zero.

Limitations:

- KS p-values assume independent continuous samples; within-job correlation and ties limit interpretation. No multiple-comparison correction is applied; numeric zero p-values may reflect underflow.
- Transitions follow top-level span step within each job; terminal spans have no successor. Entropy describes observed routing, not scheduling uncertainty.
- Overlap assumes a common timestamp clock and reports job residency and recorded span activity, not GPU contention. Pairwise seconds can double-count wall time.
- Outside-LLM time includes other phases and uninstrumented gaps. LLM spans can include waiting and CPU work, so this is not GPU utilization. The pooled fraction weights job lifetimes, not run wall time.
- All recorded jobs, including unsuccessful jobs, are included. These serial workloads do not establish behavior under concurrent load or causal differences between phases.
