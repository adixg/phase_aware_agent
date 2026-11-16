# serial-6157083

Jobs: 1305; span records: 7111; summary records: 1305.

Duration statistics are seconds; standard deviation is population SD and percentiles use linear interpolation.

| Phase | n | Mean | SD | Min | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_generate | 4266 | 1.04334139 | 0.348573155 | 0.173591614 | 1.00631285 | 1.62363964 | 2.07034929 | 5.23193288 |
| retrieve | 2845 | 0.0175286869 | 0.021998921 | 0.0144116879 | 0.0163865089 | 0.0174208641 | 0.0300589371 | 1.07828116 |

Exact two-sided pairwise KS (uncorrected):

- llm_generate vs retrieve: D=0.998008119, p=0.

Transition entropy in bits:

- llm_generate: 0.238501129.
- retrieve: -0.
- Weighted conditional entropy: 0.121633111.

Cross-job overlap (job): 0 / 850860 pairs; total 0 s; maximum 0 s.

Cross-job overlap (span): 0 / 850860 pairs; total 0 s; maximum 0 s.

Outside-LLM pooled fraction: 0.0132401745 (1.324017%); 59.7213392 / 4510.61573 job-seconds.

The JSON includes raw duration samples, full transition counts/probabilities, per-job outside-LLM fractions, and positive overlap intervals; omitted overlap pairs are zero.

Limitations:

- KS p-values assume independent continuous samples; within-job correlation and ties limit interpretation. No multiple-comparison correction is applied; numeric zero p-values may reflect underflow.
- Transitions follow top-level span step within each job; terminal spans have no successor. Entropy describes observed routing, not scheduling uncertainty.
- Overlap assumes a common timestamp clock and reports job residency and recorded span activity, not GPU contention. Pairwise seconds can double-count wall time.
- Outside-LLM time includes other phases and uninstrumented gaps. LLM spans can include waiting and CPU work, so this is not GPU utilization. The pooled fraction weights job lifetimes, not run wall time.
- All recorded jobs, including unsuccessful jobs, are included. These serial workloads do not establish behavior under concurrent load or causal differences between phases.
