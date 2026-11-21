# FreeInference Future DAG Generator

The generator draws whole first-turn DAGs from the training split, including linked nested subagent chains. It uses the admission snapshot's harness, requested model, and estimated initial prompt tokens. The already-admitted LLM request counts toward the forecast. [Three sampled DAGs and the observed validation DAG](FUTURE_DAG_EXAMPLE.md)

## Run the demo

```python
from pathlib import Path

from prediction.predictor.future_dag import FutureDAGGenerator

data = Path("data/freeinference")
generator = FutureDAGGenerator(data / "preprocessed", data / "empirical-baseline/model.json")
samples = generator.predict_dags(snapshot, num_samples=10, seed=42)
```

Pass a normalized admission snapshot. Each draw includes nodes with tool categories and chain membership, directed edges with kind, evidence, and provenance, and the matched-subagent count. `forecast` contains LLM count, tool count, and predicted observed gateway duration. `trace_metadata` and node `trace_timing` attribute source measurements to the training template and use that trace's admission as the time origin. Tool start, finish, and execution duration are unavailable. Inferred spawn links represent publisher matches, and inferred tool feedback represents a single candidate. Sibling execution order and subagent completion dependencies remain unknown where the trace leaves them unresolved.

The sampler starts with uniform mass over training templates and reuses the existing baseline's hierarchy and fitted settings (32,212-token boundary and smoothing strength 20). At each matching harness, harness/model, and harness/model/prompt-bucket level, a group of size *n* contributes *n / (n + 20)* of the mass, distributed uniformly inside the group. The remaining mass comes from the parent. When a level is unavailable, the deepest populated parent supplies the distribution. Draws use replacement. `template_probability` is the total probability of drawing that training root on any one draw. Several templates can share a topology, and repeated draws each retain their original template probability.

```sh
python3 -m prediction.predictor.evaluate_future_dag --output data/freeinference/future-dag-demo
python3 -m unittest prediction.predictor.test_freeinference_preprocess prediction.predictor.test_empirical_baseline prediction.predictor.test_future_dag
```

The evaluator writes the metrics, example JSON, and Mermaid Markdown to the fresh gitignored output directory. It uses the existing preprocessed artifacts and baseline configuration. See the [baseline milestone](FREEINFERENCE_MILESTONE.md) for how those artifacts were produced.

## Validation results

The sampler fit 1,433 training roots. Evaluation used 753 validation roots and ten seeded draws per root from each sampler. The uniform comparator gives each training root equal mass. The held-out test split was left unread by the sampler and evaluator. The conditioning settings came from the existing baseline, which selected them on this validation cohort.

All 2,186 admission graphs matched their preprocessed node and supported-edge labels, counts, categories, and gateway endpoint. All 15,060 draws passed node, edge, count, provenance, and acyclicity checks. Both the full edge sets and observed-edge subsets were acyclic. Conditional draws carried 121,320 inferred single-candidate feedback links and 427 inferred spawn links, all copied from their training templates.

| Target | CRPS uniform | CRPS conditional | Median MAE uniform → conditional | P10–P90 coverage uniform → conditional |
| --- | ---: | ---: | ---: | ---: |
| LLM requests | 12.81 | 12.61 | 16.12 → 15.93 | 90.4% → 85.9% |
| Tool calls | 15.22 | 14.80 | 19.33 → 18.90 | 91.9% → 85.9% |
| Observed duration (s) | 454.17 | 443.74 | 531.43 → 519.67 | 83.4% → 81.8% |

These are exact marginal scores from template weights. Conditional sampled means were 20.31 LLM requests, 24.22 tool calls, and 481.95 seconds. Uniform sampled means were 19.40, 23.02, and 424.78 seconds. Observed means were 19.12, 21.92, and 564.95 seconds. The exact conditional duration mean was also low at 468.43 seconds.

| Graph metric | Uniform | Conditional |
| --- | ---: | ---: |
| Mean matched-branch probability | 2.65% | 2.77% |
| Sampled matched-branch frequency | 2.55% | 2.59% |
| Matched-branch Brier score | 0.01076 | 0.01134 |
| Mean distinct structures per ten draws | 9.10 | 8.39 |
| Probability two draws have different structures | 97.33% | 94.81% |
| Mean probability of the exact validation structure | 2.25% | 2.77% |

A structure is the set of nodes and edges keyed by normalized local IDs, with node type, tool category, edge kind, and provenance. This measure excludes timing and is sensitive to local IDs. Validation contained eight matched-branch roots (1.06%). In the conditional 0–10% branch-probability bin, predicted and observed rates were 1.85% and 0.86% across 699 roots. In the 10–20% bin, they were 14.66% and 3.70% across 54 roots. These estimates describe publisher-matched branches and have few positive examples.

## Limitations and next steps

- Whole-template sampling preserves observed combinations but limits support to training topologies. Collect more traces with matched subagents, then test hierarchical branch recalibration on a later validation window.
- Gateway duration includes gaps and overlap and ends at the last in-scope LLM gateway response. Better tool and subagent instrumentation would enable evaluation of runtime dependencies and workflow completion.
- The current exact-structure score is strict, and the marginal scores assess one target at a time. Add ID-invariant topology summaries and joint calibration checks before using structural forecasts in scheduling decisions.
