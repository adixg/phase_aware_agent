# Oracle replay results

## Resource sensitivity and demotion

Feasible-turn attainment is averaged across three paired daily cohorts. The
resource sweep uses the common middle 80% arrival window per load/KV cell.
The demotion comparisons use their own paired cohorts, so the EDF percentages
differ slightly across experiments.

| Target load | KV | EDF + LRU | Cheap LLF + LRU | Oracle LLF + LRU | Oracle eviction gain over cheap eviction with EDF |
| --- | --- | --- | --- | --- | --- |
| 0.7 | Unlimited | 98.77% | 96.14% | 98.88% | — |
| 0.7 | 1M | 90.48% | 51.00% | 90.99% | +0.58 pp |
| 1.1 | Unlimited | 97.08% | 78.75% | 97.78% | — |
| 1.1 | 1M | 91.79% | 49.47% | 92.38% | +0.28 pp |

At load 1.1 with 1M KV, memory blocking consumed 72.9% of the measurement
window with EDF + LRU. Oracle SRPT reached 91.37% feasible attainment with
LRU, up 41.76 pp from cheap SRPT.

| 1M KV demotion ablation at load 1.1 | Cheap order and demotion | Oracle order only | Oracle demotion only | Oracle order and demotion |
| --- | --- | --- | --- | --- |
| LLF | 49.67% | 50.45% | 91.48% | 92.41% |
| SRPT | 49.82% | 50.33% | 91.02% | 91.37% |

EDF reached 91.83% in the 1M ablation cohort. In the overdue-only follow-up,
LLF reached 91.40% and SRPT reached 88.81%, against 91.79% for EDF. Across
both KV sizes, overdue-only demotion recovered 93.9–97.9% of the feasible
attainment improvement from the cheap policy to its Oracle counterpart.

## DAG replay and deadline sensitivity

Feasible attainment below covers all replayed turns. The nested counts pool
cases across the three days.

| Replay | Join assumption | EDF | DAG | DAG minus EDF |
| --- | --- | --- | --- | --- |
| Initial DAG replay | No join | 97.92% | 97.46% | −0.46 pp |
| Initial DAG replay | Plausible join | 97.89% | 97.51% | −0.38 pp |
| All-turn DAG replay | No join | 97.81% | 97.37% | −0.44 pp |
| All-turn DAG replay | Plausible join | 97.81% | 97.58% | −0.24 pp |

At 1× without joins, the all-turn DAG policy reduced mean nested latency
from 2,023.8 to 2,016.9 seconds. Every policy met 7 of 32 nested deadlines
under both join assumptions.

| Global scaling, no join | EDF all-turn attainment | DAG all-turn attainment | Nested deadlines met, both policies |
| --- | --- | --- | --- |
| 1× | 66.73% | 66.43% | 7/32 |
| 2× | 77.30% | 77.27% | 12/32 |
| 3× | 82.53% | 82.35% | 12/32 |
| 8× | 91.91% | 91.93% | 18/32 |

In the nested-only diagnostic at 2× without joins, both the DAG and
flattened-timing policies met one additional nested deadline relative to EDF
(12/32 versus 11/32). At 8×, all policies met 18/32 in both join scenarios.
