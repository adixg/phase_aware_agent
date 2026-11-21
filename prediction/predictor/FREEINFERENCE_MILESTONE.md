# FreeInference admission workload prediction

The baseline is a probabilistic workload predictor for the first user turn and its release-linked descendants. It forecasts joint counts and observed duration at initial request admission. Future DAG structure remains outside this milestone. The forecast contains no future node identities, dependency edges, branch topology, execution order, or critical path.

## Inputs, targets, and split

The model uses the released harness, requested model, and the initial prompt estimate (the sum of five input-role token counts, including tool definitions). The harness is a release proxy whose availability at serving admission still needs confirmation. The evaluator checks that the model inputs are available by the cutoff.

The joint outcome is the number of remaining LLM requests (including the already-admitted request and linked child requests), number of future emitted tool calls, and milliseconds to the last visible in-scope gateway response. Later user turns are outside the forecast. Subtract one from the LLM count to estimate additional requests beyond the admitted one.

| Split | Root admission dates, UTC (inclusive) | Roots before → after purge | Use |
| --- | --- | ---: | --- |
| Train | Aug 23–29, 2026 | 1,437 → 1,433 | Fit distributions and prompt buckets |
| Validation | Aug 30–Sep 1, 2026 | 783 → 753 | Select configuration and compare forecasts |
| Held out | Sep 2–5, 2026 | 1,019 → 1,019 | Reserved for frozen evaluation |

Preprocessing keeps each root and its descendants together and purges roots whose visible graph crosses a split boundary. See [data provenance and preprocessing](FREEINFERENCE.md) for the pinned download and label definitions.

## Validation comparison

The unconditional comparator assigns equal weight to all training outcomes. The conditional baseline mixes whole outcome tuples from global, harness, harness/model, and harness/model/prompt-bucket groups. Each matching group contributes a fraction of mass equal to its size divided by its size plus the smoothing strength. Sparse or unseen groups borrow from broader groups. Validation selected two prompt buckets (32,212-token boundary) and smoothing strength 20 from nine configurations. The deepest matching group was the prompt bucket for 734 validation roots and the harness for 19.

| Target | MAE, unconditional → conditional | CRPS, unconditional → conditional | P10–P90 coverage, unconditional → conditional |
| --- | ---: | ---: | ---: |
| LLM requests | 16.12 → 15.93 | 12.81 → 12.61 | 90.4% → 85.9% |
| Tool calls | 19.33 → 18.90 | 15.22 → 14.80 | 91.9% → 85.9% |
| Observed duration (s) | 531.43 → 519.67 | 454.17 → 443.74 | 83.4% → 81.8% |

The mean of the three normalized marginal CRPS scores improves by **2.24%**. Descriptive paired 95% validation-root bootstrap intervals for conditional minus unconditional CRPS are [−0.46, +0.05] for LLM requests, [−0.76, −0.09] for tool calls, and [−19.96, −1.48] seconds for duration. These intervals use the same validation roots as model selection. The mean duration interval widens from 860 to 1,052 seconds. Overall coverage leaves tail and per-harness calibration open, and marginal CRPS leaves joint calibration untested.

## Learning curve

Scores are the mean of three marginal CRPS ratios against the full-training unconditional comparator. Lower is better. Each seed uses nested random subsets of training roots, recomputes the bucket boundary, and holds the selected bucket count, smoothing, and validation roots fixed.

| Training fraction | Roots | Conditional mean | Conditional P10–P90 across draws | Same-size unconditional mean |
| --- | ---: | ---: | ---: | ---: |
| 10% | 144 | 1.0135 | 1.0051–1.0189 | 1.0051 |
| 25% | 359 | 1.0020 | 0.9959–1.0095 | 1.0037 |
| 50% | 717 | 0.9903 | 0.9806–0.9962 | 1.0008 |
| 75% | 1,075 | 0.9822 | 0.9747–0.9910 | 1.0001 |
| 100% | 1,433 | 0.9776 | Single full fit | 1.0000 |

The curve shows diminishing returns. Relative gains at the last two increments are 0.81% and 0.47%, and all three mean marginal CRPS scores still improve at the final increment. The ten-draw bands measure training-subset variability at a fixed validation set. A plateau claim needs paired uncertainty for the increments against a practical threshold, a calibration and duration pinball check, and a chronological or time-block sensitivity check. The existing curve falls short of that evidence.

## Reuse and sample

`EmpiricalPredictor.load(path).predict(snapshot, "initial_user_turn")` accepts a preprocessing admission snapshot. The minimal input carries `event`, `horizon`, `cutoff_ms`, the three model values in `features`, and matching timestamps in `feature_available_ms`. The forecast carries the cutoff, horizon, marginal quantiles, deepest matching group depth and size, and weighted joint support. Use `forecast.sample(count, seed=42)` for reproducible draws in LLM count, tool count, duration-ms order.

For the sanitized Sep 1 validation admission (shard line 141, `hermes`, `glm-5.3-flash`, 45,725 prompt tokens), the deepest group has 27 training roots.

| Target | P10 | P50 | P90 | Observed |
| --- | ---: | ---: | ---: | ---: |
| LLM requests | 1 | 7 | 34 | 7 |
| Tool calls | 0 | 7 | 38 | 6 |
| Observed duration (s) | 9.017 | 192.852 | 863.464 | 168.889 |

## Scheduling and cache use

- **Deadline-aware scheduling.** Compare deadline slack to a duration quantile or sum empirical mass above the slack for a trace-based overrun estimate. The count distributions add demand signals for possible follow-up LLM admissions and tool invocations. Live deadline policy needs serving-side calibration because the gateway endpoint includes waits and overlap and can omit trailing tool work or work beyond a release boundary.
- **KV-cache decisions.** Additional LLM request count gives a coarse signal for reuse opportunities, and the input prompt estimate gives a rough starting context-size signal. Duration gives an overall horizon. Retain, evict, and prefetch decisions still need time to next use, reusable prefix and hit probability, token growth, KV footprint, and recompute cost. Total tool count supplies no tool wait forecast.

## Reproduce and share

From the repo root with the pinned shards available, use fresh output directory names.

```sh
python3 -m unittest prediction.predictor.test_freeinference_preprocess prediction.predictor.test_empirical_baseline
python3 -m prediction.predictor.preprocess_freeinference --output milestone-preprocessed
python3 -m prediction.predictor.evaluate_empirical_baseline --source milestone-preprocessed --output milestone-baseline
```

Preprocessing materializes all splits. The evaluator opens only training and validation snapshots and targets and writes `model.json` and `validation.json` under the gitignored `data/freeinference/milestone-baseline/`.

The verification run passed all nine targeted unit tests and checked all 15 source shards against their manifests. All 30 train and validation output shards matched the previous run after decompression. Both baseline artifacts matched byte-for-byte, and loading the fresh model reproduced the sample forecast and seeded joint draws. The held-out split was never scored.

Share the validation comparison, learning curve, sample, and admission interface as the first milestone. The empirical support ends at training-observed outcomes, and the forecast has no explicit censoring model. Update-time forecasting, tail and joint calibration, plateau confirmation, serving-policy validation, future DAG prediction, and the held-out evaluation remain open.
