# FreeInference offline DAG predictor

## Reproduce the audit

From the repository root, use Python 3.10+ and the `zstd` CLI. The audit ran on Python 3.13.6. The existing `data/` ignore rule covers downloads, reports, and future derived tables. The pre-existing May sample is outside the cohort. No Python packages need to be installed for inspection.

```sh
hf download harvardMadsys/freeinference_agentic_trace --repo-type dataset \
  --revision bb26cf72c9c20f2b05f765d88548156a74d3c253 \
  --local-dir data/freeinference \
  --include 'release/2026-08-23_2026-08-29/manifest.json' 'release/2026-08-23_2026-08-29/traces/*.jsonl.zst' \
            'release/2026-08-30_2026-09-05/manifest.json' 'release/2026-08-30_2026-09-05/traces/*.jsonl.zst'
python3 prediction/predictor/inspect_freeinference.py
```

The audit compares each streamed decompressed file's SHA-256, byte count and line count with its manifest. It parses the trace structure after omitting numeric prefix arrays, whose original bytes are included in verification. The JSON report includes per-shard field coverage, tool categories, link reasons, integrity checks, and sanitized examples with file and 1-based line locators. Keep both August 29 shards and key records by release week and root session ID. The cohort is selected by top-level session start day in UTC, with nested execution attributed to its root. [Pinned Hugging Face release](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/tree/bb26cf72c9c20f2b05f765d88548156a74d3c253/release)

## Audit results

All 15 shards matched their release manifests. They contain 149,362,447 compressed bytes and 38,976,498,480 decompressed bytes. All 3,239 top-level rows parsed and match their UTC start date. The four August 29 rows in the second release week are distinct from the 194 in the first. There are no repeated root IDs.

| Metric | Count |
| --- | ---: |
| Top-level sessions | 3,239 |
| Matched nested sessions | 416 |
| LLM requests, including nested | 350,469 |
| Nested LLM requests | 7,312 |
| Emitted tool calls, including nested | 360,562 |
| Nested tool calls | 12,716 |
| Calls categorized as Subagent | 1,292 |

The publisher excluded 177 and 272 sessions in the two respective weeks because the first request did not identify a harness. Within the release, 177 TTFTs, 5,466 provider cached-token counts and 1,030 tool-batch gap values are null. There are two negative tool-batch gap values and five overlapping consecutive request pairs within the same chain. The audit found zero empty sessions, invalid step indices, reversed session or request spans, or missing provider prompt/completion token counts. Maximum nested depth is one in this cohort. The 16 released tool categories include Read, Edit, Browser, Subagent, and shell subcategories such as Interpreter and Search. Among consecutive chain links, 323,468 were tool-triggered with tool-call-ID evidence, 13 were tool-triggered with assistant-message evidence, and 23,333 were user-triggered with assistant-message evidence.

## Reconstruct the observed DAG

Use stable node IDs local to the root, combining the nested path, request step and tool call index. Each root or child session has a published ordered request chain. `step` gives chain position, `turn_index` groups user turns, and `step_trigger` and `link_reason` describe the publisher's reconstruction. Preserve timestamps and temporal overlaps independently of graph edges.

| Object | Attributes and timing |
| --- | --- |
| Session or subagent entry | Harness, nested depth, gateway-observed start and end, root and spawning-tool reference. Store the session span as a container attribute and exclude it from additive node-duration or critical-path sums. |
| LLM request | Gateway start/end and TTFT, requested model, anonymized provider, streaming, provider prompt/completion/cache tokens, input role counts, message count, tool definitions, finish reason, input mutation and block IDs. Gateway span can include network, queues and provider processing. |
| Tool call | Response call index, normalized category/name, redacted argument skeleton/programs, parse status, outcome and result tokens. Tool timestamps and exclusive service duration are unavailable. Store the released latency once per emitting-request batch. |

Represent these edges with explicit provenance:

- **Observed emission** from the LLM response to each emitted tool call. Call index is response order. Emission describes logical causality even if streaming permits execution before the response closes.
- **Publisher-supported continuation** between linked requests. A tool-ID match supports result feedback at the emitting-request level. A single emitted call supports an inferred tool-to-next-request edge when the link indicates tool feedback. For multiple calls, keep a batch-level continuation annotation with unknown contributing subset. Text-based request linking provides weaker continuation evidence. Step order and timestamps alone provide temporal evidence.
- **Publisher-inferred spawn** from the tool invocation to a nested subagent entry, then entry to the child's first request. The exporter identifies candidate spawn tools and matches the delegation prompt to a same-account child in a time window. Preserve the match as uncertain provenance. Null nesting leaves actual launch status unknown. A child-to-parent completion dependency needs additional evidence, such as a join or returned-result ID with matching spans.
- Leave sibling tool order and concurrency unknown. Gateway timestamp overlap can establish overlap between recorded request windows. It cannot place tool execution or prove concurrent GPU service. Validate acyclicity and flag timing anomalies separately.

Released `latency_ms` for qualifying tool batches is the next request's gateway start minus the emitting request's gateway end. It can include multiple tools, harness overhead, remote waits, and transport. Request gaps, tool service time, CPU time and GPU service time require distinct fields. Week boundaries and the publisher's session reconstruction limit visibility into open work and earlier context. [Tool wait calculation](https://github.com/HarvardMadSys/freeinference_agentic_trace/blob/main/export/release/requests_table.py), [request linkage](https://github.com/HarvardMadSys/freeinference_agentic_trace/blob/main/export/sessions/link.py), [spawn matching](https://github.com/HarvardMadSys/freeinference_agentic_trace/blob/main/export/sessions/spawns.py)

## Sanitized session example

September 1 shard, line 141 (1-based), under release week `2026-08-30_2026-09-05`. `R` is a root LLM request and `C` is a child LLM request. Solid arrows denote release-observed emission or child membership, and dashed arrows denote supported continuation or inferred spawn. The two Browser tools have unknown order and the batch feedback link leaves their contributing subset unspecified.

```text
R0 ──► Browser 0
 ├───► Browser 1
 └···► R1 [tool-call-ID batch feedback; gap 4.596 s]
        └──► Other ···► R2 ──► Subagent tool ···► R3 ···► R4 [user]
                                    └···► Child session ──► C0
                                           C0 ──► Git/VCS ···► C1
                                           C1 ──► Git/VCS ···► C2
```

R3 spans 142.823–168.889 s relative to the root start. The child's gateway window spans 142.479–168.232 s, with gaps between its three LLM requests. R3 begins before the child finishes. The example therefore has no supported child-completion edge into R3.

## Features and labels

Use admission of the initial root LLM request as `t=0`, immediately before gateway execution. Make the primary forecast the initial user turn and descendants of tools it emitted. Forecast full released-session totals separately since later user turns and idle intervals can dominate them. Count the initial LLM request in the initial-turn LLM total, and expose the known request separately so the scheduler can obtain the count beyond it.

- **At admission** use the harness if available, requested model, stream flag, local input token estimate, role-token/message counts and input tool definitions. Provider identity is eligible only if routing is already known. Use input-derived counts as the deployable prompt-size feature. A provider prompt-token count can serve as a measured label or offline comparison. Prefix reuse can use an admission-time serving-cache lookup if available. Published block IDs are week-scoped and should not be used as cross-week categorical features.
- **At updates** use admitted requests, completed-response token telemetry, emitted tools, tool outcomes and result sizes only after delivery, elapsed time, branch events already exposed to the scheduler, and gaps once their endpoint arrives. Put a timestamp on each feature's availability. Exclude publisher-derived features whose availability cannot be demonstrated at the cutoff.
- **Leakage** includes future selected models, completion/cache tokens from unfinished calls, future mutation and category fields, completed counts, later user turns and terminal timestamps. Session and account IDs serve as keys or slice/grouping metadata. Fit preprocessing, category vocabularies, priors and calibrators without test data.
- **Reliable within the release** are request and emitted-tool counts, category histograms, any emitted tool, and LLM gateway span. Matched nested-child counts and any matched spawn are observed release labels with incomplete recall and possible mismatches. Report spawn-tool emission and matched nesting separately, where spawn tools can be identified.
- **Duration** is wall-clock time to the last visible in-scope gateway response, including linked descendants. Keep root-only span and tool-batch gap as separate labels. This endpoint misses trailing tool-only work and may be truncated. Keep endpoint/missingness flags and use explicit censoring when known, with a sensitivity analysis for boundary cases. Actual tool duration, exact joins, resource use and GPU service require harness spans, result IDs, parent/child and join IDs, plus serving telemetry for queue, prefill, decode, device and KV residency.

## Training tables and output

| Table | Fields |
| --- | --- |
| `snapshots` | Release/root key, cutoff time and event ordinal, horizon, split, admission features, causal prefix aggregates, masks and feature schema version |
| `targets` | Snapshot key, remaining root and nested LLM counts, emitted tool counts by category, matched-spawn count and branch flags, remaining gateway horizon in ms, censoring and scope flags |
| `nodes`, `edges` | Root-local stable keys, node type, attributes, event/availability bounds, edge type and evidence. Reference known and future subgraphs by snapshot, with unknown edges left unlabeled. |

Start with empirical joint outcome distributions by harness, model and prompt-size bucket, backed off to coarser groups when sparse. Compare to regularized count/branch models and quantile or survival models for duration. For structure, sample a sequence of next-event choices (stop or continue, emitted tool batch, matched spawn, child continuation), conditioned on harness and current history. Recurse for nested branches with a budget and report probability mass truncated by limits. Sample token, duration and gap marks jointly or conditional on event type to retain correlations. Build each sample from emission and continuation templates with explicit edge provenance and unknown sibling order. Derive marginal probabilities from weighted samples and keep release-observable spawn labels distinct from runtime promises.

A minimal interface is `predict(snapshot, horizon) -> Forecast`. Return horizon and cutoff, count and duration quantiles with units and duration kind, tool-category and matched-spawn probabilities, and optional weighted DAG samples with node/edge types, token and timing marks, and unknown order. Use probabilities as forecast inputs for later GPU/KV policy decisions, with a separate serving-calibrated resource model.

## Evaluation

- Train on root UTC dates Aug 23–29, validate Aug 30–Sep 1 and reserve Sep 2–5 as a single untouched test period (1,437 / 783 / 1,019 roots before purging). Keep every snapshot and descendant with its root. Purge earlier-split examples whose feature or label observation window crosses the next split boundary. Flag weekly-cut artifacts and report post-purge sizes plus known-account and unseen-account slices. Choose calibration, hyperparameters and stopping rules only on train/validation.
- Score count forecasts with MAE and discrete CRPS. Use Brier score, log loss, PR-AUC and reliability plots for tool/subagent occurrence. Use pinball loss and interval coverage/width for duration quantiles, plus a censor-aware score when applicable. For DAGs, align nodes by deterministic matching and report typed node/edge precision and recall on known labeled relations with label coverage. Exclude unknown tool order and joins from edge scoring. Report session-bootstrap intervals and per-harness/cutoff slices.
- Run nested training prefixes at 10%, 25%, 50%, 75% and 100% of training roots in chronological order. Use the same validation roots and compare paired confidence intervals for the improvement at each increment. Predeclare a practical improvement threshold, such as under 2% relative gain in count CRPS and duration pinball loss for two consecutive increments, with stable calibration. Use repeated nested block subsamples within training as a sensitivity check for time drift. Freeze the method before scoring the test period once.

Next, materialize normalized event/edge tables and audit split-boundary/turn censoring, implement causal snapshot extraction with feature-availability checks, then add empirical baselines and validation scoring. Add serving telemetry and tool spans before evaluating resource-cost or true critical-path forecasts.

## Preprocessing implementation

From the repository root, with the raw shards in place:

```sh
python3 -m prediction.predictor.preprocess_freeinference
python3 -m unittest prediction.predictor.test_freeinference_preprocess
```

The command streams and verifies the 15 v1 shards, builds a temporary output tree and publishes it after validation. It uses the Python standard library and the `zstd` CLI. The destination is `data/freeinference/preprocessed/`. To write a new output, set `--output` to a new directory name under `data/freeinference/`. The default materializes admission snapshots. Add `--updates` to include root-response completion snapshots for the initial turn.

| Output | Row schema |
| --- | --- |
| `{split}/*.graphs.jsonl.gz` | One row per retained root. Release/root key, source coordinates, root bounds, typed nodes, typed evidence-bearing edges and tool-phase intervals. The graph covers the released root session and descendants. |
| `{split}/*.snapshots.jsonl.gz` | One row per root/cutoff. Join on `(key, ordinal)`. Includes cutoff, horizon, event, feature values and per-feature availability timestamps. Use only the `features` mapping as model input. |
| `{split}/*.targets.jsonl.gz` | One row per snapshot. Remaining LLM and emitted-tool counts, category histogram and unknown count, matched-spawn and spawn-category labels, observed duration, incompleteness flags, and references to future nodes and in-scope edges. Edge indices are zero-based into the graph's `edges` array. Inferred edges retain their provenance through that join. |
| `manifest.json`, `example.json` | Split bounds and counts, release revision, quality checks, and one sanitized validation example. The example uses relative timestamps. |

For admission snapshots, the only observable event is the initial root request input. The prompt estimate is the sum of the five released input-role counts, including tool definitions. Tool definition names describe available tools. Emitted tool categories are labels at admission and become history only after response completion. The released harness is an input-side proxy whose availability should be checked against the deployment interface. Provider usage is available only after completion. Snapshot metadata, graph fields, and label fields are outside the feature vector.

Optional completion snapshots include completed root responses by the cutoff and exclude tied new admissions and phase endpoints. Their remaining LLM count includes any unfinished calls, including in-flight calls. History uses only the root stream since the release supplies no causal child-attribution notification time. Child gateway admission acts as a label-time proxy for matched subagent branching. Actual spawn and completion labels are null. The test set is materialized in its own directory for later evaluation, with no model fitting, vocabulary learning, or scoring in preprocessing.

For each adjacent tool-triggered transition with an emitting batch, the tool-phase row records both endpoints and the signed gap. Negative gaps keep the signed observation and have a null elapsed duration. Each tool preserves its original `released_latency_ms`. The implementation verifies that these fields repeat the qualifying batch gap exactly and are null outside qualifying batches. Tool timestamps and durations are null. For responses with multiple tools, sibling execution order is explicitly unknown. A tool-ID continuation provides request-level support even when per-tool feedback is ambiguous. The pipeline emits inferred tool feedback only for single-call candidates and omits child-to-parent joins.

Split membership follows UTC root admission. The full visible graph and all descendant timestamp bounds must fit inside that split, including the final cohort cutoff at September 6 00:00 UTC. Crossing roots are removed as a unit. The manifest reports purges even when their initial-turn labels could have fit. Release-week truncation and trailing tools can hide work. Observed remaining duration therefore ends at the last visible gateway completion in the initial turn or its descendants. It includes wall-clock idle and overlap and is never a service-time sum.

The example above has a 45,725-token input estimate, model `glm-5.3-flash`, harness `hermes`, 42 input messages, 32 available tool definitions, and zero completed requests or emitted tools at admission. Its initial-turn targets include 7 LLM requests (3 nested and 1 already admitted), 6 tools (2 Browser, 1 Other, 1 Subagent, 2 Git/VCS), 1 publisher-matched child, and 168,889 ms of remaining observed duration. The later user request R4 remains in the graph and falls outside those labels.

Next, fit training-only empirical distributions conditioned on harness, model and prompt size. Choose bucket boundaries and backoff with validation only. Reserve test artifacts for the frozen baseline.

## Admission baseline

```sh
python3 -m prediction.predictor.evaluate_empirical_baseline
python3 -m unittest prediction.predictor.test_empirical_baseline
```

The evaluator reads admission snapshot and target shards under train and validation. It checks row pairing, causal feature availability, temporal bounds and key disjointness. Training uses only harness, requested model and initial input-role token estimate. The generated graph's scalar summaries are already present in the targets. Dependency edges are outside the baseline's labels.

The model mixes empirical distributions along global → harness → harness/model → harness/model/prompt bucket. A group of size `n` supplies `n / (n + strength)` of the mass, with the remainder inherited from its parent. Missing and unseen groups back off. Mixtures reweight whole `(LLM count, tool count, duration_ms)` tuples, preserving their empirical dependence. Forecast support is bounded by the observed training outcomes.

The nine validation candidates combine 2, 4 or 8 train-quantile prompt buckets and smoothing strengths of 5, 20 or 80. The selection criterion is the mean of the three marginal CRPS ratios against the full unconditional empirical distribution. The selected model uses two buckets, with a split at 32,212 tokens, and strength 20. The curve fixes that configuration and recomputes prompt boundaries using each subset. It uses nested seeded random root subsets, the same validation set throughout, and a same-size unconditional comparator. The 10–90% bands describe variability across training draws. The paired root-bootstrap intervals compare validation errors after configuration selection and are descriptive.

```python
from prediction.predictor.empirical_baseline import EmpiricalPredictor

predictor = EmpiricalPredictor.load("data/freeinference/empirical-baseline/model.json")
forecast = predictor.predict(snapshot, "initial_user_turn")
duration_p90_ms = forecast.quantiles["remaining_observed_duration_ms"][0.9]
joint_draws = forecast.sample(100, seed=42)
```

The format includes cutoff, horizon, P10/P50/P90 marginal quantiles, the deepest matching group's size and depth, and weighted joint support. The current artifact supports admission snapshots. The same interface can dispatch later cutoffs to separately fitted update models. Duration quantiles refer to the observed gateway endpoint. The evaluator writes the reusable artifact and all validation metrics under gitignored `data/freeinference/empirical-baseline/`. Pass a new `--output` name to rerun.
