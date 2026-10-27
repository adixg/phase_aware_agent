# Observed session sequences

## Overview

Eight complete observed session groups were extracted from the local TraceLab DuckDB using a read-only connection. Four use Claude and four use Codex. All recorded rounds in each selected group are included; complete observed coverage does not establish that the underlying task finished or that collection captured its full lifecycle.

## Selection methodology

Session grouping uses `(provider, user, project, session_id)`, with null-safe matching. Candidates have 3–120 rounds and at least one tool call. For each provider, select distinct sessions in this order:

1. Short: 3–8 rounds, fewest rounds first.
2. Longer: 80–120 rounds, most rounds first.
3. Tool-diverse: 10–40 rounds, most distinct tool names first, then fewest rounds.
4. Overlapping calls: at most 30 rounds, at least two calls in a round, highest maximum calls per round first, then fewest rounds; require a positive overlap within a round.

All ties use the smallest source `round_pk`. Already-selected groups are skipped. Tool counts and diversity are calculated through `round_pk` joins. The 120-round cap bounds export size; longer sessions exist outside this sample. These are purposive examples of structural variation, not a random or statistically representative sample. No timestamp-completeness filter was applied; the selected sessions happen to have no missing tool-result timestamps.

## Data schema

`sample_sessions.json` contains a format version, source database SHA-256, UTC timezone, and a `sessions` array. Session identifiers are export-local pseudonyms. Original session, user, project, model, and provenance strings are omitted. Tool-call identifiers are replaced consistently within each session; identifiers seen only in result or continuation references also receive aliases.

Each session includes provider, selection reason, ordered `rounds`, and `observed_overlaps`. Each round contains:

- `round_pk`: source surrogate key used for all joins; `ingest_seq`: source ingestion ordinal; `round_index`: source session ordering index.
- `tools`: source-order calls with `tool_index`, pseudonymous `tool_call_id`, tool name, emission/result timestamps, wall/internal/effective latency in milliseconds, effective measurement source, error flag, command status/exit code, and continuation reference.
- `timing_events`: `event_index`, event type, observed timestamp, source `round_pk`, tool index, and aliased call reference when recorded. Event content is excluded, including text, reasoning, user input, and tool-result payloads.
- `observed_input_tool_call_ids`: unique call references explicitly attached to tool-result events in that round. These expose observed input relationships, not a synthesized dependency graph.

Nulls remain null. `effective_latency_ms` selects non-null internal latency first, otherwise wall latency, including zero values. This derived selection is not a new timing measurement. `observed_overlaps` lists call pairs whose recorded emission/result intervals intersect with strictly positive duration; `overlap_ms` is calculated from their endpoints, across the entire session.

`session_summary.csv` records provider, selection reason, round/call/type counts, tool names, maximum calls per round, overlapping-pair count, missing result timestamps, and observed start/end/span. The span is the difference between the earliest and latest stored event or tool endpoint; it includes gaps and is not active execution time.

## Ordering and dependencies

Rounds are ordered by `(round_index, ingest_seq)`, tools by `tool_index`, events by `event_index`. Tool results can be attached to the following round while the originating call remains in the preceding round. Source ordinals resolve recorded order; they do not impose a serial schedule. Timestamp ties retain source indices. Overlapping intervals remain intact and are exported separately, without treating all calls in a round as either serial or parallel.

`continuation_of_tool_call_id` is a recorded command-lifecycle link for polling interactions. A reference may point outside the observed group; no missing parent is fabricated. Input references and continuation links should be resolved only when their aliased identifier exists among exported calls. No dependency is inferred merely from temporal adjacency.

## Limitations and workload modeling

Observed timestamps are log timestamps, not direct scheduler or inference-server instrumentation. Overlapping logged intervals support concurrent outstanding calls but do not prove simultaneous CPU execution. LLM start/end timestamps and inference service durations are not synthesized. Text/reasoning events retain timestamps only. Missing events, timestamp resolution, interruptions, and command polling can affect interpretation. A returned tool interaction need not mean the underlying command completed.

These sequences can provide trace-informed phase ordering, call batching, observed input relationships, and interval overlap. Empirical duration distributions can supply alternative durations, conditional on tool/provider and measurement quality. Independent resampling is an explicit modeling assumption and discards temporal and session correlations. Neither resource demand nor deadlines are measured here.

## Reproduction

Use the exact database SHA-256 embedded in the JSON, DuckDB read-only mode, the selection rules above, and child-table joins on `round_pk`. Read only timing/identity/latency/status fields, alias identifiers, and omit payloads. The source database is not included. See the parent README for provenance and source analysis paths.
