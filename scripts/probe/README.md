# Allocation & environment probe — Phase-Aware Agent Scheduling

Four files. Drop them in your scratch directory, keeping `bench/` as a subfolder.

```
~/scratch/
├── submit_probe.sh          # submits, records submission time
├── check_probe.sh           # reports queue wait + where results are
├── probe_env.sbatch         # the job itself
└── bench/
    └── contention_probe.py  # contention calibration
```

## Run

```bash
cd ~/scratch
chmod +x submit_probe.sh check_probe.sh
./submit_probe.sh a100      # or: ./submit_probe.sh all
```

Then poll:

```bash
./check_probe.sh
```

`QUEUE_WAIT` with a trailing `+` means still waiting and counting. Once the job
starts, the number freezes at the true wait. The `%R` reason column at the
bottom tells you *why* it's pending (`Resources` = GPUs busy, `Priority` =
other users ahead of you, `QOSMaxGRES` = you've hit a quota).

## Read results

```bash
cat results/probe-<JOBID>/SUMMARY.txt      # verdicts, one line per check
less results/probe-<JOBID>/detail.log      # full raw output
cat results/probe-<JOBID>/contention.json  # calibration curve
```

Every check prints `[PASS]`, `[WARN]`, or `[FAIL]`. Section 8 gives counts and a
ready / not-ready verdict.

## What it checks

| § | Area | Why it matters here |
|---|------|--------------------|
| 1 | GPU identity, VRAM, **exclusivity**, DCGM | Shared GPU ⇒ contaminated latency measurements |
| 2 | Core count, affinity, `taskset` | CPU starvation masquerades as scheduler queueing delay |
| 3 | System RAM | Dozens of concurrent sessions + retrieval index |
| 4 | Filesystem types, node-local scratch, write speed | Trace logging latency lands inside your metric |
| 5 | Internet, HF cache, staged models | Compute nodes are usually offline |
| 6 | torch / vLLM / transformers / retrieval deps | |
| 7 | **Contention calibration** | The one that can kill the project |

## Why `--time=00:30:00`

Short walltime is the single biggest lever on queue wait. This probe is
deliberately sized to schedule fast so you learn the real A100 wait quickly. It
also means the measured wait is a *lower bound* — a 12-hour request will wait
longer.

## Section 7 is the important one

The four policies in §5 of the proposal can only differ when there is a queue to
reorder. The calibration sweeps concurrency and finds the knee where latency
starts climbing. If it reports **no knee found**, your weeks 7–8 experiments
would have all four policies tie — not because phase-awareness doesn't work, but
because the server was never loaded. Fix by raising concurrency past
`max_num_seqs`, lengthening prompts/outputs, or lowering `--max-num-seqs`.

## Differences from the DSpark script

- **No Hopper gate.** That project needed FA3 (compute cap 9.x); this one needs
  nothing beyond Ampere. A100, L40S, and A40 are all valid, which is why A100
  is the default and why queue waits should be much shorter.
- **No `set -e`.** This is a diagnostic — it reports every gap in one run
  instead of dying on the first.
- **`$MODEL` is resolved, not assumed.** The DSpark script referenced `$MODEL`
  without ever defining it; under `set -u` that aborts. Here it's discovered
  from the HF cache, with `MODEL=... sbatch` as an override.
- **16 cores / 96GB**, not 8/64 — the scheduler, agent sessions, and tools are
  all CPU-side and run concurrently with serving.

## Before the real runs

Pin the server and scheduler to disjoint core sets, fix `--max-num-seqs` and
`--gpu-memory-utilization` identically across all four policies (the "fixed
resources" condition in §2), write trace logs to node-local scratch, and record
node identity in every trace so you can detect cross-node variance.
