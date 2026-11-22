# FreeInference Oracle experiments

These experiments test where future workload information helps scheduling and whether
critical-path estimates help when turns contain subagent calls. The
[results summary](FINDINGS.md) records the paired comparisons.

| Experiment | Configuration | Main finding |
| --- | --- | --- |
| [Oracle resource sensitivity](oracle-sensitivity/) | Target load 0.7 and 1.1, unlimited or 1M-token KV, EDF and cheap versus Oracle LLF/SRPT, plus an EDF eviction comparison | At load 1.1 with 1M KV, Oracle LLF gained 42.91 percentage points (pp) over cheap LLF and 0.59 pp over EDF on feasible turns. Oracle eviction gained 0.28 pp over cheap eviction under EDF. |
| [Deadline demotion ablation](oracle-ablation/) | Load 1.1, both KV sizes, independent cheap/Oracle estimates for ordering and demotion | With 1M KV, changing LLF demotion alone raised feasible attainment from 49.67% to 91.48%. Changing ordering alone raised it to 50.45%. |
| [Overdue-only follow-up](oracle-ablation/policy-followup/) | Same load and KV settings, with cheap ordering and demotion only once a deadline has passed | Overdue-only LLF recovered 97.7% of the full feasible-attainment improvement from cheap LLF to Oracle LLF at 1M KV. |
| [DAG-aware replay](dag-replay/) | Load 1.1, EDF versus critical-path, remaining-work, and flattened-timing priorities, with no-join and plausible-join scenarios | With plausible joins, the DAG policy reached 97.51% feasible attainment versus 97.89% for EDF. |
| [All-turn subagent replay](dag-replay/all-turns/) | Same policy and join comparisons, with all eligible turns considered when reconstructing nested workflows | At 1×, every policy met 7 of 32 nested deadlines in both scenarios. |
| [Deadline sensitivity](dag-replay/all-turns/deadline-sensitivity/) | Replays 1×, 2×, 3×, and 8× with global deadline scaling and a nested-only diagnostic | At 8× global scaling without joins, DAG and EDF reached 91.93% and 91.91% all-turn attainment, respectively. Both met 18 of 32 nested deadlines. |

All experiments replay August 30 through September 1, 2026, with eight slots
and a modeled L40S profile. The baseline deadline classes are the 50th, 75th,
and 95th percentiles from August 23–29. DAG replay uses seed 0 and a 2,000 ms
residual threshold for plausible joins. EDF is earliest deadline first,
LLF is least laxity first, and SRPT is shortest remaining processing time.

## Reproduce

Use Python 3.10 or newer, Git, and the system `patch` command. Install the
experiment dependencies with `python -m pip install numpy orjson zstandard`.
The runs below were verified with Python 3.10.21, orjson 3.12.0, and zstandard
0.25.0.

The workspace setup reads replay commit
`a3dbeae41b7dac6c06f3fe1863ad1300e348f405` from the local Git object database.
If the object is missing, fetch the history of `origin/track2/vllm-backend`
(with sufficient history in a shallow clone). Setup applies the demotion hooks
only to a second simulator copy in the workspace. Shared ablation helpers are
copied into the follow-up directory during setup.

The FreeInference input directory must contain
`oracle-sensitivity/compact/` for the ten training and evaluation days,
`milestone-preprocessed/validation/` for the evaluation-day graphs, and
`release/2026-08-30_2026-09-05/traces/` for the corresponding compressed
traces. Use `--data /path/to/freeinference` if the data lives elsewhere.
Run setup from the repository root. The workspace path must be new.
Outputs are written in that workspace, and the original inputs are read
through symlinks.

```sh
work="$(mktemp -d)/oracle"
python analysis/oracle_experiments/prepare_workspace.py "$work"
cd "$work"

# Resource sensitivity, then ablation and policy follow-up.
python oracle-sensitivity/run_sensitivity.py
python oracle-sensitivity/summarize_sensitivity.py
python oracle-ablation/run_ablation.py
python oracle-ablation/report_ablation.py
python oracle-ablation/policy-followup/run_followup.py
python oracle-ablation/policy-followup/report_followup.py

# First-turn DAG reconstruction, then all-turn replay and deadline scaling.
python dag-replay/prepare.py
python dag-replay/replay.py
python dag-replay/report.py
python dag-replay/all-turns/prepare.py
python dag-replay/all-turns/replay.py
python dag-replay/all-turns/report.py
python dag-replay/all-turns/deadline-sensitivity/sweep.py
python dag-replay/all-turns/deadline-sensitivity/report.py
```

Each report step writes Markdown next to its workspace scripts. The sweep
writes separate global and nested reports and diagnostic CSVs. From the
repository root, the existing test command is `python -m pytest tests`.
With pytest installed, check both simulator copies from the workspace:

```sh
PYTHONPATH="$PWD/source" python -m pytest source/tests/test_replay.py
PYTHONPATH="$PWD/ablation-source" python -m pytest ablation-source/tests/test_replay.py
```

## Limitations

These are trace-driven simulations over three validation days using synthetic
deadlines and a modeled GPU profile. Oracle estimates use future trace data.
Feasibility is based on optimistic service time or the no-contention critical
path, depending on the experiment. The DAG prototype infers parent/child
relationships and tests alternative join assumptions because join dependencies
are unobserved. Its endpoint is the last visible modeled LLM response. The
32-case nested cohort makes individual deadline flips and tail latency
descriptive. Daily ranges describe variation across these runs.
