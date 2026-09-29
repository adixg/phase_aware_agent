# phase_aware_agent
# RAG Scheduler Project — Team Schedule & Repo Structure

## Setup

What exists so far: a LangGraph ReAct agent ([agent/graph.py](agent/graph.py)), a
phase tracer ([serving/instrumentation/](serving/instrumentation/)), a toy local
retriever, and loaders for all four workloads (AgenticRAGTracer, HotpotQA, MuSiQue,
calculator). There's no vLLM backend yet — everything runs against a scripted
`MockLLM`, so only the `calculator` workload produces meaningful `correct` results.

```bash
# 1. Conda env (Python >= 3.10)
conda create -n phase-aware-agent python=3.10 -y
conda activate phase-aware-agent

# 2. Install the package + langgraph (editable, so agent/serving/... are importable)
pip install -e ".[test,data]"   # test: pytest; data: pyarrow, needed only for step 3's HotpotQA fetch

# 3. Download the datasets into ./data (gitignored, ~80MB, network required)
python scripts/fetch_workloads.py               # all four
# or just one: python scripts/fetch_workloads.py hotpotqa

# 4. Verify: run the test suite
pytest tests

# 5. Verify: run one traced job per workload
python -m agent.run_one --workload calculator --index 0
python -m agent.run_one --workload agentic_rag_tracer --index 0
```

Each `run_one` call writes a trace to `results/traces/<job_id>.jsonl` — one JSON line
per phase span (`llm_generate` / `retrieve` / `tool_exec`), plus a final
`job_summary` line with `workload`, `task_id`, `answer`, `correct`, etc.

## Team Schedule

| Week | Mallika (Agent/Workload) | Aditya (Infra/Baselines) | Andrew (Prediction/Analysis) |
|---|---|---|---|
| 1–2 | Set up AgenticRAGTracer + comparison workload | Set up vLLM, pick pilot model, confirm A100 | Scope predictor design; **all three** run the lit-watch sweep |
| 3–4 | Verify task quality on the built agent | Build phase/resource instrumentation + open-loop harness | Add calculator microbenchmark w/ Track 1; sketch confidence-gating math |
| 5 | **Lead**: collect traces, characterize resource signatures | Support trace collection | Start feature design for predictor from collected traces |
| 6–7 | Support policy integration into agent loop | **Lead**: build trace replay; implement FCFS, length+wait, attained-service, current-phase | Prep predictor training pipeline |
| 8–9 | Support corpus/query needs for predictor eval | Integrate predictive policies into replay harness | **Lead**: train predictor, build confidence gating, sweep load/latency/accuracy |
| 10 | Own task-quality checks during live run | **Lead**: live A100 validation | Run prediction-error, overhead ablations during live run |
| 11 | Write agent/workload + methodology sections | Write infra/baselines + systems sections | **Lead**: analysis, results, figures; re-run lit-watch |

### Shared, no matter the split
- **Week 10 live-run day**: all three present — the one irreversible, expensive data collection point.
- **Lit-watch (Weeks 1–2 and 11)**: rotate or do together — the subfield moves fast enough that this is cheap insurance.
- **Final report/presentation**: each person drafts their own section first, then one person (typically Track 3, since they're synthesizing the hypothesis test) does a unifying pass.

---

## Repo Structure

```
rag-scheduler/
├── agent/                 # Track 1: LangGraph agent, workloads
│   ├── graph.py
│   ├── workloads/          # AgenticRAGTracer, HotpotQA/MuSiQue, calculator loaders
│   └── retriever/
├── serving/                # Track 2: vLLM integration, instrumentation, baseline policies
│   ├── policies/
│   │   ├── fcfs.py
│   │   ├── length_wait.py
│   │   ├── attained_service.py
│   │   └── current_phase.py
│   ├── instrumentation/    # phase/resource tracking
│   └── harness/            # open-loop arrival generator, admission window
├── prediction/              # Track 3: predictor, confidence gating, predictive policies
│   ├── predictor/
│   ├── gating/              # CVaR-based confidence gating
│   └── policies/
│       ├── predicted_next_phase.py
│       └── perfect_next_phase.py
├── replay/                  # shared: trace replay engine (built Wk 6-7, used by all)
├── configs/                  # experiment configs (load levels, tool latency, corruption levels)
├── notebooks/                # analysis notebooks, kept OUT of policy code
├── results/                  # gitignored except summary CSVs/figures committed at milestones
├── scripts/                   # A100 live-run launch scripts
├── tests/
└── docs/
    ├── lit-watch.md           # running log from Weeks 1-2 and 11 sweeps
    └── decisions.md           # lightweight ADR log — pilot model choice, why threshold=10%, etc.
```

### Design notes

- **Policies are the seam.** Every policy (baseline or predictive) implements the same interface — roughly `next(ready_jobs, system_state) -> job_id` — so the replay harness and the live A100 harness in `serving/harness/` can run any policy from `serving/policies/` or `prediction/policies/` without caring who wrote it. This lets Track 2 and Track 3 work in parallel without blocking each other.
- **`docs/lit-watch.md`** is append-only: one dated entry per Week 1–2 and Week 11 sweep, with paper titles/arXiv IDs and a one-line note on whether it affects the framing. This is the evidence trail if a reviewer asks whether overlap was checked.
- **`configs/` is the source of truth for experiments.** Every sweep (load × tool-latency × prediction-corruption) should be a checked-in YAML/JSON, not a notebook cell edited by hand — this is what makes Week 10's live validation reproducible.

### Branching & review

- `main` stays runnable at all times.
- Feature branches: `track1/*`, `track2/*`, `track3/*`, PRs into `main`.
- Require one **cross-track** review per PR once integration starts (~Week 6) to catch interface mismatches early.
- Tag milestone releases (`v0-baselines`, `v0-predictor`, `v0-live-run`) so results behind any report claim can be regenerated exactly.

### CI

- Lint + a smoke test running each policy against a tiny synthetic trace (no GPU needed) on every PR.
- A separate, manually-triggered workflow for anything needing the A100 — don't burn GPU hours in CI.
