"""Check the simulator against our own measured vLLM runs: replay each run's jobs with the
measured arrivals (or concurrency), token counts and tool times, and compare job completion time.

    python analysis/replay/validate.py results/traces/run-... [more runs]
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
import sys

from analysis.replay.sim import Config, Profile, Req, Session, Simulator, _annotate

# decode and per-request overhead fitted on the closed-loop run 6420496 (22.4 ms/token, mean
# TTFT 157 ms less ~18 ms of prefill); the open-loop run 6420500 is the held-out check
L40S = Profile(decode_ms=22.4, prefill_ms=0.046, overhead_ms=140.0)
TOOL_PHASES = ("retrieve", "retrieve_delay", "tool_exec", "wait_cpu")


def load_run(run_dir: str) -> tuple[list[Session], dict, dict[str, float]]:
    manifest = json.load(open(f"{run_dir}/manifest.json"))
    arrivals = {j["job_id"]: j.get("arrival_ts") for j in manifest["jobs"]}
    measured = {j["job_id"]: j.get("jct_s") or j["wall_s"] for j in manifest["jobs"]}
    sessions, t0 = [], min((a for a in arrivals.values() if a is not None), default=0.0)
    for p in sorted(glob.glob(f"{run_dir}/*.jsonl")):
        jid = p.rsplit("/", 1)[-1][:-len(".jsonl")]
        spans = [r for r in map(json.loads, open(p)) if "phase" in r]
        reqs, gap = [], 0.0
        for s in spans:
            if s["phase"] == "llm_generate":
                if reqs:
                    reqs[-1].gap = gap
                gap = 0.0
                m = s["meta"]
                reqs.append(Req(m["prompt_tokens"], m.get("cached_tokens") or 0, m["output_tokens"], None,
                                "__none__", "__start__", len(reqs), 0, False))
            elif s["phase"] in TOOL_PHASES:
                gap += s["duration_s"]
        if not reqs:
            continue
        reqs[-1].last_in_turn = True
        z = _annotate(reqs, L40S)
        sessions.append(Session(jid, (arrivals.get(jid) or t0) - t0, reqs, z))
    return sessions, manifest, measured


def real_llm_mean(run_dir: str) -> float:
    durs = [r["duration_s"] for p in glob.glob(f"{run_dir}/*.jsonl") for r in map(json.loads, open(p))
            if r.get("phase") == "llm_generate"]
    return st.mean(durs)


def main(run_dirs: list[str]) -> None:
    for d in run_dirs:
        sessions, manifest, measured = load_run(d)
        closed = manifest.get("concurrency") if manifest.get("mode") != "open_loop" else None
        slots = manifest.get("llm_slots") or 8
        sim = Simulator(sessions, Config(slots=slots, kv_tokens=math.inf, profile=L40S, closed=closed)).run()
        ours = sorted(t.latency for t in sim.turns)
        real = sorted(measured.values())
        if closed:
            print(f"  throughput: sim {len(ours) / max(t + d for t, d in sim.busy):.2f} jobs/s, "
                  f"real {manifest['jobs_per_s']:.2f}")
        service = st.mean(d for _, d in sim.busy)
        p95 = lambda v: v[int(0.95 * len(v))]
        print(f"{d.rsplit('/', 1)[-1][:60]}\n  jobs {len(ours)}/{len(real)} | mean JCT sim {st.mean(ours):.2f} real {st.mean(real):.2f}"
              f" | p95 sim {p95(ours):.2f} real {p95(real):.2f} | LLM call sim {service:.3f} s, real {real_llm_mean(d):.3f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
