"""Check the replay simulator against a pressure_replay run on real vLLM, then report what the
simulator predicts for other schedulers on that same, now checked, workload.

The simulator gets exactly what was replayed: the workload's sessions (prompt, shared prefix, output,
gaps), vLLM's KV pool and max-num-seqs as slots, the calibrated prefill cost, the decode speed measured
during the replay, and for a CPU-tier run the measured post-eviction cost as its reload speed. vLLM
schedules first come first served, so the comparison is with the simulator's FCFS + LRU.

    python -m analysis.replay.validate_pressure results/traces/pressure-<jobid>_<mode>_pool<N> [more runs]
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

from analysis.replay.sim import Cheap, Config, Profile, Simulator, build_sessions

KV_BYTES_PER_TOKEN = 57344        # Qwen2.5-7B: 28 layers x 4 KV heads x 128 x (K, V) x 2 bytes


def sessions_of(workload: dict, speedup: float):
    """The workload as the simulator's input: shared prefixes are exactly `lcp` (outputs are not reused,
    since the replay's next prompt carries new tokens instead of the generated ones)."""
    raw = []
    for s in workload["sessions"]:
        reqs = s["requests"]
        rows = []
        for i, r in enumerate(reqs):
            nxt = reqs[i + 1] if i + 1 < len(reqs) else None
            rows.append({"prompt": r["prompt"], "lcp": r["lcp"], "completion": r["completion"], "append": False,
                         "user": r["user_turn"], "cat": "__none__" if nxt is None or nxt["user_turn"] else "Read",
                         "gap": None if r["gap_after_s"] is None else r["gap_after_s"] / speedup * 1000})
        raw.append({"sid": s["sid"], "start": s["arrival_s"] / speedup * 1000, "requests": rows})
    return raw


def profile_of(run: Path, reqs: list[dict], mode: str) -> Profile:
    cal = json.load(open(run / "calibration.json"))
    decode = st.median((r["total_s"] - r["ttft_s"]) * 1000 / max(1, r["completion"] - 1)
                       for r in reqs if r.get("total_s") and r["completion"] > 1)
    reload_ms = cal["after_eviction_ms_per_token"] if mode in ("lmcache", "native") else None
    return Profile(decode_ms=decode, prefill_ms=cal["prefill_ms_per_token"], overhead_ms=max(0.0, cal["overhead_ms"]),
                   reload_ms=reload_ms)


def session_times(sim: Simulator) -> dict[str, float]:
    out = {}
    for st_ in sim.states:
        t = next(t for t in sim.turns if abs(t.arrival - st_.turn_arrival) < 1e-9
                 and abs(t.budget - (st_.deadline - st_.turn_arrival)) < 1e-9)
        out[st_.s.sid] = t.latency
    return out


def check(run: Path) -> None:
    man = json.load(open(run / "manifest.json"))
    reqs = [json.loads(l) for l in open(run / "requests.jsonl")]
    errors = sum(r["error"] is not None for r in reqs)
    workload = json.load(open(man["workload"]))
    if man.get("limit_sessions"):
        workload["sessions"] = workload["sessions"][:man["limit_sessions"]]
    speedup = man.get("speedup", 1.0)
    mode = man.get("mode") or "recompute"
    pool = float(man["pool_tokens"])
    slots = int(man.get("max_num_seqs") or 8)
    profile = profile_of(run, reqs, mode)
    cpu_tokens = math.inf
    if profile.reload_ms is not None and man.get("cpu_gb"):
        cpu_tokens = float(man["cpu_gb"]) * 1e9 / KV_BYTES_PER_TOKEN

    real = {}
    for r in reqs:
        a = real.setdefault(r["sid"], [math.inf, 0.0])
        a[0], a[1] = min(a[0], r["ready_s"]), max(a[1], r["end_s"])
    real = {k: e - s for k, (s, e) in real.items()}
    real_hit = sum(r.get("cached_tokens") or 0 for r in reqs) / sum(r["prompt"] for r in reqs)

    raw = sessions_of(workload, speedup)
    sessions = build_sessions(raw, profile, unit="session")
    cheap = Cheap(sessions, profile)
    base = dict(slots=slots, kv_tokens=pool, profile=profile, deadline_mode="proportional", slack=100.0,
                cpu_tokens=cpu_tokens if profile.reload_ms is not None else math.inf)
    sim = Simulator(sessions, Config(policy="fcfs", **base), cheap).run()
    simulated = session_times(sim)

    keys = sorted(set(real) & set(simulated))
    a, b = [real[k] for k in keys], [simulated[k] for k in keys]
    corr = st.correlation(a, b) if len(keys) > 2 else float("nan")
    p95 = lambda v: sorted(v)[int(0.95 * (len(v) - 1))]
    print(f"\n== {run.name}: mode {mode}, pool {pool:,.0f} tokens, {slots} slots, {len(reqs)} requests ({errors} errors)")
    print(f"   profile: prefill {profile.prefill_ms:.4f} ms/token, decode {profile.decode_ms:.1f} ms/token under load, "
          f"overhead {profile.overhead_ms:.0f} ms" + (f", reload {profile.reload_ms:.4f} ms/token" if profile.reload_ms else ""))
    print(f"   session time  real: mean {st.mean(a):6.1f} s, p95 {p95(a):6.1f} s   simulated FCFS: mean {st.mean(b):6.1f} s, "
          f"p95 {p95(b):6.1f} s   (mean off by {100 * (st.mean(b) / st.mean(a) - 1):+.1f}%, per-session r = {corr:.2f})")
    print(f"   prompt tokens found in GPU cache  real {100 * real_hit:.1f}%   simulated {100 * sim.tokens_hit / sim.tokens_prompt:.1f}%"
          "   (vLLM's count may leave out CPU-tier hits)")

    print("   simulator on this workload, % of sessions on time (deadline = scale x time alone):")
    print("   " + "%-36s" % "" + "".join("%9s" % f"{s}x" for s in (1.1, 1.2, 1.5)))
    for name, extra in (("FCFS (vLLM's order)", dict(policy="fcfs")), ("EDF", dict(policy="edf")),
                        ("CacheWise", dict(policy="fewest", eviction="cachewise")),
                        ("Continuum", dict(policy="continuum", pin="ttl")),
                        ("oracle shortest-remaining", dict(policy="srpt_oracle")),
                        ("predicted shortest-remaining, 10%", dict(policy="srpt_pred", pred_error=0.1))):
        row = []
        for scale in (1.1, 1.2, 1.5):
            cfg = Config(**{**base, "slack": scale, **extra})
            s = Simulator(sessions, cfg, cheap).run()
            row.append(100 * sum(not t.missed for t in s.turns) / len(s.turns))
        print("   " + "%-36s" % name + "".join("%8.1f%%" % x for x in row))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(Path(p))
