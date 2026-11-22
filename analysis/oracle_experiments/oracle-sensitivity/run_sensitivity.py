"""Bounded validation replay using the unmodified archived simulator."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

from analysis.replay.run_gate import BUDGET_QUANTILES, PROFILES, TRAIN_DAYS, find_compress
from analysis.replay.sim import Cheap, Config, Simulator, build_sessions, summarize

DAYS = ("2026-08-30", "2026-08-31", "2026-09-01")
POLICIES = ("edf", "llf_cheap", "llf_oracle", "srpt_cheap", "srpt_oracle")
PROFILE = PROFILES["l40s"]
KV = 1_000_000


def load(days, end):
    cutoff = datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp() * 1000
    result = {}
    for day in days:
        raw = json.loads((ROOT / "compact" / f"{day}.json").read_text())["sessions"]
        valid = [s for s in raw if all(r["end"] < cutoff and r["start"] < cutoff for r in s["requests"])]
        result[day] = build_sessions(valid, PROFILE)
        print(day, "sessions", len(valid), "cross-cutoff sessions removed", len(raw) - len(valid), flush=True)
    return result


class ObservedSimulator(Simulator):
    """Record dispatch observations while delegating all policy decisions."""
    def __init__(self, *args):
        super().__init__(*args)
        self.waits = []
        self.turn_ids = []
        self.blocked_intervals = []
        self.blocked_since = None

    def _pop(self, now):
        item = super()._pop(now)
        if item is not None:
            self.popped.append(item)
        return item

    def _dispatch(self, now):
        if self.blocked_since is not None:
            self.blocked_intervals.append((self.blocked_since, now))
        self.popped = []
        old_count = len(self.busy)
        super()._dispatch(now)
        started = len(self.busy) - old_count
        # A selected request that did not start took the memory-blocked branch.
        assert len(self.popped) in (started, started + 1)
        self.waits.extend((item[3], now - item[3]) for item in self.popped[:started])
        self.blocked_since = now if len(self.popped) > started else None

    def _on_finish(self, st, now):
        if st.s.reqs[st.i].last_in_turn:
            self.turn_ids.append((st.idx, st.s.reqs[st.i].turn))
        super()._on_finish(st, now)


def quantile(values, q):
    return sorted(values)[int(q * len(values))]


def main():
    validation = load(DAYS, "2026-09-02")
    preflight = {}
    for day, sessions in validation.items():
        eligible = [s for s in sessions if all(r.prompt + r.completion <= KV for r in s.reqs)]
        validation[day] = eligible
        footprints = sorted(r.prompt + r.completion for s in eligible for r in s.reqs)
        preflight[day] = dict(sessions=len(eligible), excluded_oversize_sessions=len(sessions) - len(eligible),
                              requests=len(footprints),
                              p95_footprint=quantile(footprints, .95),
                              max_footprint=max(footprints), largest_eight=sum(footprints[-8:]),
                              ideal_gap_share=(sum(sum(s.z) for s in eligible) - sum(
                                  r.ideal for s in eligible for r in s.reqs
                              )) / sum(sum(s.z) for s in eligible))
        assert max(footprints) <= KV, "Single-request reservation would be clamped"
    (ROOT / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps(preflight, indent=2), flush=True)
    if "--preflight" in sys.argv:
        return

    train = load(TRAIN_DAYS, "2026-08-30")
    train_sessions = [s for sessions in train.values() for s in sessions]
    cheap = Cheap(train_sessions, PROFILE)
    durations = [z for s in train_sessions for z in s.z]
    budgets = tuple(quantile(durations, q) for q in BUDGET_QUANTILES)
    print("Class budgets (seconds):", budgets, flush=True)

    with (ROOT / "runs.jsonl").open("w") as out:
        for day, sessions in validation.items():
            for load_target in (.7, 1.1):
                compress, horizon = find_compress(sessions, 8, load_target)
                lo, hi = .1 * horizon, .9 * horizon
                for kv in (0, KV):
                    pairs = [(p, "lru") for p in POLICIES]
                    if kv:
                        pairs += [("edf", "cheap"), ("edf", "oracle")]
                    rows = []
                    for policy, eviction in pairs:
                        cfg = Config(8, kv or math.inf, PROFILE, policy=policy, eviction=eviction,
                                     deadline_mode="class", budgets=budgets, compress=compress, seed=0)
                        sim = ObservedSimulator(sessions, cfg, cheap).run()
                        assert len(sim.waits) == len(sim.busy)
                        assert len(sim.turn_ids) == len(sim.turns)
                        assert sim.blocked_since is None
                        turns = {key: t for key, t in zip(sim.turn_ids, sim.turns) if lo <= t.arrival < hi}
                        waits = [wait for ready, wait in sim.waits if lo <= ready < hi]
                        clipped_busy = sum(max(0, min(hi, t+d) - max(lo, t)) for t, d in sim.busy)
                        blocked = sum(max(0, min(hi, b) - max(lo, a)) for a, b in sim.blocked_intervals)
                        row = dict(day=day, load=load_target, kv=kv, policy=policy, eviction=eviction,
                                   compress=compress, horizon=horizon, seed=0, budgets=budgets,
                                   **summarize(sim, horizon),
                                   clipped_utilization=clipped_busy / (8 * (hi-lo)),
                                   mean_queue_wait=mean(waits), p95_queue_wait=quantile(waits, .95),
                                   queued_share=sum(w > 1e-9 for w in waits) / len(waits),
                                   memory_blocked_seconds=blocked, memory_blocked_share=blocked/(hi-lo))
                        rows.append((row, turns))
                    common = set.intersection(*(set(turns) for _, turns in rows))
                    for row, turns in rows:
                        same = [turns[key] for key in sorted(common)]
                        feasible = [t for t in same if t.feasible]
                        row.update(common_turns=len(same), common_feasible=len(feasible),
                                   common_attainment=mean(not t.missed for t in feasible),
                                   common_attainment_all=mean(not t.missed for t in same),
                                   common_mean_latency=mean(t.latency for t in same),
                                   common_p95_latency=quantile([t.latency for t in same], .95))
                        out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(day, load_target, kv, len(common), "common turns", flush=True)


if __name__ == "__main__":
    main()
