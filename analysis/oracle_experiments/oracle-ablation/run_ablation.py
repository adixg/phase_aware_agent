"""LLF/SRPT 2x2 factorial ablation on the previous sensitivity cohort."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from sensitivity_helpers import DAYS, PROFILE, ROOT, KV, ObservedSimulator, load, quantile
from analysis.replay.run_gate import BUDGET_QUANTILES, TRAIN_DAYS, find_compress
from analysis.replay.sim import Cheap, Config, POLICIES, estimate_remaining, summarize

VARIANTS = [("edf", "none", "none")]
VARIANTS += [(family, ordering, demotion) for family in ("llf", "srpt")
             for ordering in ("cheap", "oracle") for demotion in ("cheap", "oracle")]


class AblationSimulator(ObservedSimulator):
    def __init__(self, sessions, cfg, cheap, demotion_policy):
        super().__init__(sessions, cfg, cheap)
        self.demotion_policy = demotion_policy
        self.first_demotions = {}
        self.demotion_events = []

    def _demotion_estimate(self, req, ordering_estimate):
        if self.demotion_policy == self.policy:
            return ordering_estimate
        return estimate_remaining(req, self.demotion_policy, self.cheap, self.cfg.profile)

    def _record_demotion(self, st, item, now):
        req = st.s.reqs[st.i]
        # This is an optimistic model lower bound, excluding future waits and cache loss.
        ideal_fits = now + estimate_remaining(req, POLICIES["llf_oracle"],
                                              self.cheap, self.cfg.profile) <= st.deadline
        self.demotion_events.append(item[3])
        self.first_demotions.setdefault(
            item[1], (item[3], (st.idx, req.turn), ideal_fits, now > st.deadline)
        )


def demotions(sim, turns, lo, hi):
    counts = Counter(events=sum(lo <= ready < hi for ready in sim.demotion_events))
    for ready, key, ideal_fits, overdue in sim.first_demotions.values():
        if not lo <= ready < hi:
            continue
        counts["requests"] += 1
        if ideal_fits:
            counts["optimistic_fit"] += 1
            counts["optimistic_fit_missed" if turns[key].missed else "optimistic_fit_met"] += 1
        else:
            counts["already_overdue" if overdue else "ideal_cannot_fit_before_deadline"] += 1
    assert counts["requests"] == sum(counts[k] for k in (
        "optimistic_fit", "already_overdue", "ideal_cannot_fit_before_deadline"))
    assert counts["optimistic_fit"] == counts["optimistic_fit_missed"] + counts["optimistic_fit_met"]
    return dict(counts)


def turn_metrics(turns):
    feasible = [turn for turn in turns if turn.feasible]
    latency = [turn.latency for turn in turns]
    return dict(turns=len(turns), feasible=len(feasible),
                attainment=mean(not turn.missed for turn in feasible),
                attainment_all=mean(not turn.missed for turn in turns),
                mean_latency=mean(latency), p95_latency=quantile(latency, .95))


def main():
    validation = load(DAYS, "2026-09-02")
    train = load(TRAIN_DAYS, "2026-08-30")
    train_sessions = [s for sessions in train.values() for s in sessions]
    cheap = Cheap(train_sessions, PROFILE)
    durations = [z for s in train_sessions for z in s.z]
    budgets = tuple(quantile(durations, q) for q in BUDGET_QUANTILES)
    print("Class budgets (seconds):", budgets, flush=True)
    prior_path = ROOT.parent / "oracle-sensitivity" / "runs.jsonl"
    prior = {(r["day"], r["load"], r["kv"], r["policy"], r["eviction"]): r
             for r in map(json.loads, prior_path.read_text().splitlines())}
    metadata = dict(replay_ref="a3dbeae41b7dac6c06f3fe1863ad1300e348f405",
                    prior_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                    days=DAYS, train_days=TRAIN_DAYS, profile="l40s", slots=8,
                    load=1.1, kv_budgets=[0, KV], deadline_mode="class",
                    deadline_multiplier=1, seed=0, budgets=budgets)
    (ROOT / "setup.json").write_text(json.dumps(metadata, indent=2) + "\n")

    with (ROOT / "runs.jsonl").open("w") as out:
        for day, candidates in validation.items():
            sessions = [s for s in candidates if all(r.prompt + r.completion <= KV for r in s.reqs)]
            compress, horizon = find_compress(sessions, 8, 1.1)
            lo, hi = .1 * horizon, .9 * horizon
            for kv in (0, KV):
                rows = []
                for family, ordering, demotion in VARIANTS:
                    policy = family if family == "edf" else f"{family}_{ordering}"
                    demotion_key = family if family == "edf" else f"{family}_{demotion}"
                    cfg = Config(8, kv or math.inf, PROFILE, policy=policy, eviction="lru",
                                 deadline_mode="class", budgets=budgets, compress=compress, seed=0)
                    sim = AblationSimulator(sessions, cfg, cheap, POLICIES[demotion_key]).run()
                    assert len(sim.waits) == len(sim.busy)
                    assert len(sim.turn_ids) == len(sim.turns)
                    assert sim.blocked_since is None
                    assert all(not t.missed or t.latency > t.budget for t in sim.turns)
                    assert all(t.feasible or t.missed for t in sim.turns)
                    turns = dict(zip(sim.turn_ids, sim.turns))
                    window = {key: t for key, t in turns.items() if lo <= t.arrival < hi}
                    waits = [wait for ready, wait in sim.waits if lo <= ready < hi]
                    blocked = sum(max(0, min(hi, b) - max(lo, a)) for a, b in sim.blocked_intervals)
                    native = summarize(sim, horizon)
                    if ordering == demotion:
                        old = prior[day, 1.1, kv, policy, "lru"]
                        assert all(old[k] == v for k, v in native.items()), (day, kv, policy)
                        assert old["compress"] == compress and old["budgets"] == list(budgets)
                        assert old["mean_queue_wait"] == mean(waits)
                        assert old["memory_blocked_seconds"] == blocked
                    row = dict(day=day, kv=kv, sessions=len(sessions), family=family, ordering=ordering,
                               demotion=demotion, compress=compress, horizon=horizon, native=native,
                               mean_queue_wait=mean(waits), p95_queue_wait=quantile(waits, .95),
                               memory_blocked_seconds=blocked, memory_blocked_share=blocked / (hi-lo),
                               demotions=demotions(sim, turns, lo, hi))
                    rows.append((row, window))
                common = sorted(set.intersection(*(set(window) for _, window in rows)))
                for row, window in rows:
                    row["paired"] = turn_metrics([window[key] for key in common])
                    out.write(json.dumps(row) + "\n")
                out.flush()
                print(day, kv, len(common), "common turns, all prior policy metrics matched", flush=True)


if __name__ == "__main__":
    main()
