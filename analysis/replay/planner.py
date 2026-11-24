"""Rollout planning over predicted futures: the scheduler that a DAG predictor would drive.

A predictor gives, for one session, K sampled futures of the rest of that session (Forecast). Each
sample is its remaining user turns, each a list of LLM calls with the GPU time of the call and the
gap after it, so every sample carries per-turn times and a total time. The real predictor will come
from the DAG work; until then NoisyOracle stands in for it: the true future with controlled error,
refreshed every few user turns. Between refreshes a forecast is aged by what has happened since:
samples that end before the session has got to are dropped.

The planner (RolloutPlanner, passed to sim.Simulator as `planner`) acts whenever a GPU slot is free
and more than one request is ready. It takes the `candidates` best by the simulator's own policy
(EDF for a planner run), and for each, under each of the same `samples` futures, simulates every
active session forward for `horizon` seconds: the candidate starts now, later choices are EDF. It
runs the candidate whose rollouts get the most sessions to their deadline. The rollouts model GPU
slots, queueing and deadlines; they do not model KV memory and do not see sessions that have not
arrived yet.

Plugging in the real predictor: implement `forecast(session, i, now) -> Forecast` (sessions are
sim.Session, i the index of the session's next request) and pass it as RolloutPlanner's predictor.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from analysis.replay.sim import Session


@dataclass(frozen=True)
class FutureSample:
    """One possible rest of a session: turns[t] is a list of (LLM GPU seconds, seconds after it
    before the next call is ready). The last call's gap is 0."""
    turns: tuple[tuple[tuple[float, float], ...], ...]

    @property
    def calls(self) -> list[tuple[float, float]]:
        return [c for t in self.turns for c in t]

    @property
    def turn_times(self) -> list[float]:
        """Each remaining turn's time alone: its calls and the gaps inside it."""
        return [sum(d for d, _ in t) + sum(g for _, g in t[:-1]) for t in self.turns]

    @property
    def total(self) -> float:
        """The rest of the session alone, user pauses between turns included."""
        calls = self.calls
        return sum(d for d, _ in calls) + sum(g for _, g in calls[:-1])

    @staticmethod
    def from_turn_totals(llm_s: list[float], wall_s: list[float], calls: list[int],
                         reply_s: float = 5.0) -> "FutureSample":
        """For a predictor that gives only each turn's GPU time, wall time and call count: the
        calls split the GPU time evenly and the remaining wall time evenly as gaps between them."""
        turns = []
        for gpu, wall, n in zip(llm_s, wall_s, calls):
            n = max(1, n)
            d = gpu / n
            gap = max(0.0, wall - gpu) / (n - 1) if n > 1 else 0.0
            turns.append(tuple((d, gap) for _ in range(n - 1)) + ((d, reply_s),))
        if turns:
            last = turns[-1]
            turns[-1] = last[:-1] + ((last[-1][0], 0.0),)
        return FutureSample(tuple(turns))


@dataclass
class Forecast:
    made_at: float
    first: int                       # index of the session's request the samples start at
    samples: list[FutureSample]


class Predictor(Protocol):
    def forecast(self, session: Session, i: int, now: float) -> Forecast: ...


def true_future(session: Session, i: int) -> FutureSample:
    """What actually happens from request i on, as the simulator replays it (unlimited KV)."""
    turns, cur = [], []
    for r in session.reqs[i:]:
        cur.append((r.ideal, r.gap or 0.0))
        if r.cat == "__none__" or r.gap is None:          # the model answered: the user turn ends
            turns.append(tuple(cur))
            cur = []
    if cur:
        turns.append(tuple(cur))
    return FutureSample(tuple(turns))


@dataclass
class NoisyOracle:
    """Stand-in for the DAG predictor: the true future with lognormal error. `error` is the sigma
    of log(predicted / true) for the number of calls left, each call's GPU time and each gap;
    0 gives the truth in every sample. A longer predicted future repeats the true calls in order."""
    error: float = 0.3
    samples: int = 8
    seed: int = 0

    def forecast(self, session: Session, i: int, now: float) -> Forecast:
        truth = true_future(session, i)
        if self.error == 0:
            return Forecast(now, i, [truth] * self.samples)
        rng = random.Random(f"{self.seed}:{session.sid}:{i}")
        calls = truth.calls
        bounds = []                              # cumulative call counts at each turn's end
        n = 0
        for t in truth.turns:
            n += len(t)
            bounds.append(n)
        out = []
        for _ in range(self.samples):
            m = max(1, round(len(calls) * math.exp(rng.gauss(0, self.error))))
            noisy = [(calls[k % len(calls)][0] * math.exp(rng.gauss(0, self.error)),
                      calls[k % len(calls)][1] * math.exp(rng.gauss(0, self.error))) for k in range(m)]
            noisy[-1] = (noisy[-1][0], 0.0)
            ends = [b for b in bounds if b < m] + [m]
            turns, start = [], 0
            for e in ends:
                turns.append(tuple(noisy[start:e]))
                start = e
            out.append(FutureSample(tuple(turns)))
        return Forecast(now, i, out)


def aged(fc: Forecast, i: int) -> list[list[tuple[float, float]]]:
    """Each sample's calls from request i on, dropping samples the session has already outrun.
    If none is left, the session is assumed to make one more call as long as its samples' mean."""
    done = i - fc.first
    out = [s.calls[done:] for s in fc.samples if len(s.calls) > done]
    if not out:
        calls = [c for s in fc.samples for c in s.calls]
        mean = sum(d for d, _ in calls) / len(calls) if calls else 1.0
        out = [[(mean, 0.0)]]
    return out


@dataclass
class RolloutPlanner:
    predictor: Predictor
    refresh_turns: int = 1           # a new forecast every this many user turns of a session
    candidates: int = 3
    horizon: float = 120.0
    samples: int = 8
    forecasts: dict[int, Forecast] = field(default_factory=dict)
    turns_seen: dict[int, int] = field(default_factory=dict)
    decisions: int = 0
    changed: int = 0                 # decisions where the planner did not take the policy's first choice
    confidence: float = 0.0          # override the policy only for a candidate that beats its choice in at
                                     # least this share of the samples (0: best mean wins, ties to the policy)
    trace: list | None = None        # a list to append each decision's details to, for inspection

    def on_ready(self, sim, st, now: float) -> None:
        """A request became ready: at the start of every `refresh_turns`-th user turn, re-forecast."""
        reqs = st.s.reqs
        if not (st.i == 0 or reqs[st.i - 1].cat == "__none__"):
            return
        t = self.turns_seen.get(st.idx, 0)
        self.turns_seen[st.idx] = t + 1
        if t % self.refresh_turns == 0 or st.idx not in self.forecasts:
            self.forecasts[st.idx] = self.predictor.forecast(st.s, st.i, now)

    def choose(self, sim, now: float, cands: list) -> object:
        """The candidate with the best mean score over the samples. With `confidence`, a candidate
        other than the policy's first choice must also beat that choice, sample by sample, in at
        least that share of the samples, or the policy's choice stands."""
        self.decisions += 1
        snapshot = self._snapshot(sim, now, cands)
        scores = {c[2]: [self._rollout(sim, snapshot, j, c[2], now) for j in range(self.samples)] for c in cands}
        base = scores[cands[0][2]]
        best, best_mean = cands[0], sum(base) / len(base)
        for c in cands[1:]:
            s = scores[c[2]]
            wins = sum(a > b for a, b in zip(s, base)) / len(s)
            if sum(s) / len(s) > best_mean + 1e-9 and wins >= self.confidence:
                best, best_mean = c, sum(s) / len(s)
        self.changed += best is not cands[0]
        if self.trace is not None:
            self.trace.append({"now": now, "choice": best[2], "candidates": [
                {"session": c[2], "sid": sim.states[c[2]].s.sid, "deadline": sim.states[c[2]].deadline,
                 "request": sim.states[c[2]].i, "scores": scores[c[2]]} for c in cands],
                "active": len(snapshot)})
        return best

    def _snapshot(self, sim, now: float, cands: list) -> dict:
        """Every active session: what it is doing now and its forecast futures from its next request.
        The candidates have been taken out of the queue to be chosen from, so they are added back."""
        ready_at = {c[2]: c[3] for c in cands}
        for item in sim.ready:
            ready_at[item[2]] = item[3]
        for entry in sim.demoted:
            ready_at[entry[2][2]] = entry[2][3]
        snap = {}
        for st in sim.states:
            if st.i >= len(st.s.reqs) or st.idx not in self.forecasts:
                continue
            futures = aged(self.forecasts[st.idx], st.i)
            if st.idx in sim.active:
                a = sim.active[st.idx]
                snap[st.idx] = ("running", now - a["start"], futures, st.deadline)
            elif st.idx in ready_at:
                snap[st.idx] = ("ready", ready_at[st.idx], futures, st.deadline)
            elif st.next_use < math.inf and st.next_use > now:
                gap_left = st.next_use - now          # how long this gap still has, by the forecast
                prev = st.i - 1 - self.forecasts[st.idx].first
                fc = self.forecasts[st.idx]
                predicted = [s.calls[prev][1] for s in fc.samples if 0 <= prev < len(s.calls)]
                if predicted:
                    gap_left = max(0.0, sum(predicted) / len(predicted) - (now - st.idle_since))
                snap[st.idx] = ("gap", gap_left, futures, st.deadline)
        return snap

    def _rollout(self, sim, snap: dict, j: int, first: int, now: float) -> float:
        """Sessions on time under sample j when session `first` starts its next call now and every
        later choice is EDF. A session still going at the horizon counts as on time if what is left
        of it, run alone from there, still makes its deadline: the rest of a call or a wait it is in
        the middle of, then its remaining calls and gaps."""
        end = now + self.horizon
        slots = sim.cfg.slots
        events, ready, left, pos = [], [], {}, {}
        pending: dict[int, tuple[float, int]] = {}      # each session's next event: (time, 0 ready / 1 call done)
        busy = 0

        def push(t, kind, idx):
            pending[idx] = (t, kind)
            heapq.heappush(events, (t, kind, idx))

        for idx, (state, x, futures, deadline) in snap.items():
            calls = futures[j % len(futures)]
            left[idx], pos[idx] = calls, 0
            if state == "running":
                busy += 1
                push(now + max(1e-3, calls[0][0] - x), 1, idx)
            elif state == "ready":
                if idx != first:
                    ready.append((deadline, x, idx))
            else:
                push(now + x, 0, idx)
        heapq.heapify(ready)
        finished = {}

        def start(t, idx):
            push(t + left[idx][pos[idx]][0], 1, idx)

        # the decision being scored: `first` takes the free slot now
        start(now, first)
        busy += 1
        while events:
            t, kind, idx = heapq.heappop(events)
            if t > end:
                break
            del pending[idx]
            if kind == 1:
                busy -= 1
                gap = left[idx][pos[idx]][1]
                pos[idx] += 1
                if pos[idx] >= len(left[idx]):
                    finished[idx] = t
                else:
                    push(t + gap, 0, idx)
            else:
                heapq.heappush(ready, (snap[idx][3], t, idx))
            while busy < slots and ready:
                _, _, nxt = heapq.heappop(ready)
                start(t, nxt)
                busy += 1

        def alone(calls):
            return sum(d for d, _ in calls) + sum(g for _, g in calls[:-1])

        score = 0
        for idx, (_, _, _, deadline) in snap.items():
            if idx in finished:
                score += finished[idx] <= deadline
                continue
            calls, k = left[idx], pos[idx]
            if idx in pending and pending[idx][1] == 1:      # mid-call at the horizon
                rest = pending[idx][0] - end + (calls[k][1] + alone(calls[k + 1:]) if k + 1 < len(calls) else 0.0)
            elif idx in pending:                              # mid-gap at the horizon
                rest = pending[idx][0] - end + alone(calls[k:])
            else:                                             # waiting in the queue
                rest = alone(calls[k:])
            score += end + rest <= deadline
        return score
