"""Closed-loop discrete-event replay of agent sessions on one GPU server.

Each request becomes ready when the session's previous request finishes plus the recorded
tool or human gap, so scheduling delay feeds back into the session. The server has `slots`
concurrent requests and a KV pool of `kv_tokens`. A session's prefix stays in the pool between
requests (evictable, tail first) and later requests pay prefill only for what is missing.

Simplifications: a request reserves prompt + completion at start and is never preempted
mid-generation; decode speed does not depend on batch size; prefixes are per-session (no
sharing across sessions); subagent requests are not replayed (extract.py drops them, their
time stays in the parent's tool gap).
"""

from __future__ import annotations

import bisect
import heapq
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    decode_ms: float      # per output token, per request
    prefill_ms: float     # per uncached prompt token
    reload_ms: float | None = None  # per token reloaded from a CPU tier; None = no tier, recompute
    overhead_ms: float = 0.0        # per request, on top of prefill and decode


@dataclass(frozen=True)
class Policy:
    kind: str                 # fcfs | returning | las | edf | llf | srpt
    calls: bool = False       # true remaining-call count (else the Cheap table)
    gaps: bool = False        # true future tool gaps
    tokens: bool = False      # true future service, including this call's completion length


POLICIES = {
    "fcfs": Policy("fcfs"),
    "returning": Policy("returning"),
    "las": Policy("las"),
    "edf": Policy("edf"),
    "llf_cheap": Policy("llf"),
    "srpt_cheap": Policy("srpt"),
    "llf_calls": Policy("llf", calls=True),
    "llf_gaps": Policy("llf", gaps=True),
    "llf_tokens": Policy("llf", tokens=True),
    "llf_oracle": Policy("llf", True, True, True),
    "srpt_calls": Policy("srpt", calls=True),
    "srpt_gaps": Policy("srpt", gaps=True),
    "srpt_tokens": Policy("srpt", tokens=True),
    "srpt_oracle": Policy("srpt", True, True, True),
}
EVICTIONS = ("lru", "cheap", "oracle")
STEP_CAP = 12
NEVER_S = 86400.0     # next use of a session that never returns, in the Cheap gap tables
MIN_BUCKET = 20


@dataclass(slots=True)
class Req:
    prompt: int
    base: int             # tokens of this prompt a resident prefix could cover
    completion: int
    gap: float | None     # seconds after this request until the next; None = session ends
    cat: str              # tool category that set the gap, "__none__" if the model stopped
    prev_cat: str
    step: int             # position in its turn
    turn: int
    last_in_turn: bool
    ideal: float = 0.0    # service time with the prefix fully resident
    rem_calls: int = 0    # from this request to the end of the turn, inclusive
    rem_service: float = 0.0
    rem_gaps: float = 0.0


@dataclass
class Session:
    sid: str
    arrival: float
    reqs: list[Req]
    z: list[float]        # per turn: duration with no contention


def build_sessions(raw: list[dict], profile: Profile, max_think_s: float = 3600.0) -> list[Session]:
    """Sessions from extract.py's records; arrivals are seconds after the first session starts."""
    t0 = min(s["start"] for s in raw)
    out = []
    for s in sorted(raw, key=lambda s: (s["start"], s["sid"])):
        reqs, rs, turn, step = [], s["requests"], -1, 0
        for i, r in enumerate(rs):
            if i == 0 or r["user"]:
                turn, step = turn + 1, 0
            base = 0
            if i:
                base = r["lcp"] + (rs[i - 1]["completion"] if r["append"] else 0)
            nxt = rs[i + 1] if i + 1 < len(rs) else None
            gap = None if nxt is None else r["gap"] / 1000
            ends_turn = nxt is None or nxt["user"]
            if gap is not None and ends_turn:
                gap = min(gap, max_think_s)   # a long-idle session is effectively a new one
            reqs.append(Req(r["prompt"], min(base, r["prompt"]), r["completion"], gap, r["cat"],
                            rs[i - 1]["cat"] if i else "__start__", step, turn, ends_turn))
            step += 1
        out.append(Session(s["sid"], (s["start"] - t0) / 1000, reqs, _annotate(reqs, profile)))
    return out


def _annotate(reqs: list[Req], profile: Profile) -> list[float]:
    """Fill each request's remaining-work fields, and return each turn's no-contention duration."""
    z: dict[int, float] = {}
    calls, service, gaps = 0, 0.0, 0.0
    for r in reversed(reqs):
        if r.last_in_turn:
            calls, service, gaps = 0, 0.0, 0.0
        else:
            gaps += r.gap
        r.ideal = (profile.overhead_ms + profile.prefill_ms * (r.prompt - r.base)
                   + profile.decode_ms * r.completion) / 1000
        calls += 1
        service += r.ideal
        r.rem_calls, r.rem_service, r.rem_gaps = calls, service, gaps
        z[r.turn] = service + gaps
    return [z[t] for t in sorted(z)]


class Cheap:
    """Historical averages from training sessions: what a system knows without a predictor."""

    def __init__(self, sessions: list[Session], profile: Profile):
        self.profile = profile
        by_key, by_step, gaps = defaultdict(list), defaultdict(list), defaultdict(list)
        completions, news, tool_gaps = [], [], []
        for s in sessions:
            for r in s.reqs:
                k = min(r.step, STEP_CAP)
                by_key[(k, r.prev_cat)].append(r.rem_calls)
                by_step[k].append(r.rem_calls)
                completions.append(r.completion)
                news.append(r.prompt - r.base)
                if r.gap is not None and not r.last_in_turn:
                    tool_gaps.append(r.gap)
                gaps[r.cat if r.gap is None or not r.last_in_turn else "__none__"].append(
                    NEVER_S if r.gap is None else r.gap)
        self.calls = {k: sum(v) / len(v) for k, v in by_key.items() if len(v) >= MIN_BUCKET}
        self.calls_step = {k: sum(v) / len(v) for k, v in by_step.items()}
        self.completion = sum(completions) / len(completions)
        self.gap = sum(tool_gaps) / len(tool_gaps)
        self.service = (profile.overhead_ms + profile.prefill_ms * sum(news) / len(news)
                        + profile.decode_ms * self.completion) / 1000
        self.gaps, self.tail_sums = {}, {}
        for cat, v in gaps.items():
            v.sort()
            tail = [0.0] * (len(v) + 1)
            for i in range(len(v) - 1, -1, -1):
                tail[i] = tail[i + 1] + v[i]
            self.gaps[cat], self.tail_sums[cat] = v, tail

    def remaining_calls(self, r: Req) -> float:
        k = min(r.step, STEP_CAP)
        return max(1.0, self.calls.get((k, r.prev_cat), self.calls_step.get(k, 1.0)))

    def time_to_next_use(self, cat: str, elapsed: float) -> float:
        """Mean residual gap: E[G - t | G > t] for this tool category, given t already elapsed."""
        v = self.gaps.get(cat) or self.gaps.get("Other")
        if v is None:
            return NEVER_S
        tail = self.tail_sums[cat if cat in self.gaps else "Other"]
        i = bisect.bisect_right(v, elapsed)
        return NEVER_S if i == len(v) else tail[i] / (len(v) - i) - elapsed


def estimate_remaining(r: Req, p: Policy, cheap: Cheap, profile: Profile) -> float:
    """Remaining time in this turn, each ingredient either true (the policy's flags) or historical."""
    n = float(r.rem_calls) if p.calls else cheap.remaining_calls(r)
    prefill = (profile.overhead_ms + profile.prefill_ms * (r.prompt - r.base)) / 1000
    current = r.ideal if p.tokens else prefill + profile.decode_ms * cheap.completion / 1000
    later = r.rem_calls - 1
    service = (r.rem_service - r.ideal) / later if p.tokens and later else cheap.service
    gap = r.rem_gaps / later if p.gaps and later else cheap.gap
    return current + (n - 1) * (service + gap)


@dataclass
class Config:
    slots: int
    kv_tokens: float
    profile: Profile
    policy: str = "fcfs"
    eviction: str = "lru"
    deadline_mode: str = "class"     # class: a random class's budget; proportional: slack * z
    budgets: tuple[float, ...] = (math.inf,)
    slack: float = 2.0
    seed: int = 0
    compress: float = 1.0            # session start offsets are divided by this
    closed: int | None = None        # at most this many sessions alive; arrivals are ignored


@dataclass
class TurnResult:
    arrival: float
    latency: float
    budget: float
    z: float
    missed: bool
    feasible: bool


class _State:
    __slots__ = ("s", "idx", "i", "resident", "idle_since", "next_use", "cat", "attained", "turn_arrival", "deadline")

    def __init__(self, s: Session, idx: int):
        self.s, self.idx, self.i, self.resident = s, idx, 0, 0
        self.idle_since = self.next_use = 0.0
        self.cat = "__none__"
        self.attained = self.turn_arrival = self.deadline = 0.0


class Simulator:
    def __init__(self, sessions: list[Session], cfg: Config, cheap: Cheap | None = None):
        self.cfg, self.cheap = cfg, cheap
        self.policy = POLICIES[cfg.policy]
        self.sessions = sessions
        self.ready: list = []
        self.demoted: list = []
        self.idle: dict[int, _State] = {}
        self.events: list = []
        self.seq = 0
        self.used = 0.0
        self.running = 0
        self.turns: list[TurnResult] = []
        self.busy: list[tuple[float, float]] = []
        self.tokens_prompt = self.tokens_hit = self.tokens_evicted = 0.0
        self.alive = 0
        self.pending: list[int] = []
        self.states = [_State(s, i) for i, s in enumerate(sessions)]

    def _push(self, t: float, kind: str, st: _State) -> None:
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, st.idx))

    def run(self) -> "Simulator":
        cfg = self.cfg
        if cfg.closed is None:
            for st in self.states:
                self._push(st.s.arrival / cfg.compress, "ready", st)
        else:
            self.pending = list(range(len(self.states) - 1, -1, -1))
            for _ in range(min(cfg.closed, len(self.states))):
                self._push(0.0, "ready", self.states[self.pending.pop()])
        while self.events:
            now, _, kind, idx = heapq.heappop(self.events)
            st = self.states[idx]
            if kind == "ready":
                self._on_ready(st, now)
            else:
                self._on_finish(st, now)
            self._dispatch(now)
        return self

    def _budget(self, st: _State, turn: int) -> float:
        cfg = self.cfg
        if cfg.deadline_mode == "proportional":
            return cfg.slack * st.s.z[turn]
        h = zlib.crc32(f"{cfg.seed}:{st.s.sid}:{turn}".encode())
        return cfg.budgets[h % len(cfg.budgets)]

    def _on_ready(self, st: _State, now: float) -> None:
        r = st.s.reqs[st.i]
        if r.step == 0:
            st.turn_arrival, st.attained = now, 0.0
            st.deadline = now + self._budget(st, r.turn)
        p = self.policy
        est = 0.0
        if p.kind in ("llf", "srpt"):
            est = estimate_remaining(r, p, self.cheap, self.cfg.profile)
        key = {"fcfs": lambda: (now,),
               "returning": lambda: (r.step == 0, now),
               "las": lambda: (st.attained, now),
               "edf": lambda: (st.deadline, now),
               "llf": lambda: (st.deadline - est, now),
               "srpt": lambda: (est, now)}[p.kind]()
        self.seq += 1
        heapq.heappush(self.ready, (key, self.seq, st.idx, now, est))

    def _pop(self, now: float):
        """Best ready request. Turns the policy judges already lost wait behind all the others."""
        p = self.policy
        while self.ready:
            item = heapq.heappop(self.ready)
            st = self.states[item[2]]
            lost = (p.kind == "edf" and now > st.deadline) or \
                (p.kind in ("llf", "srpt") and now + item[4] > st.deadline)
            if lost:
                heapq.heappush(self.demoted, (item[3], item[1], item))
                continue
            return item
        return heapq.heappop(self.demoted)[2] if self.demoted else None

    def _queue_ranks(self) -> dict[int, int]:
        """Position of each queued session in dispatch order: known to any real scheduler."""
        ranks = {item[2]: i for i, item in enumerate(sorted(self.ready))}
        n = len(ranks)
        for i, entry in enumerate(sorted(self.demoted)):
            ranks[entry[2][2]] = n + i
        return ranks

    def _score(self, v: _State, now: float, ranks: dict[int, int] | None) -> float:
        """Higher = evict first: for lru time idle, otherwise time until the prefix is next
        needed. A queued session's time comes from its place in line; one in a gap uses the
        true next use (oracle) or the historical mean residual gap (cheap)."""
        e = self.cfg.eviction
        if e == "lru":
            return now - v.idle_since
        if v.idx in ranks:
            return (ranks[v.idx] + 1) * self.cheap.service / self.cfg.slots
        if e == "oracle":
            return v.next_use - now
        return self.cheap.time_to_next_use(v.cat, now - v.idle_since)

    def _dispatch(self, now: float) -> None:
        cfg = self.cfg
        while self.running < cfg.slots:
            item = self._pop(now)
            if item is None:
                return
            st = self.states[item[2]]
            r = st.s.reqs[st.i]
            usable = min(st.resident, r.base)
            self.used -= st.resident - usable    # a prefix past the shared part is stale
            st.resident = usable
            need = min(r.prompt + r.completion, cfg.kv_tokens) - usable
            was_idle = self.idle.pop(st.idx, None) is not None
            short = need - (cfg.kv_tokens - self.used)
            if short > 0 and sum(v.resident for v in self.idle.values()) < short:
                if was_idle:
                    self.idle[st.idx] = st
                heapq.heappush(self.ready, item)   # head of line waits for memory
                return
            ranks = self._queue_ranks() if short > 0 and cfg.eviction != "lru" else None
            while need > cfg.kv_tokens - self.used:
                v = max(self.idle.values(), key=lambda x: self._score(x, now, ranks))
                take = min(v.resident, need - (cfg.kv_tokens - self.used))
                v.resident -= take
                self.used -= take
                self.tokens_evicted += take
                if v.resident <= 0:
                    del self.idle[v.idx]
            self.used += need
            self.running += 1
            cpu_hit = 0.0
            if cfg.profile.reload_ms is not None:
                cpu_hit = r.base - usable
            uncached = r.prompt - usable - cpu_hit
            service = (cfg.profile.overhead_ms + cfg.profile.prefill_ms * uncached + (cfg.profile.reload_ms or 0.0) * cpu_hit
                       + cfg.profile.decode_ms * r.completion) / 1000
            self.tokens_prompt += r.prompt
            self.tokens_hit += usable
            st.resident = usable + need
            st.attained += service
            self.busy.append((now, service))
            self._push(now + service, "finish", st)

    def _on_finish(self, st: _State, now: float) -> None:
        self.running -= 1
        r = st.s.reqs[st.i]
        if r.last_in_turn:
            budget = st.deadline - st.turn_arrival
            latency = now - st.turn_arrival
            z = st.s.z[r.turn]
            self.turns.append(TurnResult(st.turn_arrival, latency, budget, z, latency > budget, z <= budget))
        st.cat, st.idle_since = r.cat if not r.last_in_turn else "__none__", now
        self.idle[st.idx] = st
        st.i += 1
        if r.gap is None:
            st.next_use = math.inf
            if self.cfg.closed is not None and self.pending:
                self._push(now, "ready", self.states[self.pending.pop()])
            return
        st.next_use = now + r.gap
        self._push(st.next_use, "ready", st)

    def utilization(self, a: float, b: float) -> float:
        """Busy share of the slots for requests starting in [a, b)."""
        return sum(d for t, d in self.busy if a <= t < b) / (self.cfg.slots * (b - a))


def summarize(sim: Simulator, horizon: float | None = None) -> dict:
    """Deadline attainment over turns that could meet their deadline with no contention. With a
    `horizon`, only turns arriving in its middle 80%, leaving out ramp-up and drain."""
    t = sim.turns if horizon is None else [x for x in sim.turns if 0.1 * horizon <= x.arrival < 0.9 * horizon]
    feas = [x for x in t if x.feasible]
    lat = sorted(x.latency for x in t)
    out = {
        "turns": len(t),
        "feasible_turns": len(feas),
        "attainment": sum(not x.missed for x in feas) / len(feas) if feas else None,
        "attainment_all": sum(not x.missed for x in t) / len(t) if t else None,
        "mean_latency": sum(lat) / len(lat) if lat else None,
        "p95_latency": lat[int(0.95 * len(lat))] if lat else None,
        "hit_rate": sim.tokens_hit / sim.tokens_prompt if sim.tokens_prompt else None,
        "tokens_evicted": sim.tokens_evicted,
    }
    if horizon is not None:
        out["utilization"] = sim.utilization(0.1 * horizon, 0.9 * horizon)
    return out
