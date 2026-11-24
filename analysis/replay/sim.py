"""Closed-loop discrete-event replay of agent sessions on one GPU server.

Each request becomes ready when the session's previous request finishes plus the recorded
tool or human gap, so scheduling delay feeds back into the session. The server has `slots`
concurrent requests and a KV pool of `kv_tokens`. A session's prefix stays in the pool between
requests (evictable, tail first) and later requests pay prefill only for what is missing.

Simplifications: a request reserves prompt + completion at start, and is preempted only by a
more urgent request (Config.preempt, off by default), never to make room for its own output;
decode speed does not depend on batch size; prefixes are per-session (no
sharing across sessions); subagent requests are not replayed (extract.py drops them, their
time stays in the parent's tool gap).
"""

from __future__ import annotations

import bisect
import heapq
import math
import random
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field


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
    lookahead: int | None = None  # true service and gaps for the next k requests of the turn (this one
                                  # included), historical averages beyond; overrides the three flags
    pure: bool = False        # no deadline logic: a turn judged lost is not pushed behind the others
    scope: str | None = None  # how much of a session's future is true when a deadline covers the whole session
                              # (the oracle decomposition): "cur" = the calls left in the current user turn, the
                              # later turns from the history table; "count" = the same plus the true number of
                              # later turns, each at the average size. Everything else stays historical.
    forecast: str | None = None   # remaining calls from a fitted admission forecaster instead of the Cheap
                                  # table: "first" = first user turn of a session only (what the predictor
                                  # was built for; later turns fall back to Cheap), "all" = every turn,
                                  # "online" = the online forecaster's estimate at this request's admission


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
    # shortest remaining time first for mean latency (JCT): the fair comparison with attained service
    "psrpt_cheap": Policy("srpt", pure=True),
    "psrpt_oracle": Policy("srpt", True, True, True, pure=True),
}
LOOKAHEAD_DEPTHS = (1, 2, 4, 8)
POLICIES["psrpt_calls"] = Policy("srpt", calls=True, pure=True)
for _fam, _kind, _pure in (("llf", "llf", False), ("srpt", "srpt", False), ("psrpt", "srpt", True)):
    POLICIES[f"{_fam}_predfirst"] = Policy(_kind, pure=_pure, forecast="first")
    POLICIES[f"{_fam}_predall"] = Policy(_kind, pure=_pure, forecast="all")
    POLICIES[f"{_fam}_online"] = Policy(_kind, pure=_pure, forecast="online")
    POLICIES[f"{_fam}_ucur"] = Policy(_kind, pure=_pure, scope="cur")
    POLICIES[f"{_fam}_ucount"] = Policy(_kind, pure=_pure, scope="count")
for _k in LOOKAHEAD_DEPTHS:
    POLICIES[f"llf_la{_k}"] = Policy("llf", lookahead=_k)
    POLICIES[f"srpt_la{_k}"] = Policy("srpt", lookahead=_k)
    POLICIES[f"psrpt_la{_k}"] = Policy("srpt", lookahead=_k, pure=True)
EVICTIONS = ("lru", "cheap", "oracle", "online")
STEP_CAP = 12     # steps 0..11 each get their own Cheap bucket; later ones share one per doubling
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
    nxt: "Req | None" = field(default=None, compare=False, repr=False)  # next request of the turn
    uturn: int = 0        # which user message this request answers (0 = the session's first), whatever the deadline unit
    rem_uturn: int = 0    # requests left in this user turn, this one included
    fut_turns: int = 0    # user turns still to come after this one
    fut_calls: int = 0    # requests in those later turns
    pred_first: float = 0.0   # forecast remaining calls (this one included), first user turns only; 0 = none
    pred_all: float = 0.0     # the same, from a forecaster fitted on every turn
    pred_online: float = 0.0  # online forecaster at this request's admission: remaining calls, this one included
    pred_stall: float = -1.0  # online forecaster when this request's response completes: wall time of its round (s)


@dataclass(slots=True)
class TurnMeta:
    """What is known when a turn is admitted (model, prompt) and how it turned out, for fitting a forecaster."""
    model: str | None
    prompt: int           # provider prompt tokens of the turn's first request
    calls: int            # requests in the turn
    tools: int            # tool calls it emitted
    duration_ms: int      # first request's start to last request's end


@dataclass
class Session:
    sid: str
    arrival: float
    reqs: list[Req]
    z: list[float]        # per turn: duration with no contention
    harness: str | None = None
    turns: list[TurnMeta] = field(default_factory=list)


def reply_time(spec: str) -> Callable[[str, int, float], float]:
    """How long a user takes to reply between turns: (session, turn, recorded seconds) -> seconds.
    "recorded" keeps the trace's own pause, "scale:K" multiplies it, and "fixed:S", "exp:MEAN"
    and "lognormal:MEDIAN,SIGMA" replace it. Drawn values depend only on session and turn, so
    every policy and every run sees the same pauses."""
    kind, _, arg = spec.partition(":")
    nums = [float(x) for x in arg.split(",")] if arg else []
    draw = lambda sid, turn: random.Random(zlib.crc32(f"{sid}:{turn}".encode()))
    if kind == "recorded" and not nums:
        return lambda sid, turn, rec: rec
    if kind == "scale" and len(nums) == 1:
        return lambda sid, turn, rec: nums[0] * rec
    if kind == "fixed" and len(nums) == 1:
        return lambda sid, turn, rec: nums[0]
    if kind == "exp" and len(nums) == 1:
        return lambda sid, turn, rec: draw(sid, turn).expovariate(1 / nums[0])
    if kind == "lognormal" and len(nums) == 2:
        return lambda sid, turn, rec: draw(sid, turn).lognormvariate(math.log(nums[0]), nums[1])
    raise ValueError(f"reply time {spec!r}: use recorded, scale:K, fixed:S, exp:MEAN or lognormal:MEDIAN,SIGMA")


def build_sessions(raw: list[dict], profile: Profile, max_think_s: float = 3600.0,
                   reply: Callable[[str, int, float], float] | None = None, unit: str = "turn") -> list[Session]:
    """Sessions from extract.py's records; arrivals are seconds after the first session starts.
    `reply` (see reply_time) sets the pause before each new user message, capped at `max_think_s`.
    `unit` is what a deadline covers: each "turn", or the whole "session" (one turn, with the
    user's pauses as waits inside it)."""
    if unit not in ("turn", "session"):
        raise ValueError(f"unit {unit!r}: use turn or session")
    t0 = min(s["start"] for s in raw)
    out = []
    for s in sorted(raw, key=lambda s: (s["start"], s["sid"])):
        reqs, rs, turn, step, uturns = [], s["requests"], -1, 0, []
        for i, r in enumerate(rs):
            if i == 0 or (r["user"] and unit == "turn"):
                turn, step = turn + 1, 0
            uturns.append((uturns[-1] + 1 if uturns else 0) if i and r["user"] else (uturns[-1] if uturns else 0))
            base = 0
            if i:
                base = r["lcp"] + (rs[i - 1]["completion"] if r["append"] else 0)
            nxt = rs[i + 1] if i + 1 < len(rs) else None
            gap = None if nxt is None else r["gap"] / 1000
            user_next = nxt is not None and nxt["user"]
            ends_turn = nxt is None or (user_next and unit == "turn")
            if user_next:
                if reply is not None:
                    gap = reply(s["sid"], i + 1, gap)
                gap = min(gap, max_think_s)   # a long-idle session is effectively a new one
            reqs.append(Req(r["prompt"], min(base, r["prompt"]), r["completion"], gap, r["cat"],
                            rs[i - 1]["cat"] if i else "__start__", step, turn, ends_turn))
            step += 1
        for a, b in zip(reqs, reqs[1:]):
            if not a.last_in_turn:
                a.nxt = b
        _user_turns(reqs, uturns)
        out.append(Session(s["sid"], (s["start"] - t0) / 1000, reqs, _annotate(reqs, profile),
                           s.get("harness"), _turn_meta(rs, reqs)))
    return out


def _user_turns(reqs: list[Req], uturns: list[int]) -> None:
    """Where each user message's turn starts and ends, which a session-level deadline otherwise hides:
    the requests left in this user turn, and the turns and requests still to come after it."""
    last, left, later = uturns[-1], 0, 0         # `later` counts the requests of turns after the current one
    for i in range(len(reqs) - 1, -1, -1):
        r, u = reqs[i], uturns[i]
        if i + 1 < len(reqs) and uturns[i + 1] != u:
            later += left                        # stepped back into the previous user turn: the one just left is "later"
            left = 0
        left += 1
        r.uturn, r.rem_uturn, r.fut_turns, r.fut_calls = u, left, last - u, later


def _turn_meta(raw: list[dict], reqs: list[Req]) -> list[TurnMeta]:
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, r in enumerate(reqs):
        first.setdefault(r.turn, i)
        last[r.turn] = i
    return [TurnMeta(raw[a].get("model"), raw[a].get("provider_prompt") or raw[a]["prompt"], b - a + 1,
                     sum(raw[i].get("n_tools") or 0 for i in range(a, b + 1)), raw[b].get("end", 0) - raw[a].get("start", 0))
            for t, a in sorted(first.items()) for b in (last[t],)]


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


def step_bucket(step: int) -> int:
    """12, 13, 14, ... for steps 12-23, 24-47, 48-95, ...: a session can run to hundreds of steps."""
    return step if step < STEP_CAP else STEP_CAP + int(math.log2(step / STEP_CAP))


class Cheap:
    """Historical averages from training sessions: what a system knows without a predictor."""

    def __init__(self, sessions: list[Session], profile: Profile):
        self.profile = profile
        by_key, by_step, gaps = defaultdict(list), defaultdict(list), defaultdict(list)
        completions, news, tool_gaps = [], [], []
        by_user_turn, turn_sizes = defaultdict(list), []      # per user turn, not per request
        for s in sessions:
            started = set()
            for r in s.reqs:
                if r.uturn not in started:
                    started.add(r.uturn)
                    by_user_turn[step_bucket(r.uturn)].append(r.fut_calls)
                    turn_sizes.append(r.rem_uturn)
                k = step_bucket(r.step)
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
        self.fut_calls_by_turn = {k: sum(v) / len(v) for k, v in by_user_turn.items() if len(v) >= MIN_BUCKET}
        every = [x for v in by_user_turn.values() for x in v]
        self.fut_calls_all = sum(every) / len(every) if every else 0.0
        self.calls_per_uturn = sum(turn_sizes) / len(turn_sizes) if turn_sizes else 1.0
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
        k = step_bucket(r.step)
        return max(1.0, self.calls.get((k, r.prev_cat), self.calls_step.get(k, 1.0)))

    def future_calls(self, uturn: int) -> float:
        """Requests expected in the user turns after this one, from the turn index alone: the average over
        training sessions that reached this turn (so it conditions on survival). Sparse late turns borrow
        the nearest earlier turn that has enough sessions."""
        for k in range(step_bucket(uturn), -1, -1):
            if k in self.fut_calls_by_turn:
                return self.fut_calls_by_turn[k]
        return self.fut_calls_all

    def time_to_next_use(self, cat: str, elapsed: float) -> float:
        """Mean residual gap: E[G - t | G > t] for this tool category, given t already elapsed."""
        v = self.gaps.get(cat) or self.gaps.get("Other")
        if v is None:
            return NEVER_S
        tail = self.tail_sums[cat if cat in self.gaps else "Other"]
        i = bisect.bisect_right(v, elapsed)
        return NEVER_S if i == len(v) else tail[i] / (len(v) - i) - elapsed


def estimate_windowed(r: Req, k: int, cheap: Cheap) -> float:
    """Remaining time in this turn when the next `k` requests are known exactly (their service and
    the gap after each) and the rest of the turn is a historical average. Exact once the window
    reaches the end of the turn, so a deep enough window equals the full oracle."""
    total, cur = 0.0, r
    for _ in range(k):
        total += cur.ideal
        if cur.nxt is None:
            return total
        total += cur.gap
        cur = cur.nxt
    return total + cheap.service + (cheap.remaining_calls(cur) - 1) * (cheap.service + cheap.gap)


def estimate_remaining(r: Req, p: Policy, cheap: Cheap, profile: Profile) -> float:
    """Remaining time in this turn, each ingredient either true (the policy's flags) or historical."""
    if p.lookahead is not None:
        return estimate_windowed(r, p.lookahead, cheap)
    n = float(r.rem_calls) if p.calls else cheap.remaining_calls(r)
    if p.scope:                       # a session-level deadline: split the remaining calls into this user turn and the rest
        later = cheap.future_calls(r.uturn) if p.scope == "cur" else r.fut_turns * cheap.calls_per_uturn
        n = float(r.rem_uturn) + later
    if p.forecast:
        forecast = getattr(r, f"pred_{p.forecast}")
        n = forecast if forecast > 0 else n
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
    preempt: str = "none"            # none, or priority: a waiting request that does not fit may
                                     # preempt running ones the policy ranks below it


@dataclass
class TurnResult:
    arrival: float
    latency: float
    budget: float
    z: float
    missed: bool
    feasible: bool


class _State:
    __slots__ = ("s", "idx", "i", "resident", "idle_since", "next_use", "cat", "attained", "turn_arrival", "deadline",
                 "pred", "resume")

    def __init__(self, s: Session, idx: int):
        self.s, self.idx, self.i, self.resident = s, idx, 0, 0
        self.idle_since = self.next_use = 0.0
        self.cat = "__none__"
        self.pred = -1.0              # online forecast of the stall its last finished request is in (-1 = none)
        self.attained = self.turn_arrival = self.deadline = 0.0
        self.resume = None   # (prompt, base, completion) of a preempted request still to finish


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
        self.active: dict[int, dict] = {}   # running requests, by session
        self.preemptions = 0
        if cfg.preempt not in ("none", "priority"):
            raise ValueError(f"preempt {cfg.preempt!r}: use none or priority")

    def _push(self, t: float, kind: str, st: _State) -> int:
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, st.idx))
        return self.seq

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
            now, seq, kind, idx = heapq.heappop(self.events)
            st = self.states[idx]
            if kind == "finish" and self.active.get(idx, {}).get("seq") != seq:
                continue                                  # its request was preempted
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

    def _lost(self, item, now: float) -> bool:
        """The policy judges this turn unable to make its deadline."""
        p, st = self.policy, self.states[item[2]]
        return not p.pure and ((p.kind == "edf" and now > st.deadline) or
                               (p.kind in ("llf", "srpt") and now + item[4] > st.deadline))

    def _pop(self, now: float):
        """Best ready request. Turns the policy judges already lost wait behind all the others."""
        while self.ready:
            item = heapq.heappop(self.ready)
            if self._lost(item, now):
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
        elapsed = now - v.idle_since
        if e == "online" and v.pred - elapsed > 0:      # the forecast stall has not run out yet
            return v.pred - elapsed
        return self.cheap.time_to_next_use(v.cat, elapsed)

    def _dispatch(self, now: float) -> None:
        cfg = self.cfg
        while self.running < cfg.slots:
            item = self._pop(now)
            if item is None:
                return
            st = self.states[item[2]]
            r = st.s.reqs[st.i]
            prompt, base, completion = st.resume or (r.prompt, r.base, r.completion)
            usable = min(st.resident, base)
            self.used -= st.resident - usable    # a prefix past the shared part is stale
            st.resident = usable
            need = min(prompt + completion, cfg.kv_tokens) - usable
            was_idle = self.idle.pop(st.idx, None) is not None
            short = need - (cfg.kv_tokens - self.used)
            if short > 0 and sum(v.resident for v in self.idle.values()) < short \
                    and not (cfg.preempt == "priority" and self._preempt_for(item, short, now)):
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
                cpu_hit = base - usable
            uncached = prompt - usable - cpu_hit
            service = (cfg.profile.overhead_ms + cfg.profile.prefill_ms * uncached + (cfg.profile.reload_ms or 0.0) * cpu_hit
                       + cfg.profile.decode_ms * completion) / 1000
            before_decode = service - cfg.profile.decode_ms * completion / 1000   # only read when preempting
            self.tokens_prompt += prompt
            self.tokens_hit += usable
            st.resident = usable + need
            st.attained += service
            self.busy.append((now, service))
            self.active[st.idx] = dict(item=item, start=now, service=service, before_decode=before_decode,
                                       prompt=prompt, usable=usable, completion=completion,
                                       busy=len(self.busy) - 1, seq=self._push(now + service, "finish", st))

    def _preempt_for(self, item, short: float, now: float) -> bool:
        """Free `short` tokens for `item` by preempting running requests the policy ranks below
        it, lowest first: turns it judges lost, then by its own key. Each victim keeps what it
        has computed (its prompt and the output so far) as an evictable prefix and goes back in
        line with its original priority, as vLLM keeps a preempted request's blocks cached.
        Returns False, preempting nothing, if that cannot free enough."""
        if self._lost(item, now):
            return False
        below = [(not self._lost(a["item"], now), a["item"]) for a in self.active.values()
                 if self._lost(a["item"], now) or a["item"][:2] > item[:2]]
        below.sort(reverse=True)
        below.sort(key=lambda x: x[0])                    # lost first, then lowest priority first
        free = sum(v.resident for v in self.idle.values())
        victims = []
        for _, v in below:
            if free >= short:
                break
            victims.append(v)
            free += self.states[v[2]].resident
        if free < short:
            return False
        for v in victims:
            self._preempt(self.states[v[2]], now)
        return True

    def _preempt(self, st: _State, now: float) -> None:
        a = self.active.pop(st.idx)
        p = self.cfg.profile
        elapsed = now - a["start"]
        if elapsed < a["before_decode"]:                  # still reading its prompt
            done = elapsed / a["before_decode"] if a["before_decode"] > 0 else 1.0
            computed, generated = a["usable"] + int(done * (a["prompt"] - a["usable"])), 0
        else:
            computed = a["prompt"]
            generated = min(a["completion"], int((elapsed - a["before_decode"]) * 1000 / p.decode_ms))
        keep = min(computed + generated, st.resident)
        self.used -= st.resident - keep
        st.resident = keep
        st.attained -= a["service"] - elapsed
        self.busy[a["busy"]] = (a["start"], elapsed)
        st.resume = (a["prompt"] + generated, a["prompt"] + generated, a["completion"] - generated)
        st.idle_since = now
        self.idle[st.idx] = st
        self.running -= 1
        self.preemptions += 1
        heapq.heappush(self.ready, a["item"])

    def _on_finish(self, st: _State, now: float) -> None:
        self.running -= 1
        self.active.pop(st.idx, None)
        st.resume = None
        r = st.s.reqs[st.i]
        if r.last_in_turn:
            budget = st.deadline - st.turn_arrival
            latency = now - st.turn_arrival
            z = st.s.z[r.turn]
            self.turns.append(TurnResult(st.turn_arrival, latency, budget, z, latency > budget, z <= budget))
        st.cat, st.idle_since = r.cat if not r.last_in_turn else "__none__", now
        st.pred = -1.0 if r.last_in_turn else r.pred_stall    # a finished turn is known to be over: the table covers it
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
        "p99_latency": lat[int(0.99 * len(lat))] if lat else None,
        "mean_slowdown": sum(x.latency / x.z for x in t if x.z > 0) / sum(1 for x in t if x.z > 0)
        if any(x.z > 0 for x in t) else None,
        "hit_rate": sim.tokens_hit / sim.tokens_prompt if sim.tokens_prompt else None,
        "tokens_evicted": sim.tokens_evicted,
        "preemptions": sim.preemptions,
    }
    if horizon is not None:
        out["utilization"] = sim.utilization(0.1 * horizon, 0.9 * horizon)
    return out
