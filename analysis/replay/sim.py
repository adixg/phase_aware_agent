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
from collections import defaultdict, deque
from collections.abc import Callable
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
    predicted: bool = False   # a predictor's remaining time (Config.pred_*) instead of the flags above


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
    "llf_pred": Policy("llf", predicted=True),
    "srpt_pred": Policy("srpt", predicted=True),
    # baselines from the literature (Config.pin="ttl" and eviction="cachewise" complete them)
    "continuum": Policy("continuum"),   # Continuum: pinned sessions first, then program (session) arrival order
    "fewest": Policy("fewest"),         # CacheWise: the request needing the fewest extra cache tokens first
}
EVICTIONS = ("lru", "cheap", "oracle", "cachewise")
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


@dataclass
class Session:
    sid: str
    arrival: float
    reqs: list[Req]
    z: list[float]        # per turn: duration with no contention


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
        reqs, rs, turn, step = [], s["requests"], -1, 0
        for i, r in enumerate(rs):
            if i == 0 or (r["user"] and unit == "turn"):
                turn, step = turn + 1, 0
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


def step_bucket(step: int) -> int:
    """12, 13, 14, ... for steps 12-23, 24-47, 48-95, ...: a session can run to hundreds of steps."""
    return step if step < STEP_CAP else STEP_CAP + int(math.log2(step / STEP_CAP))


class Cheap:
    """Historical averages from training sessions: what a system knows without a predictor."""

    def __init__(self, sessions: list[Session], profile: Profile):
        self.profile = profile
        by_key, by_step, gaps = defaultdict(list), defaultdict(list), defaultdict(list)
        completions, news, tool_gaps = [], [], []
        for s in sessions:
            for r in s.reqs:
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
    preempt: str = "none"            # none, or priority: a waiting request that does not fit may
                                     # preempt running ones the policy ranks below it
    # A stand-in for a predictor of each request's remaining time, for the *_pred policies: samples
    # are the true remaining time x a bias shared by the whole session (the predictor's error, which
    # the scheduler cannot see) x a per-sample spread (its stated uncertainty, which it can). Sessions
    # differ in how predictable they are: error sigma = pred_error x a lognormal(0, 0.5) factor.
    pred_error: float = 0.0
    pred_samples: int = 8
    pred_calibrated: bool = True     # the spread matches the session's own error; else it is pred_error for all
    pred_gate: float | None = None   # llf only: if the samples' coefficient of variation exceeds this, treat
                                     # the request as EDF would (no remaining time: order by deadline, no early give-up)
    # Predictive KV offload, with a CPU tier (profile.reload_ms). When a call ends and the session's next
    # gap is predicted to last at least offload_after_s, its cache moves to CPU at once; a reload is started
    # at the predicted return - transfer time - prefetch_margin_s if the GPU has free room then (it evicts
    # nothing). A call that starts before its reload is done waits for the rest of it. The prediction is the
    # true gap x lognormal(0, gap_error), drawn per gap; a session's end is predicted as an endless gap.
    offload_after_s: float | None = None
    prefetch_margin_s: float = 1.0
    gap_error: float = 0.0
    # A realistic CPU tier (needs profile.reload_ms). cpu_tokens: its capacity; every prefix is written to it
    # as it is computed (as LMCache does), least recently used sessions' copies are cut when it is full, and
    # a reload covers only what is still there. shared_link: reloads and prefetches queue on one GPU-CPU
    # link instead of each running at full speed.
    cpu_tokens: float = math.inf
    shared_link: bool = False
    # Continuum's KV time-to-live: after a tool call the session's cache cannot be evicted for
    # tau* = argmax_tau P(tool done by tau) x (recent queueing delay + cost to restore the cache) - tau,
    # P from the tool category's training-week gaps; with no other room, the latest-arrived pinned
    # session is evicted first.
    pin: str = "none"                # none | ttl
    # One GPU computes prompts one at a time: vLLM's prefill uses the whole GPU, so a request's uncached
    # prompt waits for the prefills ahead of it instead of every slot prefilling at full speed in parallel.
    # Decode stays per slot; use a decode speed measured under load, which already includes the stalls
    # other requests' prefills cause.
    shared_prefill: bool = False


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
                 "resume", "loaded_at", "prefetch_tokens", "prefetch_seq", "cpu", "cpu_touch", "pin_until",
                 "session_start")

    def __init__(self, s: Session, idx: int):
        self.s, self.idx, self.i, self.resident = s, idx, 0, 0
        self.idle_since = self.next_use = 0.0
        self.cat = "__none__"
        self.attained = self.turn_arrival = self.deadline = 0.0
        self.resume = None   # (prompt, base, completion) of a preempted request still to finish
        self.loaded_at = 0.0       # when a reload from CPU in progress completes
        self.prefetch_tokens = 0   # cache offloaded to CPU, waiting for its prefetch
        self.prefetch_seq = -1     # the prefetch event that is still current
        self.cpu = 0               # prefix tokens held in a finite CPU tier
        self.cpu_touch = 0.0
        self.pin_until = -1.0      # Continuum: no eviction before this
        self.session_start = 0.0


class Simulator:
    def __init__(self, sessions: list[Session], cfg: Config, cheap: Cheap | None = None, planner=None):
        self.cfg, self.cheap = cfg, cheap
        self.planner = planner     # planner.RolloutPlanner: picks among the best few ready requests
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
        self.tokens_offloaded = self.prefetches = 0
        self.transfer_wait = 0.0
        self.cpu_used = 0.0
        self.link_free = 0.0                  # when the shared GPU-CPU link is next idle
        self.link_wait = 0.0
        self.prefill_free = 0.0               # when the GPU is next free to compute a prompt
        self.prefill_wait = 0.0
        self.tokens_cpu_hit = 0.0
        self.recent_waits: deque = deque(maxlen=100)   # Continuum's sliding window of queueing delay
        self.pinned_evicted = 0
        finite_cpu = cfg.cpu_tokens < math.inf or cfg.shared_link
        if finite_cpu and cfg.profile.reload_ms is None:
            raise ValueError("cpu_tokens and shared_link describe a CPU tier: use a profile with reload_ms")
        if cfg.offload_after_s is not None and cfg.profile.reload_ms is None:
            raise ValueError("predictive offload needs a CPU tier: a profile with reload_ms")
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
                if self.planner is not None:
                    self.planner.on_ready(self, st, now)
            elif kind == "prefetch":
                if seq != st.prefetch_seq:
                    continue                              # superseded by a later offload
                self._on_prefetch(st, now)
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
        if st.i == 0:
            st.session_start = now
        p = self.policy
        est = 0.0
        if p.predicted:
            est = self._predicted_remaining(st, r)
        elif p.kind in ("llf", "srpt"):
            est = estimate_remaining(r, p, self.cheap, self.cfg.profile)
        key = {"fcfs": lambda: (now,),
               "returning": lambda: (r.step == 0, now),
               "las": lambda: (st.attained, now),
               "edf": lambda: (st.deadline, now),
               "llf": lambda: (st.deadline - est, now),
               "srpt": lambda: (est, now),
               "continuum": lambda: (st.pin_until < now, st.session_start, now),
               "fewest": lambda: (r.prompt + r.completion - min(st.resident, r.base), now)}[p.kind]()
        self.seq += 1
        heapq.heappush(self.ready, (key, self.seq, st.idx, now, est))

    def _lost(self, item, now: float) -> bool:
        """The policy judges this turn unable to make its deadline."""
        p, st = self.policy, self.states[item[2]]
        return (p.kind == "edf" and now > st.deadline) or \
            (p.kind in ("llf", "srpt") and now + item[4] > st.deadline)

    def _predicted_remaining(self, st: _State, r: Req) -> float:
        """The stand-in predictor's estimate (the median sample), or 0 for a gated-out request."""
        cfg = self.cfg
        truth = r.rem_service + r.rem_gaps
        if cfg.pred_error <= 0:
            return truth
        session = random.Random(f"{cfg.seed}:{st.s.sid}")
        sigma = cfg.pred_error * math.exp(session.gauss(0, 0.5))      # how predictable this session is
        bias = math.exp(session.gauss(0, sigma))                      # its error, the same for all its requests
        spread = sigma if cfg.pred_calibrated else cfg.pred_error
        draw = random.Random(f"{cfg.seed}:{st.s.sid}:{st.i}")
        samples = sorted(truth * bias * math.exp(draw.gauss(0, spread)) for _ in range(cfg.pred_samples))
        if cfg.pred_gate is not None and self.policy.kind == "llf":
            mean = sum(samples) / len(samples)
            sd = math.sqrt(sum((x - mean) ** 2 for x in samples) / len(samples))
            if mean <= 0 or sd / mean > cfg.pred_gate:
                return 0.0
        return samples[len(samples) // 2]

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
        if e == "cachewise" and v.cat == "__none__":     # waiting on the user or finished: these go first
            return 1e12 + now - v.idle_since
        return self.cheap.time_to_next_use(v.cat, now - v.idle_since)

    def _victim(self, now: float, ranks):
        """The idle session to cut next. With Continuum's pins, pinned ones are spared while any other
        has memory left, and then the latest-arrived pinned session goes first."""
        pool = self.idle.values()
        if self.cfg.pin == "ttl":
            free = [v for v in pool if v.pin_until < now]
            if not free:
                self.pinned_evicted += 1
                return max(pool, key=lambda x: (x.session_start, x.idx))
            pool = free
        return max(pool, key=lambda x: self._score(x, now, ranks))

    def _dispatch(self, now: float) -> None:
        cfg = self.cfg
        while self.running < cfg.slots:
            item = self._pop(now)
            if item is None:
                return
            if self.planner is not None and self.ready:
                item = self._plan(item, now)
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
                v = self._victim(now, ranks)
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
                if cfg.cpu_tokens < math.inf:            # only what the CPU tier still holds
                    cpu_hit = max(0, min(st.cpu, base) - usable)
            uncached = prompt - usable - cpu_hit
            service = (cfg.profile.overhead_ms + cfg.profile.prefill_ms * uncached + (cfg.profile.reload_ms or 0.0) * cpu_hit
                       + cfg.profile.decode_ms * completion) / 1000
            if cfg.shared_link and cpu_hit > 0:          # the reload waits its turn on the link
                start = max(now, self.link_free)
                self.link_free = start + cpu_hit * cfg.profile.reload_ms / 1000
                self.link_wait += start - now
                service += start - now
            if cfg.shared_prefill and uncached > 0:      # the prompt waits for the GPU to be free
                start = max(now, self.prefill_free)
                self.prefill_free = start + (cfg.profile.overhead_ms + cfg.profile.prefill_ms * uncached) / 1000
                self.prefill_wait += start - now
                service += start - now
            if cfg.pin == "ttl":
                self.recent_waits.append(now - item[3])
            if st.loaded_at > now and usable > 0:        # its prefetch is still copying in
                self.transfer_wait += st.loaded_at - now
                service += st.loaded_at - now
            st.loaded_at, st.prefetch_tokens, st.prefetch_seq = 0.0, 0, -1
            before_decode = service - cfg.profile.decode_ms * completion / 1000   # only read when preempting
            self.tokens_prompt += prompt
            self.tokens_hit += usable
            self.tokens_cpu_hit += cpu_hit
            st.resident = usable + need
            st.attained += service
            self.busy.append((now, service))
            self.active[st.idx] = dict(item=item, start=now, service=service, before_decode=before_decode,
                                       prompt=prompt, usable=usable, completion=completion,
                                       busy=len(self.busy) - 1, seq=self._push(now + service, "finish", st))

    def _plan(self, item, now: float):
        """Let the planner pick among the policy's best few ready requests; the rest go back in line."""
        cands = [item]
        while self.ready and len(cands) < self.planner.candidates:
            nxt = heapq.heappop(self.ready)
            if self._lost(nxt, now):
                heapq.heappush(self.demoted, (nxt[3], nxt[1], nxt))
            else:
                cands.append(nxt)
        choice = self.planner.choose(self, now, cands) if len(cands) > 1 else item
        for c in cands:
            if c is not choice:
                heapq.heappush(self.ready, c)
        return choice

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
        self.idle[st.idx] = st
        if self.cfg.cpu_tokens < math.inf:
            self._cpu_store(st, st.resident, now)
        if self.cfg.pin == "ttl" and r.gap is not None and r.cat != "__none__":
            st.pin_until = now + self._ttl(st, r.cat)
        st.i += 1
        if self.cfg.offload_after_s is not None:
            self._offload(st, r, now)
        if r.gap is None:
            st.next_use = math.inf
            if self.cfg.closed is not None and self.pending:
                self._push(now, "ready", self.states[self.pending.pop()])
            return
        st.next_use = now + r.gap
        self._push(st.next_use, "ready", st)

    def _cpu_store(self, st: _State, tokens: float, now: float) -> None:
        """Write-through: the session's prefix is in the CPU tier; least recently stored copies shrink to fit."""
        cap = self.cfg.cpu_tokens
        tokens = min(tokens, cap)
        self.cpu_used += tokens - st.cpu
        st.cpu, st.cpu_touch = tokens, now
        while self.cpu_used > cap:
            v = min((x for x in self.states if x.cpu > 0 and x is not st), key=lambda x: x.cpu_touch)
            cut = min(v.cpu, self.cpu_used - cap)
            v.cpu -= cut
            self.cpu_used -= cut

    def _ttl(self, st: _State, cat: str) -> float:
        """Continuum's time-to-live for a cache paused on a tool call of category `cat`."""
        cfg = self.cfg
        gaps = self.cheap.gaps.get(cat) or self.cheap.gaps.get("Other") or []
        if not gaps:
            return 0.0
        queueing = sum(self.recent_waits) / len(self.recent_waits) if self.recent_waits else 0.0
        restore = st.resident * (cfg.profile.reload_ms if cfg.profile.reload_ms is not None else cfg.profile.prefill_ms) / 1000
        n = len(gaps)
        best, best_tau = 0.0, 0.0                       # tau = 0: no pin
        for k in range(min(n, 64)):                      # candidate taus at quantiles of past durations
            i = (k + 1) * n // min(n, 64) - 1
            tau = gaps[i]
            value = (i + 1) / n * (queueing + restore) - tau
            if value > best:
                best, best_tau = value, tau
        return best_tau

    def _offload(self, st: _State, r: Req, now: float) -> None:
        """Move the cache to CPU if the coming gap is predicted to be long, and time its prefetch."""
        cfg = self.cfg
        predicted = math.inf
        if r.gap is not None:
            predicted = r.gap
            if cfg.gap_error > 0:
                predicted *= math.exp(random.Random(f"{cfg.seed}:gap:{st.s.sid}:{st.i}").gauss(0, cfg.gap_error))
        if predicted < cfg.offload_after_s or st.resident <= 0:
            return
        tokens = st.resident
        self.idle.pop(st.idx, None)
        self.used -= tokens
        self.tokens_offloaded += tokens
        st.resident = 0
        if predicted < math.inf:
            transfer = tokens * cfg.profile.reload_ms / 1000
            st.prefetch_tokens = tokens
            st.prefetch_seq = self._push(max(now, now + predicted - transfer - cfg.prefetch_margin_s), "prefetch", st)

    def _on_prefetch(self, st: _State, now: float) -> None:
        """Copy an offloaded cache back if the GPU has the room, without evicting anyone."""
        tokens = st.prefetch_tokens
        if self.cfg.cpu_tokens < math.inf:
            tokens = min(tokens, st.cpu)                 # the CPU tier may have cut its copy since
        if st.idx in self.active or st.resident > 0 or tokens <= 0 or self.cfg.kv_tokens - self.used < tokens:
            return
        start = now
        if self.cfg.shared_link:
            start = max(now, self.link_free)
            self.link_free = start + tokens * self.cfg.profile.reload_ms / 1000
        self.used += tokens
        st.resident = tokens
        st.loaded_at = start + tokens * self.cfg.profile.reload_ms / 1000
        st.prefetch_tokens = 0
        st.idle_since = now
        self.idle[st.idx] = st
        self.prefetches += 1

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
        "preemptions": sim.preemptions,
        "tokens_offloaded": sim.tokens_offloaded,
        "prefetches": sim.prefetches,
        "transfer_wait_s": sim.transfer_wait,
        "link_wait_s": sim.link_wait,
        "prefill_wait_s": sim.prefill_wait,
        "cpu_hit_rate": sim.tokens_cpu_hit / sim.tokens_prompt if sim.tokens_prompt else None,
        "pinned_evicted": sim.pinned_evicted,
    }
    if horizon is not None:
        out["utilization"] = sim.utilization(0.1 * horizon, 0.9 * horizon)
    return out
