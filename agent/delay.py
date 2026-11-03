"""Injected retrieval latency, for emulating a slower search backend.

Our corpus is small and in memory, so a real search takes ~16 ms; production
retrieval (large index, network hop, reranker) is 100s of ms. A delay spec adds
that time inside the RETRIEVE span, recorded apart from the real search time.

    0.5                 fixed 500 ms
    uniform:0.2,1.0     uniform between 200 ms and 1 s
    lognormal:0.3,0.8   median 300 ms, sigma 0.8 (heavy right tail)

Each draw is seeded by (seed, job_id, agent step), so a task gets the same
delays across reruns and concurrency levels, whatever order threads run in.
"""

from __future__ import annotations

import math
import random
from typing import Callable

Sampler = Callable[[random.Random], float]


def parse_delay(spec: str) -> Sampler:
    """Turn a spec into a sampler; raises ValueError on a malformed one."""
    kind, _, params = spec.partition(":")
    try:
        if not params:
            value = float(kind)
            if value < 0:
                raise ValueError
            return lambda rng: value
        args = [float(x) for x in params.split(",")]
        if kind == "uniform" and len(args) == 2 and 0 <= args[0] <= args[1]:
            low, high = args
            return lambda rng: rng.uniform(low, high)
        if kind == "lognormal" and len(args) == 2 and args[0] > 0 and args[1] >= 0:
            mu, sigma = math.log(args[0]), args[1]
            return lambda rng: rng.lognormvariate(mu, sigma)
    except ValueError:
        pass
    raise ValueError(f"bad retrieve delay {spec!r}: use SECONDS, uniform:LOW,HIGH or lognormal:MEDIAN,SIGMA")


def job_delay(sampler: Sampler, seed: int, job_id: str) -> Callable[[int], float]:
    """Per-job delay function of the agent step."""
    return lambda step: sampler(random.Random(f"{seed}:{job_id}:{step}"))
