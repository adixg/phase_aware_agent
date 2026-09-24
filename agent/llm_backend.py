"""LLM backends for the ReAct agent.

`MockLLM` is a scripted stand-in used until vLLM is wired in (see the probe/
milestone). It exists so the tracer, phase boundaries, and trace schema can
be built and validated today, on a laptop, with zero model/GPU dependency.
Swapping in a real backend later means implementing the same `generate`
signature -- nothing else in the pipeline should need to change.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, step: int) -> str:
        """Return the next ReAct turn's raw text for this step."""


class MockLLM(LLMBackend):
    """Replays a fixed ReAct trajectory, ignoring the actual prompt content.

    Simulated decode latency scales with the response's word count plus
    jitter, so even this stub produces LLM-phase durations that differ
    from retrieval/tool durations -- the thing week-5 characterization
    needs to see in the trace.
    """

    def __init__(self, script: list[str], *, tokens_per_second: float = 40.0, jitter_s: tuple[float, float] = (0.0, 0.03)):
        self.script = script
        self.tokens_per_second = tokens_per_second
        self.jitter_s = jitter_s

    def generate(self, prompt: str, step: int) -> str:
        if step >= len(self.script):
            raise IndexError(f"MockLLM script has no turn for step {step}")
        text = self.script[step]
        output_tokens = max(1, len(text.split()))
        decode_s = output_tokens / self.tokens_per_second
        time.sleep(decode_s + random.uniform(*self.jitter_s))
        return text


DEFAULT_SCRIPT = [
    "Thought: I need to find the capital of France.\nAction: retrieve[capital of France]",
    "Thought: Now I need the population of that city.\nAction: retrieve[population of Paris]",
    "Thought: The question asks for double that population.\nAction: calculator[2.1 * 2]",
    "Thought: I have everything needed to answer.\nAction: finish[The population of Paris, doubled, is approximately 4.2 million.]",
]
