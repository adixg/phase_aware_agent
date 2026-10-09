"""LLM backends for the ReAct agent.

`MockLLM` is a scripted stand-in used until vLLM is wired in (see the probe/
milestone). It exists so the tracer, phase boundaries, and trace schema can
be built and validated today, on a laptop, with zero model/GPU dependency.
Swapping in a real backend later means implementing the same `generate`
signature -- nothing else in the pipeline should need to change.

`VLLMBackend` talks to a running `vllm serve` over its OpenAI-compatible HTTP
API and returns a `Generation` carrying the server's real token counts.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Generation:
    """One ReAct turn plus the usage the serving engine reported for it."""

    text: str
    prompt_tokens: int
    output_tokens: int
    cached_tokens: int | None = None   # prefix-cache hits, when the server reports them


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, step: int) -> str | Generation:
        """Return the next ReAct turn for this step: a `Generation` with real token
        counts, or plain text (the agent then falls back to word-count estimates)."""


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


REACT_SYSTEM_PROMPT = """\
Answer the question by reasoning step by step and using tools.
Each turn, write exactly one Thought line and one Action line, then stop:
Thought: <your reasoning>
Action: <tool>[<input>]

Tools:
- retrieve[query]: search a collection of Wikipedia passages; returns the best-matching passage.
- calculator[expression]: evaluate arithmetic with + - * / and parentheses.
- finish[answer]: give the final answer and stop. Answer with a short entity, date, or number.

After each Action you will see an Observation with the tool's result. Stop right after
the Action line: never write the tool's result, a guess at it, or a second Action
yourself. Most questions need several facts: retrieve one fact at a time, with a
short, specific query. If a search does not return what you need, rephrase it with
different words; never repeat a query you have already made.

Example:
Question: Who was the father of the director of Jaws?
Thought: I need to find who directed Jaws.
Action: retrieve[Jaws film director]
Observation: Jaws is a 1975 American thriller film directed by Steven Spielberg.
Thought: Jaws was directed by Steven Spielberg. Now I need his father.
Action: retrieve[Steven Spielberg father]
Observation: Spielberg was born in Cincinnati to Arnold Spielberg, an electrical engineer.
Thought: Steven Spielberg's father is Arnold Spielberg.
Action: finish[Arnold Spielberg]"""


class VLLMBackend(LLMBackend):
    """Chat-completions client for `vllm serve` (any OpenAI-compatible server).

    The ReAct transcript so far is the user message; generation stops before the
    model can invent its own Observation. Greedy decoding by default, so reruns
    of a job follow the same trajectory.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout_s: float = 600.0,
        system_prompt: str = REACT_SYSTEM_PROMPT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt

    @staticmethod
    def served_model(base_url: str = "http://localhost:8000/v1") -> str:
        """Name of the (first) model the server is serving."""
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=30) as r:
            return json.load(r)["data"][0]["id"]

    def generate(self, prompt: str, step: int) -> Generation:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": ["Observation:"],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            out = json.load(r)
        usage = out.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return Generation(
            text=(out["choices"][0]["message"]["content"] or "").strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cached_tokens=details.get("cached_tokens"),
        )


DEFAULT_SCRIPT = [
    "Thought: I need to find the capital of France.\nAction: retrieve[capital of France]",
    "Thought: Now I need the population of that city.\nAction: retrieve[population of Paris]",
    "Thought: The question asks for double that population.\nAction: calculator[2.1 * 2]",
    "Thought: I have everything needed to answer.\nAction: finish[The population of Paris, doubled, is approximately 4.2 million.]",
]
