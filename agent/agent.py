from __future__ import annotations

import ast
import operator
import re
import time
from dataclasses import dataclass, field

from agent.llm_backend import LLMBackend
from serving.instrumentation.phases import Phase
from agent.retriever import LocalRetriever
from serving.instrumentation.tracer import Tracer

_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Arithmetic-only evaluator -- avoids a bare `eval` on tool input."""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression: {expr!r}")

    return _eval(ast.parse(expr, mode="eval").body)


@dataclass
class AgentResult:
    job_id: str
    question: str
    answer: str | None
    transcript: list[str] = field(default_factory=list)
    success: bool = False


class ReActAgent:
    """Thought -> Action -> Observation loop, instrumented with `Tracer`.

    Each iteration is exactly one LLM_GENERATE phase span followed by at
    most one RETRIEVE or TOOL_EXEC span for the action it chose. This 1:1
    mapping is what makes "phase heterogeneity" directly readable off the
    trace file without extra post-processing.
    """

    def __init__(self, llm: LLMBackend, retriever: LocalRetriever, *, max_steps: int = 8):
        self.llm = llm
        self.retriever = retriever
        self.max_steps = max_steps

    def run(self, job_id: str, question: str, tracer: Tracer) -> AgentResult:
        transcript = [f"Question: {question}"]
        answer = None
        success = False

        for step in range(self.max_steps):
            prompt = "\n".join(transcript)
            with tracer.phase(Phase.LLM_GENERATE, step=step) as meta:
                text = self.llm.generate(prompt, step)
                meta["prompt_tokens"] = len(prompt.split())
                meta["output_tokens"] = len(text.split())
            transcript.append(text)

            match = _ACTION_RE.search(text)
            if not match:
                transcript.append("Observation: could not parse an action; stopping.")
                break

            tool, arg = match.group(1).lower(), match.group(2).strip()

            if tool == "finish":
                answer = arg
                success = True
                break
            elif tool == "retrieve":
                with tracer.phase(Phase.RETRIEVE, step=step, query=arg) as meta:
                    hits = self.retriever.retrieve(arg, top_k=1)
                    meta["top_doc"] = hits[0].doc_id if hits else None
                    meta["score"] = hits[0].score if hits else 0.0
                observation = hits[0].text if hits else "No results found."
            elif tool == "calculator":
                with tracer.phase(Phase.TOOL_EXEC, step=step, tool="calculator", expr=arg) as meta:
                    try:
                        result = _safe_eval(arg)
                        meta["result"] = result
                        observation = str(result)
                    except Exception as e:  # scripted/malformed tool input
                        observation = f"error: {e}"
            else:
                observation = f"error: unknown tool '{tool}'"

            transcript.append(f"Observation: {observation}")

        return AgentResult(job_id=job_id, question=question, answer=answer, transcript=transcript, success=success)
