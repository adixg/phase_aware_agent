"""Pieces shared by every ReAct agent implementation (hand-written loop or LangGraph)."""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field

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
