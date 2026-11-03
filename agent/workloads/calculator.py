"""Synthetic arithmetic tasks for controlled tool-latency experiments.

No retrieval corpus. Questions are generated from `seed`, gold answers are
computed with the same safe evaluator the agent's calculator tool uses.
`tool_latency_s` is recorded in each task's meta so the tool can honor it.
"""

from __future__ import annotations

import random
from typing import Iterator

from agent.common import _safe_eval

from .base import Task, Workload


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(round(x, 4))


class Calculator(Workload):
    name = "calculator"

    def __init__(self, data_dir=None, *, num_tasks: int = 100, tool_latency_s: float = 0.0):
        self.num_tasks = num_tasks
        self.tool_latency_s = tool_latency_s

    def tasks(self, *, limit: int | None = None, seed: int = 0) -> Iterator[Task]:
        rng = random.Random(seed)
        for i in range(limit if limit is not None else self.num_tasks):
            operands = [rng.randint(2, 99) for _ in range(rng.randint(2, 4))]
            ops = [rng.choice("+-*") for _ in operands[1:]]
            expr = str(operands[0]) + "".join(f" {o} {n}" for o, n in zip(ops, operands[1:]))
            yield Task(
                task_id=f"{self.name}/{seed}-{i}",
                workload=self.name,
                question=f"What is {expr}?",
                gold_answers=(_fmt(_safe_eval(expr)),),
                meta={"expression": expr, "tool_latency_s": self.tool_latency_s},
            )

    def is_correct(self, task: Task, answer: str | None) -> bool:
        if answer is None:
            return False
        try:
            return abs(float(answer.strip().replace(",", "")) - float(task.gold_answers[0])) < 1e-3
        except ValueError:
            return False
