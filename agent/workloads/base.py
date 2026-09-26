"""Common interface every workload loader implements.

Downstream code (the agent CLI, the open-loop harness, quality checks) only
depends on `Task`, `Document`, `Workload` and `get_workload`. It never needs
to know which dataset a job came from beyond the `workload` / `task_id`
fields, which are recorded in every trace summary.
"""

from __future__ import annotations

import random
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Task:
    task_id: str                       # unique across workloads: "<workload>/<local id>"
    workload: str
    question: str
    gold_answers: tuple[str, ...]      # accepted answers; empty if unknown
    documents: tuple[Document, ...] = ()   # passages this task's retrieval can draw on
    meta: dict[str, Any] = field(default_factory=dict)   # dataset-specific extras (hops, type, ...)


class Workload(ABC):
    name: str

    @abstractmethod
    def tasks(self, *, limit: int | None = None, seed: int = 0) -> Iterator[Task]:
        """Yield tasks. With `limit`, a seeded random sample (deterministic for a
        given seed); without it, every task in file order."""

    def is_correct(self, task: Task, answer: str | None) -> bool:
        """Exact match after normalization against any accepted answer."""
        if answer is None:
            return False
        got = normalize_answer(answer)
        return any(got == normalize_answer(g) for g in task.gold_answers)


def pool_documents(tasks: Iterable[Task]) -> list[Document]:
    """Corpus for a retriever serving these tasks: their passages, deduplicated by id."""
    seen: dict[str, Document] = {}
    for t in tasks:
        for d in t.documents:
            seen.setdefault(d.doc_id, d)
    return list(seen.values())


def normalize_answer(s: str) -> str:
    """Standard SQuAD-style normalization: lowercase, drop punctuation/articles/extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def select(items: Sequence, limit: int | None, seed: int) -> list:
    """`limit`-sized seeded sample of `items`, or all of them in order."""
    if limit is None or limit >= len(items):
        return list(items)
    idx = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in idx]
