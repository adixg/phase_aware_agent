"""Common interface every workload loader implements.

Downstream code (the agent CLI, the open-loop harness, quality checks) only
depends on `Task`, `Document`, `Workload` and `get_workload`. It never needs
to know which dataset a job came from beyond the `workload` / `task_id`
fields, which are recorded in every trace summary.
"""

from __future__ import annotations

import random
import datetime
import re
import string
import unicodedata
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
    """Corpus for a retriever serving these tasks: their passages, deduplicated by
    (title, text). The same passage often backs several tasks under different ids;
    the first id seen is kept."""
    seen: dict[tuple[str, str], Document] = {}
    for t in tasks:
        for d in t.documents:
            seen.setdefault((d.title, d.text), d)
    return list(seen.values())


_PUNCT = set(string.punctuation) - {"."}

_MONTHS = {m: i for i, names in enumerate(
    ["jan january", "feb february", "mar march", "apr april", "may", "jun june", "jul july",
     "aug august", "sep sept september", "oct october", "nov november", "dec december"], 1)
    for m in names.split()}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_DATE_PATTERNS = [   # (regex, group order) -- matched against the whole answer only
    (re.compile(r"(\d{4})[-./ ](\d{1,2})[-./ ](\d{1,2})"), "ymd"),
    (re.compile(rf"{_DAY} ({_MON}),? (\d{{4}})"), "dmy"),
    (re.compile(rf"({_MON}) {_DAY},? (\d{{4}})"), "mdy"),
    (re.compile(rf"({_MON}),? (\d{{4}})"), "my"),
    (re.compile(r"(\d{4})-(\d{2})"), "ym"),
]


def _canonical_date(s: str) -> str | None:
    """Whole-answer dates -> "YYYY-MM-DD" (or "YYYY-MM" for month-year), else None.
    Handles ISO, "29 March 1964", "March 29th, 1964", "Mar. 29 1964" and similar.
    Numeric d/m/y vs m/d/y is ambiguous, so slash dates are left as plain text."""
    t = re.sub(r"[,.]+(?=\s|$)", "", s.strip())       # "Mar. 29, 1964." -> "Mar 29 1964"
    t = re.sub(r"^on ", "", re.sub(r"\s+", " ", t))
    for rx, order in _DATE_PATTERNS:
        m = rx.fullmatch(t)
        if not m:
            continue
        parts = dict(zip(order, m.groups()))
        month = _MONTHS.get(parts["m"], None) if not parts["m"].isdigit() else int(parts["m"])
        year, day = int(parts["y"]), int(parts.get("d", 1))
        try:
            datetime.date(year, month, day)
        except (ValueError, TypeError):
            return None
        return f"{year:04d}-{month:02d}-{day:02d}" if "d" in parts else f"{year:04d}-{month:02d}"
    return None


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip accents, drop punctuation/articles/extra
    spaces. Whole-answer dates are first converted to one format (see `_canonical_date`), so
    "1964-03-29" and "29 March 1964" compare equal. Thousands commas and decimal points inside numbers are handled so that
    "15,615" and "15.615" (or "10.3" and "103") do not collide."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if (date := _canonical_date(s)) is not None:
        return date
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def select(items: Sequence, limit: int | None, seed: int) -> list:
    """`limit`-sized seeded sample of `items`, or all of them in order."""
    if limit is None or limit >= len(items):
        return list(items)
    idx = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in idx]
