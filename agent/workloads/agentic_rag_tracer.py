"""AgenticRAGTracer (HF: YqjMartin/AgenticRAGTracer), read from its six JSONL files.

Each record is {"hop_1": {...}, ..., "hop_N": {...}, "final_question", "final_answer"}.
Every hop has a question, answer and the gold passage `doc` ("\"Title\"\\n<text>").
The benchmark itself retrieves over a Wikipedia corpus that we do not ship, so
a task's retrievable passages are its own hop passages (pooled across tasks by
`pool_documents`). Intermediate hop questions are kept in `meta` for hop-aware
diagnostics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .base import DEFAULT_DATA_DIR, Document, Task, Workload, normalize_answer, select

_HOP_RE = re.compile(r"^hop_(\d+)$")
_FILE_RE = re.compile(r"^(\d+)hop_(\w+)\.jsonl$")


def _gold_answers(final_answer: str, last_hop: dict) -> tuple[str, ...]:
    """`final_answer` plus the last hop's `optional_answers`. Aliases only appear on hop 2
    onward, and the last hop's list is the one that describes `final_answer`. In ~1.5% of
    records it describes something else (e.g. "The birth of X" for gold "X"), so aliases
    are kept only when the list contains `final_answer`. Aliases that normalize to nothing
    (e.g. "#", "An") are dropped, since they would match an empty answer."""
    aliases = last_hop.get("optional_answers", [])
    if normalize_answer(final_answer) not in {normalize_answer(a) for a in aliases}:
        aliases = []
    return tuple(a for a in dict.fromkeys([final_answer, *aliases]) if normalize_answer(a))


def _split_doc(doc: str) -> tuple[str, str]:
    title, _, body = doc.partition("\n")
    return title.strip().strip('"'), body.strip()


class AgenticRAGTracer(Workload):
    name = "agentic_rag_tracer"

    def __init__(self, data_dir: str | Path | None = None):
        self.dir = Path(data_dir or DEFAULT_DATA_DIR) / self.name

    def _records(self) -> list[tuple[str, int, dict]]:
        out = []
        for path in sorted(self.dir.glob("*hop_*.jsonl")):
            if not _FILE_RE.match(path.name):
                continue
            with path.open() as f:
                for i, line in enumerate(l for l in f if l.strip()):
                    out.append((path.stem, i, json.loads(line)))
        return out

    def tasks(self, *, limit: int | None = None, seed: int = 0) -> Iterator[Task]:
        for stem, i, r in select(self._records(), limit, seed):
            hops = sorted((int(m.group(1)), v) for k, v in r.items() if (m := _HOP_RE.match(k)))
            task_id = f"{self.name}/{stem}/{i}"
            docs = []
            for n, hop in hops:
                title, body = _split_doc(hop.get("doc", ""))
                docs.append(Document(doc_id=f"{task_id}#hop_{n}", title=title, text=body))
            yield Task(
                task_id=task_id,
                workload=self.name,
                question=r["final_question"],
                gold_answers=_gold_answers(r["final_answer"], hops[-1][1]),
                documents=tuple(docs),
                meta={
                    "num_hops": len(hops),
                    "category": _FILE_RE.match(stem + ".jsonl").group(2),
                    "hop_questions": [h["question"] for _, h in hops],
                    "final_answer": r["final_answer"],
                },
            )
