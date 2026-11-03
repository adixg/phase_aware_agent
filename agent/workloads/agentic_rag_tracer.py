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
    title = title.strip()
    if len(title) >= 2 and title[0] == title[-1] == '"':   # CSV-style quoting: inner " is doubled
        title = title[1:-1].replace('""', '"')
    return title, body.strip()


# Reduced slice for the 50-100 job experiments: (num_hops, category) -> task count.
# Deliberately skewed toward 3/4-hop tasks (the full set is ~66% 2-hop) because hop count
# drives the number of retrieve/generate rounds per job, i.e. the job-length spread the
# scheduler has to handle. Each hop level is split evenly between comparison and inference.
SLICE_PLAN: dict[tuple[int, str], int] = {
    (2, "comparison"): 20, (2, "inference"): 20,
    (3, "comparison"): 18, (3, "inference"): 18,
    (4, "comparison"): 12, (4, "inference"): 12,
}


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

    def reduced_slice(self, plan: dict[tuple[int, str], int] | None = None, *, seed: int = 0) -> list[Task]:
        """Stratified sample: `plan[(num_hops, category)]` tasks from each cell, seeded and
        returned in file order. The retriever corpus for the slice is `pool_documents(slice)`."""
        plan = SLICE_PLAN if plan is None else plan
        cells: dict[tuple[int, str], list[Task]] = {}
        for t in self.tasks():
            cells.setdefault((t.meta["num_hops"], t.meta["category"]), []).append(t)
        out = []
        for cell, n in plan.items():
            have = cells.get(cell, [])
            if len(have) < n:
                raise ValueError(f"slice wants {n} tasks for {cell} but only {len(have)} exist")
            out += select(have, n, seed)
        order = {t.task_id: i for i, t in enumerate(self.tasks())}
        return sorted(out, key=lambda t: order[t.task_id])
