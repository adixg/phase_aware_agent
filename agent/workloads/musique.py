"""MuSiQue (answerable variant), read from musique_ans_v1.0_dev.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import DEFAULT_DATA_DIR, Document, Task, Workload, select

FILENAME = "musique_ans_v1.0_dev.jsonl"


class MuSiQue(Workload):
    name = "musique"

    def __init__(self, data_dir: str | Path | None = None):
        self.path = Path(data_dir or DEFAULT_DATA_DIR) / self.name / FILENAME

    def tasks(self, *, limit: int | None = None, seed: int = 0) -> Iterator[Task]:
        with self.path.open() as f:
            records = [r for r in map(json.loads, filter(str.strip, f)) if r.get("answerable", True)]
        for r in select(records, limit, seed):
            docs = tuple(
                Document(doc_id=f"{r['id']}:{p['idx']}", title=p["title"], text=p["paragraph_text"])
                for p in r["paragraphs"]
            )
            yield Task(
                task_id=f"{self.name}/{r['id']}",
                workload=self.name,
                question=r["question"],
                gold_answers=tuple(dict.fromkeys([r["answer"], *r.get("answer_aliases", [])])),
                documents=docs,
                meta={
                    "num_hops": len(r.get("question_decomposition", [])),
                    "supporting_titles": [p["title"] for p in r["paragraphs"] if p.get("is_supporting")],
                },
            )
