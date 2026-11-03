"""HotpotQA (distractor setting), read from the original JSON release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import DEFAULT_DATA_DIR, Document, Task, Workload, select

FILENAME = "hotpot_dev_distractor_v1.json"


class HotpotQA(Workload):
    name = "hotpotqa"

    def __init__(self, data_dir: str | Path | None = None):
        self.path = Path(data_dir or DEFAULT_DATA_DIR) / self.name / FILENAME

    def tasks(self, *, limit: int | None = None, seed: int = 0) -> Iterator[Task]:
        records = json.loads(self.path.read_text())
        for r in select(records, limit, seed):
            docs = tuple(
                Document(doc_id=f"{r['_id']}:{title}", title=title, text=" ".join(s.strip() for s in sents))
                for title, sents in r["context"]
            )
            yield Task(
                task_id=f"{self.name}/{r['_id']}",
                workload=self.name,
                question=r["question"],
                gold_answers=(r["answer"],),
                documents=docs,
                meta={
                    "type": r.get("type"),
                    "level": r.get("level"),
                    "supporting_titles": sorted({t for t, _ in r.get("supporting_facts", [])}),
                },
            )
