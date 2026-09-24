"""AgenticRAGTracer -- records per-phase spans for one agent job.

Each job produces one JSONL file: one line per phase span, plus a final
summary line. This is the raw material for the week-5 characterization
(phase heterogeneity, resource signatures) and the later scheduling
replay -- so the schema here is the contract the rest of the project
builds on. Keep it stable; add fields, don't rename them.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Span:
    job_id: str
    step: int
    phase: str
    start_ts: float
    end_ts: float
    duration_s: float
    meta: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Accumulates spans for a single job and writes them as JSONL."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.spans: list[Span] = []
        self._step = 0
        self._job_start = time.time()

    @contextmanager
    def phase(self, phase: str, **meta: Any) -> Iterator[dict[str, Any]]:
        """Time one phase. `meta` is recorded as given; the block may also
        mutate the yielded dict to add fields only known after the phase
        runs (e.g. output token count)."""
        step = self._step
        self._step += 1
        t0 = time.time()
        try:
            yield meta
        finally:
            t1 = time.time()
            self.spans.append(
                Span(
                    job_id=self.job_id,
                    step=step,
                    phase=phase,
                    start_ts=t0,
                    end_ts=t1,
                    duration_s=t1 - t0,
                    meta=dict(meta),
                )
            )

    def write(self, out_path: str | Path, *, success: bool, extra: dict[str, Any] | None = None) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        job_end = time.time()
        summary = {
            "record_type": "job_summary",
            "job_id": self.job_id,
            "start_ts": self._job_start,
            "end_ts": job_end,
            "duration_s": job_end - self._job_start,
            "num_steps": len(self.spans),
            "success": success,
            **(extra or {}),
        }
        with out_path.open("w") as f:
            for span in self.spans:
                record = {"record_type": "span", **asdict(span)}
                f.write(json.dumps(record) + "\n")
            f.write(json.dumps(summary) + "\n")
