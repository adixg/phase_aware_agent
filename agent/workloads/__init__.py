from __future__ import annotations

from pathlib import Path

from .agentic_rag_tracer import AgenticRAGTracer
from .base import DEFAULT_DATA_DIR, Document, Task, Workload, pool_documents
from .calculator import Calculator
from .hotpotqa import HotpotQA
from .musique import MuSiQue

WORKLOADS: dict[str, type[Workload]] = {
    cls.name: cls for cls in (AgenticRAGTracer, HotpotQA, MuSiQue, Calculator)
}


def get_workload(name: str, data_dir: str | Path | None = None, **kwargs) -> Workload:
    try:
        cls = WORKLOADS[name]
    except KeyError:
        raise ValueError(f"unknown workload {name!r}; choose from {sorted(WORKLOADS)}") from None
    return cls(data_dir, **kwargs)


__all__ = ["Document", "Task", "Workload", "get_workload", "pool_documents", "WORKLOADS", "DEFAULT_DATA_DIR"]
