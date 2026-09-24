"""A local retriever with no external dependencies.

Step-1 scope: prove the phase boundary and latency shape of "retrieval" as
a distinct phase from LLM generation and tool execution. The ranking
quality (bag-of-words cosine) is intentionally unsophisticated -- swapping
in FAISS/BM25 later changes latency and quality but not the trace schema
or the phase boundary, which is the thing this step needs to get right.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float


class LocalRetriever:
    """Bag-of-words cosine similarity over a small in-memory corpus."""

    def __init__(self, corpus: dict[str, str], *, simulated_latency_s: tuple[float, float] = (0.02, 0.06)):
        self.corpus = corpus
        self._vectors = {doc_id: Counter(_tokenize(text)) for doc_id, text in corpus.items()}
        self._latency_range = simulated_latency_s

    @classmethod
    def from_dir(cls, path: str | Path, **kwargs) -> "LocalRetriever":
        path = Path(path)
        corpus = {p.stem: p.read_text() for p in sorted(path.glob("*.txt"))}
        return cls(corpus, **kwargs)

    def _cosine(self, a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b[t] for t in a if t in b)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 1) -> list[Hit]:
        # Simulated I/O/compute cost so trace durations are non-degenerate
        # even though the corpus is tiny; replaced by real latency once the
        # real retriever backend is wired in.
        import time

        time.sleep(random.uniform(*self._latency_range))

        q_vec = Counter(_tokenize(query))
        scored = [
            Hit(doc_id, self.corpus[doc_id], self._cosine(q_vec, vec))
            for doc_id, vec in self._vectors.items()
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]
