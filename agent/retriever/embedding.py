"""Dense retriever: cosine similarity over passage embeddings.

Same surface as `LocalRetriever` (`from_documents`, `retrieve(query, top_k) -> list[Hit]`),
but with real retrieval cost: the corpus is embedded once at construction, and `retrieve`
is one query embedding plus a matrix product, with no simulated sleep.

The embedding model sits behind the small `Embedder` protocol, so swapping it is either a
model-name change (`SentenceTransformerEmbedder("other/model")`) or, for a non-sentence-
transformers backend, any object with `embed_documents` / `embed_queries`.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .local import Hit

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE v1.5 retrieves best when queries (not passages) carry this instruction.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-base-en-v1.5": _BGE_QUERY_PREFIX,
    "BAAI/bge-large-en-v1.5": _BGE_QUERY_PREFIX,
}


class Embedder(Protocol):
    """Maps text to 2-D float arrays, one row per input. Rows need not be normalized."""

    def embed_documents(self, texts: Sequence[str]): ...

    def embed_queries(self, texts: Sequence[str]): ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        query_prefix: str | None = None,
        doc_prefix: str = "",
        device: str | None = None,
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer  # optional dependency (extra: "embedding")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = _QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size

    def _encode(self, texts: Sequence[str], prefix: str):
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_documents(self, texts: Sequence[str]):
        return self._encode(texts, self.doc_prefix)

    def embed_queries(self, texts: Sequence[str]):
        return self._encode(texts, self.query_prefix)


class EmbeddingRetriever:
    """Dense cosine retrieval over a small in-memory corpus."""

    def __init__(self, corpus: dict[str, str], embedder: Embedder | None = None):
        import numpy as np

        self._np = np
        self.corpus = corpus
        self.embedder = embedder or SentenceTransformerEmbedder()
        self._ids = list(corpus)
        self._matrix = self._normalize(self.embedder.embed_documents([corpus[i] for i in self._ids]))

    @classmethod
    def from_documents(cls, documents, embedder: Embedder | None = None) -> "EmbeddingRetriever":
        """Build from `agent.workloads.Document`s (anything with doc_id/title/text)."""
        return cls({d.doc_id: f"{d.title}. {d.text}" for d in documents}, embedder)

    def _normalize(self, vectors):
        np = self._np
        m = np.asarray(vectors, dtype=np.float32)
        if m.size == 0:
            return m.reshape(0, 0)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(norms, 1e-12)

    def retrieve(self, query: str, top_k: int = 1) -> list[Hit]:
        if not self._ids or top_k <= 0:
            return []
        np = self._np
        q = self._normalize(self.embedder.embed_queries([query]))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self._ids[i], self.corpus[self._ids[i]], float(scores[i])) for i in order]
