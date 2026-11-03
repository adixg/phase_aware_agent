"""EmbeddingRetriever behaviour, using a stub embedder so no model is needed."""

import time

import pytest

np = pytest.importorskip("numpy")

from agent.retriever import EmbeddingRetriever, Hit, LocalRetriever
from agent.workloads import Document

VOCAB = ["paris", "france", "tokyo", "japan", "river", "mountain"]


class BagEmbedder:
    """Deterministic stand-in for a model: one dimension per vocabulary word."""

    def __init__(self):
        self.query_calls = 0

    def _vec(self, text):
        words = text.lower().replace(".", " ").split()
        return [float(words.count(w)) for w in VOCAB]

    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        self.query_calls += 1
        return np.array([self._vec(t) for t in texts])


DOCS = [
    Document("a#hop_1", "Paris", "Paris is the capital of France."),
    Document("b#hop_1", "Tokyo", "Tokyo is the capital of Japan."),
    Document("c#hop_1", "Alps", "A mountain range."),
]


def test_same_interface_as_local_retriever_and_ranked():
    r = EmbeddingRetriever.from_documents(DOCS, BagEmbedder())
    hits = r.retrieve("capital of france paris", top_k=3)
    assert all(isinstance(h, Hit) for h in hits)
    assert [h.doc_id for h in hits][0] == "a#hop_1"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].text == "Paris. Paris is the capital of France."
    assert hasattr(LocalRetriever, "retrieve") and hasattr(LocalRetriever, "from_documents")


def test_top_k_is_capped_and_default_is_one():
    r = EmbeddingRetriever.from_documents(DOCS, BagEmbedder())
    assert len(r.retrieve("japan")) == 1
    assert len(r.retrieve("japan", top_k=10)) == 3
    assert r.retrieve("japan", top_k=0) == []


def test_empty_corpus_returns_nothing():
    assert EmbeddingRetriever({}, BagEmbedder()).retrieve("anything", top_k=3) == []


def test_zero_vector_query_does_not_crash():
    hits = EmbeddingRetriever.from_documents(DOCS, BagEmbedder()).retrieve("zzz unknown words", top_k=3)
    assert len(hits) == 3 and all(h.score == 0.0 for h in hits)


def test_no_simulated_sleep():
    r = EmbeddingRetriever.from_documents(DOCS, BagEmbedder())
    start = time.perf_counter()
    for _ in range(50):
        r.retrieve("paris")
    assert time.perf_counter() - start < 0.5   # LocalRetriever would take >= 1s (20 ms floor x 50)


def test_corpus_embedded_once_at_build_time():
    e = BagEmbedder()
    r = EmbeddingRetriever.from_documents(DOCS, e)
    r.retrieve("paris")
    r.retrieve("tokyo")
    assert e.query_calls == 2          # documents were not re-embedded per query


def test_real_model_ranks_the_gold_passage_first():
    pytest.importorskip("sentence_transformers")
    from agent.retriever import SentenceTransformerEmbedder
    try:
        embedder = SentenceTransformerEmbedder()
    except Exception as e:          # model not cached and no network
        pytest.skip(f"embedding model unavailable: {e}")
    r = EmbeddingRetriever.from_documents(DOCS, embedder)
    assert r.retrieve("What is the capital of France?")[0].doc_id == "a#hop_1"
    assert r.retrieve("Which city is Japan's capital?")[0].doc_id == "b#hop_1"
