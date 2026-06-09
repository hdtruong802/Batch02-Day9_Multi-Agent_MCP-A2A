"""
Task 6 — Lexical Search Module (BM25).

BM25: TF × IDF với length normalization (k1=1.5, b=0.75).
Corpus được load từ chunks đã index ở Task 4.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .rag_store import load_chunks_cache

CORPUS: list[dict] = []
_BM25_INDEX: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[\w]+", text, flags=re.UNICODE)


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xây dựng BM25 index từ corpus."""
    global CORPUS, _BM25_INDEX
    CORPUS = corpus
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    _BM25_INDEX = BM25Okapi(tokenized_corpus)
    return _BM25_INDEX


def _ensure_bm25_index() -> BM25Okapi:
    global _BM25_INDEX
    if _BM25_INDEX is None:
        corpus = load_chunks_cache()
        if not corpus:
            from .task4_chunking_indexing import chunk_documents, load_documents
            from .rag_store import embed_and_cache_chunks

            docs = load_documents()
            chunks = chunk_documents(docs)
            embed_and_cache_chunks(chunks, show_progress=False)
            corpus = load_chunks_cache()
        build_bm25_index(corpus)
    return _BM25_INDEX  # type: ignore[return-value]


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    bm25 = _ensure_bm25_index()
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx].get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
