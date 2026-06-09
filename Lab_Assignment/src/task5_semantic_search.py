"""
Task 5 — Semantic Search Module.

Dense retrieval trên vector store đã index ở Task 4 (Weaviate hoặc local fallback).
"""

from .rag_store import vector_search


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    return vector_search(query, top_k=top_k)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
