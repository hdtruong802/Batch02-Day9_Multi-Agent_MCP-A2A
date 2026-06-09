"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Semantic + Lexical → RRF merge → Cross-encoder rerank → PageIndex fallback.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _normalize_scores(results: list[dict]) -> list[dict]:
    if not results:
        return []
    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r["score"] = 1.0
        return results
    for r in results:
        r["score"] = (r["score"] - min_s) / (max_s - min_s)
    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.
    """
    fetch_k = max(top_k * 2, 10)

    dense_results = semantic_search(query, top_k=fetch_k)
    sparse_results = lexical_search(query, top_k=fetch_k)

    if not dense_results and not sparse_results:
        fallback = pageindex_search(query, top_k=top_k)
        return fallback

    dense_norm = _normalize_scores([r.copy() for r in dense_results])
    sparse_norm = _normalize_scores([r.copy() for r in sparse_results])

    merged = rerank_rrf([dense_norm, sparse_norm], top_k=fetch_k)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    if not final_results or final_results[0]["score"] < score_threshold:
        best = final_results[0]["score"] if final_results else 0.0
        print(
            f"  ⚠ Hybrid score ({best:.3f}) < threshold ({score_threshold}). "
            "Fallback → PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
