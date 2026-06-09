"""
Task 7 — Reranking Module.

Default: cross-encoder via Jina API (multilingual).
Fallback: keyword overlap reranker khi không có JINA_API_KEY.
Cũng hỗ trợ MMR và RRF.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import requests

from .rag_config import JINA_API_KEY
from .rag_store import embed_query, get_embedding_model


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _keyword_overlap_score(query: str, content: str) -> float:
    q_tokens = set(re.findall(r"[\w]+", query.lower(), flags=re.UNICODE))
    c_tokens = set(re.findall(r"[\w]+", content.lower(), flags=re.UNICODE))
    if not q_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank bằng Jina cross-encoder API hoặc keyword overlap fallback."""
    if not candidates:
        return []

    if JINA_API_KEY:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": min(top_k, len(candidates)),
                },
                timeout=30,
            )
            response.raise_for_status()
            reranked = response.json().get("results", [])
            results = []
            for item in reranked:
                candidate = candidates[item["index"]].copy()
                candidate["score"] = float(item["relevance_score"])
                results.append(candidate)
            return results
        except Exception:
            pass

    scored = []
    for candidate in candidates:
        overlap = _keyword_overlap_score(query, candidate["content"])
        combined = 0.6 * overlap + 0.4 * float(candidate.get("score", 0))
        item = candidate.copy()
        item["score"] = combined
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    model = get_embedding_model()
    doc_embeddings = []
    for candidate in candidates:
        if "embedding" in candidate:
            doc_embeddings.append(candidate["embedding"])
        else:
            doc_embeddings.append(
                model.encode(candidate["content"], normalize_embeddings=True).tolist()
            )

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(query_embedding, doc_embeddings[idx])
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(doc_embeddings[idx], doc_embeddings[sel_idx])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = _cosine_sim(query_embedding, doc_embeddings[idx])
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """RRF(d) = Σ 1 / (k + rank_r(d))"""
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        result = content_map[content].copy()
        result["score"] = score
        results.append(result)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        query_embedding = embed_query(query)
        return rerank_mmr(query_embedding, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
