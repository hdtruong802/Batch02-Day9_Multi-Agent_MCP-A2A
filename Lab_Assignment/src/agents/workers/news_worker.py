"""News Research Worker — truy xuất tin tức / vụ án nghệ sĩ."""

from __future__ import annotations

from .base import BaseWorker, WorkerResult
from ...task9_retrieval_pipeline import retrieve


class NewsWorker(BaseWorker):
    """Worker chuyên domain tin tức (bài báo, vụ án, nghệ sĩ)."""

    worker_id = "news"
    worker_name = "News Research Worker"

    def run(self, query: str, *, top_k: int = 5) -> WorkerResult:
        fetch_k = max(top_k * 3, 12)
        all_results = retrieve(query, top_k=fetch_k)

        news_chunks = [
            r for r in all_results
            if r.get("metadata", {}).get("type") == "news"
            or r.get("metadata", {}).get("source", "").startswith("news_")
        ][:top_k]

        if not news_chunks and all_results:
            news_chunks = all_results[:top_k]

        for chunk in news_chunks:
            chunk["worker"] = self.worker_id

        return WorkerResult(
            worker_id=self.worker_id,
            worker_name=self.worker_name,
            chunks=news_chunks,
            summary=f"Truy xuất {len(news_chunks)} chunk tin tức",
            metadata={"domain": "news", "requested_top_k": top_k},
        )
