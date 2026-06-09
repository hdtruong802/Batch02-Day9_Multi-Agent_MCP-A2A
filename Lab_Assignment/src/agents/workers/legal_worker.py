"""Legal Research Worker — truy xuất văn bản pháp luật."""

from __future__ import annotations

from .base import BaseWorker, WorkerResult
from ...task9_retrieval_pipeline import retrieve


class LegalWorker(BaseWorker):
    """Worker chuyên domain pháp luật (luật, nghị định, điều khoản)."""

    worker_id = "legal"
    worker_name = "Legal Research Worker"

    def run(self, query: str, *, top_k: int = 5) -> WorkerResult:
        fetch_k = max(top_k * 3, 12)
        all_results = retrieve(query, top_k=fetch_k)

        legal_chunks = [
            r for r in all_results
            if r.get("metadata", {}).get("type") == "legal"
            or "luat" in r.get("metadata", {}).get("source", "").lower()
            or "nghi-dinh" in r.get("metadata", {}).get("source", "").lower()
        ][:top_k]

        if not legal_chunks and all_results:
            legal_chunks = all_results[:top_k]

        for chunk in legal_chunks:
            chunk["worker"] = self.worker_id

        return WorkerResult(
            worker_id=self.worker_id,
            worker_name=self.worker_name,
            chunks=legal_chunks,
            summary=f"Truy xuất {len(legal_chunks)} chunk pháp luật",
            metadata={"domain": "legal", "requested_top_k": top_k},
        )
