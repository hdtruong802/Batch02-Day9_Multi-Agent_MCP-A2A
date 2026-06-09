"""Answer Synthesis Worker — tổng hợp câu trả lời có citation."""

from __future__ import annotations

from .base import BaseWorker, WorkerResult
from ...task10_generation import generate_from_chunks


class AnswerWorker(BaseWorker):
    """Worker sinh câu trả lời cuối từ context đã gom."""

    worker_id = "answer"
    worker_name = "Answer Synthesis Worker"

    def run(
        self,
        query: str,
        *,
        top_k: int = 5,
        context_chunks: list[dict] | None = None,
    ) -> WorkerResult:
        if context_chunks is None:
            context_chunks = []

        result = generate_from_chunks(query, context_chunks, top_k=top_k)

        return WorkerResult(
            worker_id=self.worker_id,
            worker_name=self.worker_name,
            chunks=result.get("sources", []),
            summary="Đã tổng hợp câu trả lời có citation",
            metadata={
                "answer": result.get("answer", ""),
                "retrieval_source": result.get("retrieval_source", "hybrid"),
            },
        )
