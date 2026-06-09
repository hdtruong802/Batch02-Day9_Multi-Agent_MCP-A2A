"""
Supervisor Agent — phân công workers theo loại câu hỏi.

Pattern: Supervisor nhận query → chọn 1+ research workers → gom context → Answer worker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..rag_config import OPENAI_API_KEY
from .workers import AnswerWorker, LegalWorker, NewsWorker, WorkerResult

LEGAL_KEYWORDS = (
    "luật", "luat", "điều", "dieu", "nghị định", "nghi dinh", "hình phạt",
    "hinh phat", "tội", "toi", "cai nghiện", "cai nghien", "pháp luật",
    "phap luat", "bộ luật", "bo luat", "quy định", "quy dinh", "tàng trữ",
    "tang tru", "sử dụng trái phép", "chat ma tuy", "chất ma túy",
)

NEWS_KEYWORDS = (
    "nghệ sĩ", "nghe si", "ca sĩ", "ca si", "bị bắt", "bi bat", "tin tức",
    "tin tuc", "báo", "bao", "vnexpress", "vụ án", "vu an", "chi dân",
    "miu lê", "miu le", "andrea", "long nhật", "sao việt",
)


@dataclass
class SupervisorPlan:
    """Kế hoạch điều phối của Supervisor."""

    query: str
    workers_to_run: list[str]
    reasoning: str
    worker_results: list[WorkerResult] = field(default_factory=list)


def _keyword_score(query: str, keywords: tuple[str, ...]) -> int:
    q = query.lower()
    return sum(1 for kw in keywords if kw in q)


def route_query_rule_based(query: str) -> SupervisorPlan:
    """Routing bằng keyword khi không có LLM."""
    legal_score = _keyword_score(query, LEGAL_KEYWORDS)
    news_score = _keyword_score(query, NEWS_KEYWORDS)

    if legal_score > 0 and news_score > 0:
        workers = ["legal", "news"]
        reasoning = "Câu hỏi liên quan cả pháp luật và tin tức — kích hoạt cả 2 research workers."
    elif news_score > legal_score:
        workers = ["news"]
        reasoning = "Câu hỏi thiên về tin tức / vụ án / nghệ sĩ — giao cho News Worker."
    elif legal_score > 0:
        workers = ["legal"]
        reasoning = "Câu hỏi thiên về pháp luật — giao cho Legal Worker."
    else:
        workers = ["legal", "news"]
        reasoning = "Không rõ domain — kích hoạt cả Legal và News workers để đủ evidence."

    return SupervisorPlan(query=query, workers_to_run=workers, reasoning=reasoning)


def route_query_llm(query: str) -> SupervisorPlan | None:
    """Routing bằng LLM (gpt-4o-mini) nếu có API key."""
    api_key = (OPENAI_API_KEY or "").strip()
    if not api_key or api_key.startswith("sk-xxx"):
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là Supervisor agent cho hệ thống RAG về ma tuý Việt Nam. "
                        "Chọn workers cần chạy từ: legal, news (có thể chọn một hoặc cả hai). "
                        'Trả về JSON: {"workers": ["legal"], "reasoning": "..."}'
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        workers = [w for w in data.get("workers", []) if w in ("legal", "news")]
        if not workers:
            return None
        return SupervisorPlan(
            query=query,
            workers_to_run=workers,
            reasoning=data.get("reasoning", "LLM routing"),
        )
    except Exception:
        return None


def plan_query(query: str) -> SupervisorPlan:
    """Tạo kế hoạch: ưu tiên LLM, fallback keyword."""
    return route_query_llm(query) or route_query_rule_based(query)


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in chunks:
        key = chunk.get("content", "")[:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    unique.sort(key=lambda x: x.get("score", 0), reverse=True)
    return unique


def merge_worker_chunks(results: list[WorkerResult], top_k: int = 5) -> list[dict]:
    """Gom và dedupe chunks từ các research workers."""
    combined: list[dict] = []
    for result in results:
        if result.worker_id in ("legal", "news"):
            combined.extend(result.chunks)
    return _dedupe_chunks(combined)[:top_k]


class Supervisor:
    """Điều phối Legal + News workers, sau đó Answer worker."""

    def __init__(self):
        self.legal_worker = LegalWorker()
        self.news_worker = NewsWorker()
        self.answer_worker = AnswerWorker()
        self._workers = {
            "legal": self.legal_worker,
            "news": self.news_worker,
        }

    def run(self, query: str, *, top_k: int = 5) -> dict:
        plan = plan_query(query)

        for worker_id in plan.workers_to_run:
            worker = self._workers.get(worker_id)
            if worker:
                plan.worker_results.append(worker.run(query, top_k=top_k))

        merged_chunks = merge_worker_chunks(plan.worker_results, top_k=top_k)
        answer_result = self.answer_worker.run(
            query, top_k=top_k, context_chunks=merged_chunks
        )
        plan.worker_results.append(answer_result)

        return {
            "answer": answer_result.metadata.get("answer", ""),
            "sources": answer_result.chunks,
            "retrieval_source": answer_result.metadata.get("retrieval_source", "hybrid"),
            "supervisor_plan": plan.reasoning,
            "workers_used": plan.workers_to_run + ["answer"],
            "worker_trace": [
                {
                    "worker_id": r.worker_id,
                    "worker_name": r.worker_name,
                    "summary": r.summary,
                    "chunk_count": len(r.chunks),
                }
                for r in plan.worker_results
            ],
        }
