"""
Supervisor-Workers pipeline — entry point cho multi-agent RAG.

Usage:
    from src.agents.pipeline import run_supervisor_pipeline
    result = run_supervisor_pipeline("Hình phạt tàng trữ ma tuý?")
"""

from __future__ import annotations

from .supervisor import Supervisor

_supervisor: Supervisor | None = None


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor


def run_supervisor_pipeline(query: str, *, top_k: int = 5) -> dict:
    """
    Chạy full Supervisor-Workers flow.

    Returns:
        dict với answer, sources, workers_used, worker_trace, supervisor_plan
    """
    return get_supervisor().run(query, top_k=top_k)
