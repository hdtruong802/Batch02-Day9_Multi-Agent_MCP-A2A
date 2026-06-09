"""Customer Agent LangGraph — direct delegation to Law Agent (no ReAct loop)."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class CustomerState(TypedDict):
    question: str
    answer: str


def build_graph(trace_id: str, context_id: str, depth: int) -> Any:
    """Build a per-request graph that delegates immediately to the Law Agent."""

    async def delegate_to_law(state: CustomerState) -> dict:
        from common.a2a_client import delegate
        from common.registry_client import discover

        question = state["question"]
        logger.info(
            "Customer delegate_to_legal_agent | trace=%s context=%s depth=%d",
            trace_id,
            context_id,
            depth,
        )
        try:
            endpoint = await discover("legal_question")
            result = await delegate(
                endpoint=endpoint,
                question=question,
                context_id=context_id,
                trace_id=trace_id,
                depth=depth + 1,
            )
            if not result:
                return {"answer": "The Law Agent returned an empty response. Please try again."}
            return {"answer": result}
        except Exception as exc:
            logger.exception("delegate_to_legal_agent failed: %s", exc)
            return {"answer": f"Could not reach the Law Agent: {exc}"}

    graph = StateGraph(CustomerState)
    graph.add_node("delegate", delegate_to_law)
    graph.set_entry_point("delegate")
    graph.add_edge("delegate", END)
    return graph.compile()
