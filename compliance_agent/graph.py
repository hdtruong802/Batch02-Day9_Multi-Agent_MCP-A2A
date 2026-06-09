"""Compliance Agent LangGraph definition — single LLM call (no ReAct loop)."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from common.llm import get_llm

COMPLIANCE_SYSTEM_PROMPT = """You are a senior regulatory compliance officer.

Answer in concise bullet points only. Keep your entire response under 150 words.
Cover key agencies (SEC, FTC, DOJ), civil/criminal remedies, and officer liability.
No disclaimers unless essential.
"""


class ComplianceState(TypedDict):
    question: str
    answer: str


async def answer_compliance(state: ComplianceState) -> dict:
    llm = get_llm()
    result = await llm.ainvoke(
        [
            SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"answer": result.content}


def create_graph():
    graph = StateGraph(ComplianceState)
    graph.add_node("answer_compliance", answer_compliance)
    graph.set_entry_point("answer_compliance")
    graph.add_edge("answer_compliance", END)
    return graph.compile()
