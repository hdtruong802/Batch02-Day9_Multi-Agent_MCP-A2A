"""Tax Agent LangGraph definition — single LLM call (no ReAct loop)."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA.

Answer in concise bullet points only. Keep your entire response under 150 words.
Cover only the most critical penalties, agencies (IRS, DOJ), and liability points.
Do not repeat the question. No disclaimers unless essential.

Topics you may cover: tax evasion, IRC penalties, FBAR/FATCA, officer liability.
"""


class TaxState(TypedDict):
    question: str
    answer: str


async def answer_tax(state: TaxState) -> dict:
    llm = get_llm()
    result = await llm.ainvoke(
        [
            SystemMessage(content=TAX_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"answer": result.content}


def create_graph():
    graph = StateGraph(TaxState)
    graph.add_node("answer_tax", answer_tax)
    graph.set_entry_point("answer_tax")
    graph.add_edge("answer_tax", END)
    return graph.compile()
