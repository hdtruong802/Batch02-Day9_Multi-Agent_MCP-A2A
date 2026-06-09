"""Phần 6: Tổng kết, ôn tập và đo latency.

Chạy:
    uv run python exercises/exercise_6_summary.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from common.llm import get_llm

QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


def print_review_answers() -> None:
    """Câu hỏi ôn tập — Phần 6 CODELAB."""
    print("=" * 70)
    print("CAU HOI ON TAP (Phan 6)")
    print("=" * 70)

    answers = [
        (
            "1. Khi nao dung single agent thay vi multi-agent?",
            "Dung single agent khi: domain don (1 chuyen gia du), luong xu ly don gian, "
            "khong can chay song song, team nho va muon deploy don gian. Multi-agent khi: "
            "nhieu domain (law/tax/compliance), can chuyen mon hoa prompt/tool, can scale "
            "hoac deploy doc lap tung agent.",
        ),
        (
            "2. Uu diem A2A so voi gRPC/REST thong thuong?",
            "A2A co Agent Card (self-description), task/message model chuan cho agent, "
            "dynamic discovery qua Registry, trace_id propagation, va hop dong ro rang cho "
            "agent-to-agent — khong can hardcode API tuy bien cho tung agent.",
        ),
        (
            "3. Prevent infinite delegation loops trong A2A?",
            "Dung MAX_DELEGATION_DEPTH (repo: depth=3), tang depth moi hop delegate, "
            "tu choi delegate khi dat max. Law Agent skip sub-agents khi depth >= max.",
        ),
        (
            "4. Tai sao can Registry? Co hardcode URL khong?",
            "Registry cho phep dynamic discovery — agents tu dang ky, scale/restart khong "
            "can sua config cac agent khac. Hardcode URL duoc cho demo nho nhung kem linh "
            "hoat khi deploy production (nhieu instance, load balance, failover).",
        ),
    ]

    for q, a in answers:
        print(f"\n{q}\n   -> {a}")

    print("\n" + "=" * 70)
    print("SO SANH 5 STAGES")
    print("=" * 70)
    print("Stage 1: Direct LLM          — don gian, khong tools")
    print("Stage 2: LLM + Tools         — RAG/tools, manual loop")
    print("Stage 3: ReAct Agent         — tu dong hoa tool loop")
    print("Stage 4: Multi-Agent         — chuyen gia song song in-process")
    print("Stage 5: Distributed A2A     — nhieu service HTTP + Registry")


def keyword_route(question: str) -> list[str]:
    """Routing toi uu: keyword-based, khong goi LLM."""
    q = question.lower()
    targets = []
    if any(kw in q for kw in ["tax", "irs", "taxes", "thuế"]):
        targets.append("call_tax")
    if any(kw in q for kw in ["compliance", "sec", "regulation", "regulatory"]):
        targets.append("call_compliance")
    return targets or ["aggregate"]


async def llm_route(question: str) -> dict:
    """Routing baseline (cu): 1 LLM call de quyet dinh routing."""
    llm = get_llm()
    messages = [
        SystemMessage(
            content=(
                'Reply with ONLY valid JSON: {"needs_tax": <bool>, "needs_compliance": <bool>}'
            )
        ),
        HumanMessage(content=question),
    ]
    result = await llm.ainvoke(messages)
    return {"raw": result.content}


async def benchmark_routing() -> None:
    """So sanh thoi gian routing keyword vs LLM."""
    print("\n" + "=" * 70)
    print("BENCHMARK ROUTING (Bai tap cong diem)")
    print("=" * 70)
    print(f"Cau hoi: {QUESTION}\n")

    t0 = time.perf_counter()
    keyword_targets = keyword_route(QUESTION)
    keyword_ms = (time.perf_counter() - t0) * 1000
    print(f"[Keyword routing] targets={keyword_targets}  time={keyword_ms:.2f}ms")

    t0 = time.perf_counter()
    llm_result = await llm_route(QUESTION)
    llm_s = time.perf_counter() - t0
    print(f"[LLM routing]     raw={llm_result['raw'][:80]}...")
    print(f"[LLM routing]     time={llm_s:.2f}s")

    saved = llm_s - (keyword_ms / 1000)
    print(f"\nTiet kiem routing: ~{saved:.2f}s / request (bo 1 LLM call o Law Agent)")


async def benchmark_stage4() -> None:
    """Do latency multi-agent in-process (Stage 4 — da toi uu)."""
    from stages.stage_4_milti_agent.main import create_graph

    print("\n" + "=" * 70)
    print("BENCHMARK STAGE 4 (Multi-Agent in-process)")
    print("=" * 70)

    graph = create_graph()
    t0 = time.perf_counter()
    result = await graph.ainvoke({
        "question": QUESTION,
        "law_analysis": "",
        "tax_result": "",
        "compliance_result": "",
        "privacy_result": "",
        "final_answer": "",
    })
    elapsed = time.perf_counter() - t0

    print(f"Latency Stage 4: {elapsed:.2f}s")
    print(f"Answer length: {len(result.get('final_answer', ''))} chars")
    print(f"Preview: {result.get('final_answer', '')[:300]}...")


def _read_stage5_latency() -> str | None:
    """Doc latency tu stage5_e2e_output.txt hoac test_client log."""
    import re

    root = os.path.join(os.path.dirname(__file__), "..")
    for name in ("stage5_e2e_output.txt",):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        match = re.search(r"Latency:\s*([\d.]+)s", text)
        if match:
            return match.group(1)
    return None


def print_stage5_latency_table() -> None:
    """Bang latency Stage 5 (bai cong diem). Cap nhat sau khi chay test_client."""
    measured = _read_stage5_latency()
    optimized_note = f"{measured}s (Gemini)" if measured else "(chay test_client.py de do)"

    print("\n" + "=" * 70)
    print("STAGE 5 — BANG LATENCY (BAI CONG DIEM)")
    print("=" * 70)
    print("| Cau hinh              | Latency (s) | Ghi chu                          |")
    print("|-----------------------|-------------|----------------------------------|")
    print("| Truoc toi uu          | ~900+       | LLM routing + OpenRouter free    |")
    print(f"| Sau toi uu (hien tai) | {optimized_note:<11} | Gemini + keyword + direct delegate |")
    print("")
    print("Cach do: .\\start_all.ps1 -> uv run python test_client.py")
    print("Doc dong 'Latency: X.XXs' trong output hoac stage5_e2e_output.txt")
    print("")
    print("Toi uu da ap dung:")
    print("  - Law Agent: keyword routing thay LLM check_routing (-1 LLM call)")
    print("  - Law/Tax/Compliance/Customer: prompt ngan hon")
    print("  - Tax + Compliance: 1 LLM call moi (bo ReAct), chay song song")
    print("  - Customer: delegate truc tiep (bo ReAct)")


async def main() -> None:
    load_dotenv()
    print_review_answers()
    await benchmark_routing()
    await benchmark_stage4()
    print_stage5_latency_table()
    print("\n" + "=" * 70)
    print("Hoan thanh Phan 6!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
