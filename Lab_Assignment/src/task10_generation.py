"""
Task 10 — Generation Có Citation.

top_k=5: đủ evidence, tránh context quá dài.
top_p=0.9: đa dạng vừa phải; temperature=0.3: factual, ít hallucination.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .rag_config import OPENAI_API_KEY
from .task9_retrieval_pipeline import retrieve

# top_k=5: cân bằng evidence vs lost-in-the-middle
TOP_K = 5
# top_p=0.9: nucleus sampling — giữ token có xác suất tích lũy 90%
TOP_P = 0.9
# temperature thấp cho RAG factual
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2025, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Tránh lost in the middle: chunk quan trọng nhất ở đầu và cuối.

    Input (by score):  [1, 2, 3, 4, 5]
    Output:            [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    reordered: list[dict] = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    start = len(chunks) - 1
    if len(chunks) % 2 == 0:
        start = len(chunks) - 2

    for i in range(start, 0, -2):
        reordered.append(chunks[i])

    return reordered


def format_context(chunks: list[dict]) -> str:
    """Format chunks với source label để LLM cite."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        article = meta.get("article", "")
        label = f"[Document {i} | Source: {source} | Type: {doc_type}"
        if article:
            label += f" | {article}"
        label += "]"
        context_parts.append(f"{label}\n{chunk['content']}\n")
    return "\n---\n".join(context_parts)


def _generate_fallback_answer(query: str, context: str, chunks: list[dict]) -> str:
    """Trả lời rule-based khi không có OpenAI API key."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    top = chunks[0]
    source = top.get("metadata", {}).get("source", "Nguồn không xác định")
    article = top.get("metadata", {}).get("article", "")
    cite = f"[{source}"
    if article:
        cite += f", {article}"
    cite += "]"

    excerpt = top["content"][:500].strip()
    return (
        f"Dựa trên ngữ cảnh truy xuất {cite}: {excerpt}...\n\n"
        f"(Trả lời tóm tắt tự động — cần OPENAI_API_KEY để sinh câu trả lời đầy đủ cho: {query})"
    )


def generate_from_chunks(
    query: str,
    chunks: list[dict],
    *,
    top_k: int = TOP_K,
) -> dict:
    """
    Sinh câu trả lời từ chunks đã được Supervisor/workers gom sẵn.
    Dùng bởi Answer Worker trong pattern Supervisor-Workers.
    """
    reordered = reorder_for_llm(chunks[:top_k])
    context = format_context(reordered)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    def _fallback_response() -> dict:
        return {
            "answer": _generate_fallback_answer(query, context, reordered),
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid"),
        }

    api_key = (OPENAI_API_KEY or "").strip()
    if not api_key or api_key.startswith("sk-xxx"):
        return _fallback_response()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content or ""
        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid"),
        }
    except Exception:
        return _fallback_response()


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """End-to-end RAG generation có citation (retrieve + generate)."""
    chunks = retrieve(query, top_k=top_k)
    return generate_from_chunks(query, chunks, top_k=top_k)


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
