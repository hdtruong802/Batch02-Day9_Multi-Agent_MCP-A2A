"""
Task 8 — PageIndex Vectorless RAG.

Ưu tiên PageIndex SDK (cloud). Fallback: structural search theo section/Điều
trong markdown khi không có API key hoặc SDK lỗi.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from .rag_config import PAGEINDEX_API_KEY, PAGEINDEX_DOC_IDS_PATH, STANDARDIZED_DIR
from .rag_store import ensure_processed_dir

load_dotenv()

STANDARDIZED_DIR = STANDARDIZED_DIR


def _split_into_sections(content: str, source: str) -> list[dict]:
    """Tách markdown thành sections theo heading / Điều luật."""
    pattern = r"(?=\n(?:#+\s+|\*\*Điều\s+\d+|\*\*Chương\s+[IVXLC\d]+))"
    parts = re.split(pattern, content)
    sections = []
    for part in parts:
        text = part.strip()
        if len(text) < 80:
            continue
        title_match = re.search(
            r"(?:^#+\s+(.+)$|^\*\*(Điều\s+\d+[^*]*)\*\*|^\*\*(Chương\s+[^*]+)\*\*)",
            text,
            flags=re.MULTILINE,
        )
        title = ""
        if title_match:
            title = next(g for g in title_match.groups() if g)
        sections.append(
            {
                "content": text,
                "metadata": {"source": source, "title": title.strip()},
            }
        )
    if not sections and content.strip():
        sections.append({"content": content.strip(), "metadata": {"source": source}})
    return sections


def _local_structural_search(query: str, top_k: int) -> list[dict]:
    """Vectorless fallback: keyword scoring trên cấu trúc section."""
    query_tokens = set(re.findall(r"[\w]+", query.lower(), flags=re.UNICODE))
    sections: list[dict] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        sections.extend(_split_into_sections(content, md_file.name))

    scored = []
    for section in sections:
        text = section["content"].lower()
        tokens = set(re.findall(r"[\w]+", text, flags=re.UNICODE))
        overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
        title_bonus = 0.2 if any(t in section["metadata"].get("title", "").lower() for t in query_tokens) else 0
        score = overlap + title_bonus
        if score > 0:
            scored.append((score, section))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, section in scored[:top_k]:
        results.append(
            {
                "content": section["content"],
                "score": float(score),
                "metadata": section["metadata"],
                "source": "pageindex",
            }
        )
    return results


def _load_doc_ids() -> list[str]:
    if PAGEINDEX_DOC_IDS_PATH.exists():
        return json.loads(PAGEINDEX_DOC_IDS_PATH.read_text(encoding="utf-8"))
    return []


def _save_doc_ids(doc_ids: list[str]) -> None:
    ensure_processed_dir()
    PAGEINDEX_DOC_IDS_PATH.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upload_documents() -> list[str]:
    """Upload markdown documents lên PageIndex (nếu có API key)."""
    if not PAGEINDEX_API_KEY:
        print("⚠ Không có PAGEINDEX_API_KEY — bỏ qua upload cloud.")
        return []

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids: list[str] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        try:
            if hasattr(client, "submit_document"):
                result = client.submit_document(str(md_file))
                doc_id = result.get("doc_id") or result.get("id")
            elif hasattr(client, "documents") and hasattr(client.documents, "create"):
                doc = client.documents.create(
                    file=content.encode("utf-8"),
                    filename=md_file.name,
                    media_type="text/markdown",
                )
                doc_id = getattr(doc, "id", None) or doc.get("id")
            else:
                raise AttributeError("Unsupported PageIndex SDK version")

            if doc_id:
                doc_ids.append(str(doc_id))
                print(f"  ✓ Uploaded: {md_file.name} → {doc_id}")
        except Exception as exc:
            print(f"  ⚠ Upload failed for {md_file.name}: {exc}")

    if doc_ids:
        _save_doc_ids(doc_ids)
    return doc_ids


def _wait_for_ready(client, doc_id: str, timeout: int = 120) -> bool:
    if not hasattr(client, "is_retrieval_ready"):
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.is_retrieval_ready(doc_id):
                return True
        except Exception:
            return True
        time.sleep(3)
    return False


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex hoặc structural fallback.
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            doc_ids = _load_doc_ids()

            if not doc_ids:
                doc_ids = upload_documents()

            if doc_ids and hasattr(client, "chat_completions"):
                for doc_id in doc_ids:
                    if hasattr(client, "is_retrieval_ready"):
                        _wait_for_ready(client, doc_id, timeout=30)
                response = client.chat_completions(
                    messages=[{"role": "user", "content": query}],
                    doc_id=doc_ids[0],
                )
                answer = ""
                if isinstance(response, dict):
                    answer = (
                        response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                if answer:
                    return [
                        {
                            "content": answer,
                            "score": 1.0,
                            "metadata": {"doc_id": doc_ids[0]},
                            "source": "pageindex",
                        }
                    ]

            if doc_ids and hasattr(client, "documents") and hasattr(client.documents, "search"):
                all_results = []
                for doc_id in doc_ids[:3]:
                    hits = client.documents.search(doc_id, query=query)
                    items = hits if isinstance(hits, list) else hits.get("results", [])
                    for hit in items[:top_k]:
                        content = getattr(hit, "text", None) or hit.get("text", "") or str(hit)
                        score = float(getattr(hit, "score", None) or hit.get("score", 0.8))
                        all_results.append(
                            {
                                "content": content,
                                "score": score,
                                "metadata": {"doc_id": doc_id},
                                "source": "pageindex",
                            }
                        )
                if all_results:
                    all_results.sort(key=lambda x: x["score"], reverse=True)
                    return all_results[:top_k]

            if hasattr(client, "collection"):
                col = client.collection("drug_law_docs")
                response = col.query(query, top_k=top_k)
                if response:
                    results = []
                    for item in response:
                        content = getattr(item, "content", None) or str(item)
                        results.append(
                            {
                                "content": content,
                                "score": 0.9,
                                "metadata": {},
                                "source": "pageindex",
                            }
                        )
                    return results[:top_k]
        except Exception:
            pass

    return _local_structural_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env (hoặc dùng local fallback)")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
