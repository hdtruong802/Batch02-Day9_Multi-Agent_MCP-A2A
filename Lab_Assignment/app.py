"""
RAG Chat UI — Vietnamese Drug Law Q&A (Supervisor-Workers)
Run: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.agents.pipeline import run_supervisor_pipeline

st.set_page_config(
    page_title="RAG Chat · Luật Ma Túy",
    page_icon="⚖️",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations = [{"id": 0, "title": "Cuộc trò chuyện 1", "messages": []}]
    st.session_state.active_id = 0
    st.session_state.next_id = 1


def active_conv():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_id:
            return c
    return st.session_state.conversations[0]


def new_conversation():
    cid = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.conversations.append({
        "id": cid,
        "title": f"Cuộc trò chuyện {cid + 1}",
        "messages": [],
    })
    st.session_state.active_id = cid


def delete_conversation(cid):
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != cid]
    if not st.session_state.conversations:
        new_conversation()
    elif st.session_state.active_id == cid:
        st.session_state.active_id = st.session_state.conversations[0]["id"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ RAG Chat")
    st.caption("Supervisor-Workers: Legal · News · Answer")

    if st.button("+ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()

    st.divider()

    st.markdown("**Lịch sử**")
    for conv in reversed(st.session_state.conversations):
        is_active = conv["id"] == st.session_state.active_id
        col_title, col_del = st.columns([5, 1])
        label = ("▶ " if is_active else "") + conv["title"]
        if col_title.button(label, key=f"conv_{conv['id']}", use_container_width=True,
                            type="secondary" if not is_active else "primary"):
            st.session_state.active_id = conv["id"]
            st.rerun()
        if col_del.button("X", key=f"del_{conv['id']}", help="Xoá"):
            delete_conversation(conv["id"])
            st.rerun()

    st.divider()

    with st.expander("Tuỳ chọn"):
        top_k = st.slider("top_k", min_value=1, max_value=10, value=5,
                          help="Số chunks tìm kiếm mỗi worker")
        show_sources = st.toggle("Hiển thị nguồn", value=True)
        show_agents = st.toggle("Hiển thị agent trace", value=True)

    st.caption("Luật PCMT · NĐ · Tin tức nghệ sĩ")

# ── Active conversation ───────────────────────────────────────────────────────
conv = active_conv()
messages = conv["messages"]

st.title("RAG Chat · Luật Ma Túy Việt Nam")
st.caption(
    "Multi-agent: Supervisor điều phối Legal Worker, News Worker và Answer Worker — "
    "mọi câu trả lời có trích dẫn nguồn."
)


def render_agent_trace(result: dict):
    with st.expander("Agent trace (Supervisor-Workers)"):
        st.markdown(f"**Supervisor:** {result.get('supervisor_plan', '')}")
        st.markdown(f"**Workers:** {', '.join(result.get('workers_used', []))}")
        for step in result.get("worker_trace", []):
            st.markdown(
                f"- `{step['worker_id']}` **{step['worker_name']}** — "
                f"{step['summary']} ({step['chunk_count']} chunks)"
            )


def render_sources(sources, retrieval_source="hybrid"):
    with st.expander(f"Nguồn tài liệu ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            score = src.get("score", 0.0)
            worker = src.get("worker", meta.get("type", "unknown"))
            st.markdown(
                f"**{i}. {meta.get('source', f'Source {i}')}** · "
                f"`{worker}` · score `{score:.4f}` · via `{retrieval_source}`"
            )
            preview = src["content"][:300]
            st.markdown(f"> {preview}{'…' if len(src['content']) > 300 else ''}")
            if i < len(sources):
                st.divider()


for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if show_agents and msg.get("worker_trace"):
                render_agent_trace(msg)
            if show_sources and msg.get("sources"):
                render_sources(msg["sources"], msg.get("retrieval_source", "hybrid"))

if prompt := st.chat_input("Nhập câu hỏi về luật ma túy Việt Nam..."):
    messages.append({"role": "user", "content": prompt})

    if conv["title"].startswith("Cuộc trò chuyện"):
        conv["title"] = prompt[:40] + ("…" if len(prompt) > 40 else "")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Supervisor đang phân công workers..."):
            result = run_supervisor_pipeline(prompt, top_k=top_k)

        answer = result["answer"]
        sources = result["sources"]
        retrieval_source = result.get("retrieval_source", "hybrid")

        st.markdown(answer)
        if show_agents:
            render_agent_trace(result)
        if show_sources and sources:
            render_sources(sources, retrieval_source)

    messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "supervisor_plan": result.get("supervisor_plan", ""),
        "workers_used": result.get("workers_used", []),
        "worker_trace": result.get("worker_trace", []),
    })
