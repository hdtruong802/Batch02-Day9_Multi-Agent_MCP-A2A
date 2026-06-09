"""Demo UI server — chat + live agent graph + latency."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from a2a.client import A2AClient
from a2a.types import AgentCard, Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart

from common.a2a_client import _extract_text
from demo.trace_hub import TraceHub

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [demo] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")
A2A_HTTP_TIMEOUT = float(os.getenv("A2A_HTTP_TIMEOUT", "1800"))
DEMO_PORT = int(os.getenv("DEMO_PORT", "10200"))

STATIC_DIR = Path(__file__).parent / "static"
hub = TraceHub()
app = FastAPI(title="Legal Multi-Agent Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str


class TraceEvent(BaseModel):
    trace_id: str
    agent: str
    status: str
    detail: str = ""


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    customer_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{CUSTOMER_AGENT_URL}/.well-known/agent.json")
            customer_ok = resp.status_code == 200
    except Exception:
        pass
    return {
        "demo": True,
        "customer_agent": customer_ok,
        "customer_url": CUSTOMER_AGENT_URL,
    }


@app.post("/internal/trace")
async def receive_trace(event: TraceEvent) -> dict:
    await hub.publish(event.model_dump())
    return {"ok": True}


@app.get("/api/trace/{trace_id}/stream")
async def trace_stream(trace_id: str) -> StreamingResponse:
    queue = await hub.subscribe(trace_id)

    async def generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("agent") == "system" and event.get("status") == "completed":
                    break
        finally:
            await hub.unsubscribe(trace_id, queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _call_customer_agent(question: str, trace_id: str) -> tuple[str, float]:
    http_timeout = httpx.Timeout(A2A_HTTP_TIMEOUT, connect=30.0)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=http_timeout) as http_client:
        card_resp = await http_client.get(f"{CUSTOMER_AGENT_URL}/.well-known/agent.json")
        card_resp.raise_for_status()
        agent_card = AgentCard.model_validate(card_resp.json())
        client = A2AClient(httpx_client=http_client, agent_card=agent_card)

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
            message_id=str(uuid4()),
            metadata={
                "trace_id": trace_id,
                "delegation_depth": 0,
            },
        )
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )
        response = await client.send_message(request, http_kwargs={"timeout": http_timeout})
        latency = time.perf_counter() - started
        return _extract_text(response), latency


async def _run_chat(question: str, trace_id: str) -> None:
    try:
        await hub.publish(
            {
                "trace_id": trace_id,
                "agent": "user",
                "status": "started",
                "detail": "question_sent",
            }
        )
        answer, latency = await _call_customer_agent(question, trace_id)
        result = {
            "answer": answer or "No response received.",
            "latency_s": round(latency, 2),
            "trace_id": trace_id,
        }
        await hub.complete(trace_id, result)
    except Exception as exc:
        logger.exception("chat failed trace=%s", trace_id)
        await hub.publish(
            {
                "trace_id": trace_id,
                "agent": "system",
                "status": "error",
                "detail": str(exc),
            }
        )


@app.post("/api/chat")
async def chat(body: ChatRequest) -> dict:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    trace_id = str(uuid4())
    await hub.create(trace_id)
    asyncio.create_task(_run_chat(question, trace_id))
    return {"trace_id": trace_id, "status": "processing"}


def main() -> None:
    logger.info("Demo UI at http://localhost:%d", DEMO_PORT)
    logger.info("Set DEMO_TRACE_URL=http://localhost:%d in .env for live agent graph", DEMO_PORT)
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="info")


if __name__ == "__main__":
    main()
