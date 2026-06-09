"""Optional trace events for the demo UI (fire-and-forget HTTP posts)."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_trace_id: ContextVar[str | None] = ContextVar("demo_trace_id", default=None)

AGENT_PORTS: dict[str, str] = {
    "10000": "registry",
    "10100": "customer",
    "10101": "law",
    "10102": "tax",
    "10103": "compliance",
}


def set_trace_id(trace_id: str | None) -> None:
    _trace_id.set(trace_id)


def get_trace_id() -> str | None:
    return _trace_id.get()


def agent_from_endpoint(endpoint: str) -> str:
    for port, name in AGENT_PORTS.items():
        if f":{port}" in endpoint:
            return name
    return "agent"


async def emit(
    agent: str,
    status: str,
    *,
    detail: str = "",
    trace_id: str | None = None,
) -> None:
    """Post a trace event to the demo server if DEMO_TRACE_URL is configured."""
    url = os.getenv("DEMO_TRACE_URL", "").rstrip("/")
    tid = trace_id or get_trace_id()
    if not url or not tid:
        return

    payload: dict[str, Any] = {
        "trace_id": tid,
        "agent": agent,
        "status": status,
        "detail": detail,
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{url}/internal/trace", json=payload)
    except Exception as exc:
        logger.debug("demo trace emit failed: %s", exc)
