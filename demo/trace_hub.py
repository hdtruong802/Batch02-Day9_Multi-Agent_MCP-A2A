"""In-memory hub for demo trace events (SSE subscribers)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceSession:
    trace_id: str
    created_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    done: bool = False
    result: dict[str, Any] | None = None


class TraceHub:
    def __init__(self) -> None:
        self._sessions: dict[str, TraceSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, trace_id: str) -> TraceSession:
        async with self._lock:
            session = TraceSession(trace_id=trace_id)
            self._sessions[trace_id] = session
            return session

    async def get(self, trace_id: str) -> TraceSession | None:
        async with self._lock:
            return self._sessions.get(trace_id)

    async def publish(self, event: dict[str, Any]) -> None:
        trace_id = event.get("trace_id")
        if not trace_id:
            return
        async with self._lock:
            session = self._sessions.get(trace_id)
            if session is None:
                session = TraceSession(trace_id=trace_id)
                self._sessions[trace_id] = session
            stamped = {**event, "ts": time.time()}
            session.events.append(stamped)
            for queue in list(session.subscribers):
                try:
                    queue.put_nowait(stamped)
                except asyncio.QueueFull:
                    pass

    async def complete(self, trace_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            session = self._sessions.get(trace_id)
            if session is None:
                session = TraceSession(trace_id=trace_id)
                self._sessions[trace_id] = session
            session.done = True
            session.result = result
        await self.publish(
            {
                "trace_id": trace_id,
                "agent": "system",
                "status": "completed",
                "detail": "request_finished",
                **result,
            }
        )

    async def subscribe(self, trace_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        async with self._lock:
            session = self._sessions.get(trace_id)
            if session is None:
                session = TraceSession(trace_id=trace_id)
                self._sessions[trace_id] = session
            for past in session.events:
                queue.put_nowait(past)
            session.subscribers.append(queue)
            if session.done and session.result:
                queue.put_nowait(
                    {
                        "trace_id": trace_id,
                        "agent": "system",
                        "status": "completed",
                        "detail": "request_finished",
                        **session.result,
                    }
                )
        return queue

    async def unsubscribe(self, trace_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            session = self._sessions.get(trace_id)
            if session and queue in session.subscribers:
                session.subscribers.remove(queue)
