"""Tax Agent — AgentExecutor bridge between A2A SDK and LangGraph."""

from __future__ import annotations

import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart

from tax_agent.graph import create_graph

logger = logging.getLogger(__name__)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph


class TaxAgentExecutor(AgentExecutor):
    """Bridges A2A RequestContext to the Tax LangGraph agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract question from message parts
        question = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        task_id = context.task_id or str(uuid4())
        metadata = context.message.metadata or {} if context.message else {}
        trace_id = metadata.get("trace_id", str(uuid4()))
        depth = int(metadata.get("delegation_depth", 0))

        from common.demo_trace import emit, set_trace_id

        set_trace_id(trace_id)
        logger.info(
            "TaxAgent executing | task=%s context=%s trace=%s depth=%d",
            task_id, context_id, trace_id, depth,
        )
        await emit("tax", "started", detail="execute", trace_id=trace_id)

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.start_work()

        try:
            result = await _get_graph().ainvoke(
                {"question": question, "answer": ""},
                config={"configurable": {"thread_id": context_id}},
            )

            answer = result.get("answer", "") or "I was unable to generate a tax analysis at this time."

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=answer))],
                name="tax_analysis",
            )
            await updater.complete()
            await emit("tax", "completed", detail="execute", trace_id=trace_id)

        except Exception as exc:
            logger.exception("TaxAgent execution error: %s", exc)
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Tax analysis failed: {exc}"))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    @staticmethod
    def _extract_question(context: RequestContext) -> str:
        if context.message and context.message.parts:
            parts = []
            for part in context.message.parts:
                inner = getattr(part, "root", part)
                text = getattr(inner, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""