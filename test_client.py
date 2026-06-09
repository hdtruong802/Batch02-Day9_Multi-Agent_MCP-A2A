"""End-to-end test client for the Legal Multi-Agent System.

Sends a legal question to the Customer Agent and prints the response.
"""

import asyncio
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")
# Stage 5 chains 6-8 LLM calls; free models often need >5 minutes end-to-end.
A2A_HTTP_TIMEOUT = float(os.getenv("A2A_HTTP_TIMEOUT", "1800"))

QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


async def main() -> None:
    print(f"Connecting to Customer Agent at {CUSTOMER_AGENT_URL}")
    print(f"Question: {QUESTION}")
    print("-" * 60)

    http_timeout = httpx.Timeout(A2A_HTTP_TIMEOUT, connect=30.0)
    async with httpx.AsyncClient(timeout=http_timeout) as http_client:
        # Resolve agent card
        card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
        try:
            card_resp = await http_client.get(card_url)
            card_resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: Could not reach Customer Agent at {card_url}")
            print(f"  {e}")
            print("Make sure all services are running:")
            print("  Windows: .\\start_all.ps1")
            print("  Linux/Mac: ./start_all.sh")
            sys.exit(1)

        from a2a.types import AgentCard, Message, Part, Role, TextPart, MessageSendParams
        from a2a.client import A2AClient
        from uuid import uuid4

        agent_card = AgentCard.model_validate(card_resp.json())
        print(f"Connected to agent: {agent_card.name} v{agent_card.version}")
        print("-" * 60)

        # Build the legacy A2AClient
        client = A2AClient(httpx_client=http_client, agent_card=agent_card)

        # Construct the message
        from a2a.types import SendMessageRequest, MessageSendParams as MSP
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=QUESTION))],
            message_id=str(uuid4()),
        )
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MSP(message=message),
        )

        print(
            f"Sending request (may take 3-15 min with free models, "
            f"timeout={A2A_HTTP_TIMEOUT:.0f}s)..."
        )
        print("Tip: grep trace_id in service logs to follow the request flow.\n")
        started_at = time.perf_counter()
        try:
            response = await client.send_message(
                request,
                http_kwargs={"timeout": http_timeout},
            )
        except Exception as e:
            if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower():
                elapsed = time.perf_counter() - started_at
                print(f"\nERROR: Request timed out after {elapsed:.0f}s.")
                print("The agent chain is still slow (many LLM calls). Try:")
                print("  1. Increase A2A_HTTP_TIMEOUT in .env (e.g. 1200)")
                print("  2. Use a faster model in GEMINI_MODEL (or OPENROUTER_MODEL)")
                print("  3. Ensure all 5 services are running (.\\start_all.ps1)")
            raise
        latency_s = time.perf_counter() - started_at

        # Parse response
        result_text = ""
        if hasattr(response, "root"):
            root = response.root
            if hasattr(root, "result"):
                result = root.result
                # Task with artifacts
                if hasattr(result, "artifacts") and result.artifacts:
                    for artifact in result.artifacts:
                        for part in artifact.parts:
                            p = part.root if hasattr(part, "root") else part
                            if hasattr(p, "text"):
                                result_text += p.text
                # Message with parts
                elif hasattr(result, "parts") and result.parts:
                    for part in result.parts:
                        p = part.root if hasattr(part, "root") else part
                        if hasattr(p, "text"):
                            result_text += p.text

        print(f"Latency: {latency_s:.2f}s")
        print("-" * 60)

        if result_text:
            print("RESPONSE:")
            print("=" * 60)
            print(result_text)
            print("=" * 60)
        else:
            print("No text response received. Raw response:")
            print(response)


if __name__ == "__main__":
    asyncio.run(main())