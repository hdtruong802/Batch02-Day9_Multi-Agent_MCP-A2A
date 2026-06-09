"""Shared LLM factory for all agents.

Supports:
  - gemini (default): Google Gemini API via langchain-google-genai
  - openrouter: OpenAI-compatible API via OpenRouter
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def get_llm() -> BaseChatModel:
    """Return a chat model based on LLM_PROVIDER in .env."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    if provider == "openrouter":
        return _get_openrouter_llm()
    if provider == "gemini":
        return _get_gemini_llm()
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'gemini' or 'openrouter'."
    )


def _get_gemini_llm() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key in ("your_key_here", "your_google_api_key_here"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Get a key at https://aistudio.google.com/apikey "
            "and set it in .env"
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        google_api_key=api_key,
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
    )


def _get_openrouter_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Copy .env.example to .env and set your key."
        )

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        openai_api_key=api_key,
        openai_api_base=os.getenv(
            "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
        ),
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.3")),
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "legal-multiagent"),
        },
    )
