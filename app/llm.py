"""Generation backend: Google Gemini, wrapped as a LangChain chat model."""
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import settings

ChatMessage = dict[str, Any]  # {"role": "system"|"user"|"assistant"|"tool", "content": "...", ...}

_ROLE_TO_MESSAGE_CLS = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def _to_lc_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    return [_ROLE_TO_MESSAGE_CLS[m["role"]](content=m["content"]) for m in messages]


def _extract_text(content: str | list) -> str:
    """Newer langchain_google_genai versions return `.content` as a list of
    content blocks (e.g. [{"type": "text", "text": "..."}]) instead of a
    plain string; normalize both shapes to a single string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def generate(messages: list[ChatMessage]) -> str:
    return _invoke(_get_chat_model(), messages)


def generate_stream(messages: list[ChatMessage]):
    """Yield the answer incrementally as text deltas, as Gemini generates them.

    Mirrors generate() but uses the chat model's streaming interface so the API
    can push tokens to the client instead of waiting for the full response."""
    for chunk in _get_chat_model().stream(_to_lc_messages(messages)):
        text = _extract_text(chunk.content)
        if text:
            yield text


def _invoke(chat_model, messages: list[ChatMessage]) -> str:
    response = chat_model.invoke(_to_lc_messages(messages))
    return _extract_text(response.content).strip()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMToolResponse:
    content: str | None
    tool_calls: list[ToolCall]


def generate_with_tools(messages: list[ChatMessage], tools: list[dict]) -> LLMToolResponse:
    model = get_tool_chat_model()
    if tools:
        model = model.bind_tools(tools)
    response = model.invoke(_to_lc_messages(messages))
    tool_calls = [
        ToolCall(id=tc["id"], name=tc["name"], arguments=tc["args"] or {})
        for tc in response.tool_calls or []
    ]
    return LLMToolResponse(content=_extract_text(response.content) or None, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_chat_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model_name,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_new_tokens,
        # Gemini 3+ models spend part of max_output_tokens on internal
        # "thinking" tokens before the visible answer; on a small budget this
        # can eat the whole response and leave it truncated. RAG QA doesn't
        # need deep reasoning, so keep thinking minimal to leave the budget
        # for the actual answer (ignored by models that don't support it).
        thinking_level="minimal",
    )


@lru_cache(maxsize=1)
def get_tool_chat_model():
    """Entry point for graph_agent.py: the tool-capable chat model to drive
    the LangGraph agent<->tools loop."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.tool_model_name,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_new_tokens,
        thinking_level="minimal",
    )
