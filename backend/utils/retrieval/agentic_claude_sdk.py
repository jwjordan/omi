"""
claude-agent-sdk path for the agentic chat loop.

Drives chat through host-side llm-proxy's /v1/agent/chat endpoint instead of
calling anthropic_client.messages.stream() directly. Selected by setting
OMI_CHAT_BACKEND=claude-agent-sdk; otherwise the original Anthropic path runs.

The wire format from llm-proxy:
    data: {"type": "thought", "label": "...", "tool": "..."}
    data: {"type": "text", "text": "..."}
    data: {"type": "done", "text": "...", "duration_ms": int, "num_turns": int, ...}
    data: {"type": "error", "message": "..."}

This is translated back into the AsyncStreamingCallback contract that the
existing chat router already consumes ("data: ..." / "think: ..." chunks).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import AsyncGenerator, List, Optional

import httpx

import database.conversations as conversations_db
from models.app import App
from models.chat import ChatSession, Message, PageContext

logger = logging.getLogger(__name__)


LLM_PROXY_URL = os.environ.get("LLM_PROXY_URL", "http://host.docker.internal:8090").rstrip("/")
AGENT_CHAT_TIMEOUT_S = float(os.environ.get("AGENT_CHAT_TIMEOUT_S", "300"))


def _messages_to_openai(messages: List[Message]) -> list[dict]:
    """Flatten chat history into the OpenAI-style array llm-proxy expects."""
    out: list[dict] = []
    for m in messages:
        role = "user" if m.sender == "human" else "assistant"
        out.append({"role": role, "content": m.text})
    return out


async def execute_agentic_claude_sdk_stream(
    uid: str,
    messages: List[Message],
    app: Optional[App],
    callback_data: Optional[dict],
    chat_session: Optional[ChatSession],
    context: Optional[PageContext],
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    """Stream chat via llm-proxy /v1/agent/chat. Yields the same "data: ..." /
    "think: ..." chunks the chat router already consumes, and populates
    callback_data with answer / ask_for_nps / langsmith_run_id like the
    Anthropic path does."""

    payload = {
        "uid": uid,
        "system_prompt": system_prompt,
        "messages": _messages_to_openai(messages),
    }

    langsmith_run_id = str(uuid.uuid4())
    if callback_data is not None:
        callback_data["langsmith_run_id"] = langsmith_run_id

    answer_parts: list[str] = []
    tool_usage_count = 0
    final_text: Optional[str] = None
    error_msg: Optional[str] = None
    conversation_ids: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=AGENT_CHAT_TIMEOUT_S) as client:
            async with client.stream(
                "POST",
                f"{LLM_PROXY_URL}/v1/agent/chat",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"llm-proxy /v1/agent/chat HTTP {response.status_code}: {body!r}")

                # SSE frames are separated by blank lines; httpx aiter_lines gives
                # us one logical line at a time. We only care about `data: ` lines.
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):]
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("agent-sdk: skipping non-JSON SSE line: %r", raw[:200])
                        continue

                    et = event.get("type")
                    if et == "text":
                        chunk = event.get("text", "")
                        if chunk:
                            answer_parts.append(chunk)
                            yield f"data: {chunk}"
                    elif et == "thought":
                        label = event.get("label") or event.get("tool") or "Thinking"
                        tool_usage_count += 1
                        yield f"think: {label}"
                    elif et == "done":
                        final_text = event.get("text") or "".join(answer_parts)
                        ids = event.get("conversation_ids") or []
                        if isinstance(ids, list):
                            conversation_ids = [c for c in ids if isinstance(c, str)]
                        logger.info(
                            "agent-sdk done uid=%s duration_ms=%s turns=%s cost_usd=%s citations=%d",
                            uid,
                            event.get("duration_ms"),
                            event.get("num_turns"),
                            event.get("total_cost_usd"),
                            len(conversation_ids),
                        )
                        break
                    elif et == "error":
                        error_msg = event.get("message", "agent error")
                        logger.error("agent-sdk error uid=%s: %s", uid, error_msg)
                        break
                    else:
                        logger.warning("agent-sdk: unknown event type %r", et)

    except Exception as e:  # noqa: BLE001
        logger.exception("agent-sdk stream failed uid=%s", uid)
        error_msg = error_msg or f"agent-sdk transport failed: {e}"

    # Populate callback_data using the same shape the Anthropic path produces
    # so chat.py:process_message doesn't care which backend we used.
    if callback_data is not None:
        callback_data["answer"] = final_text if final_text is not None else "".join(answer_parts)
        callback_data["memories_found"] = _build_citations(uid, conversation_ids)
        callback_data["ask_for_nps"] = tool_usage_count > 0
        if error_msg:
            callback_data["error"] = error_msg

    yield None  # signal completion to the chat router


def _build_citations(uid: str, conversation_ids: list[str]) -> list[dict]:
    """Look up the conversations the agent touched and shape them as
    MessageConversation dicts ({id, structured:{title,emoji}, created_at})
    for the chat router. Returns [] on any error so chat never fails over
    citation rendering."""
    if not conversation_ids:
        return []
    try:
        # Cap at the same threshold the chat router applies (it slices to 5).
        # Fetching more is wasted DB work.
        ids = list(dict.fromkeys(conversation_ids))[:25]
        raw = conversations_db.get_conversations_by_id(uid, ids)
    except Exception as e:  # noqa: BLE001
        logger.warning("agent-sdk citation lookup failed uid=%s: %s", uid, e)
        return []

    citations: list[dict] = []
    for conv in raw:
        structured = conv.get("structured") or {}
        citations.append(
            {
                "id": conv.get("id"),
                "structured": {
                    "title": structured.get("title") or "",
                    "emoji": structured.get("emoji") or "",
                },
                "created_at": conv.get("created_at") or conv.get("started_at"),
            }
        )
    # Preserve the model's order (most relevant first per vector search) so
    # the UI shows the most-referenced conversation at the top.
    by_id = {c["id"]: c for c in citations}
    ordered = [by_id[i] for i in ids if i in by_id]
    return ordered
