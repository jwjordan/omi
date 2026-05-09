"""
Pre-hosted MCP Server via Streamable HTTP Transport

This module provides a streamable HTTP transport for MCP (Model Context Protocol),
allowing clients to connect without running a local MCP server.

Implements the MCP 2025-03-26 Streamable HTTP Transport specification.
"""

import asyncio
import hmac
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Union, List, Any

from fastapi import APIRouter, HTTPException, Header, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from utils.other.endpoints import check_rate_limit_inline

import database.memories as memories_db
import database.conversations as conversations_db
import database.mcp_api_key as mcp_api_key_db
import database.users as users_db
import database.vector_db as vector_db
from models.memories import MemoryDB, Memory, MemoryCategory
from models.other import Person
from utils.conversations.factory import deserialize_conversation
from utils.conversations.render import redact_conversation_for_list, conversations_to_string
from models.conversation_enums import CategoryEnum
from utils.llm.memories import identify_category_for_memory

router = APIRouter()

# Store active sessions
active_sessions: dict = {}


class MCPSession:
    """Represents an active MCP session."""

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.initialized = False


def authenticate_api_key(authorization: Optional[str]) -> Optional[str]:
    """Validate API key from Authorization header and return user_id if valid."""
    if not authorization:
        return None

    # Support both "Bearer <key>" and just "<key>" formats
    token = authorization
    if authorization.startswith("Bearer "):
        token = authorization[7:]

    if not token.startswith("omi_mcp_"):
        return None

    return mcp_api_key_db.get_user_id_by_api_key(token)


def authenticate_local_trust(local_trust: Optional[str], uid: Optional[str]) -> Optional[str]:
    """Pendant-stack local-trust auth: shared-secret header lets a trusted host-side
    process (llm-proxy on host.docker.internal) act as any uid. Bound to the
    127.0.0.1:18080 docker port mapping; never expose this header on the public
    api.omi.me endpoint.

    Returns uid if (a) MCP_LOCAL_TRUST_TOKEN is configured, (b) the supplied token
    matches it via constant-time compare, and (c) a uid header was supplied.
    Returns None otherwise — caller should fall back to other auth schemes.
    """
    expected = os.environ.get("MCP_LOCAL_TRUST_TOKEN", "").strip()
    if not expected or not local_trust or not uid:
        return None
    if not hmac.compare_digest(expected, local_trust.strip()):
        return None
    return uid


# MCP Tool Definitions
MCP_TOOLS = [
    {
        "name": "get_memories",
        "description": "Retrieve a list of memories. A memory is a known fact about the user across multiple domains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": [c.value for c in MemoryCategory]},
                    "description": "Categories to filter by",
                    "default": [],
                },
                "limit": {"type": "integer", "description": "Number of memories to retrieve", "default": 100},
                "offset": {"type": "integer", "description": "Offset for pagination", "default": 0},
            },
        },
    },
    {
        "name": "create_memory",
        "description": "Create a new memory. A memory is a known fact about the user across multiple domains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content of the memory"},
                "category": {
                    "type": "string",
                    "enum": [c.value for c in MemoryCategory],
                    "description": "The category of the memory",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "delete_memory",
        "description": "Delete a memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string", "description": "The ID of the memory to delete"}},
            "required": ["memory_id"],
        },
    },
    {
        "name": "edit_memory",
        "description": "Edit a memory's content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The ID of the memory to edit"},
                "content": {"type": "string", "description": "The new content for the memory"},
            },
            "required": ["memory_id", "content"],
        },
    },
    {
        "name": "get_conversations",
        "description": "Retrieve a list of conversation metadata. To get full transcripts, use get_conversation_by_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Filter after this date (yyyy-mm-dd)"},
                "end_date": {"type": "string", "description": "Filter before this date (yyyy-mm-dd)"},
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": [c.value for c in CategoryEnum]},
                    "description": "Categories to filter by",
                    "default": [],
                },
                "limit": {"type": "integer", "description": "Number of conversations to retrieve", "default": 20},
                "offset": {"type": "integer", "description": "Offset for pagination", "default": 0},
            },
        },
    },
    {
        "name": "get_conversation_by_id",
        "description": "Retrieve a conversation by ID including each segment of the transcript.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string", "description": "The ID of the conversation to retrieve"}
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "search_memories",
        "description": "Semantic search across the user's memories. Returns memories ranked by relevance to the query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_conversations",
        "description": "Semantic search across the user's conversations. Returns conversations ranked by relevance to the query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "start_date": {"type": "string", "description": "Filter after this date (yyyy-mm-dd)"},
                "end_date": {"type": "string", "description": "Filter before this date (yyyy-mm-dd)"},
                "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "deep_search_conversations",
        "description": (
            "Semantic vector search across the user's conversations that returns full "
            "structured summaries AND inline transcript segments with speaker attribution. "
            "Use this for content-level questions like 'what did Eric ask me to do', 'what "
            "was decided in the Tuesday meeting', or 'find when I discussed X' — anything "
            "where you need the actual words spoken, not just titles. One round-trip; "
            "prefer this over chaining search_conversations + get_conversation_by_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what to find."},
                "start_date": {
                    "type": "string",
                    "description": (
                        "Filter after this date. Accepts YYYY-MM-DD (interpreted as UTC midnight) "
                        "or ISO 8601 with timezone (e.g. '2026-04-28T00:00:00-04:00')."
                    ),
                },
                "end_date": {
                    "type": "string",
                    "description": (
                        "Filter before this date. Accepts YYYY-MM-DD (interpreted as UTC end-of-day) "
                        "or ISO 8601 with timezone."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max conversations returned (default 8, hard cap 20).",
                    "default": 8,
                },
                "max_segments_per_conversation": {
                    "type": "integer",
                    "description": (
                        "Cap transcript segments per conversation to avoid context overflow. "
                        "Default 40; raise to 100 only when you need exhaustive detail; never above 200."
                    ),
                    "default": 40,
                },
            },
            "required": ["query"],
        },
    },
]


def _parse_flexible_date(value: str, *, end_of_day: bool) -> int:
    """Parse YYYY-MM-DD or ISO 8601 datetime to a UTC epoch int.

    YYYY-MM-DD with no time component is treated as UTC midnight (or 23:59:59
    for end_of_day). Anything with a time component is parsed via fromisoformat
    and converted to UTC.
    """
    s = value.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.timestamp())
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails."""

    def __init__(self, message: str, code: int = -32000):
        self.message = message
        self.code = code
        super().__init__(self.message)


def execute_tool(user_id: str, tool_name: str, arguments: dict) -> dict:
    """Execute an MCP tool and return the result. Raises ToolExecutionError on failure."""

    if tool_name == "get_memories":
        categories = arguments.get("categories", [])
        limit = arguments.get("limit", 100)
        offset = arguments.get("offset", 0)

        # Validate categories
        valid_categories = []
        for cat in categories:
            try:
                valid_categories.append(MemoryCategory(cat).value)
            except ValueError:
                raise ToolExecutionError(f"Invalid memory category: '{cat}'", code=-32602)

        memories = memories_db.get_memories(user_id, limit, offset, valid_categories)
        # Apply locked content truncation
        for memory in memories:
            if memory.get('is_locked', False):
                content = memory.get('content', '')
                memory['content'] = (content[:70] + '...') if len(content) > 70 else content

        return {"memories": memories}

    elif tool_name == "create_memory":
        content = arguments.get("content")
        if not content:
            raise ToolExecutionError("Content is required")

        # Auto-categorize memories from MCP clients
        category = identify_category_for_memory(content)
        memory = Memory(content=content, category=category)
        memory_db = MemoryDB.from_memory(memory, user_id, None, True)
        memories_db.create_memory(user_id, memory_db.model_dump())

        return {"success": True, "memory": memory_db.model_dump()}

    elif tool_name == "delete_memory":
        memory_id = arguments.get("memory_id")
        if not memory_id:
            raise ToolExecutionError("memory_id is required")

        memory = memories_db.get_memory(user_id, memory_id)
        if not memory:
            raise ToolExecutionError("Memory not found", code=-32001)
        if memory.get('is_locked', False):
            raise ToolExecutionError("A paid plan is required to access this memory.", code=-32002)

        memories_db.delete_memory(user_id, memory_id)
        return {"success": True}

    elif tool_name == "edit_memory":
        memory_id = arguments.get("memory_id")
        content = arguments.get("content")
        if not memory_id or not content:
            raise ToolExecutionError("memory_id and content are required")

        memory = memories_db.get_memory(user_id, memory_id)
        if not memory:
            raise ToolExecutionError("Memory not found", code=-32001)
        if memory.get('is_locked', False):
            raise ToolExecutionError("A paid plan is required to access this memory.", code=-32002)

        memories_db.edit_memory(user_id, memory_id, content)
        return {"success": True}

    elif tool_name == "get_conversations":
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")
        categories = arguments.get("categories", [])
        limit = arguments.get("limit", 20)
        offset = arguments.get("offset", 0)

        # Parse dates. start_date is interpreted as midnight UTC of that day,
        # end_date as 23:59:59 UTC of that day so a same-day window (e.g.
        # start_date == end_date == "2026-05-05") matches everything that
        # happened on that calendar day in UTC. Without the end-of-day shift,
        # end_date acted as midnight and same-day queries always returned [].
        start_dt = None
        end_dt = None
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ToolExecutionError(
                    f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.", code=-32602
                )
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
            except ValueError:
                raise ToolExecutionError(f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.", code=-32602)

        # Validate categories
        valid_categories = []
        for cat in categories:
            try:
                valid_categories.append(CategoryEnum(cat).value)
            except ValueError:
                pass

        conversations = conversations_db.get_conversations(
            user_id,
            limit,
            offset,
            include_discarded=False,
            statuses=["completed"],
            start_date=start_dt,
            end_date=end_dt,
            categories=valid_categories,
        )

        # Simplify conversation data
        simple_conversations = []
        for conv in conversations:
            redact_conversation_for_list(conv)
            simple_conversations.append(
                {
                    "id": conv.get("id"),
                    "started_at": conv.get("started_at"),
                    "finished_at": conv.get("finished_at"),
                    "structured": conv.get("structured"),
                    "language": conv.get("language"),
                }
            )

        return {"conversations": simple_conversations}

    elif tool_name == "get_conversation_by_id":
        conversation_id = arguments.get("conversation_id")
        if not conversation_id:
            raise ToolExecutionError("conversation_id is required")

        conversation = conversations_db.get_conversation(user_id, conversation_id)
        if not conversation:
            raise ToolExecutionError("Conversation not found", code=-32001)

        if conversation.get('is_locked', False):
            raise ToolExecutionError("A paid plan is required to access this conversation.", code=-32002)

        return {"conversation": conversation}

    elif tool_name == "search_memories":
        query = arguments.get("query")
        if not query:
            raise ToolExecutionError("query is required")

        limit = arguments.get("limit", 10)

        matches = vector_db.find_similar_memories(user_id, query, threshold=0.0, limit=limit)
        if not matches:
            return {"memories": []}

        memory_ids = [m['memory_id'] for m in matches]
        memories = memories_db.get_memories_by_ids(user_id, memory_ids)

        # Build score lookup and filter locked
        score_map = {m['memory_id']: m.get('score', 0) for m in matches}
        results = []
        for mem in memories:
            if mem.get('is_locked', False):
                content = mem.get('content', '')
                mem['content'] = (content[:70] + '...') if len(content) > 70 else content
            mem['relevance_score'] = round(score_map.get(mem.get('id'), 0), 4)
            results.append(mem)

        # Sort by relevance
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return {"memories": results}

    elif tool_name == "search_conversations":
        query = arguments.get("query")
        if not query:
            raise ToolExecutionError("query is required")

        limit = arguments.get("limit", 10)
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")

        # Parse dates to epoch for vector search
        starts_at = None
        ends_at = None
        if start_date:
            try:
                starts_at = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            except ValueError:
                raise ToolExecutionError(
                    f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.", code=-32602
                )
        if end_date:
            try:
                ends_at = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            except ValueError:
                raise ToolExecutionError(f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.", code=-32602)

        conversation_ids = vector_db.query_vectors(query, user_id, starts_at=starts_at, ends_at=ends_at, k=limit)
        if not conversation_ids:
            return {"conversations": []}

        conversations = conversations_db.get_conversations_by_id(user_id, conversation_ids)

        # Simplify and handle locked content
        results = []
        for conv in conversations:
            structured = conv.get("structured")
            if conv.get("is_locked", False) and structured:
                structured = dict(structured)
                structured['action_items'] = []
                structured['events'] = []
            results.append(
                {
                    "id": conv.get("id"),
                    "started_at": conv.get("started_at"),
                    "finished_at": conv.get("finished_at"),
                    "structured": structured,
                    "language": conv.get("language"),
                }
            )

        return {"conversations": results}

    elif tool_name == "deep_search_conversations":
        query = arguments.get("query")
        if not query:
            raise ToolExecutionError("query is required")

        limit = min(int(arguments.get("limit", 8) or 8), 20)
        max_segments = min(int(arguments.get("max_segments_per_conversation", 40) or 40), 200)
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")

        starts_at = ends_at = None
        try:
            if start_date:
                starts_at = _parse_flexible_date(start_date, end_of_day=False)
            if end_date:
                ends_at = _parse_flexible_date(end_date, end_of_day=True)
        except ValueError as e:
            raise ToolExecutionError(
                f"Invalid date format: {e}. Use YYYY-MM-DD or ISO 8601 with timezone.", code=-32602
            )

        conversation_ids = vector_db.query_vectors(
            query, user_id, starts_at=starts_at, ends_at=ends_at, k=limit
        )
        if not conversation_ids:
            return {
                "summary": f"No conversations matched '{query}' in the requested window.",
                "conversations": [],
            }

        raw = conversations_db.get_conversations_by_id(user_id, conversation_ids)
        raw = [c for c in raw if not c.get("is_locked", False)]

        # Resolve speaker names so the rendered transcript is readable.
        person_ids = set()
        for conv in raw:
            for seg in conv.get("transcript_segments") or []:
                pid = seg.get("person_id")
                if pid:
                    person_ids.add(pid)
        people: List[Person] = []
        if person_ids:
            try:
                people = [Person(**p) for p in users_db.get_people_by_ids(user_id, list(person_ids))]
            except Exception as e:
                logging.warning(f"deep_search_conversations: people fetch failed: {e}")

        rendered: List[Any] = []
        for conv_data in raw:
            try:
                conv = deserialize_conversation(conv_data)
                if max_segments != -1 and conv.transcript_segments and len(conv.transcript_segments) > max_segments:
                    conv.transcript_segments = conv.transcript_segments[:max_segments]
                rendered.append(conv)
            except Exception as e:
                logging.warning(f"deep_search_conversations: deserialize failed for {conv_data.get('id')}: {e}")

        body = conversations_to_string(rendered, use_transcript=True, include_timestamps=True, people=people)

        return {
            "summary": f"Found {len(rendered)} conversations semantically matching '{query}'.",
            "result": body,
            "conversation_ids": [c.id for c in rendered],
        }

    else:
        raise ToolExecutionError(f"Unknown tool: {tool_name}", code=-32601)


def create_mcp_response(id: Any, result: dict) -> dict:
    """Create a JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": id, "result": result}


def create_mcp_error(id: Any, code: int, message: str) -> dict:
    """Create a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def handle_mcp_message(
    user_id: str, message: dict, session: Optional[MCPSession] = None
) -> tuple[Optional[dict], Optional[str]]:
    """
    Process an incoming MCP JSON-RPC message and return a response.
    Returns (response, new_session_id) tuple.
    """
    msg_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})
    new_session_id = None

    if method == "initialize":
        # Create a new session
        session_id = str(uuid.uuid4())
        new_session = MCPSession(session_id, user_id)
        new_session.initialized = True
        active_sessions[session_id] = new_session
        new_session_id = session_id

        return (
            create_mcp_response(
                msg_id,
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "omi-mcp-server", "version": "1.0.0"},
                },
            ),
            new_session_id,
        )

    elif method == "notifications/initialized":
        # This is a notification, no response needed
        return None, None

    elif method == "tools/list":
        return create_mcp_response(msg_id, {"tools": MCP_TOOLS}), None

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return create_mcp_error(msg_id, -32602, "Tool name is required"), None

        try:
            result = execute_tool(user_id, tool_name, arguments)
        except ToolExecutionError as e:
            return create_mcp_error(msg_id, e.code, e.message), None

        return (
            create_mcp_response(
                msg_id, {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]}
            ),
            None,
        )

    elif method == "ping":
        return create_mcp_response(msg_id, {}), None

    else:
        return create_mcp_error(msg_id, -32601, f"Method not found: {method}"), None


@router.get("/authorize", tags=["mcp"])
async def mcp_authorize(
    response_type: str,
    client_id: str,
    redirect_uri: str,
    state: Optional[str] = None,
    scope: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = None,
):
    """OAuth authorize endpoint."""
    if client_id != "omi":
        raise HTTPException(status_code=400, detail="Invalid client_id")

    redirect_url = f"{redirect_uri}?code=omi"
    if state:
        redirect_url += f"&state={state}"

    return Response(status_code=302, headers={"Location": redirect_url})


@router.post("/token", tags=["mcp"])
async def mcp_token(request: Request):
    """OAuth token endpoint."""
    try:
        form_data = await request.form()
        client_secret = form_data.get("client_secret")
        client_id = form_data.get("client_id")
    except Exception:
        try:
            body = await request.json()
            client_secret = body.get("client_secret")
            client_id = body.get("client_id")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid request body")

    if not client_secret:
        raise HTTPException(status_code=400, detail="client_secret is required")

    if client_id != "omi":
        raise HTTPException(status_code=400, detail="Invalid client_id")

    user_id = authenticate_api_key(client_secret)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {
        "access_token": client_secret,
        "token_type": "Bearer",
    }


@router.post("/v1/mcp/sse", tags=["mcp"])
async def mcp_streamable_http(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    accept: Optional[str] = Header(None, alias="Accept"),
    pendant_local_trust: Optional[str] = Header(None, alias="X-Pendant-Local-Trust"),
    pendant_uid: Optional[str] = Header(None, alias="X-Pendant-Uid"),
):
    """
    Streamable HTTP Transport endpoint for MCP clients.

    This implements the MCP 2025-03-26 Streamable HTTP Transport specification.

    - POST JSON-RPC messages to this endpoint
    - Responses are returned as SSE stream or JSON depending on Accept header
    - Session ID is returned in Mcp-Session-Id header after initialization
    """
    user_id = authenticate_local_trust(pendant_local_trust, pendant_uid) or authenticate_api_key(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Provide via Authorization header.")

    # Rate limit per-user
    check_rate_limit_inline(user_id, "mcp:sse")

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Get session if provided
    session = None
    if mcp_session_id and mcp_session_id in active_sessions:
        session = active_sessions[mcp_session_id]
        # Verify session belongs to this user
        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session does not belong to this user")

    # Handle batch requests (array of messages)
    messages = body if isinstance(body, list) else [body]

    # Check if all messages are notifications/responses (no id)
    all_notifications = all(msg.get("id") is None for msg in messages)

    if all_notifications:
        # Process notifications without response
        for msg in messages:
            handle_mcp_message(user_id, msg, session)
        return Response(status_code=202)

    # Process messages and collect responses
    responses = []
    new_session_id = None

    for msg in messages:
        response, session_id = handle_mcp_message(user_id, msg, session)
        if session_id:
            new_session_id = session_id
        if response:
            responses.append(response)

    # Prepare headers
    headers = {}
    if new_session_id:
        headers["Mcp-Session-Id"] = new_session_id

    # Check if client accepts SSE
    wants_sse = accept and "text/event-stream" in accept

    if wants_sse:
        # Return as SSE stream
        async def event_generator():
            for resp in responses:
                yield f"event: message\ndata: {json.dumps(resp, default=str)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    else:
        # Return as JSON
        if len(responses) == 1:
            return JSONResponse(content=responses[0], headers=headers)
        else:
            return JSONResponse(content=responses, headers=headers)


@router.get("/v1/mcp/sse", tags=["mcp"])
async def mcp_sse_get(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    SSE endpoint for server-initiated messages (optional).

    Clients can GET this endpoint to listen for server-initiated notifications.
    This is optional per the MCP spec and mainly used for long-polling scenarios.
    """
    # Authenticate
    user_id = authenticate_api_key(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Provide via Authorization header.")

    # For backwards compatibility, also support the old SSE flow
    # Return an empty SSE stream that just sends keepalives
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                yield f"event: ping\ndata: {{}}\n\n"
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            # Normal cancellation when client disconnects
            pass
        except Exception as e:
            logging.warning(f"MCP SSE event generator error: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.delete("/v1/mcp/sse", tags=["mcp"])
async def mcp_delete_session(
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Delete/terminate an MCP session.
    """
    user_id = authenticate_api_key(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if not mcp_session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")

    if mcp_session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[mcp_session_id]
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this session")

    # Delete the session
    del active_sessions[mcp_session_id]
    return Response(status_code=204)


@router.get("/v1/mcp/sse/info", tags=["mcp"])
async def mcp_sse_info(request: Request):
    """
    Get information about the pre-hosted MCP server.
    """
    base_url = str(request.base_url).rstrip("/")
    return {
        "endpoint": "/v1/mcp/sse",
        "transport": "streamable-http",
        "protocol_version": "2025-03-26",
        "authentication": {
            "methods": ["api_key"],
            "api_key": {"header": "Authorization", "format": "Bearer <api_key>"},
        },
        "instructions": {
            "step1": "Create an MCP API key in the Omi app (Settings > Developer > MCP)",
            "step2": f"Set Server URL to: {base_url}/v1/mcp/sse",
            "step3": "Set Authorization header to your key",
        },
    }
