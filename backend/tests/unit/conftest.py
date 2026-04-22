"""Pytest configuration for unit tests - mock heavy dependencies."""

import os
import sys
import types
from unittest.mock import MagicMock

# Set encryption secret before any imports that depend on it
os.environ.setdefault("ENCRYPTION_SECRET", "omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-for-unit-tests")
os.environ.setdefault("DEEPGRAM_API_KEY", "dg-test-fake-for-unit-tests")
os.environ.setdefault("GOOGLE_API_KEY", "goog-test-fake-for-unit-tests")
os.environ.setdefault("ANTHROPIC_API_KEY", "ant-test-fake-for-unit-tests")

# Mock psycopg_pool and pgvector before any imports
sys.modules["psycopg_pool"] = MagicMock()
sys.modules["pgvector"] = MagicMock()
sys.modules["pgvector.psycopg"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["stripe"] = MagicMock()
sys.modules["opuslib"] = MagicMock()
sys.modules["pydub"] = MagicMock()
sys.modules["pycountry"] = MagicMock()

# Also mock database.redis_db since it fails on Python 3.9 with dict | None syntax
sys.modules["database.redis_db"] = MagicMock()

# Mock utils.subscription since it uses Python 3.10+ syntax (int | None)
sys.modules["utils.subscription"] = MagicMock()

# Mock utils.other.storage to avoid GCS credential check at import time
sys.modules["utils.other.storage"] = MagicMock()

# Stub utils.llm.clients so modules that import `embeddings` from it can be
# loaded without the full anthropic/langchain/tiktoken chain.  Tests that need
# a specific embeddings behaviour patch the name on the module under test.
sys.modules["utils.llm.clients"] = MagicMock()

# ---------------------------------------------------------------------------
# Stub heavy third-party packages that are not installed in the test venv.
# These are audio/ML packages needed by conversations.py transitive imports.
# We use AutoAttrModule so that subpackage attribute access returns MagicMock.
# ---------------------------------------------------------------------------

class _AutoAttrModule(types.ModuleType):
    """A ModuleType whose missing attributes resolve to MagicMock instances."""
    def __getattr__(self, item):
        v = MagicMock()
        setattr(self, item, v)
        return v

def _stub_pkg(name: str) -> "_AutoAttrModule":
    m = _AutoAttrModule(name)
    m.__path__ = [name]
    sys.modules[name] = m
    return m

for _pkg in [
    # LangChain ecosystem
    "langchain_core",
    "langchain_core.output_parsers",
    "langchain_core.prompts",
    "langchain_core.messages",
    "langchain_core.callbacks",
    "langchain_core.outputs",
    "langchain_core.runnables",
    "langchain_core.tools",
    "langchain_core.tracers",
    "langchain_openai",
    "langchain",
    "langchain.schema",
    # Typesense (search backend)
    "typesense",
    # Audio / ML
    "av",
    "fal_client",
    "deepgram",
    "onnxruntime",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.distance",
]:
    if _pkg not in sys.modules:
        _stub_pkg(_pkg)

# ---------------------------------------------------------------------------
# Stub routers that pull in the heavy dependencies above, so that tests that
# import `main.app` can spin up FastAPI without loading audio/ML code.
# The conversations router is intentionally left real so its endpoints exist.
# Each stub exposes a real (empty) APIRouter so that app.include_router()
# works at import time without installing audio/ML packages.
# ---------------------------------------------------------------------------
from fastapi import APIRouter as _APIRouter

_HEAVY_ROUTERS = [
    "chat", "firmware", "transcribe", "notifications", "speech_profile",
    "agents", "users", "trends", "sync", "apps", "payment", "integration",
    "memories", "mcp", "mcp_sse", "oauth", "auth", "action_items",
    "task_integrations", "integrations", "other", "developer", "updates",
    "calendar_meetings", "imports", "knowledge_graph", "wrapped", "folders",
    "goals", "announcements", "phone_calls", "agent_tools", "tools",
    "metrics", "fair_use_admin", "staged_tasks", "focus_sessions", "advice",
    "chat_sessions", "scores",
]

for _rname in _HEAVY_ROUTERS:
    _key = f"routers.{_rname}"
    if _key not in sys.modules:
        _rm = types.ModuleType(_key)
        _rm.__path__ = []
        _rm.router = _APIRouter()
        sys.modules[_key] = _rm
