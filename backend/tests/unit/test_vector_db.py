"""Tests for database.vector_db after the Pinecone -> Postgres+pgvector migration.

These tests must NOT hit a live Postgres. We stub `psycopg_pool.ConnectionPool`
at import time (before `database.vector_db` is loaded) so the module picks up
a fake pool whose `.connection()` returns a mock connection that records every
`execute()` call. Each test then asserts on SQL text + params.

The stubbing pattern mirrors tests/unit/test_byok_security.py and
tests/unit/test_endpoints_verify_token.py.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Module-level stubs. Run BEFORE importing database.vector_db.
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")


class _FakeCursor:
    """Records execute() calls and replays rows from a queue."""

    def __init__(self, rows_queue):
        self.rows_queue = rows_queue  # list of lists; each execute() pops the next list
        self.executed = []            # list of (sql, params)
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self._rows = self.rows_queue.pop(0) if self.rows_queue else []
        return self

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cursor

    def execute(self, sql, params=None):
        return self._cursor.execute(sql, params)


class _FakePoolContext:
    """Context manager returned by pool.connection()."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *a):
        return False


class _FakePool:
    def __init__(self, *args, **kwargs):
        self._conn = None

    def connection(self):
        return _FakePoolContext(self._conn)


# Stub psycopg_pool.ConnectionPool so vector_db's module-level `_build_pool()`
# doesn't open a real connection. The shape has to match: ConnectionPool(...)
# returns something with `.connection()` -> context manager -> connection.
_fake_pool_module = types.ModuleType("psycopg_pool")
_fake_pool_module.ConnectionPool = _FakePool
sys.modules["psycopg_pool"] = _fake_pool_module

# Stub pgvector.psycopg.register_vector to a no-op.
_fake_pgvector_pkg = types.ModuleType("pgvector")
_fake_pgvector_psycopg = types.ModuleType("pgvector.psycopg")
_fake_pgvector_psycopg.register_vector = lambda conn: None
sys.modules["pgvector"] = _fake_pgvector_pkg
sys.modules["pgvector.psycopg"] = _fake_pgvector_psycopg

# Stub utils.llm.clients.embeddings so importing vector_db doesn't pull OpenAI.
if "utils.llm.clients" not in sys.modules:
    clients_stub = types.ModuleType("utils.llm.clients")
    clients_stub.embeddings = MagicMock()
    sys.modules["utils.llm.clients"] = clients_stub

from database import vector_db  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_pool(monkeypatch):
    """Install a fake pool on vector_db and return (pool, cursor)."""
    cursor = _FakeCursor(rows_queue=[])
    conn = _FakeConnection(cursor)
    pool = _FakePool()
    pool._conn = conn
    monkeypatch.setattr(vector_db, "_pool", pool)
    return pool, cursor


@pytest.fixture
def fake_embeddings(monkeypatch):
    fake = MagicMock()
    fake.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    fake.embed_documents = MagicMock(
        side_effect=lambda texts: [[0.01 * i, 0.02 * i] for i, _ in enumerate(texts, start=1)]
    )
    monkeypatch.setattr(vector_db, "embeddings", fake)
    return fake


# ---------------------------------------------------------------------------
# ns1 / conversation_vectors tests
# ---------------------------------------------------------------------------
class TestUpsertVector:
    def test_writes_insert_on_conflict(self, fake_pool):
        _, cursor = fake_pool
        vector_db.upsert_vector("uid1", "conv1", [0.1, 0.2, 0.3])

        assert len(cursor.executed) == 1
        sql, params = cursor.executed[0]
        assert "conversation_vectors" in sql
        assert "INSERT" in sql.upper()
        assert "ON CONFLICT" in sql.upper()
        # id, uid, memory_id, created_at, embedding, metadata
        assert params[0] == "uid1-conv1"
        assert params[1] == "uid1"
        assert params[2] == "conv1"
        assert params[4] == [0.1, 0.2, 0.3]


class TestUpsertVector2:
    def test_merges_metadata(self, fake_pool):
        _, cursor = fake_pool
        vector_db.upsert_vector2(
            "uid1", "conv1", [0.1, 0.2], {"topics": ["t1"], "entities": ["e1"]}
        )

        sql, params = cursor.executed[0]
        assert "conversation_vectors" in sql
        # metadata is the last param and is a JSON string
        meta = params[5]
        # it's serialized as JSON
        import json as _json
        parsed = _json.loads(meta)
        assert parsed["uid"] == "uid1"
        assert parsed["memory_id"] == "conv1"
        assert parsed["topics"] == ["t1"]
        assert parsed["entities"] == ["e1"]


class TestUpdateVectorMetadata:
    def test_forces_uid_and_memory_id(self, fake_pool):
        _, cursor = fake_pool
        vector_db.update_vector_metadata("uidX", "convY", {"custom": "val"})

        sql, params = cursor.executed[0]
        assert "UPDATE conversation_vectors" in sql
        assert "SET metadata" in sql
        import json as _json
        meta = _json.loads(params[0])
        assert meta["uid"] == "uidX"
        assert meta["memory_id"] == "convY"
        assert meta["custom"] == "val"
        # WHERE id = ...
        assert params[-1] == "uidX-convY"


class TestUpsertVectors:
    def test_batch(self, fake_pool):
        _, cursor = fake_pool
        vector_db.upsert_vectors("uid1", [[0.1], [0.2]], ["c1", "c2"])
        # Expect 2 execute calls (one per row) OR 1 call with executemany.
        # Accept either, but SQL must reference conversation_vectors.
        assert any("conversation_vectors" in call[0] for call in cursor.executed)


class TestQueryVectors:
    def test_returns_stripped_ids(self, fake_pool, fake_embeddings):
        _, cursor = fake_pool
        cursor.rows_queue = [[("uid1-conv1", 0.9), ("uid1-conv2", 0.8)]]

        ids = vector_db.query_vectors("hello", "uid1", k=5)

        assert ids == ["conv1", "conv2"]
        fake_embeddings.embed_query.assert_called_once_with("hello")
        sql, params = cursor.executed[0]
        assert "conversation_vectors" in sql
        assert "<=>" in sql
        assert "uid = " in sql or "uid=" in sql or params[1] == "uid1" or "uid1" in params

    def test_with_date_range(self, fake_pool, fake_embeddings):
        _, cursor = fake_pool
        cursor.rows_queue = [[("uid1-conv1", 0.9)]]
        ids = vector_db.query_vectors("hi", "uid1", starts_at=100, ends_at=200, k=3)
        assert ids == ["conv1"]
        sql, params = cursor.executed[0]
        assert "created_at" in sql
        assert 100 in params and 200 in params


class TestQueryVectorsByMetadata:
    def test_applies_jsonb_filter_and_sorts_by_match_count(self, fake_pool):
        _, cursor = fake_pool
        # Return three rows. First has 1 topic match, second has 2 matches,
        # third has 0 matches (fell through on another filter).
        cursor.rows_queue = [[
            ("uid1-c_a", {"memory_id": "c_a", "topics": ["t1"], "entities": [], "people_mentioned": []}),
            ("uid1-c_b", {"memory_id": "c_b", "topics": ["t1"], "entities": ["e1"], "people_mentioned": []}),
            ("uid1-c_c", {"memory_id": "c_c", "topics": [], "entities": [], "people_mentioned": ["p1"]}),
        ]]

        ids = vector_db.query_vectors_by_metadata(
            uid="uid1",
            vector=[0.1, 0.2, 0.3],
            dates_filter=[],
            people=["p1"],
            topics=["t1"],
            entities=["e1"],
            dates=[],
            limit=5,
        )

        # c_b has the most matches (2), then c_a or c_c with 1 each.
        assert ids[0] == "c_b"
        assert set(ids) == {"c_a", "c_b", "c_c"}

        sql, _params = cursor.executed[0]
        assert "?|" in sql  # JSONB "contains any of" operator
        assert "topics" in sql
        assert "entities" in sql
        assert "people_mentioned" in sql

    def test_fallback_when_no_results(self, fake_pool):
        _, cursor = fake_pool
        # First query: empty. Second (fallback without metadata filter): one row.
        cursor.rows_queue = [
            [],
            [("uid1-c1", {"memory_id": "c1", "topics": [], "entities": [], "people_mentioned": []})],
        ]
        # Must have both date filter AND people/topics to trigger the 3-clause path
        from datetime import datetime, timezone
        ids = vector_db.query_vectors_by_metadata(
            uid="uid1",
            vector=[0.1, 0.2, 0.3],
            dates_filter=[
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 12, 31, tzinfo=timezone.utc),
            ],
            people=["p1"],
            topics=["t1"],
            entities=[],
            dates=[],
            limit=5,
        )
        assert ids == ["c1"]
        assert len(cursor.executed) == 2  # retried


class TestDeleteVector:
    def test_issues_delete(self, fake_pool):
        _, cursor = fake_pool
        vector_db.delete_vector("uid1", "conv1")
        sql, params = cursor.executed[0]
        assert "DELETE FROM conversation_vectors" in sql
        assert "WHERE id" in sql
        assert params[0] == "uid1-conv1"


# ---------------------------------------------------------------------------
# ns2 / memory_vectors tests
# ---------------------------------------------------------------------------
class TestUpsertMemoryVector:
    def test_writes_memory_row(self, fake_pool, fake_embeddings):
        _, cursor = fake_pool
        vector = vector_db.upsert_memory_vector("uid1", "m1", "hello world", "identity")

        assert vector == [0.1, 0.2, 0.3]
        sql, params = cursor.executed[0]
        assert "memory_vectors" in sql
        assert "INSERT" in sql.upper()
        assert params[0] == "uid1-m1"
        assert params[1] == "uid1"
        assert params[2] == "m1"
        assert params[3] == "identity"  # category
        assert params[5] == [0.1, 0.2, 0.3]


class TestFindSimilarMemories:
    def test_threshold_filter(self, fake_pool, fake_embeddings):
        _, cursor = fake_pool
        # score = 1 - distance; simulate three rows with varying distances
        cursor.rows_queue = [[
            ("m1", "identity", 0.05),    # score 0.95 -> above 0.85
            ("m2", "preferences", 0.10),  # score 0.90 -> above 0.85
            ("m3", "skills", 0.25),      # score 0.75 -> below 0.85
        ]]

        results = vector_db.find_similar_memories(
            "uid1", "hello", threshold=0.85, limit=5
        )

        assert len(results) == 2
        assert results[0]["memory_id"] == "m1"
        assert results[0]["category"] == "identity"
        assert results[0]["score"] == pytest.approx(0.95)
        assert results[1]["memory_id"] == "m2"
        assert results[1]["score"] == pytest.approx(0.90)

    def test_duplicate_check_uses_top_result(self, fake_pool, fake_embeddings):
        _, cursor = fake_pool
        cursor.rows_queue = [[("m1", "identity", 0.05)]]
        dup = vector_db.check_memory_duplicate("uid1", "hi", threshold=0.85)
        assert dup is not None
        assert dup["memory_id"] == "m1"

    def test_duplicate_check_returns_none_below_threshold(self, fake_pool, fake_embeddings):
        _, cursor = fake_pool
        cursor.rows_queue = [[("m1", "identity", 0.5)]]  # score 0.5, below 0.85
        dup = vector_db.check_memory_duplicate("uid1", "hi", threshold=0.85)
        assert dup is None


class TestSearchMemoriesByVector:
    def test_returns_memory_ids(self, fake_pool, fake_embeddings):
        _, cursor = fake_pool
        cursor.rows_queue = [[("m1",), ("m2",), ("m3",)]]
        ids = vector_db.search_memories_by_vector("uid1", "query", limit=10)
        assert ids == ["m1", "m2", "m3"]


class TestDeleteMemoryVector:
    def test_deletes(self, fake_pool):
        _, cursor = fake_pool
        vector_db.delete_memory_vector("uid1", "m1")
        sql, params = cursor.executed[0]
        assert "DELETE FROM memory_vectors" in sql
        assert params[0] == "uid1-m1"


# ---------------------------------------------------------------------------
# No DATABASE_URL
# ---------------------------------------------------------------------------
class TestNoDatabaseUrl:
    def test_functions_raise_when_pool_none(self, monkeypatch):
        monkeypatch.setattr(vector_db, "_pool", None)
        with pytest.raises(RuntimeError, match="DATABASE_URL not configured"):
            vector_db.upsert_vector("uid1", "conv1", [0.1])


# ---------------------------------------------------------------------------
# ns3 stubs
# ---------------------------------------------------------------------------
class TestScreenActivityStubs:
    def test_upsert_returns_zero(self, caplog):
        with caplog.at_level("WARNING"):
            result = vector_db.upsert_screen_activity_vectors("uid1", [{"id": 1}])
        assert result == 0
        assert any(
            "screen activity" in r.message.lower() for r in caplog.records
        )

    def test_search_returns_empty(self, caplog):
        with caplog.at_level("WARNING"):
            result = vector_db.search_screen_activity_vectors(
                "uid1", [0.1, 0.2], start_date=0, end_date=1, app_filter=None, k=10
            )
        assert result == []

    def test_delete_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            result = vector_db.delete_screen_activity_vectors("uid1", [1, 2, 3])
        assert result is None

    def test_namespace_constants_preserved(self):
        assert vector_db.MEMORIES_NAMESPACE == "ns2"
        assert vector_db.SCREEN_ACTIVITY_NAMESPACE == "ns3"
