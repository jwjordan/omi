"""Tests for the Edwin/pendant_memory endpoints in routers/memories.py.

These cover:
  * /v3/memories/search — semantic search wrapper around
    search_memories_by_vector + get_memories_by_ids that returns full
    content. Verifies vector ranking is preserved (rank ascending) and
    the response shape includes the fields Edwin's MCP tool consumes.
  * /v3/memories/{id}/promote-to-edwin — flag-stamp endpoint.
  * mark_memory_promoted_to_edwin DB helper — payload merging behavior.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# DB helper: mark_memory_promoted_to_edwin
# ---------------------------------------------------------------------------


def _setup_db_mock(rowcount=1):
    cursor = MagicMock()
    cursor.rowcount = rowcount
    cursor_ctx = MagicMock()
    cursor_ctx.__enter__ = MagicMock(return_value=cursor)
    cursor_ctx.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor_ctx
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    db_mock = MagicMock()
    db_mock.connection.return_value = conn_ctx
    return db_mock, cursor


class TestMarkPromotedToEdwinDB:
    def test_writes_flag_and_timestamp(self):
        from database import memories as mdb

        db_mock, cursor = _setup_db_mock(rowcount=1)
        with patch.object(mdb, 'db', db_mock):
            ok = mdb.mark_memory_promoted_to_edwin('uid-1', 'mem-1')
        assert ok is True
        sql, params = cursor.execute.call_args[0]
        assert 'data || %s::jsonb' in sql
        # First param is the JSONB delta — parse and assert keys
        import json
        delta = json.loads(params[0])
        assert delta['promoted_to_edwin'] is True
        assert 'promoted_to_edwin_at' in delta
        assert 'promoted_to_edwin_note' not in delta  # no note → key omitted
        assert params[2] == 'uid-1'
        assert params[3] == 'mem-1'

    def test_writes_note_when_provided(self):
        from database import memories as mdb

        db_mock, cursor = _setup_db_mock(rowcount=1)
        with patch.object(mdb, 'db', db_mock):
            ok = mdb.mark_memory_promoted_to_edwin('uid-1', 'mem-1', note='good rule')
        assert ok is True
        import json
        delta = json.loads(cursor.execute.call_args[0][1][0])
        assert delta['promoted_to_edwin_note'] == 'good rule'

    def test_returns_false_when_memory_missing(self):
        from database import memories as mdb

        db_mock, _ = _setup_db_mock(rowcount=0)
        with patch.object(mdb, 'db', db_mock):
            ok = mdb.mark_memory_promoted_to_edwin('uid-1', 'missing-id')
        assert ok is False


# ---------------------------------------------------------------------------
# Endpoint: /v3/memories/search
# ---------------------------------------------------------------------------


class TestSemanticSearchEndpoint:
    def test_preserves_vector_ranking(self):
        from routers import memories as memories_router

        # search_memories_by_vector returns memory_ids in best-first order.
        # get_memories_by_ids does NOT preserve order (it ANY()'s an array),
        # so the endpoint must reassemble by the search-returned id order.
        with (
            patch.object(memories_router, 'search_memories_by_vector', return_value=['c', 'a', 'b']),
            patch.object(memories_router.memories_db, 'get_memories_by_ids', return_value=[
                {
                    'id': 'a', 'content': 'memory A', 'category': 'system',
                    'created_at': datetime(2026, 4, 20, tzinfo=timezone.utc),
                },
                {
                    'id': 'b', 'content': 'memory B', 'category': 'interesting',
                    'created_at': datetime(2026, 4, 21, tzinfo=timezone.utc),
                },
                {
                    'id': 'c', 'content': 'memory C', 'category': 'manual',
                    'created_at': datetime(2026, 4, 22, tzinfo=timezone.utc),
                },
            ]),
        ):
            req = memories_router.MemoriesSearchRequest(query='kelvin', limit=5)
            resp = memories_router.semantic_search_memories(req, uid='uid-1')

        assert resp.query == 'kelvin'
        assert [h.id for h in resp.hits] == ['c', 'a', 'b']
        assert [h.rank for h in resp.hits] == [0, 1, 2]
        assert resp.hits[0].content == 'memory C'
        assert resp.hits[0].category == 'manual'
        assert resp.hits[0].created_at == '2026-04-22T00:00:00+00:00'
        # All categories surface — endpoint does not filter category
        assert {h.category for h in resp.hits} == {'system', 'interesting', 'manual'}

    def test_empty_when_no_vectors_match(self):
        from routers import memories as memories_router

        with patch.object(memories_router, 'search_memories_by_vector', return_value=[]):
            req = memories_router.MemoriesSearchRequest(query='nothing', limit=5)
            resp = memories_router.semantic_search_memories(req, uid='uid-1')
        assert resp.hits == []

    def test_promoted_flag_surfaces(self):
        """Edwin's MCP tool relies on promoted_to_edwin to filter "show me
        unpromoted memories" UX. The flag must round-trip from JSONB to hit."""
        from routers import memories as memories_router

        with (
            patch.object(memories_router, 'search_memories_by_vector', return_value=['p1']),
            patch.object(memories_router.memories_db, 'get_memories_by_ids', return_value=[
                {
                    'id': 'p1',
                    'content': 'already promoted',
                    'promoted_to_edwin': True,
                    'promoted_to_edwin_at': '2026-04-25T00:00:00+00:00',
                },
            ]),
        ):
            req = memories_router.MemoriesSearchRequest(query='x', limit=5)
            resp = memories_router.semantic_search_memories(req, uid='uid-1')
        assert resp.hits[0].promoted_to_edwin is True
        assert resp.hits[0].promoted_to_edwin_at == '2026-04-25T00:00:00+00:00'

    def test_skips_ids_with_no_decryptable_memory(self):
        """If get_memories_by_ids drops a row (decrypt failure, soft-delete),
        the endpoint must not crash and must still rank the surviving rows."""
        from routers import memories as memories_router

        with (
            patch.object(memories_router, 'search_memories_by_vector', return_value=['gone', 'present']),
            patch.object(memories_router.memories_db, 'get_memories_by_ids', return_value=[
                {'id': 'present', 'content': 'kept'},
            ]),
        ):
            req = memories_router.MemoriesSearchRequest(query='x', limit=5)
            resp = memories_router.semantic_search_memories(req, uid='uid-1')
        assert [h.id for h in resp.hits] == ['present']
        # rank is the position in the original vector ordering — survivor was at index 1
        assert resp.hits[0].rank == 1


# ---------------------------------------------------------------------------
# Endpoint: /v3/memories/{id}/promote-to-edwin
# ---------------------------------------------------------------------------


class TestPromoteToEdwinEndpoint:
    def test_404_when_memory_missing(self):
        from fastapi import HTTPException
        from routers import memories as memories_router

        with patch.object(memories_router.memories_db, 'get_memory', return_value=None):
            req = memories_router.PromoteToEdwinRequest()
            with pytest.raises(HTTPException) as exc:
                memories_router.promote_memory_to_edwin('missing', req, uid='uid-1')
            assert exc.value.status_code == 404

    def test_marks_when_memory_exists(self):
        from routers import memories as memories_router

        with (
            patch.object(memories_router.memories_db, 'get_memory', return_value={'id': 'mem-1'}),
            patch.object(memories_router.memories_db, 'mark_memory_promoted_to_edwin', return_value=True) as mark_mock,
        ):
            req = memories_router.PromoteToEdwinRequest(note='useful preference')
            resp = memories_router.promote_memory_to_edwin('mem-1', req, uid='uid-1')
        assert resp == {'status': 'ok', 'memory_id': 'mem-1'}
        mark_mock.assert_called_once_with('uid-1', 'mem-1', note='useful preference')

    def test_idempotent_repeated_calls(self):
        """Calling twice in a row should both succeed and re-stamp the flag."""
        from routers import memories as memories_router

        with (
            patch.object(memories_router.memories_db, 'get_memory', return_value={'id': 'mem-1'}),
            patch.object(memories_router.memories_db, 'mark_memory_promoted_to_edwin', return_value=True),
        ):
            req = memories_router.PromoteToEdwinRequest()
            r1 = memories_router.promote_memory_to_edwin('mem-1', req, uid='uid-1')
            r2 = memories_router.promote_memory_to_edwin('mem-1', req, uid='uid-1')
        assert r1 == r2 == {'status': 'ok', 'memory_id': 'mem-1'}
