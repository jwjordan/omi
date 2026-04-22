"""Unit tests for database/knowledge_graph.py — Postgres impl.

Tests exercise every function and SQL pathway:
- get_knowledge_nodes, get_knowledge_node
- upsert_knowledge_node with merging (aliases, memory_ids)
- find_node_by_label_or_alias (label_lower and aliases_lower queries)
- get_knowledge_edges
- upsert_knowledge_edge with ID sanitization (replace '/' with '_')
- get_knowledge_graph (combined nodes + edges)
- delete_knowledge_graph (atomicity: batch() touches both tables)
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


def _mock_conn():
    """Returns (conn_mock, cursor_mock) with context-manager semantics."""
    cursor_mock = MagicMock()
    cursor_mock.__enter__ = MagicMock(return_value=cursor_mock)
    cursor_mock.__exit__ = MagicMock(return_value=None)

    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    conn_mock.__enter__ = MagicMock(return_value=conn_mock)
    conn_mock.__exit__ = MagicMock(return_value=None)
    return conn_mock, cursor_mock


def _mock_batch(db_mock, conn):
    """Set up db.batch() to return a transaction context."""
    batch_ctx = MagicMock()
    batch_ctx.__enter__ = MagicMock(return_value=conn)
    batch_ctx.__exit__ = MagicMock(return_value=None)
    db_mock.batch.return_value = batch_ctx


# ============================================================================
# get_knowledge_nodes
# ============================================================================


def test_get_knowledge_nodes_empty():
    """When no nodes exist, return empty list."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.knowledge_graph import get_knowledge_nodes
        result = get_knowledge_nodes('uid-1')

        assert result == []
        cur.execute.assert_called_once()
        sql = cur.execute.call_args.args[0]
        assert "SELECT id, label, node_type, data" in sql
        assert "FROM knowledge_graph_nodes" in sql
        assert "WHERE uid = %s" in sql


def test_get_knowledge_nodes_returns_list():
    """Return all nodes for a user."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('node-1', 'Alice', 'person', {'aliases': ['Alice W.'], 'memory_ids': ['m1']}),
            ('node-2', 'Bob', 'person', {'aliases': [], 'memory_ids': []}),
        ]

        from database.knowledge_graph import get_knowledge_nodes
        result = get_knowledge_nodes('uid-1')

        assert len(result) == 2
        assert result[0]['id'] == 'node-1'
        assert result[0]['label'] == 'Alice'
        assert result[0]['node_type'] == 'person'
        assert result[0]['aliases'] == ['Alice W.']
        assert result[1]['id'] == 'node-2'
        assert result[1]['label'] == 'Bob'


# ============================================================================
# get_knowledge_node
# ============================================================================


def test_get_knowledge_node_not_found():
    """Return None if node doesn't exist."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.knowledge_graph import get_knowledge_node
        result = get_knowledge_node('uid-1', 'node-xyz')

        assert result is None


def test_get_knowledge_node_found():
    """Fetch a single node by ID."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (
            'node-1',
            'Alice',
            'person',
            {'aliases': ['Alice W.'], 'memory_ids': ['m1']},
        )

        from database.knowledge_graph import get_knowledge_node
        result = get_knowledge_node('uid-1', 'node-1')

        assert result is not None
        assert result['id'] == 'node-1'
        assert result['label'] == 'Alice'
        assert result['node_type'] == 'person'
        assert result['aliases'] == ['Alice W.']


# ============================================================================
# upsert_knowledge_node
# ============================================================================


def test_upsert_knowledge_node_new_with_generated_id():
    """Insert a new node with auto-generated UUID."""
    with patch("database.knowledge_graph.db") as db_mock:
        with patch("database.knowledge_graph.uuid.uuid4") as uuid_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None  # Not found by label
            uuid_mock.return_value = 'uuid-123'

            from database.knowledge_graph import upsert_knowledge_node

            result = upsert_knowledge_node('uid-1', {
                'label': 'Alice',
                'node_type': 'person',
                'aliases': ['Alice W.'],
                'memory_ids': ['m1'],
            })

            assert result['id'] == 'uuid-123'
            assert result['label'] == 'Alice'
            assert result['node_type'] == 'person'
            assert result['aliases'] == ['Alice W.']
            # Check SQL was INSERT with UPSERT
            sql = cur.execute.call_args.args[0]
            assert "INSERT INTO knowledge_graph_nodes" in sql
            assert "ON CONFLICT (uid, id) DO UPDATE" in sql


def test_upsert_knowledge_node_new_with_explicit_id():
    """Insert a new node with caller-provided ID."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.knowledge_graph import upsert_knowledge_node

        result = upsert_knowledge_node('uid-1', {
            'id': 'node-custom',
            'label': 'Alice',
            'node_type': 'person',
            'memory_ids': ['m1'],
        })

        assert result['id'] == 'node-custom'
        assert result['label'] == 'Alice'


def test_upsert_knowledge_node_updates_existing_merges_fields():
    """Update existing node merges aliases and memory_ids."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        existing_data = {
            'aliases': ['Alice W.'],
            'memory_ids': ['m1', 'm2'],
            'created_at': now,
            'label_lower': 'alice',
        }
        cur.fetchone.return_value = (existing_data,)

        from database.knowledge_graph import upsert_knowledge_node

        result = upsert_knowledge_node('uid-1', {
            'id': 'node-1',
            'label': 'Alice',
            'node_type': 'person',
            'aliases': ['Ally'],  # New alias
            'memory_ids': ['m3'],  # New memory
        })

        # Merged: {Alice W., Ally}, {m1, m2, m3}
        assert set(result['aliases']) == {'Alice W.', 'Ally'}
        assert set(result['memory_ids']) == {'m1', 'm2', 'm3'}
        assert result['created_at'] == now  # preserved


# ============================================================================
# find_node_by_label_or_alias
# ============================================================================


def test_find_node_by_label_or_alias_empty_label():
    """Empty label returns None."""
    with patch("database.knowledge_graph.db") as db_mock:
        from database.knowledge_graph import find_node_by_label_or_alias
        result = find_node_by_label_or_alias('uid-1', '')
        assert result is None


def test_find_node_by_label_or_alias_label_match():
    """Find node by exact label (case-insensitive)."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # First query (label_lower) succeeds
        cur.fetchone.return_value = (
            'node-1',
            'Alice',
            'person',
            {'aliases': [], 'memory_ids': ['m1'], 'label_lower': 'alice'},
        )

        from database.knowledge_graph import find_node_by_label_or_alias
        result = find_node_by_label_or_alias('uid-1', 'ALICE')

        assert result is not None
        assert result['id'] == 'node-1'
        assert result['label'] == 'Alice'
        # Verify it queried label_lower
        sql = cur.execute.call_args.args[0]
        assert "data->>'label_lower'" in sql


def test_find_node_by_label_or_alias_alias_match():
    """Find node by alias when label doesn't match."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # First query (label_lower) returns None
        cur.fetchone.return_value = None
        # Reset for second call
        cur.fetchone.side_effect = [
            None,  # label_lower query
            ('node-2', 'Alice W.', 'person', {'aliases_lower': ['alice'], 'memory_ids': []}),  # aliases_lower query
        ]

        from database.knowledge_graph import find_node_by_label_or_alias
        result = find_node_by_label_or_alias('uid-1', 'alice')

        assert result is not None
        assert result['id'] == 'node-2'


# ============================================================================
# get_knowledge_edges
# ============================================================================


def test_get_knowledge_edges_empty():
    """When no edges exist, return empty list."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.knowledge_graph import get_knowledge_edges
        result = get_knowledge_edges('uid-1')

        assert result == []


def test_get_knowledge_edges_returns_list():
    """Return all edges for a user."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ('edge-1', 'node-1', 'node-2', 'likes', {'memory_ids': ['m1']}),
            ('edge-2', 'node-2', 'node-3', 'works_with', {'memory_ids': []}),
        ]

        from database.knowledge_graph import get_knowledge_edges
        result = get_knowledge_edges('uid-1')

        assert len(result) == 2
        assert result[0]['id'] == 'edge-1'
        assert result[0]['source_id'] == 'node-1'
        assert result[0]['target_id'] == 'node-2'
        assert result[0]['label'] == 'likes'
        assert result[1]['label'] == 'works_with'


# ============================================================================
# upsert_knowledge_edge (with ID sanitization)
# ============================================================================


def test_upsert_knowledge_edge_sanitizes_slash_in_generated_id():
    """Edge ID generated from label 'works/with' has '/' replaced with '_'."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None  # Not found

        from database.knowledge_graph import upsert_knowledge_edge

        result = upsert_knowledge_edge('uid-1', {
            'source_id': 'abc',
            'target_id': 'def',
            'label': 'works/with',
            'memory_ids': ['m1'],
        })

        assert '/' not in result['id']
        assert result['id'] == 'abc_works_with_def'


def test_upsert_knowledge_edge_sanitizes_slash_in_caller_provided_id():
    """Caller-provided edge ID with '/' gets sanitized."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.knowledge_graph import upsert_knowledge_edge

        result = upsert_knowledge_edge('uid-1', {
            'id': 'custom/edge/id',
            'source_id': 'x',
            'target_id': 'y',
            'label': 'test',
            'memory_ids': ['m1'],
        })

        assert '/' not in result['id']
        assert result['id'] == 'custom_edge_id'


def test_upsert_knowledge_edge_normal_id_unchanged():
    """Edge IDs without '/' pass through unchanged."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.knowledge_graph import upsert_knowledge_edge

        result = upsert_knowledge_edge('uid-1', {
            'source_id': 'abc',
            'target_id': 'def',
            'label': 'likes',
            'memory_ids': ['m1'],
        })

        assert result['id'] == 'abc_likes_def'


def test_upsert_knowledge_edge_merges_memory_ids():
    """Update existing edge merges memory_ids."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        existing_data = {
            'memory_ids': ['m1', 'm2'],
            'created_at': now,
        }
        cur.fetchone.return_value = (existing_data,)

        from database.knowledge_graph import upsert_knowledge_edge

        result = upsert_knowledge_edge('uid-1', {
            'id': 'edge-1',
            'source_id': 'src',
            'target_id': 'tgt',
            'label': 'likes',
            'memory_ids': ['m3'],
        })

        assert set(result['memory_ids']) == {'m1', 'm2', 'm3'}
        assert result['created_at'] == now  # preserved


# ============================================================================
# get_knowledge_graph
# ============================================================================


def test_get_knowledge_graph_combined():
    """Get both nodes and edges together."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # First call to get_knowledge_nodes
        # Second call to get_knowledge_edges
        cur.fetchall.side_effect = [
            [('node-1', 'Alice', 'person', {'aliases': []})],
            [('edge-1', 'node-1', 'node-2', 'likes', {'memory_ids': []})],
        ]

        from database.knowledge_graph import get_knowledge_graph
        result = get_knowledge_graph('uid-1')

        assert 'nodes' in result
        assert 'edges' in result
        assert len(result['nodes']) == 1
        assert len(result['edges']) == 1
        assert result['nodes'][0]['label'] == 'Alice'
        assert result['edges'][0]['label'] == 'likes'


# ============================================================================
# delete_knowledge_graph (atomicity)
# ============================================================================


def test_delete_knowledge_graph_uses_batch_transaction():
    """delete_knowledge_graph wraps both DELETEs in a batch() transaction."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.knowledge_graph import delete_knowledge_graph
        delete_knowledge_graph('uid-1')

        # Verify db.batch() was called (not db.connection())
        db_mock.batch.assert_called_once()
        # Verify two DELETE calls on the same cursor
        assert cur.execute.call_count == 2
        # First call deletes edges
        first_sql = cur.execute.call_args_list[0].args[0]
        assert "DELETE FROM knowledge_graph_edges" in first_sql
        assert "WHERE uid = %s" in first_sql
        # Second call deletes nodes
        second_sql = cur.execute.call_args_list[1].args[0]
        assert "DELETE FROM knowledge_graph_nodes" in second_sql
        assert "WHERE uid = %s" in second_sql


def test_delete_knowledge_graph_params():
    """delete_knowledge_graph passes uid as parameter."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.knowledge_graph import delete_knowledge_graph
        delete_knowledge_graph('uid-test-123')

        # Check both DELETE calls used the uid parameter
        params_list = [call.args[1] for call in cur.execute.call_args_list]
        assert ('uid-test-123',) in params_list
        assert ('uid-test-123',) in params_list


# ============================================================================
# Edge case: empty data JSONB
# ============================================================================


def test_get_knowledge_node_with_null_data():
    """Handle NULL data JSONB gracefully (defaults to empty dict)."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ('node-1', 'Alice', 'person', None)

        from database.knowledge_graph import get_knowledge_node
        result = get_knowledge_node('uid-1', 'node-1')

        assert result is not None
        assert result['id'] == 'node-1'
        assert result['label'] == 'Alice'


def test_get_knowledge_edges_with_null_data():
    """Handle NULL data JSONB in edges."""
    with patch("database.knowledge_graph.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ('edge-1', 'n1', 'n2', 'likes', None),
        ]

        from database.knowledge_graph import get_knowledge_edges
        result = get_knowledge_edges('uid-1')

        assert len(result) == 1
        assert result[0]['id'] == 'edge-1'
        assert result[0]['label'] == 'likes'
