"""
Tests for issue #4929: Edge ID sanitization in knowledge graph.

Firestore document IDs cannot contain '/'. When the LLM generates edge labels
like 'works/with', the '/' in the constructed edge_id breaks the Firestore path.
Fix: replace '/' with '_' in edge_id before using as document ID.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault(
    "ENCRYPTION_SECRET",
    "omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv",
)

# Import the module directly
from database.knowledge_graph import upsert_knowledge_edge


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


class TestEdgeIdSanitization:
    """Tests for '/' sanitization in edge document IDs (Postgres version)."""

    def setup_method(self):
        """Set up mocks for db.connection() calls."""
        pass

    def test_slash_in_label_replaced(self):
        """Edge label 'works/with' should produce edge_id with '_' not '/'."""
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None  # No existing edge

            edge_data = {
                'source_id': 'abc',
                'target_id': 'def',
                'label': 'works/with',
                'memory_ids': ['m1'],
            }
            result = upsert_knowledge_edge('uid-1', edge_data)
            edge_id = result['id']
            assert '/' not in edge_id
            assert edge_id == 'abc_works_with_def'

    def test_multiple_slashes_replaced(self):
        """Multiple '/' characters should all be replaced."""
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None

            edge_data = {
                'source_id': 'a',
                'target_id': 'b',
                'label': 'is/was/related',
                'memory_ids': ['m1'],
            }
            result = upsert_knowledge_edge('uid-1', edge_data)
            assert '/' not in result['id']
            assert result['id'] == 'a_is_was_related_b'

    def test_caller_provided_id_with_slash_sanitized(self):
        """Even caller-provided edge IDs with '/' should be sanitized."""
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None

            edge_data = {
                'id': 'custom/edge/id',
                'source_id': 'x',
                'target_id': 'y',
                'label': 'test',
                'memory_ids': ['m1'],
            }
            result = upsert_knowledge_edge('uid-1', edge_data)
            assert '/' not in result['id']
            assert result['id'] == 'custom_edge_id'

    def test_label_without_slash_unchanged(self):
        """Normal labels without '/' should produce correct edge IDs."""
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None

            edge_data = {
                'source_id': 'abc',
                'target_id': 'def',
                'label': 'likes',
                'memory_ids': ['m1'],
            }
            result = upsert_knowledge_edge('uid-1', edge_data)
            assert result['id'] == 'abc_likes_def'

    def test_document_called_with_sanitized_id(self):
        """The database INSERT/UPDATE must use the sanitized ID."""
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None

            edge_data = {
                'source_id': 'src',
                'target_id': 'tgt',
                'label': 'has/a',
                'memory_ids': ['m1'],
            }
            upsert_knowledge_edge('uid-1', edge_data)
            # Verify the edge_id passed to SQL has no '/'
            sql = cur.execute.call_args.args[0]
            params = cur.execute.call_args.args[1]
            # edge_id is the second parameter (after uid)
            edge_id_param = params[1]
            assert '/' not in edge_id_param
            assert edge_id_param == 'src_has_a_tgt'

    def test_empty_label_produces_valid_id(self):
        """Empty label should produce a valid edge_id with no slash."""
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None

            edge_data = {
                'source_id': 'abc',
                'target_id': 'def',
                'label': '',
                'memory_ids': ['m1'],
            }
            result = upsert_knowledge_edge('uid-1', edge_data)
            assert '/' not in result['id']
            assert result['id'] == 'abc__def'

    def test_label_only_slash_produces_valid_id(self):
        """Label that is just '/' should be sanitized to '_'."""
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None

            edge_data = {
                'source_id': 'abc',
                'target_id': 'def',
                'label': '/',
                'memory_ids': ['m1'],
            }
            result = upsert_knowledge_edge('uid-1', edge_data)
            assert '/' not in result['id']
            assert result['id'] == 'abc___def'

    def test_caller_provided_dotdot_id_unchanged(self):
        """Caller-provided edge_id '..' is not a slash issue — passes through.

        Note: '..' as a standalone ID is not a slash-related concern. In practice
        edge IDs are always '{source}_{label}_{target}' format so '..' cannot occur
        from normal construction. This test documents current behavior.
        """
        with patch("database.knowledge_graph.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = None

            edge_data = {
                'id': '..',
                'source_id': 's',
                'target_id': 't',
                'label': 'x',
                'memory_ids': ['m1'],
            }
            result = upsert_knowledge_edge('uid-1', edge_data)
            assert result['id'] == '..'
