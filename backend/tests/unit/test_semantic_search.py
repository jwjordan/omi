"""Unit tests for utils/conversations/semantic_search.py.

We mock both `embeddings.embed_query` and `database._client.db.connection()`.
The function under test:
  1. Embeds the query to a 768-dim vector.
  2. Runs a single SELECT against conversation_vectors JOIN conversations.
  3. For each hit, decrypts transcript_segments and picks up to 3 segments
     matching any query term (case-insensitive substring).
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


def _mock_db_connection():
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=None)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=None)
    return conn, cursor


def _fake_conversation_row(conv_id: str, title: str, overview: str, segments: list) -> tuple:
    """Shape of one DB row returned by the helper's SELECT."""
    started = datetime(2026, 4, 22, 12, 51, 23, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 22, 13, 56, 44, tzinfo=timezone.utc)
    data = {
        "structured": {"title": title, "overview": overview},
        "transcript_segments": segments,  # pre-decrypted list; monkeypatch decrypt to passthrough
    }
    similarity = 0.85
    return (conv_id, data, started, finished, similarity)


def test_embeds_query_and_runs_sql_with_vector_param():
    from utils.conversations.semantic_search import semantic_search_conversations

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock, \
         patch("utils.conversations.semantic_search.db") as db_mock, \
         patch("utils.conversations.semantic_search._decrypt_conversation_data", side_effect=lambda d, uid: d):
        emb_mock.embed_query.return_value = [0.1] * 768
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        result = semantic_search_conversations(uid="james", query="minecraft", limit=5)

        emb_mock.embed_query.assert_called_once_with("minecraft")
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM conversation_vectors" in sql
        assert "JOIN conversations" in sql
        assert "ORDER BY" in sql and "<=>" in sql
        # The vector param is passed twice (ORDER BY + SELECT similarity).
        assert params[0] == [0.1] * 768
        assert "james" in params
        assert result == {"query": "minecraft", "hits": []}


def test_excerpt_picks_segments_matching_query_terms():
    from utils.conversations.semantic_search import semantic_search_conversations

    segments = [
        {"text": "We talked about Grayson's soccer practice.", "start": 10.0, "speaker": "SPEAKER_00"},
        {"text": "Then the minecraft server came up.", "start": 42.0, "speaker": "SPEAKER_00"},
        {"text": "Later discussed JMI demo.", "start": 120.0, "speaker": "SPEAKER_01"},
        {"text": "Back to the minecraft realm for the kids.", "start": 180.0, "speaker": "SPEAKER_01"},
    ]
    row = _fake_conversation_row("c1", "T", "O", segments)

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock, \
         patch("utils.conversations.semantic_search.db") as db_mock, \
         patch("utils.conversations.semantic_search._decrypt_conversation_data", side_effect=lambda d, uid: d):
        emb_mock.embed_query.return_value = [0.1] * 768
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [row]

        result = semantic_search_conversations(uid="james", query="minecraft server", limit=5)

        assert len(result["hits"]) == 1
        excerpts = result["hits"][0]["excerpts"]
        assert len(excerpts) == 2  # only 2 segments contain "minecraft"; "server" matches same
        texts = [e["text"] for e in excerpts]
        assert any("minecraft server" in t for t in texts)
        assert all("Grayson" not in t for t in texts)
        # Excerpts preserve speaker + start
        assert all("speaker" in e and "start_seconds" in e for e in excerpts)


def test_excerpt_capped_at_three():
    from utils.conversations.semantic_search import semantic_search_conversations

    segments = [{"text": f"minecraft discussion {i}", "start": float(i), "speaker": "SPEAKER_00"} for i in range(10)]
    row = _fake_conversation_row("c1", "T", "O", segments)

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock, \
         patch("utils.conversations.semantic_search.db") as db_mock, \
         patch("utils.conversations.semantic_search._decrypt_conversation_data", side_effect=lambda d, uid: d):
        emb_mock.embed_query.return_value = [0.1] * 768
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [row]

        result = semantic_search_conversations(uid="james", query="minecraft", limit=5)
        assert len(result["hits"][0]["excerpts"]) == 3


def test_no_excerpts_when_no_segments_match():
    from utils.conversations.semantic_search import semantic_search_conversations

    segments = [{"text": "unrelated content about grocery shopping", "start": 5.0, "speaker": "SPEAKER_00"}]
    row = _fake_conversation_row("c1", "T", "O", segments)

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock, \
         patch("utils.conversations.semantic_search.db") as db_mock, \
         patch("utils.conversations.semantic_search._decrypt_conversation_data", side_effect=lambda d, uid: d):
        emb_mock.embed_query.return_value = [0.1] * 768
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [row]

        result = semantic_search_conversations(uid="james", query="minecraft", limit=5)
        assert result["hits"][0]["excerpts"] == []
        # Summary still included
        assert result["hits"][0]["title"] == "T"
        assert result["hits"][0]["overview"] == "O"
        assert result["hits"][0]["similarity"] == 0.85


def test_since_and_until_passed_as_sql_params():
    from utils.conversations.semantic_search import semantic_search_conversations

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock, \
         patch("utils.conversations.semantic_search.db") as db_mock, \
         patch("utils.conversations.semantic_search._decrypt_conversation_data", side_effect=lambda d, uid: d):
        emb_mock.embed_query.return_value = [0.1] * 768
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        semantic_search_conversations(
            uid="james",
            query="x",
            since="2026-04-22T00:00:00Z",
            until="2026-04-23T00:00:00Z",
            limit=5,
        )
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]

        # Both bounds made it into the WHERE clause
        assert "c.started_at >= %s" in sql
        assert "c.started_at <= %s" in sql

        # Both datetimes appear in the param tuple, in chronological order
        dt_params = [p for p in params if isinstance(p, datetime)]
        assert len(dt_params) == 2
        expected_since = datetime(2026, 4, 22, tzinfo=timezone.utc)
        expected_until = datetime(2026, 4, 23, tzinfo=timezone.utc)
        assert dt_params[0] == expected_since
        assert dt_params[1] == expected_until


def test_since_only_appends_only_lower_bound():
    from utils.conversations.semantic_search import semantic_search_conversations

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock, \
         patch("utils.conversations.semantic_search.db") as db_mock, \
         patch("utils.conversations.semantic_search._decrypt_conversation_data", side_effect=lambda d, uid: d):
        emb_mock.embed_query.return_value = [0.1] * 768
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        semantic_search_conversations(
            uid="james",
            query="x",
            since="2026-04-22T00:00:00Z",
            limit=5,
        )
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]

        assert "c.started_at >= %s" in sql
        assert "c.started_at <= %s" not in sql

        dt_params = [p for p in params if isinstance(p, datetime)]
        assert len(dt_params) == 1
        assert dt_params[0] == datetime(2026, 4, 22, tzinfo=timezone.utc)


def test_decryption_failure_yields_empty_excerpts_not_exception():
    from utils.conversations.semantic_search import semantic_search_conversations

    row = _fake_conversation_row("c1", "T", "O", "raw-ciphertext-never-decrypted")

    def raising_decrypt(data, uid):
        raise RuntimeError("boom")

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock, \
         patch("utils.conversations.semantic_search.db") as db_mock, \
         patch("utils.conversations.semantic_search._decrypt_conversation_data", side_effect=raising_decrypt):
        emb_mock.embed_query.return_value = [0.1] * 768
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [row]

        result = semantic_search_conversations(uid="james", query="x", limit=5)
        assert len(result["hits"]) == 1
        assert result["hits"][0]["excerpts"] == []
        assert result["hits"][0]["title"] == "T"


def test_embed_service_down_raises_runtime_error():
    from utils.conversations.semantic_search import (
        EmbedServiceUnavailable,
        semantic_search_conversations,
    )
    import pytest

    with patch("utils.conversations.semantic_search.embeddings") as emb_mock:
        emb_mock.embed_query.side_effect = RuntimeError("llm-proxy 503")
        with pytest.raises(EmbedServiceUnavailable):
            semantic_search_conversations(uid="james", query="x", limit=5)
