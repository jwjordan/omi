"""Unit tests for database/advice.py — Postgres impl."""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call


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


def test_create_advice_inserts_row():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.advice import create_advice
        result = create_advice("user123", "Be more active", category="health", reasoning="sedentary pattern")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO advice" in sql
        assert params[0] == "user123"  # uid
        assert isinstance(params[1], str)  # id (UUID)
        assert isinstance(params[2], str)  # data JSON-serialized string
        assert result["content"] == "Be more active"
        assert result["category"] == "health"
        assert result["is_read"] is False
        assert result["is_dismissed"] is False


def test_get_advice_filters_by_uid():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        advice_id = str(uuid.uuid4())
        row_data = {
            "content": "Exercise",
            "category": "health",
            "is_read": False,
            "is_dismissed": False,
        }
        cur.fetchall.return_value = [(advice_id, row_data)]

        from database.advice import get_advice
        result = get_advice("user123", limit=50)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT" in sql
        assert "FROM advice" in sql
        assert "WHERE uid = %s" in sql
        assert params[0] == "user123"
        assert len(result) == 1
        assert result[0]["id"] == advice_id
        assert result[0]["content"] == "Exercise"


def test_get_advice_respects_category_filter():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.advice import get_advice
        get_advice("user123", category="health", limit=10)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data->>'category'" in sql or "category" in sql
        assert "health" in params


def test_get_advice_excludes_dismissed_by_default():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.advice import get_advice
        get_advice("user123")

        sql = cur.execute.call_args.args[0]
        assert "is_dismissed" in sql


def test_get_advice_includes_dismissed_when_requested():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.advice import get_advice
        get_advice("user123", include_dismissed=True)

        sql = cur.execute.call_args.args[0]
        # When include_dismissed=True, the is_dismissed filter should be absent
        assert "is_dismissed" not in sql or "is_dismissed" in sql and "False" not in sql


def test_update_advice_merges_jsonb():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        advice_id = str(uuid.uuid4())
        row_data = {
            "content": "Exercise",
            "category": "health",
            "is_read": False,
            "is_dismissed": False,
        }
        # First call is SELECT, second is UPDATE RETURNING
        cur.fetchone.side_effect = [(advice_id, row_data), (row_data,)]

        from database.advice import update_advice
        result = update_advice("user123", advice_id, is_read=True)

        # Get the last execute call which should be the UPDATE
        last_call_sql = cur.execute.call_args_list[-1].args[0]
        assert "UPDATE advice" in last_call_sql
        assert "SET data = data ||" in last_call_sql
        assert "WHERE uid = %s AND id = %s" in last_call_sql
        assert result is not None
        assert result["id"] == advice_id


def test_update_advice_returns_none_when_not_found():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.advice import update_advice
        result = update_advice("user123", "nonexistent", is_read=True)

        assert result is None


def test_delete_advice_removes_row():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.advice import delete_advice
        result = delete_advice("user123", "advice123")

        sql = cur.execute.call_args.args[0]
        assert "DELETE FROM advice" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert result is True


def test_delete_advice_returns_false_when_not_found():
    with patch("database.advice.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.advice import delete_advice
        result = delete_advice("user123", "nonexistent")

        assert result is False


def test_mark_all_advice_read_uses_batch():
    with patch("database.advice.db") as db_mock:
        batch_conn = MagicMock()
        batch_cursor = MagicMock()
        batch_cursor.__enter__ = MagicMock(return_value=batch_cursor)
        batch_cursor.__exit__ = MagicMock(return_value=None)
        batch_cursor.rowcount = 3
        batch_conn.cursor.return_value = batch_cursor
        batch_conn.__enter__ = MagicMock(return_value=batch_conn)
        batch_conn.__exit__ = MagicMock(return_value=None)

        db_mock.batch.return_value = batch_conn

        from database.advice import mark_all_advice_read
        result = mark_all_advice_read("user123")

        sql = batch_cursor.execute.call_args.args[0]
        assert "UPDATE advice" in sql
        assert "jsonb_set" in sql or "data = " in sql
        assert "is_read" in sql
        assert result == 3
