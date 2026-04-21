"""Unit tests for database/daily_summaries.py — Postgres impl."""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


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


def test_create_daily_summary_inserts_with_date_typed_column():
    """Test that create_daily_summary inserts uid, id, date, and data columns."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.daily_summaries import create_daily_summary
        summary_data = {
            "id": "summary1",
            "date": "2026-04-21",
            "headline": "Busy day",
            "overview": "Lots of meetings",
        }
        result = create_daily_summary("user123", summary_data)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO daily_summaries" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == "user123"  # uid
        assert params[1] == "summary1"  # id
        assert params[2] == "2026-04-21"  # date
        assert result == "summary1"


def test_get_daily_summary_returns_dict_when_found():
    """Test that get_daily_summary merges typed columns with data JSONB."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        row_data = {"headline": "Busy day", "overview": "Lots of meetings"}
        cur.fetchone.return_value = ("summary1", "2026-04-21", row_data)

        from database.daily_summaries import get_daily_summary
        result = get_daily_summary("user123", "summary1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT" in sql
        assert "FROM daily_summaries" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ("user123", "summary1")
        assert result is not None
        assert result["id"] == "summary1"
        assert result["date"] == "2026-04-21"
        assert result["headline"] == "Busy day"


def test_get_daily_summary_returns_none_when_not_found():
    """Test that get_daily_summary returns None if not found."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.daily_summaries import get_daily_summary
        result = get_daily_summary("user123", "nonexistent")

        assert result is None


def test_get_daily_summary_by_date_uses_typed_date_column():
    """Test that get_daily_summary_by_date queries the typed date column."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        row_data = {"headline": "Busy day", "overview": "Lots of meetings"}
        cur.fetchone.return_value = ("summary1", "2026-04-21", row_data)

        from database.daily_summaries import get_daily_summary_by_date
        result = get_daily_summary_by_date("user123", "2026-04-21")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT" in sql
        assert "WHERE uid = %s AND date = %s" in sql
        assert "LIMIT 1" in sql
        assert params == ("user123", "2026-04-21")
        assert result is not None
        assert result["date"] == "2026-04-21"


def test_get_daily_summary_by_date_returns_none_when_not_found():
    """Test that get_daily_summary_by_date returns None if not found."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.daily_summaries import get_daily_summary_by_date
        result = get_daily_summary_by_date("user123", "2026-04-21")

        assert result is None


def test_get_daily_summaries_orders_by_date_descending():
    """Test that get_daily_summaries returns summaries ordered by date DESC."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        row1 = ("summary2", "2026-04-21", {"headline": "Recent"})
        row2 = ("summary1", "2026-04-20", {"headline": "Older"})
        cur.fetchall.return_value = [row1, row2]

        from database.daily_summaries import get_daily_summaries
        result = get_daily_summaries("user123", limit=10, offset=0)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "ORDER BY date DESC" in sql
        assert "LIMIT %s" in sql
        assert "OFFSET %s" in sql
        assert params[0] == "user123"
        assert params[1] == 10  # limit
        assert params[2] == 0  # offset
        assert len(result) == 2
        assert result[0]["id"] == "summary2"
        assert result[1]["id"] == "summary1"


def test_get_daily_summaries_respects_start_date_filter():
    """Test that get_daily_summaries filters by start_date."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.daily_summaries import get_daily_summaries
        get_daily_summaries("user123", start_date="2026-04-20")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "date >= %s" in sql
        assert "2026-04-20" in params


def test_get_daily_summaries_respects_end_date_filter():
    """Test that get_daily_summaries filters by end_date."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.daily_summaries import get_daily_summaries
        get_daily_summaries("user123", end_date="2026-04-21")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "date <= %s" in sql
        assert "2026-04-21" in params


def test_delete_daily_summary_removes_row():
    """Test that delete_daily_summary removes a row and returns True."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.daily_summaries import delete_daily_summary
        result = delete_daily_summary("user123", "summary1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM daily_summaries" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ("user123", "summary1")
        assert result is True


def test_delete_daily_summary_returns_false_when_not_found():
    """Test that delete_daily_summary returns False if not found."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.daily_summaries import delete_daily_summary
        result = delete_daily_summary("user123", "nonexistent")

        assert result is False


def test_get_summaries_count_returns_count_for_user():
    """Test that get_summaries_count returns the count of summaries for a user."""
    with patch("database.daily_summaries.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (5,)

        from database.daily_summaries import get_summaries_count
        result = get_summaries_count("user123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT COUNT(*)" in sql
        assert "FROM daily_summaries" in sql
        assert "WHERE uid = %s" in sql
        assert params == ("user123",)
        assert result == 5
