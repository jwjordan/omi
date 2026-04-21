"""Unit tests for database/focus_sessions.py — Postgres impl."""

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


def test_create_focus_session_inserts_row_with_generated_uuid():
    with patch("database.focus_sessions.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.focus_sessions import create_focus_session
        result = create_focus_session(
            "user123",
            "focused",
            "VSCode",
            "Coding",
            duration_seconds=600,
            message="Great session"
        )

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO focus_sessions" in sql
        assert params[0] == "user123"  # uid
        assert "focused" in str(params)  # Check in the whole params
        assert "VSCode" in str(params)
        assert "Coding" in str(params)
        assert result["status"] == "focused"
        assert result["app_or_site"] == "VSCode"
        assert result["description"] == "Coding"


def test_get_focus_sessions_returns_list_ordered_by_created_at_desc():
    with patch("database.focus_sessions.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # Fake rows: (uid, id, created_at, data_jsonb)
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ("user123", "session-1", now, {
                "status": "focused",
                "app_or_site": "VSCode",
                "description": "Coding",
                "duration_seconds": 600
            }),
            ("user123", "session-2", now, {
                "status": "distracted",
                "app_or_site": "Twitter",
                "description": "Scrolling",
                "duration_seconds": 300
            })
        ]

        from database.focus_sessions import get_focus_sessions
        result = get_focus_sessions("user123", limit=10, offset=0)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM focus_sessions" in sql
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT 10" in sql
        assert "OFFSET 0" in sql
        assert "user123" in params
        assert len(result) == 2
        assert result[0]["id"] == "session-1"
        assert result[0]["status"] == "focused"


def test_get_focus_sessions_filters_by_date_when_provided():
    with patch("database.focus_sessions.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.focus_sessions import get_focus_sessions
        get_focus_sessions("user123", date="2024-01-15", limit=100, offset=0)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "WHERE uid = %s" in sql
        assert "data->>'date'" in sql or "data" in sql
        assert "2024-01-15" in params


def test_delete_focus_session_removes_row_and_returns_true():
    with patch("database.focus_sessions.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.focus_sessions import delete_focus_session
        result = delete_focus_session("user123", "session-1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM focus_sessions" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ("user123", "session-1")
        assert result is True


def test_delete_focus_session_returns_false_when_not_found():
    with patch("database.focus_sessions.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.focus_sessions import delete_focus_session
        result = delete_focus_session("user123", "nonexistent")

        assert result is False


def test_get_focus_stats_aggregates_session_data():
    with patch("database.focus_sessions.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # Fake rows for aggregation
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ("user123", "session-1", now, {
                "status": "focused",
                "app_or_site": "VSCode",
                "duration_seconds": 600
            }),
            ("user123", "session-2", now, {
                "status": "focused",
                "app_or_site": "VSCode",
                "duration_seconds": 900
            }),
            ("user123", "session-3", now, {
                "status": "distracted",
                "app_or_site": "Twitter",
                "duration_seconds": 300
            }),
            ("user123", "session-4", now, {
                "status": "distracted",
                "app_or_site": "Twitter",
                "duration_seconds": 300
            })
        ]

        from database.focus_sessions import get_focus_stats
        result = get_focus_stats("user123")

        assert result["focused_count"] == 2
        assert result["distracted_count"] == 2
        assert result["focused_minutes"] == 25  # (600 + 900) / 60
        assert result["distracted_minutes"] == 10  # (300 + 300) / 60
        assert result["session_count"] == 4
        assert len(result["top_distractions"]) > 0
