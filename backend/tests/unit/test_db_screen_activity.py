"""Unit tests for database/screen_activity.py — Postgres impl."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import json


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


def test_upsert_screen_activity_inserts_rows_via_batch():
    """Test that upsert_screen_activity uses db.batch() and returns count."""
    with patch("database.screen_activity.db") as db_mock:
        batch_conn, batch_cur = _mock_conn()
        batch_cur.rowcount = 2
        db_mock.batch.return_value = batch_conn

        from database.screen_activity import upsert_screen_activity

        rows = [
            {
                "id": "screen1",
                "timestamp": "2026-04-21 10:30:00.000",
                "appName": "Chrome",
                "windowTitle": "GitHub",
                "ocrText": "some text",
            },
            {
                "id": "screen2",
                "timestamp": "2026-04-21 10:31:00.000",
                "appName": "Safari",
                "windowTitle": "Claude",
                "ocrText": "more text",
            },
        ]
        result = upsert_screen_activity("user123", rows)

        # Should use db.batch()
        db_mock.batch.assert_called()

        # Should return count of rows
        assert result == 2

        # Should execute INSERT ... ON CONFLICT
        sql = batch_cur.execute.call_args.args[0]
        assert "INSERT INTO screen_activity" in sql
        assert "ON CONFLICT" in sql


def test_upsert_screen_activity_returns_zero_for_empty_rows():
    """Test that upsert_screen_activity returns 0 for empty list."""
    with patch("database.screen_activity.db") as db_mock:
        from database.screen_activity import upsert_screen_activity

        result = upsert_screen_activity("user123", [])
        assert result == 0
        db_mock.batch.assert_not_called()


def test_get_screen_activity_filters_by_uid():
    """Test that get_screen_activity queries screen_activity by uid."""
    with patch("database.screen_activity.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        row_data = {
            "appName": "Chrome",
            "windowTitle": "GitHub",
            "ocrText": "some text",
            "timestamp": "2026-04-21 10:30:00.000",
        }
        cur.fetchall.return_value = [("screen1", row_data)]

        from database.screen_activity import get_screen_activity

        result = get_screen_activity("user123", limit=100)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT" in sql
        assert "FROM screen_activity" in sql
        assert "WHERE uid = %s" in sql
        assert params[0] == "user123"
        assert len(result) == 1
        assert result[0]["id"] == "screen1"
        assert result[0]["appName"] == "Chrome"


def test_get_screen_activity_filters_by_start_date():
    """Test that get_screen_activity respects start_date filter."""
    with patch("database.screen_activity.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.screen_activity import get_screen_activity

        start = datetime(2026, 4, 21, 10, 0, 0, tzinfo=timezone.utc)
        get_screen_activity("user123", start_date=start)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "created_at" in sql or "timestamp" in sql
        # Should have start date in params
        assert len(params) >= 2


def test_get_screen_activity_filters_by_end_date():
    """Test that get_screen_activity respects end_date filter."""
    with patch("database.screen_activity.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.screen_activity import get_screen_activity

        end = datetime(2026, 4, 21, 23, 59, 59, tzinfo=timezone.utc)
        get_screen_activity("user123", end_date=end)

        sql = cur.execute.call_args.args[0]
        assert "created_at" in sql or "timestamp" in sql


def test_get_screen_activity_filters_by_app_filter():
    """Test that get_screen_activity respects app_filter."""
    with patch("database.screen_activity.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.screen_activity import get_screen_activity

        get_screen_activity("user123", app_filter="Chrome")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "appName" in sql or "data" in sql
        assert "Chrome" in params


def test_get_screen_activity_respects_limit():
    """Test that get_screen_activity respects limit parameter."""
    with patch("database.screen_activity.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.screen_activity import get_screen_activity

        get_screen_activity("user123", limit=50)

        sql = cur.execute.call_args.args[0]
        assert "LIMIT" in sql


def test_get_screen_activity_summary_aggregates_apps():
    """Test that get_screen_activity_summary groups by appName and counts."""
    with patch("database.screen_activity.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # Mock return data for get_screen_activity
        rows_data = [
            {
                "appName": "Chrome",
                "windowTitle": "GitHub",
                "timestamp": "2026-04-21 10:30:00.000",
                "ocrText": "code",
            },
            {
                "appName": "Chrome",
                "windowTitle": "Claude",
                "timestamp": "2026-04-21 10:31:00.000",
                "ocrText": "chat",
            },
            {
                "appName": "Safari",
                "windowTitle": "News",
                "timestamp": "2026-04-21 10:32:00.000",
                "ocrText": "news",
            },
        ]
        cur.fetchall.return_value = [
            ("screen1", rows_data[0]),
            ("screen2", rows_data[1]),
            ("screen3", rows_data[2]),
        ]

        from database.screen_activity import get_screen_activity_summary

        result = get_screen_activity_summary("user123")

        assert "apps" in result
        assert "total_screenshots" in result
        assert result["total_screenshots"] == 3
        assert "Chrome" in result["apps"]
        assert "Safari" in result["apps"]
        assert result["apps"]["Chrome"]["count"] == 2
        assert result["apps"]["Safari"]["count"] == 1


def test_get_screen_activity_summary_returns_empty_when_no_rows():
    """Test that get_screen_activity_summary returns empty structure for no data."""
    with patch("database.screen_activity.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.screen_activity import get_screen_activity_summary

        result = get_screen_activity_summary("user123")

        assert result["total_screenshots"] == 0
        assert result["apps"] == {}
