"""Unit tests for database/wrapped.py — Postgres impl."""

import json
from datetime import datetime, timedelta, timezone
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


def test_get_wrapped_returns_dict_when_found():
    """Test that get_wrapped merges typed columns with data JSONB."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        row_data = {
            "year": 2026,
            "status": "processing",
            "started_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
            "schema_version": 1,
        }
        cur.fetchone.return_value = (row_data,)

        from database.wrapped import get_wrapped
        result = get_wrapped("user123", 2026)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT data FROM wrapped" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ("user123", "2026")
        assert result is not None
        assert result["year"] == 2026
        assert result["status"] == "processing"


def test_get_wrapped_returns_none_when_not_found():
    """Test that get_wrapped returns None if not found."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.wrapped import get_wrapped
        result = get_wrapped("user123", 2026)

        assert result is None


def test_create_wrapped_inserts_with_initial_state():
    """Test that create_wrapped inserts with initial processing status."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.wrapped import create_wrapped
        result = create_wrapped("user123", 2026)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO wrapped" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == "user123"  # uid
        assert params[1] == "2026"  # id (year as string)
        # Verify the inserted data contains required fields
        assert result is not None
        assert result["year"] == 2026
        assert result["status"] == "processing"
        assert result["started_at"] is not None
        assert result["updated_at"] is not None
        assert result["completed_at"] is None
        assert result["result"] is None
        assert result["error"] is None
        assert result["schema_version"] == 1


def test_update_wrapped_status_sets_done():
    """Test that update_wrapped_status updates status and completed_at when status=done."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.wrapped import update_wrapped_status, WrappedStatus
        result_payload = {"summary": "test", "stats": {}}
        result = update_wrapped_status("user123", 2026, WrappedStatus.DONE, result=result_payload)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE wrapped" in sql
        assert "data = data ||" in sql
        assert "WHERE uid=%s AND id=%s" in sql
        # params[0] is the jsonb dict (as string), params[1] is uid, params[2] is id
        assert params[1] == "user123"  # uid
        assert params[2] == "2026"  # id
        # Verify the update dict contains done status
        import json
        update_data = json.loads(params[0])
        assert update_data["status"] == WrappedStatus.DONE
        assert result is True


def test_update_wrapped_status_sets_error():
    """Test that update_wrapped_status updates status and error when status=error."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.wrapped import update_wrapped_status, WrappedStatus
        result = update_wrapped_status("user123", 2026, WrappedStatus.ERROR, error="Test error message")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE wrapped" in sql
        assert params[1] == "user123"  # uid
        assert params[2] == "2026"  # id
        # Verify the update dict contains error status
        import json
        update_data = json.loads(params[0])
        assert update_data["status"] == WrappedStatus.ERROR
        assert update_data["error"] == "Test error message"
        assert result is True


def test_update_wrapped_status_returns_false_when_not_found():
    """Test that update_wrapped_status returns False when row doesn't exist."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.wrapped import update_wrapped_status, WrappedStatus
        result = update_wrapped_status("user123", 2026, WrappedStatus.PROCESSING)

        assert result is False


def test_update_wrapped_progress_merges_progress():
    """Test that update_wrapped_progress merges progress dict into data."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.wrapped import update_wrapped_progress
        progress = {"step": "computing_stats", "pct": 0.5}
        result = update_wrapped_progress("user123", 2026, progress)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE wrapped" in sql
        assert "data = data ||" in sql
        assert "WHERE uid=%s AND id=%s" in sql
        # Verify uid and id are in correct positions (1 and 2)
        assert params[1] == "user123"  # uid
        assert params[2] == "2026"  # id
        # Verify progress is merged into the update dict
        import json
        update_data = json.loads(params[0])
        assert update_data["progress"] == progress
        assert result is True


def test_update_wrapped_progress_returns_false_when_not_found():
    """Test that update_wrapped_progress returns False when row doesn't exist."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.wrapped import update_wrapped_progress
        progress = {"step": "computing_stats", "pct": 0.5}
        result = update_wrapped_progress("user123", 2026, progress)

        assert result is False


def test_reset_wrapped_for_regeneration_overwrites_with_reset_state():
    """Test that reset_wrapped_for_regeneration resets to processing state."""
    with patch("database.wrapped.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.wrapped import reset_wrapped_for_regeneration
        result = reset_wrapped_for_regeneration("user123", 2026)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO wrapped" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == "user123"  # uid
        assert params[1] == "2026"  # id
        # Verify returned data is in reset state
        assert result is not None
        assert result["status"] == "processing"
        assert result["completed_at"] is None
        assert result["result"] is None
        assert result["error"] is None
        assert result["progress"] is None


def test_is_wrapped_stuck_returns_false_for_non_processing():
    """Test that is_wrapped_stuck returns False when status is not processing."""
    from database.wrapped import is_wrapped_stuck

    wrapped_data = {
        "status": "done",
        "updated_at": datetime.now(timezone.utc),
    }
    assert is_wrapped_stuck(wrapped_data) is False


def test_is_wrapped_stuck_returns_true_when_no_updated_at():
    """Test that is_wrapped_stuck returns True when updated_at is missing."""
    from database.wrapped import is_wrapped_stuck

    wrapped_data = {
        "status": "processing",
    }
    assert is_wrapped_stuck(wrapped_data) is True


def test_is_wrapped_stuck_returns_true_when_stale():
    """Test that is_wrapped_stuck returns True when elapsed > stale_minutes."""
    from database.wrapped import is_wrapped_stuck

    now = datetime.now(timezone.utc)
    old_time = now - timedelta(minutes=20)
    wrapped_data = {
        "status": "processing",
        "updated_at": old_time,
    }
    assert is_wrapped_stuck(wrapped_data, stale_minutes=15) is True


def test_is_wrapped_stuck_returns_false_when_fresh():
    """Test that is_wrapped_stuck returns False when recently updated."""
    from database.wrapped import is_wrapped_stuck

    now = datetime.now(timezone.utc)
    recent_time = now - timedelta(minutes=5)
    wrapped_data = {
        "status": "processing",
        "updated_at": recent_time,
    }
    assert is_wrapped_stuck(wrapped_data, stale_minutes=15) is False
