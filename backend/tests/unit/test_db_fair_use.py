"""Unit tests for database/fair_use.py — Postgres impl."""

import json
from datetime import datetime, timedelta
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


def test_get_fair_use_state_returns_dict_when_found():
    """Happy path: get_fair_use_state returns the stored data dict."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        state_data = {"stage": "warning", "violation_count_7d": 2}
        cur.fetchone.return_value = (state_data,)

        from database.fair_use import get_fair_use_state

        result = get_fair_use_state("user123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT data FROM fair_use_state" in sql
        assert "WHERE uid = %s" in sql
        assert params == ("user123",)
        assert result == state_data


def test_get_fair_use_state_returns_empty_dict_when_missing():
    """get_fair_use_state returns {} if no row exists."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.fair_use import get_fair_use_state

        result = get_fair_use_state("user123")

        assert result == {}


def test_update_fair_use_state_inserts_or_updates():
    """update_fair_use_state uses UPSERT pattern."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.fair_use import update_fair_use_state

        updates = {"stage": "throttle", "violation_count_7d": 5}
        update_fair_use_state("user123", updates)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO fair_use_state" in sql
        assert "ON CONFLICT" in sql
        assert "||" in sql  # JSONB merge operator
        assert "user123" in params


def test_set_fair_use_stage_delegates_to_update():
    """set_fair_use_stage builds a dict and calls update_fair_use_state."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.fair_use import set_fair_use_stage

        set_fair_use_stage("user123", "restrict", extra_field="value")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO fair_use_state" in sql
        assert "user123" in params
        # stage and extra_field should be in the JSONB payload
        assert any("restrict" in str(p) for p in params)


def test_create_fair_use_event_returns_event_id():
    """create_fair_use_event inserts and returns the generated event_id."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # Simulate the INSERT...RETURNING id behavior
        cur.fetchone.return_value = ("evt-uuid-123",)

        from database.fair_use import create_fair_use_event

        event_data = {"violation_type": "rate_limit"}
        event_id = create_fair_use_event("user123", event_data)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO fair_use_events" in sql
        assert "user123" in params
        assert "RETURNING id" in sql
        assert event_id == "evt-uuid-123"


def test_get_fair_use_events_returns_list_ordered_by_created_at():
    """get_fair_use_events returns list of events sorted newest first."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        event1 = {"id": "evt1", "resolved": False, "violation_type": "rate_limit"}
        event2 = {"id": "evt2", "resolved": True, "violation_type": "upload_limit"}
        cur.fetchall.return_value = [(event1["id"], event1["resolved"], event1),
                                      (event2["id"], event2["resolved"], event2)]

        from database.fair_use import get_fair_use_events

        result = get_fair_use_events("user123", limit=50)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT id, resolved, data FROM fair_use_events" in sql
        assert "WHERE uid = %s" in sql
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT %s" in sql
        assert params == ("user123", 50)
        assert len(result) == 2


def test_get_violation_counts_counts_unresolved_events():
    """get_violation_counts returns 7d and 30d counts."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # Return a single row with counts
        cur.fetchone.return_value = (3, 8)

        from database.fair_use import get_violation_counts

        result = get_violation_counts("user123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM fair_use_events" in sql
        assert "WHERE uid = %s" in sql
        assert "NOT resolved" in sql
        assert "user123" in params
        assert result["violation_count_7d"] == 3
        assert result["violation_count_30d"] == 8


def test_resolve_fair_use_event_updates_event():
    """resolve_fair_use_event marks event as resolved and stores metadata."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.fair_use import resolve_fair_use_event

        resolve_fair_use_event("user123", "evt1", "admin456", "Case dismissed")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE fair_use_events" in sql
        assert "SET resolved = TRUE" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert "user123" in params
        assert "evt1" in params
        assert "admin456" in params
        assert "Case dismissed" in params


def test_reset_fair_use_state_uses_batch_to_delete_both():
    """reset_fair_use_state uses db.batch() to atomically clear state and events."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        # Mock db.batch() to return the connection
        batch_ctx = MagicMock()
        batch_ctx.__enter__ = MagicMock(return_value=conn)
        batch_ctx.__exit__ = MagicMock(return_value=None)
        db_mock.batch.return_value = batch_ctx

        from database.fair_use import reset_fair_use_state

        reset_fair_use_state("user123", "admin456")

        # Verify db.batch() was called
        db_mock.batch.assert_called_once()
        # Two execute calls: one for DELETE fair_use_state, one for DELETE fair_use_events
        assert cur.execute.call_count == 2


def test_get_flagged_users_returns_list_filtered_by_stage():
    """get_flagged_users returns users with active enforcement."""
    with patch("database.fair_use.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        user_data = {"stage": "throttle", "violation_count_7d": 5}
        cur.fetchall.return_value = [("user123", user_data)]

        from database.fair_use import get_flagged_users

        result = get_flagged_users(stage_filter=None, limit=100)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT uid, data FROM fair_use_state" in sql
        assert "ORDER BY updated_at DESC" in sql
        assert "LIMIT %s" in sql
        # Should filter for active stages
        assert any(stage in sql for stage in ["warning", "throttle", "restrict"])


def test_generate_case_ref_returns_fu_format():
    """_generate_case_ref returns FU-{12 hex chars} format."""
    from database.fair_use import _generate_case_ref

    case_ref = _generate_case_ref()

    assert case_ref.startswith("FU-")
    assert len(case_ref) == 15  # FU- + 12 hex chars
    # hex chars are 0-9 and A-F
    hex_part = case_ref[3:]
    assert all(c in "0123456789ABCDEF" for c in hex_part)
