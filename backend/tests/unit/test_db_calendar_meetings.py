"""Unit tests for database/calendar_meetings.py — Postgres impl."""

import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Mock _client before importing calendar_meetings to avoid DB connection attempts
sys.modules["database._client"] = MagicMock()


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


def test_create_meeting_inserts_row_with_generated_id():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("generated-uuid",)

        from database.calendar_meetings import create_meeting

        meeting_data = {"start_time": "2024-01-01T10:00:00Z", "title": "Meeting"}
        result = create_meeting("uid123", meeting_data)

        sql = cur.execute.call_args_list[0].args[0]
        assert "INSERT INTO calendar_meetings" in sql
        assert result == "generated-uuid"


def test_create_meeting_returns_existing_id_if_provided():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("existing-id",)

        from database.calendar_meetings import create_meeting

        meeting_data = {"id": "existing-id", "start_time": "2024-01-01T10:00:00Z"}
        result = create_meeting("uid123", meeting_data)

        assert result == "existing-id"


def test_update_meeting_merges_into_data():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.calendar_meetings import update_meeting

        update_meeting("uid123", "meeting1", {"title": "Updated"})

        sql = cur.execute.call_args.args[0]
        assert "UPDATE calendar_meetings" in sql
        assert "data = data || %s::jsonb" in sql
        assert "WHERE uid=%s AND id=%s" in sql


def test_get_meeting_returns_dict_when_found():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (
            "uid123",
            "meeting1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            {"title": "Meeting", "start_time": "2024-01-01T10:00:00Z"},
        )

        from database.calendar_meetings import get_meeting

        result = get_meeting("uid123", "meeting1")

        assert result is not None
        assert result["id"] == "meeting1"
        assert result["uid"] == "uid123"
        assert result["title"] == "Meeting"


def test_get_meeting_returns_none_when_missing():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.calendar_meetings import get_meeting

        result = get_meeting("uid123", "missing")
        assert result is None


def test_get_meeting_id_by_calendar_event_filters_by_event_id_and_source():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("meeting1",)

        from database.calendar_meetings import get_meeting_id_by_calendar_event

        result = get_meeting_id_by_calendar_event("uid123", "cal-event-456", "google")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data->>'calendar_event_id' = %s" in sql
        assert "data->>'calendar_source' = %s" in sql
        assert "LIMIT 1" in sql
        assert result == "meeting1"


def test_get_meeting_id_by_calendar_event_returns_none_when_not_found():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.calendar_meetings import get_meeting_id_by_calendar_event

        result = get_meeting_id_by_calendar_event("uid123", "missing", "google")
        assert result is None


def test_list_meetings_returns_ordered_results():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("uid123", "m1", datetime(2024, 1, 3, tzinfo=timezone.utc), {"title": "M1"}),
            ("uid123", "m2", datetime(2024, 1, 2, tzinfo=timezone.utc), {"title": "M2"}),
        ]

        from database.calendar_meetings import list_meetings

        result = list_meetings("uid123", limit=50)

        sql = cur.execute.call_args.args[0]
        assert "ORDER BY (data->>'start_time')::timestamptz DESC" in sql
        assert len(result) == 2
        assert result[0]["id"] == "m1"


def test_list_meetings_filters_by_date_range():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.calendar_meetings import list_meetings

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)
        list_meetings("uid123", start_date=start, end_date=end)

        sql = cur.execute.call_args.args[0]
        assert "(data->>'start_time')::timestamptz >= %s" in sql
        assert "(data->>'start_time')::timestamptz <= %s" in sql


def test_delete_meeting_deletes_row():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.calendar_meetings import delete_meeting

        delete_meeting("uid123", "meeting1")

        sql = cur.execute.call_args.args[0]
        assert "DELETE FROM calendar_meetings" in sql
        assert "WHERE uid=%s AND id=%s" in sql


def test_delete_old_meetings_deletes_by_end_time():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 5

        from database.calendar_meetings import delete_old_meetings

        before_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = delete_old_meetings("uid123", before_date)

        sql = cur.execute.call_args.args[0]
        assert "DELETE FROM calendar_meetings" in sql
        assert "(data->>'end_time')::timestamptz < %s" in sql
        assert result == 5


def test_get_meetings_in_time_range_filters_overlaps():
    with patch("database.calendar_meetings.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("uid123", "m1", datetime(2024, 1, 1, tzinfo=timezone.utc), {"title": "M1"}),
        ]

        from database.calendar_meetings import get_meetings_in_time_range

        start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
        result = get_meetings_in_time_range("uid123", start, end)

        sql = cur.execute.call_args.args[0]
        assert "(data->>'start_time')::timestamptz < %s" in sql
        assert "(data->>'end_time')::timestamptz > %s" in sql
        assert "ORDER BY (data->>'start_time')::timestamptz ASC" in sql
        assert len(result) == 1
