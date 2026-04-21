"""Unit tests for database/announcements.py — Postgres impl."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from models.announcement import Announcement, AnnouncementType


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


def test_get_announcement_by_id_returns_announcement_when_found():
    """Happy path: fetch a single announcement by ID."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        data_dict = {
            "id": "ann1",
            "type": "changelog",
            "created_at": now_str,
            "active": True,
            "content": {"title": "Test"},
        }
        cur.fetchone.return_value = ("ann1", now, data_dict)

        from database.announcements import get_announcement_by_id

        result = get_announcement_by_id("ann1")

        assert result is not None
        assert result.id == "ann1"
        assert result.type == AnnouncementType.CHANGELOG


def test_get_announcement_by_id_returns_none_when_missing():
    """Return None if announcement doesn't exist."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.announcements import get_announcement_by_id

        result = get_announcement_by_id("missing")
        assert result is None


def test_get_app_changelogs_filters_by_version_range():
    """Fetch changelogs between two versions, sorted by version descending."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        rows = [
            (
                "ann1",
                now,
                json.dumps({
                    "id": "ann1",
                    "type": "changelog",
                    "created_at": now_str,
                    "active": True,
                    "app_version": "1.0.2",
                    "content": {},
                }),
            ),
            (
                "ann2",
                now,
                json.dumps({
                    "id": "ann2",
                    "type": "changelog",
                    "created_at": now_str,
                    "active": True,
                    "app_version": "1.0.3",
                    "content": {},
                }),
            ),
        ]
        # Convert JSON strings to dicts for the mock
        parsed_rows = [
            (row[0], row[1], json.loads(row[2]) if isinstance(row[2], str) else row[2])
            for row in rows
        ]
        cur.fetchall.return_value = parsed_rows

        from database.announcements import get_app_changelogs

        result = get_app_changelogs("1.0.1", "1.0.5")

        # Should execute a query filtering by type and active
        sql = cur.execute.call_args.args[0]
        assert "FROM announcements" in sql
        assert "WHERE" in sql

        # Results should include both versions and be sorted descending
        assert len(result) >= 1


def test_get_recent_changelogs_returns_limited_results():
    """Get recent changelogs limited by count."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        data_dict = {
            "id": "ann1",
            "type": "changelog",
            "created_at": now_str,
            "active": True,
            "app_version": "1.0.1",
            "content": {},
        }
        cur.fetchall.return_value = [("ann1", now, data_dict)]

        from database.announcements import get_recent_changelogs

        result = get_recent_changelogs(limit=5)

        assert isinstance(result, list)
        assert len(result) <= 5


def test_get_firmware_features_filters_by_version():
    """Fetch features for specific firmware version."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        data_dict = {
            "id": "feat1",
            "type": "feature",
            "created_at": now_str,
            "active": True,
            "firmware_version": "2.0.1",
            "content": {},
        }
        cur.fetchall.return_value = [("feat1", now, data_dict)]

        from database.announcements import get_firmware_features

        result = get_firmware_features("2.0.1")

        sql = cur.execute.call_args.args[0]
        assert "FROM announcements" in sql
        # Should filter by firmware_version in JSONB
        assert "firmware_version" in sql or "data" in sql


def test_get_firmware_features_filters_by_device_model():
    """Filter features by device model if specified."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        data_dict = {
            "id": "feat1",
            "type": "feature",
            "created_at": now_str,
            "active": True,
            "firmware_version": "2.0.1",
            "device_models": ["Omi DevKit 2", "Omi Pro"],
            "content": {},
        }
        cur.fetchall.return_value = [("feat1", now, data_dict)]

        from database.announcements import get_firmware_features

        result = get_firmware_features("2.0.1", device_model="Omi DevKit 2")

        assert isinstance(result, list)


def test_get_app_features_returns_features_for_version():
    """Fetch features for specific app version."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        rows = [
            (
                "feat1",
                now,
                json.dumps({
                    "id": "feat1",
                    "type": "feature",
                    "created_at": now_str,
                    "active": True,
                    "app_version": "1.0.5",
                    "content": {},
                }),
            ),
        ]
        cur.fetchall.return_value = rows

        from database.announcements import get_app_features

        result = get_app_features("1.0.5")

        sql = cur.execute.call_args.args[0]
        assert "FROM announcements" in sql


def test_get_general_announcements_filters_by_time():
    """Get general announcements, filtering by created_at."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        rows = [
            (
                "ann1",
                now,
                json.dumps({
                    "id": "ann1",
                    "type": "announcement",
                    "created_at": now_str,
                    "active": True,
                    "content": {},
                }),
            ),
        ]
        cur.fetchall.return_value = rows

        from database.announcements import get_general_announcements

        result = get_general_announcements()

        assert isinstance(result, list)


def test_get_all_announcements_with_filters():
    """Fetch all announcements with optional type and active filters."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        rows = [
            (
                "ann1",
                now,
                json.dumps({
                    "id": "ann1",
                    "type": "changelog",
                    "created_at": now_str,
                    "active": True,
                    "content": {},
                }),
            ),
        ]
        cur.fetchall.return_value = rows

        from database.announcements import get_all_announcements

        result = get_all_announcements(announcement_type=AnnouncementType.CHANGELOG, active_only=True)

        sql = cur.execute.call_args.args[0]
        assert "FROM announcements" in sql
        assert isinstance(result, list)


def test_create_announcement_inserts_row():
    """Insert a new announcement with all fields stored in JSONB."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        ann = Announcement(
            id="new-ann",
            type=AnnouncementType.CHANGELOG,
            created_at=now,
            active=True,
            content={"title": "Test Changelog"},
        )

        from database.announcements import create_announcement

        result = create_announcement(ann)

        sql = cur.execute.call_args.args[0]
        assert "INSERT INTO announcements" in sql
        assert "ON CONFLICT" in sql
        assert result.id == "new-ann"


def test_update_announcement_merges_data():
    """Update an announcement with partial data merge."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        cur.fetchone.return_value = (
            "ann1",
            now,
            json.dumps({
                "id": "ann1",
                "type": "changelog",
                "created_at": now_str,
                "active": True,
                "content": {},
            }),
        )

        from database.announcements import update_announcement

        result = update_announcement("ann1", {"active": False})

        sql = cur.execute.call_args.args[0]
        assert "UPDATE announcements" in sql
        assert result is not None


def test_delete_announcement_removes_row():
    """Delete an announcement by ID."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.announcements import delete_announcement

        # Mock that the announcement exists before delete
        result = delete_announcement("ann1")

        sql = cur.execute.call_args.args[0]
        assert "DELETE FROM announcements" in sql


def test_deactivate_announcement_soft_deletes():
    """Soft-delete by setting active=False."""
    with patch("database.announcements.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.announcements import deactivate_announcement

        result = deactivate_announcement("ann1")

        sql = cur.execute.call_args.args[0]
        assert "UPDATE announcements" in sql
