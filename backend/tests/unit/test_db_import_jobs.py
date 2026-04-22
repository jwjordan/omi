"""Unit tests for database/import_jobs.py — Postgres impl."""

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


def test_create_import_job_inserts_row():
    with patch("database.import_jobs.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.import_jobs import create_import_job
        job_id = create_import_job({"id": "job1", "uid": "user123", "status": "pending"})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO import_jobs" in sql
        assert "ON CONFLICT" in sql
        assert "job1" in params
        assert job_id == "job1"


def test_update_import_job_merges_data():
    with patch("database.import_jobs.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.import_jobs import update_import_job
        update_import_job("job1", {"status": "completed"})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE import_jobs" in sql
        assert "WHERE id =" in sql
        assert "job1" in params


def test_get_import_job_returns_dict_when_found():
    with patch("database.import_jobs.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # fake a row: (id, uid, data_jsonb)
        cur.fetchone.return_value = ("job1", "user123", {"status": "pending", "extra": "data"})

        from database.import_jobs import get_import_job
        result = get_import_job("job1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM import_jobs" in sql
        assert "WHERE id =" in sql
        assert params == ("job1",)
        assert result is not None
        assert result["id"] == "job1"
        assert result["uid"] == "user123"
        assert result["status"] == "pending"
        assert result["extra"] == "data"


def test_get_import_job_returns_none_when_missing():
    with patch("database.import_jobs.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.import_jobs import get_import_job
        result = get_import_job("missing")
        assert result is None


def test_get_import_jobs_filters_by_uid_and_orders_by_created():
    with patch("database.import_jobs.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("job1", "user123", {"status": "done"}),
            ("job2", "user123", {"status": "pending"}),
        ]

        from database.import_jobs import get_import_jobs
        results = get_import_jobs("user123", limit=50)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        # Verify uid is in WHERE clause
        assert "WHERE uid =" in sql
        # Verify ordering by created_at DESC
        assert "ORDER BY created_at DESC" in sql
        # Verify limit
        assert "LIMIT %s" in sql
        # Verify params: uid and limit
        assert params == ("user123", 50)
        assert len(results) == 2
        assert results[0]["id"] == "job1"
        assert results[1]["id"] == "job2"


def test_delete_import_job_deletes_row():
    with patch("database.import_jobs.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.import_jobs import delete_import_job
        delete_import_job("job1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM import_jobs" in sql
        assert "WHERE id =" in sql
        assert params == ("job1",)
