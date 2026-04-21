"""Unit tests for database/tasks.py — Postgres impl."""

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


def test_create_inserts_row_with_action_and_request_id():
    with patch("database.tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.tasks import create
        create({"id": "task1", "action": "my_action", "request_id": "req-42", "extra": "stuff"})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO tasks" in sql
        assert "ON CONFLICT" in sql
        assert "task1" in params
        assert "my_action" in params
        assert "req-42" in params


def test_update_merges_into_data_and_typed_cols():
    with patch("database.tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.tasks import update
        update("task1", {"status": "done", "action": "updated_action"})

        sql = cur.execute.call_args.args[0]
        assert "UPDATE tasks" in sql
        assert "WHERE id =" in sql


def test_get_task_by_action_request_returns_dict_when_found():
    with patch("database.tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # fake a row: (id, action, request_id, data_jsonb)
        cur.fetchone.return_value = ("task1", "my_action", "req-42", {"status": "pending", "extra": "stuff"})

        from database.tasks import get_task_by_action_request
        result = get_task_by_action_request("my_action", "req-42")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM tasks" in sql
        assert "action = " in sql
        assert "request_id = " in sql
        assert "LIMIT 1" in sql
        assert params == ("my_action", "req-42")
        assert result is not None
        assert result["id"] == "task1"
        assert result["action"] == "my_action"
        assert result["request_id"] == "req-42"
        assert result["status"] == "pending"
        assert result["extra"] == "stuff"


def test_get_task_by_action_request_returns_none_when_missing():
    with patch("database.tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.tasks import get_task_by_action_request
        result = get_task_by_action_request("x", "y")
        assert result is None
