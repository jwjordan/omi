"""Unit tests for database/apps.py — Postgres impl."""

import uuid
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


# ---------------------------------------------------------------------------
# CRUD on apps
# ---------------------------------------------------------------------------


def test_get_app_by_id_db_returns_dict_when_found():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("app1", {"id": "app1", "name": "Test App", "private": False})

        from database.apps import get_app_by_id_db
        result = get_app_by_id_db("app1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT" in sql
        assert "FROM apps" in sql
        assert "WHERE id = %s" in sql
        assert params == ("app1",)
        assert result is not None
        assert result["id"] == "app1"
        assert result["name"] == "Test App"


def test_get_app_by_id_db_returns_none_when_missing():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.apps import get_app_by_id_db
        result = get_app_by_id_db("nonexistent")

        assert result is None


def test_add_app_to_db_inserts_with_id():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import add_app_to_db
        add_app_to_db({"id": "app1", "name": "Test", "private": False})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO apps" in sql
        assert "app1" in params


def test_upsert_app_to_db_uses_on_conflict():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import upsert_app_to_db
        upsert_app_to_db({"id": "app1", "name": "Upserted"})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO apps" in sql
        assert "ON CONFLICT" in sql
        assert "app1" in params


def test_update_app_in_db_merges_data():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import update_app_in_db
        update_app_in_db({"id": "app1", "name": "Updated"})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE apps" in sql
        assert "data = data ||" in sql
        assert "WHERE id = %s" in sql
        assert "app1" in params


def test_delete_app_from_db_removes_row():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import delete_app_from_db
        delete_app_from_db("app1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM apps" in sql
        assert "WHERE id = %s" in sql
        assert params == ("app1",)


# ---------------------------------------------------------------------------
# Filter queries
# ---------------------------------------------------------------------------


def test_get_public_approved_apps_db_filters_by_approved_and_public():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [("app1", {"id": "app1", "approved": True, "private": False})]

        from database.apps import get_public_approved_apps_db
        result = get_public_approved_apps_db()

        sql = cur.execute.call_args.args[0]
        assert "FROM apps" in sql
        assert "approved" in sql
        assert "private" in sql
        assert len(result) == 1
        assert result[0]["id"] == "app1"


def test_get_popular_apps_db_filters_by_approved_and_is_popular():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.apps import get_popular_apps_db
        get_popular_apps_db()

        sql = cur.execute.call_args.args[0]
        assert "FROM apps" in sql
        assert "approved" in sql
        assert "is_popular" in sql


def test_get_private_apps_db_filters_by_uid_and_private():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.apps import get_private_apps_db
        get_private_apps_db("user123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM apps" in sql
        assert "data->>'uid'" in sql
        assert "private" in sql
        assert "user123" in params


def test_search_apps_db_my_apps_filters_by_uid():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.apps import search_apps_db
        search_apps_db("user123", my_apps=True)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM apps" in sql
        assert "data->>'uid'" in sql
        assert "user123" in params


def test_search_apps_db_default_returns_public_approved():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.apps import search_apps_db
        search_apps_db("user123", category="productivity")

        sql = cur.execute.call_args.args[0]
        assert "approved" in sql
        assert "private" in sql
        assert "category" in sql


def test_get_audio_apps_count_returns_count():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (3,)

        from database.apps import get_audio_apps_count
        result = get_audio_apps_count(["app1", "app2", "app3"])

        assert result == 3


def test_get_audio_apps_count_empty_returns_zero():
    from database.apps import get_audio_apps_count
    assert get_audio_apps_count([]) == 0


# ---------------------------------------------------------------------------
# Testers
# ---------------------------------------------------------------------------


def test_add_tester_db_inserts_with_conflict():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import add_tester_db
        add_tester_db({"uid": "user123", "apps": ["app1"]})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO testers" in sql
        assert "ON CONFLICT" in sql
        assert "user123" in params


def test_is_tester_db_returns_true_when_found():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)

        from database.apps import is_tester_db
        assert is_tester_db("user123") is True


def test_is_tester_db_returns_false_when_missing():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.apps import is_tester_db
        assert is_tester_db("user123") is False


def test_can_tester_access_app_db_true_when_in_apps():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"uid": "user123", "apps": ["app1", "app2"]},)

        from database.apps import can_tester_access_app_db
        assert can_tester_access_app_db("app1", "user123") is True


def test_can_tester_access_app_db_false_when_not_in_apps():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"uid": "user123", "apps": ["app2"]},)

        from database.apps import can_tester_access_app_db
        assert can_tester_access_app_db("app1", "user123") is False


def test_add_app_access_for_tester_db_updates_apps_array():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import add_app_access_for_tester_db
        add_app_access_for_tester_db("app1", "user123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE testers" in sql
        assert "user123" in params


# ---------------------------------------------------------------------------
# Usage history + reviews
# ---------------------------------------------------------------------------


def test_record_app_usage_inserts_row():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import record_app_usage
        from models.app import UsageHistoryType
        data = record_app_usage(
            uid="user123",
            app_id="app1",
            usage_type=UsageHistoryType.memory_created_external_integration,
            conversation_id="conv1",
        )

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO app_usage_history" in sql
        assert "app1" in params
        assert "user123" in params
        assert data["uid"] == "user123"


def test_record_app_usage_requires_conversation_or_message_id():
    from database.apps import record_app_usage
    from models.app import UsageHistoryType

    try:
        record_app_usage(
            uid="user123",
            app_id="app1",
            usage_type=UsageHistoryType.memory_created_external_integration,
        )
        assert False, "should have raised"
    except ValueError:
        pass


def test_get_app_usage_count_db_returns_count():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (7,)

        from database.apps import get_app_usage_count_db
        result = get_app_usage_count_db("app1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "COUNT" in sql.upper()
        assert "FROM app_usage_history" in sql
        assert "app_id" in sql
        assert "app1" in params
        assert result == 7


def test_set_app_review_in_db_upserts():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.apps import set_app_review_in_db
        set_app_review_in_db("app1", "user123", {"rating": 5, "comment": "great"})

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO app_reviews" in sql
        assert "ON CONFLICT" in sql
        assert "app1" in params
        assert "user123" in params


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


def test_get_persona_by_username_db_filters_by_username_and_capability():
    with patch("database.apps.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("app1", {"username": "bob", "capabilities": ["persona"]})

        from database.apps import get_persona_by_username_db
        result = get_persona_by_username_db("bob")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM apps" in sql
        assert "username" in sql
        assert "capabilities" in sql
        assert "bob" in params
        assert result is not None
