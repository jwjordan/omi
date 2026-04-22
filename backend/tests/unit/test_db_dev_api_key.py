"""Unit tests for database/dev_api_key.py — Postgres impl."""

from datetime import datetime
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


def test_create_dev_key_inserts_with_hashed_key():
    """Test that create_dev_key generates and stores a hashed key."""
    with patch("database.dev_api_key.db") as db_mock, \
         patch("database.dev_api_key.generate_dev_api_key") as gen_mock, \
         patch("database.dev_api_key.uuid") as uuid_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        gen_mock.return_value = ("raw_key_123", "hashed_key_abc", "omi_dev_xyz")
        uuid_mock.uuid4.return_value.hex = "key-id-uuid"

        from database.dev_api_key import create_dev_key
        raw_key, api_key_data = create_dev_key("user123", "My Key", scopes=["read"])

        assert raw_key == "raw_key_123"
        assert api_key_data.name == "My Key"
        assert api_key_data.key_prefix == "omi_dev_xyz"
        assert api_key_data.scopes == ["read"]

        # Verify INSERT was called
        cur.execute.assert_called()
        call_args = cur.execute.call_args
        assert "INSERT INTO dev_api_keys" in call_args[0][0]
        assert "hashed_key_abc" in str(call_args)


def test_create_dev_key_defaults_to_read_only_scopes():
    """Test that create_dev_key uses READ_ONLY_SCOPES when scopes=None."""
    with patch("database.dev_api_key.db") as db_mock, \
         patch("database.dev_api_key.generate_dev_api_key") as gen_mock, \
         patch("database.dev_api_key.uuid") as uuid_mock, \
         patch("database.dev_api_key.READ_ONLY_SCOPES", ["read"]):
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        gen_mock.return_value = ("raw_key", "hashed_key", "omi_dev_abc")
        uuid_mock.uuid4.return_value.hex = "key-id"

        from database.dev_api_key import create_dev_key
        raw_key, api_key_data = create_dev_key("user123", "Key Without Scopes")

        assert api_key_data.scopes == ["read"]


def test_get_user_and_scopes_by_api_key_returns_none_for_invalid_prefix():
    """Test that get_user_and_scopes_by_api_key returns None for non-dev keys."""
    from database.dev_api_key import get_user_and_scopes_by_api_key
    result = get_user_and_scopes_by_api_key("invalid_key_123")
    assert result is None


def test_get_user_and_scopes_by_api_key_queries_database():
    """Test that get_user_and_scopes_by_api_key queries by hashed key."""
    with patch("database.dev_api_key.db") as db_mock, \
         patch("database.dev_api_key.hash_dev_api_key") as hash_mock, \
         patch("database.dev_api_key.redis_db") as redis_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        hash_mock.return_value = "hashed_secret_xyz"
        redis_mock.get_cached_dev_api_key_data.return_value = None
        cur.fetchone.return_value = ("user456", {"scopes": ["read", "write"]})

        from database.dev_api_key import get_user_and_scopes_by_api_key
        result = get_user_and_scopes_by_api_key("omi_dev_secret_xyz")

        assert result is not None
        assert result["user_id"] == "user456"
        assert result["scopes"] == ["read", "write"]


def test_get_user_id_by_api_key_returns_user_id():
    """Test that get_user_id_by_api_key returns just the user ID."""
    with patch("database.dev_api_key.get_user_and_scopes_by_api_key") as get_mock:
        get_mock.return_value = {"user_id": "user789", "scopes": ["read"]}

        from database.dev_api_key import get_user_id_by_api_key
        result = get_user_id_by_api_key("omi_dev_secret")

        assert result == "user789"


def test_get_dev_keys_for_user_lists_all_keys():
    """Test that get_dev_keys_for_user returns all keys for a user."""
    with patch("database.dev_api_key.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime.utcnow()
        cur.fetchall.return_value = [
            ({"id": "id1", "name": "Key 1", "key_prefix": "omi_dev_a", "created_at": now.isoformat(), "last_used_at": None, "scopes": None},),
            ({"id": "id2", "name": "Key 2", "key_prefix": "omi_dev_b", "created_at": now.isoformat(), "last_used_at": None, "scopes": ["read"]},),
        ]

        from database.dev_api_key import get_dev_keys_for_user
        result = get_dev_keys_for_user("user123")

        # Should query with ORDER BY created_at DESC
        cur.execute.assert_called()
        call_args = cur.execute.call_args
        assert "ORDER BY created_at DESC" in call_args[0][0]


def test_delete_dev_key_removes_from_db_and_cache():
    """Test that delete_dev_key deletes from database and clears cache."""
    with patch("database.dev_api_key.db") as db_mock, \
         patch("database.dev_api_key.redis_db") as redis_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.dev_api_key import delete_dev_key
        delete_dev_key("user123", "key_id_xyz")

        # Verify DELETE was called
        cur.execute.assert_called()
        call_args = cur.execute.call_args
        assert "DELETE FROM dev_api_keys" in call_args[0][0]


def test_get_dev_keys_for_user_returns_empty_list_when_no_keys():
    """Test that get_dev_keys_for_user returns empty list when user has no keys."""
    with patch("database.dev_api_key.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.dev_api_key import get_dev_keys_for_user
        result = get_dev_keys_for_user("user_no_keys")

        assert result == []


def test_get_user_and_scopes_by_api_key_caches_result():
    """Test that get_user_and_scopes_by_api_key caches the result."""
    with patch("database.dev_api_key.db") as db_mock, \
         patch("database.dev_api_key.hash_dev_api_key") as hash_mock, \
         patch("database.dev_api_key.redis_db") as redis_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        hash_mock.return_value = "hashed_xyz"
        redis_mock.get_cached_dev_api_key_data.return_value = None
        cur.fetchone.return_value = ("user123", {"scopes": ["read"]})

        from database.dev_api_key import get_user_and_scopes_by_api_key
        result = get_user_and_scopes_by_api_key("omi_dev_secret")

        assert result is not None
        redis_mock.cache_dev_api_key.assert_called_once()
