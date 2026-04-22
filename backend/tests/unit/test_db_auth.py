"""Unit tests for database/auth.py — Postgres impl."""

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


def test_get_user_from_uid_returns_dict_when_row_exists():
    """Test that get_user_from_uid reads from users.data and returns fields."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        user_data = {
            "email": "alice@example.com",
            "email_verified": True,
            "phone_number": "+1234567890",
            "display_name": "Alice Smith",
            "photo_url": "https://example.com/photo.jpg",
            "disabled": False,
        }
        cur.fetchone.return_value = (user_data,)

        from database.auth import get_user_from_uid
        result = get_user_from_uid("user123")

        assert result is not None
        assert result["uid"] == "user123"
        assert result["email"] == "alice@example.com"
        assert result["email_verified"] is True
        assert result["phone_number"] == "+1234567890"
        assert result["display_name"] == "Alice Smith"
        assert result["photo_url"] == "https://example.com/photo.jpg"
        assert result["disabled"] is False


def test_get_user_from_uid_returns_none_when_row_missing():
    """Test that get_user_from_uid returns None when user row doesn't exist."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.auth import get_user_from_uid
        result = get_user_from_uid("nonexistent")

        assert result is None


def test_get_user_from_uid_returns_none_when_empty_uid():
    """Test that get_user_from_uid returns None for empty uid."""
    from database.auth import get_user_from_uid
    result = get_user_from_uid("")
    assert result is None

    result = get_user_from_uid(None)
    assert result is None


def test_get_user_from_uid_handles_missing_optional_fields():
    """Test that get_user_from_uid provides defaults for missing fields."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        user_data = {
            "email": "bob@example.com",
            # Missing: email_verified, phone_number, display_name, photo_url, disabled
        }
        cur.fetchone.return_value = (user_data,)

        from database.auth import get_user_from_uid
        result = get_user_from_uid("user456")

        assert result is not None
        assert result["uid"] == "user456"
        assert result["email"] == "bob@example.com"
        assert result["email_verified"] is True  # default
        assert result["phone_number"] is None
        assert result["display_name"] is None
        assert result["photo_url"] is None
        assert result["disabled"] is False  # default


def test_get_user_name_returns_first_word_from_display_name():
    """Test that get_user_name extracts first word from display_name."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        user_data = {
            "display_name": "Charles Anderson",
        }
        cur.fetchone.return_value = (user_data,)

        with patch("database.auth.cache_user_name") as cache_mock:
            from database.auth import get_user_name
            result = get_user_name("user789")

            assert result == "Charles"
            cache_mock.assert_called_once_with("user789", "Charles", ttl=3600)


def test_get_user_name_returns_first_word_from_name_field():
    """Test that get_user_name falls back to name field when display_name is missing."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # First call for get_user_from_uid returns no display_name
        user_data = {"name": "Diana Prince"}

        # We need to mock multiple calls: one in get_user_from_uid, one in _get_user_profile_name
        call_count = [0]

        def fetchone_side_effect():
            call_count[0] += 1
            return (user_data,)

        cur.fetchone.side_effect = fetchone_side_effect

        with patch("database.auth.cache_user_name") as cache_mock:
            from database.auth import get_user_name
            result = get_user_name("user999")

            assert result == "Diana"
            cache_mock.assert_called_once_with("user999", "Diana", ttl=3600)


def test_get_user_name_returns_default_when_row_missing():
    """Test that get_user_name returns 'The User' when use_default=True and no row."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.auth import get_user_name
        result = get_user_name("missing_user", use_default=True)

        assert result == "The User"


def test_get_user_name_returns_none_when_use_default_false():
    """Test that get_user_name returns None when use_default=False and no row."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.auth import get_user_name
        result = get_user_name("missing_user", use_default=False)

        assert result is None


def test_get_user_name_replaces_anonymous_user_with_name():
    """Test that get_user_name replaces 'AnonymousUser' display_name with name field."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        user_data = {
            "display_name": "AnonymousUser",
            "name": "Eve Walker",
        }
        cur.fetchone.return_value = (user_data,)

        with patch("database.auth.cache_user_name") as cache_mock:
            from database.auth import get_user_name
            result = get_user_name("anon_user")

            assert result == "Eve"
            cache_mock.assert_called_once_with("anon_user", "Eve", ttl=3600)


def test_get_user_name_replaces_anonymous_user_with_default():
    """Test that get_user_name uses default when display_name is 'AnonymousUser' and no name."""
    with patch("database.auth.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        user_data = {
            "display_name": "AnonymousUser",
        }
        cur.fetchone.return_value = (user_data,)

        with patch("database.auth.cache_user_name") as cache_mock:
            from database.auth import get_user_name
            result = get_user_name("anon_user2", use_default=True)

            assert result == "The User"
            cache_mock.assert_called_once_with("anon_user2", "The User", ttl=3600)
