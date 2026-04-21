"""Unit tests for database/app_review_config.py — Postgres impl."""

from unittest.mock import MagicMock, patch
import pytest


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


@pytest.mark.parametrize("patches_needed", [[
    "database.app_review_config.db",
    "database.app_review_config.get_memory_cache",
]])
def test_fetch_review_config_returns_empty_dict_when_missing(patches_needed):
    """_fetch_review_config returns {} when platform not in database."""
    with patch("database.app_review_config.db") as db_mock, \
         patch("database.app_review_config.get_memory_cache"):
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.app_review_config import _fetch_review_config

        result = _fetch_review_config("ios")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT data" in sql
        assert "FROM app_review_config" in sql
        assert "WHERE id = %s" in sql
        assert params == ("ios",)
        assert result == {}


def test_fetch_review_config_returns_data_when_found():
    """_fetch_review_config returns data JSONB when platform exists."""
    with patch("database.app_review_config.db") as db_mock, \
         patch("database.app_review_config.get_memory_cache"):
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        test_data = {"hidden_versions": ["1.0.531"], "reviewer_uids": ["uid-123"]}
        cur.fetchone.return_value = (test_data,)

        from database.app_review_config import _fetch_review_config

        result = _fetch_review_config("ios")

        assert result == test_data


def test_get_review_config_returns_config():
    """get_review_config returns config for a platform."""
    with patch("database.app_review_config.db") as db_mock, \
         patch("database.app_review_config.get_memory_cache") as cache_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        test_data = {"hidden_versions": ["1.0.531"], "reviewer_uids": []}
        cur.fetchone.return_value = (test_data,)

        # Mock cache to call through to _fetch_review_config
        mock_cache = MagicMock()
        cache_mock.return_value = mock_cache

        def cache_get_or_fetch(key, fetch_fn, ttl=None):
            return fetch_fn()

        mock_cache.get_or_fetch.side_effect = cache_get_or_fetch

        from database.app_review_config import get_review_config

        result = get_review_config("ios")

        assert result == test_data
        # Verify cache was called with correct key and TTL
        mock_cache.get_or_fetch.assert_called_once()
        args = mock_cache.get_or_fetch.call_args
        assert args[0][0] == "app_review_config:ios"
        assert args[1]["ttl"] == 60


def test_should_hide_subscription_ui_hides_for_reviewer_uid():
    """should_hide_subscription_ui returns True when UID is in reviewer_uids."""
    with patch("database.app_review_config.get_review_config") as get_config_mock, \
         patch("database.app_review_config.db"), \
         patch("database.app_review_config.get_memory_cache"):
        get_config_mock.return_value = {
            "hidden_versions": [],
            "reviewer_uids": ["uid-123", "uid-456"],
        }

        from database.app_review_config import should_hide_subscription_ui

        result = should_hide_subscription_ui("uid-123", "ios", "1.0.0")

        assert result is True
        get_config_mock.assert_called_once_with("ios")


def test_should_hide_subscription_ui_hides_for_hidden_version():
    """should_hide_subscription_ui returns True when version is in hidden_versions."""
    with patch("database.app_review_config.get_review_config") as get_config_mock, \
         patch("database.app_review_config._compare_versions") as compare_mock, \
         patch("database.app_review_config.db"), \
         patch("database.app_review_config.get_memory_cache"):
        get_config_mock.return_value = {
            "hidden_versions": ["1.0.531"],
            "reviewer_uids": [],
        }
        compare_mock.return_value = 0  # versions match

        from database.app_review_config import should_hide_subscription_ui

        result = should_hide_subscription_ui("uid-unknown", "ios", "1.0.531")

        assert result is True
        get_config_mock.assert_called_once_with("ios")
        compare_mock.assert_called_once_with("1.0.531", "1.0.531")


def test_should_hide_subscription_ui_returns_false_when_not_hidden():
    """should_hide_subscription_ui returns False when not in reviewer_uids or hidden_versions."""
    with patch("database.app_review_config.get_review_config") as get_config_mock, \
         patch("database.app_review_config.db"), \
         patch("database.app_review_config.get_memory_cache"):
        get_config_mock.return_value = {
            "hidden_versions": ["1.0.531"],
            "reviewer_uids": ["uid-123"],
        }

        from database.app_review_config import should_hide_subscription_ui

        result = should_hide_subscription_ui("uid-unknown", "ios", "2.0.0")

        assert result is False


def test_should_hide_subscription_ui_ignores_unsupported_platform():
    """should_hide_subscription_ui returns False for unsupported platforms."""
    with patch("database.app_review_config.db"), \
         patch("database.app_review_config.get_memory_cache"):
        from database.app_review_config import should_hide_subscription_ui

        result = should_hide_subscription_ui("uid-123", "android", "1.0.0")
        assert result is False

        result = should_hide_subscription_ui("uid-123", None, "1.0.0")
        assert result is False

        result = should_hide_subscription_ui("uid-123", "", "1.0.0")
        assert result is False


def test_should_hide_subscription_ui_normalizes_platform():
    """should_hide_subscription_ui normalizes platform to lowercase."""
    with patch("database.app_review_config.get_review_config") as get_config_mock, \
         patch("database.app_review_config.db"), \
         patch("database.app_review_config.get_memory_cache"):
        get_config_mock.return_value = {
            "hidden_versions": [],
            "reviewer_uids": ["uid-123"],
        }

        from database.app_review_config import should_hide_subscription_ui

        result = should_hide_subscription_ui("uid-123", "iOS", "1.0.0")

        assert result is True
        # Verify lowercase was passed to get_review_config
        get_config_mock.assert_called_once_with("ios")


def test_should_hide_subscription_ui_handles_none_config():
    """should_hide_subscription_ui handles None config gracefully."""
    with patch("database.app_review_config.get_review_config") as get_config_mock, \
         patch("database.app_review_config.db"), \
         patch("database.app_review_config.get_memory_cache"):
        get_config_mock.return_value = None

        from database.app_review_config import should_hide_subscription_ui

        result = should_hide_subscription_ui("uid-123", "ios", "1.0.0")

        assert result is False


def test_should_hide_subscription_ui_handles_missing_uid_and_version():
    """should_hide_subscription_ui returns False when uid and app_version are None."""
    with patch("database.app_review_config.get_review_config") as get_config_mock, \
         patch("database.app_review_config.db"), \
         patch("database.app_review_config.get_memory_cache"):
        get_config_mock.return_value = {
            "hidden_versions": ["1.0.0"],
            "reviewer_uids": ["uid-123"],
        }

        from database.app_review_config import should_hide_subscription_ui

        result = should_hide_subscription_ui(None, "ios", None)

        assert result is False
