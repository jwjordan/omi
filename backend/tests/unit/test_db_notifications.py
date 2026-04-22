"""Unit tests for database/notifications.py — Postgres impl.

Covers:
- save_token: legacy migration, unknown_default collapse, normal upsert
- users.data accessor pairs (getters/setters for daily_summary_*, mentor_notification_*)
- get_all_tokens, remove_invalid_token, remove_bulk_tokens
- Async cross-user queries: get_users_token_in_timezones, get_users_id_in_timezones,
  get_users_for_daily_summary
"""

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Force a fresh import of database.notifications (replacing any MagicMock stub that
# another test module may have left in sys.modules via `setdefault`). This
# makes the test file independent of collection order.
for _mod in ("database.notifications", "database._client"):
    sys.modules.pop(_mod, None)
import database.notifications  # noqa: E402,F401  (populates sys.modules with the real module)


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


def _mock_batch(db_mock, conn):
    """Wire db.batch() to return a context manager yielding conn."""
    batch_ctx = MagicMock()
    batch_ctx.__enter__ = MagicMock(return_value=conn)
    batch_ctx.__exit__ = MagicMock(return_value=None)
    db_mock.batch.return_value = batch_ctx


# ---------------------------------------------------------------------------
# save_token: Legacy Migration
# ---------------------------------------------------------------------------


def test_save_token_migrates_legacy_fcm_token_to_unknown_default():
    """Test that legacy fcm_token field is migrated to user_fcm_tokens."""
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # save_token has multiple fetchone calls:
        # 1) check legacy token existence
        # 2) check if legacy token already in user_fcm_tokens
        # 3) check unknown_default (won't happen since device_key != unknown_default)
        cur.fetchone.side_effect = [
            ("old_token", "America/New_York"),  # legacy token + tz exists
            None,  # token not yet in user_fcm_tokens
            None,  # (device_key != unknown_default, so won't check this)
        ]

        from database.notifications import save_token
        save_token("u1", {"device_key": "device1", "fcm_token": "new_token", "time_zone": "America/New_York"})

        # Should have: 1) read legacy, 2) check if exists in user_fcm_tokens,
        # 3) insert to unknown_default, 4) delete legacy from users.data,
        # 5) upsert new token, 6) update time_zone
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = "\n".join(sqls)

        assert "SELECT data->>'fcm_token'" in joined
        assert "INSERT INTO user_fcm_tokens" in joined and "unknown_default" in joined
        assert "UPDATE users SET data = data - 'fcm_token'" in joined


def test_save_token_skips_migration_if_legacy_token_already_in_subcollection():
    """Test that migration is skipped if legacy token already in user_fcm_tokens."""
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # First: SELECT legacy token; second: token DOES exist in user_fcm_tokens
        cur.fetchone.side_effect = [
            ("old_token", "America/New_York"),
            (1,),  # already exists - skip INSERT to unknown_default
            None,  # device_key != 'unknown_default', so no unknown_default to collapse
        ]

        from database.notifications import save_token
        save_token("u1", {"device_key": "device1", "fcm_token": "new_token", "time_zone": "America/New_York"})

        # Should NOT insert to unknown_default, should only update users.data
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        # Filter for INSERT ... unknown_default - should NOT find a plain INSERT (only the ON CONFLICT upsert for device1)
        unknown_inserts = [s for s in sqls if "INSERT INTO user_fcm_tokens" in s and "'unknown_default'" in s]
        assert len(unknown_inserts) == 0


def test_save_token_collapses_unknown_default_when_same_token_with_proper_device_key():
    """Test that unknown_default is deleted when same token is saved with proper device_key."""
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # First: no legacy token; second: unknown_default exists with same token
        cur.fetchone.side_effect = [
            None,  # no legacy token
            ("new_token",),  # unknown_default has the same token
        ]

        from database.notifications import save_token
        save_token("u1", {"device_key": "device1", "fcm_token": "new_token", "time_zone": "America/New_York"})

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = "\n".join(sqls)

        # Should delete unknown_default
        assert "DELETE FROM user_fcm_tokens WHERE uid=%s AND device_key='unknown_default'" in joined


def test_save_token_keeps_unknown_default_if_different_token():
    """Test that unknown_default is kept if it has a different token."""
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # First: no legacy token; second: unknown_default has different token
        cur.fetchone.side_effect = [
            None,
            ("old_token",),  # different from new_token
        ]

        from database.notifications import save_token
        save_token("u1", {"device_key": "device1", "fcm_token": "new_token", "time_zone": "America/New_York"})

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = "\n".join(sqls)

        # Should NOT delete unknown_default
        delete_statements = [s for s in sqls if "DELETE FROM user_fcm_tokens" in s]
        assert len(delete_statements) == 0


def test_save_token_upserts_new_token():
    """Test that new token is inserted/updated."""
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.notifications import save_token
        save_token("u1", {"device_key": "device1", "fcm_token": "new_token", "time_zone": "America/New_York"})

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = "\n".join(sqls)

        assert "INSERT INTO user_fcm_tokens" in joined
        assert "ON CONFLICT (uid, device_key)" in joined
        assert "new_token" in str(cur.execute.call_args_list)


def test_save_token_updates_users_data_time_zone():
    """Test that users.data.time_zone is updated."""
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.notifications import save_token
        save_token("u1", {"device_key": "device1", "fcm_token": "token", "time_zone": "UTC"})

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = "\n".join(sqls)

        assert "jsonb_build_object('time_zone'" in joined


# ---------------------------------------------------------------------------
# get_user_time_zone
# ---------------------------------------------------------------------------


def test_get_user_time_zone_returns_time_zone():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("America/New_York",)

        from database.notifications import get_user_time_zone
        result = get_user_time_zone("u1")
        assert result == "America/New_York"

        sql = cur.execute.call_args.args[0]
        assert "data->>'time_zone'" in sql


def test_get_user_time_zone_returns_none_when_missing():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.notifications import get_user_time_zone
        assert get_user_time_zone("u1") is None


# ---------------------------------------------------------------------------
# Daily Summary Hour
# ---------------------------------------------------------------------------


def test_get_daily_summary_hour_local_returns_hour():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (22,)

        from database.notifications import get_daily_summary_hour_local
        result = get_daily_summary_hour_local("u1")
        assert result == 22


def test_get_daily_summary_hour_local_returns_none_when_unset():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (None,)

        from database.notifications import get_daily_summary_hour_local
        assert get_daily_summary_hour_local("u1") is None


def test_set_daily_summary_hour_local_upserts_users_data():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.notifications import set_daily_summary_hour_local
        result = set_daily_summary_hour_local("u1", 10)
        assert result is True

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO users" in sql
        assert "ON CONFLICT (uid) DO UPDATE" in sql
        assert "daily_summary_hour_local" in sql
        assert params[0] == "u1"
        assert 10 in params


def test_set_daily_summary_hour_local_rejects_invalid_hour():
    import pytest
    from database.notifications import set_daily_summary_hour_local

    with pytest.raises(ValueError):
        set_daily_summary_hour_local("u1", 25)

    with pytest.raises(ValueError):
        set_daily_summary_hour_local("u1", -1)


# ---------------------------------------------------------------------------
# Daily Summary Enabled
# ---------------------------------------------------------------------------


def test_get_daily_summary_enabled_returns_bool():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (True,)

        from database.notifications import get_daily_summary_enabled
        assert get_daily_summary_enabled("u1") is True


def test_get_daily_summary_enabled_defaults_true_when_unset():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (None,)

        from database.notifications import get_daily_summary_enabled
        assert get_daily_summary_enabled("u1") is True


def test_set_daily_summary_enabled_upserts():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.notifications import set_daily_summary_enabled
        result = set_daily_summary_enabled("u1", False)
        assert result is True

        sql = cur.execute.call_args.args[0]
        assert "INSERT INTO users" in sql
        assert "daily_summary_enabled" in sql


# ---------------------------------------------------------------------------
# Mentor Notification Frequency
# ---------------------------------------------------------------------------


def test_get_mentor_notification_frequency_returns_cached_value():
    with patch("database.notifications.db") as db_mock:
        with patch("database.notifications.get_memory_cache") as cache_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            cur.fetchone.return_value = (3,)

            cache = MagicMock()
            cache.get_or_fetch.side_effect = lambda key, fetch_fn, ttl: fetch_fn()
            cache_mock.return_value = cache

            from database.notifications import get_mentor_notification_frequency
            result = get_mentor_notification_frequency("u1")
            assert result == 3


def test_set_mentor_notification_frequency_upserts_and_invalidates_cache():
    with patch("database.notifications.db") as db_mock:
        with patch("database.notifications.get_memory_cache") as cache_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn

            cache = MagicMock()
            cache_mock.return_value = cache

            from database.notifications import set_mentor_notification_frequency
            result = set_mentor_notification_frequency("u1", 4)
            assert result is True

            sql = cur.execute.call_args.args[0]
            assert "INSERT INTO users" in sql
            assert "mentor_notification_frequency" in sql
            cache.delete.assert_called_once()


def test_set_mentor_notification_frequency_rejects_invalid_frequency():
    import pytest
    from database.notifications import set_mentor_notification_frequency

    with pytest.raises(ValueError):
        set_mentor_notification_frequency("u1", 6)

    with pytest.raises(ValueError):
        set_mentor_notification_frequency("u1", -1)


# ---------------------------------------------------------------------------
# get_all_tokens
# ---------------------------------------------------------------------------


def test_get_all_tokens_returns_tokens_from_user_fcm_tokens():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [("token1",), ("token2",)]

        from database.notifications import get_all_tokens
        result = get_all_tokens("u1")
        assert result == ["token1", "token2"]

        sql = cur.execute.call_args.args[0]
        assert "SELECT token FROM user_fcm_tokens" in sql
        assert "WHERE uid=%s AND token IS NOT NULL" in sql


def test_get_all_tokens_returns_empty_list_when_no_tokens():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.notifications import get_all_tokens
        assert get_all_tokens("u1") == []


# ---------------------------------------------------------------------------
# remove_invalid_token
# ---------------------------------------------------------------------------


def test_remove_invalid_token_deletes_by_token():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.notifications import remove_invalid_token
        remove_invalid_token("bad_token")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM user_fcm_tokens WHERE token=%s" in sql
        assert params == ("bad_token",)


# ---------------------------------------------------------------------------
# remove_bulk_tokens
# ---------------------------------------------------------------------------


def test_remove_bulk_tokens_deletes_multiple_tokens():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.notifications import remove_bulk_tokens
        remove_bulk_tokens(["token1", "token2", "token3"])

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM user_fcm_tokens WHERE token = ANY(%s::text[])" in sql
        assert params[0] == ["token1", "token2", "token3"]


def test_remove_bulk_tokens_noop_when_empty():
    with patch("database.notifications.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.notifications import remove_bulk_tokens
        remove_bulk_tokens([])

        cur.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Async cross-user queries
# ---------------------------------------------------------------------------


def test_get_users_token_in_timezones_returns_flat_token_list():
    """Test that get_users_token_in_timezones returns a flat list of tokens."""
    async def run_test():
        with patch("database.notifications.db") as db_mock:
            with patch("database.notifications._get_users_in_timezones") as mock_get:
                mock_get.return_value = ["token1", "token2", "token3"]

                from database.notifications import get_users_token_in_timezones
                result = await get_users_token_in_timezones(["America/New_York"])
                assert result == ["token1", "token2", "token3"]
                mock_get.assert_called_once_with(["America/New_York"], "fcm_token")

    asyncio.run(run_test())


def test_get_users_id_in_timezones_returns_tuples():
    """Test that get_users_id_in_timezones returns (uid, tokens, tz) tuples."""
    async def run_test():
        with patch("database.notifications.db") as db_mock:
            with patch("database.notifications._get_users_in_timezones") as mock_get:
                expected = [("u1", ["token1"], "America/New_York"), ("u2", ["token2"], "America/Chicago")]
                mock_get.return_value = expected

                from database.notifications import get_users_id_in_timezones
                result = await get_users_id_in_timezones(["America/New_York", "America/Chicago"])
                assert result == expected
                mock_get.assert_called_once_with(["America/New_York", "America/Chicago"], "id")

    asyncio.run(run_test())


def test_get_users_for_daily_summary_filters_by_hour_and_enabled():
    """Test that get_users_for_daily_summary applies hour and enabled filters."""
    async def run_test():
        with patch("database.notifications.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn

            # Mock the user query
            user_data_1 = {
                "daily_summary_enabled": True,
                "daily_summary_hour_local": 22,
                "time_zone": "America/New_York",
            }
            user_data_2 = {
                "daily_summary_enabled": False,  # disabled
                "daily_summary_hour_local": 22,
                "time_zone": "America/New_York",
            }
            user_data_3 = {
                "daily_summary_enabled": True,
                "daily_summary_hour_local": 10,  # wrong hour
                "time_zone": "America/Chicago",
            }

            cur.fetchall.side_effect = [
                [("u1", user_data_1), ("u2", user_data_2), ("u3", user_data_3)],
                [("token1",)],  # tokens for u1
                [("token2",)],  # tokens for u2 (won't be used)
                [("token3",)],  # tokens for u3 (won't be used)
            ]

            from database.notifications import get_users_for_daily_summary
            result = await get_users_for_daily_summary(["America/New_York", "America/Chicago"], 22)

            # Only u1 should be returned (enabled=True, hour=22)
            assert len(result) == 1
            assert result[0][0] == "u1"
            assert result[0][1] == ["token1"]

    asyncio.run(run_test())


def test_get_users_for_daily_summary_returns_empty_when_no_timezones():
    """Test that empty timezone list returns empty result."""
    async def run_test():
        from database.notifications import get_users_for_daily_summary
        result = await get_users_for_daily_summary([], 22)
        assert result == []

    asyncio.run(run_test())


def test_get_users_in_timezones_chunks_large_timezone_lists():
    """Test that _get_users_in_timezones chunks timezone queries (30 max per query)."""
    async def run_test():
        with patch("database.notifications.db") as db_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn

            # Create 35 timezones (more than 30) to force chunking
            timezones = [f"TZ_{i}" for i in range(35)]

            cur.fetchall.side_effect = [
                [("u1", {"time_zone": "TZ_0"})],  # first chunk (30 results)
                [("u2", {"time_zone": "TZ_31"})],  # second chunk (5 results)
            ]
            cur.fetchall = MagicMock(side_effect=[
                [("u1", {"time_zone": "TZ_0"})],
                [("token1",)],
                [("u2", {"time_zone": "TZ_31"})],
                [("token2",)],
            ])

            from database.notifications import _get_users_in_timezones
            result = await _get_users_in_timezones(timezones, "id")

            # Both queries should have been made (chunked)
            assert cur.execute.call_count >= 2
