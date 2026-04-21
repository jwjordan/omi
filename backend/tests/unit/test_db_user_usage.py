"""Unit tests for database/user_usage.py — Postgres impl."""

from datetime import datetime, timezone
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
# get_monthly_chat_usage
# ---------------------------------------------------------------------------


def test_get_monthly_chat_usage_sums_desktop_and_backend_chat():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # Two llm_usage rows for April 2026
        cur.fetchall.return_value = [
            (
                {
                    "desktop_chat.call_count": 3,
                    "desktop_chat.cost_usd": 0.12,
                    "chat.gpt_4.call_count": 2,
                },
            ),
            (
                {
                    "desktop_chat.call_count": 1,
                    "desktop_chat.cost_usd": 0.03,
                    "chat.gemini.call_count": 4,
                    # excluded non-user-initiated category:
                    "conversation_processing.gpt_4.call_count": 99,
                },
            ),
        ]

        from database.user_usage import get_monthly_chat_usage
        now = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc)
        result = get_monthly_chat_usage("user123", now=now)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM llm_usage" in sql
        assert "WHERE uid = %s" in sql
        assert "id LIKE" in sql
        assert params == ("user123", "2026-04-%")

        # 3 + 1 desktop_chat.call_count + 2 + 4 backend chat.*.call_count = 10
        assert result["questions"] == 10
        assert result["cost_usd"] == round(0.15, 4)
        # reset_at is start of May 2026 UTC
        assert result["reset_at"] == int(
            datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        )


def test_get_monthly_chat_usage_december_rolls_to_next_year():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.user_usage import get_monthly_chat_usage
        now = datetime(2026, 12, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = get_monthly_chat_usage("user123", now=now)

        assert result["questions"] == 0
        assert result["cost_usd"] == 0.0
        assert result["reset_at"] == int(
            datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
        )


def test_get_monthly_chat_usage_defaults_now_to_utc():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        with patch("database.user_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
            # datetime() still needs to be callable for constructing reset_at
            dt_mock.side_effect = lambda *a, **kw: datetime(*a, **kw)

            from database.user_usage import get_monthly_chat_usage
            result = get_monthly_chat_usage("user123")

            params = cur.execute.call_args.args[1]
            assert params[1] == "2026-04-%"
            assert result["questions"] == 0


# ---------------------------------------------------------------------------
# update_hourly_usage
# ---------------------------------------------------------------------------


def test_update_hourly_usage_upserts_increments_into_jsonb():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.user_usage import update_hourly_usage
        date = datetime(2026, 4, 21, 14, 30, 0, tzinfo=timezone.utc)
        update_hourly_usage(
            "user123",
            date,
            {"transcription_seconds": 60, "words_transcribed": 120},
        )

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO user_hourly_usage" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == "user123"
        assert params[1] == "2026-04-21-14"


def test_update_hourly_usage_skips_when_no_positive_increments():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.user_usage import update_hourly_usage
        date = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        update_hourly_usage("user123", date, {"transcription_seconds": 0})

        cur.execute.assert_not_called()


def test_update_hourly_usage_records_platform_when_provided():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.user_usage import update_hourly_usage
        date = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        update_hourly_usage(
            "user123",
            date,
            {"transcription_seconds": 60},
            platform="desktop",
        )

        # The jsonb payload is the third param
        params = cur.execute.call_args.args[1]
        import json as _json
        payload = _json.loads(params[2])
        assert payload.get("transcription_seconds") == 60
        assert "desktop" in payload.get("platforms", [])


# ---------------------------------------------------------------------------
# batch_update_hourly_usage
# ---------------------------------------------------------------------------


def test_batch_update_hourly_usage_upserts_each_hour():
    with patch("database.user_usage.db") as db_mock:
        batch_conn, batch_cur = _mock_conn()
        db_mock.batch.return_value = batch_conn

        from database.user_usage import batch_update_hourly_usage
        d1 = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 4, 21, 15, 0, 0, tzinfo=timezone.utc)
        batch_update_hourly_usage(
            "user123",
            {
                d1: {"transcription_seconds": 30},
                d2: {"words_transcribed": 200},
            },
        )

        # Two upsert calls, both against user_hourly_usage
        calls = batch_cur.execute.call_args_list
        assert len(calls) == 2
        for c in calls:
            assert "INSERT INTO user_hourly_usage" in c.args[0]
            assert "ON CONFLICT" in c.args[0]

        hour_keys = {c.args[1][1] for c in calls}
        assert hour_keys == {"2026-04-21-14", "2026-04-21-15"}


# ---------------------------------------------------------------------------
# get_today_usage_stats
# ---------------------------------------------------------------------------


def test_get_today_usage_stats_aggregates_hourly_rows():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchall.return_value = [
            ({"transcription_seconds": 60, "words_transcribed": 100, "insights_gained": 1},),
            ({"transcription_seconds": 30, "words_transcribed": 50, "memories_created": 2, "speech_seconds": 10},),
        ]

        from database.user_usage import get_today_usage_stats
        date = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        result = get_today_usage_stats("user123", date)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM user_hourly_usage" in sql
        assert "hour_key LIKE" in sql
        assert params == ("user123", "2026-04-21-%")

        assert result["transcription_seconds"] == 90
        assert result["words_transcribed"] == 150
        assert result["insights_gained"] == 1
        assert result["memories_created"] == 2
        assert result["speech_seconds"] == 10


def test_get_today_usage_stats_returns_zeros_when_empty():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.user_usage import get_today_usage_stats
        date = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        result = get_today_usage_stats("user123", date)

        assert result == {
            "transcription_seconds": 0,
            "words_transcribed": 0,
            "insights_gained": 0,
            "memories_created": 0,
            "speech_seconds": 0,
        }


# ---------------------------------------------------------------------------
# get_monthly_usage_stats
# ---------------------------------------------------------------------------


def test_get_monthly_usage_stats_filters_by_month_prefix():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ({"transcription_seconds": 10},),
            ({"transcription_seconds": 5, "words_transcribed": 40},),
        ]

        from database.user_usage import get_monthly_usage_stats
        date = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        result = get_monthly_usage_stats("user123", date)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM user_hourly_usage" in sql
        assert "hour_key LIKE" in sql
        assert params == ("user123", "2026-04-%")

        assert result["transcription_seconds"] == 15
        assert result["words_transcribed"] == 40


# ---------------------------------------------------------------------------
# get_monthly_usage_stats_since
# ---------------------------------------------------------------------------


def test_get_monthly_usage_stats_since_uses_lower_bound_hour_key():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [({"transcription_seconds": 7},)]

        from database.user_usage import get_monthly_usage_stats_since
        date = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        start = datetime(2026, 4, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = get_monthly_usage_stats_since("user123", date, start)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM user_hourly_usage" in sql
        assert "hour_key LIKE" in sql
        assert "hour_key >=" in sql
        # Params: (uid, month-pattern, start-hour-key)
        assert params[0] == "user123"
        assert params[1] == "2026-04-%"
        assert params[2] == "2026-04-15-00"

        assert result["transcription_seconds"] == 7


# ---------------------------------------------------------------------------
# get_yearly_usage_stats
# ---------------------------------------------------------------------------


def test_get_yearly_usage_stats_filters_by_year_prefix():
    with patch("database.user_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ({"transcription_seconds": 100, "memories_created": 1},),
            ({"transcription_seconds": 200, "memories_created": 3},),
        ]

        from database.user_usage import get_yearly_usage_stats
        date = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)
        result = get_yearly_usage_stats("user123", date)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "hour_key LIKE" in sql
        assert params == ("user123", "2026-%")

        assert result["transcription_seconds"] == 300
        assert result["memories_created"] == 4
