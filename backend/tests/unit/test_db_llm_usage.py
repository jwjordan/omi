"""Unit tests for database/llm_usage.py — Postgres impl."""

import json
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


def test_record_llm_usage_increments_with_batch():
    with patch("database.llm_usage.db") as db_mock:
        batch_conn, batch_cur = _mock_conn()
        db_mock.batch.return_value = batch_conn
        batch_cur.fetchone.return_value = None

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import record_llm_usage
            record_llm_usage("user123", "chat", "gpt-4", 100, 50)

            # Verify INSERT was called
            calls = batch_cur.execute.call_args_list
            last_sql = calls[-1].args[0]
            assert "INSERT INTO llm_usage" in last_sql
            assert "ON CONFLICT" in last_sql


def test_record_llm_usage_skips_zero_tokens():
    with patch("database.llm_usage.db") as db_mock:
        from database.llm_usage import record_llm_usage
        record_llm_usage("user123", "chat", "gpt-4", 0, 0)

        db_mock.batch.assert_not_called()


def test_get_daily_usage_fetches_by_uid_and_date():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        row_data = {"chat": {"gpt_4": {"input_tokens": 100, "output_tokens": 50, "call_count": 1}}}
        cur.fetchone.return_value = (row_data,)

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import get_daily_usage
            result = get_daily_usage("user123")

            sql = cur.execute.call_args.args[0]
            params = cur.execute.call_args.args[1]
            assert "WHERE uid = %s AND id = %s" in sql
            assert params == ("user123", "2026-04-21")
            assert result["chat"]["gpt_4"]["input_tokens"] == 100


def test_get_daily_usage_returns_empty_dict_when_not_found():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import get_daily_usage
            result = get_daily_usage("user123")

            assert result == {}


def test_get_daily_usage_respects_custom_date():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.llm_usage import get_daily_usage
        custom_date = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
        get_daily_usage("user123", date=custom_date)

        params = cur.execute.call_args.args[1]
        assert params[1] == "2026-04-20"


def test_get_usage_summary_aggregates_multi_day_window():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchall.return_value = [
            ("2026-04-20", {"chat": {"gpt_4": {"input_tokens": 100, "output_tokens": 50, "call_count": 1}}}),
            ("2026-04-21", {"chat": {"gpt_4": {"input_tokens": 200, "output_tokens": 100, "call_count": 2}}}),
        ]

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import get_usage_summary
            result = get_usage_summary("user123", days=30)

            assert result["chat"]["input_tokens"] == 300
            assert result["chat"]["output_tokens"] == 150
            assert result["chat"]["call_count"] == 3


def test_get_usage_summary_returns_empty_when_no_data():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import get_usage_summary
            result = get_usage_summary("user123", days=30)

            assert result == {}


def test_get_top_features_sorts_by_total_tokens_descending():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchall.return_value = [
            (
                "2026-04-21",
                {
                    "chat": {"gpt_4": {"input_tokens": 1000, "output_tokens": 500, "call_count": 10}},
                    "rag": {"gpt_4": {"input_tokens": 100, "output_tokens": 50, "call_count": 1}},
                },
            )
        ]

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import get_top_features
            result = get_top_features("user123", days=30, limit=3)

            assert result[0]["feature"] == "chat"
            assert result[0]["total_tokens"] == 1500
            assert result[1]["feature"] == "rag"


def test_get_top_features_respects_limit():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchall.return_value = [
            (
                "2026-04-21",
                {
                    "f1": {"gpt_4": {"input_tokens": 300, "output_tokens": 0, "call_count": 1}},
                    "f2": {"gpt_4": {"input_tokens": 200, "output_tokens": 0, "call_count": 1}},
                    "f3": {"gpt_4": {"input_tokens": 100, "output_tokens": 0, "call_count": 1}},
                },
            )
        ]

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import get_top_features
            result = get_top_features("user123", days=30, limit=2)

            assert len(result) == 2


def test_get_global_top_features_queries_all_users():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchall.return_value = [
            ("2026-04-21", {"chat": {"gpt_4": {"input_tokens": 5000, "output_tokens": 2000, "call_count": 50}}})
        ]

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import get_global_top_features
            result = get_global_top_features(days=30, limit=3)

            sql = cur.execute.call_args.args[0]
            # No uid in the WHERE clause
            assert "WHERE uid" not in sql
            assert result[0]["feature"] == "chat"
            assert result[0]["total_tokens"] == 7000


def test_record_llm_usage_bucket_upserts_with_batch():
    with patch("database.llm_usage.db") as db_mock:
        batch_conn, batch_cur = _mock_conn()
        db_mock.batch.return_value = batch_conn
        batch_cur.fetchone.return_value = None

        with patch("database.llm_usage.datetime") as dt_mock:
            dt_mock.now.return_value = datetime(2026, 4, 21, 12, 30, 45, tzinfo=timezone.utc)

            from database.llm_usage import record_llm_usage_bucket
            record_llm_usage_bucket(
                "user123",
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=10,
                cache_write_tokens=20,
                total_tokens=180,
                cost_usd=0.05,
                bucket="desktop_chat",
                account="omi",
            )

            calls = batch_cur.execute.call_args_list
            last_sql = calls[-1].args[0]
            assert "INSERT INTO llm_usage" in last_sql


def test_get_total_llm_cost_sums_across_all_days():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchall.return_value = [
            ("2026-04-20", {"desktop_chat": {"cost_usd": 0.12}}),
            ("2026-04-21", {"desktop_chat": {"cost_usd": 0.18}}),
        ]

        from database.llm_usage import get_total_llm_cost
        result = get_total_llm_cost("user123", bucket="desktop_chat")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "WHERE uid = %s" in sql
        assert params[0] == "user123"
        assert result == 0.30


def test_get_total_llm_cost_returns_zero_when_no_data():
    with patch("database.llm_usage.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.llm_usage import get_total_llm_cost
        result = get_total_llm_cost("user123", bucket="desktop_chat")

        assert result == 0.0
