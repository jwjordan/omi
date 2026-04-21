"""Unit tests for database/trends.py — Postgres impl."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from collections import defaultdict


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


def test_get_trends_data_returns_nested_structure():
    """Verify get_trends_data fetches categories and topics, returning nested dict."""
    with patch("database.trends.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # Mock category query
        cat_id = "abc123"
        cat_data = {"category": "ceo", "type": "best"}

        # Mock topic query
        topic_id_1 = "topic-001"
        topic_data_1 = {"topic": "Elon Musk", "memories_count": 5}
        topic_id_2 = "topic-002"
        topic_data_2 = {"topic": "Sam Altman", "memories_count": 3}

        # First execute: SELECT FROM trends_categories
        # Second execute: SELECT FROM trends_topics
        cur.fetchall.side_effect = [
            [(cat_id, cat_data)],  # categories
            [(cat_id, topic_id_1, topic_data_1), (cat_id, topic_id_2, topic_data_2)]  # topics
        ]

        from database.trends import get_trends_data
        result = get_trends_data()

        assert len(result) == 1
        assert result[0]["id"] == cat_id
        assert result[0]["category"] == "ceo"
        assert result[0]["type"] == "best"
        assert "topics" in result[0]
        assert len(result[0]["topics"]) == 2
        assert result[0]["topics"][0]["id"] == topic_id_1
        assert result[0]["topics"][0]["topic"] == "Elon Musk"
        assert result[0]["topics"][1]["id"] == topic_id_2
        assert result[0]["topics"][1]["topic"] == "Sam Altman"


def test_get_trends_data_uses_connection_not_batch():
    """Verify get_trends_data uses db.connection() for reads."""
    with patch("database.trends.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.side_effect = [[], []]  # empty results

        from database.trends import get_trends_data
        get_trends_data()

        # Should call db.connection(), not db.batch()
        db_mock.connection.assert_called_once()
        db_mock.batch.assert_not_called()


def test_get_trends_data_joins_categories_with_topics():
    """Verify topics are correctly nested by category_id."""
    with patch("database.trends.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # Multiple categories, each with topics
        cat_id_1 = "cat-1"
        cat_data_1 = {"category": "company"}
        cat_id_2 = "cat-2"
        cat_data_2 = {"category": "software_product"}

        topic_1_1 = "topic-1-1"
        topic_data_1_1 = {"topic": "Microsoft", "memories_count": 2}
        topic_2_1 = "topic-2-1"
        topic_data_2_1 = {"topic": "Slack", "memories_count": 3}

        cur.fetchall.side_effect = [
            [(cat_id_1, cat_data_1), (cat_id_2, cat_data_2)],  # categories
            [(cat_id_1, topic_1_1, topic_data_1_1), (cat_id_2, topic_2_1, topic_data_2_1)]  # topics
        ]

        from database.trends import get_trends_data
        result = get_trends_data()

        assert len(result) == 2
        assert result[0]["id"] == cat_id_1
        assert len(result[0]["topics"]) == 1
        assert result[0]["topics"][0]["topic"] == "Microsoft"
        assert result[1]["id"] == cat_id_2
        assert len(result[1]["topics"]) == 1
        assert result[1]["topics"][0]["topic"] == "Slack"


def test_save_trends_uses_batch_for_atomicity():
    """Verify save_trends uses db.batch() for transactional writes."""
    with patch("database.trends.db") as db_mock:
        batch_conn = MagicMock()
        batch_cursor = MagicMock()
        batch_cursor.__enter__ = MagicMock(return_value=batch_cursor)
        batch_cursor.__exit__ = MagicMock(return_value=None)
        batch_conn.cursor.return_value = batch_cursor
        batch_conn.__enter__ = MagicMock(return_value=batch_conn)
        batch_conn.__exit__ = MagicMock(return_value=None)

        db_mock.batch.return_value = batch_conn

        # Create a mock Trend object
        from models.trend import Trend, TrendEnum, TrendType
        trend = Trend(category=TrendEnum.ceo, type=TrendType.best, topics=["Elon Musk"])

        from database.trends import save_trends
        save_trends("mem-001", [trend])

        # Should call db.batch()
        db_mock.batch.assert_called_once()


def test_save_trends_inserts_categories_and_topics():
    """Verify save_trends creates both category and topic rows."""
    with patch("database.trends.db") as db_mock:
        batch_conn = MagicMock()
        batch_cursor = MagicMock()
        batch_cursor.__enter__ = MagicMock(return_value=batch_cursor)
        batch_cursor.__exit__ = MagicMock(return_value=None)
        batch_conn.cursor.return_value = batch_cursor
        batch_conn.__enter__ = MagicMock(return_value=batch_conn)
        batch_conn.__exit__ = MagicMock(return_value=None)

        db_mock.batch.return_value = batch_conn

        from models.trend import Trend, TrendEnum, TrendType
        trend = Trend(category=TrendEnum.company, type=TrendType.worst, topics=["Boeing", "WeWork"])

        from database.trends import save_trends
        save_trends("mem-001", [trend])

        # Should have multiple execute calls: one for category, at least one per topic
        execute_calls = batch_cursor.execute.call_args_list
        assert len(execute_calls) >= 3  # 1 category + 2 topics

        # Verify INSERT into trends_categories is called
        category_insert_found = False
        for call in execute_calls:
            sql = call.args[0]
            if "INSERT INTO trends_categories" in sql:
                category_insert_found = True
                break
        assert category_insert_found, "Category INSERT not found in execute calls"

        # Verify INSERT into trends_topics is called
        topic_insert_found = False
        for call in execute_calls:
            sql = call.args[0]
            if "INSERT INTO trends_topics" in sql:
                topic_insert_found = True
                break
        assert topic_insert_found, "Topic INSERT not found in execute calls"


def test_save_trends_handles_multiple_trends():
    """Verify save_trends processes multiple Trend objects."""
    with patch("database.trends.db") as db_mock:
        batch_conn = MagicMock()
        batch_cursor = MagicMock()
        batch_cursor.__enter__ = MagicMock(return_value=batch_cursor)
        batch_cursor.__exit__ = MagicMock(return_value=None)
        batch_conn.cursor.return_value = batch_cursor
        batch_conn.__enter__ = MagicMock(return_value=batch_conn)
        batch_conn.__exit__ = MagicMock(return_value=None)

        db_mock.batch.return_value = batch_conn

        from models.trend import Trend, TrendEnum, TrendType
        trends = [
            Trend(category=TrendEnum.ceo, type=TrendType.best, topics=["Elon Musk"]),
            Trend(category=TrendEnum.company, type=TrendType.worst, topics=["Boeing"]),
        ]

        from database.trends import save_trends
        save_trends("mem-001", trends)

        # Should execute multiple times for 2 trends + their topics
        execute_calls = batch_cursor.execute.call_args_list
        assert len(execute_calls) >= 4  # At least 2 categories + 2 topics


def test_save_trends_uses_on_conflict_for_idempotency():
    """Verify save_trends uses ON CONFLICT DO UPDATE for idempotency."""
    with patch("database.trends.db") as db_mock:
        batch_conn = MagicMock()
        batch_cursor = MagicMock()
        batch_cursor.__enter__ = MagicMock(return_value=batch_cursor)
        batch_cursor.__exit__ = MagicMock(return_value=None)
        batch_conn.cursor.return_value = batch_cursor
        batch_conn.__enter__ = MagicMock(return_value=batch_conn)
        batch_conn.__exit__ = MagicMock(return_value=None)

        db_mock.batch.return_value = batch_conn

        from models.trend import Trend, TrendEnum, TrendType
        trend = Trend(category=TrendEnum.ai_product, type=TrendType.best, topics=["ChatGPT"])

        from database.trends import save_trends
        save_trends("mem-001", [trend])

        # Verify ON CONFLICT clause appears in SQL
        execute_calls = batch_cursor.execute.call_args_list
        on_conflict_found = False
        for call in execute_calls:
            sql = call.args[0]
            if "ON CONFLICT" in sql:
                on_conflict_found = True
                break
        assert on_conflict_found, "ON CONFLICT clause not found in execute calls"
