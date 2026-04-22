"""Unit tests for scripts/backfill_vectors.py."""

from unittest.mock import MagicMock, patch


def _mock_db_connection():
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=None)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=None)
    return conn, cursor


def test_backfill_skips_conversations_already_vectorized():
    from scripts.backfill_vectors import backfill_missing_vectors

    with patch("scripts.backfill_vectors.db") as db_mock, \
         patch("scripts.backfill_vectors.embeddings") as emb_mock, \
         patch("scripts.backfill_vectors.upsert_vector") as upsert_mock:
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        # Return zero rows = nothing to backfill
        cur.fetchall.return_value = []

        stats = backfill_missing_vectors()
        assert stats["embedded"] == 0
        assert stats["skipped"] == 0
        emb_mock.embed_query.assert_not_called()
        upsert_mock.assert_not_called()


def test_backfill_embeds_one_row_per_missing_conversation():
    from scripts.backfill_vectors import backfill_missing_vectors

    with patch("scripts.backfill_vectors.db") as db_mock, \
         patch("scripts.backfill_vectors.embeddings") as emb_mock, \
         patch("scripts.backfill_vectors.upsert_vector") as upsert_mock:
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("uid1", "c1", {"structured": {"title": "T1", "overview": "O1"}}),
            ("uid1", "c2", {"structured": {"title": "T2", "overview": "O2"}}),
        ]
        emb_mock.embed_query.return_value = [0.0] * 768

        stats = backfill_missing_vectors()
        assert stats["embedded"] == 2
        assert emb_mock.embed_query.call_count == 2
        assert upsert_mock.call_count == 2
        # Called with uid, id, vector in that order.
        assert upsert_mock.call_args_list[0].args[:2] == ("uid1", "c1")
        assert upsert_mock.call_args_list[1].args[:2] == ("uid1", "c2")


def test_backfill_uses_left_join_to_find_missing():
    from scripts.backfill_vectors import backfill_missing_vectors

    with patch("scripts.backfill_vectors.db") as db_mock, \
         patch("scripts.backfill_vectors.embeddings"), \
         patch("scripts.backfill_vectors.upsert_vector"):
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        backfill_missing_vectors()
        sql = cur.execute.call_args.args[0]
        assert "LEFT JOIN conversation_vectors" in sql
        assert "IS NULL" in sql
        assert "status = 'completed'" in sql
        assert "NOT" in sql and "discarded" in sql


def test_backfill_continues_on_per_conversation_failure():
    from scripts.backfill_vectors import backfill_missing_vectors

    with patch("scripts.backfill_vectors.db") as db_mock, \
         patch("scripts.backfill_vectors.embeddings") as emb_mock, \
         patch("scripts.backfill_vectors.upsert_vector") as upsert_mock:
        conn, cur = _mock_db_connection()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("uid1", "c1", {"structured": {"title": "T1", "overview": "O1"}}),
            ("uid1", "c2", {"structured": {"title": "T2", "overview": "O2"}}),
        ]
        # First embed raises; second succeeds.
        emb_mock.embed_query.side_effect = [RuntimeError("boom"), [0.0] * 768]

        stats = backfill_missing_vectors()
        assert stats["embedded"] == 1
        assert stats["failed"] == 1
        assert upsert_mock.call_count == 1
