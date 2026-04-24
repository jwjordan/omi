"""Verify _finalize inserts a pending_diarizations row."""

from unittest.mock import MagicMock, patch


def test_finalize_enqueues_diarization_job():
    """When a non-reprocess conversation finalizes, an INSERT ... ON CONFLICT
    DO NOTHING row should hit pending_diarizations."""
    from utils.conversations.process_conversation import enqueue_diarization_job

    with patch("utils.conversations.process_conversation.db") as db_mock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=None)
        db_mock.connection.return_value = conn

        enqueue_diarization_job(uid="u1", conversation_id="c1")

        sql, params = cursor.execute.call_args.args
        assert "INSERT INTO pending_diarizations" in sql
        assert "ON CONFLICT" in sql
        assert params == ("c1", "u1")


def test_enqueue_is_idempotent_on_conflict():
    """Re-enqueue (e.g. reprocess) should not error."""
    from utils.conversations.process_conversation import enqueue_diarization_job

    with patch("utils.conversations.process_conversation.db") as db_mock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        conn = MagicMock()
        conn.cursor.return_value = cursor
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=None)
        db_mock.connection.return_value = conn

        enqueue_diarization_job("u1", "c1")
        enqueue_diarization_job("u1", "c1")
        # Two calls, both produce ON CONFLICT DO NOTHING SQL.
        assert cursor.execute.call_count == 2
