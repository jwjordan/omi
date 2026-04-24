"""Tests for pusher/diarization_worker.py."""

from unittest.mock import MagicMock, patch


def _mock_db():
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=None)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=None)
    return conn, cursor


def test_no_pending_jobs_returns_false():
    from pusher.diarization_worker import run_one

    with patch("pusher.diarization_worker.db") as db_mock:
        conn, cur = _mock_db()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None  # no pending job

        processed = run_one()
        assert processed is False


def test_run_one_claims_job_and_marks_done_on_success():
    from pusher.diarization_worker import run_one

    with patch("pusher.diarization_worker.db") as db_mock, \
         patch("pusher.diarization_worker._run_job") as run_job:
        conn, cur = _mock_db()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("c1", "u1", 0)
        run_job.return_value = True

        processed = run_one()
        assert processed is True

        # Expect at least 2 execute calls: claim + mark done.
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any("UPDATE pending_diarizations" in s and "running" in s for s in sqls)
        assert any("UPDATE pending_diarizations" in s and "done" in s for s in sqls)


def test_run_one_requeues_on_failure_under_retry_cap():
    from pusher.diarization_worker import run_one

    with patch("pusher.diarization_worker.db") as db_mock, \
         patch("pusher.diarization_worker._run_job") as run_job:
        conn, cur = _mock_db()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("c1", "u1", 0)
        run_job.side_effect = RuntimeError("boom")

        run_one()
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        # State goes back to 'pending' for retry.
        assert any("state = 'pending'" in s for s in sqls)


def test_run_one_marks_failed_after_max_attempts():
    from pusher.diarization_worker import run_one

    with patch("pusher.diarization_worker.db") as db_mock, \
         patch("pusher.diarization_worker._run_job") as run_job:
        conn, cur = _mock_db()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("c1", "u1", 3)  # already at max attempts
        run_job.side_effect = RuntimeError("boom")

        run_one()
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any("state = 'failed'" in s for s in sqls)


def test_requeue_stale_running_jobs_at_startup():
    from pusher.diarization_worker import requeue_stale_running

    with patch("pusher.diarization_worker.db") as db_mock:
        conn, cur = _mock_db()
        db_mock.connection.return_value = conn
        cur.rowcount = 2

        n = requeue_stale_running()
        sql = cur.execute.call_args.args[0]
        assert "UPDATE pending_diarizations" in sql
        assert "state = 'pending'" in sql
        assert "WHERE state = 'running'" in sql
        assert n == 2
