"""Tests for the sync_local_files file-level dedup guard.

When the iOS app's "Store Audio on Cloud" is enabled, it uploads the same
audio both via the live WS stream and via sync-local-files. The file-level
guard skips any .bin file whose start timestamp already sits inside a
long-enough live-streamed conversation, to prevent duplicate transcript
insertion (Omi's existing segment-level dedup is fragile under our
self-hosted whisper-shim where live and sync use different chunking and
produce slightly different word boundaries).
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# timestamp_inside_long_conversation: DB helper behavior
# ---------------------------------------------------------------------------


def _setup_db_mock(fetchone_return):
    """Build a fake `db` with a connection/cursor whose fetchone returns
    whatever the test wants. Returns (db_mock, cursor_mock) so the test
    can inspect the SQL + bound params."""
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone_return
    cursor_ctx = MagicMock()
    cursor_ctx.__enter__ = MagicMock(return_value=cursor)
    cursor_ctx.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor_ctx
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    db_mock = MagicMock()
    db_mock.connection.return_value = conn_ctx
    return db_mock, cursor


def test_returns_true_when_db_finds_matching_long_conversation():
    from database import conversations as cdb

    db_mock, cursor = _setup_db_mock(fetchone_return=(1,))
    with patch.object(cdb, 'db', db_mock):
        assert cdb.timestamp_inside_long_conversation('uid-123', 1777000000.0) is True

    # Verify the SQL query bound the min-duration guard and timestamp correctly.
    assert cursor.execute.call_count == 1
    sql, params = cursor.execute.call_args[0]
    assert 'started_at <= %s' in sql
    assert 'finished_at >= %s' in sql
    assert 'EXTRACT(epoch FROM (finished_at - started_at)) >= %s' in sql
    assert params[0] == 'uid-123'
    # params[1] and [2] are the same datetime (started_at <= ts AND finished_at >= ts)
    assert params[1] == params[2]
    assert params[1] == datetime.fromtimestamp(1777000000.0, tz=timezone.utc)
    assert params[3] == 120  # default min_duration_seconds


def test_returns_false_when_db_finds_no_matching_conversation():
    from database import conversations as cdb

    db_mock, _ = _setup_db_mock(fetchone_return=None)
    with patch.object(cdb, 'db', db_mock):
        assert cdb.timestamp_inside_long_conversation('uid-123', 1777000000.0) is False


def test_min_duration_guard_configurable():
    """Short-conversation threshold can be overridden — the guard keys off
    this to avoid skipping ~2-min sync files that would extend way past a
    tiny 5-second fragment conversation."""
    from database import conversations as cdb

    db_mock, cursor = _setup_db_mock(fetchone_return=None)
    with patch.object(cdb, 'db', db_mock):
        cdb.timestamp_inside_long_conversation('uid-123', 1777000000.0, min_duration_seconds=300)

    _, params = cursor.execute.call_args[0]
    assert params[3] == 300


# ---------------------------------------------------------------------------
# Source-level verification that sync.py wires the guard + bytes path
# ---------------------------------------------------------------------------


class TestSyncSourceWiring:
    @staticmethod
    def _sync_source():
        import os
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'routers', 'sync.py')
        with open(path) as f:
            return f.read()

    def test_imports_storage_disabled(self):
        src = self._sync_source()
        assert 'STORAGE_DISABLED' in src and 'from utils.other.storage import' in src

    def test_imports_deepgram_prerecorded_from_bytes(self):
        assert 'deepgram_prerecorded_from_bytes' in self._sync_source()

    def test_process_segment_has_storage_disabled_branch(self):
        src = self._sync_source()
        start = src.index('def process_segment(')
        next_def = src.index('\ndef ', start + 1)
        body = src[start:next_def]
        assert 'STORAGE_DISABLED' in body, 'process_segment must branch on STORAGE_DISABLED'
        assert 'deepgram_prerecorded_from_bytes' in body

    def test_sync_local_files_has_file_level_dedup_guard(self):
        src = self._sync_source()
        start = src.index('async def sync_local_files(')
        next_def = src.index('\nasync def ', start + 1)
        body = src[start:next_def]
        assert 'timestamp_inside_long_conversation' in body, (
            'sync_local_files must call the file-level dedup guard'
        )

    def test_audio_bytes_cache_reused_for_speaker_id(self):
        src = self._sync_source()
        start = src.index('def process_segment(')
        next_def = src.index('\ndef ', start + 1)
        body = src[start:next_def]
        # The bytes loaded for STT must be reused for speaker ID — re-reading
        # would be wasteful and, in the STORAGE_DISABLED branch, `url` is None
        # so _download_audio_bytes(url) would fail.
        assert 'audio_bytes_cache' in body
