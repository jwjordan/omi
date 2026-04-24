"""Tests for pusher/audio_reaper.py."""

import os
import time


def test_reap_deletes_old_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("PENDANT_AUDIO_RETENTION_DAYS", "7")
    import importlib
    import pusher.audio_reaper as reaper
    importlib.reload(reaper)

    # Create old + new conversation folders with one file each.
    old_dir = tmp_path / "uid" / "u1" / "conv" / "old"
    old_dir.mkdir(parents=True)
    (old_dir / "1.opus").write_bytes(b"x")
    # Set mtime to 10 days ago.
    old_mtime = time.time() - (10 * 86400)
    os.utime(old_dir / "1.opus", (old_mtime, old_mtime))

    new_dir = tmp_path / "uid" / "u1" / "conv" / "new"
    new_dir.mkdir(parents=True)
    (new_dir / "1.opus").write_bytes(b"x")

    deleted = reaper.reap_once()
    assert deleted == 1
    assert not old_dir.exists()
    assert new_dir.exists()


def test_reap_no_root_returns_zero(monkeypatch):
    monkeypatch.delenv("PENDANT_LOCAL_AUDIO_ROOT", raising=False)
    import importlib
    import pusher.audio_reaper as reaper
    importlib.reload(reaper)

    assert reaper.reap_once() == 0


def test_reap_skips_empty_conv_dir(tmp_path, monkeypatch):
    """An empty conversation folder has no files → mtime calc must handle
    the no-files case without crashing. Policy: skip, don't delete."""
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("PENDANT_AUDIO_RETENTION_DAYS", "7")
    import importlib
    import pusher.audio_reaper as reaper
    importlib.reload(reaper)

    empty = tmp_path / "uid" / "u1" / "conv" / "empty"
    empty.mkdir(parents=True)

    assert reaper.reap_once() == 0
    assert empty.exists()


def test_reap_continues_on_per_directory_error(tmp_path, monkeypatch, caplog):
    """If deletion of one conversation fails, the reaper logs and moves on
    to the next."""
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", str(tmp_path))
    monkeypatch.setenv("PENDANT_AUDIO_RETENTION_DAYS", "7")
    import importlib
    import pusher.audio_reaper as reaper
    importlib.reload(reaper)

    old_dir = tmp_path / "uid" / "u1" / "conv" / "old"
    old_dir.mkdir(parents=True)
    (old_dir / "1.opus").write_bytes(b"x")
    old_mtime = time.time() - (10 * 86400)
    os.utime(old_dir / "1.opus", (old_mtime, old_mtime))

    from unittest.mock import patch
    with patch("pusher.audio_reaper.shutil.rmtree", side_effect=PermissionError("x")):
        deleted = reaper.reap_once()

    # Could not delete; counter stays 0; directory still exists.
    assert deleted == 0
    assert old_dir.exists()
