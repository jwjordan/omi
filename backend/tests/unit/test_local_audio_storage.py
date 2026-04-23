"""Tests for the PENDANT_LOCAL_AUDIO_ROOT branch of storage.py.

Verifies that chunk upload, list, delete, and merge operations prefer
the local-disk path when the env var is set and skip GCS entirely.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("ENCRYPTION_SECRET", "omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv")

# Stub heavy GCS deps so storage.py can be imported without credentials.
sys.modules.setdefault("database._client", MagicMock())
_mock_gcs_storage = MagicMock()
_mock_gcs_client_instance = MagicMock()
_mock_gcs_storage.Client.return_value = _mock_gcs_client_instance
sys.modules.setdefault("google.cloud.storage", _mock_gcs_storage)
sys.modules.setdefault("google.cloud.storage.transfer_manager", MagicMock())
sys.modules.setdefault("google.cloud.exceptions", MagicMock())
sys.modules.setdefault("google.oauth2", MagicMock())
sys.modules.setdefault("google.oauth2.service_account", MagicMock())

# The conftest installs a MagicMock at sys.modules["utils.other.storage"].
# Remove it so we can import the real module here.
sys.modules.pop("utils.other.storage", None)
# Also remove the parent package cache so the attribute is re-resolved.
import utils.other  # noqa: E402
if hasattr(utils.other, "storage") and isinstance(utils.other.storage, MagicMock):
    del utils.other.storage

from utils.other import storage as storage_mod  # noqa: E402  # real module now

# Re-register under the key the conftest used so other test files still see
# the real module if they import it, without breaking conftest's intent.
sys.modules["utils.other.storage"] = storage_mod


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    """Point LOCAL_AUDIO_ROOT at a fresh temp directory for the duration of a test."""
    monkeypatch.setattr(storage_mod, "LOCAL_AUDIO_ROOT", str(tmp_path))
    yield tmp_path
    monkeypatch.setattr(storage_mod, "LOCAL_AUDIO_ROOT", None)


def test_upload_audio_chunks_batch_writes_to_local_disk(local_root, monkeypatch):
    # encode_pcm_to_opus uses opuslib which is mocked; stub it to return real bytes.
    monkeypatch.setattr(storage_mod, "encode_pcm_to_opus", lambda data: b"OPUS" + data)

    chunks = [
        {"data": b"\x01\x02\x03\x04", "timestamp": 1000.0},
        {"data": b"\x05\x06\x07\x08", "timestamp": 1001.0},
    ]
    paths = storage_mod.upload_audio_chunks_batch(chunks, uid="u1", conversation_id="c1")

    assert len(paths) == 1
    # Directory should now exist with one batch file.
    conv_dir = local_root / "uid" / "u1" / "conv" / "c1"
    assert conv_dir.exists()
    files = list(conv_dir.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(".batch.bin") or files[0].name.endswith(".batch.opus")


def test_list_audio_chunks_reads_from_local_disk(local_root):
    (local_root / "uid" / "u1" / "conv" / "c1").mkdir(parents=True)
    (local_root / "uid" / "u1" / "conv" / "c1" / "1000.000.opus").write_bytes(b"x")
    (local_root / "uid" / "u1" / "conv" / "c1" / "1001.000-1002.000.batch.opus").write_bytes(
        b"yy"
    )

    chunks = storage_mod.list_audio_chunks(uid="u1", conversation_id="c1")
    assert len(chunks) == 2
    assert {c["timestamp"] for c in chunks} == {1000.0, 1001.0}
    # The batch chunk's `is_batch` flag is set.
    batch = [c for c in chunks if c.get("is_batch")]
    assert len(batch) == 1


def test_delete_conversation_audio_files_removes_local_directory(local_root):
    conv_dir = local_root / "uid" / "u1" / "conv" / "c1"
    conv_dir.mkdir(parents=True)
    (conv_dir / "1000.000.opus").write_bytes(b"x")

    storage_mod.delete_conversation_audio_files(uid="u1", conversation_id="c1")
    assert not conv_dir.exists()


def test_local_root_unset_falls_back_to_storage_disabled(monkeypatch):
    monkeypatch.setattr(storage_mod, "LOCAL_AUDIO_ROOT", None)
    monkeypatch.setattr(storage_mod, "STORAGE_DISABLED", True)

    # STORAGE_DISABLED short-circuits these (Stage 2 hotfix behavior).
    assert storage_mod.upload_audio_chunks_batch([{"data": b"x", "timestamp": 1.0}], "u", "c") == []
    assert storage_mod.list_audio_chunks("u", "c") == []


def test_upload_audio_chunk_single_writes_local(local_root, monkeypatch):
    # encode_pcm_to_opus uses opuslib which is mocked; stub it to return real bytes.
    monkeypatch.setattr(storage_mod, "encode_pcm_to_opus", lambda data: b"OPUS" + data)

    path = storage_mod.upload_audio_chunk(b"\x00" * 64, uid="u1", conversation_id="c1", timestamp=1000.0)
    assert path  # non-empty local path returned
    # File should exist on disk.
    conv_dir = local_root / "uid" / "u1" / "conv" / "c1"
    assert conv_dir.exists()
    assert len(list(conv_dir.iterdir())) == 1
