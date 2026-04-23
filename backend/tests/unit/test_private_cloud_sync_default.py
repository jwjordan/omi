"""Verify private_cloud_sync_enabled default honors PENDANT_LOCAL_AUDIO_ROOT."""

import importlib
import sys
from datetime import datetime, timezone


def test_default_is_false_when_local_root_unset(monkeypatch):
    monkeypatch.delenv("PENDANT_LOCAL_AUDIO_ROOT", raising=False)
    # Remove cached module to force reimport with new env
    sys.modules.pop("models.conversation", None)
    import models.conversation
    importlib.reload(models.conversation)
    from models.conversation import Conversation
    from models.structured import Structured

    now = datetime.now(timezone.utc)
    conv = Conversation(
        id="c1",
        created_at=now,
        started_at=now,
        finished_at=now,
        structured=Structured(title="Test"),
    )
    assert conv.private_cloud_sync_enabled is False


def test_default_is_true_when_local_root_set(monkeypatch):
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", "/tmp/x")
    # Remove cached module to force reimport with new env
    sys.modules.pop("models.conversation", None)
    import models.conversation
    importlib.reload(models.conversation)
    from models.conversation import Conversation
    from models.structured import Structured

    now = datetime.now(timezone.utc)
    conv = Conversation(
        id="c1",
        created_at=now,
        started_at=now,
        finished_at=now,
        structured=Structured(title="Test"),
    )
    assert conv.private_cloud_sync_enabled is True
