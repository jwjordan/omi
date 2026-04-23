"""Tests for utils/stt/enrollment.py."""

from unittest.mock import MagicMock, patch


def test_learn_from_cluster_user_calls_append_user(tmp_path, monkeypatch):
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", str(tmp_path))
    from utils.stt import enrollment
    import importlib
    importlib.reload(enrollment)

    with patch("utils.stt.enrollment._extract_cluster_embedding") as extract, \
         patch("utils.stt.enrollment.users_db") as users_db:
        extract.return_value = [0.1] * 256
        users_db.append_user_speaker_embedding.return_value = True

        ok = enrollment.learn_from_cluster(
            uid="u1",
            conversation_id="c1",
            cluster_label="SPEAKER_00",
            target={"is_user": True, "person_id": None},
        )

        assert ok is True
        users_db.append_user_speaker_embedding.assert_called_once_with("u1", [0.1] * 256)
        users_db.append_person_speech_sample_embedding.assert_not_called()


def test_learn_from_cluster_person_calls_append_person(tmp_path, monkeypatch):
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", str(tmp_path))
    from utils.stt import enrollment
    import importlib
    importlib.reload(enrollment)

    with patch("utils.stt.enrollment._extract_cluster_embedding") as extract, \
         patch("utils.stt.enrollment.users_db") as users_db:
        extract.return_value = [0.2] * 256

        ok = enrollment.learn_from_cluster(
            uid="u1",
            conversation_id="c1",
            cluster_label="SPEAKER_02",
            target={"is_user": False, "person_id": "grayson-id"},
        )

        assert ok is True
        users_db.append_person_speech_sample_embedding.assert_called_once_with(
            "u1", "grayson-id", [0.2] * 256
        )
        users_db.append_user_speaker_embedding.assert_not_called()


def test_learn_returns_false_on_extract_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", str(tmp_path))
    from utils.stt import enrollment
    import importlib
    importlib.reload(enrollment)

    with patch("utils.stt.enrollment._extract_cluster_embedding") as extract, \
         patch("utils.stt.enrollment.users_db"):
        extract.return_value = None  # simulate cluster audio too short / ffmpeg fail

        ok = enrollment.learn_from_cluster(
            uid="u1",
            conversation_id="c1",
            cluster_label="SPEAKER_02",
            target={"is_user": False, "person_id": "p1"},
        )
        assert ok is False


def test_learn_noop_when_cluster_label_is_none(tmp_path, monkeypatch):
    """If raw_speaker is None on the tagged segment (old conversation,
    pre-Stage-1c, or diarization never ran), skip enrollment gracefully."""
    monkeypatch.setenv("PENDANT_LOCAL_AUDIO_ROOT", str(tmp_path))
    from utils.stt import enrollment
    import importlib
    importlib.reload(enrollment)

    with patch("utils.stt.enrollment._extract_cluster_embedding") as extract, \
         patch("utils.stt.enrollment.users_db"):
        ok = enrollment.learn_from_cluster(
            uid="u1",
            conversation_id="c1",
            cluster_label=None,
            target={"is_user": False, "person_id": "p1"},
        )
        assert ok is False
        extract.assert_not_called()
