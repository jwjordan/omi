"""Tests for utils/stt/diarization_overlay.py."""

from models.transcript_segment import TranscriptSegment


def _seg(text: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, end=end, is_user=False)


def test_maps_word_midpoint_into_cluster_range():
    from utils.stt.diarization_overlay import apply

    segments = [_seg("hello", 0.0, 1.0), _seg("there", 1.5, 2.5)]
    clusters = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.2},
        {"speaker": "SPEAKER_01", "start": 1.3, "end": 3.0},
    ]
    matches = {
        "SPEAKER_00": {"is_user": True, "person_id": None},
        "SPEAKER_01": {"is_user": False, "person_id": "grayson-id"},
    }
    name_lookup = {"grayson-id": "Grayson"}

    apply(segments, clusters, matches, name_lookup)

    assert segments[0].is_user is True
    assert segments[0].person_id is None
    assert segments[0].speaker == "user"
    assert segments[0].raw_speaker == "SPEAKER_00"

    assert segments[1].is_user is False
    assert segments[1].person_id == "grayson-id"
    assert segments[1].speaker == "Grayson"
    assert segments[1].raw_speaker == "SPEAKER_01"


def test_unmatched_cluster_keeps_speaker_label():
    from utils.stt.diarization_overlay import apply

    segments = [_seg("x", 1.0, 2.0)]
    clusters = [{"speaker": "SPEAKER_02", "start": 0.0, "end": 3.0}]
    matches = {"SPEAKER_02": {"is_user": False, "person_id": None}}

    apply(segments, clusters, matches, name_lookup={})

    assert segments[0].speaker == "SPEAKER_02"
    assert segments[0].raw_speaker == "SPEAKER_02"
    assert segments[0].is_user is False
    assert segments[0].person_id is None


def test_word_outside_all_clusters_left_alone():
    from utils.stt.diarization_overlay import apply

    segments = [_seg("x", 10.0, 11.0)]
    clusters = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]
    matches = {"SPEAKER_00": {"is_user": False, "person_id": None}}

    apply(segments, clusters, matches, name_lookup={})

    # Default speaker="SPEAKER_00", raw_speaker stays None (no cluster claimed this word).
    assert segments[0].raw_speaker is None


def test_tie_break_later_cluster_wins():
    """If two clusters have overlapping ranges around a word's midpoint,
    iteration order means the later cluster wins."""
    from utils.stt.diarization_overlay import apply

    segments = [_seg("x", 1.0, 2.0)]
    clusters = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
        {"speaker": "SPEAKER_01", "start": 0.0, "end": 3.0},
    ]
    matches = {
        "SPEAKER_00": {"is_user": False, "person_id": None},
        "SPEAKER_01": {"is_user": False, "person_id": None},
    }

    apply(segments, clusters, matches, name_lookup={})
    assert segments[0].raw_speaker == "SPEAKER_01"


def test_missing_match_for_cluster_treated_as_unmatched():
    from utils.stt.diarization_overlay import apply

    segments = [_seg("x", 1.0, 2.0)]
    clusters = [{"speaker": "SPEAKER_99", "start": 0.0, "end": 3.0}]
    matches = {}  # matcher didn't produce an entry for this cluster

    apply(segments, clusters, matches, name_lookup={})
    assert segments[0].speaker == "SPEAKER_99"
    assert segments[0].raw_speaker == "SPEAKER_99"


def test_person_id_without_name_falls_back_to_person_id():
    """If the name_lookup is missing an entry, render the cluster label
    rather than an opaque UUID."""
    from utils.stt.diarization_overlay import apply

    segments = [_seg("x", 1.0, 2.0)]
    clusters = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 3.0}]
    matches = {"SPEAKER_01": {"is_user": False, "person_id": "uuid-no-name"}}

    apply(segments, clusters, matches, name_lookup={})
    # person_id stored, speaker falls back to cluster label
    assert segments[0].person_id == "uuid-no-name"
    assert segments[0].speaker == "SPEAKER_01"
