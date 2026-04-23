"""Verify TranscriptSegment carries a raw_speaker field for Stage 1c."""

from models.transcript_segment import TranscriptSegment


def test_raw_speaker_defaults_to_none():
    seg = TranscriptSegment(text="hi", is_user=True, start=0.0, end=1.0)
    assert seg.raw_speaker is None


def test_raw_speaker_round_trips_through_dict():
    seg = TranscriptSegment(
        text="hi", is_user=False, start=0.0, end=1.0, raw_speaker="SPEAKER_02"
    )
    d = seg.dict()
    assert d["raw_speaker"] == "SPEAKER_02"
    round_tripped = TranscriptSegment(**d)
    assert round_tripped.raw_speaker == "SPEAKER_02"


def test_raw_speaker_accepts_none_from_dict():
    seg = TranscriptSegment(
        text="hi", is_user=False, start=0.0, end=1.0, raw_speaker=None
    )
    assert seg.raw_speaker is None
