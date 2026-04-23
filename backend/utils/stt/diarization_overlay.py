"""Overlay diarization cluster labels onto word-level transcript segments.

Given:
  - segments: list[TranscriptSegment] (word-level from whisper-shim)
  - clusters: [{'speaker': 'SPEAKER_00', 'start': float, 'end': float}, ...]
  - matches:  {cluster_label: {'is_user': bool, 'person_id': Optional[str]}}
  - name_lookup: {person_id: human_name}

Mutates segments in place: for each word, find the cluster whose range
contains the word's midpoint. Later cluster wins on tie (see spec
Appendix A). If no cluster claims the word, leave the segment untouched
(keeps its default SPEAKER_00 fallback but raw_speaker stays None so
retagging can detect this state).
"""

from typing import Dict, List, Optional

from models.transcript_segment import TranscriptSegment


def apply(
    segments: List[TranscriptSegment],
    clusters: List[Dict[str, float]],
    matches: Dict[str, Dict[str, object]],
    name_lookup: Dict[str, str],
) -> None:
    for seg in segments:
        midpoint = (seg.start + seg.end) / 2.0
        hit: Optional[Dict[str, float]] = None
        for c in clusters:
            if c['start'] <= midpoint <= c['end']:
                hit = c  # later cluster wins on overlap
        if hit is None:
            continue

        cluster_label = hit['speaker']
        seg.raw_speaker = cluster_label
        m = matches.get(cluster_label, {})

        if m.get('is_user'):
            seg.is_user = True
            seg.person_id = None
            seg.speaker = 'user'
        elif m.get('person_id'):
            seg.is_user = False
            seg.person_id = m['person_id']
            seg.speaker = name_lookup.get(m['person_id'], cluster_label)
        else:
            seg.is_user = False
            seg.person_id = None
            seg.speaker = cluster_label
