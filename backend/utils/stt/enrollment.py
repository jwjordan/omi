"""Extract a voice fingerprint from a tagged diarization cluster.

Called from the PATCH /v1/conversations/{id}/...assign* handlers when the
user tags a segment as `is_user` or `person_id`. Reads the conversation's
stored transcript, finds all segments in the cluster, pulls audio for
the longest contiguous sub-range, POSTs to the diarizer's /v2/embedding
endpoint, and persists the embedding on the user or person.

Silent fallback: returns False when cluster_label is None, audio can't
be extracted, or the embedding service returns an error. The assign
handler's mutation of person_id / is_user still succeeds; we just fail
to learn a fingerprint this time.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import wave
from typing import Dict, List, Optional

import httpx

import database.users as users_db
from database.conversations import get_conversation as _get_conversation, _decrypt_conversation_data
from utils.other.storage import list_audio_chunks

logger = logging.getLogger(__name__)

DIARIZER_URL = os.getenv('HOSTED_DIARIZER_API_URL', 'http://diarizer:8080')
MIN_EMBED_AUDIO_SECONDS = 0.5


def _find_cluster_windows(
    segments: list, cluster_label: str
) -> List[Dict[str, float]]:
    """Return [{'start': float, 'end': float}, ...] for each contiguous
    span where raw_speaker matches cluster_label."""
    windows: List[Dict[str, float]] = []
    current: Optional[Dict[str, float]] = None
    for seg in segments:
        raw = seg.get('raw_speaker') if isinstance(seg, dict) else getattr(seg, 'raw_speaker', None)
        start = seg.get('start') if isinstance(seg, dict) else seg.start
        end = seg.get('end') if isinstance(seg, dict) else seg.end
        if raw == cluster_label:
            if current is None:
                current = {'start': start, 'end': end}
            else:
                current['end'] = end
        else:
            if current is not None:
                windows.append(current)
                current = None
    if current is not None:
        windows.append(current)
    return windows


def _extract_cluster_embedding(
    uid: str, conversation_id: str, cluster_label: str
) -> Optional[List[float]]:
    """Build a WAV from the longest window of this cluster's audio and POST
    to the embedding API. Returns the embedding vector or None on failure."""
    raw_conv = _get_conversation(uid, conversation_id)
    if not raw_conv:
        logger.warning("enrollment: conversation %s/%s not found", uid, conversation_id)
        return None
    conv = _decrypt_conversation_data(raw_conv, uid)
    segments = conv.get('transcript_segments') or []
    if not segments:
        logger.warning("enrollment: conversation %s/%s has no transcript", uid, conversation_id)
        return None

    windows = _find_cluster_windows(segments, cluster_label)
    if not windows:
        logger.warning(
            "enrollment: no segments with raw_speaker=%s in %s/%s",
            cluster_label, uid, conversation_id,
        )
        return None
    longest = max(windows, key=lambda w: w['end'] - w['start'])
    duration = longest['end'] - longest['start']
    if duration < MIN_EMBED_AUDIO_SECONDS:
        logger.warning(
            "enrollment: longest %s window is %.2fs (< %.2fs) in %s/%s",
            cluster_label, duration, MIN_EMBED_AUDIO_SECONDS, uid, conversation_id,
        )
        return None

    chunks = list_audio_chunks(uid, conversation_id)
    if not chunks:
        logger.warning("enrollment: no chunks for %s/%s", uid, conversation_id)
        return None

    # Decode all chunks to one PCM16 16kHz mono stream via ffmpeg.
    sample_rate = 16000
    pcm_parts = []
    for c in chunks:
        proc = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', c['path'], '-f', 's16le',
             '-ar', str(sample_rate), '-ac', '1', '-'],
            capture_output=True, check=False,
        )
        if proc.returncode != 0:
            logger.warning(
                "enrollment: ffmpeg failed for %s: %s",
                c['path'], proc.stderr.decode(errors='replace')[:200],
            )
            continue
        pcm_parts.append(proc.stdout)
    if not pcm_parts:
        return None
    full_pcm = b''.join(pcm_parts)

    window_start_bytes = int(longest['start'] * sample_rate) * 2
    window_end_bytes = int(longest['end'] * sample_rate) * 2
    window_pcm = full_pcm[window_start_bytes:window_end_bytes]
    if not window_pcm:
        return None

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(window_pcm)
    wav_buf.seek(0)

    try:
        resp = httpx.post(
            f"{DIARIZER_URL}/v2/embedding",
            files={'file': ('cluster.wav', wav_buf, 'audio/wav')},
            timeout=120.0,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning("enrollment: embedding call failed: %s", e)
        return None

    body = resp.json()
    if isinstance(body, list):
        return body
    emb = body.get('embedding')
    if isinstance(emb, list):
        return emb
    logger.warning("enrollment: unexpected embedding response shape: %r", body)
    return None


def learn_from_cluster(
    uid: str,
    conversation_id: str,
    cluster_label: Optional[str],
    target: Dict[str, object],
) -> bool:
    """Extract + persist a voice embedding for the tagged cluster.

    Args:
        uid: user id.
        conversation_id: conversation whose audio is on disk.
        cluster_label: diarizer cluster label (e.g. "SPEAKER_02"). If None,
            this is an old conversation that predates Stage 1c and we
            quietly skip enrollment.
        target: {'is_user': bool, 'person_id': Optional[str]}. Exactly one
            of is_user / person_id identifies the tag target.

    Returns:
        True if an embedding was successfully extracted and persisted;
        False on any failure (including cluster_label is None).
    """
    if not cluster_label:
        return False
    embedding = _extract_cluster_embedding(uid, conversation_id, cluster_label)
    if embedding is None:
        return False

    if target.get('is_user'):
        users_db.append_user_speaker_embedding(uid, embedding)
        return True
    pid = target.get('person_id')
    if pid:
        users_db.append_person_speech_sample_embedding(uid, pid, embedding)
        return True
    return False
