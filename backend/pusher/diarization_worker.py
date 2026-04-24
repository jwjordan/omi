"""Async diarization worker thread.

Runs inside pusher. Polls pending_diarizations every POLL_INTERVAL_S; for
each job, merges chunks from the local audio volume, calls the diarizer
service, extracts per-cluster voice embeddings, matches against the
user's stored fingerprints, overlays speaker labels onto the transcript,
and updates the conversation.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import threading
import time
import wave
from typing import Dict, List, Optional, Tuple

import httpx

from database._client import db
import database.conversations as conversations_db
import database.users as users_db
from utils.other.storage import list_audio_chunks
from utils.stt import diarization_overlay, speaker_matching

logger = logging.getLogger(__name__)

DIARIZER_URL = os.getenv('HOSTED_DIARIZER_API_URL', 'http://diarizer:8080')
POLL_INTERVAL_S = float(os.getenv('DIARIZATION_POLL_INTERVAL_S', '5'))
MAX_ATTEMPTS = int(os.getenv('DIARIZATION_MAX_ATTEMPTS', '3'))
STALE_RUNNING_MINUTES = int(os.getenv('DIARIZATION_STALE_MINUTES', '10'))
MIN_CLUSTER_EMBED_SECONDS = 0.5
SAMPLE_RATE = 16000


def _merge_chunks_to_pcm(uid: str, conversation_id: str) -> Optional[bytes]:
    chunks = list_audio_chunks(uid, conversation_id)
    if not chunks:
        return None
    parts = []
    for c in chunks:
        proc = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', c['path'], '-f', 's16le',
             '-ar', str(SAMPLE_RATE), '-ac', '1', '-'],
            capture_output=True, check=False,
        )
        if proc.returncode != 0:
            logger.warning("ffmpeg failed on %s: %s",
                           c['path'], proc.stderr.decode(errors='replace')[:200])
            continue
        parts.append(proc.stdout)
    if not parts:
        return None
    return b''.join(parts)


def _pcm_to_wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def _post_diarize(wav_bytes: bytes) -> List[Dict[str, float]]:
    resp = httpx.post(
        f"{DIARIZER_URL}/v1/diarization",
        files={'file': ('conv.wav', io.BytesIO(wav_bytes), 'audio/wav')},
        timeout=3600.0,
    )
    resp.raise_for_status()
    return resp.json()


def _post_embedding(wav_bytes: bytes) -> Optional[List[float]]:
    try:
        resp = httpx.post(
            f"{DIARIZER_URL}/v2/embedding",
            files={'file': ('cluster.wav', io.BytesIO(wav_bytes), 'audio/wav')},
            timeout=120.0,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning("embedding call failed: %s", e)
        return None
    body = resp.json()
    if isinstance(body, list):
        return body
    return body.get('embedding')


def _slice_pcm(full_pcm: bytes, start: float, end: float) -> bytes:
    start_bytes = int(start * SAMPLE_RATE) * 2
    end_bytes = int(end * SAMPLE_RATE) * 2
    return full_pcm[start_bytes:end_bytes]


def _longest_cluster_window(
    clusters: List[Dict[str, float]], speaker: str
) -> Optional[Tuple[float, float]]:
    windows = [(c['start'], c['end']) for c in clusters if c['speaker'] == speaker]
    if not windows:
        return None
    return max(windows, key=lambda w: w[1] - w[0])


def _run_job(uid: str, conversation_id: str) -> bool:
    full_pcm = _merge_chunks_to_pcm(uid, conversation_id)
    if full_pcm is None:
        logger.warning("diarize: no audio for %s/%s, marking failed", uid, conversation_id)
        return False

    wav = _pcm_to_wav_bytes(full_pcm)
    clusters = _post_diarize(wav)

    distinct_speakers = {c['speaker'] for c in clusters}
    matches: Dict[str, Dict[str, object]] = {}
    name_lookup: Dict[str, str] = {}
    for speaker in distinct_speakers:
        window = _longest_cluster_window(clusters, speaker)
        if window is None:
            matches[speaker] = {'is_user': False, 'person_id': None}
            continue
        duration = window[1] - window[0]
        if duration < MIN_CLUSTER_EMBED_SECONDS:
            matches[speaker] = {'is_user': False, 'person_id': None}
            continue
        cluster_pcm = _slice_pcm(full_pcm, window[0], window[1])
        cluster_wav = _pcm_to_wav_bytes(cluster_pcm)
        emb = _post_embedding(cluster_wav)
        if emb is None:
            matches[speaker] = {'is_user': False, 'person_id': None}
            continue
        matches[speaker] = speaker_matching.match(uid, emb)

    matched_person_ids = {
        m['person_id'] for m in matches.values()
        if m.get('person_id')
    }
    if matched_person_ids:
        for p in users_db.get_people_with_embeddings(uid):
            if p['person_id'] in matched_person_ids and p.get('name'):
                name_lookup[p['person_id']] = p['name']

    raw_conv = conversations_db.get_conversation(uid, conversation_id)
    if not raw_conv:
        logger.warning("diarize: conversation %s/%s vanished", uid, conversation_id)
        return False
    decrypted = conversations_db._decrypt_conversation_data(raw_conv, uid)
    from models.transcript_segment import TranscriptSegment
    segments = [TranscriptSegment(**s) for s in (decrypted.get('transcript_segments') or [])]

    diarization_overlay.apply(segments, clusters, matches, name_lookup)

    conversations_db.update_conversation_segments(
        uid, conversation_id, [s.dict() for s in segments]
    )
    return True


def run_one() -> bool:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT conversation_id, uid, attempts
                FROM pending_diarizations
                WHERE state = 'pending'
                ORDER BY enqueued_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """
            )
            row = cur.fetchone()
            if row is None:
                return False
            conv_id, uid, attempts = row
            cur.execute(
                """
                UPDATE pending_diarizations
                SET state = 'running', attempts = attempts + 1, started_at = now()
                WHERE conversation_id = %s
                """,
                (conv_id,),
            )

    try:
        ok = _run_job(uid, conv_id)
    except Exception as e:
        logger.exception("diarization job %s/%s failed: %s", uid, conv_id, e)
        with db.connection() as conn:
            with conn.cursor() as cur:
                if attempts + 1 >= MAX_ATTEMPTS:
                    cur.execute(
                        """
                        UPDATE pending_diarizations
                        SET state = 'failed', error = %s, finished_at = now()
                        WHERE conversation_id = %s
                        """,
                        (str(e)[:1000], conv_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE pending_diarizations
                        SET state = 'pending', error = %s
                        WHERE conversation_id = %s
                        """,
                        (str(e)[:1000], conv_id),
                    )
        return True

    with db.connection() as conn:
        with conn.cursor() as cur:
            if ok:
                cur.execute(
                    """
                    UPDATE pending_diarizations
                    SET state = 'done', finished_at = now()
                    WHERE conversation_id = %s
                    """,
                    (conv_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE pending_diarizations
                    SET state = 'failed', error = 'no audio or conversation missing',
                        finished_at = now()
                    WHERE conversation_id = %s
                    """,
                    (conv_id,),
                )
    return True


def requeue_stale_running() -> int:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE pending_diarizations
                SET state = 'pending'
                WHERE state = 'running'
                  AND started_at < now() - interval '{STALE_RUNNING_MINUTES} minutes'
                """
            )
            return cur.rowcount


_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def _loop():
    logger.info("diarization worker started; polling every %ss", POLL_INTERVAL_S)
    try:
        n = requeue_stale_running()
        if n:
            logger.info("requeued %d stale running jobs at startup", n)
    except Exception as e:
        logger.warning("stale-running requeue failed: %s", e)

    while not _stop.is_set():
        try:
            processed = run_one()
        except Exception as e:
            logger.exception("worker loop error: %s", e)
            processed = False
        if not processed:
            _stop.wait(POLL_INTERVAL_S)


def start() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name='diarization-worker', daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
