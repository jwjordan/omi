import os
import re
from typing import Optional

import av

from fastapi import APIRouter, UploadFile, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydub import AudioSegment

from database.redis_db import set_speech_profile_duration
from database.users import set_user_speaker_embedding
from utils.other import endpoints as auth
from utils.other.storage import (
    upload_profile_audio,
    get_profile_audio_if_exists,
    delete_additional_profile_audio,
    get_additional_profile_recordings,
    delete_user_person_speech_sample,
    get_user_person_speech_samples,
    get_user_has_speech_profile,
    verify_signed_local_profile_url,
)
from utils.stt.speaker_embedding import extract_embedding
from utils.stt.vad import apply_vad_for_speech_profile
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/v3/speech-profile', tags=['v3'])
def has_speech_profile(uid: str = Depends(auth.get_current_user_uid)):
    return {'has_profile': get_user_has_speech_profile(uid, max_age_days=90)}


@router.get('/v4/speech-profile', tags=['v3'])
def get_speech_profile(uid: str = Depends(auth.get_current_user_uid)):
    return {'url': get_profile_audio_if_exists(uid, download=False)}


_RANGE_RE = re.compile(r'^bytes=(\d*)-(\d*)$')


def _range_iter(path: str, start: int, end: int, chunk: int = 65536):
    with open(path, 'rb') as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get('/v4/speech-profile/audio', tags=['v3'])
def serve_signed_local_profile_audio(
    request: Request,
    p: str = Query(...),
    exp: int = Query(...),
    sig: str = Query(...),
):
    """Serve speech-profile or people-profile audio under STORAGE_DISABLED via
    HMAC-signed URLs minted by storage.py. Anonymous (no bearer) so the
    iOS just_audio (AVPlayer) can fetch it the same way it fetches a GCS
    signed URL. Honors HTTP Range so AVPlayer can probe duration without
    stalling. Signature covers path+expiry; expired or tampered links 403.
    """
    full = verify_signed_local_profile_url(p, exp, sig)
    if not full:
        raise HTTPException(status_code=403, detail='invalid or expired signature')

    size = os.path.getsize(full)
    range_header = request.headers.get('range') or request.headers.get('Range')

    if not range_header:
        return FileResponse(
            full,
            media_type='audio/wav',
            headers={'Accept-Ranges': 'bytes', 'Content-Length': str(size)},
        )

    m = _RANGE_RE.match(range_header.strip())
    if not m:
        return Response(status_code=416, headers={'Content-Range': f'bytes */{size}'})
    start_s, end_s = m.group(1), m.group(2)
    if start_s == '' and end_s == '':
        return Response(status_code=416, headers={'Content-Range': f'bytes */{size}'})
    if start_s == '':
        # suffix range: last N bytes
        length = int(end_s)
        if length <= 0:
            return Response(status_code=416, headers={'Content-Range': f'bytes */{size}'})
        start = max(size - length, 0)
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    if start >= size or end >= size or start > end:
        return Response(status_code=416, headers={'Content-Range': f'bytes */{size}'})

    return StreamingResponse(
        _range_iter(full, start, end),
        status_code=206,
        media_type='audio/wav',
        headers={
            'Accept-Ranges': 'bytes',
            'Content-Range': f'bytes {start}-{end}/{size}',
            'Content-Length': str(end - start + 1),
        },
    )


# ******************************************
# ************* UPLOAD SAMPLE **************
# ******************************************

# Consist of bytes (for initiating deepgram)
# and audio itself, which we use on post-processing to use speechbrain model


@router.post('/v3/upload-audio', tags=['v3'])
def upload_profile(file: UploadFile, uid: str = Depends(auth.get_current_user_uid)):
    os.makedirs(f'_temp/{uid}', exist_ok=True)
    file_path = f"_temp/{uid}/{file.filename}"
    with open(file_path, 'wb') as f:
        f.write(file.file.read())

    aseg = AudioSegment.from_wav(file_path)
    if aseg.frame_rate != 16000:
        raise HTTPException(status_code=400, detail="Invalid codec, must be opus 16khz.")

    if aseg.duration_seconds < 5 or aseg.duration_seconds > 120:
        raise HTTPException(status_code=400, detail="Audio duration is invalid (must be 5-120 seconds)")

    apply_vad_for_speech_profile(file_path)

    # Write-ahead: Cache exact duration after VAD processing (use av for fast header-only read)
    with av.open(file_path) as container:
        duration = (float(container.duration) / av.time_base) + 5 if container.duration else 0
    set_speech_profile_duration(uid, duration)

    url = upload_profile_audio(file_path, uid)

    # Extract and store speaker embedding for user identification in listen sessions
    try:
        embedding = extract_embedding(file_path)
        set_user_speaker_embedding(uid, embedding.flatten().tolist())
        logger.info(f"Speech profile: stored speaker embedding for {uid}")
    except Exception as e:
        logger.error(f"Speech profile: failed to extract/store speaker embedding for {uid}: {e}")

    return {"url": url}


# ******************************************************
# ********** SPEECH SAMPLES FROM CONVERSATION **********
# ******************************************************


@router.delete('/v3/speech-profile/expand', tags=['v3'])
def delete_extra_speech_profile_sample(
    memory_id: str, segment_idx: int, person_id: Optional[str] = None, uid: str = Depends(auth.get_current_user_uid)
):
    logger.info(f'delete_extra_speech_profile_sample {memory_id} {segment_idx} {person_id} {uid}')
    file_name = f'{memory_id}_segment_{segment_idx}.wav'
    if person_id == 'null':
        person_id = None

    if person_id:
        delete_user_person_speech_sample(uid, person_id, file_name)
    else:
        delete_additional_profile_audio(uid, file_name)

    return {'status': 'ok'}


@router.get('/v3/speech-profile/expand', tags=['v3'])
def get_extra_speech_profile_samples(person_id: Optional[str] = None, uid: str = Depends(auth.get_current_user_uid)):
    if person_id:
        return get_user_person_speech_samples(uid, person_id)
    return get_additional_profile_recordings(uid)
