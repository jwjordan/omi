import base64
import datetime
import hashlib
import hmac
import io
import json
import os
import struct
import time as _time
import wave
from typing import List
from concurrent.futures import as_completed

from utils.executors import storage_executor

import opuslib
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound as BlobNotFound
from google.cloud.exceptions import NotFound

from database.redis_db import cache_signed_url, get_cached_signed_url
from utils import encryption
from database import users as users_db
import logging

logger = logging.getLogger(__name__)

# Opus encoding constants
OPUS_SAMPLE_RATE = 16000
OPUS_CHANNELS = 1
OPUS_FRAME_DURATION_MS = 20  # 20ms frames (standard for voice)
OPUS_FRAME_SIZE = OPUS_SAMPLE_RATE * OPUS_FRAME_DURATION_MS // 1000  # 320 samples per frame

# Valid private cloud sync extensions (longest first for correct matching).
# .batch.wav / .wav are the Stage 1c local-disk format (raw PCM in a WAV
# container — ffmpeg and soundfile decode natively). Upstream emits
# .batch.bin / .batch.enc / .opus variants.
PRIVATE_CLOUD_EXTENSIONS = ['.batch.wav', '.batch.enc', '.batch.bin', '.opus.enc', '.opus', '.wav', '.enc', '.bin']

if os.environ.get('SERVICE_ACCOUNT_JSON'):
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    storage_client = storage.Client(credentials=credentials)
else:
    storage_client = storage.Client()

speech_profiles_bucket = os.getenv('BUCKET_SPEECH_PROFILES')
postprocessing_audio_bucket = os.getenv('BUCKET_POSTPROCESSING')
memories_recordings_bucket = os.getenv('BUCKET_MEMORIES_RECORDINGS')
private_cloud_sync_bucket = os.getenv('BUCKET_PRIVATE_CLOUD_SYNC', 'omi-private-cloud-sync')
syncing_local_bucket = os.getenv('BUCKET_TEMPORAL_SYNC_LOCAL')
omi_apps_bucket = os.getenv('BUCKET_PLUGINS_LOGOS')
app_thumbnails_bucket = os.getenv('BUCKET_APP_THUMBNAILS')
chat_files_bucket = os.getenv('BUCKET_CHAT_FILES')
desktop_updates_bucket = os.getenv('BUCKET_DESKTOP_UPDATES')

# pendant-stack: when STORAGE_DISABLED=true, short-circuit storage functions to
# return empty / False instead of calling GCS. Firebase Storage now requires the
# Blaze plan (billing) and the pendant use case doesn't need blob storage for
# Stage 1b — speech profile training and conversation-audio uploads are both
# deferred. Writes become no-ops, existence checks return False, listings empty.
STORAGE_DISABLED = os.getenv('STORAGE_DISABLED', 'false').lower() in ('true', '1', 'yes')
if STORAGE_DISABLED:
    logger.warning("STORAGE_DISABLED=true: all cloud-storage functions will short-circuit")

# Stage 1c: local-disk audio storage. When set, chunk upload/list/delete
# operations target this directory instead of GCS. The layout mirrors
# GCS: <root>/uid/<uid>/conv/<conversation_id>/<timestamp>.opus
LOCAL_AUDIO_ROOT = os.getenv('PENDANT_LOCAL_AUDIO_ROOT')
if LOCAL_AUDIO_ROOT:
    logger.info("PENDANT_LOCAL_AUDIO_ROOT=%s: chunk storage is local-disk", LOCAL_AUDIO_ROOT)

# Speech-profile + people-profile audio store. When STORAGE_DISABLED=true,
# upload/list/delete target this directory and the public-URL path returns
# HMAC-signed links to a backend route that streams the file. Mirrors GCS
# layout: <root>/<uid>/speech_profile.wav,
# <root>/<uid>/additional_profile_recordings/<file>,
# <root>/<uid>/people_profiles/<person_id>/<file>.
LOCAL_PROFILE_ROOT = os.getenv('PENDANT_LOCAL_PROFILE_ROOT', '_temp/profiles')
PUBLIC_API_BASE = os.getenv('PENDANT_PUBLIC_API_BASE', 'https://api.omi.me').rstrip('/')
PROFILE_URL_TTL_S = int(os.getenv('PENDANT_PROFILE_URL_TTL_S', '300'))
if STORAGE_DISABLED:
    logger.info(
        "STORAGE_DISABLED=true: profile audio stored at %s, public URLs base %s",
        LOCAL_PROFILE_ROOT,
        PUBLIC_API_BASE,
    )


def _local_conv_dir(uid: str, conversation_id: str) -> str:
    """Return the local directory for a conversation's chunks (not created)."""
    return os.path.join(LOCAL_AUDIO_ROOT, 'uid', uid, 'conv', conversation_id)


# *******************************************
# ******** LOCAL PROFILE STORAGE HELPERS ****
# *******************************************
def _profile_signing_key() -> bytes:
    secret = os.getenv('PENDANT_PROFILE_SIGNING_KEY') or os.getenv('ENCRYPTION_SECRET') or ''
    if not secret:
        raise RuntimeError(
            'PENDANT_PROFILE_SIGNING_KEY or ENCRYPTION_SECRET must be set when STORAGE_DISABLED=true'
        )
    return hashlib.sha256(secret.encode('utf-8') + b'|profile_audio').digest()


def _local_profile_full_path(rel_path: str) -> str:
    """Return absolute local-disk path for rel_path (e.g. '{uid}/speech_profile.wav')."""
    return os.path.normpath(os.path.join(LOCAL_PROFILE_ROOT, rel_path))


def _make_signed_local_profile_url(rel_path: str, ttl_seconds: int = None) -> str:
    """Build an HMAC-signed URL the iOS app can fetch anonymously (just_audio
    plays GCS-style signed URLs without sending any auth header). Format:
    {base}/v4/speech-profile/audio?p={b64url(rel_path)}&exp={ts}&sig={hex}.
    """
    if ttl_seconds is None:
        ttl_seconds = PROFILE_URL_TTL_S
    p = base64.urlsafe_b64encode(rel_path.encode('utf-8')).rstrip(b'=').decode('ascii')
    exp = int(_time.time()) + ttl_seconds
    msg = f'{p}|{exp}'.encode('ascii')
    sig = hmac.new(_profile_signing_key(), msg, hashlib.sha256).hexdigest()
    return f'{PUBLIC_API_BASE}/v4/speech-profile/audio?p={p}&exp={exp}&sig={sig}'


def verify_signed_local_profile_url(p: str, exp: int, sig: str) -> str | None:
    """Validate a signature and return the absolute file path, or None on failure.
    Used by the audio-serve route in routers/speech_profile.py.
    """
    if not p or not exp or not sig:
        return None
    if int(exp) < int(_time.time()):
        return None
    expected = hmac.new(_profile_signing_key(), f'{p}|{exp}'.encode('ascii'), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    # b64-decode rel_path. Pad back the trailing '=' chars b64decode requires.
    pad = '=' * (-len(p) % 4)
    try:
        rel_path = base64.urlsafe_b64decode(p + pad).decode('utf-8')
    except (ValueError, UnicodeDecodeError):
        return None
    # Reject path-traversal attempts.
    if '..' in rel_path.split('/'):
        return None
    full = _local_profile_full_path(rel_path)
    return full if os.path.isfile(full) else None


def _local_profile_write(rel_path: str, src_path: str) -> str:
    full = _local_profile_full_path(rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    # Copy in chunks to avoid loading large WAVs into memory all at once.
    with open(src_path, 'rb') as src, open(full, 'wb') as dst:
        while True:
            buf = src.read(64 * 1024)
            if not buf:
                break
            dst.write(buf)
    return full


def _local_profile_write_bytes(rel_path: str, data: bytes) -> str:
    full = _local_profile_full_path(rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'wb') as f:
        f.write(data)
    return full


def _local_profile_list(rel_dir: str) -> List[str]:
    """Return list of relative paths (under LOCAL_PROFILE_ROOT) inside rel_dir."""
    full_dir = _local_profile_full_path(rel_dir)
    if not os.path.isdir(full_dir):
        return []
    return [f'{rel_dir}/{name}' for name in sorted(os.listdir(full_dir))]


def _local_profile_delete(rel_path: str) -> bool:
    full = _local_profile_full_path(rel_path)
    if os.path.isfile(full):
        try:
            os.remove(full)
            return True
        except OSError as e:
            logger.warning('local profile delete failed for %s: %s', rel_path, e)
    return False


# *******************************************
# ************* SPEECH PROFILE **************
# *******************************************
def upload_profile_audio(file_path: str, uid: str):
    rel = f'{uid}/speech_profile.wav'
    if STORAGE_DISABLED:
        _local_profile_write(rel, file_path)
        return _make_signed_local_profile_url(rel)
    bucket = storage_client.bucket(speech_profiles_bucket)
    blob = bucket.blob(rel)
    blob.upload_from_filename(file_path)
    return f'https://storage.googleapis.com/{speech_profiles_bucket}/{rel}'


def get_user_has_speech_profile(uid: str, max_age_days: int = None) -> bool:
    if STORAGE_DISABLED:
        # GCS blob check is disabled. Treat presence of a stored speaker
        # embedding as the source of truth — that's the artifact actually
        # consumed by the speaker-ID code path in transcribe.py. The original
        # GCS blob existed only to allow re-extracting the embedding on
        # demand, which we don't need when the embedding is persisted at
        # upload time. max_age_days is ignored here (we don't track creation
        # time for the embedding).
        return users_db.get_user_speaker_embedding(uid) is not None
    bucket = storage_client.bucket(speech_profiles_bucket)
    blob = bucket.blob(f'{uid}/speech_profile.wav')
    if not blob.exists():
        return False

    # Check age if max_age_days is specified
    if max_age_days is not None:
        blob.reload()
        if blob.time_created:
            age = datetime.datetime.now(datetime.timezone.utc) - blob.time_created
            if age.days > max_age_days:
                return False

    return True


def get_profile_audio_if_exists(uid: str, download: bool = True) -> str:
    rel = f'{uid}/speech_profile.wav'
    if STORAGE_DISABLED:
        full = _local_profile_full_path(rel)
        if not os.path.isfile(full):
            return None
        return full if download else _make_signed_local_profile_url(rel)
    bucket = storage_client.bucket(speech_profiles_bucket)
    blob = bucket.blob(rel)
    if blob.exists():
        if download:
            file_path = f'_temp/{uid}_speech_profile.wav'
            blob.download_to_filename(file_path)
            return file_path
        return _get_signed_url(blob, 60)

    return None


def delete_additional_profile_audio(uid: str, file_name: str) -> None:
    rel = f'{uid}/additional_profile_recordings/{file_name}'
    if STORAGE_DISABLED:
        if _local_profile_delete(rel):
            logger.info(f'delete_additional_profile_audio deleting {file_name}')
        return
    bucket = storage_client.bucket(speech_profiles_bucket)
    blob = bucket.blob(rel)
    if blob.exists():
        logger.info(f'delete_additional_profile_audio deleting {file_name}')
        blob.delete()


def get_additional_profile_recordings(uid: str, download: bool = False) -> List[str]:
    rel_dir = f'{uid}/additional_profile_recordings'
    if STORAGE_DISABLED:
        rels = _local_profile_list(rel_dir)
        if download:
            return [_local_profile_full_path(rel) for rel in rels]
        return [_make_signed_local_profile_url(rel) for rel in rels]
    bucket = storage_client.bucket(speech_profiles_bucket)
    blobs = bucket.list_blobs(prefix=f'{rel_dir}/')
    if download:
        paths = []
        for blob in blobs:
            file_path = f'_temp/{uid}_{blob.name.split("/")[-1]}'
            blob.download_to_filename(file_path)
            paths.append(file_path)
        return paths

    return [_get_signed_url(blob, 60) for blob in blobs]


# ********************************************
# ************* PEOPLE PROFILES **************
# ********************************************


def delete_user_person_speech_sample(uid: str, person_id: str, file_name: str) -> None:
    rel = f'{uid}/people_profiles/{person_id}/{file_name}'
    if STORAGE_DISABLED:
        _local_profile_delete(rel)
        return
    bucket = storage_client.bucket(speech_profiles_bucket)
    blob = bucket.blob(rel)
    if blob.exists():
        blob.delete()


def delete_user_person_speech_samples(uid: str, person_id: str) -> None:
    rel_dir = f'{uid}/people_profiles/{person_id}'
    if STORAGE_DISABLED:
        for rel in _local_profile_list(rel_dir):
            _local_profile_delete(rel)
        return
    bucket = storage_client.bucket(speech_profiles_bucket)
    blobs = bucket.list_blobs(prefix=f'{rel_dir}/')
    for blob in blobs:
        blob.delete()


def upload_person_speech_sample_from_bytes(
    audio_bytes: bytes,
    uid: str,
    person_id: str,
    sample_rate: int = 16000,
) -> str:
    """Upload PCM audio bytes as WAV speech sample. Returns the storage path."""
    import uuid as uuid_module

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    filename = f"{uuid_module.uuid4()}.wav"
    rel = f'{uid}/people_profiles/{person_id}/{filename}'
    if STORAGE_DISABLED:
        _local_profile_write_bytes(rel, wav_buffer.getvalue())
        return rel
    bucket = storage_client.bucket(speech_profiles_bucket)
    blob = bucket.blob(rel)
    blob.upload_from_string(wav_buffer.getvalue(), content_type='audio/wav')

    return rel


def get_user_people_ids(uid: str) -> List[str]:
    if STORAGE_DISABLED:
        full_dir = _local_profile_full_path(f'{uid}/people_profiles')
        if not os.path.isdir(full_dir):
            return []
        return [name for name in sorted(os.listdir(full_dir)) if os.path.isdir(os.path.join(full_dir, name))]
    bucket = storage_client.bucket(speech_profiles_bucket)
    blobs = bucket.list_blobs(prefix=f'{uid}/people_profiles/')
    return [blob.name.split("/")[-2] for blob in blobs]


def get_user_person_speech_samples(uid: str, person_id: str, download: bool = False) -> List[str]:
    rel_dir = f'{uid}/people_profiles/{person_id}'
    if STORAGE_DISABLED:
        rels = _local_profile_list(rel_dir)
        if download:
            return [_local_profile_full_path(rel) for rel in rels]
        return [_make_signed_local_profile_url(rel) for rel in rels]
    bucket = storage_client.bucket(speech_profiles_bucket)
    blobs = bucket.list_blobs(prefix=f'{rel_dir}/')
    if download:
        paths = []
        for blob in blobs:
            file_path = f'_temp/{uid}_person_{blob.name.split("/")[-1]}'
            blob.download_to_filename(file_path)
            paths.append(file_path)
        return paths

    return [_get_signed_url(blob, 60) for blob in blobs]


def get_speech_sample_signed_urls(paths: List[str]) -> List[str]:
    """
    Generate signed URLs for speech samples given their GCS paths.
    Uses the paths stored in Firestore instead of listing GCS blobs.

    Args:
        paths: List of storage paths (e.g., '{uid}/people_profiles/{person_id}/{filename}')

    Returns:
        List of signed URLs
    """
    if not paths:
        return []
    if STORAGE_DISABLED:
        return [_make_signed_local_profile_url(p) for p in paths]
    bucket = storage_client.bucket(speech_profiles_bucket)
    signed_urls = []
    for path in paths:
        blob = bucket.blob(path)
        signed_urls.append(_get_signed_url(blob, 60))
    return signed_urls


# ********************************************
# ************* POST PROCESSING **************
# ********************************************
def upload_postprocessing_audio(file_path: str):
    bucket = storage_client.bucket(postprocessing_audio_bucket)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    return f'https://storage.googleapis.com/{postprocessing_audio_bucket}/{file_path}'


def delete_postprocessing_audio(file_path: str):
    bucket = storage_client.bucket(postprocessing_audio_bucket)
    blob = bucket.blob(file_path)
    blob.delete()


# ***********************************
# ************* SDCARD **************
# ***********************************


def upload_sdcard_audio(file_path: str):
    bucket = storage_client.bucket(postprocessing_audio_bucket)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    return f'https://storage.googleapis.com/{postprocessing_audio_bucket}/sdcard/{file_path}'


def download_postprocessing_audio(file_path: str, destination_file_path: str):
    bucket = storage_client.bucket(postprocessing_audio_bucket)
    blob = bucket.blob(file_path)
    blob.download_to_filename(destination_file_path)


# ************************************************
# *********** CONVERSATIONS RECORDINGS ***********
# ************************************************


def upload_conversation_recording(file_path: str, uid: str, conversation_id: str):
    bucket = storage_client.bucket(memories_recordings_bucket)
    path = f'{uid}/{conversation_id}.wav'
    blob = bucket.blob(path)
    blob.upload_from_filename(file_path)
    return f'https://storage.googleapis.com/{memories_recordings_bucket}/{path}'


def get_conversation_recording_if_exists(uid: str, memory_id: str) -> str:
    logger.info(f'get_conversation_recording_if_exists {uid} {memory_id}')
    bucket = storage_client.bucket(memories_recordings_bucket)
    path = f'{uid}/{memory_id}.wav'
    blob = bucket.blob(path)
    if blob.exists():
        file_path = f'_temp/{memory_id}.wav'
        blob.download_to_filename(file_path)
        return file_path
    return None


def delete_all_conversation_recordings(uid: str):
    if not uid:
        return
    bucket = storage_client.bucket(memories_recordings_bucket)
    blobs = bucket.list_blobs(prefix=uid)
    for blob in blobs:
        blob.delete()


# ********************************************
# ************* SYNCING FILES **************
# ********************************************
def get_syncing_file_temporal_url(file_path: str):
    bucket = storage_client.bucket(syncing_local_bucket)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    return f'https://storage.googleapis.com/{syncing_local_bucket}/{file_path}'


def get_syncing_file_temporal_signed_url(file_path: str):
    bucket = storage_client.bucket(syncing_local_bucket)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
    return _get_signed_url(blob, 15)


def delete_syncing_temporal_file(file_path: str):
    bucket = storage_client.bucket(syncing_local_bucket)
    blob = bucket.blob(file_path)
    try:
        blob.delete()
    except BlobNotFound:
        pass


# ************************************************
# *********** PRIVATE CLOUD SYNC *****************
# ************************************************


def encode_pcm_to_opus(pcm_data: bytes, sample_rate: int = OPUS_SAMPLE_RATE, channels: int = OPUS_CHANNELS) -> bytes:
    """
    Encode PCM16 audio to Opus.

    Format: 4-byte little-endian packet count, then for each packet:
    2-byte little-endian length prefix followed by the Opus packet bytes.
    This allows exact reconstruction on decode.

    Args:
        pcm_data: Raw PCM16 audio bytes
        sample_rate: Sample rate in Hz (default 16000)
        channels: Number of audio channels (default 1)

    Returns:
        Length-prefixed Opus packets as bytes
    """
    encoder = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_VOIP)
    frame_size = sample_rate * OPUS_FRAME_DURATION_MS // 1000
    bytes_per_frame = frame_size * channels * 2  # 16-bit = 2 bytes per sample

    packets = []
    offset = 0
    while offset + bytes_per_frame <= len(pcm_data):
        frame = pcm_data[offset : offset + bytes_per_frame]
        encoded = encoder.encode(frame, frame_size)
        packets.append(encoded)
        offset += bytes_per_frame

    # Encode remaining samples (pad with silence)
    if offset < len(pcm_data):
        remaining = pcm_data[offset:]
        padded = remaining + b'\x00' * (bytes_per_frame - len(remaining))
        encoded = encoder.encode(padded, frame_size)
        packets.append(encoded)

    # Pack: [packet_count (4 bytes)] + [original_pcm_len (4 bytes)] + [len (2 bytes) + data] per packet
    output = struct.pack('<I', len(packets))
    output += struct.pack('<I', len(pcm_data))
    for pkt in packets:
        output += struct.pack('<H', len(pkt)) + pkt

    return output


def decode_opus_to_pcm(opus_data: bytes, sample_rate: int = OPUS_SAMPLE_RATE, channels: int = OPUS_CHANNELS) -> bytes:
    """
    Decode length-prefixed Opus packets back to PCM16.

    Args:
        opus_data: Length-prefixed Opus packets (from encode_pcm_to_opus)
        sample_rate: Sample rate in Hz (default 16000)
        channels: Number of audio channels (default 1)

    Returns:
        Raw PCM16 audio bytes

    Raises:
        ValueError: If opus_data is too short or has invalid header/packet structure
    """
    if len(opus_data) < 8:
        raise ValueError(f"Opus data too short: {len(opus_data)} bytes (need at least 8 for header)")

    decoder = opuslib.Decoder(sample_rate, channels)
    frame_size = sample_rate * OPUS_FRAME_DURATION_MS // 1000

    offset = 0
    packet_count = struct.unpack_from('<I', opus_data, offset)[0]
    offset += 4
    original_pcm_len = struct.unpack_from('<I', opus_data, offset)[0]
    offset += 4

    pcm_parts = []
    for i in range(packet_count):
        if offset + 2 > len(opus_data):
            raise ValueError(f"Truncated Opus data: expected packet {i}/{packet_count} length at offset {offset}")
        pkt_len = struct.unpack_from('<H', opus_data, offset)[0]
        offset += 2
        if offset + pkt_len > len(opus_data):
            raise ValueError(
                f"Truncated Opus data: packet {i} needs {pkt_len} bytes at offset {offset}, only {len(opus_data) - offset} available"
            )
        pkt_data = opus_data[offset : offset + pkt_len]
        offset += pkt_len
        decoded = decoder.decode(pkt_data, frame_size)
        pcm_parts.append(decoded)

    result = b''.join(pcm_parts)
    # Trim to original PCM length to remove padding from partial final frame
    if original_pcm_len > 0 and original_pcm_len < len(result):
        result = result[:original_pcm_len]
    return result


def _get_extension_for_path(path: str) -> str:
    """Extract the private cloud sync extension from a GCS path."""
    if path.endswith('.batch.enc'):
        return 'batch.enc'
    elif path.endswith('.batch.bin'):
        return 'batch.bin'
    elif path.endswith('.opus.enc'):
        return 'opus.enc'
    elif path.endswith('.opus'):
        return 'opus'
    elif path.endswith('.enc'):
        return 'enc'
    elif path.endswith('.bin'):
        return 'bin'
    return 'bin'


def _strip_extension(filename: str) -> str:
    """Strip private cloud sync extension to get the timestamp string.

    Handles both single-chunk filenames (e.g. '1000.000.opus') and
    batch filenames (e.g. '1000.000-1010.000.batch.bin',
    '1000.000.batch.opus'). Iterates PRIVATE_CLOUD_EXTENSIONS in
    longest-first order so compound extensions like `.batch.opus`
    match before the single `.opus` suffix.
    """
    for ext in PRIVATE_CLOUD_EXTENSIONS:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename.rsplit('.', 1)[0]


def upload_audio_chunk(
    chunk_data: bytes, uid: str, conversation_id: str, timestamp: float, data_protection_level: str = None
) -> str:
    """
    Upload an audio chunk to Google Cloud Storage with optional encryption.

    Args:
        chunk_data: Raw audio bytes (PCM16)
        uid: User ID
        conversation_id: Conversation ID
        timestamp: Unix timestamp when chunk was recorded
        data_protection_level: Optional cached protection level. When provided,
            skips the per-chunk Firestore read. Falls back to DB read when None.

    Returns:
        GCS path of the uploaded chunk
    """
    if LOCAL_AUDIO_ROOT:
        # Stage 1c: write raw PCM wrapped in a WAV container. Upstream's
        # custom Opus packet format (encode_pcm_to_opus) isn't a container
        # ffmpeg can decode — we'd have to re-parse length prefixes and
        # pipe through opuslib. WAV is ~10x larger but ffmpeg/soundfile
        # decode it natively, which the worker + enrollment both rely on.
        conv_dir = _local_conv_dir(uid, conversation_id)
        os.makedirs(conv_dir, exist_ok=True)
        formatted_timestamp = f'{timestamp:.3f}'
        path = os.path.join(conv_dir, f'{formatted_timestamp}.wav')
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(OPUS_CHANNELS)
            wf.setsampwidth(2)  # PCM16
            wf.setframerate(OPUS_SAMPLE_RATE)
            wf.writeframes(chunk_data)
        return path
    if STORAGE_DISABLED:
        return ''
    bucket = storage_client.bucket(private_cloud_sync_bucket)
    protection_level = (
        data_protection_level if data_protection_level is not None else users_db.get_data_protection_level(uid)
    )

    # Format timestamp to 3 decimal places for cleaner filenames
    formatted_timestamp = f'{timestamp:.3f}'

    upload_data = encode_pcm_to_opus(chunk_data)

    if protection_level == 'enhanced':
        encrypted_chunk = encryption.encrypt_audio_chunk(upload_data, uid)
        path = f'chunks/{uid}/{conversation_id}/{formatted_timestamp}.opus.enc'
        blob = bucket.blob(path)
        blob.upload_from_string(encrypted_chunk, content_type='application/octet-stream')
    else:
        path = f'chunks/{uid}/{conversation_id}/{formatted_timestamp}.opus'
        blob = bucket.blob(path)
        blob.upload_from_string(upload_data, content_type='application/octet-stream')

    del upload_data
    return path


def upload_audio_chunks_batch(
    chunks: List[dict],
    uid: str,
    conversation_id: str,
    data_protection_level: str = None,
) -> List[str]:
    """
    Upload multiple audio chunks to GCS in a single streaming write.

    Concatenates all chunk data into one GCS object (1 write op instead of N).

    Args:
        chunks: List of dicts with 'data' (bytes) and 'timestamp' (float).
        uid: User ID.
        conversation_id: Conversation ID.
        data_protection_level: Optional cached protection level. When provided,
            skips the Firestore read. Falls back to DB read when None.

    Returns:
        List of GCS paths for the uploaded batch.
    """
    if not chunks:
        return []
    if LOCAL_AUDIO_ROOT:
        # Stage 1c: write raw PCM wrapped in a WAV container. See the
        # upload_audio_chunk local-root branch for rationale.
        conv_dir = _local_conv_dir(uid, conversation_id)
        os.makedirs(conv_dir, exist_ok=True)
        sorted_chunks = sorted(chunks, key=lambda c: c['timestamp'])
        first_ts = f'{sorted_chunks[0]["timestamp"]:.3f}'
        last_ts = f'{sorted_chunks[-1]["timestamp"]:.3f}'
        batch_name = f'{first_ts}-{last_ts}' if len(sorted_chunks) > 1 else first_ts
        path = os.path.join(conv_dir, f'{batch_name}.batch.wav')
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(OPUS_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(OPUS_SAMPLE_RATE)
            for chunk in sorted_chunks:
                wf.writeframes(chunk['data'])
        return [path]
    if STORAGE_DISABLED:
        return []

    # Sort by timestamp for consistent ordering
    sorted_chunks = sorted(chunks, key=lambda c: c['timestamp'])

    # Resolve protection level once for the entire batch
    protection_level = (
        data_protection_level if data_protection_level is not None else users_db.get_data_protection_level(uid)
    )

    bucket = storage_client.bucket(private_cloud_sync_bucket)

    # Build batch filename from first and last timestamps
    first_ts = f'{sorted_chunks[0]["timestamp"]:.3f}'
    last_ts = f'{sorted_chunks[-1]["timestamp"]:.3f}'
    batch_name = f'{first_ts}-{last_ts}' if len(sorted_chunks) > 1 else first_ts

    if protection_level == 'enhanced':
        # Encrypt each chunk individually (length-prefixed), stream to GCS
        path = f'chunks/{uid}/{conversation_id}/{batch_name}.batch.enc'
        blob = bucket.blob(path)
        with blob.open('wb', content_type='application/octet-stream') as f:
            for chunk in sorted_chunks:
                encrypted_chunk = encryption.encrypt_audio_chunk(chunk['data'], uid)
                f.write(encrypted_chunk)
                del encrypted_chunk
    else:
        # Standard — stream raw PCM data to GCS
        path = f'chunks/{uid}/{conversation_id}/{batch_name}.batch.bin'
        blob = bucket.blob(path)
        with blob.open('wb', content_type='application/octet-stream') as f:
            for chunk in sorted_chunks:
                f.write(chunk['data'])

    return [path]


def delete_audio_chunks(uid: str, conversation_id: str, timestamps: List[float]) -> None:
    """Delete audio chunks after they've been merged.

    Handles both single-chunk blobs (per-timestamp lookup) and batch blobs
    (listed and matched by start timestamp).
    """
    if LOCAL_AUDIO_ROOT:
        conv_dir = _local_conv_dir(uid, conversation_id)
        if not os.path.isdir(conv_dir):
            return
        ts_set = {f'{ts:.3f}' for ts in timestamps}
        for filename in os.listdir(conv_dir):
            timestamp_str = _strip_extension(filename)
            if '-' in timestamp_str:
                start_ts = timestamp_str.split('-', 1)[0]
                if start_ts in ts_set:
                    os.unlink(os.path.join(conv_dir, filename))
            elif timestamp_str in ts_set:
                os.unlink(os.path.join(conv_dir, filename))
        return
    if STORAGE_DISABLED:
        return
    bucket = storage_client.bucket(private_cloud_sync_bucket)
    deleted_batch_paths = set()

    for timestamp in timestamps:
        # Format timestamp to match upload format (3 decimal places)
        formatted_timestamp = f'{timestamp:.3f}'

        # Try single-chunk extensions first
        for extension in PRIVATE_CLOUD_EXTENSIONS:
            if extension in ('.batch.enc', '.batch.bin'):
                continue  # batch blobs handled separately below
            chunk_path = f'chunks/{uid}/{conversation_id}/{formatted_timestamp}{extension}'
            blob = bucket.blob(chunk_path)
            if blob.exists():
                blob.delete()

        # Try batch blobs: exact single-timestamp batch (e.g. "1000.000.batch.bin")
        for batch_ext in ('.batch.enc', '.batch.bin'):
            batch_path = f'chunks/{uid}/{conversation_id}/{formatted_timestamp}{batch_ext}'
            if batch_path not in deleted_batch_paths:
                blob = bucket.blob(batch_path)
                if blob.exists():
                    blob.delete()
                    deleted_batch_paths.add(batch_path)

    # Scan for range-named batch blobs whose start timestamp matches any requested timestamp
    ts_set = {f'{ts:.3f}' for ts in timestamps}
    prefix = f'chunks/{uid}/{conversation_id}/'
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name in deleted_batch_paths:
            continue
        filename = blob.name.split('/')[-1]
        if '.batch.' not in filename:
            continue
        timestamp_str = _strip_extension(filename)
        if '-' in timestamp_str:
            start_ts = timestamp_str.split('-', 1)[0]
            if start_ts in ts_set:
                blob.delete()
                deleted_batch_paths.add(blob.name)


def list_audio_chunks(uid: str, conversation_id: str) -> List[dict]:
    """
    List all audio chunks for a conversation.

    Returns:
        List of dicts with chunk info: {'timestamp': float, 'path': str, 'size': int}
    """
    if LOCAL_AUDIO_ROOT:
        conv_dir = _local_conv_dir(uid, conversation_id)
        if not os.path.isdir(conv_dir):
            return []
        chunks = []
        for filename in os.listdir(conv_dir):
            has_valid_ext = any(filename.endswith(ext) for ext in PRIVATE_CLOUD_EXTENSIONS)
            if not has_valid_ext:
                continue
            try:
                timestamp_str = _strip_extension(filename)
                is_batch = '.batch.' in filename
                if is_batch and '-' in timestamp_str:
                    timestamp = float(timestamp_str.split('-', 1)[0])
                else:
                    timestamp = float(timestamp_str)
                path = os.path.join(conv_dir, filename)
                chunks.append({
                    'timestamp': timestamp,
                    'path': path,
                    'size': os.path.getsize(path),
                    'is_batch': is_batch,
                })
            except ValueError:
                continue
        return sorted(chunks, key=lambda x: x['timestamp'])
    if STORAGE_DISABLED:
        return []
    bucket = storage_client.bucket(private_cloud_sync_bucket)
    prefix = f'chunks/{uid}/{conversation_id}/'
    blobs = bucket.list_blobs(prefix=prefix)

    chunks = []
    for blob in blobs:
        # Extract timestamp from filename
        # Supports single-chunk: '1234567890.123.opus', '1234567890.123.opus.enc', etc.
        # Supports batch: '1234567890.123-1234567900.123.batch.bin', '1234567890.123.batch.enc'
        filename = blob.name.split('/')[-1]
        has_valid_ext = any(filename.endswith(ext) for ext in PRIVATE_CLOUD_EXTENSIONS)
        if has_valid_ext:
            try:
                timestamp_str = _strip_extension(filename)
                is_batch = '.batch.' in filename

                if is_batch and '-' in timestamp_str:
                    # Batch blob with timestamp range: "first_ts-last_ts"
                    first_ts_str, last_ts_str = timestamp_str.split('-', 1)
                    timestamp = float(first_ts_str)
                else:
                    timestamp = float(timestamp_str)

                chunks.append(
                    {
                        'timestamp': timestamp,
                        'path': blob.name,
                        'size': blob.size,
                        'is_batch': is_batch,
                    }
                )
            except ValueError:
                continue

    return sorted(chunks, key=lambda x: x['timestamp'])


def delete_conversation_audio_files(uid: str, conversation_id: str) -> None:
    """Delete all audio files (chunks and merged) for a conversation."""
    if LOCAL_AUDIO_ROOT:
        import shutil
        conv_dir = _local_conv_dir(uid, conversation_id)
        if os.path.isdir(conv_dir):
            shutil.rmtree(conv_dir)
        return
    if STORAGE_DISABLED:
        return
    bucket = storage_client.bucket(private_cloud_sync_bucket)

    # Delete chunks
    chunks_prefix = f'chunks/{uid}/{conversation_id}/'
    for blob in bucket.list_blobs(prefix=chunks_prefix):
        blob.delete()

    # Delete merged files
    audio_prefix = f'audio/{uid}/{conversation_id}/'
    for blob in bucket.list_blobs(prefix=audio_prefix):
        blob.delete()


def download_audio_chunks_and_merge(
    uid: str,
    conversation_id: str,
    timestamps: List[float],
    fill_gaps: bool = True,
    sample_rate: int = 16000,
) -> bytes:
    """
    Download and merge audio chunks on-demand, handling mixed encryption states.
    Downloads chunks in parallel.
    Normalizes all chunks to unencrypted PCM format for consistent merging.
    Supports both single-chunk blobs and batch blobs (from upload_audio_chunks_batch).

    Args:
        uid: User ID
        conversation_id: Conversation ID
        timestamps: List of chunk timestamps to merge
        fill_gaps: If True, insert silence (zero bytes) between chunks to maintain
                   continuous time-aligned audio. Default True.
        sample_rate: Audio sample rate in Hz (default 16000)

    Returns:
        Merged audio bytes (PCM16)
    """
    if LOCAL_AUDIO_ROOT:
        conv_dir = _local_conv_dir(uid, conversation_id)
        if not os.path.isdir(conv_dir):
            return b''
        # Collect all chunks whose timestamp (or batch range) covers any
        # requested timestamp; decode Opus → PCM16; concat in timestamp order.
        import subprocess
        pcm_parts = []
        ts_set = {round(ts, 3) for ts in timestamps}
        filenames = sorted(os.listdir(conv_dir))
        for fn in filenames:
            if not fn.endswith('.opus') and not fn.endswith('.batch.opus'):
                continue
            try:
                ts_str = _strip_extension(fn)
                ts = round(float(ts_str.split('-', 1)[0] if '-' in ts_str else ts_str), 3)
            except ValueError:
                continue
            if ts not in ts_set and not fn.endswith('.batch.opus'):
                continue
            # Decode Opus → PCM16 via ffmpeg (already in the image).
            src = os.path.join(conv_dir, fn)
            pcm = subprocess.run(
                ['ffmpeg', '-loglevel', 'error', '-i', src, '-f', 's16le',
                 '-ar', str(sample_rate), '-ac', '1', '-'],
                capture_output=True, check=True,
            ).stdout
            pcm_parts.append(pcm)
        return b''.join(pcm_parts)
    if STORAGE_DISABLED:
        return b''

    bucket = storage_client.bucket(private_cloud_sync_bucket)

    # Resolve actual GCS paths — needed to find batch blobs whose filenames
    # contain timestamp ranges instead of single timestamps
    actual_chunks = list_audio_chunks(uid, conversation_id)
    ts_set = {round(ts, 3) for ts in timestamps}

    # Build batch blob map: for batch blobs, track which timestamps they cover
    batch_paths = {}  # path -> chunk_info (deduplicate downloads)
    ts_to_batch_path = {}  # timestamp -> batch_path (for timestamps inside batch range)
    single_chunk_timestamps = []  # timestamps that have individual blobs

    for chunk in actual_chunks:
        if chunk.get('is_batch'):
            path = chunk['path']
            batch_paths[path] = chunk

            # Parse batch range to determine covered timestamps
            filename = path.split('/')[-1]
            ts_str = _strip_extension(filename)
            if '-' in ts_str:
                start_str, end_str = ts_str.split('-', 1)
                batch_start = float(start_str)
                batch_end = float(end_str)
            else:
                batch_start = batch_end = float(ts_str)

            # Map requested timestamps that fall within this batch's range
            for ts in timestamps:
                if batch_start <= round(ts, 3) <= batch_end:
                    ts_to_batch_path[round(ts, 3)] = path
        elif round(chunk['timestamp'], 3) in ts_set:
            single_chunk_timestamps.append(chunk['timestamp'])

    def _download_and_decode_blob(path: str) -> bytes | None:
        """Download a blob and decode/decrypt based on extension."""
        ext = _get_extension_for_path(path)
        encrypted = ext in ('opus.enc', 'enc', 'batch.enc')
        is_opus = ext in ('opus.enc', 'opus')

        try:
            chunk_data = bucket.blob(path).download_as_bytes()
        except NotFound:
            return None

        try:
            if encrypted:
                raw_data = encryption.decrypt_audio_file(chunk_data, uid)
            else:
                raw_data = chunk_data

            if is_opus:
                pcm_data = decode_opus_to_pcm(raw_data, sample_rate=sample_rate)
                del raw_data
            else:
                pcm_data = raw_data

            return pcm_data
        except Exception as e:
            logger.warning(f"Failed to decode/decrypt {path}: {e}")
            return None

    def download_single_chunk(timestamp: float) -> tuple[float, bytes | None]:
        """Download a single-chunk blob by trying extensions in priority order."""
        formatted_timestamp = f'{timestamp:.3f}'

        extensions_to_try = [
            ('opus.enc', True, True),  # (ext, encrypted, opus)
            ('enc', True, False),
            ('opus', False, True),
            ('bin', False, False),
        ]

        for ext, encrypted, opus in extensions_to_try:
            chunk_path = f'chunks/{uid}/{conversation_id}/{formatted_timestamp}.{ext}'
            try:
                chunk_data = bucket.blob(chunk_path).download_as_bytes()
            except NotFound:
                continue

            try:
                if encrypted:
                    raw_data = encryption.decrypt_audio_file(chunk_data, uid)
                else:
                    raw_data = chunk_data

                if opus:
                    pcm_data = decode_opus_to_pcm(raw_data, sample_rate=sample_rate)
                    del raw_data
                else:
                    pcm_data = raw_data

                return (timestamp, pcm_data)
            except Exception as e:
                logger.warning(
                    f"Failed to decode/decrypt {ext} chunk at {formatted_timestamp}: {e}, trying next format"
                )
                continue

        logger.warning(f"Warning: Chunk not found for timestamp {formatted_timestamp}")
        return (timestamp, None)

    # Download all data in parallel
    chunk_results = {}

    # Determine which timestamps need individual downloads vs batch downloads
    individual_timestamps = [ts for ts in timestamps if round(ts, 3) not in ts_to_batch_path]
    unique_batch_paths = set(ts_to_batch_path.values())

    # Submit individual chunk downloads via shared storage executor
    individual_futures = {storage_executor.submit(download_single_chunk, ts): ts for ts in individual_timestamps}

    # Submit batch blob downloads (once per unique path)
    batch_futures = {storage_executor.submit(_download_and_decode_blob, path): path for path in unique_batch_paths}

    # Collect individual results
    for future in as_completed(individual_futures):
        timestamp, pcm_data = future.result()
        if pcm_data is not None:
            chunk_results[timestamp] = pcm_data

    # Collect batch results — assign full batch data at the batch's start timestamp
    for future in as_completed(batch_futures):
        path = batch_futures[future]
        pcm_data = future.result()
        if pcm_data is not None:
            batch_info = batch_paths[path]
            chunk_results[batch_info['timestamp']] = pcm_data

    # Merge chunks
    merged_data = bytearray()

    if fill_gaps and timestamps and chunk_results:
        # Sort timestamps to ensure proper ordering
        sorted_timestamps = sorted(timestamps)
        first_timestamp = sorted_timestamps[0]
        current_time = first_timestamp  # Track current audio end time in seconds

        for timestamp in sorted_timestamps:
            if timestamp not in chunk_results:
                continue

            pcm_data = chunk_results[timestamp]

            # Calculate gap from current position to this chunk's start
            gap_seconds = timestamp - current_time
            if gap_seconds > 0:
                # Insert silence: 16-bit mono = 2 bytes per sample
                gap_samples = int(gap_seconds * sample_rate)
                silence_bytes = bytes(gap_samples * 2)  # Zero bytes for silence
                merged_data.extend(silence_bytes)
                logger.info(f"Filled {gap_seconds:.3f}s gap ({len(silence_bytes)} bytes) before chunk at {timestamp}")

            merged_data.extend(pcm_data)

            # Update current time based on chunk duration
            # PCM16 mono: 2 bytes per sample
            chunk_duration = len(pcm_data) / (sample_rate * 2)
            current_time = timestamp + chunk_duration
    else:
        # Original behavior - just concatenate without gap filling
        for timestamp in timestamps:
            if timestamp in chunk_results:
                merged_data.extend(chunk_results[timestamp])

    # Free memory from chunk results immediately after merging
    chunk_results.clear()

    if not merged_data:
        raise FileNotFoundError(f"No chunks found for conversation {conversation_id}")

    return bytes(merged_data)


def get_cached_merged_audio_path(uid: str, conversation_id: str, audio_file_id: str) -> str:
    """Get the GCS path for a cached merged audio file."""
    return f'merged/{uid}/{conversation_id}/{audio_file_id}.wav'


def get_or_create_merged_audio(
    uid: str,
    conversation_id: str,
    audio_file_id: str,
    timestamps: List[float],
    pcm_to_wav_func,
    fill_gaps: bool = True,
    sample_rate: int = 16000,
) -> tuple[bytes, bool]:
    """
    Get merged audio from cache or create it.
    Cached files are stored in GCS with 1-day TTL (via lifecycle policy).

    Args:
        uid: User ID
        conversation_id: Conversation ID
        audio_file_id: Audio file ID
        timestamps: List of chunk timestamps
        pcm_to_wav_func: Function to convert PCM to WAV
        fill_gaps: If True, insert silence between chunks to maintain time alignment. Default True.
        sample_rate: Audio sample rate in Hz (default 16000)

    Returns:
        Tuple of (audio_data_bytes, was_cached)
    """
    if STORAGE_DISABLED:
        return b'', False
    bucket = storage_client.bucket(private_cloud_sync_bucket)
    cache_path = get_cached_merged_audio_path(uid, conversation_id, audio_file_id)
    cache_blob = bucket.blob(cache_path)

    # Check if cached version exists and is not expired
    if cache_blob.exists():
        # Check custom metadata for expiry
        cache_blob.reload()
        metadata = cache_blob.metadata or {}
        expires_at_str = metadata.get('expires_at')

        if expires_at_str:
            try:
                expires_at = datetime.datetime.fromisoformat(expires_at_str)
                if datetime.datetime.now(datetime.timezone.utc) < expires_at:
                    # Cache is valid, return it
                    logger.info(f"Serving merged audio from cache: {cache_path}")
                    return cache_blob.download_as_bytes(), True
                else:
                    logger.warning(f"Cache expired for: {cache_path}")
            except (ValueError, TypeError):
                pass

    # Cache miss or expired - create new merged file
    logger.info(f"Cache miss, merging audio for: {cache_path}")

    # Download and merge chunks
    pcm_data = download_audio_chunks_and_merge(
        uid, conversation_id, timestamps, fill_gaps=fill_gaps, sample_rate=sample_rate
    )

    # Convert to WAV
    wav_data = pcm_to_wav_func(pcm_data)
    del pcm_data  # Free PCM data immediately after WAV conversion

    # Upload to cache in background thread with 3-day TTL
    def _upload_to_cache():
        try:
            expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3)
            cache_blob.metadata = {
                'expires_at': expires_at.isoformat(),
                'audio_file_id': audio_file_id,
            }
            cache_blob.upload_from_string(wav_data, content_type='audio/wav')
            logger.info(f"Cached merged audio at: {cache_path}")
        except Exception as e:
            logger.error(f"Error uploading audio cache: {e}")

    storage_executor.submit(_upload_to_cache)

    return wav_data, False


def get_merged_audio_signed_url(uid: str, conversation_id: str, audio_file_id: str) -> str | None:
    """
    Get a signed URL for cached merged audio if it exists and is valid.

    Returns:
        Signed URL valid for 1 hour, or None if cache doesn't exist
    """
    if STORAGE_DISABLED:
        return None
    bucket = storage_client.bucket(private_cloud_sync_bucket)
    cache_path = get_cached_merged_audio_path(uid, conversation_id, audio_file_id)
    cache_blob = bucket.blob(cache_path)

    if not cache_blob.exists():
        return None

    # Check expiry
    cache_blob.reload()
    metadata = cache_blob.metadata or {}
    expires_at_str = metadata.get('expires_at')

    if expires_at_str:
        try:
            expires_at = datetime.datetime.fromisoformat(expires_at_str)
            if datetime.datetime.now(datetime.timezone.utc) >= expires_at:
                return None  # Expired
        except (ValueError, TypeError):
            pass

    # Generate signed URL valid for 1 hour
    return _get_signed_url(cache_blob, 60)


def delete_cached_merged_audio(uid: str, conversation_id: str) -> None:
    """Delete all cached merged audio for a conversation."""
    if STORAGE_DISABLED:
        return
    bucket = storage_client.bucket(private_cloud_sync_bucket)
    prefix = f'merged/{uid}/{conversation_id}/'
    for blob in bucket.list_blobs(prefix=prefix):
        blob.delete()


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert PCM16 data to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


def precache_conversation_audio(
    uid: str, conversation_id: str, audio_files: list, fill_gaps: bool = True, sample_rate: int = 16000
) -> None:
    """
    Pre-cache all audio files for a conversation in a background thread.

    Args:
        uid: User ID
        conversation_id: Conversation ID
        audio_files: List of audio file dicts with 'id' and 'chunk_timestamps'
        fill_gaps: If True, insert silence between chunks to maintain time alignment. Default True.
        sample_rate: Audio sample rate in Hz (default 16000)
    """
    if not audio_files:
        return

    def _precache_all():

        def _cache_single(af):
            try:
                audio_file_id = af.get('id')
                timestamps = af.get('chunk_timestamps')
                if not audio_file_id or not timestamps:
                    return
                get_or_create_merged_audio(
                    uid=uid,
                    conversation_id=conversation_id,
                    audio_file_id=audio_file_id,
                    timestamps=timestamps,
                    pcm_to_wav_func=_pcm_to_wav,
                    fill_gaps=fill_gaps,
                    sample_rate=sample_rate,
                )
            except Exception as e:
                logger.error(f"[PRECACHE] Error caching audio file {af.get('id')}: {e}")

        futures = [storage_executor.submit(_cache_single, af) for af in audio_files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass

    storage_executor.submit(_precache_all)


# **********************************
# ************* UTILS **************
# **********************************


def download_blob_bytes(bucket_name: str, path: str) -> bytes:
    """
    Download blob content as bytes from GCS.

    Args:
        bucket_name: Name of the GCS bucket
        path: Path to the blob within the bucket

    Returns:
        Blob content as bytes

    Raises:
        NotFound: If the blob doesn't exist
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.download_as_bytes()


def delete_blob(bucket_name: str, path: str) -> bool:
    """
    Delete a blob from GCS.

    Args:
        bucket_name: Name of the GCS bucket
        path: Path to the blob within the bucket

    Returns:
        True if deleted, False if not found
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    try:
        blob.delete()
        return True
    except NotFound:
        return False


def download_speech_profile_bytes(path: str) -> bytes:
    """
    Download speech profile/sample audio from GCS.

    Args:
        path: GCS path to the sample (e.g., '{uid}/people_profiles/{person_id}/{filename}.wav')

    Returns:
        Audio bytes (WAV format)

    Raises:
        NotFound: If the sample doesn't exist
    """
    return download_blob_bytes(speech_profiles_bucket, path)


def delete_speech_profile_blob(path: str) -> bool:
    """
    Delete speech profile/sample from GCS.

    Args:
        path: GCS path to the sample

    Returns:
        True if deleted, False if not found
    """
    return delete_blob(speech_profiles_bucket, path)


def _get_signed_url(blob, minutes):
    if cached := get_cached_signed_url(blob.name):
        return cached

    signed_url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=minutes), method="GET")
    cache_signed_url(blob.name, signed_url, minutes * 60)
    return signed_url


def upload_app_logo(file_path: str, app_id: str):
    bucket = storage_client.bucket(omi_apps_bucket)
    path = f'{app_id}.png'
    blob = bucket.blob(path)
    blob.cache_control = 'public, no-cache'
    blob.upload_from_filename(file_path)
    return f'https://storage.googleapis.com/{omi_apps_bucket}/{path}'


def delete_app_logo(img_url: str):
    bucket = storage_client.bucket(omi_apps_bucket)
    path = img_url.split(f'https://storage.googleapis.com/{omi_apps_bucket}/')[1]
    logger.info(f'delete_app_logo {path}')
    blob = bucket.blob(path)
    blob.delete()


def upload_app_thumbnail(file_path: str, thumbnail_id: str) -> str:
    bucket = storage_client.bucket(app_thumbnails_bucket)
    path = f'{thumbnail_id}.jpg'
    blob = bucket.blob(path)
    blob.cache_control = 'public, no-cache'
    blob.upload_from_filename(file_path)
    public_url = f'https://storage.googleapis.com/{app_thumbnails_bucket}/{path}'
    return public_url


def get_app_thumbnail_url(thumbnail_id: str) -> str:
    path = f'{thumbnail_id}.jpg'
    return f'https://storage.googleapis.com/{app_thumbnails_bucket}/{path}'


# **********************************
# ************* CHAT FILES **************
# **********************************
def upload_multi_chat_files(files_name: List[str], uid: str) -> dict:
    """
    Upload multiple files to Google Cloud Storage in the chat files bucket.

    Args:
        files_name: List of file paths to upload
        uid: User ID to use as part of the storage path

    Returns:
        dict: A dictionary mapping original filenames to their Google Cloud Storage URLs
    """
    bucket = storage_client.bucket(chat_files_bucket)
    dictFiles = {}
    for name in files_name:
        try:
            blob = bucket.blob(f'{uid}/{name}')
            blob.cache_control = 'public, no-cache'
            blob.upload_from_filename(f'./{name}')
            try:
                blob.make_public()
            except Exception as e:
                logger.warning(f"Could not make blob public (may need bucket-level IAM): {e}")
            dictFiles[name] = f'https://storage.googleapis.com/{chat_files_bucket}/{uid}/{name}'
        except Exception as e:
            logger.error("Failed to upload {} due to exception: {}".format(name, e))
    return dictFiles


# **************************************************
# ************* DESKTOP UPDATES ********************
# **************************************************


def get_desktop_update_signed_url(blob_path: str, expiration_hours: int = 1) -> str:
    """
    Generate a signed URL for a desktop update file (ZIP).

    Args:
        blob_path: Path to the blob in GCS (e.g., "1.0.78+474-macos/1.0.78+474-macos.zip")
        expiration_hours: Hours until the URL expires (default: 1 hour)

    Returns:
        Signed URL valid for the specified duration
    """
    bucket = storage_client.bucket(desktop_updates_bucket)
    blob = bucket.blob(blob_path)

    # Use existing _get_signed_url helper with caching
    return _get_signed_url(blob, expiration_hours * 60)
