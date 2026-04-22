"""Conversations, photos, and per-model transcripts. Postgres-backed port.

Four tables:

- conversations                    (uid, id) PK — main conversation record.
  Typed columns: status, discarded, created_at, started_at, finished_at.
  Everything else lives inside `data` JSONB (including the encrypted
  `transcript_segments` blob — encryption happens in
  `_prepare_conversation_for_write` before we hit the DB).

- conversation_photos              (uid, conversation_id, photo_id) PK.

- conversation_model_transcripts   (uid, conversation_id, model_name, segment_id) PK.
  `model_name` is one of 'deepgram_streaming', 'soniox_streaming',
  'speechmatics_streaming', 'fal_whisperx'. `start_ts` is denormalized out
  of the JSONB for ORDER BY efficiency.

- action_items                     (uid, id) PK — referenced by delete_conversation
  cascade only. Primary CRUD lives in database/action_items.py.

Encryption semantics are preserved unchanged from the Firestore version:
`_prepare_conversation_for_write` / `_for_read` still handle the zlib compression
and optional AES encryption of the transcript_segments array. The
Postgres `data` column stores whatever the write-prep produces.
"""

import copy
import json
import uuid
import zlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

import utils.other.hume as hume
from database import users as users_db
from models.audio_file import AudioFile
from models.conversation_enums import ConversationStatus, PostProcessingModel, PostProcessingStatus
from models.conversation_photo import ConversationPhoto
from models.transcript_segment import TranscriptSegment
from utils import encryption
from ._client import db
from .helpers import set_data_protection_level, prepare_for_write, prepare_for_read, with_photos
from utils.other.storage import list_audio_chunks
import logging

logger = logging.getLogger(__name__)

conversations_collection = 'conversations'


def _ensure_timezone_aware(dt: datetime) -> datetime:
    """
    Ensure a datetime object is timezone-aware.
    If naive, assume UTC timezone.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# *********************************
# ******* ENCRYPTION HELPERS ******
# *********************************


def _decrypt_conversation_data(conversation_data: Dict[str, Any], uid: str) -> Dict[str, Any]:
    data = copy.deepcopy(conversation_data)

    if 'transcript_segments' not in data:
        return data

    if isinstance(data['transcript_segments'], str):
        try:
            decrypted_payload = encryption.decrypt(data['transcript_segments'], uid)
            if data.get('transcript_segments_compressed'):
                compressed_bytes = bytes.fromhex(decrypted_payload)
                decompressed_json = zlib.decompress(compressed_bytes).decode('utf-8')
                data['transcript_segments'] = json.loads(decompressed_json)
            # backward compatibility, will be removed soon
            else:
                data['transcript_segments'] = json.loads(decrypted_payload)
        except (json.JSONDecodeError, TypeError, zlib.error, ValueError) as e:
            logger.error(f"{e} {uid}")
            data['transcript_segments'] = []
    # backward compatibility, will be removed soon
    elif isinstance(data['transcript_segments'], bytes):
        try:
            compressed_bytes = data['transcript_segments']
            if data.get('transcript_segments_compressed'):
                decompressed_json = zlib.decompress(compressed_bytes).decode('utf-8')
                data['transcript_segments'] = json.loads(decompressed_json)
        except (json.JSONDecodeError, TypeError, zlib.error, ValueError) as e:
            logger.error(f"{e} {uid}")
            data['transcript_segments'] = []

    return data


def _prepare_conversation_for_write(data: Dict[str, Any], uid: str, level: str) -> Dict[str, Any]:
    data = copy.deepcopy(data)
    if 'transcript_segments' in data and isinstance(data['transcript_segments'], list):
        segments_json = json.dumps(data['transcript_segments'])
        compressed_segments_bytes = zlib.compress(segments_json.encode('utf-8'))
        data['transcript_segments_compressed'] = True

        if level == 'enhanced':
            encrypted_segments = encryption.encrypt(compressed_segments_bytes.hex(), uid)
            data['transcript_segments'] = encrypted_segments
        else:
            data['transcript_segments'] = compressed_segments_bytes
    return data


def _prepare_conversation_for_read(conversation_data: Optional[Dict[str, Any]], uid: str) -> Optional[Dict[str, Any]]:
    if not conversation_data:
        return None

    data = copy.deepcopy(conversation_data)
    level = data.get('data_protection_level')

    if level == 'enhanced':
        return _decrypt_conversation_data(data, uid)

    # Handle standard level with potential compression
    if data.get('transcript_segments_compressed'):
        if 'transcript_segments' in data and isinstance(data['transcript_segments'], bytes):
            try:
                decompressed_json = zlib.decompress(data['transcript_segments']).decode('utf-8')
                data['transcript_segments'] = json.loads(decompressed_json)
            except (json.JSONDecodeError, TypeError, zlib.error) as e:
                logger.error(e)
                pass

    return data


def _prepare_photo_for_write(data: Dict[str, Any], uid: str, level: str) -> Dict[str, Any]:
    data = copy.deepcopy(data)
    data['data_protection_level'] = level
    if level == 'enhanced' and 'base64' in data and isinstance(data['base64'], str):
        data['base64'] = encryption.encrypt(data['base64'], uid)
    return data


def _prepare_photo_for_read(photo_data: Optional[Dict[str, Any]], uid: str) -> Optional[Dict[str, Any]]:
    if not photo_data:
        return None
    data = copy.deepcopy(photo_data)
    level = data.get('data_protection_level')
    if level == 'enhanced' and 'base64' in data and isinstance(data['base64'], str):
        try:
            data['base64'] = encryption.decrypt(data['base64'], uid)
        except Exception:
            # If decryption fails, it might be already decrypted or not encrypted.
            # We can log this, but for now, we'll just pass.
            pass
    return data


# *********************************************************************
# **** Row helpers: promote typed columns into the JSONB data dict ****
# *********************************************************************
#
# Tables keep status/discarded/created_at/started_at/finished_at in typed
# columns for indexing. But callers have always treated these as keys on the
# conversation dict. These helpers bridge the two views.


_CONVERSATION_COLS = 'uid, id, status, discarded, created_at, started_at, finished_at, data'


def _row_to_conversation(row) -> Dict[str, Any]:
    """Merge typed columns back into the JSONB data dict so callers see one flat dict."""
    _uid, cid, status, discarded, created_at, started_at, finished_at, data = row
    out = dict(data or {})
    out['id'] = cid
    out['status'] = status
    out['discarded'] = discarded
    out['created_at'] = created_at
    out['started_at'] = started_at
    out['finished_at'] = finished_at
    return out


def _extract_typed_cols(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return dict of typed columns (status, discarded, *_at) from a conversation dict."""
    return {
        'status': conversation_data.get('status'),
        'discarded': bool(conversation_data.get('discarded', False)),
        'created_at': conversation_data.get('created_at'),
        'started_at': conversation_data.get('started_at'),
        'finished_at': conversation_data.get('finished_at'),
    }


def _data_without_typed_cols(conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of conversation_data with typed columns stripped out of the JSONB body."""
    out = copy.deepcopy(conversation_data)
    for key in ('status', 'discarded', 'created_at', 'started_at', 'finished_at'):
        out.pop(key, None)
    return out


def _json_default(value):
    """JSON serializer for datetime / bytes values going into JSONB."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        # zlib-compressed transcript_segments for standard-level conversations.
        return value.hex()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@prepare_for_read(decrypt_func=_prepare_photo_for_read)
def get_conversation_photos(uid: str, conversation_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data FROM conversation_photos
                WHERE uid = %s AND conversation_id = %s
                ORDER BY created_at
                """,
                (uid, conversation_id),
            )
            return [dict(row[0] or {}) for row in cur.fetchall()]


# *****************************
# ********** CRUD *************
# *****************************


@set_data_protection_level(data_arg_name='conversation_data')
@prepare_for_write(data_arg_name='conversation_data', prepare_func=_prepare_conversation_for_write)
def upsert_conversation(uid: str, conversation_data: dict):
    if 'audio_base64_url' in conversation_data:
        del conversation_data['audio_base64_url']
    if 'photos' in conversation_data:
        del conversation_data['photos']

    conv_id = conversation_data['id']
    typed = _extract_typed_cols(conversation_data)
    data_body = _data_without_typed_cols(conversation_data)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations
                    (uid, id, status, discarded, created_at, started_at, finished_at, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET status = EXCLUDED.status,
                        discarded = EXCLUDED.discarded,
                        created_at = EXCLUDED.created_at,
                        started_at = EXCLUDED.started_at,
                        finished_at = EXCLUDED.finished_at,
                        data = EXCLUDED.data
                """,
                (
                    uid,
                    conv_id,
                    typed['status'],
                    typed['discarded'],
                    typed['created_at'],
                    typed['started_at'],
                    typed['finished_at'],
                    json.dumps(data_body, default=_json_default),
                ),
            )


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
@with_photos(get_conversation_photos)
def get_conversation(uid, conversation_id):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_CONVERSATION_COLS} FROM conversations WHERE uid = %s AND id = %s",
                (uid, conversation_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_conversation(row)


def _build_list_query(
    uid: str,
    limit: int,
    offset: int,
    include_discarded: bool,
    statuses: List[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    categories: Optional[List[str]],
    folder_id: Optional[str],
    starred: Optional[bool],
) -> tuple[str, tuple]:
    parts = [f"SELECT {_CONVERSATION_COLS} FROM conversations WHERE uid = %s"]
    params: List[Any] = [uid]

    if not include_discarded:
        parts.append("AND discarded = FALSE")
    if statuses and len(statuses) > 0:
        parts.append("AND status = ANY(%s::text[])")
        params.append(list(statuses))
    if categories:
        parts.append("AND data->'structured'->>'category' = ANY(%s::text[])")
        params.append(list(categories))
    if folder_id:
        parts.append("AND data->>'folder_id' = %s")
        params.append(folder_id)
    if starred is not None:
        # JSONB boolean comparison: data->'starred' serializes to 'true'/'false'.
        parts.append("AND (data->>'starred')::boolean = %s")
        params.append(bool(starred))
    if start_date:
        parts.append("AND created_at >= %s")
        params.append(start_date)
    if end_date:
        parts.append("AND created_at <= %s")
        params.append(end_date)

    parts.append("ORDER BY created_at DESC")
    parts.append("LIMIT %s OFFSET %s")
    params.extend([limit, offset])

    return ("\n".join(parts), tuple(params))


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
@with_photos(get_conversation_photos)
def get_conversations(
    uid: str,
    limit: int = 100,
    offset: int = 0,
    include_discarded: bool = False,
    statuses: List[str] = [],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    categories: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
    starred: Optional[bool] = None,
):
    sql, params = _build_list_query(
        uid, limit, offset, include_discarded, statuses,
        start_date, end_date, categories, folder_id, starred,
    )
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_conversation(r) for r in cur.fetchall()]


def get_conversations_count(uid: str, include_discarded: bool = False, statuses: List[str] = []):
    parts = ["SELECT COUNT(*) FROM conversations WHERE uid = %s"]
    params: List[Any] = [uid]
    if not include_discarded:
        parts.append("AND discarded = FALSE")
    if statuses:
        parts.append("AND status = ANY(%s::text[])")
        params.append(list(statuses))

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            row = cur.fetchone()
            return int(row[0]) if row else 0


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
def get_conversations_without_photos(
    uid: str,
    limit: int = 100,
    offset: int = 0,
    include_discarded: bool = False,
    statuses: List[str] = [],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    categories: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
    starred: Optional[bool] = None,
):
    """
    Same as get_conversations but without loading photos.
    Much faster for list endpoints and bulk operations where full photo base64 isn't needed.
    """
    sql, params = _build_list_query(
        uid, limit, offset, include_discarded, statuses,
        start_date, end_date, categories, folder_id, starred,
    )
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_conversation(r) for r in cur.fetchall()]


def iter_all_conversations(uid: str, batch_size: int = 400, include_discarded: bool = True):
    """Yield all conversations for a user, decrypted, in batches. Used for streaming data export."""
    parts = [f"SELECT {_CONVERSATION_COLS} FROM conversations WHERE uid = %s"]
    params: List[Any] = [uid]
    if not include_discarded:
        parts.append("AND discarded = FALSE")
    parts.append("ORDER BY created_at DESC")
    sql = "\n".join(parts)

    with db.connection() as conn:
        cursor_name = f"iter_conv_{uuid.uuid4().hex}"
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = batch_size
            cur.execute(sql, tuple(params))
            while True:
                batch_rows = cur.fetchmany(batch_size)
                if not batch_rows:
                    break
                for row in batch_rows:
                    conv = _row_to_conversation(row)
                    yield _prepare_conversation_for_read(conv, uid) or conv


def update_conversation(uid: str, conversation_id: str, update_data: dict):
    """Partial-update a conversation. Nested keys with `.` separators (e.g.
    'structured.title') are honored — they update one leaf inside the JSONB
    `data` blob via jsonb_set. Flat keys shallow-merge into `data`.

    Typed columns (status/discarded/created_at/started_at/finished_at) are
    updated on their own columns when present in `update_data`.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            # Load the existing row (need data_protection_level for write-prep).
            cur.execute(
                "SELECT data FROM conversations WHERE uid = %s AND id = %s FOR UPDATE",
                (uid, conversation_id),
            )
            row = cur.fetchone()
            if row is None:
                return

            doc_level = (row[0] or {}).get('data_protection_level', 'standard')
            prepared_data = _prepare_conversation_for_write(update_data, uid, doc_level)

            typed_updates: Dict[str, Any] = {}
            jsonb_flat_updates: Dict[str, Any] = {}
            jsonb_nested_updates: Dict[str, Any] = {}

            for key, value in prepared_data.items():
                if key in ('status', 'discarded', 'created_at', 'started_at', 'finished_at'):
                    typed_updates[key] = value
                elif '.' in key:
                    jsonb_nested_updates[key] = value
                else:
                    jsonb_flat_updates[key] = value

            # Shallow-merge flat JSONB keys.
            if jsonb_flat_updates:
                cur.execute(
                    "UPDATE conversations SET data = data || %s::jsonb WHERE uid = %s AND id = %s",
                    (json.dumps(jsonb_flat_updates, default=_json_default), uid, conversation_id),
                )

            # Nested keys: jsonb_set per path.
            for dotted, value in jsonb_nested_updates.items():
                path = '{' + ','.join(dotted.split('.')) + '}'
                cur.execute(
                    """
                    UPDATE conversations
                    SET data = jsonb_set(data, %s::text[], %s::jsonb, TRUE)
                    WHERE uid = %s AND id = %s
                    """,
                    (path, json.dumps(value, default=_json_default), uid, conversation_id),
                )

            # Typed column updates.
            if typed_updates:
                set_parts = []
                params: List[Any] = []
                for col, val in typed_updates.items():
                    set_parts.append(f"{col} = %s")
                    params.append(val)
                params.extend([uid, conversation_id])
                cur.execute(
                    f"UPDATE conversations SET {', '.join(set_parts)} WHERE uid = %s AND id = %s",
                    tuple(params),
                )


def create_audio_files_from_chunks(
    uid: str,
    conversation_id: str,
) -> List[AudioFile]:
    """
    Create audio file records by merging chunks from a conversation.
    Chunks are merged unless there's a gap > 30 seconds between segments.

    Args:
        uid: User ID
        conversation_id: Conversation ID

    Returns:
        List of AudioFile objects
    """
    # Get all chunks for this conversation
    chunks = list_audio_chunks(uid, conversation_id)
    if not chunks:
        return []

    # Group chunks based on gap rule (90s threshold accommodates both 5s and 60s chunk durations)
    audio_files = []
    current_group = []
    gap_threshold = 90  # seconds — must exceed max chunk duration (60s) to avoid false splits

    for i, chunk in enumerate(chunks):
        if not current_group:
            current_group.append(chunk)
        else:
            # Check if there's a gap between chunks exceeding the threshold
            prev_chunk = current_group[-1]
            time_gap = chunk['timestamp'] - prev_chunk['timestamp']
            if time_gap > gap_threshold:
                # Gap detected, finalize current group
                audio_file = _finalize_audio_file_group(uid, conversation_id, current_group, audio_files)
                if audio_file:
                    audio_files.append(audio_file)
                current_group = [chunk]
            else:
                current_group.append(chunk)

    # Finalize last group
    if current_group:
        audio_file = _finalize_audio_file_group(uid, conversation_id, current_group, audio_files)
        if audio_file:
            audio_files.append(audio_file)

    return audio_files


def _finalize_audio_file_group(
    uid: str, conversation_id: str, chunk_group: List[dict], existing_files: List[AudioFile]
) -> Optional[AudioFile]:
    """
    Create an AudioFile record that references chunks (no merging).

    Args:
        uid: User ID
        conversation_id: Conversation ID
        chunk_group: List of chunk dicts to reference
        existing_files: List of existing audio files

    Returns:
        AudioFile object or None if failed
    """
    if not chunk_group:
        return None

    # Generate file ID
    file_id = str(uuid.uuid4())

    # Extract timestamps
    timestamps = [chunk['timestamp'] for chunk in chunk_group]

    # Calculate started_at and duration from timestamps and blob sizes
    started_at = datetime.fromtimestamp(chunk_group[0]['timestamp'], tz=timezone.utc)
    last_chunk_start = datetime.fromtimestamp(chunk_group[-1]['timestamp'], tz=timezone.utc)
    # Estimate last chunk duration from blob size (PCM16 mono at 8kHz = 16000 bytes/sec)
    last_chunk_size = chunk_group[-1].get('size', 0)
    last_chunk_duration = last_chunk_size / 16000.0 if last_chunk_size > 0 else 5.0
    duration = (last_chunk_start - started_at).total_seconds() + last_chunk_duration

    return AudioFile(
        id=file_id,
        uid=uid,
        conversation_id=conversation_id,
        chunk_timestamps=timestamps,
        provider='gcp',
        started_at=started_at,
        duration=duration,
    )


def update_conversation_title(uid: str, conversation_id: str, title: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM conversations WHERE uid = %s AND id = %s",
                (uid, conversation_id),
            )
            if cur.fetchone() is None:
                return

            cur.execute(
                """
                UPDATE conversations
                SET data = jsonb_set(data, '{structured,title}', %s::jsonb, TRUE)
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(title), uid, conversation_id),
            )


def update_conversation_segment_text(uid: str, conversation_id: str, segment_id: str, text: str) -> str:
    """
    Update a single segment's text in a conversation.

    Returns:
        'ok' on success, 'not_found' if conversation missing, 'locked' if conversation is locked,
        'segment_not_found' if segment_id not found.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM conversations WHERE uid = %s AND id = %s FOR UPDATE",
                (uid, conversation_id),
            )
            row = cur.fetchone()
            if row is None:
                return 'not_found'

            raw_data = dict(row[0] or {})
            if raw_data.get('is_locked', False):
                return 'locked'

            conversation_data = _prepare_conversation_for_read(raw_data, uid)
            if not conversation_data:
                return 'not_found'

            segments = conversation_data.get('transcript_segments', [])
            found = False
            for segment in segments:
                if isinstance(segment, dict) and segment.get('id') == segment_id:
                    segment['text'] = text
                    found = True
                    break

            if not found:
                return 'segment_not_found'

            doc_level = conversation_data.get('data_protection_level', 'standard')
            prepared_payload = _prepare_conversation_for_write({'transcript_segments': segments}, uid, doc_level)

            # Merge prepared transcript_segments (+ _compressed flag) back into data.
            cur.execute(
                "UPDATE conversations SET data = data || %s::jsonb WHERE uid = %s AND id = %s",
                (json.dumps(prepared_payload, default=_json_default), uid, conversation_id),
            )
            return 'ok'


def delete_conversation_photos(uid: str, conversation_id: str) -> int:
    """
    Delete all photos for a conversation.

    Returns:
        Number of photos deleted
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversation_photos WHERE uid = %s AND conversation_id = %s",
                (uid, conversation_id),
            )
            return cur.rowcount or 0


def delete_conversation(uid, conversation_id):
    """
    Delete a conversation plus every related row: photos, per-model transcripts,
    and action_items scoped to this conversation. All wrapped in a single
    transaction via db.batch() so a failure rolls the whole thing back.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversation_photos WHERE uid = %s AND conversation_id = %s",
                (uid, conversation_id),
            )
            cur.execute(
                "DELETE FROM conversation_model_transcripts WHERE uid = %s AND conversation_id = %s",
                (uid, conversation_id),
            )
            cur.execute(
                "DELETE FROM action_items WHERE uid = %s AND conversation_id = %s",
                (uid, conversation_id),
            )
            cur.execute(
                "DELETE FROM conversations WHERE uid = %s AND id = %s",
                (uid, conversation_id),
            )


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
@with_photos(get_conversation_photos)
def get_conversations_by_id(uid, conversation_ids):
    if not conversation_ids:
        return []

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_CONVERSATION_COLS} FROM conversations
                WHERE uid = %s
                  AND id = ANY(%s::text[])
                  AND discarded = FALSE
                """,
                (uid, [str(cid) for cid in conversation_ids]),
            )
            return [_row_to_conversation(r) for r in cur.fetchall()]


# **************************************
# ********* MIGRATION HELPERS **********
# **************************************


def get_conversations_to_migrate(uid: str, target_level: str) -> List[dict]:
    """
    Finds all conversations that are not at the target protection level.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       data->>'data_protection_level' AS level,
                       data->>'visibility' AS visibility
                FROM conversations
                WHERE uid = %s
                """,
                (uid,),
            )
            to_migrate: List[dict] = []
            for cid, level, visibility in cur.fetchall():
                if visibility in ('public', 'shared'):
                    continue
                current_level = level or 'standard'
                if target_level != current_level:
                    to_migrate.append({'id': cid, 'type': 'conversation'})
            return to_migrate


def migrate_conversations_level_batch(uid: str, conversation_ids: List[str], target_level: str):
    """
    Migrates a batch of conversations (and their photos) to the target protection level,
    wrapped in a single transaction.
    """
    if not conversation_ids:
        return

    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_CONVERSATION_COLS}
                FROM conversations
                WHERE uid = %s AND id = ANY(%s::text[])
                FOR UPDATE
                """,
                (uid, list(conversation_ids)),
            )
            conversation_rows = cur.fetchall()

            for row in conversation_rows:
                conversation_dict = _row_to_conversation(row)
                conv_id = conversation_dict['id']
                current_level = conversation_dict.get('data_protection_level', 'standard')

                if current_level == target_level:
                    continue

                # Decrypt/decompress to get a clean slate.
                plain_data = _prepare_conversation_for_read(conversation_dict, uid)

                update_payload = {'transcript_segments': plain_data.get('transcript_segments')}
                prepared_payload = _prepare_conversation_for_write(update_payload, uid, target_level)

                update_data: Dict[str, Any] = {'data_protection_level': target_level}
                if 'transcript_segments' in prepared_payload:
                    update_data['transcript_segments'] = prepared_payload['transcript_segments']
                    update_data['transcript_segments_compressed'] = prepared_payload.get(
                        'transcript_segments_compressed', False
                    )

                # Shallow-merge. If the new level doesn't need _compressed, clear it.
                cur.execute(
                    "UPDATE conversations SET data = data || %s::jsonb WHERE uid = %s AND id = %s",
                    (json.dumps(update_data, default=_json_default), uid, conv_id),
                )
                if not update_data.get('transcript_segments_compressed'):
                    cur.execute(
                        """
                        UPDATE conversations
                        SET data = data - 'transcript_segments_compressed'
                        WHERE uid = %s AND id = %s
                        """,
                        (uid, conv_id),
                    )

                # Migrate photos for this conversation.
                cur.execute(
                    """
                    SELECT photo_id, data
                    FROM conversation_photos
                    WHERE uid = %s AND conversation_id = %s
                    FOR UPDATE
                    """,
                    (uid, conv_id),
                )
                for photo_id, photo_data in cur.fetchall():
                    photo_dict = dict(photo_data or {})
                    current_photo_level = photo_dict.get('data_protection_level', 'standard')
                    if current_photo_level == target_level:
                        continue

                    plain_photo_data = _prepare_photo_for_read(photo_dict, uid)
                    photo_update_payload: Dict[str, Any] = {'data_protection_level': target_level}
                    if target_level == 'enhanced':
                        photo_update_payload['base64'] = encryption.encrypt(plain_photo_data['base64'], uid)
                    else:
                        photo_update_payload['base64'] = plain_photo_data['base64']

                    cur.execute(
                        """
                        UPDATE conversation_photos
                        SET data = data || %s::jsonb
                        WHERE uid = %s AND conversation_id = %s AND photo_id = %s
                        """,
                        (json.dumps(photo_update_payload), uid, conv_id, photo_id),
                    )


# **************************************
# ********** STATUS *************
# **************************************


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
@with_photos(get_conversation_photos)
def get_in_progress_conversation(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_CONVERSATION_COLS}
                FROM conversations
                WHERE uid = %s AND status = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (uid, 'in_progress'),
            )
            row = cur.fetchone()
            return _row_to_conversation(row) if row else None


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
@with_photos(get_conversation_photos)
def get_processing_conversations(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_CONVERSATION_COLS}
                FROM conversations
                WHERE uid = %s AND status = %s
                """,
                (uid, 'processing'),
            )
            return [_row_to_conversation(r) for r in cur.fetchall()]


def update_conversation_status(uid: str, conversation_id: str, status: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET status = %s WHERE uid = %s AND id = %s",
                (status, uid, conversation_id),
            )


def set_conversation_as_discarded(uid: str, conversation_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET discarded = TRUE WHERE uid = %s AND id = %s",
                (uid, conversation_id),
            )


# *********************************
# ********** CALENDAR *************
# *********************************


def update_conversation_events(uid: str, conversation_id: str, events: List[dict]):
    update_conversation(uid, conversation_id, {'structured.events': events})


# *********************************
# ******** ACTION ITEMS ***********
# *********************************


def update_conversation_action_items(uid: str, conversation_id: str, action_items: List[dict]):
    update_conversation(uid, conversation_id, {'structured.action_items': action_items})


def get_action_items(
    uid: str,
    limit: int = 100,
    offset: int = 0,
    include_completed: bool = True,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """Fetch action items embedded in completed conversations' structured.action_items array.

    This preserves the Firestore-era shape: action_items are read out of the
    conversation's `data->structured->action_items` JSONB array. (The separate
    `action_items` table exists for the new standalone action-items router; this
    helper powers the legacy path.)
    """
    parts = [
        f"SELECT {_CONVERSATION_COLS} FROM conversations WHERE uid = %s AND status = %s",
    ]
    params: List[Any] = [uid, 'completed']
    if start_date:
        parts.append("AND created_at >= %s")
        params.append(start_date)
    if end_date:
        parts.append("AND created_at <= %s")
        params.append(end_date)
    parts.append("ORDER BY created_at DESC")

    conversations: List[dict] = []
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            for row in cur.fetchall():
                conversation_dict = _row_to_conversation(row)
                structured = conversation_dict.get('structured', {})
                raw_action_items = structured.get('action_items', [])
                if raw_action_items:
                    decrypted_data = _prepare_conversation_for_read(conversation_dict, uid)
                    conversations.append(decrypted_data)

    # Extract and flatten action items with metadata
    action_items: List[dict] = []
    for conversation in conversations:
        conversation_id = conversation['id']
        conversation_title = conversation.get('structured', {}).get('title', 'Untitled')
        conversation_created_at = _ensure_timezone_aware(conversation['created_at'])

        raw_items = conversation.get('structured', {}).get('action_items', [])

        for idx, item in enumerate(raw_items):
            # Skip deleted items
            if isinstance(item, dict) and item.get('deleted', False):
                continue

            is_completed = False
            if isinstance(item, dict):
                is_completed = item.get('completed', False)

            if not include_completed and is_completed:
                continue

            created_at = None
            completed_at = None

            if isinstance(item, dict):
                created_at = item.get('created_at')
                completed_at = item.get('completed_at')

            if created_at is not None:
                created_at = _ensure_timezone_aware(created_at)
            if completed_at is not None:
                completed_at = _ensure_timezone_aware(completed_at)

            if created_at is None:
                created_at = conversation_created_at

            if is_completed and completed_at is None:
                completed_at = conversation_created_at

            action_item_data = {
                'id': f"{conversation_id}_{idx}",
                'conversation_id': conversation_id,
                'conversation_title': conversation_title,
                'conversation_created_at': conversation_created_at,
                'index': idx,
                'description': item.get('description', item) if isinstance(item, dict) else item,
                'completed': is_completed,
                'deleted': item.get('deleted', False) if isinstance(item, dict) else False,
                'created_at': created_at,
                'completed_at': completed_at,
            }
            action_items.append(action_item_data)

    # Sort by newest first
    action_items.sort(key=lambda x: -x['conversation_created_at'].timestamp())

    # Apply pagination
    start_idx = offset
    end_idx = offset + limit

    return action_items[start_idx:end_idx]


# ******************************
# ********** OTHER *************
# ******************************


def update_conversation_finished_at(uid: str, conversation_id: str, finished_at: datetime):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET finished_at = %s WHERE uid = %s AND id = %s",
                (finished_at, uid, conversation_id),
            )


def update_conversation_segments(
    uid: str,
    conversation_id: str,
    segments: List[dict],
    finished_at: datetime = None,
    data_protection_level: str = None,
):
    with db.batch() as conn:
        with conn.cursor() as cur:
            if data_protection_level is not None:
                doc_level = data_protection_level
            else:
                cur.execute(
                    "SELECT data->>'data_protection_level' FROM conversations WHERE uid = %s AND id = %s",
                    (uid, conversation_id),
                )
                row = cur.fetchone()
                if row is None:
                    return
                doc_level = row[0] or 'standard'

            update_payload: Dict[str, Any] = {'transcript_segments': segments}
            prepared_payload = _prepare_conversation_for_write(update_payload, uid, doc_level)

            cur.execute(
                "UPDATE conversations SET data = data || %s::jsonb WHERE uid = %s AND id = %s",
                (json.dumps(prepared_payload, default=_json_default), uid, conversation_id),
            )
            if finished_at is not None:
                cur.execute(
                    "UPDATE conversations SET finished_at = %s WHERE uid = %s AND id = %s",
                    (finished_at, uid, conversation_id),
                )


# ***********************************
# ********** VISIBILITY *************
# ***********************************


def set_conversation_visibility(uid: str, conversation_id: str, visibility: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET data = data || %s::jsonb WHERE uid = %s AND id = %s",
                (json.dumps({'visibility': visibility}), uid, conversation_id),
            )


def set_conversation_starred(uid: str, conversation_id: str, starred: bool):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET data = data || %s::jsonb WHERE uid = %s AND id = %s",
                (json.dumps({'starred': starred}), uid, conversation_id),
            )


def unlock_all_conversations(uid: str):
    """
    Find all conversations for a user with is_locked: True and update them to is_locked = False.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversations
                SET data = data || '{"is_locked": false}'::jsonb
                WHERE uid = %s AND (data->>'is_locked')::boolean = TRUE
                """,
                (uid,),
            )
    logger.info(f"Unlocked all conversations for user {uid}")


# ****************************************
# ********** POSTPROCESSING **************
# ****************************************


def set_postprocessing_status(
    uid: str,
    conversation_id: str,
    status: PostProcessingStatus,
    fail_reason: str = None,
    model: PostProcessingModel = PostProcessingModel.fal_whisperx,
):
    # Firestore used dotted keys ("postprocessing.status") so the three fields
    # landed on the postprocessing sub-object. In JSONB we write the whole
    # sub-object at once via jsonb_set so a single update replaces it.
    status_val = status.value if hasattr(status, 'value') else status
    model_val = model.value if hasattr(model, 'value') else model
    payload = {
        'status': status_val,
        'model': model_val,
        'fail_reason': fail_reason,
    }
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversations
                SET data = jsonb_set(data, '{postprocessing}', %s::jsonb, TRUE)
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(payload, default=_json_default), uid, conversation_id),
            )


def store_model_segments_result(uid: str, conversation_id: str, model_name: str, segments: List[TranscriptSegment]):
    if not segments:
        return
    with db.batch() as conn:
        with conn.cursor() as cur:
            for segment in segments:
                segment_id = str(uuid.uuid4())
                segment_data = segment.dict()
                start_ts = segment_data.get('start')
                cur.execute(
                    """
                    INSERT INTO conversation_model_transcripts
                        (uid, conversation_id, model_name, segment_id, start_ts, data)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (uid, conversation_id, model_name, segment_id) DO UPDATE
                        SET start_ts = EXCLUDED.start_ts,
                            data = EXCLUDED.data
                    """,
                    (
                        uid,
                        conversation_id,
                        model_name,
                        segment_id,
                        start_ts,
                        json.dumps(segment_data, default=_json_default),
                    ),
                )


def store_model_emotion_predictions_result(
    uid: str, conversation_id: str, model_name: str, predictions: List[hume.HumeJobModelPredictionResponseModel]
):
    if not predictions:
        return
    now = datetime.now()
    with db.batch() as conn:
        with conn.cursor() as cur:
            for prediction in predictions:
                prediction_id = str(uuid.uuid4())
                payload = {
                    "created_at": now,
                    "start": prediction.time[0],
                    "end": prediction.time[1],
                    "emotions": json.dumps(hume.HumePredictionEmotionResponseModel.to_multi_dict(prediction.emotions)),
                }
                cur.execute(
                    """
                    INSERT INTO conversation_model_transcripts
                        (uid, conversation_id, model_name, segment_id, start_ts, data)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (uid, conversation_id, model_name, segment_id) DO UPDATE
                        SET start_ts = EXCLUDED.start_ts,
                            data = EXCLUDED.data
                    """,
                    (
                        uid,
                        conversation_id,
                        model_name,
                        prediction_id,
                        prediction.time[0],
                        json.dumps(payload, default=_json_default),
                    ),
                )


def get_conversation_transcripts_by_model(uid: str, conversation_id: str):
    result: Dict[str, List[dict]] = {}
    with db.connection() as conn:
        with conn.cursor() as cur:
            for short_name, model_name in [
                ('deepgram', 'deepgram_streaming'),
                ('soniox', 'soniox_streaming'),
                ('speechmatics', 'speechmatics_streaming'),
                ('whisperx', 'fal_whisperx'),
            ]:
                cur.execute(
                    """
                    SELECT data FROM conversation_model_transcripts
                    WHERE uid = %s AND conversation_id = %s AND model_name = %s
                    ORDER BY start_ts
                    """,
                    (uid, conversation_id, model_name),
                )
                result[short_name] = [dict(row[0] or {}) for row in cur.fetchall()]
    return result


# ***********************************
# ********** OPENGLASS **************
# ***********************************


def store_conversation_photos(uid: str, conversation_id: str, photos: List[ConversationPhoto]):
    if not photos:
        return
    with db.batch() as conn:
        with conn.cursor() as cur:
            # Pull the conversation's protection level so photos get the same treatment.
            cur.execute(
                "SELECT data->>'data_protection_level' FROM conversations WHERE uid = %s AND id = %s",
                (uid, conversation_id),
            )
            row = cur.fetchone()
            level = (row[0] if row else None) or 'standard'

            for photo in photos:
                photo_id = photo.id or str(uuid.uuid4())
                data = photo.dict()
                data['id'] = photo_id
                prepared_data = _prepare_photo_for_write(data, uid, level)
                cur.execute(
                    """
                    INSERT INTO conversation_photos
                        (uid, conversation_id, photo_id, data)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (uid, conversation_id, photo_id) DO UPDATE
                        SET data = EXCLUDED.data
                    """,
                    (uid, conversation_id, photo_id, json.dumps(prepared_data, default=_json_default)),
                )


# ********************************
# ********** SYNCING *************
# ********************************


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
@with_photos(get_conversation_photos)
def get_closest_conversation_to_timestamps(uid: str, start_timestamp: int, end_timestamp: int) -> Optional[dict]:
    start_threshold = datetime.fromtimestamp(start_timestamp, tz=timezone.utc) - timedelta(minutes=2)
    end_threshold = datetime.fromtimestamp(end_timestamp, tz=timezone.utc) + timedelta(minutes=2)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_CONVERSATION_COLS}
                FROM conversations
                WHERE uid = %s
                  AND finished_at >= %s
                  AND started_at <= %s
                ORDER BY created_at DESC
                """,
                (uid, start_threshold, end_threshold),
            )
            conversations = [_row_to_conversation(r) for r in cur.fetchall()]

    logger.info(f'get_closest_conversation_to_timestamps len(conversations) {len(conversations)}')
    if not conversations:
        return None

    logger.info('get_closest_conversation_to_timestamps found:')
    for conversation in conversations:
        logger.info(f"- {conversation['id']} {conversation['started_at']} {conversation['finished_at']}")

    # get the conversation that has the closest start timestamp or end timestamp
    closest_conversation = None
    min_diff = float('inf')
    for conversation in conversations:
        conversation_start_timestamp = conversation['started_at'].timestamp()
        conversation_end_timestamp = conversation['finished_at'].timestamp()
        diff1 = abs(conversation_start_timestamp - start_timestamp)
        diff2 = abs(conversation_end_timestamp - end_timestamp)
        if diff1 < min_diff or diff2 < min_diff:
            min_diff = min(diff1, diff2)
            closest_conversation = conversation

    logger.info(f"get_closest_conversation_to_timestamps closest_conversation: {closest_conversation['id']}")
    return closest_conversation


@prepare_for_read(decrypt_func=_prepare_conversation_for_read)
@with_photos(get_conversation_photos)
def get_last_completed_conversation(uid: str) -> Optional[dict]:
    status_val = (
        ConversationStatus.completed.value
        if hasattr(ConversationStatus.completed, 'value')
        else ConversationStatus.completed
    )
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_CONVERSATION_COLS}
                FROM conversations
                WHERE uid = %s AND status = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (uid, status_val),
            )
            row = cur.fetchone()
            return _row_to_conversation(row) if row else None
