"""Chat messages, chat sessions, chat files. Postgres-backed port.

Three tables:

- chat_messages (uid, id) PK — per-message row.
  Typed columns: session_id, plugin_id, created_at. Everything else lives
  inside the `data` JSONB column (including `rating`, `reported`,
  `memories_id`, `files_id`, `chat_session_id`, encrypted `text`, etc.).

- chat_sessions (uid, id) PK — per-session row.
  Typed columns: plugin_id, created_at. Everything else lives inside
  `data` JSONB (title, preview, starred, updated_at, message_count,
  message_ids, file_ids, openai_thread_id, openai_assistant_id).

- chat_files (uid, id) PK — per-file row.
  Typed columns: created_at. Everything else lives inside `data` JSONB.

Also READS:
- conversations (uid, id) — cross-table hydrate for `include_conversations=True`.

Encryption semantics preserved unchanged: only messages may be
encrypted, and only the `text` field. `_prepare_data_for_write` and
`_prepare_message_for_read` still handle the per-message AES path.
"""

import copy
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from models.chat import Message
from utils import encryption
from ._client import db
from .helpers import set_data_protection_level, prepare_for_write, prepare_for_read
import logging

logger = logging.getLogger(__name__)

# Matches the old Firestore batch cap — kept so paging semantics don't change.
BATCH_LIMIT = 500

# *********************************
# ******* ENCRYPTION HELPERS ******
# *********************************


def _encrypt_chat_data(chat_data: Dict[str, Any], uid: str) -> Dict[str, Any]:
    data = copy.deepcopy(chat_data)

    if 'text' in data and isinstance(data['text'], str):
        data['text'] = encryption.encrypt(data['text'], uid)
    return data


def _decrypt_chat_data(chat_data: Dict[str, Any], uid: str) -> Dict[str, Any]:
    data = copy.deepcopy(chat_data)

    if 'text' in data and isinstance(data['text'], str):
        try:
            data['text'] = encryption.decrypt(data['text'], uid)
        except Exception:
            pass

    return data


def _prepare_data_for_write(data: Dict[str, Any], uid: str, level: str) -> Dict[str, Any]:
    if level == 'enhanced':
        return _encrypt_chat_data(data, uid)
    return data


def _prepare_message_for_read(message_data: Optional[Dict[str, Any]], uid: str) -> Optional[Dict[str, Any]]:
    if not message_data:
        return None

    level = message_data.get('data_protection_level')
    if level == 'enhanced':
        return _decrypt_chat_data(message_data, uid)

    return message_data


# *********************************************************************
# **** Row helpers: promote typed columns into the JSONB data dict ****
# *********************************************************************


_MESSAGE_COLS = 'uid, id, session_id, plugin_id, created_at, data'
_SESSION_COLS = 'uid, id, plugin_id, created_at, data'
_FILE_COLS = 'uid, id, created_at, data'


def _json_default(value):
    """JSON serializer for datetime / bytes values going into JSONB."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _row_to_message(row) -> Dict[str, Any]:
    _uid, mid, session_id, plugin_id, created_at, data = row
    out = dict(data or {})
    out['id'] = mid
    # The typed cols win — they're the source of truth for these fields.
    out['chat_session_id'] = session_id if session_id is not None else out.get('chat_session_id')
    out['plugin_id'] = plugin_id
    out['created_at'] = created_at
    return out


def _extract_message_typed(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the typed columns carved out of a message dict."""
    # chat_session_id is the field name in the Message model; session_id is the
    # desktop/chat_sessions-era alias. Either may be present on the dict.
    session_id = message_data.get('chat_session_id') or message_data.get('session_id')
    return {
        'session_id': session_id,
        'plugin_id': message_data.get('plugin_id') or message_data.get('app_id'),
        'created_at': message_data.get('created_at'),
    }


def _message_data_without_typed(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of message_data with the typed columns dropped from the JSONB body."""
    out = copy.deepcopy(message_data)
    for key in ('created_at',):
        out.pop(key, None)
    # Keep plugin_id / chat_session_id / session_id inside the data blob too so
    # reads reconstruct the full Message shape. Only `created_at` is the "sole
    # source of truth" typed col (legacy readers expect it as a datetime, not
    # a JSON string).
    return out


def _row_to_session(row) -> Dict[str, Any]:
    _uid, sid, plugin_id, created_at, data = row
    out = dict(data or {})
    out['id'] = sid
    out['plugin_id'] = plugin_id
    # created_at only overwritten if the JSONB version wasn't a datetime already.
    if 'created_at' not in out or out.get('created_at') is None:
        out['created_at'] = created_at
    return out


def _row_to_file(row) -> Dict[str, Any]:
    _uid, fid, created_at, data = row
    out = dict(data or {})
    out['id'] = fid
    if 'created_at' not in out or out.get('created_at') is None:
        out['created_at'] = created_at
    return out


# *****************************
# ********** CRUD *************
# *****************************


@set_data_protection_level(data_arg_name='message_data')
@prepare_for_write(data_arg_name='message_data', prepare_func=_prepare_data_for_write)
def add_message(uid: str, message_data: dict):
    # Match the Firestore original — `memories` is the hydrated front-facing
    # list; only `memories_id` (strings) gets persisted. Callers always pass
    # `Message.dict()` which has a default `memories: []`, so this key exists.
    if 'memories' in message_data:
        del message_data['memories']

    msg_id = message_data.get('id') or str(uuid.uuid4())
    # Ensure the id is on the stored dict too.
    message_data['id'] = msg_id

    typed = _extract_message_typed(message_data)
    data_body = _message_data_without_typed(message_data)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages
                    (uid, id, session_id, plugin_id, created_at, data)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET session_id = EXCLUDED.session_id,
                        plugin_id = EXCLUDED.plugin_id,
                        created_at = EXCLUDED.created_at,
                        data = EXCLUDED.data
                """,
                (
                    uid,
                    msg_id,
                    typed['session_id'],
                    typed['plugin_id'],
                    typed['created_at'] or datetime.now(timezone.utc),
                    json.dumps(data_body, default=_json_default),
                ),
            )
    return message_data


def add_app_message(text: str, app_id: str, uid: str, conversation_id: Optional[str] = None) -> Message:
    ai_message = Message(
        id=str(uuid.uuid4()),
        text=text,
        created_at=datetime.now(timezone.utc),
        sender='ai',
        app_id=app_id,
        from_external_integration=False,
        type='text',
        memories_id=[conversation_id] if conversation_id else [],
    )
    add_message(uid, ai_message.dict())
    return ai_message


def add_integration_chat_message(text: str, app_id: Optional[str], uid: str) -> Message:
    """Add a chat message from an external integration (e.g. notification API),
    linking it to the user's existing chat session so it appears in the chat feed."""
    chat_session = get_chat_session(uid, app_id=app_id)
    chat_session_id = chat_session['id'] if chat_session else None

    ai_message = Message(
        id=str(uuid.uuid4()),
        text=text,
        created_at=datetime.now(timezone.utc),
        sender='ai',
        app_id=app_id,
        from_external_integration=True,
        type='text',
        chat_session_id=chat_session_id,
    )
    add_message(uid, ai_message.dict())
    if chat_session_id:
        add_message_to_chat_session(uid, chat_session_id, ai_message.id)
    return ai_message


def add_summary_message(text: str, uid: str) -> Message:
    ai_message = Message(
        id=str(uuid.uuid4()),
        text=text,
        created_at=datetime.now(timezone.utc),
        sender='ai',
        app_id=None,
        from_external_integration=False,
        type='day_summary',
        memories_id=[],
    )
    add_message(uid, ai_message.dict())
    return ai_message


def _fetch_conversations_by_ids(uid: str, conversation_ids: List[str]) -> Dict[str, dict]:
    """Cross-table hydrate: pull conversations named by `memories_id`. Not
    decrypted — callers already get whatever the conversations module writes
    into the JSONB body. Matches the old db.get_all(doc_refs) behavior."""
    if not conversation_ids:
        return {}
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data
                FROM conversations
                WHERE uid = %s AND id = ANY(%s::text[])
                """,
                (uid, [str(cid) for cid in conversation_ids]),
            )
            out: Dict[str, dict] = {}
            for cid, data in cur.fetchall():
                merged = dict(data or {})
                merged['id'] = cid
                out[cid] = merged
            return out


def _fetch_files_by_ids(uid: str, file_ids: List[str]) -> Dict[str, dict]:
    if not file_ids:
        return {}
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_FILE_COLS}
                FROM chat_files
                WHERE uid = %s AND id = ANY(%s::text[])
                """,
                (uid, [str(fid) for fid in file_ids]),
            )
            out: Dict[str, dict] = {}
            for row in cur.fetchall():
                f = _row_to_file(row)
                out[f['id']] = f
            return out


@prepare_for_read(decrypt_func=_prepare_message_for_read)
def get_app_messages(uid: str, app_id: str, limit: int = 20, offset: int = 0, include_conversations: bool = False):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_MESSAGE_COLS}
                FROM chat_messages
                WHERE uid = %s AND plugin_id IS NOT DISTINCT FROM %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (uid, app_id, limit, offset),
            )
            rows = cur.fetchall()

    messages = []
    conversations_id: set = set()
    for row in rows:
        message = _row_to_message(row)
        if message.get('reported') is True:
            continue
        messages.append(message)
        conversations_id.update(message.get('memories_id', []))

    if not include_conversations:
        return messages

    conversations = _fetch_conversations_by_ids(uid, list(conversations_id))
    for message in messages:
        message['memories'] = [
            conversations[conversation_id]
            for conversation_id in message.get('memories_id', [])
            if conversation_id in conversations
        ]
    return messages


@prepare_for_read(decrypt_func=_prepare_message_for_read)
def get_messages(
    uid: str,
    limit: int = 20,
    offset: int = 0,
    include_conversations: bool = False,
    app_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
):
    logger.info(f'get_messages {uid} {limit} {offset} {app_id} {include_conversations}')

    # Session-scoped query: filter by session only. Otherwise filter by plugin_id
    # (None == main chat). `IS NOT DISTINCT FROM` handles the NULL-equals-NULL case.
    parts = [f"SELECT {_MESSAGE_COLS} FROM chat_messages WHERE uid = %s"]
    params: List[Any] = [uid]
    if chat_session_id:
        parts.append("AND session_id = %s")
        params.append(chat_session_id)
    else:
        parts.append("AND plugin_id IS NOT DISTINCT FROM %s")
        params.append(app_id)
    parts.append("ORDER BY created_at DESC")
    parts.append("LIMIT %s OFFSET %s")
    params.extend([limit, offset])

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            rows = cur.fetchall()

    messages: List[dict] = []
    conversations_id: set = set()
    files_id: set = set()
    for row in rows:
        message = _row_to_message(row)
        if message.get('reported') is True:
            continue
        messages.append(message)
        conversations_id.update(message.get('memories_id', []))
        files_id.update(message.get('files_id', []))

    if not include_conversations:
        return messages

    conversations = _fetch_conversations_by_ids(uid, list(conversations_id))
    for message in messages:
        message['memories'] = [
            conversations[conversation_id]
            for conversation_id in message.get('memories_id', [])
            if conversation_id in conversations
        ]

    files = _fetch_files_by_ids(uid, list(files_id))
    for message in messages:
        message['files'] = [files[file_id] for file_id in message.get('files_id', []) if file_id in files]

    return messages


def get_message_count(uid: str) -> int:
    """Return the total number of chat messages for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chat_messages WHERE uid = %s", (uid,))
            row = cur.fetchone()
            return int(row[0]) if row else 0


def iter_all_messages(uid: str, batch_size: int = 1000):
    """Yield all chat messages for a user, decrypted, in batches. Used for
    streaming data export. Uses a server-side cursor (matching the
    conversations.py pattern) so we don't load everything into memory."""
    sql = f"SELECT {_MESSAGE_COLS} FROM chat_messages WHERE uid = %s ORDER BY created_at DESC"
    with db.connection() as conn:
        cursor_name = f"iter_msgs_{uuid.uuid4().hex}"
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = batch_size
            cur.execute(sql, (uid,))
            while True:
                batch_rows = cur.fetchmany(batch_size)
                if not batch_rows:
                    break
                for row in batch_rows:
                    msg = _row_to_message(row)
                    yield _prepare_message_for_read(msg, uid) or msg


def get_message(uid: str, message_id: str) -> tuple[Message, str] | None:
    """Fetch a single message by id.  Returns (Message, doc_id) or None.

    The Firestore impl returned a separate doc_id alongside the Message, since
    Firestore auto-generated doc ids independent of the `id` field. In Postgres
    the primary key IS the id, so doc_id == message_id."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_MESSAGE_COLS} FROM chat_messages WHERE uid = %s AND id = %s LIMIT 1",
                (uid, message_id),
            )
            row = cur.fetchone()
            if row is None:
                return None

    message_data = _row_to_message(row)
    if not message_data:
        return None

    decrypted_data = _prepare_message_for_read(message_data, uid)
    message = Message(**decrypted_data)

    return message, message_id


def report_message(uid: str, msg_doc_id: str):
    """Flip `reported=true` on a message's JSONB body. `msg_doc_id` is the PK
    `id` in Postgres (callers pass whatever `get_message` returned as the
    second tuple element)."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    UPDATE chat_messages
                    SET data = data || '{"reported": true}'::jsonb
                    WHERE uid = %s AND id = %s
                    """,
                    (uid, msg_doc_id),
                )
                return {"message": "Message reported"}
            except Exception as e:
                logger.error(f"Update failed: {e}")
                return {"message": f"Update failed: {e}"}


def update_message_rating(uid: str, message_id: str, rating: int | None):
    """Update the rating on a message row's JSONB body.

    Args:
        uid: User ID
        message_id: Message ID (PK in chat_messages)
        rating: Rating value (1 = thumbs up, -1 = thumbs down, None = no rating)
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM chat_messages WHERE uid = %s AND id = %s",
                (uid, message_id),
            )
            if cur.fetchone() is None:
                logger.warning(f"Message {message_id} not found for user {uid}")
                return False

            try:
                cur.execute(
                    """
                    UPDATE chat_messages
                    SET data = data || %s::jsonb
                    WHERE uid = %s AND id = %s
                    """,
                    (json.dumps({'rating': rating}), uid, message_id),
                )
                logger.info(f"Updated message {message_id} rating to {rating}")
                return True
            except Exception as e:
                logger.error(f"Failed to update message rating: {e}")
                return False


def batch_delete_messages(
    parent_doc_ref=None,
    batch_size: int = 450,
    app_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    uid: Optional[str] = None,
):
    """Bulk-delete messages matching the given filters.

    Signature accepts `parent_doc_ref` (legacy Firestore user_ref) for call-site
    compatibility — when callers pass a Firestore-era ref, we coerce its id to
    `uid`. When `uid` is passed explicitly, we use it directly.
    """
    if uid is None and parent_doc_ref is not None:
        # Legacy path: the Firestore impl took `parent_doc_ref` (a
        # user document). The new impl just needs the uid.
        uid = getattr(parent_doc_ref, 'id', None) or getattr(parent_doc_ref, 'uid', None)
    if uid is None:
        raise ValueError("batch_delete_messages requires `uid`")

    logger.info(f'batch_delete_messages uid={uid} app_id={app_id} chat_session_id={chat_session_id}')

    parts = ["DELETE FROM chat_messages WHERE uid = %s"]
    params: List[Any] = [uid]
    # Match the Firestore semantics: app_id was always applied, chat_session_id
    # optional on top.
    parts.append("AND plugin_id IS NOT DISTINCT FROM %s")
    params.append(app_id)
    if chat_session_id:
        parts.append("AND session_id = %s")
        params.append(chat_session_id)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            deleted = cur.rowcount or 0
            logger.info(f'Deleted {deleted} messages')


def clear_chat(uid: str, app_id: Optional[str] = None, chat_session_id: Optional[str] = None):
    """Delete every message for a user scoped by app/session. Returns None on
    success, or {'message': ...} on error (matches the Firestore contract)."""
    try:
        # Mirror the old user-exists check so callers keep getting that message
        # on a bogus uid. The check is cheap and preserves behavior.
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE uid = %s", (uid,))
                if cur.fetchone() is None:
                    return {"message": "User not found"}

        batch_delete_messages(uid=uid, app_id=app_id, chat_session_id=chat_session_id)
        return None
    except Exception as e:
        return {"message": str(e)}


def add_multi_files(uid: str, files_data: list):
    """Insert many files atomically."""
    if not files_data:
        return
    with db.batch() as conn:
        with conn.cursor() as cur:
            for file_data in files_data:
                fid = file_data['id']
                created_at = file_data.get('created_at') or datetime.now(timezone.utc)
                cur.execute(
                    """
                    INSERT INTO chat_files (uid, id, created_at, data)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (uid, id) DO UPDATE
                        SET created_at = EXCLUDED.created_at,
                            data = EXCLUDED.data
                    """,
                    (uid, fid, created_at, json.dumps(file_data, default=_json_default)),
                )


def get_chat_files(uid: str, files_id: List[str] = []):
    """Return all files if `files_id` is empty, else only matching ids."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            if len(files_id) == 0:
                cur.execute(
                    f"SELECT {_FILE_COLS} FROM chat_files WHERE uid = %s",
                    (uid,),
                )
            else:
                cur.execute(
                    f"""
                    SELECT {_FILE_COLS} FROM chat_files
                    WHERE uid = %s AND id = ANY(%s::text[])
                    """,
                    (uid, [str(fid) for fid in files_id]),
                )
            return [_row_to_file(r) for r in cur.fetchall()]


def get_chat_files_desc(uid: str, files_id: List[str] = [], limit: int = 10):
    """Most recent chat files, optionally filtered by file IDs."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            if len(files_id) == 0:
                cur.execute(
                    f"""
                    SELECT {_FILE_COLS} FROM chat_files
                    WHERE uid = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (uid, limit),
                )
            else:
                cur.execute(
                    f"""
                    SELECT {_FILE_COLS} FROM chat_files
                    WHERE uid = %s AND id = ANY(%s::text[])
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (uid, [str(fid) for fid in files_id], limit),
                )
            return [_row_to_file(r) for r in cur.fetchall()]


def delete_multi_files(uid: str, files_data: list):
    """Delete many files atomically. `files_data` is a list of dicts with 'id'."""
    if not files_data:
        return
    ids = [fd["id"] for fd in files_data if fd.get("id")]
    if not ids:
        return
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_files WHERE uid = %s AND id = ANY(%s::text[])",
                (uid, ids),
            )


# *****************************
# ********** SESSIONS *********
# *****************************


def _extract_session_typed(session_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'plugin_id': session_data.get('plugin_id') or session_data.get('app_id'),
        'created_at': session_data.get('created_at') or datetime.now(timezone.utc),
    }


def add_chat_session(uid: str, chat_session_data: dict):
    sid = chat_session_data['id']
    typed = _extract_session_typed(chat_session_data)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (uid, id, plugin_id, created_at, data)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET plugin_id = EXCLUDED.plugin_id,
                        created_at = EXCLUDED.created_at,
                        data = EXCLUDED.data
                """,
                (
                    uid,
                    sid,
                    typed['plugin_id'],
                    typed['created_at'],
                    json.dumps(chat_session_data, default=_json_default),
                ),
            )
    return chat_session_data


def get_chat_session(uid: str, app_id: Optional[str] = None):
    """Return any single session matching plugin_id. Used for 'does this app
    already have a session' probes. `IS NOT DISTINCT FROM` matches NULL ==
    NULL (main chat)."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_SESSION_COLS}
                FROM chat_sessions
                WHERE uid = %s AND plugin_id IS NOT DISTINCT FROM %s
                LIMIT 1
                """,
                (uid, app_id),
            )
            row = cur.fetchone()
            return _row_to_session(row) if row else None


def get_chat_session_by_id(uid: str, chat_session_id: str):
    """Get a specific chat session by its ID."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_SESSION_COLS}
                FROM chat_sessions
                WHERE uid = %s AND id = %s
                """,
                (uid, chat_session_id),
            )
            row = cur.fetchone()
            return _row_to_session(row) if row else None


def delete_chat_session(uid, chat_session_id, cascade_messages: bool = False):
    """Delete a chat session; optionally cascade to all its messages.

    When `cascade_messages=True` and the session doesn't exist, returns False
    (matches the Firestore check). The whole thing runs inside a single
    transaction so partial failures don't leave orphan messages.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            if cascade_messages:
                cur.execute(
                    "SELECT 1 FROM chat_sessions WHERE uid = %s AND id = %s",
                    (uid, chat_session_id),
                )
                if cur.fetchone() is None:
                    return False
                cur.execute(
                    "DELETE FROM chat_messages WHERE uid = %s AND session_id = %s",
                    (uid, chat_session_id),
                )
            cur.execute(
                "DELETE FROM chat_sessions WHERE uid = %s AND id = %s",
                (uid, chat_session_id),
            )


def add_message_to_chat_session(uid: str, chat_session_id: str, message_id: str):
    """Append a message id to the session's `message_ids` array (JSONB).
    The Firestore impl used an array-union primitive that dedupes; we use
    jsonb_set + a uniqueness guard so we don't double-append."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM chat_sessions WHERE uid = %s AND id = %s FOR UPDATE",
                (uid, chat_session_id),
            )
            row = cur.fetchone()
            if row is None:
                return
            current = dict(row[0] or {})
            ids = list(current.get('message_ids') or [])
            if message_id in ids:
                return
            ids.append(message_id)
            cur.execute(
                """
                UPDATE chat_sessions
                SET data = jsonb_set(data, '{message_ids}', %s::jsonb, TRUE)
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(ids), uid, chat_session_id),
            )


def add_files_to_chat_session(uid: str, chat_session_id: str, file_ids: List[str]):
    if not file_ids:
        return
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM chat_sessions WHERE uid = %s AND id = %s FOR UPDATE",
                (uid, chat_session_id),
            )
            row = cur.fetchone()
            if row is None:
                return
            current = dict(row[0] or {})
            existing = list(current.get('file_ids') or [])
            for fid in file_ids:
                if fid not in existing:
                    existing.append(fid)
            cur.execute(
                """
                UPDATE chat_sessions
                SET data = jsonb_set(data, '{file_ids}', %s::jsonb, TRUE)
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(existing), uid, chat_session_id),
            )


def update_chat_session_openai_ids(uid: str, chat_session_id: str, thread_id: str, assistant_id: str):
    """Update OpenAI thread and assistant IDs on a chat session."""
    update_data: Dict[str, Any] = {}
    if thread_id:
        update_data['openai_thread_id'] = thread_id
    if assistant_id:
        update_data['openai_assistant_id'] = assistant_id
    if not update_data:
        return

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions
                SET data = data || %s::jsonb
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(update_data), uid, chat_session_id),
            )
    logger.info(f"Updated session {chat_session_id} with thread {thread_id} and assistant {assistant_id}")


# ============================================================================
# CHAT SESSIONS (v2)
#
# v2 sessions support: title, preview, message_count, starred, updated_at.
# v1 sessions store: message_ids, file_ids, openai_thread_id.
# Both schemas coexist. Both write plugin_id alongside app_id for cross-platform
# query compat.
# ============================================================================


def create_chat_session(uid: str, title: str = None, app_id: str = None) -> dict:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = {
        'id': session_id,
        'title': title or 'New Chat',
        'preview': None,
        'created_at': now,
        'updated_at': now,
        'app_id': app_id,
        'plugin_id': app_id,
        'message_count': 0,
        'starred': False,
    }
    add_chat_session(uid, doc)
    return doc


def acquire_chat_session(uid: str, app_id: str = None) -> str:
    """Get or create a chat session for the given app_id (None = main chat)."""
    existing = get_chat_session(uid, app_id=app_id)
    if existing:
        return existing['id']
    session = create_chat_session(uid, app_id=app_id)
    return session['id']


def get_chat_sessions(
    uid: str, app_id: str = None, limit: int = 50, offset: int = 0, starred: bool = None
) -> List[dict]:
    """Order by updated_at — v2 sessions always have this field.

    Legacy v1 sessions without updated_at in the JSONB body return NULL from
    `data->>'updated_at'`. In Postgres, NULLs sort last under DESC by default,
    which effectively excludes them from the top of the list — matching the
    Firestore behavior where `order_by('updated_at')` silently dropped rows
    missing that field.
    """
    parts = [f"SELECT {_SESSION_COLS} FROM chat_sessions WHERE uid = %s"]
    params: List[Any] = [uid]

    parts.append("AND plugin_id IS NOT DISTINCT FROM %s")
    params.append(app_id)
    if starred is not None:
        parts.append("AND (data->>'starred')::boolean = %s")
        params.append(bool(starred))
    # Exclude rows with no updated_at (legacy v1) to match Firestore's drop-on-missing.
    parts.append("AND data ? 'updated_at'")
    parts.append("ORDER BY (data->>'updated_at') DESC")
    parts.append("LIMIT %s OFFSET %s")
    params.extend([limit, offset])

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            return [_row_to_session(r) for r in cur.fetchall()]


def update_chat_session(uid: str, session_id: str, title: str = None, starred: bool = None) -> Optional[dict]:
    now = datetime.now(timezone.utc)
    updates: Dict[str, Any] = {'updated_at': now}
    if title is not None:
        updates['title'] = title
    if starred is not None:
        updates['starred'] = starred

    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_SESSION_COLS} FROM chat_sessions WHERE uid = %s AND id = %s",
                (uid, session_id),
            )
            row = cur.fetchone()
            if row is None:
                return None

            cur.execute(
                """
                UPDATE chat_sessions
                SET data = data || %s::jsonb
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(updates, default=_json_default), uid, session_id),
            )

            cur.execute(
                f"SELECT {_SESSION_COLS} FROM chat_sessions WHERE uid = %s AND id = %s",
                (uid, session_id),
            )
            row = cur.fetchone()

    if row is None:
        return None
    result = _row_to_session(row)
    result['id'] = session_id
    return result


# **************************************
# ********* MIGRATION HELPERS **********
# **************************************


def get_chats_to_migrate(uid: str, target_level: str) -> List[dict]:
    """Find all chat messages not at the target protection level."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data->>'data_protection_level' AS level
                FROM chat_messages
                WHERE uid = %s
                """,
                (uid,),
            )
            to_migrate: List[dict] = []
            for mid, level in cur.fetchall():
                current_level = level or 'standard'
                if target_level != current_level:
                    to_migrate.append({'id': mid, 'type': 'chat'})
            return to_migrate


def migrate_chats_level_batch(uid: str, message_doc_ids: List[str], target_level: str):
    """Migrate a batch of chat messages to `target_level`."""
    if not message_doc_ids:
        return

    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_MESSAGE_COLS}
                FROM chat_messages
                WHERE uid = %s AND id = ANY(%s::text[])
                FOR UPDATE
                """,
                (uid, list(message_doc_ids)),
            )
            rows = cur.fetchall()

            for row in rows:
                msg_dict = _row_to_message(row)
                mid = msg_dict['id']
                current_level = msg_dict.get('data_protection_level', 'standard')
                if current_level == target_level:
                    continue

                plain_data = _prepare_message_for_read(msg_dict, uid) or msg_dict
                plain_text = plain_data.get('text')
                migrated_text = plain_text
                if target_level == 'enhanced' and isinstance(plain_text, str):
                    migrated_text = encryption.encrypt(plain_text, uid)

                update_data = {'data_protection_level': target_level, 'text': migrated_text}
                cur.execute(
                    """
                    UPDATE chat_messages
                    SET data = data || %s::jsonb
                    WHERE uid = %s AND id = %s
                    """,
                    (json.dumps(update_data, default=_json_default), uid, mid),
                )


# ============================================================================
# MESSAGES (v2)
#
# Persistence-only message writes (no LLM streaming). They write the same
# field set as the Message model for cross-platform compatibility:
#   plugin_id, app_id, type='text', chat_session_id, from_external_integration
#
# When session_id is not provided, acquire_chat_session() auto-creates one.
# ============================================================================


def save_message(
    uid: str, text: str, sender: str, app_id: str = None, session_id: str = None, metadata: str = None
) -> dict:
    """Save a chat message for the desktop app."""
    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    if not session_id:
        session_id = acquire_chat_session(uid, app_id=app_id)

    doc = {
        'id': msg_id,
        'text': text,
        'created_at': now,
        'sender': sender,
        'type': 'text',
        'app_id': app_id,
        'plugin_id': app_id,
        'session_id': session_id,
        'chat_session_id': session_id,
        'from_external_integration': False,
        'rating': None,
        'reported': False,
        'memories_id': [],
        'metadata': metadata,
    }

    with db.batch() as conn:
        with conn.cursor() as cur:
            typed = _extract_message_typed(doc)
            data_body = _message_data_without_typed(doc)
            cur.execute(
                """
                INSERT INTO chat_messages (uid, id, session_id, plugin_id, created_at, data)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    uid,
                    msg_id,
                    typed['session_id'],
                    typed['plugin_id'],
                    typed['created_at'] or now,
                    json.dumps(data_body, default=_json_default),
                ),
            )

            # Session update: message_count += 1, updated_at, preview.
            if session_id:
                cur.execute(
                    "SELECT data FROM chat_sessions WHERE uid = %s AND id = %s FOR UPDATE",
                    (uid, session_id),
                )
                srow = cur.fetchone()
                if srow is not None:
                    sdata = dict(srow[0] or {})
                    sdata['updated_at'] = now
                    sdata['message_count'] = int(sdata.get('message_count') or 0) + 1
                    sdata['preview'] = text[:100] if text else None
                    cur.execute(
                        """
                        UPDATE chat_sessions
                        SET data = data || %s::jsonb
                        WHERE uid = %s AND id = %s
                        """,
                        (
                            json.dumps(
                                {
                                    'updated_at': now,
                                    'message_count': sdata['message_count'],
                                    'preview': sdata['preview'],
                                },
                                default=_json_default,
                            ),
                            uid,
                            session_id,
                        ),
                    )

    return {'id': msg_id, 'created_at': now.isoformat()}


def delete_messages(uid: str, app_id: str = None, session_id: str = None) -> int:
    """Delete messages matching app_id/session_id. Returns count deleted."""
    parts = ["DELETE FROM chat_messages WHERE uid = %s"]
    params: List[Any] = [uid]
    if session_id:
        parts.append("AND session_id = %s")
        params.append(session_id)
    else:
        parts.append("AND plugin_id IS NOT DISTINCT FROM %s")
        params.append(app_id)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            return cur.rowcount or 0
