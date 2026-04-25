"""Memories (distilled facts/preferences). Postgres-backed port.

Table: memories (uid, id) PK
  Typed columns: category, scoring, is_locked, created_at, updated_at
  Everything else lives inside `data` JSONB (content, tags, headline, visibility,
  reviewed, user_review, manually_added, edited, app_id, conversation_id,
  memory_id, kg_extracted, data_protection_level, ...).

Indexes: (uid, created_at DESC), (uid, category), (scoring DESC NULLS LAST),
GIN on data.

Encryption helpers (_prepare_*_for_write/read, _encrypt_memory_data,
_decrypt_memory_data) are preserved unchanged from the Firestore version.
"""

import copy
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from database import users as users_db
from utils import encryption
from ._client import db
from .helpers import set_data_protection_level, prepare_for_write, prepare_for_read
import logging

logger = logging.getLogger(__name__)


# *********************************
# ******* ENCRYPTION HELPERS ******
# *********************************


def _encrypt_memory_data(memory_data: Dict[str, Any], uid: str) -> Dict[str, Any]:
    data = copy.deepcopy(memory_data)

    if 'content' in data and isinstance(data['content'], str):
        data['content'] = encryption.encrypt(data['content'], uid)
    return data


def _decrypt_memory_data(memory_data: Dict[str, Any], uid: str) -> Dict[str, Any]:
    data = copy.deepcopy(memory_data)

    if 'content' in data and isinstance(data['content'], str):
        try:
            data['content'] = encryption.decrypt(data['content'], uid)
        except Exception:
            pass
    return data


def _prepare_data_for_write(data: Dict[str, Any], uid: str, level: str) -> Dict[str, Any]:
    if level == 'enhanced':
        return _encrypt_memory_data(data, uid)
    return data


def _prepare_memory_for_read(memory_data: Optional[Dict[str, Any]], uid: str) -> Optional[Dict[str, Any]]:
    if not memory_data:
        return None

    level = memory_data.get('data_protection_level')
    if level == 'enhanced':
        return _decrypt_memory_data(memory_data, uid)

    return memory_data


# *********************************************************************
# **** Row helpers: promote typed columns into the JSONB data dict ****
# *********************************************************************


_MEMORY_COLS = 'id, category, scoring, is_locked, created_at, updated_at, data'
_TYPED_KEYS = ('category', 'scoring', 'is_locked', 'created_at', 'updated_at')


def _row_to_memory(row) -> Dict[str, Any]:
    """Merge typed columns back into the JSONB data dict so callers see one flat dict."""
    mid, category, scoring, is_locked, created_at, updated_at, data = row
    out = dict(data or {})
    out['id'] = mid
    out['category'] = category
    out['scoring'] = scoring
    out['is_locked'] = is_locked
    out['created_at'] = created_at
    out['updated_at'] = updated_at
    return out


def _extract_typed_cols(memory_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'category': memory_data.get('category'),
        'scoring': memory_data.get('scoring'),
        'is_locked': bool(memory_data.get('is_locked', False)),
        'created_at': memory_data.get('created_at'),
        'updated_at': memory_data.get('updated_at'),
    }


def _data_without_typed_cols(memory_data: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(memory_data)
    for key in _TYPED_KEYS:
        out.pop(key, None)
    return out


def _json_default(value):
    """JSON serializer for datetime values going into JSONB."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# *****************************
# ********** CRUD *************
# *****************************


@prepare_for_read(decrypt_func=_prepare_memory_for_read)
def get_memories(
    uid: str,
    limit: int = 100,
    offset: int = 0,
    categories: List[str] = [],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    logger.info(f'get_memories db {uid} {limit} {offset} {categories} {start_date} {end_date}')

    parts = [f"SELECT {_MEMORY_COLS} FROM memories WHERE uid = %s"]
    params: List[Any] = [uid]

    if categories:
        parts.append("AND category = ANY(%s::text[])")
        params.append(list(categories))

    if start_date:
        parts.append("AND created_at >= %s")
        params.append(start_date)

    if end_date:
        parts.append("AND created_at <= %s")
        params.append(end_date)

    parts.append("ORDER BY scoring DESC NULLS LAST, created_at DESC")
    parts.append("LIMIT %s OFFSET %s")
    params.extend([limit, offset])

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            memories = [_row_to_memory(r) for r in cur.fetchall()]

    logger.info(f"get_memories {len(memories)}")
    # TODO: push user_review filter into the SQL query once indexed.
    result = [memory for memory in memories if memory.get('user_review') is not False]
    return result


@prepare_for_read(decrypt_func=_prepare_memory_for_read)
def get_user_public_memories(uid: str, limit: int = 100, offset: int = 0):
    logger.info(f'get_public_memories {limit} {offset}')

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_MEMORY_COLS} FROM memories
                WHERE uid = %s
                ORDER BY scoring DESC NULLS LAST, created_at DESC
                LIMIT %s OFFSET %s
                """,
                (uid, limit, offset),
            )
            memories = [_row_to_memory(r) for r in cur.fetchall()]

    # Consider visibility as 'public' if it's missing
    public_memories = [memory for memory in memories if memory.get('visibility', 'public') == 'public']
    return public_memories


@prepare_for_read(decrypt_func=_prepare_memory_for_read)
def get_non_filtered_memories(uid: str, limit: int = 100, offset: int = 0):
    logger.info(f'get_non_filtered_memories {uid} {limit} {offset}')
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_MEMORY_COLS} FROM memories
                WHERE uid = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (uid, limit, offset),
            )
            return [_row_to_memory(r) for r in cur.fetchall()]


@set_data_protection_level(data_arg_name='data')
@prepare_for_write(data_arg_name='data', prepare_func=_prepare_data_for_write)
def create_memory(uid: str, data: dict):
    typed = _extract_typed_cols(data)
    data_body = _data_without_typed_cols(data)
    created_at = typed['created_at'] or datetime.now(timezone.utc)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories
                    (uid, id, category, scoring, is_locked, created_at, updated_at, data)
                VALUES (%s, %s, %s, %s, %s, %s, now(), %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET category = EXCLUDED.category,
                        scoring = EXCLUDED.scoring,
                        is_locked = EXCLUDED.is_locked,
                        updated_at = now(),
                        data = EXCLUDED.data
                """,
                (
                    uid,
                    data['id'],
                    typed['category'],
                    typed['scoring'],
                    typed['is_locked'],
                    created_at,
                    json.dumps(data_body, default=_json_default),
                ),
            )


@set_data_protection_level(data_arg_name='data')
@prepare_for_write(data_arg_name='data', prepare_func=_prepare_data_for_write)
def save_memories(uid: str, data: List[dict]):
    if not data:
        return

    with db.batch() as conn:
        with conn.cursor() as cur:
            for memory in data:
                typed = _extract_typed_cols(memory)
                data_body = _data_without_typed_cols(memory)
                created_at = typed['created_at'] or datetime.now(timezone.utc)

                cur.execute(
                    """
                    INSERT INTO memories
                        (uid, id, category, scoring, is_locked, created_at, updated_at, data)
                    VALUES (%s, %s, %s, %s, %s, %s, now(), %s::jsonb)
                    ON CONFLICT (uid, id) DO UPDATE
                        SET category = EXCLUDED.category,
                            scoring = EXCLUDED.scoring,
                            is_locked = EXCLUDED.is_locked,
                            updated_at = now(),
                            data = EXCLUDED.data
                    """,
                    (
                        uid,
                        memory['id'],
                        typed['category'],
                        typed['scoring'],
                        typed['is_locked'],
                        created_at,
                        json.dumps(data_body, default=_json_default),
                    ),
                )


def delete_memories(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE uid = %s", (uid,))


@prepare_for_read(decrypt_func=_prepare_memory_for_read)
def get_memory(uid: str, memory_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_MEMORY_COLS} FROM memories WHERE uid = %s AND id = %s",
                (uid, memory_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_memory(row)


def get_memories_by_ids(uid: str, memory_ids: List[str]) -> List[dict]:
    """
    Batch fetch multiple memories by their IDs.
    """
    if not memory_ids:
        return []

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_MEMORY_COLS} FROM memories
                WHERE uid = %s AND id = ANY(%s::text[])
                """,
                (uid, [str(m) for m in memory_ids]),
            )
            rows = cur.fetchall()

    memories = []
    for row in rows:
        memory_data = _row_to_memory(row)
        # Apply decryption if needed
        decrypted = _prepare_memory_for_read(memory_data, uid)
        if decrypted:
            memories.append(decrypted)
    return memories


def mark_memory_promoted_to_edwin(
    uid: str, memory_id: str, note: Optional[str] = None
) -> bool:
    """Flag an Omi memory as promoted into Edwin's curated memory store.

    Edwin's memory pipeline (see container/agent-runner/src/ipc-mcp-stdio.ts
    `pendant_memory_promote`) writes a markdown file under
    agents/edwin/memory/ with frontmatter linking back to this memory_id.
    Stamping the source memory lets us answer "show me memories I haven't
    promoted yet" queries later, and gives us provenance in both directions.
    Returns False if the memory does not exist.
    """
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        'promoted_to_edwin': True,
        'promoted_to_edwin_at': now.isoformat(),
    }
    if note:
        payload['promoted_to_edwin_note'] = note
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories
                SET data = data || %s::jsonb,
                    updated_at = %s
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(payload), now, uid, memory_id),
            )
            return cur.rowcount > 0


def review_memory(uid: str, memory_id: str, value: bool):
    now = datetime.now(timezone.utc)
    payload = {'reviewed': True, 'user_review': value}
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories
                SET data = data || %s::jsonb,
                    updated_at = %s
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(payload), now, uid, memory_id),
            )


def set_memory_kg_extracted(uid: str, memory_id: str):
    now = datetime.now(timezone.utc)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories
                SET data = data || %s::jsonb,
                    updated_at = %s
                WHERE uid = %s AND id = %s
                """,
                (json.dumps({'kg_extracted': True}), now, uid, memory_id),
            )


def change_memory_visibility(uid: str, memory_id: str, value: str):
    now = datetime.now(timezone.utc)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories
                SET data = data || %s::jsonb,
                    updated_at = %s
                WHERE uid = %s AND id = %s
                """,
                (json.dumps({'visibility': value}), now, uid, memory_id),
            )


def update_memory_fields(uid: str, memory_id: str, data: dict):
    """Updates specified fields for a memory and sets the updated_at timestamp.

    Typed columns (category, scoring, is_locked) are promoted when present;
    everything else is shallow-merged into the JSONB `data` column.
    """
    if not data:
        return

    payload = dict(data)
    # Peel off any typed-column updates
    typed_updates: Dict[str, Any] = {}
    for key in ('category', 'scoring', 'is_locked'):
        if key in payload:
            typed_updates[key] = payload.pop(key)

    now = datetime.now(timezone.utc)

    with db.batch() as conn:
        with conn.cursor() as cur:
            if payload:
                cur.execute(
                    """
                    UPDATE memories
                    SET data = data || %s::jsonb,
                        updated_at = %s
                    WHERE uid = %s AND id = %s
                    """,
                    (json.dumps(payload, default=_json_default), now, uid, memory_id),
                )
            if typed_updates:
                set_parts = []
                params: List[Any] = []
                for col, val in typed_updates.items():
                    set_parts.append(f"{col} = %s")
                    params.append(val)
                set_parts.append("updated_at = %s")
                params.append(now)
                params.extend([uid, memory_id])
                cur.execute(
                    f"UPDATE memories SET {', '.join(set_parts)} WHERE uid = %s AND id = %s",
                    tuple(params),
                )
            elif not payload:
                # Nothing to update, still bump updated_at for parity with Firestore.
                cur.execute(
                    "UPDATE memories SET updated_at = %s WHERE uid = %s AND id = %s",
                    (now, uid, memory_id),
                )


def edit_memory(uid: str, memory_id: str, value: str):
    now = datetime.now(timezone.utc)
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM memories WHERE uid = %s AND id = %s FOR UPDATE",
                (uid, memory_id),
            )
            row = cur.fetchone()
            if row is None:
                return

            doc_level = (row[0] or {}).get('data_protection_level', 'standard')
            content = value
            if doc_level == 'enhanced':
                content = encryption.encrypt(content, uid)

            payload = {'content': content, 'edited': True}
            cur.execute(
                """
                UPDATE memories
                SET data = data || %s::jsonb,
                    updated_at = %s
                WHERE uid = %s AND id = %s
                """,
                (json.dumps(payload), now, uid, memory_id),
            )


def delete_memory(uid: str, memory_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memories WHERE uid = %s AND id = %s",
                (uid, memory_id),
            )


def delete_all_memories(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE uid = %s", (uid,))


def get_memory_ids_for_conversation(uid: str, conversation_id: str) -> List[str]:
    """Get all memory IDs associated with a conversation."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM memories
                WHERE uid = %s AND data->>'memory_id' = %s
                """,
                (uid, conversation_id),
            )
            return [row[0] for row in cur.fetchall()]


def delete_memories_for_conversation(uid: str, memory_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM memories
                WHERE uid = %s AND data->>'memory_id' = %s
                """,
                (uid, memory_id),
            )
            removed = cur.rowcount or 0
    logger.info(f'delete_memories_for_conversation {memory_id} {removed}')


def unlock_all_memories(uid: str):
    """
    Find all memories for a user with is_locked = TRUE and update them to is_locked = FALSE.
    """
    now = datetime.now(timezone.utc)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories
                SET is_locked = FALSE,
                    updated_at = %s
                WHERE uid = %s AND is_locked = TRUE
                """,
                (now, uid),
            )
    logger.info(f"Unlocked all memories for user {uid}")


# **************************************
# ********* MIGRATION HELPERS **********
# **************************************


def get_memories_to_migrate(uid: str, target_level: str) -> List[dict]:
    """
    Finds all memories that are not at the target protection level.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data->>'data_protection_level' AS level
                FROM memories
                WHERE uid = %s
                """,
                (uid,),
            )
            to_migrate: List[dict] = []
            for mid, level in cur.fetchall():
                current_level = level or 'standard'
                if target_level != current_level:
                    to_migrate.append({'id': mid, 'type': 'memory'})
            return to_migrate


def migrate_memories_level_batch(uid: str, memory_ids: List[str], target_level: str):
    """
    Migrates a batch of memories to the target protection level,
    wrapped in a single transaction.
    """
    if not memory_ids:
        return

    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_MEMORY_COLS}
                FROM memories
                WHERE uid = %s AND id = ANY(%s::text[])
                FOR UPDATE
                """,
                (uid, list(memory_ids)),
            )
            rows = cur.fetchall()

            for row in rows:
                memory_dict = _row_to_memory(row)
                mem_id = memory_dict['id']
                current_level = memory_dict.get('data_protection_level', 'standard')

                if current_level == target_level:
                    continue

                # Decrypt to get a clean slate, then re-encrypt at the new level.
                plain_data = _prepare_memory_for_read(memory_dict, uid) or memory_dict
                plain_content = plain_data.get('content')
                migrated_content = plain_content
                if target_level == 'enhanced':
                    if isinstance(plain_content, str):
                        migrated_content = encryption.encrypt(plain_content, uid)

                update_payload = {
                    'data_protection_level': target_level,
                    'content': migrated_content,
                }
                cur.execute(
                    "UPDATE memories SET data = data || %s::jsonb WHERE uid = %s AND id = %s",
                    (json.dumps(update_payload, default=_json_default), uid, mem_id),
                )


def migrate_memories(prev_uid: str, new_uid: str, app_id: str = None):
    """
    Migrate memories from one user to another.
    If app_id is provided, only migrate memories related to that app.
    """
    logger.info(f'Migrating memories from {prev_uid} to {new_uid}')

    parts = [f"SELECT {_MEMORY_COLS} FROM memories WHERE uid = %s"]
    params: List[Any] = [prev_uid]
    if app_id:
        parts.append("AND data->>'app_id' = %s")
        params.append(app_id)

    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(parts), tuple(params))
            src_rows = cur.fetchall()

            if not src_rows:
                logger.info(f'No memories to migrate for user {prev_uid}')
                return 0

            for row in src_rows:
                memory = _row_to_memory(row)
                typed = _extract_typed_cols(memory)
                data_body = _data_without_typed_cols(memory)
                created_at = typed['created_at'] or datetime.now(timezone.utc)
                updated_at = typed['updated_at'] or datetime.now(timezone.utc)

                cur.execute(
                    """
                    INSERT INTO memories
                        (uid, id, category, scoring, is_locked, created_at, updated_at, data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (uid, id) DO UPDATE
                        SET category = EXCLUDED.category,
                            scoring = EXCLUDED.scoring,
                            is_locked = EXCLUDED.is_locked,
                            updated_at = EXCLUDED.updated_at,
                            data = EXCLUDED.data
                    """,
                    (
                        new_uid,
                        memory['id'],
                        typed['category'],
                        typed['scoring'],
                        typed['is_locked'],
                        created_at,
                        updated_at,
                        json.dumps(data_body, default=_json_default),
                    ),
                )

    logger.info(f'Migrated {len(src_rows)} memories from {prev_uid} to {new_uid}')
    return len(src_rows)
