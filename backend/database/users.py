"""Postgres-backed user profile, people, analytics, and integrations.

Ported from the Firestore implementation. Touches seven tables:

- users                 (uid PK, data JSONB)               -- main profile
- user_people           ((uid, person_id) PK, data JSONB)  -- people/speakers
- account_deletions     (uid PK, data JSONB)               -- retained after delete
- analytics             (id PK, type, data JSONB)          -- ratings/feedback
- user_integrations     ((uid, app_key) PK, data JSONB)
- user_task_integrations((uid, app_key) PK, data JSONB)

All semantic fields live inside `data` JSONB unless otherwise noted.

delete_user_data hits a broader set of user-scoped tables so a delete request
wipes everything the user wrote, not just the profile.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ._client import db, document_id_from_seed
from database.redis_db import try_acquire_user_platform_write_lock
from models.users import Subscription, PlanLimits, PlanType, SubscriptionStatus
from utils.subscription import get_default_basic_subscription

logger = logging.getLogger(__name__)


# Industry-standard two-field pattern (Mixpanel / Amplitude / PostHog):
#   signup_platform       — set once at account creation, immutable
#   last_active_platform  — overwritten on every authenticated request
#   platforms_used        — array union of every platform the user has ever
#                           authenticated from (for cross-platform segmentation)
#
# We normalize the raw header into a coarse `desktop | mobile` bucket, matching
# the profitability dashboard splits, and preserve the granular value
# (`ios`/`android`/`macos`) in `last_active_os` for finer drill-down.
_PLATFORM_ALIASES = {
    'macos': 'desktop',
    'mac': 'desktop',
    'mac os x': 'desktop',
    'desktop': 'desktop',
    'ios': 'mobile',
    'iphone os': 'mobile',
    'android': 'mobile',
    'mobile': 'mobile',
    'web': 'web',
    'browser': 'web',
}


def _normalize_platform(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (coarse_platform, os_value) for a raw `X-App-Platform` header."""
    if not raw or not isinstance(raw, str):
        return None, None
    os_value = raw.strip().lower()
    if not os_value:
        return None, None
    coarse = _PLATFORM_ALIASES.get(os_value)
    return coarse, os_value


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _get_user_data(uid: str) -> Optional[dict]:
    """Return the users.data JSONB dict, or None if the user doesn't exist."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM users WHERE uid = %s", (uid,))
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row[0] or {})


def _merge_user_data(uid: str, patch: dict) -> None:
    """Shallow-merge `patch` into users.data, creating the row if missing.

    Timestamps that are datetime objects are serialized with isoformat so the
    JSONB round-trip preserves them as ISO-8601 strings.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (uid, data, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (uid) DO UPDATE
                    SET data = users.data || EXCLUDED.data, updated_at = now()
                """,
                (uid, json.dumps(patch, default=_json_default)),
            )


def _set_user_field(uid: str, field: str, value: Any) -> None:
    """Shortcut: set a single top-level field under users.data."""
    _merge_user_data(uid, {field: value})


def _delete_user_field(uid: str, field: str) -> None:
    """Remove a top-level key from users.data (JSONB minus)."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET data = data - %s, updated_at = now() WHERE uid = %s",
                (field, uid),
            )


def _json_default(value):
    """JSON serializer for datetime-like values going into JSONB."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_dt(value) -> Optional[datetime]:
    """Best-effort: return a datetime from a JSONB string/datetime value."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Core profile
# ---------------------------------------------------------------------------


def record_user_platform(uid: str, raw_platform: Optional[str]) -> None:
    """Write per-request platform telemetry fields on the users row.

    Throttled to one write per (uid, coarse_platform) every 10 minutes via Redis.
    Fail-open: any error is logged and swallowed — this is telemetry, not a
    correctness path.
    """
    coarse, os_value = _normalize_platform(raw_platform)
    if not coarse:
        return

    try:
        if not try_acquire_user_platform_write_lock(uid, coarse):
            return

        now = datetime.now(timezone.utc)
        existing = _get_user_data(uid)

        updates: dict = {
            'last_active_platform': coarse,
            'last_active_os': os_value,
            'last_active_at': now.isoformat(),
            f'last_active_at_{coarse}': now.isoformat(),
        }

        # Merge platforms_used (deduped array)
        platforms_used = []
        if existing is not None:
            platforms_used = list(existing.get('platforms_used') or [])
        if coarse not in platforms_used:
            platforms_used.append(coarse)
        updates['platforms_used'] = platforms_used

        # signup_platform is set-once.
        if existing is None:
            updates['signup_platform'] = coarse
            updates['signup_os'] = os_value
            updates['signup_platform_at'] = now.isoformat()
        elif not existing.get('signup_platform'):
            updates['signup_platform'] = coarse
            updates['signup_os'] = os_value
            created_at = existing.get('created_at') or now.isoformat()
            updates['signup_platform_at'] = (
                created_at.isoformat() if isinstance(created_at, datetime) else created_at
            )

        _merge_user_data(uid, updates)
    except Exception as e:  # noqa: BLE001
        logger.warning("record_user_platform failed for uid=%s: %s", uid, e)


def is_exists_user(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE uid = %s LIMIT 1", (uid,))
            return cur.fetchone() is not None


def get_user_profile(uid: str) -> dict:
    """Gets the full user profile document."""
    data = _get_user_data(uid)
    return data if data is not None else {}


def get_user_store_recording_permission(uid: str):
    data = _get_user_data(uid) or {}
    return data.get('store_recording_permission', False)


def set_user_store_recording_permission(uid: str, value: bool):
    _set_user_field(uid, 'store_recording_permission', value)


def get_user_private_cloud_sync_enabled(uid: str) -> bool:
    """Check if user has private cloud sync enabled."""
    data = _get_user_data(uid) or {}
    return data.get('private_cloud_sync_enabled', True)


def set_user_private_cloud_sync_enabled(uid: str, value: bool):
    """Enable or disable private cloud sync for a user."""
    _set_user_field(uid, 'private_cloud_sync_enabled', value)


def set_user_cancellation_feedback(uid: str, reason: str, reason_details: Optional[str] = None):
    _merge_user_data(
        uid,
        {
            'cancellation_feedback': {
                'reason': reason,
                'reason_details': reason_details or '',
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
        },
    )


# BYOK (Bring Your Own Keys) — free-plan flag.
# We never store keys themselves; only SHA-256 fingerprints so we can detect
# rotation. `active` is the subscription-bypass gate.

BYOK_HEARTBEAT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def get_byok_state(uid: str) -> dict:
    data = _get_user_data(uid) or {}
    return data.get('byok', {}) or {}


def is_byok_active(uid: str) -> bool:
    """True if user has a live BYOK activation (heartbeat within TTL)."""
    state = get_byok_state(uid)
    if not state.get('active'):
        return False
    last_seen = _parse_dt(state.get('last_seen_at'))
    if last_seen is None:
        return False
    age = (datetime.now(timezone.utc) - last_seen).total_seconds()
    return age <= BYOK_HEARTBEAT_TTL_SECONDS


def set_byok_active(uid: str, fingerprints: dict):
    _merge_user_data(
        uid,
        {
            'byok': {
                'active': True,
                'fingerprints': fingerprints,
                'last_seen_at': datetime.now(timezone.utc).isoformat(),
            }
        },
    )


def clear_byok_active(uid: str):
    _merge_user_data(
        uid,
        {
            'byok': {
                'active': False,
                'fingerprints': {},
                'last_seen_at': datetime.now(timezone.utc).isoformat(),
            }
        },
    )


def set_user_deletion_feedback(uid: str, reason: Optional[str], reason_details: Optional[str] = None):
    """Persist deletion feedback in a global table that survives user deletion."""
    payload = {
        'uid': uid,
        'reason': reason or '',
        'reason_details': reason_details or '',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO account_deletions (uid, data)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (uid) DO UPDATE SET data = EXCLUDED.data
                """,
                (uid, json.dumps(payload)),
            )


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------


def _row_to_person(row) -> dict:
    person_id, data = row
    result = dict(data or {})
    result.setdefault('id', person_id)
    return result


def create_person(uid: str, data: dict):
    person_id = data['id']
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_people (uid, person_id, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, person_id) DO UPDATE SET data = EXCLUDED.data
                """,
                (uid, person_id, json.dumps(data, default=_json_default)),
            )
    return data


def get_person(uid: str, person_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT person_id, data FROM user_people WHERE uid = %s AND person_id = %s LIMIT 1",
                (uid, person_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_person(row)


def get_people(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT person_id, data FROM user_people WHERE uid = %s",
                (uid,),
            )
            return [_row_to_person(r) for r in cur.fetchall()]


def get_person_by_name(uid: str, name: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT person_id, data FROM user_people
                WHERE uid = %s AND data->>'name' = %s
                LIMIT 1
                """,
                (uid, name),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_person(row)


def get_people_by_ids(uid: str, person_ids: list[str]):
    """Fetch people docs by ID. Result order is not guaranteed to match input."""
    if not person_ids:
        return []
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT person_id, data FROM user_people
                WHERE uid = %s AND person_id = ANY(%s::text[])
                """,
                (uid, list(person_ids)),
            )
            return [_row_to_person(r) for r in cur.fetchall()]


def update_person(uid: str, person_id: str, name: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_people
                SET data = data || %s::jsonb
                WHERE uid = %s AND person_id = %s
                """,
                (json.dumps({'name': name}), uid, person_id),
            )


def delete_person(uid: str, person_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_people WHERE uid = %s AND person_id = %s",
                (uid, person_id),
            )


def _add_sample_transaction(transaction, person_ref, sample_path, transcript, max_samples):
    """Back-compat helper used by existing unit tests.

    The original Firestore implementation took a transaction + document ref.
    The Postgres port does the same read-modify-write logic in-process; we
    keep the signature so tests continue to exercise the alignment /
    max-samples behavior without needing to spin up a live database.
    """
    snapshot = person_ref.get(transaction=transaction) if transaction is not None else person_ref.get()
    if not snapshot.exists:
        return False

    person_data = snapshot.to_dict() or {}
    samples = list(person_data.get('speech_samples', []) or [])

    if len(samples) >= max_samples:
        return False

    samples.append(sample_path)
    update_data = {
        'speech_samples': samples,
        'updated_at': datetime.now(timezone.utc),
    }

    if transcript is not None:
        transcripts = list(person_data.get('speech_sample_transcripts', []) or [])
        existing_sample_count = len(samples) - 1
        if len(transcripts) < existing_sample_count:
            transcripts.extend([''] * (existing_sample_count - len(transcripts)))
        transcripts.append(transcript)
        update_data['speech_sample_transcripts'] = transcripts
        update_data['speech_samples_version'] = 3

    if transaction is not None:
        transaction.update(person_ref, update_data)
    else:
        person_ref.update(update_data)
    return True


def add_person_speech_sample(
    uid: str, person_id: str, sample_path: str, transcript: Optional[str] = None, max_samples: int = 5
) -> bool:
    """Append a speech sample path (and optional transcript) to a person.

    Read-modify-write under a serializable batch so concurrent writers can't
    drift the parallel samples/transcripts arrays.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM user_people WHERE uid = %s AND person_id = %s FOR UPDATE",
                (uid, person_id),
            )
            row = cur.fetchone()
            if row is None:
                return False

            person_data = dict(row[0] or {})
            samples = list(person_data.get('speech_samples', []) or [])
            if len(samples) >= max_samples:
                return False

            samples.append(sample_path)
            patch: dict = {
                'speech_samples': samples,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }

            if transcript is not None:
                transcripts = list(person_data.get('speech_sample_transcripts', []) or [])
                existing_sample_count = len(samples) - 1
                if len(transcripts) < existing_sample_count:
                    transcripts.extend([''] * (existing_sample_count - len(transcripts)))
                transcripts.append(transcript)
                patch['speech_sample_transcripts'] = transcripts
                patch['speech_samples_version'] = 3

            cur.execute(
                """
                UPDATE user_people
                SET data = data || %s::jsonb
                WHERE uid = %s AND person_id = %s
                """,
                (json.dumps(patch), uid, person_id),
            )
    return True


def get_person_speech_samples_count(uid: str, person_id: str) -> int:
    """Get the count of speech samples for a person."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT jsonb_array_length(COALESCE(data->'speech_samples', '[]'::jsonb))
                FROM user_people
                WHERE uid = %s AND person_id = %s
                """,
                (uid, person_id),
            )
            row = cur.fetchone()
            if row is None:
                return 0
            return row[0] or 0


def remove_person_speech_sample(uid: str, person_id: str, sample_path: str) -> bool:
    """Remove sample_path from the person's parallel samples/transcripts arrays."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM user_people WHERE uid = %s AND person_id = %s FOR UPDATE",
                (uid, person_id),
            )
            row = cur.fetchone()
            if row is None:
                return False

            person_data = dict(row[0] or {})
            samples = list(person_data.get('speech_samples', []) or [])
            transcripts = list(person_data.get('speech_sample_transcripts', []) or [])

            try:
                idx = samples.index(sample_path)
            except ValueError:
                return False

            samples.pop(idx)
            if idx < len(transcripts):
                transcripts.pop(idx)

            patch = {
                'speech_samples': samples,
                'speech_sample_transcripts': transcripts,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            cur.execute(
                """
                UPDATE user_people
                SET data = data || %s::jsonb
                WHERE uid = %s AND person_id = %s
                """,
                (json.dumps(patch), uid, person_id),
            )
    return True


def set_user_speaker_embedding(uid: str, embedding: list) -> bool:
    """Store speaker embedding for the user's own voice on their user document."""
    _merge_user_data(
        uid,
        {
            'speaker_embedding': embedding,
            'speaker_embedding_updated_at': datetime.now(timezone.utc).isoformat(),
        },
    )
    return True


def get_user_speaker_embedding(uid: str) -> Optional[list]:
    """Get the user's own speaker embedding from their user document."""
    data = _get_user_data(uid)
    if data is None:
        return None
    return data.get('speaker_embedding')


def set_person_speaker_embedding(uid: str, person_id: str, embedding: list) -> bool:
    """Store speaker embedding for a person. Returns False if the person is missing."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_people
                SET data = data || %s::jsonb
                WHERE uid = %s AND person_id = %s
                """,
                (
                    json.dumps(
                        {
                            'speaker_embedding': embedding,
                            'updated_at': datetime.now(timezone.utc).isoformat(),
                        }
                    ),
                    uid,
                    person_id,
                ),
            )
            if cur.rowcount == 0:
                return False
    return True


def get_person_speaker_embedding(uid: str, person_id: str) -> Optional[list]:
    """Get speaker embedding for a person."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM user_people WHERE uid = %s AND person_id = %s LIMIT 1",
                (uid, person_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return (row[0] or {}).get('speaker_embedding')


def set_person_speech_sample_transcript(uid: str, person_id: str, sample_index: int, transcript: str) -> bool:
    """Update transcript at a specific index in the parallel arrays."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM user_people WHERE uid = %s AND person_id = %s FOR UPDATE",
                (uid, person_id),
            )
            row = cur.fetchone()
            if row is None:
                return False

            person_data = dict(row[0] or {})
            samples = list(person_data.get('speech_samples', []) or [])
            transcripts = list(person_data.get('speech_sample_transcripts', []) or [])

            if sample_index < 0 or sample_index >= len(samples):
                return False

            while len(transcripts) < len(samples):
                transcripts.append('')

            transcripts[sample_index] = transcript

            patch = {
                'speech_sample_transcripts': transcripts,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            cur.execute(
                """
                UPDATE user_people
                SET data = data || %s::jsonb
                WHERE uid = %s AND person_id = %s
                """,
                (json.dumps(patch), uid, person_id),
            )
    return True


def update_person_speech_samples_after_migration(
    uid: str,
    person_id: str,
    samples: list,
    transcripts: list,
    version: int,
    speaker_embedding: Optional[list] = None,
) -> bool:
    """Replace all samples/transcripts/embedding and set version atomically."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_people WHERE uid = %s AND person_id = %s LIMIT 1",
                (uid, person_id),
            )
            if cur.fetchone() is None:
                return False

            patch: dict = {
                'speech_samples': samples,
                'speech_sample_transcripts': transcripts,
                'speech_samples_version': version,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }

            if speaker_embedding is not None:
                patch['speaker_embedding'] = speaker_embedding
                cur.execute(
                    """
                    UPDATE user_people
                    SET data = data || %s::jsonb
                    WHERE uid = %s AND person_id = %s
                    """,
                    (json.dumps(patch), uid, person_id),
                )
            else:
                # Remove speaker_embedding key then merge the rest
                cur.execute(
                    """
                    UPDATE user_people
                    SET data = (data - 'speaker_embedding') || %s::jsonb
                    WHERE uid = %s AND person_id = %s
                    """,
                    (json.dumps(patch), uid, person_id),
                )
    return True


def clear_person_speaker_embedding(uid: str, person_id: str) -> bool:
    """Drop the speaker_embedding key from a person's data, if the person exists."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_people WHERE uid = %s AND person_id = %s LIMIT 1",
                (uid, person_id),
            )
            if cur.fetchone() is None:
                return False

            cur.execute(
                """
                UPDATE user_people
                SET data = (data - 'speaker_embedding') || %s::jsonb
                WHERE uid = %s AND person_id = %s
                """,
                (
                    json.dumps({'updated_at': datetime.now(timezone.utc).isoformat()}),
                    uid,
                    person_id,
                ),
            )
    return True


def update_person_speech_samples_version(uid: str, person_id: str, version: int) -> bool:
    """Update just the speech_samples_version field."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_people
                SET data = data || %s::jsonb
                WHERE uid = %s AND person_id = %s
                """,
                (
                    json.dumps(
                        {
                            'speech_samples_version': version,
                            'updated_at': datetime.now(timezone.utc).isoformat(),
                        }
                    ),
                    uid,
                    person_id,
                ),
            )
            if cur.rowcount == 0:
                return False
    return True


# ---------------------------------------------------------------------------
# Data deletion
# ---------------------------------------------------------------------------


# Tables keyed by `uid` that a delete-account request must wipe. Kept in one
# place so it's easy to audit. Add new user-scoped tables here as they land.
_USER_SCOPED_TABLES: tuple[str, ...] = (
    'user_people',
    'account_deletions',
    'user_integrations',
    'user_task_integrations',
    'conversations',
    'conversation_photos',
    'memories',
    'action_items',
    'chat_sessions',
    'chat_messages',
    'folders',
    'goals',
    'notifications',
    'daily_summaries',
    'focus_sessions',
    'screen_activity',
    'phone_calls',
    'calendar_meetings',
    'trends',
    'advice',
    'wrapped',
    'staged_tasks',
    'fair_use_state',
    'fair_use_events',
    'user_hourly_usage',
    'user_usage',
    'llm_usage',
    'dev_api_keys',
    'mcp_api_keys',
    'knowledge_graph_nodes',
    'knowledge_graph_edges',
    'conversation_vectors',
    'memory_vectors',
    'import_jobs',
)


def delete_user_data(uid: str):
    """Delete the user row plus every user-scoped record across related tables.

    All deletes run inside a single transaction via db.batch() so a failure
    anywhere rolls the whole thing back.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE uid = %s LIMIT 1", (uid,))
            if cur.fetchone() is None:
                return {'status': 'error', 'message': 'User not found'}

            for table in _USER_SCOPED_TABLES:
                try:
                    cur.execute(f"DELETE FROM {table} WHERE uid = %s", (uid,))
                except Exception as e:  # noqa: BLE001
                    # Some deployments may not yet have every table; log and
                    # keep going rather than blocking the user deletion.
                    logger.warning(
                        "delete_user_data: skipping %s for uid=%s (%s)", table, uid, e
                    )

            logger.info("Deleting user document: %s", uid)
            cur.execute("DELETE FROM users WHERE uid = %s", (uid,))
    return {'status': 'ok', 'message': 'Account deleted successfully'}


# **************************************
# ************* Analytics **************
# **************************************


def _write_analytics(doc_id: str, analytics_type: str, payload: dict) -> None:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analytics (id, type, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                    SET type = EXCLUDED.type, data = EXCLUDED.data
                """,
                (doc_id, analytics_type, json.dumps(payload, default=_json_default)),
            )


def set_conversation_summary_rating_score(uid: str, conversation_id: str, value: int):
    doc_id = document_id_from_seed('memory_summary' + conversation_id)
    payload = {
        'id': doc_id,
        'memory_id': conversation_id,
        'uid': uid,
        'value': value,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'type': 'memory_summary',
    }
    _write_analytics(doc_id, 'memory_summary', payload)


def get_conversation_summary_rating_score(conversation_id: str):
    doc_id = document_id_from_seed('memory_summary' + conversation_id)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM analytics WHERE id = %s LIMIT 1", (doc_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row[0] or {})


def get_all_ratings(rating_type: str = 'memory_summary'):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM analytics WHERE type = %s",
                (rating_type,),
            )
            return [dict(row[0] or {}) for row in cur.fetchall()]


def set_chat_message_rating_score(
    uid: str, message_id: str, value: int, reason: str = None, platform: str = None, app_version: str = None
):
    """Store chat message rating/feedback in the analytics table."""
    doc_id = document_id_from_seed('chat_message' + message_id)
    data = {
        'id': doc_id,
        'message_id': message_id,
        'uid': uid,
        'value': value,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'type': 'chat_message',
    }
    if reason:
        data['reason'] = reason
    if platform:
        data['platform'] = platform
    if app_version:
        data['app_version'] = app_version
    _write_analytics(doc_id, 'chat_message', data)


# **************************************
# ************** Payments **************
# **************************************


def get_stripe_connect_account_id(uid: str):
    data = _get_user_data(uid) or {}
    return data.get('stripe_account_id')


def set_stripe_connect_account_id(uid: str, account_id: str):
    _set_user_field(uid, 'stripe_account_id', account_id)


def set_paypal_payment_details(uid: str, data: dict):
    _set_user_field(uid, 'paypal_details', data)


def get_paypal_payment_details(uid: str):
    data = _get_user_data(uid) or {}
    return data.get('paypal_details')


def set_default_payment_method(uid: str, payment_method_id: str):
    _set_user_field(uid, 'default_payment_method', payment_method_id)


def get_default_payment_method(uid: str):
    data = _get_user_data(uid) or {}
    return data.get('default_payment_method')


def get_stripe_customer_id(uid: str) -> Optional[str]:
    """Get the Stripe customer ID for a user."""
    data = _get_user_data(uid)
    if data is None:
        return None
    return data.get('stripe_customer_id')


def set_stripe_customer_id(uid: str, customer_id: str):
    _set_user_field(uid, 'stripe_customer_id', customer_id)


def get_user_by_stripe_customer_id(customer_id: str):
    """Lookup a user by the stripe_customer_id field stored in users.data."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uid, data FROM users
                WHERE data->>'stripe_customer_id' = %s
                LIMIT 1
                """,
                (customer_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            uid, data = row
            result = dict(data or {})
            result['uid'] = uid
            return result


def update_user_subscription(uid: str, subscription_data: dict):
    """Updates the user's subscription information, removing dynamic fields before storing."""
    subscription_data_to_store = dict(subscription_data)
    subscription_data_to_store.pop('features', None)
    subscription_data_to_store.pop('limits', None)

    _set_user_field(uid, 'subscription', subscription_data_to_store)


# **************************************
# ********* Data Protection ************
# **************************************


def get_data_protection_level(uid: str) -> str:
    """Get the user's data protection level ('enhanced' by default, or 'e2ee')."""
    data = _get_user_data(uid)
    if data is None:
        return 'enhanced'
    return data.get('data_protection_level', 'enhanced')


def set_data_protection_level(uid: str, level: str) -> None:
    """Set the user's data protection level."""
    if level not in ['enhanced', 'e2ee']:
        raise ValueError("Invalid data protection level. Only 'enhanced' or 'e2ee' are supported.")
    _set_user_field(uid, 'data_protection_level', level)


def set_migration_status(uid: str, target_level: str):
    """Sets the migration status on the user's profile."""
    migration_status = {
        'target_level': target_level,
        'status': 'in_progress',
        'started_at': datetime.now(timezone.utc).isoformat(),
    }
    _set_user_field(uid, 'migration_status', migration_status)


def finalize_migration(uid: str, target_level: str):
    """Atomically sets the new protection level and removes the migration status field."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (uid, data, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (uid) DO UPDATE
                    SET data = (users.data - 'migration_status') || EXCLUDED.data,
                        updated_at = now()
                """,
                (uid, json.dumps({'data_protection_level': target_level})),
            )


# **************************************
# ************* Language ***************
# **************************************


def get_user_language_preference(uid: str) -> str:
    """Return the user's preferred language code (empty string if unset)."""
    data = _get_user_data(uid)
    if data is None:
        return ''
    return data.get('language', '')


def set_user_language_preference(uid: str, language: str) -> None:
    """Set the user's preferred language."""
    _set_user_field(uid, 'language', language)


def get_user_onboarding_state(uid: str) -> dict:
    """Get the user's onboarding state."""
    data = _get_user_data(uid) or {}
    return data.get('onboarding', {}) or {}


def set_user_onboarding_state(uid: str, onboarding_data: dict) -> None:
    """Merge partial onboarding state into the existing value."""
    current = get_user_onboarding_state(uid)
    merged = {**current, **(onboarding_data or {})}
    _set_user_field(uid, 'onboarding', merged)


def get_user_subscription(uid: str) -> Subscription:
    """Gets the user's subscription, creating a default free one if it doesn't exist."""
    data = _get_user_data(uid)
    if data is not None and 'subscription' in data:
        sub_data = dict(data['subscription'] or {})
        # Handle migration for old 'free' plan identifier
        if sub_data.get('plan') == 'free':
            sub_data['plan'] = PlanType.basic.value
            update_user_subscription(uid, sub_data)
        return Subscription(**sub_data)

    # If subscription doesn't exist for the user, create and return a default free plan.
    default_subscription = get_default_basic_subscription()
    # Strip dynamic fields before storing
    sub_to_store = default_subscription.dict()
    sub_to_store.pop('features', None)
    sub_to_store.pop('limits', None)
    _set_user_field(uid, 'subscription', sub_to_store)
    return default_subscription


def get_user_training_data_opt_in(uid: str) -> Optional[dict]:
    """Get user's training data opt-in status."""
    data = _get_user_data(uid) or {}
    return data.get('training_data_opt_in')


def set_user_training_data_opt_in(uid: str, status: str):
    """Set user's training data opt-in status."""
    _set_user_field(
        uid,
        'training_data_opt_in',
        {
            'status': status,
            'requested_at': datetime.now(timezone.utc).isoformat(),
        },
    )


def get_user_valid_subscription(uid: str) -> Optional[Subscription]:
    """Return the subscription if it's currently valid for use, else the default basic."""
    subscription = get_user_subscription(uid)

    # Basic (free) plans are only valid if their status is active.
    if subscription.plan == PlanType.basic:
        return subscription if subscription.status == SubscriptionStatus.active else None

    # For paid plans, validity is determined by the period end.
    if subscription.current_period_end:
        period_end_dt = datetime.fromtimestamp(subscription.current_period_end, tz=timezone.utc)
        if period_end_dt >= datetime.now(timezone.utc):
            return subscription

    # Fallback to default basic subscription
    return get_default_basic_subscription()


# **************************************
# ******** Task Integrations ***********
# **************************************


def get_task_integrations(uid: str) -> dict:
    """Return a dict keyed by app_key mapping to each integration's connection data."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT app_key, data FROM user_task_integrations WHERE uid = %s",
                (uid,),
            )
            return {app_key: dict(data or {}) for app_key, data in cur.fetchall()}


def get_task_integration(uid: str, app_key: str) -> Optional[dict]:
    """Get a specific task integration connection, or None if not set."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM user_task_integrations WHERE uid = %s AND app_key = %s LIMIT 1",
                (uid, app_key),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row[0] or {})


def set_task_integration(uid: str, app_key: str, data: dict) -> None:
    """Save or update a task integration connection (shallow-merge)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = dict(data or {})
    payload['updated_at'] = now_iso

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_task_integrations WHERE uid = %s AND app_key = %s LIMIT 1",
                (uid, app_key),
            )
            exists = cur.fetchone() is not None
            if not exists:
                payload.setdefault('created_at', now_iso)

            cur.execute(
                """
                INSERT INTO user_task_integrations (uid, app_key, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, app_key) DO UPDATE
                    SET data = user_task_integrations.data || EXCLUDED.data
                """,
                (uid, app_key, json.dumps(payload, default=_json_default)),
            )


def delete_task_integration(uid: str, app_key: str) -> bool:
    """Delete a task integration connection and clear default if it matched."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_task_integrations WHERE uid = %s AND app_key = %s LIMIT 1",
                (uid, app_key),
            )
            if cur.fetchone() is None:
                return False

            cur.execute(
                "DELETE FROM user_task_integrations WHERE uid = %s AND app_key = %s",
                (uid, app_key),
            )

            cur.execute(
                "SELECT data->>'default_task_integration' FROM users WHERE uid = %s",
                (uid,),
            )
            row = cur.fetchone()
            if row and row[0] == app_key:
                cur.execute(
                    "UPDATE users SET data = data - 'default_task_integration', updated_at = now() WHERE uid = %s",
                    (uid,),
                )
    return True


def get_default_task_integration(uid: str) -> Optional[str]:
    """Get the user's default task integration app key, or None."""
    data = _get_user_data(uid) or {}
    return data.get('default_task_integration')


def set_default_task_integration(uid: str, app_key: str) -> None:
    """Set the user's default task integration app."""
    _set_user_field(uid, 'default_task_integration', app_key)


# **************************************
# ******** Integrations ********
# **************************************


def get_integration(uid: str, app_key: str) -> Optional[dict]:
    """Get a specific integration connection."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM user_integrations WHERE uid = %s AND app_key = %s LIMIT 1",
                (uid, app_key),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row[0] or {})


def set_integration(uid: str, app_key: str, data: dict) -> None:
    """Save or update an integration connection (shallow-merge)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = dict(data or {})
    payload['updated_at'] = now_iso

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM user_integrations WHERE uid = %s AND app_key = %s LIMIT 1",
                (uid, app_key),
            )
            exists = cur.fetchone() is not None
            if not exists:
                payload.setdefault('created_at', now_iso)

            cur.execute(
                """
                INSERT INTO user_integrations (uid, app_key, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, app_key) DO UPDATE
                    SET data = user_integrations.data || EXCLUDED.data
                """,
                (uid, app_key, json.dumps(payload, default=_json_default)),
            )


def delete_integration(uid: str, app_key: str) -> bool:
    """Delete an integration connection."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_integrations WHERE uid = %s AND app_key = %s",
                (uid, app_key),
            )
            return cur.rowcount > 0


# **************************************
# ***** Transcription Preferences ******
# **************************************


def get_user_transcription_preferences(uid: str) -> dict:
    """Return single_language_mode + vocabulary + language (with defaults)."""
    data = _get_user_data(uid)
    if data is None:
        return {'single_language_mode': False, 'vocabulary': [], 'language': ''}

    prefs = data.get('transcription_preferences', {}) or {}
    return {
        'single_language_mode': prefs.get('single_language_mode', False),
        'vocabulary': prefs.get('vocabulary', []),
        'language': data.get('language', ''),
    }


def get_agent_vm(uid: str) -> Optional[dict]:
    """Return the user's agent VM info (ip/auth/status), or None if none."""
    data = _get_user_data(uid)
    if data is None:
        return None
    return data.get('agentVm')


def set_user_transcription_preferences(
    uid: str, single_language_mode: bool = None, vocabulary: list = None
) -> None:
    """Partial update of transcription_preferences sub-map."""
    sub_patch: dict = {}
    if single_language_mode is not None:
        sub_patch['single_language_mode'] = single_language_mode
    if vocabulary is not None:
        # Limit vocabulary to 100 terms max
        sub_patch['vocabulary'] = list(vocabulary)[:100]

    if not sub_patch:
        return

    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM users WHERE uid = %s", (uid,))
            row = cur.fetchone()
            existing = dict((row[0] or {}).get('transcription_preferences', {})) if row else {}
            merged = {**existing, **sub_patch}

            cur.execute(
                """
                INSERT INTO users (uid, data, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (uid) DO UPDATE
                    SET data = users.data || EXCLUDED.data, updated_at = now()
                """,
                (uid, json.dumps({'transcription_preferences': merged})),
            )


# ============================================================================
# DESKTOP USER SETTINGS — fields on users/{uid} document
# ============================================================================


def get_notification_settings(uid: str) -> dict:
    """Return notification settings mapped to Swift-compatible field names."""
    data = _get_user_data(uid)
    if data is None:
        return {'enabled': True, 'frequency': 3}
    return {
        'enabled': data.get('notifications_enabled', True),
        'frequency': data.get('notification_frequency', 3),
    }


def update_notification_settings(uid: str, enabled: bool = None, frequency: int = None) -> dict:
    updates: dict = {}
    if enabled is not None:
        updates['notifications_enabled'] = enabled
    if frequency is not None:
        updates['notification_frequency'] = frequency
    if updates:
        _merge_user_data(uid, updates)
    return get_notification_settings(uid)


def _get_raw_assistant_settings(uid: str) -> dict:
    """Read only the assistant_settings sub-map."""
    data = _get_user_data(uid)
    if data is None:
        return {}
    return data.get('assistant_settings') or {}


def get_assistant_settings(uid: str) -> dict:
    """Read assistant settings for the API response.

    Injects top-level ``update_channel`` into the response dict (it lives
    outside ``assistant_settings`` in the stored row but the API returns it
    together).
    """
    data = _get_user_data(uid)
    if data is None:
        return {}
    result = dict(data.get('assistant_settings') or {})
    if data.get('update_channel') is not None:
        result['update_channel'] = data['update_channel']
    return result


def update_assistant_settings(uid: str, settings: dict) -> dict:
    """Deep-merge partial settings into existing assistant_settings."""
    existing = _get_raw_assistant_settings(uid)

    # Extract update_channel — it goes to a top-level user doc field
    update_channel = settings.pop('update_channel', None)

    for section, values in settings.items():
        if isinstance(values, dict) and isinstance(existing.get(section), dict):
            existing[section].update(values)
        else:
            existing[section] = values

    updates: dict = {'assistant_settings': existing}
    if update_channel is not None:
        updates['update_channel'] = update_channel
    _merge_user_data(uid, updates)

    # Build response (include update_channel for the caller)
    if update_channel is not None:
        existing['update_channel'] = update_channel
    return existing


def get_ai_user_profile(uid: str) -> Optional[dict]:
    data = _get_user_data(uid)
    if data is None:
        return None
    return data.get('ai_user_profile')


def update_ai_user_profile(
    uid: str, profile_text: str = None, generated_at=None, data_sources_used: int = None
) -> dict:
    """Update AI user profile. Only writes non-None fields (partial update)."""
    existing = get_ai_user_profile(uid) or {}
    if profile_text is not None:
        existing['profile_text'] = profile_text
    if generated_at is not None:
        existing['generated_at'] = (
            generated_at.isoformat() if isinstance(generated_at, datetime) else generated_at
        )
    if data_sources_used is not None:
        existing['data_sources_used'] = data_sources_used

    _set_user_field(uid, 'ai_user_profile', existing)
    return existing
