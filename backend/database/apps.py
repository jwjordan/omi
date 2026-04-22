"""Apps registry + testers + per-app usage history + per-app reviews.

Postgres-backed port of the Firestore implementation. Four tables:

- apps              (id PK, data JSONB)
- testers           (uid PK, data JSONB -- data['apps'] is the access list)
- app_usage_history (id PK, app_id, uid, usage_type, data JSONB)
- app_reviews       ((app_id, uid) PK, data JSONB)

All structured fields (approved, private, uid, category, capabilities,
username, twitter.username, is_popular, status) live inside `data`. We
filter them with jsonb accessors/containment.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ulid import ULID

from models.app import UsageHistoryType
from ._client import db

logger = logging.getLogger(__name__)

# *****************************
# ********** CRUD *************
# *****************************


def _row_to_app(row) -> Dict[str, Any]:
    """(id, data) tuple -> flattened dict with 'id' merged in."""
    app_id, data = row
    result = dict(data or {})
    result["id"] = app_id
    return result


def get_app_by_id_db(app_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM apps WHERE id = %s LIMIT 1",
                (app_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_app(row)


def get_audio_apps_count(app_ids: List[str]):
    if not app_ids or len(app_ids) == 0:
        return 0
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM apps
                WHERE id = ANY(%s::text[])
                  AND data #>> '{external_integration,triggers_on}' = %s
                """,
                (list(app_ids), 'audio_bytes'),
            )
            row = cur.fetchone()
            return row[0] if row else 0


def get_private_apps_db(uid: str) -> List:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'uid' = %s
                  AND data @> '{"private": true}'::jsonb
                """,
                (uid,),
            )
            return [_row_to_app(r) for r in cur.fetchall()]


# This returns public unapproved apps of all users
def get_unapproved_public_apps_db() -> List:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data @> '{"approved": false}'::jsonb
                  AND data @> '{"private": false}'::jsonb
                """
            )
            return [_row_to_app(r) for r in cur.fetchall()]


def get_public_approved_apps_db() -> List:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data @> '{"approved": true}'::jsonb
                  AND data @> '{"private": false}'::jsonb
                """
            )
            return [_row_to_app(r) for r in cur.fetchall()]


def get_popular_apps_db() -> List:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data @> '{"approved": true}'::jsonb
                  AND data @> '{"is_popular": true}'::jsonb
                """
            )
            return [_row_to_app(r) for r in cur.fetchall()]


def set_app_popular_db(app_id: str, popular: bool):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE apps
                SET data = data || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps({'is_popular': popular}), app_id),
            )


def search_apps_db(
    uid: str,
    category: str | None = None,
    capability: str | None = None,
    my_apps: bool = False,
    installed_apps: bool = False,
    enabled_app_ids: List[str] | None = None,
) -> List:
    """Optimized search function applying filters at DB level.

    Rating filter is NOT applied here (rating_avg lives in Redis); apply it
    after fetching.
    """
    where_parts: List[str] = []
    params: List[Any] = []

    # 1. Most restrictive filter first.
    if my_apps:
        where_parts.append("data->>'uid' = %s")
        params.append(uid)

    elif installed_apps:
        if not enabled_app_ids or len(enabled_app_ids) == 0:
            return []

        # Unlike Firestore (30-item 'in' cap), Postgres handles ANY() well, so
        # query by specific IDs in all cases.
        where_parts.append("id = ANY(%s::text[])")
        params.append(list(enabled_app_ids))

    else:
        where_parts.append("data @> '{\"approved\": true}'::jsonb")
        where_parts.append("data @> '{\"private\": false}'::jsonb")

    # 2. Category filter (not when searching my_apps -- post-filter below).
    if category and not my_apps:
        where_parts.append("data->>'category' = %s")
        params.append(category)

    # 3. Capability filter: capabilities is an array in JSONB, use ? operator
    # which tests if the string is a top-level array element / object key.
    if capability and not my_apps:
        where_parts.append("data->'capabilities' ? %s")
        params.append(capability)

    where_clause = " AND ".join(where_parts) if where_parts else "TRUE"
    sql = f"SELECT id, data FROM apps WHERE {where_clause}"

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            apps = [_row_to_app(r) for r in cur.fetchall()]

    # Post-filter for category if my_apps.
    if my_apps and category:
        apps = [app for app in apps if app.get('category') == category]

    # Post-filter for capability if my_apps.
    if my_apps and capability:
        apps = [app for app in apps if capability in app.get('capabilities', [])]

    return apps


# This returns public unapproved apps for a user
def get_public_unapproved_apps_db(uid: str) -> List:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data @> '{"approved": false}'::jsonb
                  AND data @> '{"private": false}'::jsonb
                  AND data->>'uid' = %s
                """,
                (uid,),
            )
            return [_row_to_app(r) for r in cur.fetchall()]


def get_apps_for_tester_db(uid: str) -> List:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM testers WHERE uid = %s LIMIT 1",
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                return []
            apps = (row[0] or {}).get('apps', [])
            if not apps:
                return []
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE id = ANY(%s::text[])
                  AND data @> '{"approved": false}'::jsonb
                """,
                (list(apps),),
            )
            return [_row_to_app(r) for r in cur.fetchall()]


def add_app_to_db(app_data: dict):
    app_id = app_data.get('id') or str(ULID())
    app_data = {**app_data, 'id': app_id}
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO apps (id, data)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                """,
                (app_id, json.dumps(app_data)),
            )


def upsert_app_to_db(app_data: dict):
    app_id = app_data['id']
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO apps (id, data)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                """,
                (app_id, json.dumps(app_data)),
            )


def update_app_in_db(app_data: dict):
    """Shallow-merge app_data into the existing row's data JSONB."""
    app_id = app_data['id']
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE apps
                SET data = data || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps(app_data), app_id),
            )


def delete_app_from_db(app_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM apps WHERE id = %s", (app_id,))


def update_app_visibility_in_db(app_id: str, private: bool):
    # Special case: flipping a private app back to public gets a new id.
    if 'private' in app_id and not private:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM apps WHERE id = %s LIMIT 1", (app_id,))
                row = cur.fetchone()
                if row is None:
                    return
                app = dict(row[0] or {})
                new_app_id = app_id.split('-private')[0] + '-' + str(ULID())
                app['id'] = new_app_id
                app['private'] = private
                cur.execute("DELETE FROM apps WHERE id = %s", (app_id,))
                cur.execute(
                    """
                    INSERT INTO apps (id, data)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                    """,
                    (new_app_id, json.dumps(app)),
                )
    else:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE apps
                    SET data = data || %s::jsonb
                    WHERE id = %s
                    """,
                    (json.dumps({'private': private}), app_id),
                )


def change_app_approval_status(app_id: str, approved: bool):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE apps
                SET data = data || %s::jsonb
                WHERE id = %s
                """,
                (
                    json.dumps(
                        {'approved': approved, 'status': 'approved' if approved else 'rejected'}
                    ),
                    app_id,
                ),
            )


def get_app_usage_history_db(app_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM app_usage_history
                WHERE app_id = %s
                ORDER BY created_at DESC
                """,
                (app_id,),
            )
            items = []
            for row_id, data in cur.fetchall():
                item = dict(data or {})
                item.setdefault('id', row_id)
                items.append(item)
            return items


def _usage_count_by_type(app_id: str, usage_type) -> int:
    usage_val = usage_type.value if hasattr(usage_type, 'value') else usage_type
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM app_usage_history
                WHERE app_id = %s AND usage_type = %s
                """,
                (app_id, usage_val),
            )
            row = cur.fetchone()
            return row[0] if row else 0


def get_app_memory_created_integration_usage_count_db(app_id: str):
    return _usage_count_by_type(app_id, UsageHistoryType.memory_created_external_integration)


def get_app_memory_prompt_usage_count_db(app_id: str):
    return _usage_count_by_type(app_id, UsageHistoryType.memory_created_prompt)


def get_app_chat_message_sent_usage_count_db(app_id: str):
    return _usage_count_by_type(app_id, UsageHistoryType.chat_message_sent)


def get_app_usage_count_db(app_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM app_usage_history WHERE app_id = %s",
                (app_id,),
            )
            row = cur.fetchone()
            return row[0] if row else 0


# ********************************
# *********** REVIEWS ************
# ********************************


def set_app_review_in_db(app_id: str, uid: str, review: dict):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_reviews (app_id, uid, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (app_id, uid) DO UPDATE SET data = EXCLUDED.data
                """,
                (app_id, uid, json.dumps(review)),
            )


# ********************************
# ************ TESTER ************
# ********************************


def add_tester_db(data: dict):
    uid = data['uid']
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO testers (uid, data)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (uid) DO UPDATE SET data = EXCLUDED.data
                """,
                (uid, json.dumps(data)),
            )


def add_app_access_for_tester_db(app_id: str, uid: str):
    """Append app_id to testers.data['apps'] (array), avoiding duplicates."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Emulate ArrayUnion: remove existing occurrence then append.
            cur.execute(
                """
                UPDATE testers
                SET data = jsonb_set(
                    data,
                    '{apps}',
                    COALESCE(
                        (
                            SELECT jsonb_agg(DISTINCT e)
                            FROM jsonb_array_elements(
                                COALESCE(data->'apps', '[]'::jsonb) || to_jsonb(%s::text)
                            ) AS e
                        ),
                        '[]'::jsonb
                    ),
                    true
                )
                WHERE uid = %s
                """,
                (app_id, uid),
            )


def remove_app_access_for_tester_db(app_id: str, uid: str):
    """Remove app_id from testers.data['apps'] (array)."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE testers
                SET data = jsonb_set(
                    data,
                    '{apps}',
                    COALESCE(
                        (
                            SELECT jsonb_agg(e)
                            FROM jsonb_array_elements(COALESCE(data->'apps', '[]'::jsonb)) AS e
                            WHERE e <> to_jsonb(%s::text)
                        ),
                        '[]'::jsonb
                    ),
                    true
                )
                WHERE uid = %s
                """,
                (app_id, uid),
            )


def remove_tester_db(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM testers WHERE uid = %s", (uid,))


def can_tester_access_app_db(app_id: str, uid: str) -> bool:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM testers WHERE uid = %s LIMIT 1",
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            apps = (row[0] or {}).get('apps', []) or []
            return app_id in apps


def is_tester_db(uid: str) -> bool:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM testers WHERE uid = %s LIMIT 1",
                (uid,),
            )
            return cur.fetchone() is not None


# ********************************
# *********** APPS USAGE *********
# ********************************


def record_app_usage(
    uid: str,
    app_id: str,
    usage_type: UsageHistoryType,
    conversation_id: str = None,
    message_id: str = None,
    timestamp: datetime = None,
):
    if not conversation_id and not message_id:
        raise ValueError('memory_id or message_id must be provided')

    ts = datetime.now(timezone.utc) if timestamp is None else timestamp
    usage_val = usage_type.value if hasattr(usage_type, 'value') else usage_type

    data = {
        'uid': uid,
        'memory_id': conversation_id,
        'message_id': message_id,
        'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else ts,
        'type': usage_val,
    }

    row_id = conversation_id or message_id
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_usage_history (id, app_id, uid, usage_type, data)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                    SET app_id = EXCLUDED.app_id,
                        uid = EXCLUDED.uid,
                        usage_type = EXCLUDED.usage_type,
                        data = EXCLUDED.data
                """,
                (row_id, app_id, uid, usage_val, json.dumps(data)),
            )

    # Preserve original return contract (callers may inspect the dict).
    data['timestamp'] = ts
    data['type'] = usage_type
    return data


# ********************************
# *********** PERSONAS ***********
# ********************************


def delete_persona_db(persona_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM apps WHERE id = %s", (persona_id,))


def get_personas_by_username_db(persona_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'username' = %s
                """,
                (persona_id,),
            )
            rows = cur.fetchall()
            if not rows:
                return None
            result = []
            for row_id, data in rows:
                d = dict(data or {})
                d['doc_id'] = row_id
                result.append(d)
            return result


def get_persona_by_username_db(username: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'username' = %s
                  AND data->'capabilities' ? 'persona'
                LIMIT 1
                """,
                (username,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_app(row)


def get_persona_by_id_db(persona_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM apps WHERE id = %s LIMIT 1",
                (persona_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_app(row)


def get_persona_by_uid_db(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'uid' = %s
                  AND data->'capabilities' ? 'persona'
                LIMIT 1
                """,
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_app(row)


def get_user_persona_by_uid(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'uid' = %s
                  AND data->>'category' = %s
                  AND data->'capabilities' ? 'persona'
                LIMIT 1
                """,
                (uid, 'personality-emulation'),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, data = row
            return {'id': row_id, **(data or {})}


def get_persona_by_twitter_handle_db(handle: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'category' = %s
                  AND data #>> '{twitter,username}' = %s
                LIMIT 1
                """,
                ('personality-emulation', handle),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, data = row
            return {'id': row_id, **(data or {})}


def get_persona_by_username_twitter_handle_db(username: str, handle: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'username' = %s
                  AND data->>'category' = %s
                  AND data #>> '{twitter,username}' = %s
                LIMIT 1
                """,
                (username, 'personality-emulation', handle),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, data = row
            return {'id': row_id, **(data or {})}


def get_omi_personas_by_uid_db(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'uid' = %s
                  AND data->'capabilities' ? 'persona'
                  AND data->'connected_accounts' ? 'omi'
                """,
                (uid,),
            )
            rows = cur.fetchall()
            if not rows:
                return []
            return [_row_to_app(r) for r in rows]


def get_omi_persona_apps_by_uid_db(uid: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM apps
                WHERE data->>'uid' = %s
                  AND data->>'category' = %s
                """,
                (uid, 'personality-emulation'),
            )
            rows = cur.fetchall()
            if not rows:
                return []
            return [_row_to_app(r) for r in rows]


def update_persona_in_db(persona_data: dict):
    persona_id = persona_data['id']
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE apps
                SET data = data || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps(persona_data), persona_id),
            )


def migrate_app_owner_id_db(new_id: str, old_id: str):
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE apps
                SET data = data || %s::jsonb
                WHERE data->>'uid' = %s
                """,
                (json.dumps({'uid': new_id}), old_id),
            )


# ********************************
# *********** API KEYS ***********
# ********************************
#
# API keys live inside the owning app's JSONB under data['api_keys'] as a
# dict keyed by api_key id. This keeps the existing 4-table schema and avoids
# adding a subtable; callers only ever scope queries by (app_id, key_id) or
# (app_id, hashed).


def create_api_key_db(app_id: str, api_key_data: dict):
    """Create a new API key for an app."""
    key_id = api_key_data['id']
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE apps
                SET data = jsonb_set(
                    data,
                    ARRAY['api_keys', %s],
                    %s::jsonb,
                    true
                )
                WHERE id = %s
                """,
                (key_id, json.dumps(api_key_data), app_id),
            )
    return api_key_data


def get_api_key_by_hash_db(app_id: str, hashed_key: str):
    """Get an API key for app_id by its hash value."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT value
                FROM apps,
                     LATERAL jsonb_each(COALESCE(data->'api_keys', '{}'::jsonb))
                WHERE id = %s AND value->>'hashed' = %s
                LIMIT 1
                """,
                (app_id, hashed_key),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]


def list_api_keys_db(app_id: str):
    """List all API keys for an app, excluding the `hashed` values."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data->'api_keys' FROM apps WHERE id = %s LIMIT 1",
                (app_id,),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return []
            api_keys = row[0]
            items = list(api_keys.values())
            items.sort(key=lambda k: k.get('created_at') or '', reverse=True)
            return [{k: v for k, v in item.items() if k != 'hashed'} for item in items]


def delete_api_key_db(app_id: str, key_id: str):
    """Delete an API key from an app."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE apps
                SET data = jsonb_set(
                    data,
                    '{api_keys}',
                    COALESCE(data->'api_keys', '{}'::jsonb) - %s,
                    true
                )
                WHERE id = %s
                """,
                (key_id, app_id),
            )
    return True
