import json
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import database.redis_db as redis_db
from database._client import db
from models.mcp_api_key import McpApiKey
from utils.mcp_api_keys import generate_api_key, hash_api_key


def create_mcp_key(user_id: str, name: str) -> Tuple[str, McpApiKey]:
    """
    Creates a new MCP API key for a user.
    Returns the raw key and the key's metadata.
    """
    raw_key, hashed_key, key_prefix = generate_api_key()
    key_id = str(uuid.uuid4())
    now = datetime.utcnow()

    data = {
        "id": key_id,
        "name": name,
        "key_prefix": key_prefix,
        "last_used_at": None,
    }

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mcp_api_keys (api_key_hash, uid, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (api_key_hash) DO UPDATE
                    SET uid = EXCLUDED.uid, data = EXCLUDED.data
                """,
                (hashed_key, user_id, json.dumps(data)),
            )

    api_key_data = McpApiKey(
        id=key_id,
        name=name,
        key_prefix=key_prefix,
        created_at=now,
        last_used_at=None,
    )
    return raw_key, api_key_data


def get_mcp_keys_for_user(user_id: str) -> List[McpApiKey]:
    """
    Retrieves all MCP API keys for a user.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data
                FROM mcp_api_keys
                WHERE uid = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    keys = []
    for (data,) in rows:
        keys.append(McpApiKey.model_validate(data))
    return keys


def delete_mcp_key(user_id: str, key_id: str):
    """
    Deletes an MCP API key by ID.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            # First, get the hashed_key for cache invalidation
            cur.execute(
                """
                SELECT api_key_hash FROM mcp_api_keys
                WHERE uid = %s AND data->>'id' = %s
                """,
                (user_id, key_id),
            )
            row = cur.fetchone()

            if row:
                hashed_key = row[0]
                # Delete the key
                cur.execute(
                    """
                    DELETE FROM mcp_api_keys
                    WHERE uid = %s AND data->>'id' = %s
                    """,
                    (user_id, key_id),
                )
                if hashed_key:
                    redis_db.delete_cached_mcp_api_key(hashed_key)


def get_user_id_by_api_key(api_key: str) -> Optional[str]:
    """
    Verifies an API key and returns the associated user ID.
    Uses a cache to avoid frequent database lookups.
    Also updates the last_used_at timestamp on cache miss.
    """
    if not api_key.startswith("omi_mcp_"):
        return None
    secret_part = api_key.replace("omi_mcp_", "", 1)
    hashed_key = hash_api_key(secret_part)

    # Check cache first
    user_id = redis_db.get_cached_mcp_api_key_user_id(hashed_key)
    if user_id:
        return user_id

    # If not in cache, query database
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uid
                FROM mcp_api_keys
                WHERE api_key_hash = %s
                LIMIT 1
                """,
                (hashed_key,),
            )
            row = cur.fetchone()

    if not row:
        return None

    user_id = row[0]

    if user_id:
        # Cache the key and update last_used_at
        redis_db.cache_mcp_api_key(hashed_key, user_id)
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE mcp_api_keys
                    SET data = jsonb_set(data, '{last_used_at}', %s::jsonb)
                    WHERE api_key_hash = %s
                    """,
                    (json.dumps(datetime.utcnow().isoformat()), hashed_key),
                )

    return user_id
