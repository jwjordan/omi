"""
Notifications database module (Postgres).

Tables:
- users (uid PK, data JSONB) — user profile + time_zone
- user_fcm_tokens (uid, device_key) → token, time_zone, created_at, data JSONB
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from ._client import db
from .cache import get_memory_cache

logger = logging.getLogger(__name__)


def save_token(uid: str, data: dict):
    """
    Store token in user_fcm_tokens table with device key.
    Migrates legacy fcm_token from users.data to user_fcm_tokens.
    Collapses unknown_default if same token is saved with proper device_key.
    Updates users.data.time_zone for efficient cross-user timezone queries.
    """
    device_key = data.get('device_key', 'unknown_default')
    token = data.get('fcm_token')
    time_zone = data.get('time_zone')

    with db.batch() as conn:
        with conn.cursor() as cur:
            # Step 1: Migrate legacy fcm_token if present and not yet in subcollection
            cur.execute(
                "SELECT data->>'fcm_token', data->>'time_zone' FROM users WHERE uid=%s",
                (uid,),
            )
            row = cur.fetchone()
            if row and row[0]:
                legacy_token, legacy_tz = row
                cur.execute(
                    "SELECT 1 FROM user_fcm_tokens WHERE uid=%s AND token=%s LIMIT 1",
                    (uid, legacy_token),
                )
                if not cur.fetchone():
                    cur.execute(
                        """
                        INSERT INTO user_fcm_tokens (uid, device_key, token, time_zone)
                        VALUES (%s, 'unknown_default', %s, %s)
                        ON CONFLICT (uid, device_key) DO UPDATE
                        SET token=EXCLUDED.token, time_zone=EXCLUDED.time_zone
                        """,
                        (uid, legacy_token, legacy_tz),
                    )
                # Remove legacy fcm_token field from users.data
                cur.execute(
                    "UPDATE users SET data = data - 'fcm_token' WHERE uid=%s",
                    (uid,),
                )

            # Step 2: If new token has proper device_key, collapse unknown_default if same token
            if device_key != 'unknown_default':
                cur.execute(
                    "SELECT token FROM user_fcm_tokens WHERE uid=%s AND device_key='unknown_default'",
                    (uid,),
                )
                row = cur.fetchone()
                if row and row[0] == token:
                    cur.execute(
                        "DELETE FROM user_fcm_tokens WHERE uid=%s AND device_key='unknown_default'",
                        (uid,),
                    )

            # Step 3: Upsert new token
            cur.execute(
                """
                INSERT INTO user_fcm_tokens (uid, device_key, token, time_zone)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (uid, device_key) DO UPDATE
                SET token=EXCLUDED.token, time_zone=EXCLUDED.time_zone
                """,
                (uid, device_key, token, time_zone),
            )

            # Step 4: Update users.data.time_zone for cross-user timezone queries
            if time_zone:
                cur.execute(
                    """
                    INSERT INTO users (uid, data)
                    VALUES (%s, jsonb_build_object('time_zone', %s::text))
                    ON CONFLICT (uid) DO UPDATE
                    SET data = users.data || jsonb_build_object('time_zone', %s::text)
                    """,
                    (uid, time_zone, time_zone),
                )


def get_user_time_zone(uid: str):
    """Get timezone from users.data"""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data->>'time_zone' FROM users WHERE uid=%s", (uid,))
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
            return None


# **************************************
# *** Daily Summary Time Preferences ***
# **************************************

# Default: 22:00 local time (10 PM)
DEFAULT_DAILY_SUMMARY_HOUR_LOCAL = 22


def get_daily_summary_hour_local(uid: str) -> int | None:
    """Get user's preferred daily summary hour in local time. Returns None if not set."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (data->>'daily_summary_hour_local')::integer FROM users WHERE uid=%s",
                (uid,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return row[0]
            return None


def set_daily_summary_hour_local(uid: str, hour_local: int) -> bool:
    """
    Set user's preferred daily summary hour in local time.

    Args:
        uid: User ID
        hour_local: Hour in local timezone (0-23)

    Returns:
        True if successful
    """
    if not (0 <= hour_local <= 23):
        raise ValueError(f"Invalid hour: {hour_local}. Must be 0-23.")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (uid, data)
                VALUES (%s, jsonb_build_object('daily_summary_hour_local', %s))
                ON CONFLICT (uid) DO UPDATE
                SET data = users.data || jsonb_build_object('daily_summary_hour_local', %s)
                """,
                (uid, hour_local, hour_local),
            )
    return True


def get_daily_summary_enabled(uid: str) -> bool:
    """Check if daily summary is enabled for user. Enabled by default."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (data->>'daily_summary_enabled')::boolean FROM users WHERE uid=%s",
                (uid,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return row[0]
            return True


def set_daily_summary_enabled(uid: str, enabled: bool) -> bool:
    """Enable or disable daily summary for user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (uid, data)
                VALUES (%s, jsonb_build_object('daily_summary_enabled', %s::boolean))
                ON CONFLICT (uid) DO UPDATE
                SET data = users.data || jsonb_build_object('daily_summary_enabled', %s::boolean)
                """,
                (uid, enabled, enabled),
            )
    return True


# **************************************
# *** Mentor Notification Frequency ***
# **************************************

# Default: 0 (disabled by default, user must explicitly enable)
# Range: 0-5 where 0=disabled, 1=most selective, 5=most proactive
DEFAULT_MENTOR_NOTIFICATION_FREQUENCY = 0


def get_mentor_notification_frequency(uid: str) -> int:
    """
    Get user's mentor notification frequency preference.
    Returns 0-5 where:
    - 0 = disabled
    - 1 = ultra selective (least frequent)
    - 3 = balanced (default)
    - 5 = very proactive (most frequent)

    Uses in-memory cache (30s TTL) to avoid reading the full user doc every 1s per stream.
    """
    cache = get_memory_cache()

    def fetch():
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT (data->>'mentor_notification_frequency')::integer FROM users WHERE uid=%s",
                    (uid,),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    return row[0]
                return DEFAULT_MENTOR_NOTIFICATION_FREQUENCY

    return cache.get_or_fetch(f"mentor_frequency:{uid}", fetch, ttl=30)


def set_mentor_notification_frequency(uid: str, frequency: int) -> bool:
    """
    Set user's mentor notification frequency preference.

    Args:
        uid: User ID
        frequency: Notification frequency (0-5)

    Returns:
        True if successful

    Raises:
        ValueError if frequency is not in valid range
    """
    if not (0 <= frequency <= 5):
        raise ValueError(f"Invalid frequency: {frequency}. Must be 0-5.")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (uid, data)
                VALUES (%s, jsonb_build_object('mentor_notification_frequency', %s))
                ON CONFLICT (uid) DO UPDATE
                SET data = users.data || jsonb_build_object('mentor_notification_frequency', %s)
                """,
                (uid, frequency, frequency),
            )
    # Invalidate local cache so this instance sees the update immediately
    get_memory_cache().delete(f"mentor_frequency:{uid}")
    return True


def get_all_tokens(uid: str) -> list[str]:
    """Get all device tokens for a user from user_fcm_tokens table"""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT token FROM user_fcm_tokens WHERE uid=%s AND token IS NOT NULL",
                (uid,),
            )
            return [row[0] for row in cur.fetchall()]


def remove_invalid_token(token: str):
    """Remove invalid token from user_fcm_tokens table (cross-user)"""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_fcm_tokens WHERE token=%s", (token,))


def remove_bulk_tokens(tokens: list[str]):
    """Remove multiple invalid tokens efficiently"""
    if not tokens:
        return
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_fcm_tokens WHERE token = ANY(%s::text[])",
                (tokens,),
            )


async def get_users_token_in_timezones(timezones: list[str]):
    """Get list of unique FCM tokens for users in given timezones"""
    return await _get_users_in_timezones(timezones, 'fcm_token')


async def get_users_id_in_timezones(timezones: list[str]):
    """Get list of (uid, [tokens], time_zone) tuples for users in given timezones"""
    return await _get_users_in_timezones(timezones, 'id')


async def get_users_for_daily_summary(timezones: list[str], target_local_hour: int):
    """
    Get users who should receive daily summary notifications.

    This function queries users who:
    1. Are in one of the provided timezones (where it's currently target_local_hour)
    2. Have daily_summary_hour_local set to target_local_hour OR have no preference (uses default)
    3. Have daily_summary_enabled not explicitly set to False

    Args:
        timezones: List of IANA timezone names where it's currently target_local_hour
        target_local_hour: The local hour we're sending notifications for (0-23)

    Returns:
        List of (uid, [tokens], time_zone) tuples.
    """
    if not timezones:
        return []

    users = []

    # Split into chunks of 30 (Postgres ANY array query limit equivalent)
    timezone_chunks = [timezones[i : i + 30] for i in range(0, len(timezones), 30)]

    async def query_chunk(chunk):
        def sync_query():
            chunk_users = []
            try:
                with db.connection() as conn:
                    with conn.cursor() as cur:
                        # Query users in these timezones
                        cur.execute(
                            """
                            SELECT DISTINCT u.uid, u.data
                            FROM users u
                            WHERE u.data->>'time_zone' = ANY(%s::text[])
                            """,
                            (chunk,),
                        )
                        for uid, user_data in cur.fetchall():
                            user_data = user_data or {}

                            # Check if daily summary is enabled (default: True)
                            daily_enabled = user_data.get('daily_summary_enabled')
                            if daily_enabled is False:
                                continue

                            # Check if user's preferred hour matches target hour
                            # If not set, use default (22 = 10 PM)
                            user_hour = user_data.get('daily_summary_hour_local', DEFAULT_DAILY_SUMMARY_HOUR_LOCAL)
                            if user_hour != target_local_hour:
                                continue

                            # Collect tokens from user_fcm_tokens
                            with conn.cursor() as cur2:
                                cur2.execute(
                                    "SELECT token FROM user_fcm_tokens WHERE uid=%s AND token IS NOT NULL",
                                    (uid,),
                                )
                                tokens = [row[0] for row in cur2.fetchall()]

                            # Skip users with no tokens
                            if not tokens:
                                continue

                            time_zone = user_data.get('time_zone')
                            chunk_users.append((uid, tokens, time_zone))

            except Exception as e:
                logger.error(f"Error querying chunk for daily summary: {e}")
            return chunk_users

        return await asyncio.to_thread(sync_query)

    tasks = [query_chunk(chunk) for chunk in timezone_chunks]
    results = await asyncio.gather(*tasks)

    for chunk_users in results:
        users.extend(chunk_users)

    return users


async def _get_users_in_timezones(timezones: list[str], filter_type: str):
    """Query users by timezone, then get tokens from user_fcm_tokens"""
    users = []

    # Split into chunks of 30
    timezone_chunks = [timezones[i : i + 30] for i in range(0, len(timezones), 30)]

    async def query_chunk(chunk):
        def sync_query():
            chunk_users = []
            try:
                with db.connection() as conn:
                    with conn.cursor() as cur:
                        # Query users by time_zone
                        cur.execute(
                            """
                            SELECT DISTINCT u.uid, u.data
                            FROM users u
                            WHERE u.data->>'time_zone' = ANY(%s::text[])
                            """,
                            (chunk,),
                        )
                        for uid, user_data in cur.fetchall():
                            # Collect tokens from user_fcm_tokens
                            with conn.cursor() as cur2:
                                cur2.execute(
                                    "SELECT token FROM user_fcm_tokens WHERE uid=%s AND token IS NOT NULL",
                                    (uid,),
                                )
                                tokens = [row[0] for row in cur2.fetchall()]

                            # Skip users with no tokens
                            if not tokens:
                                continue

                            if filter_type == 'fcm_token':
                                # Return flat list of tokens
                                chunk_users.extend(tokens)
                            else:
                                # Return list of (uid, [tokens], time_zone) tuples
                                user_data = user_data or {}
                                time_zone = user_data.get('time_zone')
                                chunk_users.append((uid, tokens, time_zone))

            except Exception as e:
                logger.error(f"Error querying chunk {chunk}: {e}")
            return chunk_users

        return await asyncio.to_thread(sync_query)

    tasks = [query_chunk(chunk) for chunk in timezone_chunks]
    results = await asyncio.gather(*tasks)

    for chunk_users in results:
        users.extend(chunk_users)

    return users
