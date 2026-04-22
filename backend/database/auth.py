"""Postgres-backed user profile lookups.

Replaces the firebase_admin.auth + Firestore hybrid with a single
read from our users table. Email verification / disabled / etc. are
not tracked here; the iOS app still signs in via Firebase Auth and
those semantics live there. We only expose what Omi's business logic
actually consumes: uid, email, display_name, name, photo_url, phone_number.
"""

import logging
from typing import Optional

from database._client import db
from database.redis_db import cache_user_name

logger = logging.getLogger(__name__)


def _fetch_user_row(uid: str) -> Optional[dict]:
    """Read users.data JSONB column for a uid; return None if not found."""
    if not uid:
        return None
    try:
        with db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM users WHERE uid = %s", (uid,))
                row = cur.fetchone()
                if row is None:
                    return None
                return row[0] or {}
    except Exception as e:
        logger.error("Postgres user lookup failed: %s", e)
        return None


def get_user_from_uid(uid: str):
    """Fetch user profile from Postgres users table.

    Returns dict with uid, email, email_verified, phone_number, display_name,
    photo_url, disabled; or None if not found.
    """
    data = _fetch_user_row(uid)
    if data is None:
        return None
    return {
        'uid': uid,
        'email': data.get('email'),
        'email_verified': data.get('email_verified', True),
        'phone_number': data.get('phone_number'),
        'display_name': data.get('display_name'),
        'photo_url': data.get('photo_url'),
        'disabled': data.get('disabled', False),
    }


def _get_user_profile_name(uid: str) -> Optional[str]:
    """Read the 'name' field from users.data; return first word or None."""
    data = _fetch_user_row(uid)
    if data is None:
        return None
    name = data.get('name')
    if name and isinstance(name, str):
        return name.split(' ')[0]
    return None


def get_user_name(uid: str, use_default: bool = True):
    """Get display name for uid, falling back to name field, then default.

    Extracts first word of display_name or name. Returns 'The User' if
    use_default=True and no name found; otherwise None.

    Caches the result in Redis.
    """
    default_name = 'The User' if use_default else None
    user = get_user_from_uid(uid)
    if not user:
        # No user row; try name field
        profile_name = _get_user_profile_name(uid)
        if profile_name:
            cache_user_name(uid, profile_name, ttl=60 * 60)
            return profile_name
        return default_name

    display_name = user.get('display_name')
    if not display_name:
        # No display_name; try name field
        profile_name = _get_user_profile_name(uid)
        if profile_name:
            cache_user_name(uid, profile_name, ttl=60 * 60)
            return profile_name
        return default_name

    display_name = display_name.split(' ')[0]
    if display_name == 'AnonymousUser':
        # Replace anonymous with name field or default
        profile_name = _get_user_profile_name(uid)
        if profile_name:
            display_name = profile_name
        else:
            display_name = default_name

    cache_user_name(uid, display_name, ttl=60 * 60)
    return display_name
