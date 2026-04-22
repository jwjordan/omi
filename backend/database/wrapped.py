"""
Database operations for Wrapped (yearly recap) stored in wrapped table.

Table: wrapped (per-user, keyed by uid + id)
    uid TEXT NOT NULL
    id TEXT NOT NULL       -- year as string, e.g. '2026'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    data JSONB NOT NULL DEFAULT '{}'::jsonb
    PRIMARY KEY (uid, id)

The year is stored as a string in the id column; all other data
(status, started_at, updated_at, completed_at, result, error, progress, schema_version)
lives in the data JSONB column.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from ._client import db


class WrappedStatus:
    NOT_GENERATED = 'not_generated'
    PROCESSING = 'processing'
    DONE = 'done'
    ERROR = 'error'


def get_wrapped(uid: str, year: int) -> Optional[dict]:
    """
    Get the wrapped document for a user and year.

    Args:
        uid: User ID
        year: Year (e.g., 2025)

    Returns:
        Wrapped document data or None if not found
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data FROM wrapped
                WHERE uid = %s AND id = %s
                """,
                (uid, str(year)),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0] or {}


def create_wrapped(uid: str, year: int) -> dict:
    """
    Create a new wrapped document with status=processing.

    Args:
        uid: User ID
        year: Year (e.g., 2025)

    Returns:
        The created wrapped document data
    """
    now = datetime.now(timezone.utc)
    wrapped_data = {
        'year': year,
        'status': WrappedStatus.PROCESSING,
        'started_at': now.isoformat(),
        'updated_at': now.isoformat(),
        'completed_at': None,
        'result': None,
        'error': None,
        'schema_version': 1,
    }

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO wrapped (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET data = EXCLUDED.data
                """,
                (uid, str(year), json.dumps(wrapped_data)),
            )

    return wrapped_data


def update_wrapped_status(
    uid: str,
    year: int,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> bool:
    """
    Update the status of a wrapped document.

    Args:
        uid: User ID
        year: Year (e.g., 2025)
        status: New status (processing, done, error)
        result: Result payload (only when status=done)
        error: Error message (only when status=error)

    Returns:
        True if updated successfully
    """
    now = datetime.now(timezone.utc)
    update_dict = {
        'status': status,
        'updated_at': now.isoformat(),
    }

    if status == WrappedStatus.DONE:
        update_dict['completed_at'] = now.isoformat()
        update_dict['result'] = result
        update_dict['error'] = None
    elif status == WrappedStatus.ERROR:
        update_dict['error'] = error
        update_dict['result'] = None

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE wrapped
                SET data = data || %s::jsonb
                WHERE uid=%s AND id=%s
                """,
                (json.dumps(update_dict), uid, str(year)),
            )
            return cur.rowcount > 0


def update_wrapped_progress(uid: str, year: int, progress: dict) -> bool:
    """
    Update the progress of a wrapped generation (heartbeat).

    Args:
        uid: User ID
        year: Year (e.g., 2025)
        progress: Progress info (e.g., {"step": "computing_stats", "pct": 0.5})

    Returns:
        True if updated successfully
    """
    update_dict = {
        'progress': progress,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE wrapped
                SET data = data || %s::jsonb
                WHERE uid=%s AND id=%s
                """,
                (json.dumps(update_dict), uid, str(year)),
            )
            return cur.rowcount > 0


def reset_wrapped_for_regeneration(uid: str, year: int) -> dict:
    """
    Reset a stuck or errored wrapped document for regeneration.

    Args:
        uid: User ID
        year: Year (e.g., 2025)

    Returns:
        The updated wrapped document data
    """
    now = datetime.now(timezone.utc)
    wrapped_data = {
        'year': year,
        'status': WrappedStatus.PROCESSING,
        'started_at': now.isoformat(),
        'updated_at': now.isoformat(),
        'completed_at': None,
        'result': None,
        'error': None,
        'progress': None,
        'schema_version': 1,
    }

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO wrapped (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET data = EXCLUDED.data
                """,
                (uid, str(year), json.dumps(wrapped_data)),
            )

    return wrapped_data


def is_wrapped_stuck(wrapped_data: dict, stale_minutes: int = 15) -> bool:
    """
    Check if a wrapped generation is stuck (no heartbeat for stale_minutes).

    Args:
        wrapped_data: The wrapped document data
        stale_minutes: Minutes after which a processing job is considered stuck

    Returns:
        True if the job appears stuck
    """
    if wrapped_data.get('status') != WrappedStatus.PROCESSING:
        return False

    updated_at = wrapped_data.get('updated_at')
    if not updated_at:
        return True

    # Ensure updated_at is a datetime
    if isinstance(updated_at, str):
        updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
    elif hasattr(updated_at, 'timestamp'):
        updated_at = datetime.fromtimestamp(updated_at.timestamp(), tz=timezone.utc)
    elif updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    elapsed = (now - updated_at).total_seconds() / 60

    return elapsed > stale_minutes
