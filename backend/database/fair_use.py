"""Postgres CRUD for fair-use tracking.

Two tables:
  - fair_use_state (uid TEXT PRIMARY KEY): singleton per user, stores stage/counts/metadata
  - fair_use_events (uid, id PRIMARY KEY): many per user, stores violation events

Required indexes:
  - fair_use_state: none beyond PK
  - fair_use_events: (uid, created_at DESC), (uid, resolved)
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from ._client import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fair-use state (users/{uid}/fair_use_state/current)
# ---------------------------------------------------------------------------


def get_fair_use_state(uid: str) -> dict:
    """Get the current fair-use enforcement state for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM fair_use_state WHERE uid = %s",
                (uid,)
            )
            row = cur.fetchone()
            if row:
                return row[0]
            return {}


def update_fair_use_state(uid: str, updates: dict) -> None:
    """Update fair-use state atomically."""
    updates['updated_at'] = datetime.utcnow()
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO fair_use_state (uid, data, updated_at)
                   VALUES (%s, %s::jsonb, now())
                   ON CONFLICT (uid) DO UPDATE
                       SET data = fair_use_state.data || EXCLUDED.data,
                           updated_at = now()""",
                (uid, updates)
            )


def set_fair_use_stage(uid: str, stage: str, **kwargs) -> None:
    """Set enforcement stage with optional extra fields."""
    updates = {'stage': stage, **kwargs}
    update_fair_use_state(uid, updates)


# ---------------------------------------------------------------------------
# Fair-use events (users/{uid}/fair_use_events/{event_id})
# ---------------------------------------------------------------------------


def _generate_case_ref() -> str:
    """Generate a human-readable case reference like FU-A1B2C3D4E5F6.

    Uses 12 hex chars from UUID4 (16^12 ≈ 281 trillion possibilities),
    safe for public unauthenticated lookup without enumeration risk.
    """
    return f'FU-{uuid.uuid4().hex[:12].upper()}'


def create_fair_use_event(uid: str, event_data: dict) -> str:
    """Create a new fair-use violation event. Returns the event ID."""
    event_id = str(uuid.uuid4())
    event_data['created_at'] = datetime.utcnow()
    event_data['case_ref'] = _generate_case_ref()

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO fair_use_events (uid, id, data)
                   VALUES (%s, %s, %s::jsonb)
                   RETURNING id""",
                (uid, event_id, event_data)
            )
            row = cur.fetchone()
            return row[0] if row else event_id


def get_fair_use_events(uid: str, limit: int = 50) -> list:
    """Get recent fair-use events for a user, newest first."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, resolved, data FROM fair_use_events
                   WHERE uid = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (uid, limit)
            )
            events = []
            for row in cur.fetchall():
                event_id, resolved, data = row
                data['id'] = event_id
                data['resolved'] = resolved
                events.append(data)
            return events


def get_violation_counts(uid: str) -> dict:
    """Count violations in the last 7 and 30 days."""
    cutoff_30d = datetime.utcnow() - timedelta(days=30)
    cutoff_7d = datetime.utcnow() - timedelta(days=7)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT
                     SUM(CASE WHEN created_at >= %s THEN 1 ELSE 0 END) as count_7d,
                     SUM(CASE WHEN created_at >= %s THEN 1 ELSE 0 END) as count_30d
                   FROM fair_use_events
                   WHERE uid = %s AND NOT resolved""",
                (cutoff_7d, cutoff_30d, uid)
            )
            row = cur.fetchone()
            if row:
                count_7d, count_30d = row
                return {
                    'violation_count_7d': count_7d or 0,
                    'violation_count_30d': count_30d or 0
                }
            return {'violation_count_7d': 0, 'violation_count_30d': 0}


def resolve_fair_use_event(uid: str, event_id: str, admin_uid: str, notes: str = "") -> None:
    """Mark a fair-use event as resolved by admin."""
    resolved_at = datetime.utcnow()

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE fair_use_events
                   SET resolved = TRUE,
                       data = data || jsonb_build_object(
                           'resolved_by', %s,
                           'admin_notes', %s,
                           'resolved_at', %s::text
                       )
                   WHERE uid = %s AND id = %s""",
                (admin_uid, notes, resolved_at.isoformat(), uid, event_id)
            )


def reset_fair_use_state(uid: str, admin_uid: str) -> None:
    """Reset a user's fair-use state to clean (admin action)."""
    reset_at = datetime.utcnow()

    with db.batch() as conn:
        with conn.cursor() as cur:
            # Reset state to clean
            cur.execute(
                """INSERT INTO fair_use_state (uid, data, updated_at)
                   VALUES (%s, %s::jsonb, now())
                   ON CONFLICT (uid) DO UPDATE
                       SET data = %s::jsonb,
                           updated_at = now()""",
                (uid, {
                    'stage': 'none',
                    'violation_count_7d': 0,
                    'violation_count_30d': 0,
                    'last_violation_at': None,
                    'throttle_until': None,
                    'restrict_until': None,
                    'last_classifier_score': 0.0,
                    'last_classifier_type': 'none',
                    'reset_by': admin_uid,
                    'reset_at': reset_at.isoformat(),
                }, {
                    'stage': 'none',
                    'violation_count_7d': 0,
                    'violation_count_30d': 0,
                    'last_violation_at': None,
                    'throttle_until': None,
                    'restrict_until': None,
                    'last_classifier_score': 0.0,
                    'last_classifier_type': 'none',
                    'reset_by': admin_uid,
                    'reset_at': reset_at.isoformat(),
                })
            )
            # Clear all events for this user
            cur.execute("DELETE FROM fair_use_events WHERE uid = %s", (uid,))


# ---------------------------------------------------------------------------
# Admin queries
# ---------------------------------------------------------------------------


def get_flagged_users(stage_filter: Optional[str] = None, limit: int = 100) -> list:
    """Get users with active fair-use enforcement, for admin dashboard."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            if stage_filter:
                cur.execute(
                    """SELECT uid, data FROM fair_use_state
                       WHERE data->>'stage' = %s
                       ORDER BY updated_at DESC
                       LIMIT %s""",
                    (stage_filter, limit)
                )
            else:
                cur.execute(
                    """SELECT uid, data FROM fair_use_state
                       WHERE data->>'stage' IN ('warning', 'throttle', 'restrict')
                       ORDER BY updated_at DESC
                       LIMIT %s""",
                    (limit,)
                )

            results = []
            for uid, data in cur.fetchall():
                data['uid'] = uid
                data['id'] = 'current'
                results.append(data)
            return results
