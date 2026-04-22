"""Goals tracking database operations.

Table: goals (uid, id) PK
  Typed columns: is_active, created_at, updated_at
  Everything else lives inside `data` JSONB (current_value, target_date, etc.).

Indices: (uid, created_at DESC), (uid, is_active)

Separate table: goal_history (uid, goal_id, date) PK
  Stores progress data points with (date, value, recorded_at).
"""

import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from ._client import db
import logging

logger = logging.getLogger(__name__)


def _ensure_timestamp(ts) -> Optional[datetime]:
    """Ensure timestamp is a datetime object."""
    if ts is None:
        return None
    if isinstance(ts, str):
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return None


def _prepare_goal_for_write(goal_data: dict) -> tuple:
    """
    Prepare goal data for writing.

    Promotes id, is_active, created_at to typed columns.
    Returns (id, is_active, created_at, rest_as_jsonb).
    """
    data = dict(goal_data)  # shallow copy

    # Extract typed columns, leave rest in JSONB
    goal_id = data.pop('id', None)
    is_active = data.pop('is_active', True)
    created_at = data.pop('created_at', None)

    # Ensure timestamps
    created_at = _ensure_timestamp(created_at)
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    # Convert datetime objects to ISO strings for JSON serialization
    for field in ['updated_at', 'target_date', 'ended_at']:
        if field in data and data[field]:
            ts = _ensure_timestamp(data[field])
            if ts:
                data[field] = ts.isoformat()

    return goal_id, is_active, created_at, data


def _prepare_goal_for_read(row: tuple) -> dict:
    """
    Reconstruct goal from DB row (id, is_active, created_at, updated_at, data).
    """
    goal_id, is_active, created_at, updated_at, data = row
    result = dict(data or {})
    result['id'] = goal_id
    result['is_active'] = is_active
    result['created_at'] = created_at
    result['updated_at'] = updated_at
    return result


# *****************************
# ********** CREATE ***********
# *****************************


def create_goal(uid: str, goal_data: Dict[str, Any], max_goals: int = 4) -> Dict[str, Any]:
    """
    Create a new goal for a user. Supports up to max_goals active goals.

    Args:
        uid: User ID
        goal_data: Goal data including title, target_date, etc.
        max_goals: Maximum number of active goals allowed

    Returns:
        The created goal data
    """
    import uuid

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Check current active goal count
            cur.execute(
                "SELECT COUNT(*) FROM goals WHERE uid = %s AND is_active = true",
                (uid,),
            )
            count = cur.fetchone()[0]

            # If at max, deactivate the oldest one
            if count >= max_goals:
                cur.execute(
                    """
                    SELECT id FROM goals
                    WHERE uid = %s AND is_active = true
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (uid,),
                )
                oldest = cur.fetchone()
                if oldest:
                    oldest_id = oldest[0]
                    now = datetime.now(timezone.utc)
                    cur.execute(
                        """
                        UPDATE goals
                        SET is_active = false, updated_at = %s,
                            data = data || %s::jsonb
                        WHERE uid = %s AND id = %s
                        """,
                        (now, json.dumps({'ended_at': now.isoformat()}), uid, oldest_id),
                    )

    # Create new goal
    goal_id, is_active, created_at, rest_data = _prepare_goal_for_write(goal_data)

    # Generate ID if not provided
    if not goal_id:
        goal_id = str(uuid.uuid4())

    # Set timestamps
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    updated_at = datetime.now(timezone.utc)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO goals (uid, id, is_active, created_at, updated_at, data)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET is_active = EXCLUDED.is_active,
                        updated_at = EXCLUDED.updated_at,
                        data = EXCLUDED.data
                """,
                (uid, goal_id, is_active, created_at, updated_at, json.dumps(rest_data)),
            )

    # Return the created goal
    result = dict(rest_data)
    result['id'] = goal_id
    result['is_active'] = is_active
    result['created_at'] = created_at
    result['updated_at'] = updated_at
    return result


# *****************************
# ********** READ *************
# *****************************


def get_user_goal(uid: str) -> Optional[Dict[str, Any]]:
    """
    Get the current active goal for a user (backward compatibility - returns first active goal).

    Args:
        uid: User ID

    Returns:
        First active goal or None
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, is_active, created_at, updated_at, data
                FROM goals
                WHERE uid = %s AND is_active = true
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _prepare_goal_for_read(row)


def get_user_goals(uid: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Get all active goals for a user (up to limit).

    Args:
        uid: User ID
        limit: Maximum number of goals to return

    Returns:
        List of active goals sorted by created_at ASC
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, is_active, created_at, updated_at, data
                FROM goals
                WHERE uid = %s AND is_active = true
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (uid, limit),
            )
            rows = cur.fetchall()

    goals = [_prepare_goal_for_read(row) for row in rows]
    return goals


def get_all_goals(uid: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
    """
    Get all goals for a user.

    Args:
        uid: User ID
        include_inactive: If True, include inactive goals; otherwise only active

    Returns:
        List of goals sorted by created_at DESC
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            if include_inactive:
                cur.execute(
                    """
                    SELECT id, is_active, created_at, updated_at, data
                    FROM goals
                    WHERE uid = %s
                    ORDER BY created_at DESC
                    """,
                    (uid,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, is_active, created_at, updated_at, data
                    FROM goals
                    WHERE uid = %s AND is_active = true
                    ORDER BY created_at DESC
                    """,
                    (uid,),
                )
            rows = cur.fetchall()

    goals = [_prepare_goal_for_read(row) for row in rows]
    return goals


def get_goal(uid: str, goal_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single goal by ID.

    Args:
        uid: User ID
        goal_id: Goal ID

    Returns:
        Goal data or None if not found
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, is_active, created_at, updated_at, data
                FROM goals
                WHERE uid = %s AND id = %s
                """,
                (uid, goal_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _prepare_goal_for_read(row)


# *****************************
# ********** UPDATE ***********
# *****************************


def update_goal(uid: str, goal_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update an existing goal.

    Args:
        uid: User ID
        goal_id: Goal ID
        updates: Fields to update

    Returns:
        Updated goal data or None if not found
    """
    now = datetime.now(timezone.utc)

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Check if exists
            cur.execute(
                "SELECT 1 FROM goals WHERE uid = %s AND id = %s",
                (uid, goal_id),
            )
            if cur.fetchone() is None:
                return None

            # Extract typed columns from updates
            goal_id_update, is_active_update, _, rest_data = _prepare_goal_for_write(updates)

            # Update with typed column promotion
            cur.execute(
                """
                UPDATE goals
                SET data = data || %s::jsonb,
                    is_active = COALESCE(%s, is_active),
                    updated_at = %s
                WHERE uid = %s AND id = %s
                """,
                (
                    json.dumps(rest_data),
                    is_active_update if 'is_active' in updates else None,
                    now,
                    uid,
                    goal_id,
                ),
            )

            # Fetch and return updated goal
            cur.execute(
                """
                SELECT id, is_active, created_at, updated_at, data
                FROM goals
                WHERE uid = %s AND id = %s
                """,
                (uid, goal_id),
            )
            row = cur.fetchone()
            if row:
                return _prepare_goal_for_read(row)
    return None


def update_goal_progress(uid: str, goal_id: str, current_value: float) -> Optional[Dict[str, Any]]:
    """
    Update the current progress value of a goal.

    Args:
        uid: User ID
        goal_id: Goal ID
        current_value: New current value

    Returns:
        Updated goal or None if not found
    """
    result = update_goal(uid, goal_id, {'current_value': current_value})
    if result:
        # Also save to history
        save_goal_progress_history(uid, goal_id, current_value)
    return result


# *****************************
# ********** DELETE ***********
# *****************************


def delete_goal(uid: str, goal_id: str) -> bool:
    """
    Delete a goal.

    Args:
        uid: User ID
        goal_id: Goal ID

    Returns:
        True if deleted successfully, False otherwise
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Check if exists
            cur.execute(
                "SELECT 1 FROM goals WHERE uid = %s AND id = %s",
                (uid, goal_id),
            )
            if cur.fetchone() is None:
                return False

            # Delete goal
            cur.execute(
                "DELETE FROM goals WHERE uid = %s AND id = %s",
                (uid, goal_id),
            )

            # Also delete associated history
            cur.execute(
                "DELETE FROM goal_history WHERE uid = %s AND goal_id = %s",
                (uid, goal_id),
            )

            return True


# *****************************
# ***** PROGRESS HISTORY ******
# *****************************


def save_goal_progress_history(uid: str, goal_id: str, value: float) -> None:
    """
    Save a progress data point to history.

    Args:
        uid: User ID
        goal_id: Goal ID
        value: Progress value
    """
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    recorded_at = datetime.now(timezone.utc)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO goal_history (uid, goal_id, date, value, recorded_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (uid, goal_id, date) DO UPDATE
                    SET value = EXCLUDED.value,
                        recorded_at = EXCLUDED.recorded_at
                """,
                (uid, goal_id, today, value, recorded_at),
            )


def get_goal_history(uid: str, goal_id: str, days: int = 30) -> List[Dict[str, Any]]:
    """
    Get progress history for a goal.

    Args:
        uid: User ID
        goal_id: Goal ID
        days: Maximum number of days to return

    Returns:
        List of history entries ordered by date DESC
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, value, recorded_at
                FROM goal_history
                WHERE uid = %s AND goal_id = %s
                ORDER BY date DESC
                LIMIT %s
                """,
                (uid, goal_id, days),
            )
            rows = cur.fetchall()

    history = []
    for row in rows:
        history.append({
            'date': row[0],
            'value': row[1],
            'recorded_at': row[2],
        })
    return history
