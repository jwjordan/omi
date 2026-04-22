"""Action items. Postgres-backed port.

Table: action_items (uid, id) PK
  Typed columns: conversation_id, completed, created_at, updated_at
  Everything else lives inside `data` JSONB (including sort_order,
  indent_level, due_at, completed_at, sync_requested, exported,
  export_platform, apple_reminder_id, is_locked, etc.).

Indices: (uid, created_at DESC), (uid, conversation_id)
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List
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


def _prepare_action_item_for_write(action_item_data: dict) -> tuple:
    """
    Prepare action item data for writing.

    Promotes id, conversation_id, completed, created_at to typed columns.
    Returns (id, conversation_id, completed, created_at, rest_as_jsonb).
    """
    data = dict(action_item_data)  # shallow copy

    # Extract typed columns, leave rest in JSONB
    item_id = data.pop('id', None)
    conversation_id = data.pop('conversation_id', None)
    completed = data.pop('completed', False)
    created_at = data.pop('created_at', None)

    # Ensure timestamps
    created_at = _ensure_timestamp(created_at)
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    # Convert datetime objects to ISO strings for JSON serialization
    for field in ['updated_at', 'due_at', 'completed_at']:
        if field in data and data[field]:
            ts = _ensure_timestamp(data[field])
            if ts:
                data[field] = ts.isoformat()

    return item_id, conversation_id, completed, created_at, data


def _prepare_action_item_for_read(row: tuple) -> dict:
    """
    Reconstruct action item from DB row (id, conversation_id, completed, created_at, updated_at, data).
    """
    item_id, conversation_id, completed, created_at, updated_at, data = row
    result = dict(data or {})
    result['id'] = item_id
    result['conversation_id'] = conversation_id
    result['completed'] = completed
    result['created_at'] = created_at
    result['updated_at'] = updated_at
    return result


# *****************************
# ********** CREATE ***********
# *****************************


def create_action_item(uid: str, action_item_data: dict) -> str:
    """
    Create a new action item for a user.

    Args:
        uid: User ID
        action_item_data: Action item data including description, dates, etc.

    Returns:
        The ID of the created action item
    """
    import uuid
    item_id, conversation_id, completed, created_at, rest_data = _prepare_action_item_for_write(
        action_item_data
    )

    # Generate ID if not provided
    if not item_id:
        item_id = str(uuid.uuid4())

    # Set timestamps
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    updated_at = datetime.now(timezone.utc)

    # Set completed_at if being created as completed
    if completed and 'completed_at' not in rest_data:
        rest_data['completed_at'] = updated_at.isoformat()

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO action_items (uid, id, conversation_id, completed, created_at, updated_at, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET conversation_id = EXCLUDED.conversation_id,
                        completed = EXCLUDED.completed,
                        updated_at = EXCLUDED.updated_at,
                        data = EXCLUDED.data
                """,
                (uid, item_id, conversation_id, completed, created_at, updated_at, json.dumps(rest_data)),
            )

    return item_id


def create_action_items_batch(uid: str, action_items_data: List[dict]) -> List[str]:
    """
    Create multiple action items in a batch operation.

    Args:
        uid: User ID
        action_items_data: List of action item data dictionaries

    Returns:
        List of created action item IDs
    """
    import uuid
    if not action_items_data:
        return []

    item_ids = []
    now = datetime.now(timezone.utc)

    with db.batch() as conn:
        with conn.cursor() as cur:
            for action_item_data in action_items_data:
                item_id, conversation_id, completed, created_at, rest_data = _prepare_action_item_for_write(
                    action_item_data
                )

                # Generate ID if not provided
                if not item_id:
                    item_id = str(uuid.uuid4())

                # Set timestamps
                if created_at is None:
                    created_at = now
                updated_at = now

                # Set completed_at if being created as completed
                if completed and 'completed_at' not in rest_data:
                    rest_data['completed_at'] = updated_at

                cur.execute(
                    """
                    INSERT INTO action_items (uid, id, conversation_id, completed, created_at, updated_at, data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (uid, id) DO UPDATE
                        SET conversation_id = EXCLUDED.conversation_id,
                            completed = EXCLUDED.completed,
                            updated_at = EXCLUDED.updated_at,
                            data = EXCLUDED.data
                    """,
                    (uid, item_id, conversation_id, completed, created_at, updated_at, json.dumps(rest_data)),
                )
                item_ids.append(item_id)

    return item_ids


# *****************************
# ********** READ *************
# *****************************


def get_action_item(uid: str, action_item_id: str) -> Optional[dict]:
    """
    Get a single action item by ID.

    Args:
        uid: User ID
        action_item_id: Action item ID

    Returns:
        Action item data or None if not found
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, conversation_id, completed, created_at, updated_at, data
                FROM action_items
                WHERE uid = %s AND id = %s
                """,
                (uid, action_item_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _prepare_action_item_for_read(row)


def get_action_items(
    uid: str,
    conversation_id: Optional[str] = None,
    completed: Optional[bool] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    due_start_date: Optional[datetime] = None,
    due_end_date: Optional[datetime] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[dict]:
    """
    Get action items for a user with optional filters.

    Args:
        uid: User ID
        conversation_id: Filter by conversation ID
        completed: Filter by completion status
        start_date: Filter by created_at start date (inclusive)
        end_date: Filter by created_at end date (inclusive)
        due_start_date: Filter by due_at start date (inclusive) - in data JSONB
        due_end_date: Filter by due_at end date (inclusive) - in data JSONB
        limit: Maximum number of items to return
        offset: Number of items to skip

    Returns:
        List of action items
    """
    # Build WHERE clause dynamically
    where_clauses = ["uid = %s"]
    params = [uid]

    if conversation_id is not None:
        where_clauses.append("conversation_id = %s")
        params.append(conversation_id)

    if completed is not None:
        where_clauses.append("completed = %s")
        params.append(completed)

    if start_date is not None:
        where_clauses.append("created_at >= %s")
        params.append(start_date)

    if end_date is not None:
        where_clauses.append("created_at <= %s")
        params.append(end_date)

    # due_at filtering on JSONB data
    if due_start_date is not None:
        where_clauses.append("(data->>'due_at')::timestamptz >= %s")
        params.append(due_start_date)

    if due_end_date is not None:
        where_clauses.append("(data->>'due_at')::timestamptz <= %s")
        params.append(due_end_date)

    where_sql = " AND ".join(where_clauses)

    # Determine ORDER BY based on whether due_at filtering is active
    due_at_filtering = due_start_date is not None or due_end_date is not None
    if due_at_filtering:
        order_by = "ORDER BY (data->>'due_at')::timestamptz DESC NULLS LAST"
    else:
        order_by = "ORDER BY created_at DESC"

    sql = f"""
        SELECT id, conversation_id, completed, created_at, updated_at, data
        FROM action_items
        WHERE {where_sql}
        {order_by}
        OFFSET %s
    """
    params.append(offset)

    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    action_items = [_prepare_action_item_for_read(row) for row in rows]

    # Apply client-side sorting: due_at first (nulls last), then created_at DESC
    action_items.sort(
        key=lambda x: (
            x.get('due_at') is None,
            x.get('due_at') or datetime.max.replace(tzinfo=timezone.utc),
            -(x.get('created_at', datetime.min.replace(tzinfo=timezone.utc)).timestamp()),
        )
    )

    return action_items


def get_action_items_by_conversation(uid: str, conversation_id: str) -> List[dict]:
    """
    Get all action items for a specific conversation.

    Args:
        uid: User ID
        conversation_id: Conversation ID

    Returns:
        List of action items for the conversation
    """
    return get_action_items(uid, conversation_id=conversation_id)


def get_action_items_by_ids(uid: str, action_item_ids: List[str]) -> List[dict]:
    """
    Get multiple action items by their IDs.

    Args:
        uid: User ID
        action_item_ids: List of action item IDs

    Returns:
        List of action items (only those that exist), in the same order as the input IDs
    """
    if not action_item_ids:
        return []

    # Create action_items_map to preserve order
    action_items_map = {}

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Use IN for multiple IDs
            placeholders = ",".join(["%s"] * len(action_item_ids))
            cur.execute(
                f"""
                SELECT id, conversation_id, completed, created_at, updated_at, data
                FROM action_items
                WHERE uid = %s AND id IN ({placeholders})
                """,
                [uid] + action_item_ids,
            )
            rows = cur.fetchall()

    for row in rows:
        action_item = _prepare_action_item_for_read(row)
        action_items_map[action_item['id']] = action_item

    # Return in the same order as input IDs
    action_items = []
    for item_id in action_item_ids:
        if item_id in action_items_map:
            action_items.append(action_items_map[item_id])

    return action_items


# *****************************
# ********** UPDATE ***********
# *****************************


def update_action_item(uid: str, action_item_id: str, update_data: dict) -> bool:
    """
    Update an action item.

    Args:
        uid: User ID
        action_item_id: Action item ID
        update_data: Fields to update

    Returns:
        True if updated successfully, False otherwise
    """
    item_id, conversation_id, completed, created_at, rest_data = _prepare_action_item_for_write(update_data)

    now = datetime.now(timezone.utc)

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Check if exists
            cur.execute(
                "SELECT 1 FROM action_items WHERE uid = %s AND id = %s",
                (uid, action_item_id),
            )
            if cur.fetchone() is None:
                return False

            # Update with typed column promotion
            cur.execute(
                """
                UPDATE action_items
                SET data = data || %s::jsonb,
                    completed = COALESCE(%s, completed),
                    conversation_id = COALESCE(%s, conversation_id),
                    updated_at = %s
                WHERE uid = %s AND id = %s
                """,
                (
                    json.dumps(rest_data),
                    completed if 'completed' in update_data else None,
                    conversation_id,
                    now,
                    uid,
                    action_item_id,
                ),
            )

    return True


def batch_update_action_items(uid: str, items: list) -> None:
    """
    Batch update sort_order and/or indent_level for multiple action items.

    Args:
        uid: User ID
        items: List of objects with id, sort_order (optional), indent_level (optional)
    """
    if not items:
        return

    now = datetime.now(timezone.utc)

    with db.batch() as conn:
        with conn.cursor() as cur:
            for item in items:
                update_data = {}
                if hasattr(item, 'sort_order') and item.sort_order is not None:
                    update_data['sort_order'] = item.sort_order
                if hasattr(item, 'indent_level') and item.indent_level is not None:
                    update_data['indent_level'] = item.indent_level

                if update_data:  # Only update if there are fields to update
                    cur.execute(
                        """
                        UPDATE action_items
                        SET data = data || %s::jsonb,
                            updated_at = %s
                        WHERE uid = %s AND id = %s
                        """,
                        (json.dumps(update_data), now, uid, item.id),
                    )


def mark_action_item_completed(uid: str, action_item_id: str, completed: bool = True) -> bool:
    """
    Mark an action item as completed or uncompleted.

    Args:
        uid: User ID
        action_item_id: Action item ID
        completed: Completion status

    Returns:
        True if updated successfully, False otherwise
    """
    update_data = {
        'completed': completed,
        'completed_at': datetime.now(timezone.utc) if completed else None
    }
    return update_action_item(uid, action_item_id, update_data)


# *****************************
# ********** DELETE ***********
# *****************************


def delete_action_item(uid: str, action_item_id: str) -> bool:
    """
    Delete an action item.

    Args:
        uid: User ID
        action_item_id: Action item ID

    Returns:
        True if deleted successfully, False otherwise
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM action_items WHERE uid = %s AND id = %s",
                (uid, action_item_id),
            )
            return cur.rowcount > 0


def delete_action_items_for_conversation(uid: str, conversation_id: str) -> int:
    """
    Delete all action items for a specific conversation.

    Args:
        uid: User ID
        conversation_id: Conversation ID

    Returns:
        Number of deleted items
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM action_items WHERE uid = %s AND conversation_id = %s",
                (uid, conversation_id),
            )
            return cur.rowcount


# *****************************
# ****** REMINDERS SYNC *******
# *****************************


def batch_set_sync_requested(uid: str, item_ids: List[str]) -> None:
    """Mark multiple action items as sync_requested in a single batch write."""
    if not item_ids:
        return

    now = datetime.now(timezone.utc)

    with db.batch() as conn:
        with conn.cursor() as cur:
            for item_id in item_ids:
                cur.execute(
                    """
                    UPDATE action_items
                    SET data = data || %s::jsonb,
                        updated_at = %s
                    WHERE uid = %s AND id = %s
                    """,
                    (json.dumps({'sync_requested': True}), now, uid, item_id),
                )


def get_pending_apple_reminders_sync(uid: str) -> dict:
    """
    Get items needing Apple Reminders sync:
    - pending_export: sync_requested=True but not yet exported
    - synced_items: exported to apple_reminders with apple_reminder_id
    """
    pending_export = []
    synced_items = []

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Pending export: sync_requested=True, exported != True
            cur.execute(
                """
                SELECT id, conversation_id, completed, created_at, updated_at, data
                FROM action_items
                WHERE uid = %s AND (data->>'sync_requested')::boolean = true
                LIMIT 50
                """,
                (uid,),
            )
            for row in cur.fetchall():
                action_item = _prepare_action_item_for_read(row)
                # Filter out already exported items in Python
                if action_item.get('data', {}).get('exported') is not True:
                    pending_export.append(action_item)

            # Synced items: exported to apple_reminders
            cur.execute(
                """
                SELECT id, conversation_id, completed, created_at, updated_at, data
                FROM action_items
                WHERE uid = %s AND data->>'export_platform' = %s AND (data->>'exported')::boolean = true
                LIMIT 100
                """,
                (uid, 'apple_reminders'),
            )
            synced_items = [_prepare_action_item_for_read(row) for row in cur.fetchall()]

    # Sort by updated_at desc
    synced_items.sort(
        key=lambda x: x.get('updated_at') or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )

    return {"pending_export": pending_export, "synced_items": synced_items}


def batch_sync_update_action_items(uid: str, updates: List[dict]) -> None:
    """
    Batch update action items during reminders sync.

    Args:
        uid: User ID
        updates: List of {'id': str, 'data': dict} entries
    """
    if not updates:
        return

    now = datetime.now(timezone.utc)

    with db.batch() as conn:
        with conn.cursor() as cur:
            for entry in updates:
                item_id, _, _, _, rest_data = _prepare_action_item_for_write(entry['data'])
                # rest_data already has timestamps serialized by _prepare_action_item_for_write

                # Clear sync_requested when item is successfully exported
                if rest_data.get('exported') is True:
                    rest_data['sync_requested'] = False

                cur.execute(
                    """
                    UPDATE action_items
                    SET data = data || %s::jsonb,
                        updated_at = %s
                    WHERE uid = %s AND id = %s
                    """,
                    (json.dumps(rest_data), now, uid, item_id),
                )


def unlock_all_action_items(uid: str):
    """
    Finds all action items for a user with is_locked: True and updates them to is_locked = False.
    """
    now = datetime.now(timezone.utc)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE action_items
                SET data = data || %s::jsonb,
                    updated_at = %s
                WHERE uid = %s AND (data->>'is_locked')::boolean = true
                """,
                (json.dumps({'is_locked': False}), now, uid),
            )
            count = cur.rowcount

    logger.info(f"Unlocked {count} action items for user {uid}")


# ============================================================================
# DAILY SCORE — computed from action_items
# ============================================================================


def get_daily_score(uid: str, date: str = None) -> dict:
    """Compute productivity score for a single day from action_items."""
    if date:
        day = datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    day_end = day + timedelta(days=1)

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Count tasks due today, excluding deleted ones
            cur.execute(
                """
                SELECT COUNT(*) as total, SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed
                FROM action_items
                WHERE uid = %s
                  AND (data->>'due_at')::timestamptz >= %s
                  AND (data->>'due_at')::timestamptz < %s
                  AND (data->>'deleted')::boolean IS NOT TRUE
                """,
                (uid, day, day_end),
            )
            row = cur.fetchone()
            total = row[0] or 0
            completed = row[1] or 0

    score = round((completed / total * 100) if total > 0 else 0)
    return {
        'date': day.strftime('%Y-%m-%d'),
        'score': score,
        'completed_tasks': completed,
        'total_tasks': total
    }


def get_scores(uid: str, date: str = None) -> dict:
    """Compute daily, weekly, and overall scores.

    Takes a single date (or defaults to today) and returns:
      daily  — tasks due on that date
      weekly — tasks due in the 7 days ending on that date
      overall — all non-deleted tasks
    """
    if date:
        day = datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    day_start = day
    day_end = day + timedelta(days=1)
    week_start = day - timedelta(days=7)

    def _score(completed, total):
        return round((completed / total * 100) if total > 0 else 0, 1)

    with db.connection() as conn:
        with conn.cursor() as cur:
            # Daily: tasks due today
            cur.execute(
                """
                SELECT COUNT(*) as total, SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed
                FROM action_items
                WHERE uid = %s
                  AND (data->>'due_at')::timestamptz >= %s
                  AND (data->>'due_at')::timestamptz < %s
                  AND (data->>'deleted')::boolean IS NOT TRUE
                """,
                (uid, day_start, day_end),
            )
            row = cur.fetchone()
            daily_total = row[0] or 0
            daily_completed = row[1] or 0

            # Weekly: tasks created in last 7 days
            cur.execute(
                """
                SELECT COUNT(*) as total, SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed
                FROM action_items
                WHERE uid = %s
                  AND created_at >= %s
                  AND created_at < %s
                  AND (data->>'deleted')::boolean IS NOT TRUE
                """,
                (uid, week_start, day_end),
            )
            row = cur.fetchone()
            weekly_total = row[0] or 0
            weekly_completed = row[1] or 0

            # Overall: all non-deleted tasks
            cur.execute(
                """
                SELECT COUNT(*) as total, SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed
                FROM action_items
                WHERE uid = %s
                  AND (data->>'deleted')::boolean IS NOT TRUE
                """,
                (uid,),
            )
            row = cur.fetchone()
            overall_total = row[0] or 0
            overall_completed = row[1] or 0

    daily = {
        'score': _score(daily_completed, daily_total),
        'completed_tasks': daily_completed,
        'total_tasks': daily_total,
    }
    weekly = {
        'score': _score(weekly_completed, weekly_total),
        'completed_tasks': weekly_completed,
        'total_tasks': weekly_total,
    }
    overall = {
        'score': _score(overall_completed, overall_total),
        'completed_tasks': overall_completed,
        'total_tasks': overall_total,
    }

    # Determine default tab (highest score, prefer daily > weekly > overall)
    if daily['total_tasks'] > 0 and daily['score'] >= weekly['score'] and daily['score'] >= overall['score']:
        default_tab = 'daily'
    elif weekly['score'] >= overall['score']:
        default_tab = 'weekly'
    else:
        default_tab = 'overall'

    return {
        'daily': daily,
        'weekly': weekly,
        'overall': overall,
        'default_tab': default_tab,
        'date': day.strftime('%Y-%m-%d'),
    }
