"""Staged tasks — AI-generated tasks awaiting user promotion to action items.

Table: staged_tasks (uid, id, created_at, data)
Migration: can read/write action_items for bulk moves.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ._client import db
import database.action_items as action_items_db

logger = logging.getLogger(__name__)

BATCH_LIMIT = 500  # Postgres limit for batch operations


def create_staged_task(uid: str, description: str, **kwargs) -> dict:
    """Create a staged task. Deduplicates by case-insensitive description."""
    # Deduplicate: check for existing task with matching description
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM staged_tasks WHERE uid = %s ORDER BY created_at DESC",
                (uid,),
            )
            for task_id, data in cur.fetchall():
                if data.get("description", "").strip().lower() == description.strip().lower():
                    existing = {**data, "id": task_id}
                    return existing

    # Create new task
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = {
        "id": task_id,
        "description": description,
        "completed": False,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    for field in ("due_at", "source", "priority", "metadata", "category", "relevance_score"):
        if field in kwargs and kwargs[field] is not None:
            doc[field] = kwargs[field]

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO staged_tasks (uid, id, created_at, data) VALUES (%s, %s, %s, %s)",
                (uid, task_id, now, json.dumps(doc)),
            )

    return doc


def get_staged_tasks(uid: str, limit: int = 100, offset: int = 0) -> List[dict]:
    """Fetch uncompleted staged tasks ordered by relevance_score (ascending)."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM staged_tasks
                WHERE uid = %s AND (data->>'completed')::boolean = false
                ORDER BY (data->>'relevance_score')::numeric ASC NULLS LAST
                LIMIT %s OFFSET %s
                """,
                (uid, limit, offset),
            )
            items = []
            for task_id, data in cur.fetchall():
                item = {**data, "id": task_id}
                items.append(item)
            return items


def delete_staged_task(uid: str, task_id: str) -> bool:
    """Delete a staged task. Returns True if found and deleted, False otherwise."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM staged_tasks WHERE uid = %s AND id = %s",
                (uid, task_id),
            )
            return cur.rowcount > 0


def batch_update_staged_scores(uid: str, scores: List[dict]) -> None:
    """Update relevance_score for staged tasks in batches of 500.

    Pre-filters to active (uncompleted) document IDs so stale/deleted/promoted
    task references from the client don't cause errors.
    """
    if not scores:
        return

    # Fetch active task IDs
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM staged_tasks WHERE uid = %s AND (data->>'completed')::boolean = false",
                (uid,),
            )
            existing_ids = {row[0] for row in cur.fetchall()}

    valid_scores = [s for s in scores if s["id"] in existing_ids]
    if not valid_scores:
        return

    # Update in batches using transaction
    now = datetime.now(timezone.utc)
    batch_count = 0

    with db.batch() as conn:
        with conn.cursor() as cur:
            for item in valid_scores:
                cur.execute(
                    """
                    UPDATE staged_tasks
                    SET data = jsonb_set(data, '{relevance_score}', %s::jsonb),
                        data = jsonb_set(data, '{updated_at}', %s::jsonb)
                    WHERE uid = %s AND id = %s
                    """,
                    (json.dumps(item["relevance_score"]), json.dumps(now.isoformat()), uid, item["id"]),
                )
                batch_count += 1
                if batch_count >= BATCH_LIMIT:
                    # Continue in next batch (db.batch handles commit)
                    batch_count = 0


def promote_staged_task(uid: str) -> Optional[dict]:
    """Promote the top-scored staged task to an action_item.

    Returns the new action_item dict or None if no staged tasks exist.
    Uses database.action_items.create_action_item() for consistent field handling.
    """
    with db.batch() as conn:
        with conn.cursor() as cur:
            # Select top task with FOR UPDATE lock
            cur.execute(
                """
                SELECT id, data FROM staged_tasks
                WHERE uid = %s AND (data->>'completed')::boolean = false
                ORDER BY (data->>'relevance_score')::numeric ASC NULLS LAST
                LIMIT 1
                FOR UPDATE
                """,
                (uid,),
            )
            row = cur.fetchone()
            if not row:
                return None

            task_id, data = row
            staged = {**data, "id": task_id}

            # Build action_item data from staged task fields
            action_data = {
                "description": staged["description"],
                "completed": False,
                "from_staged": True,
            }
            for field in ("due_at", "source", "priority", "metadata", "category", "relevance_score"):
                if staged.get(field) is not None:
                    action_data[field] = staged[field]

            # Create action item outside transaction (it uses its own db.connection)
            action_id = action_items_db.create_action_item(uid, action_data)

            # Delete from staged_tasks inside transaction
            cur.execute(
                "DELETE FROM staged_tasks WHERE uid = %s AND id = %s",
                (uid, task_id),
            )

    # Fetch and return the created action item
    action_item = action_items_db.get_action_item(uid, action_id)
    return action_item


def migrate_ai_tasks(uid: str) -> dict:
    """One-time migration: move excess AI tasks from action_items to staged_tasks.

    Keeps top 3 AI tasks in action_items, moves the rest to staged_tasks.
    Uses a 'source' field marker to identify AI-created tasks.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM action_items
                WHERE uid = %s AND (data->>'completed')::boolean = false
                ORDER BY created_at ASC
                """,
                (uid,),
            )
            all_items = []
            for item_id, data in cur.fetchall():
                item = {**data, "id": item_id}
                if not item.get("deleted"):
                    all_items.append(item)

    # Separate AI-generated tasks from manual ones
    ai_tasks = [item for item in all_items if "screenshot" in (item.get("source") or "")]
    if len(ai_tasks) <= 3:
        return {"moved": 0, "kept": len(ai_tasks)}

    # Sort by relevance_score ascending (best first)
    ai_tasks.sort(key=lambda x: x.get("relevance_score") or 999)
    keep = ai_tasks[:3]
    to_move = ai_tasks[3:]

    # Move tasks atomically
    with db.batch() as conn:
        with conn.cursor() as cur:
            for task in to_move:
                # Insert into staged_tasks
                cur.execute(
                    "INSERT INTO staged_tasks (uid, id, created_at, data) VALUES (%s, %s, %s, %s)",
                    (uid, task["id"], task.get("created_at"), json.dumps(task)),
                )
                # Delete from action_items
                cur.execute(
                    "DELETE FROM action_items WHERE uid = %s AND id = %s",
                    (uid, task["id"]),
                )

    return {"moved": len(to_move), "kept": len(keep)}


def migrate_conversation_items_to_staged(uid: str) -> dict:
    """Move conversation-sourced action items (with conversation_id, no source) to staged_tasks."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data FROM action_items
                WHERE uid = %s
                """,
                (uid,),
            )
            items_to_move = []
            for item_id, data in cur.fetchall():
                item = {**data, "id": item_id}
                if (
                    not item.get("deleted")
                    and not item.get("completed")
                    and item.get("conversation_id")
                    and not item.get("source")
                ):
                    items_to_move.append(item)

    if not items_to_move:
        return {"moved": 0}

    # Move items atomically
    with db.batch() as conn:
        with conn.cursor() as cur:
            for item in items_to_move:
                item["source"] = "conversation_migration"
                cur.execute(
                    "INSERT INTO staged_tasks (uid, id, created_at, data) VALUES (%s, %s, %s, %s)",
                    (uid, item["id"], item.get("created_at"), json.dumps(item)),
                )
                cur.execute(
                    "DELETE FROM action_items WHERE uid = %s AND id = %s",
                    (uid, item["id"]),
                )

    return {"moved": len(items_to_move)}
