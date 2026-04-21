"""Postgres-backed task tracking. Global collection (not per-user)."""

import json
from typing import Any, Dict, Optional

from ._client import db


def create(task_data: dict):
    """Insert a task document keyed by its 'id' field.

    Promotes 'action' and 'request_id' into typed columns for fast lookup;
    stores the whole dict into the `data` JSONB column.
    """
    task_id = task_data["id"]
    action = task_data.get("action")
    request_id = task_data.get("request_id")
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tasks (id, action, request_id, data)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                    SET action = EXCLUDED.action,
                        request_id = EXCLUDED.request_id,
                        data = EXCLUDED.data
                """,
                (task_id, action, request_id, json.dumps(task_data)),
            )


def update(task_id: str, task_data: dict):
    """Shallow-merge task_data into the existing row's data JSONB.

    If 'action' or 'request_id' keys are present, also update the typed columns.
    """
    action = task_data.get("action")  # None if not present
    request_id = task_data.get("request_id")
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks
                SET data = data || %s::jsonb,
                    action = COALESCE(%s, action),
                    request_id = COALESCE(%s, request_id)
                WHERE id = %s
                """,
                (json.dumps(task_data), action, request_id, task_id),
            )


def get_task_by_action_request(action: str, request_id: str) -> Optional[Dict[str, Any]]:
    """Return the first task matching (action, request_id) or None."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, action, request_id, data
                FROM tasks
                WHERE action = %s AND request_id = %s
                LIMIT 1
                """,
                (action, request_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, row_action, row_request_id, row_data = row
            # Merge typed columns into the dict so callers see the full document.
            result = dict(row_data or {})
            result.setdefault("id", row_id)
            result.setdefault("action", row_action)
            result.setdefault("request_id", row_request_id)
            return result
