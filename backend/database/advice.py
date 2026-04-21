"""Advice — proactive coaching items.

Table: advice (per-user, keyed by uid + id)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from ._client import db

logger = logging.getLogger(__name__)


def create_advice(uid: str, content: str, category: str = "other", **kwargs) -> dict:
    """Insert a new advice item. Returns the created document."""
    advice_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    data = {
        "id": advice_id,
        "content": content,
        "category": category,
        "reasoning": kwargs.get("reasoning"),
        "source_app": kwargs.get("source_app"),
        "confidence": kwargs.get("confidence", 0.5),
        "context_summary": kwargs.get("context_summary"),
        "current_activity": kwargs.get("current_activity"),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "is_read": False,
        "is_dismissed": False,
    }
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO advice (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                """,
                (uid, advice_id, json.dumps(data)),
            )
    return data


def get_advice(
    uid: str, category: str = None, limit: int = 50, offset: int = 0, include_dismissed: bool = False
) -> List[dict]:
    """Query advice items for a user. Returns list of documents."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Build WHERE clause
            where_parts = ["uid = %s"]
            params = [uid]

            if category:
                where_parts.append("data->>'category' = %s")
                params.append(category)

            if not include_dismissed:
                where_parts.append("data->>'is_dismissed' != 'true'")

            where_clause = " AND ".join(where_parts)

            # Build full query
            sql = f"""
                SELECT id, data
                FROM advice
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
                OFFSET %s
            """
            params.extend([limit, offset])

            cur.execute(sql, params)
            items = []
            for row in cur.fetchall():
                advice_id, row_data = row
                result = dict(row_data or {})
                result["id"] = advice_id
                items.append(result)
            return items


def update_advice(uid: str, advice_id: str, is_read: bool = None, is_dismissed: bool = None) -> Optional[dict]:
    """Update an advice item. Returns updated document or None if not found."""
    # First check if it exists
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data
                FROM advice
                WHERE uid = %s AND id = %s
                """,
                (uid, advice_id),
            )
            row = cur.fetchone()
            if row is None:
                return None

            # Build the update data
            updates = {}
            if is_read is not None:
                updates["is_read"] = is_read
            if is_dismissed is not None:
                updates["is_dismissed"] = is_dismissed

            updates["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Update the row
            cur.execute(
                """
                UPDATE advice
                SET data = data || %s::jsonb
                WHERE uid = %s AND id = %s
                RETURNING data
                """,
                (json.dumps(updates), uid, advice_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            result = dict(row[0] or {})
            result["id"] = advice_id
            return result


def delete_advice(uid: str, advice_id: str) -> bool:
    """Delete an advice item. Returns True if deleted, False if not found."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM advice
                WHERE uid = %s AND id = %s
                """,
                (uid, advice_id),
            )
            return cur.rowcount > 0


def mark_all_advice_read(uid: str) -> int:
    """Mark all unread advice as read for a user. Returns count of updated rows."""
    with db.batch() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE advice
                SET data = jsonb_set(data, '{is_read}', 'true'::jsonb)
                WHERE uid = %s AND data->>'is_read' != 'true'
                """,
                (uid,),
            )
            return cur.rowcount
