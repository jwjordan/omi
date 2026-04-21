"""Postgres-backed daily summaries — per-user summaries indexed by date.

Table: daily_summaries (per-user, keyed by uid + id)
    uid TEXT NOT NULL
    id TEXT NOT NULL
    date DATE
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    data JSONB NOT NULL DEFAULT '{}'::jsonb
    PRIMARY KEY (uid, id)
    INDEX idx_daily_summaries_uid_date (uid, date DESC)

Fired fields (uid, id, date) are used for efficient lookups;
headline, overview, day_emoji, highlights, etc. live in data JSONB.
"""

import json
from typing import List, Optional

from ._client import db


def create_daily_summary(uid: str, summary_data: dict) -> str:
    """Insert a new daily summary. Returns the summary ID.

    Promotes uid, id, and date into typed columns for fast lookup;
    stores the whole dict into the `data` JSONB column.
    """
    summary_id = summary_data["id"]
    date_str = summary_data.get("date")
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO daily_summaries (uid, id, date, data)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET date = EXCLUDED.date,
                        data = EXCLUDED.data
                """,
                (uid, summary_id, date_str, json.dumps(summary_data)),
            )
    return summary_id


def get_daily_summary(uid: str, summary_id: str) -> Optional[dict]:
    """Get a single daily summary by ID.

    Returns summary data dict or None if not found.
    Merges typed columns (id, date) into the returned dict.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, date, data
                FROM daily_summaries
                WHERE uid = %s AND id = %s
                """,
                (uid, summary_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, row_date, row_data = row
            result = dict(row_data or {})
            result.setdefault("id", row_id)
            if row_date:
                # Handle both date objects and string dates
                date_str = row_date.isoformat() if hasattr(row_date, 'isoformat') else str(row_date)
                result.setdefault("date", date_str)
            return result


def get_daily_summary_by_date(uid: str, date: str) -> Optional[dict]:
    """Get a daily summary by date (YYYY-MM-DD format).

    Returns summary data dict or None if not found.
    Uses the typed date column for efficient lookup.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, date, data
                FROM daily_summaries
                WHERE uid = %s AND date = %s
                LIMIT 1
                """,
                (uid, date),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, row_date, row_data = row
            result = dict(row_data or {})
            result.setdefault("id", row_id)
            if row_date:
                # Handle both date objects and string dates
                date_str = row_date.isoformat() if hasattr(row_date, 'isoformat') else str(row_date)
                result.setdefault("date", date_str)
            return result


def get_daily_summaries(
    uid: str,
    limit: int = 30,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    """Get list of daily summaries for a user, ordered by date descending.

    Args:
        uid: User ID
        limit: Maximum number of summaries to return
        offset: Number of summaries to skip
        start_date: Filter summaries from this date (YYYY-MM-DD)
        end_date: Filter summaries until this date (YYYY-MM-DD)

    Returns:
        List of summary data dicts
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Build WHERE clause
            where_parts = ["uid = %s"]
            params = [uid]

            if start_date:
                where_parts.append("date >= %s")
                params.append(start_date)
            if end_date:
                where_parts.append("date <= %s")
                params.append(end_date)

            where_clause = " AND ".join(where_parts)

            # Build full query
            sql = f"""
                SELECT id, date, data
                FROM daily_summaries
                WHERE {where_clause}
                ORDER BY date DESC
                LIMIT %s
                OFFSET %s
            """
            params.extend([limit, offset])

            cur.execute(sql, params)
            items = []
            for row in cur.fetchall():
                row_id, row_date, row_data = row
                result = dict(row_data or {})
                result.setdefault("id", row_id)
                if row_date:
                    # Handle both date objects and string dates
                    date_str = row_date.isoformat() if hasattr(row_date, 'isoformat') else str(row_date)
                    result.setdefault("date", date_str)
                items.append(result)
            return items


def delete_daily_summary(uid: str, summary_id: str) -> bool:
    """Delete a daily summary. Returns True if deleted, False if not found."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM daily_summaries
                WHERE uid = %s AND id = %s
                """,
                (uid, summary_id),
            )
            return cur.rowcount > 0


def get_summaries_count(uid: str) -> int:
    """Get total count of daily summaries for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM daily_summaries
                WHERE uid = %s
                """,
                (uid,),
            )
            row = cur.fetchone()
            return row[0] if row else 0
