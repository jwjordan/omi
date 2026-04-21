"""Postgres-backed calendar meetings. Per-user collection."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ._client import db


def create_meeting(uid: str, meeting_data: Dict) -> str:
    """
    Create a new calendar meeting in Postgres.
    Returns the meeting ID (UUID).

    NOTE: Times should already be in UTC before calling this function.
    """
    from datetime import timezone

    # Generate ID if not provided
    meeting_id = meeting_data.get("id", str(uuid.uuid4()))

    # Add timestamps (always in UTC for consistent querying)
    now = datetime.now(timezone.utc)
    data_to_store = dict(meeting_data)
    data_to_store["id"] = meeting_id

    # Convert datetime objects to ISO strings for JSON serialization
    for key, value in data_to_store.items():
        if isinstance(value, datetime):
            data_to_store[key] = value.isoformat()

    data_to_store["created_at"] = now.isoformat()
    data_to_store["synced_at"] = now.isoformat()

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO calendar_meetings (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                    SET data = EXCLUDED.data
                """,
                (uid, meeting_id, json.dumps(data_to_store)),
            )
            # Return the generated ID
            cur.execute("SELECT %s", (meeting_id,))
            result = cur.fetchone()
            return result[0] if result else meeting_id


def update_meeting(uid: str, meeting_id: str, meeting_data: Dict) -> None:
    """
    Update an existing calendar meeting.

    NOTE: Times should already be in UTC before calling this function.
    """
    from datetime import timezone

    data_to_merge = dict(meeting_data)

    # Convert datetime objects to ISO strings for JSON serialization
    for key, value in data_to_merge.items():
        if isinstance(value, datetime):
            data_to_merge[key] = value.isoformat()

    data_to_merge["synced_at"] = datetime.now(timezone.utc).isoformat()

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE calendar_meetings
                SET data = data || %s::jsonb
                WHERE uid=%s AND id=%s
                """,
                (json.dumps(data_to_merge), uid, meeting_id),
            )


def get_meeting(uid: str, meeting_id: str) -> Optional[Dict]:
    """Get a calendar meeting by its ID"""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uid, id, created_at, data
                FROM calendar_meetings
                WHERE uid=%s AND id=%s
                """,
                (uid, meeting_id),
            )
            row = cur.fetchone()
            if row is None:
                return None

            row_uid, row_id, row_created_at, row_data = row
            # Merge typed columns into the dict so callers see the full document.
            result = dict(row_data or {})
            result.setdefault("id", row_id)
            result.setdefault("uid", row_uid)
            result.setdefault("created_at", row_created_at.isoformat() if row_created_at else None)
            return result


def get_meeting_id_by_calendar_event(uid: str, calendar_event_id: str, calendar_source: str) -> Optional[str]:
    """
    Find a meeting by its external calendar event ID and source.
    Returns the meeting ID if found, None otherwise.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM calendar_meetings
                WHERE uid=%s
                  AND data->>'calendar_event_id' = %s
                  AND data->>'calendar_source' = %s
                LIMIT 1
                """,
                (uid, calendar_event_id, calendar_source),
            )
            row = cur.fetchone()
            return row[0] if row else None


def list_meetings(
    uid: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, limit: int = 50
) -> List[Dict]:
    """
    List calendar meetings, optionally filtered by date range.
    Returns meetings sorted by start_time descending.
    """
    query = """
        SELECT uid, id, created_at, data
        FROM calendar_meetings
        WHERE uid=%s
    """
    params = [uid]

    if start_date:
        query += """
            AND (data->>'start_time')::timestamptz >= %s
        """
        params.append(start_date)

    if end_date:
        query += """
            AND (data->>'start_time')::timestamptz <= %s
        """
        params.append(end_date)

    query += """
        ORDER BY (data->>'start_time')::timestamptz DESC
        LIMIT %s
    """
    params.append(limit)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            meetings = []
            for row in cur.fetchall():
                row_uid, row_id, row_created_at, row_data = row
                result = dict(row_data or {})
                result.setdefault("id", row_id)
                result.setdefault("uid", row_uid)
                result.setdefault("created_at", row_created_at.isoformat() if row_created_at else None)
                meetings.append(result)
            return meetings


def delete_meeting(uid: str, meeting_id: str) -> None:
    """Delete a calendar meeting"""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM calendar_meetings
                WHERE uid=%s AND id=%s
                """,
                (uid, meeting_id),
            )


def delete_old_meetings(uid: str, before_date: datetime) -> int:
    """
    Delete meetings that ended before a certain date.
    Returns the number of meetings deleted.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM calendar_meetings
                WHERE uid=%s
                  AND (data->>'end_time')::timestamptz < %s
                """,
                (uid, before_date),
            )
            return cur.rowcount


def get_meetings_in_time_range(uid: str, start_time: datetime, end_time: datetime) -> List[Dict]:
    """
    Find meetings that overlap with the given time range.
    A meeting overlaps if: meeting.start_time < range.end_time AND meeting.end_time > range.start_time

    Returns meetings sorted by start_time ascending.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uid, id, created_at, data
                FROM calendar_meetings
                WHERE uid=%s
                  AND (data->>'start_time')::timestamptz < %s
                  AND (data->>'end_time')::timestamptz > %s
                ORDER BY (data->>'start_time')::timestamptz ASC
                LIMIT 10
                """,
                (uid, end_time, start_time),
            )
            meetings = []
            for row in cur.fetchall():
                row_uid, row_id, row_created_at, row_data = row
                result = dict(row_data or {})
                result.setdefault("id", row_id)
                result.setdefault("uid", row_uid)
                result.setdefault("created_at", row_created_at.isoformat() if row_created_at else None)
                meetings.append(result)
            return meetings
