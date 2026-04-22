"""Focus sessions — focus/distraction tracking and statistics.

Table: focus_sessions (per-user)
- uid        TEXT NOT NULL
- id         TEXT NOT NULL
- created_at TIMESTAMPTZ NOT NULL DEFAULT now()
- data       JSONB NOT NULL DEFAULT '{}'::jsonb
- PRIMARY KEY (uid, id)
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from ._client import db

logger = logging.getLogger(__name__)


def create_focus_session(uid: str, status: str, app_or_site: str, description: str, **kwargs) -> dict:
    """Insert a focus session with generated UUID. Stores whole doc in data JSONB."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc = {
        'id': session_id,
        'status': status,
        'app_or_site': app_or_site,
        'description': description,
        'message': kwargs.get('message'),
        'created_at': now.isoformat(),
        'duration_seconds': kwargs.get('duration_seconds'),
    }
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO focus_sessions (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                """,
                (uid, session_id, json.dumps(doc)),
            )
    return doc


def get_focus_sessions(uid: str, date: str = None, limit: int = 100, offset: int = 0) -> List[dict]:
    """Get focus sessions for a user, optionally filtered by date.

    If date is provided (YYYY-MM-DD format), returns sessions where the data
    'created_at' falls within that calendar day (UTC).

    Returns sessions ordered by created_at DESC.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            if date:
                # Parse the date and compute day range in UTC
                day_start = datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)
                cur.execute(
                    """
                    SELECT uid, id, created_at, data
                    FROM focus_sessions
                    WHERE uid = %s
                      AND created_at >= %s
                      AND created_at < %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (uid, day_start, day_end, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT uid, id, created_at, data
                    FROM focus_sessions
                    WHERE uid = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (uid, limit, offset),
                )
            items = []
            for row in cur.fetchall():
                row_uid, row_id, row_created_at, row_data = row
                # Merge data JSONB into a dict so callers see the full document
                result = dict(row_data or {})
                result.setdefault('id', row_id)
                result.setdefault('uid', row_uid)
                result.setdefault('created_at', row_created_at.isoformat() if row_created_at else None)
                items.append(result)
            return items


def delete_focus_session(uid: str, session_id: str) -> bool:
    """Delete a focus session. Returns True if a row was deleted, False otherwise."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM focus_sessions
                WHERE uid = %s AND id = %s
                """,
                (uid, session_id),
            )
            return cur.rowcount > 0


def get_focus_stats(uid: str, date: str = None) -> dict:
    """Aggregate focus session statistics for a user.

    If date is provided, returns stats for that calendar day.
    Returns count, total durations, and top distractions.
    """
    sessions = get_focus_sessions(uid, date=date, limit=5000, offset=0)
    focused_count = 0
    distracted_count = 0
    total_focus_seconds = 0
    total_distracted_seconds = 0
    distractions: Dict[str, Dict[str, Any]] = {}

    for s in sessions:
        if s.get('status') == 'focused':
            focused_count += 1
            total_focus_seconds += s.get('duration_seconds') or 0
        elif s.get('status') == 'distracted':
            distracted_count += 1
            total_distracted_seconds += s.get('duration_seconds') or 60
            app = s.get('app_or_site', 'Unknown')
            entry = distractions.setdefault(app, {'total_seconds': 0, 'count': 0})
            entry['total_seconds'] += s.get('duration_seconds') or 60
            entry['count'] += 1

    top = sorted(distractions.items(), key=lambda x: x[1]['total_seconds'], reverse=True)[:5]

    return {
        'date': date or datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'focused_minutes': total_focus_seconds // 60,
        'distracted_minutes': total_distracted_seconds // 60,
        'session_count': focused_count + distracted_count,
        'focused_count': focused_count,
        'distracted_count': distracted_count,
        'top_distractions': [
            {'app_or_site': app, 'total_seconds': v['total_seconds'], 'count': v['count']} for app, v in top
        ],
    }
