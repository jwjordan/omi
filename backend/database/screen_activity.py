from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import json
import logging

from ._client import db

logger = logging.getLogger(__name__)


def upsert_screen_activity(uid: str, rows: List[Dict[str, Any]]) -> int:
    """Batch write screen activity rows to Postgres users/{uid}/screen_activity/{id}."""
    if not rows:
        return 0

    written = 0

    # Process in batches of 500 (Firestore compat)
    for i in range(0, len(rows), 500):
        chunk = rows[i : i + 500]
        with db.batch() as conn:
            with conn.cursor() as cur:
                for row in chunk:
                    doc_id = str(row['id'])
                    doc_data = {
                        'timestamp': row['timestamp'],
                        'appName': row.get('appName', ''),
                        'windowTitle': row.get('windowTitle', ''),
                        'ocrText': (row.get('ocrText') or '')[:1000],
                    }
                    cur.execute(
                        """
                        INSERT INTO screen_activity (uid, id, data)
                        VALUES (%s, %s, %s::jsonb)
                        ON CONFLICT (uid, id) DO UPDATE
                        SET data = EXCLUDED.data
                        """,
                        (uid, doc_id, json.dumps(doc_data)),
                    )
                written += len(chunk)

    return written


def get_screen_activity(
    uid: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    app_filter: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Query screen activity by date range with optional app filter."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            # Build WHERE clause
            where_parts = ["uid = %s"]
            params = [uid]

            if start_date:
                where_parts.append("created_at >= %s")
                params.append(start_date)
            if end_date:
                where_parts.append("created_at <= %s")
                params.append(end_date)
            if app_filter:
                where_parts.append("data->>'appName' = %s")
                params.append(app_filter)

            where_clause = " AND ".join(where_parts)

            # Build full query
            sql = f"""
                SELECT id, data
                FROM screen_activity
                WHERE {where_clause}
                ORDER BY created_at ASC
                LIMIT %s
            """
            params.append(limit)

            cur.execute(sql, params)
            results = []
            for row in cur.fetchall():
                screen_id, row_data = row
                result = dict(row_data or {})
                result['id'] = screen_id
                results.append(result)

            return results


def get_screen_activity_summary(
    uid: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Get aggregated app usage summary — groups by appName, counts screenshots, estimates time."""
    rows = get_screen_activity(uid, start_date=start_date, end_date=end_date, limit=5000)

    if not rows:
        return {'apps': {}, 'total_screenshots': 0}

    apps: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        app_name = row.get('appName') or 'Unknown'
        if app_name not in apps:
            apps[app_name] = {
                'count': 0,
                'first_seen': row.get('timestamp'),
                'last_seen': row.get('timestamp'),
                'window_titles': set(),
            }
        apps[app_name]['count'] += 1
        apps[app_name]['last_seen'] = row.get('timestamp')
        title = row.get('windowTitle', '')
        if title:
            apps[app_name]['window_titles'].add(title)

    # Convert sets to lists for serialization
    for app_name in apps:
        titles = apps[app_name]['window_titles']
        apps[app_name]['window_titles'] = list(titles)[:10]  # Top 10 titles

    return {
        'apps': apps,
        'total_screenshots': len(rows),
    }
