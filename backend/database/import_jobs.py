"""Postgres-backed import job tracking. Global collection (not per-user)."""

import json
from typing import List, Optional

from ._client import db


def create_import_job(job_data: dict) -> str:
    """Insert an import job document keyed by its 'id' field.

    Stores the whole dict into the `data` JSONB column.
    Also stores 'uid' as a typed column for fast filtering.
    """
    job_id = job_data["id"]
    uid = job_data.get("uid")
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO import_jobs (id, uid, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                    SET uid = EXCLUDED.uid,
                        data = EXCLUDED.data
                """,
                (job_id, uid, json.dumps(job_data)),
            )
    return job_id


def update_import_job(job_id: str, updates: dict) -> None:
    """Shallow-merge updates into the existing row's data JSONB.

    If 'uid' key is present, also update the typed column.
    """
    uid = updates.get("uid")
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE import_jobs
                SET data = data || %s::jsonb,
                    uid = COALESCE(%s, uid)
                WHERE id = %s
                """,
                (json.dumps(updates), uid, job_id),
            )


def get_import_job(job_id: str) -> Optional[dict]:
    """Get a single import job by ID."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, uid, data
                FROM import_jobs
                WHERE id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, row_uid, row_data = row
            # Merge typed columns into the dict so callers see the full document.
            result = dict(row_data or {})
            result.setdefault("id", row_id)
            result.setdefault("uid", row_uid)
            return result


def get_import_jobs(uid: str, limit: int = 50) -> List[dict]:
    """Get all import jobs for a user, ordered by created_at descending."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, uid, data
                FROM import_jobs
                WHERE uid = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (uid, limit),
            )
            rows = cur.fetchall()
            result = []
            for row in rows:
                row_id, row_uid, row_data = row
                job_dict = dict(row_data or {})
                job_dict.setdefault("id", row_id)
                job_dict.setdefault("uid", row_uid)
                result.append(job_dict)
            return result


def delete_import_job(job_id: str) -> None:
    """Delete an import job."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM import_jobs
                WHERE id = %s
                """,
                (job_id,),
            )
