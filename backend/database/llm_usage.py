"""
LLM Usage Database Operations.

Stores and queries LLM token usage by feature in Postgres.
Schema: llm_usage (per-user) — uid, id (date string), created_at, data (JSONB)
Table structure:
- uid        TEXT NOT NULL
- id         TEXT NOT NULL (date string like '2026-04-21')
- created_at TIMESTAMPTZ NOT NULL DEFAULT now()
- data       JSONB NOT NULL DEFAULT '{}'::jsonb
- PRIMARY KEY (uid, id)
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ._client import db


def record_llm_usage(
    uid: str,
    feature: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
):
    """
    Record LLM token usage for a user and feature.

    Uses Postgres JSONB merge for safe concurrent updates.

    Args:
        uid: User ID
        feature: Feature name (e.g., "chat", "rag", "conversation_processing")
        model: Model name (e.g., "gpt-4.1-mini", "o4-mini")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
    """
    if input_tokens == 0 and output_tokens == 0:
        return

    now = datetime.now(timezone.utc)
    doc_id = f"{now.year}-{now.month:02d}-{now.day:02d}"

    # Sanitize model name (Firestore constraint; carry forward for consistency)
    if not isinstance(model, str) or not model:
        model = "unknown"

    safe_model = (
        model.replace(".", "_")
        .replace("/", "_")
        .replace("~", "_")
        .replace("*", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("`", "_")
    )

    # Read current data, merge, then write atomically
    with db.batch() as conn:
        with conn.cursor() as cur:
            # Fetch existing row with row-level lock
            cur.execute(
                "SELECT data FROM llm_usage WHERE uid = %s AND id = %s FOR UPDATE",
                (uid, doc_id),
            )
            row = cur.fetchone()
            current_data = row[0] if row else {}

            # Merge new counters into current data
            if feature not in current_data:
                current_data[feature] = {}

            if safe_model not in current_data[feature]:
                current_data[feature][safe_model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "call_count": 0,
                }

            current_data[feature][safe_model]["input_tokens"] += input_tokens
            current_data[feature][safe_model]["output_tokens"] += output_tokens
            current_data[feature][safe_model]["call_count"] += 1

            # Update metadata
            current_data["date"] = doc_id
            current_data["last_updated"] = now.isoformat()

            # Upsert
            cur.execute(
                """
                INSERT INTO llm_usage (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                SET data = EXCLUDED.data
                """,
                (uid, doc_id, json.dumps(current_data)),
            )


def get_daily_usage(uid: str, date: Optional[datetime] = None) -> Dict:
    """
    Get LLM usage for a specific day.

    Args:
        uid: User ID
        date: Date to query (defaults to today)

    Returns:
        Dict with usage data by feature and model
    """
    if date is None:
        date = datetime.now(timezone.utc)

    doc_id = f"{date.year}-{date.month:02d}-{date.day:02d}"

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM llm_usage WHERE uid = %s AND id = %s",
                (uid, doc_id),
            )
            row = cur.fetchone()
            if row:
                return row[0] or {}
            return {}


def get_usage_summary(uid: str, days: int = 30) -> Dict:
    """
    Get aggregated LLM usage summary for the last N days.

    Args:
        uid: User ID
        days: Number of days to aggregate

    Returns:
        Dict with total usage by feature
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_id = f"{cutoff.year}-{cutoff.month:02d}-{cutoff.day:02d}"

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM llm_usage WHERE uid = %s AND id >= %s ORDER BY id DESC",
                (uid, cutoff_id),
            )
            rows = cur.fetchall()

    # Aggregate by feature
    summary: Dict[str, Dict[str, int]] = {}

    for row in rows:
        doc_id, data = row
        if data is None:
            continue

        for feature, models in data.items():
            if feature in ("date", "last_updated"):
                continue
            if not isinstance(models, dict):
                continue

            if feature not in summary:
                summary[feature] = {"input_tokens": 0, "output_tokens": 0, "call_count": 0}

            for model, tokens in models.items():
                if isinstance(tokens, dict):
                    summary[feature]["input_tokens"] += tokens.get("input_tokens", 0)
                    summary[feature]["output_tokens"] += tokens.get("output_tokens", 0)
                    summary[feature]["call_count"] += tokens.get("call_count", 0)

    return summary


def get_top_features(uid: str, days: int = 30, limit: int = 3) -> List[Dict]:
    """
    Get top features by total token usage.

    Args:
        uid: User ID
        days: Number of days to aggregate
        limit: Number of top features to return

    Returns:
        List of dicts with feature name and total tokens, sorted by usage
    """
    summary = get_usage_summary(uid, days)

    features = []
    for feature, tokens in summary.items():
        total = tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0)
        features.append(
            {
                "feature": feature,
                "input_tokens": tokens.get("input_tokens", 0),
                "output_tokens": tokens.get("output_tokens", 0),
                "total_tokens": total,
                "call_count": tokens.get("call_count", 0),
            }
        )

    features.sort(key=lambda x: x["total_tokens"], reverse=True)
    return features[:limit]


def get_global_top_features(days: int = 30, limit: int = 3) -> List[Dict]:
    """
    Get top features across all users by total token usage.

    Args:
        days: Number of days to aggregate
        limit: Number of top features to return

    Returns:
        List of dicts with feature name and total tokens
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_id = f"{cutoff.year}-{cutoff.month:02d}-{cutoff.day:02d}"

    # Query all users' llm_usage rows (no uid filter)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM llm_usage WHERE id >= %s",
                (cutoff_id,),
            )
            rows = cur.fetchall()

    global_summary: Dict[str, Dict[str, int]] = {}

    for row in rows:
        doc_id, data = row
        if data is None:
            continue

        for feature, models in data.items():
            if feature in ("date", "last_updated"):
                continue
            if not isinstance(models, dict):
                continue

            if feature not in global_summary:
                global_summary[feature] = {"input_tokens": 0, "output_tokens": 0, "call_count": 0}

            for model, tokens in models.items():
                if isinstance(tokens, dict):
                    global_summary[feature]["input_tokens"] += tokens.get("input_tokens", 0)
                    global_summary[feature]["output_tokens"] += tokens.get("output_tokens", 0)
                    global_summary[feature]["call_count"] += tokens.get("call_count", 0)

    features = []
    for feature, tokens in global_summary.items():
        total = tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0)
        features.append(
            {
                "feature": feature,
                "input_tokens": tokens.get("input_tokens", 0),
                "output_tokens": tokens.get("output_tokens", 0),
                "total_tokens": total,
                "call_count": tokens.get("call_count", 0),
            }
        )

    features.sort(key=lambda x: x["total_tokens"], reverse=True)
    return features[:limit]


# ============================================================================
# BUCKET-BASED LLM USAGE
#
# Flat key scheme ("desktop_chat" / "desktop_chat_{account}") with fields:
# input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
# total_tokens, cost_usd, call_count.
#
# This differs from the {feature}.{model} nesting above.  Both schemas
# coexist in the same date-keyed documents using Postgres's schemaless JSONB design.
# ============================================================================


def record_llm_usage_bucket(
    uid: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
    bucket: str = 'desktop_chat',
    account: str = 'omi',
) -> None:
    """Record LLM token usage into a flat bucket with atomic increments.

    Dual-writes to both the primary bucket and a per-account alias
    (``{bucket}_{account}``) for per-account breakdown.
    """
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    # Read current data, merge, then write atomically
    with db.batch() as conn:
        with conn.cursor() as cur:
            # Fetch existing row with row-level lock
            cur.execute(
                "SELECT data FROM llm_usage WHERE uid = %s AND id = %s FOR UPDATE",
                (uid, today),
            )
            row = cur.fetchone()
            current_data = row[0] if row else {}

            # Merge bucket data
            acct_key = f'{bucket}_{account}'

            for key in [bucket, acct_key]:
                if key not in current_data:
                    current_data[key] = {}

                current_data[key]["input_tokens"] = current_data[key].get("input_tokens", 0) + input_tokens
                current_data[key]["output_tokens"] = current_data[key].get("output_tokens", 0) + output_tokens
                current_data[key]["cache_read_tokens"] = current_data[key].get("cache_read_tokens", 0) + cache_read_tokens
                current_data[key]["cache_write_tokens"] = current_data[key].get("cache_write_tokens", 0) + cache_write_tokens
                current_data[key]["total_tokens"] = current_data[key].get("total_tokens", 0) + total_tokens
                current_data[key]["cost_usd"] = current_data[key].get("cost_usd", 0.0) + cost_usd
                current_data[key]["call_count"] = current_data[key].get("call_count", 0) + 1

            # Update metadata
            current_data["date"] = today
            current_data["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Upsert
            cur.execute(
                """
                INSERT INTO llm_usage (uid, id, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, id) DO UPDATE
                SET data = EXCLUDED.data
                """,
                (uid, today, json.dumps(current_data)),
            )


def get_total_llm_cost(uid: str, bucket: str = 'desktop_chat') -> float:
    """Sum cost_usd from the given bucket.

    When the bucket dual-writes to both ``{bucket}`` and ``{bucket}_{account}``,
    this reads only the primary bucket to avoid double-counting.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, data FROM llm_usage WHERE uid = %s",
                (uid,),
            )
            rows = cur.fetchall()

    total = 0.0
    for row in rows:
        doc_id, data = row
        if data is None:
            continue
        bucket_data = data.get(bucket)
        if isinstance(bucket_data, dict):
            total += bucket_data.get('cost_usd', 0.0)

    return round(total, 6)
