"""User usage database operations — Postgres impl.

Replaces the Firestore-backed hourly_usage subcollection with two tables:

- ``user_hourly_usage`` (uid, hour_key, data JSONB, updated_at) — one row per
  (uid, 'YYYY-MM-DD-HH'). Counter updates merge into ``data`` via JSONB
  arithmetic so concurrent writes don't clobber each other.
- ``llm_usage`` — read-only here; monthly chat aggregates are computed from
  the shared per-day ``data`` JSONB column. Writes are handled by
  ``database/llm_usage.py``.

All public signatures match the previous Firestore version so call sites stay
unchanged.
"""

import json
from calendar import monthrange
from datetime import datetime, timezone
from typing import Optional

from ._client import db
from models.user_usage import UsageStats


# ---------------------------------------------------------------------------
# LLM-usage-backed monthly chat counter
# ---------------------------------------------------------------------------


def get_monthly_chat_usage(uid: str, now: Optional[datetime] = None) -> dict:
    """Sum current-month chat usage from ``llm_usage`` rows.

    Returns keys:
      - questions: total user-initiated chat calls (desktop_chat* + backend `chat.*`)
      - cost_usd:  total desktop_chat* cost_usd (backend GPT/Gemini chat has no cost field)
      - reset_at:  unix seconds of the start of next UTC month (when the bucket resets)

    Proactive, memory-extraction, knowledge-graph, conversation-processing etc. are
    excluded on purpose — those are company-driven, not user-initiated questions.
    """
    now = now or datetime.now(timezone.utc)
    month_pattern = f'{now.year}-{now.month:02d}-%'

    questions = 0
    cost_usd = 0.0

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM llm_usage WHERE uid = %s AND id LIKE %s",
                (uid, month_pattern),
            )
            rows = cur.fetchall()

    for row in rows:
        data = row[0] or {}
        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith('desktop_chat'):
                if key.endswith('.call_count'):
                    questions += int(value)
                elif key.endswith('.cost_usd'):
                    cost_usd += float(value)
            elif key.startswith('chat.') and key.endswith('.call_count'):
                # user-initiated backend chat (any model)
                questions += int(value)

    # Compute end-of-month boundary in UTC for the reset timestamp.
    # monthrange is kept for parity with the previous impl even though we
    # don't use last_day directly any more.
    _ = monthrange(now.year, now.month)[1]
    if now.month == 12:
        next_year, next_month = now.year + 1, 1
    else:
        next_year, next_month = now.year, now.month + 1
    reset_at = int(datetime(next_year, next_month, 1, tzinfo=timezone.utc).timestamp())

    return {
        'questions': questions,
        'cost_usd': round(cost_usd, 4),
        'reset_at': reset_at,
    }


# ---------------------------------------------------------------------------
# Hourly usage writes
# ---------------------------------------------------------------------------

_INCREMENTABLE_KEYS = (
    'transcription_seconds',
    'words_transcribed',
    'insights_gained',
    'memories_created',
    'speech_seconds',
)


def _hour_key(date: datetime) -> str:
    return f'{date.year}-{date.month:02d}-{date.day:02d}-{date.hour:02d}'


def _build_increment_payload(updates: dict, platform: Optional[str] = None) -> Optional[dict]:
    """Return the JSONB payload to merge into ``user_hourly_usage.data`` or ``None``
    if there's nothing to write.
    """
    payload: dict = {}
    for key, value in updates.items():
        if key in _INCREMENTABLE_KEYS and isinstance(value, (int, float)) and value > 0:
            payload[key] = value

    if not payload:
        return None

    if platform in ('desktop', 'mobile'):
        payload['platforms'] = [platform]

    return payload


def update_hourly_usage(
    uid: str, date: datetime, updates: dict, platform: Optional[str] = None
):
    """Upsert a single hour's usage counters into ``user_hourly_usage``.

    New counters are added to any existing values by summing the incoming
    JSONB with the stored JSONB on conflict. Platforms are merged as a
    unique-element JSON array.
    """
    payload = _build_increment_payload(updates, platform)
    if payload is None:
        return

    hour_key = _hour_key(date)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_hourly_usage (uid, hour_key, data)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (uid, hour_key) DO UPDATE
                SET data = jsonb_build_object(
                        'transcription_seconds',
                            COALESCE((user_hourly_usage.data->>'transcription_seconds')::bigint, 0)
                          + COALESCE((EXCLUDED.data->>'transcription_seconds')::bigint, 0),
                        'words_transcribed',
                            COALESCE((user_hourly_usage.data->>'words_transcribed')::bigint, 0)
                          + COALESCE((EXCLUDED.data->>'words_transcribed')::bigint, 0),
                        'insights_gained',
                            COALESCE((user_hourly_usage.data->>'insights_gained')::bigint, 0)
                          + COALESCE((EXCLUDED.data->>'insights_gained')::bigint, 0),
                        'memories_created',
                            COALESCE((user_hourly_usage.data->>'memories_created')::bigint, 0)
                          + COALESCE((EXCLUDED.data->>'memories_created')::bigint, 0),
                        'speech_seconds',
                            COALESCE((user_hourly_usage.data->>'speech_seconds')::bigint, 0)
                          + COALESCE((EXCLUDED.data->>'speech_seconds')::bigint, 0),
                        'platforms',
                            (
                                SELECT COALESCE(jsonb_agg(DISTINCT p), '[]'::jsonb)
                                FROM jsonb_array_elements_text(
                                    COALESCE(user_hourly_usage.data->'platforms', '[]'::jsonb)
                                    || COALESCE(EXCLUDED.data->'platforms', '[]'::jsonb)
                                ) AS p
                            )
                    ),
                    updated_at = now()
                """,
                (uid, hour_key, json.dumps(payload)),
            )


def batch_update_hourly_usage(uid: str, hourly_updates: dict):
    """Upsert a batch of hour buckets atomically."""
    if not hourly_updates:
        return

    items = list(hourly_updates.items())
    batch_size = 400

    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        with db.batch() as conn:
            with conn.cursor() as cur:
                for date, updates in chunk:
                    payload = _build_increment_payload(updates)
                    if payload is None:
                        continue
                    hour_key = _hour_key(date)
                    cur.execute(
                        """
                        INSERT INTO user_hourly_usage (uid, hour_key, data)
                        VALUES (%s, %s, %s::jsonb)
                        ON CONFLICT (uid, hour_key) DO UPDATE
                        SET data = jsonb_build_object(
                                'transcription_seconds',
                                    COALESCE((user_hourly_usage.data->>'transcription_seconds')::bigint, 0)
                                  + COALESCE((EXCLUDED.data->>'transcription_seconds')::bigint, 0),
                                'words_transcribed',
                                    COALESCE((user_hourly_usage.data->>'words_transcribed')::bigint, 0)
                                  + COALESCE((EXCLUDED.data->>'words_transcribed')::bigint, 0),
                                'insights_gained',
                                    COALESCE((user_hourly_usage.data->>'insights_gained')::bigint, 0)
                                  + COALESCE((EXCLUDED.data->>'insights_gained')::bigint, 0),
                                'memories_created',
                                    COALESCE((user_hourly_usage.data->>'memories_created')::bigint, 0)
                                  + COALESCE((EXCLUDED.data->>'memories_created')::bigint, 0),
                                'speech_seconds',
                                    COALESCE((user_hourly_usage.data->>'speech_seconds')::bigint, 0)
                                  + COALESCE((EXCLUDED.data->>'speech_seconds')::bigint, 0),
                                'platforms',
                                    (
                                        SELECT COALESCE(jsonb_agg(DISTINCT p), '[]'::jsonb)
                                        FROM jsonb_array_elements_text(
                                            COALESCE(user_hourly_usage.data->'platforms', '[]'::jsonb)
                                            || COALESCE(EXCLUDED.data->'platforms', '[]'::jsonb)
                                        ) AS p
                                    )
                            ),
                            updated_at = now()
                        """,
                        (uid, hour_key, json.dumps(payload)),
                    )


# ---------------------------------------------------------------------------
# Aggregate reads
# ---------------------------------------------------------------------------


def _empty_stats() -> dict:
    return {
        'transcription_seconds': 0,
        'words_transcribed': 0,
        'insights_gained': 0,
        'memories_created': 0,
        'speech_seconds': 0,
    }


def _aggregate_stats(rows) -> dict:
    """Sum stat fields across a list of ``(data,)`` rows from ``user_hourly_usage``."""
    stats = _empty_stats()
    for row in rows:
        data = row[0] if row else None
        if not data:
            continue
        stats['transcription_seconds'] += int(data.get('transcription_seconds', 0) or 0)
        stats['words_transcribed'] += int(data.get('words_transcribed', 0) or 0)
        stats['insights_gained'] += int(data.get('insights_gained', 0) or 0)
        stats['memories_created'] += int(data.get('memories_created', 0) or 0)
        stats['speech_seconds'] += int(data.get('speech_seconds', 0) or 0)
    return stats


def _fetch_hourly_rows(uid: str, where_sql: str, params: tuple) -> list:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT data FROM user_hourly_usage WHERE uid = %s AND {where_sql}",
                (uid, *params),
            )
            return cur.fetchall()


def _fetch_hourly_rows_with_hour(uid: str, where_sql: str, params: tuple) -> list:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT hour_key, data FROM user_hourly_usage WHERE uid = %s AND {where_sql}",
                (uid, *params),
            )
            return cur.fetchall()


def get_today_usage_stats(uid: str, date: datetime) -> dict:
    """Aggregates hourly usage stats for a given day."""
    day_pattern = f'{date.year}-{date.month:02d}-{date.day:02d}-%'
    rows = _fetch_hourly_rows(uid, "hour_key LIKE %s", (day_pattern,))
    return _aggregate_stats(rows)


def get_monthly_usage_stats(uid: str, date: datetime) -> dict:
    """Aggregates hourly usage stats for a given month."""
    month_pattern = f'{date.year}-{date.month:02d}-%'
    rows = _fetch_hourly_rows(uid, "hour_key LIKE %s", (month_pattern,))
    return _aggregate_stats(rows)


def get_monthly_usage_stats_since(uid: str, date: datetime, start_date: datetime) -> dict:
    """Aggregates hourly usage stats within ``date``'s month from ``start_date`` onward."""
    month_pattern = f'{date.year}-{date.month:02d}-%'
    start_key = f'{start_date.year}-{start_date.month:02d}-{start_date.day:02d}-00'
    rows = _fetch_hourly_rows(
        uid,
        "hour_key LIKE %s AND hour_key >= %s",
        (month_pattern, start_key),
    )
    return _aggregate_stats(rows)


def get_yearly_usage_stats(uid: str, date: datetime) -> dict:
    """Aggregates hourly usage stats for a given year."""
    year_pattern = f'{date.year}-%'
    rows = _fetch_hourly_rows(uid, "hour_key LIKE %s", (year_pattern,))
    return _aggregate_stats(rows)


def get_all_time_usage_stats(uid: str) -> dict:
    """Aggregates all hourly usage stats for a user."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM user_hourly_usage WHERE uid = %s", (uid,))
            rows = cur.fetchall()
    return _aggregate_stats(rows)


# ---------------------------------------------------------------------------
# History breakdowns
# ---------------------------------------------------------------------------


def _empty_history_stats() -> dict:
    return {
        'transcription_seconds': 0,
        'words_transcribed': 0,
        'insights_gained': 0,
        'memories_created': 0,
    }


def _accumulate_history(bucket: dict, data: dict) -> None:
    bucket['transcription_seconds'] += int(data.get('transcription_seconds', 0) or 0)
    bucket['words_transcribed'] += int(data.get('words_transcribed', 0) or 0)
    bucket['insights_gained'] += int(data.get('insights_gained', 0) or 0)
    bucket['memories_created'] += int(data.get('memories_created', 0) or 0)


def get_hourly_history_for_today(uid: str, date: datetime) -> list[dict]:
    """Gets hourly usage for a specific day."""
    day_pattern = f'{date.year}-{date.month:02d}-{date.day:02d}-%'
    rows = _fetch_hourly_rows_with_hour(uid, "hour_key LIKE %s", (day_pattern,))

    hourly_totals: dict = {}
    for hour_key, data in rows:
        if not data:
            continue
        # hour_key format: 'YYYY-MM-DD-HH'
        try:
            hour = int(hour_key.split('-')[-1])
        except (ValueError, AttributeError):
            continue
        if hour not in hourly_totals:
            hourly_totals[hour] = _empty_history_stats()
        _accumulate_history(hourly_totals[hour], data)

    history = [
        {'date': f"{date.year}-{date.month:02d}-{date.day:02d}T{hour:02d}:00:00Z", **stats}
        for hour, stats in hourly_totals.items()
    ]
    history.sort(key=lambda x: x['date'])
    return history


def get_daily_history_for_month(uid: str, date: datetime) -> list[dict]:
    """Gets daily usage for a specific month."""
    month_pattern = f'{date.year}-{date.month:02d}-%'
    rows = _fetch_hourly_rows_with_hour(uid, "hour_key LIKE %s", (month_pattern,))

    daily_totals: dict = {}
    for hour_key, data in rows:
        if not data:
            continue
        try:
            day = int(hour_key.split('-')[2])
        except (ValueError, AttributeError, IndexError):
            continue
        if day not in daily_totals:
            daily_totals[day] = _empty_history_stats()
        _accumulate_history(daily_totals[day], data)

    history = [
        {'date': f"{date.year}-{date.month:02d}-{day:02d}", **stats}
        for day, stats in daily_totals.items()
    ]
    history.sort(key=lambda x: x['date'])
    return history


def get_monthly_history_for_year(uid: str, date: datetime) -> list[dict]:
    """Gets monthly usage for a specific year."""
    year_pattern = f'{date.year}-%'
    rows = _fetch_hourly_rows_with_hour(uid, "hour_key LIKE %s", (year_pattern,))

    monthly_totals: dict = {}
    for hour_key, data in rows:
        if not data:
            continue
        try:
            month = int(hour_key.split('-')[1])
        except (ValueError, AttributeError, IndexError):
            continue
        if month not in monthly_totals:
            monthly_totals[month] = _empty_history_stats()
        _accumulate_history(monthly_totals[month], data)

    history = [
        {'date': f"{date.year}-{month:02d}-01", **stats}
        for month, stats in monthly_totals.items()
    ]
    history.sort(key=lambda x: x['date'])
    return history


def get_yearly_history(uid: str) -> list[dict]:
    """Gets yearly usage for all time."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hour_key, data FROM user_hourly_usage WHERE uid = %s",
                (uid,),
            )
            rows = cur.fetchall()

    yearly_totals: dict = {}
    for hour_key, data in rows:
        if not data:
            continue
        try:
            year = int(hour_key.split('-')[0])
        except (ValueError, AttributeError, IndexError):
            continue
        if year not in yearly_totals:
            yearly_totals[year] = _empty_history_stats()
        _accumulate_history(yearly_totals[year], data)

    history = [
        {'date': f"{year}-01-01", **stats} for year, stats in yearly_totals.items()
    ]
    history.sort(key=lambda x: x['date'])
    return history


def get_current_user_usage(uid: str, period: str) -> dict:
    """Gets usage for the current user for a specific period."""
    now = datetime.now(timezone.utc)
    response: dict = {}

    if period == 'today':
        response['today'] = UsageStats(**get_today_usage_stats(uid, now)).dict()
        response['history'] = get_hourly_history_for_today(uid, now)
    elif period == 'monthly':
        response['monthly'] = UsageStats(**get_monthly_usage_stats(uid, now)).dict()
        response['history'] = get_daily_history_for_month(uid, now)
    elif period == 'yearly':
        response['yearly'] = UsageStats(**get_yearly_usage_stats(uid, now)).dict()
        response['history'] = get_monthly_history_for_year(uid, now)
    elif period == 'all_time':
        response['all_time'] = UsageStats(**get_all_time_usage_stats(uid)).dict()
        response['history'] = get_yearly_history(uid)

    return response
