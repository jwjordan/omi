"""Postgres-backed announcements (global collection, not per-user)."""

import json
from datetime import datetime, timezone
from typing import List, Optional

from ._client import db
from models.announcement import Announcement, AnnouncementType, TriggerType


def get_announcement_by_id(announcement_id: str) -> Optional[Announcement]:
    """Get a single announcement by ID."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, data
                FROM announcements
                WHERE id = %s
                """,
                (announcement_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            row_id, created_at, data = row
            # Merge created_at from typed column
            data_dict = dict(data or {})
            data_dict.setdefault("id", row_id)
            data_dict.setdefault("created_at", created_at)
            return Announcement.from_dict(data_dict)


def get_app_changelogs(from_version: str, to_version: str) -> List[Announcement]:
    """
    Get all app changelog announcements between two versions.
    Returns changelogs where from_version < app_version <= to_version.
    Sorted by app_version descending (newest first).
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, data
                FROM announcements
                WHERE data->>'type' = %s
                  AND data->>'active' = 'true'
                ORDER BY created_at DESC
                """,
                (AnnouncementType.CHANGELOG.value,),
            )
            rows = cur.fetchall()

    changelogs = []
    for row_id, created_at, data in rows:
        data_dict = dict(data or {})
        data_dict.setdefault("id", row_id)
        data_dict.setdefault("created_at", created_at)
        announcement = Announcement.from_dict(data_dict)

        app_version = announcement.app_version
        # Skip entries without app_version, then filter by version range
        if (
            app_version
            and _compare_versions(from_version, app_version) < 0
            and _compare_versions(app_version, to_version) <= 0
        ):
            changelogs.append(announcement)

    # Sort by version descending (newest first)
    changelogs.sort(key=lambda x: _version_tuple(x.app_version), reverse=True)
    return changelogs


def get_recent_changelogs(limit: int = 5, max_version: Optional[str] = None) -> List[Announcement]:
    """
    Get the most recent app changelog announcements.
    Returns up to `limit` changelogs sorted by version descending.
    If max_version is provided, only returns changelogs with app_version <= max_version.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, data
                FROM announcements
                WHERE data->>'type' = %s
                  AND data->>'active' = 'true'
                ORDER BY created_at DESC
                """,
                (AnnouncementType.CHANGELOG.value,),
            )
            rows = cur.fetchall()

    changelogs = []
    for row_id, created_at, data in rows:
        data_dict = dict(data or {})
        data_dict.setdefault("id", row_id)
        data_dict.setdefault("created_at", created_at)
        announcement = Announcement.from_dict(data_dict)

        app_version = announcement.app_version
        if app_version:
            # Filter out versions newer than max_version if specified
            if max_version and _compare_versions(app_version, max_version) > 0:
                continue
            changelogs.append(announcement)

    # Sort by version descending
    changelogs.sort(key=lambda x: _version_tuple(x.app_version), reverse=True)

    # Return only the most recent N changelogs
    return changelogs[:limit]


def get_firmware_features(firmware_version: str, device_model: Optional[str] = None) -> List[Announcement]:
    """
    Get feature announcements for a specific firmware version.
    Optionally filter by device model.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, data
                FROM announcements
                WHERE data->>'type' = %s
                  AND data->>'active' = 'true'
                  AND data->>'firmware_version' = %s
                """,
                (AnnouncementType.FEATURE.value, firmware_version),
            )
            rows = cur.fetchall()

    features = []
    for row_id, created_at, data in rows:
        data_dict = dict(data or {})
        data_dict.setdefault("id", row_id)
        data_dict.setdefault("created_at", created_at)
        announcement = Announcement.from_dict(data_dict)

        # Filter by device model if specified
        if device_model and announcement.device_models:
            if device_model not in announcement.device_models:
                continue

        features.append(announcement)

    return features


def get_app_features(app_version: str) -> List[Announcement]:
    """
    Get feature announcements for a specific app version.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, data
                FROM announcements
                WHERE data->>'type' = %s
                  AND data->>'active' = 'true'
                  AND data->>'app_version' = %s
                """,
                (AnnouncementType.FEATURE.value, app_version),
            )
            rows = cur.fetchall()

    announcements = []
    for row_id, created_at, data in rows:
        data_dict = dict(data or {})
        data_dict.setdefault("id", row_id)
        data_dict.setdefault("created_at", created_at)
        announcements.append(Announcement.from_dict(data_dict))

    return announcements


def get_general_announcements(last_checked_at: Optional[datetime] = None) -> List[Announcement]:
    """
    Get active, non-expired general announcements.
    If last_checked_at is provided, only returns announcements created after that time.
    """
    now = datetime.now(timezone.utc)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, data
                FROM announcements
                WHERE data->>'type' = %s
                  AND data->>'active' = 'true'
                ORDER BY created_at DESC
                """,
                (AnnouncementType.ANNOUNCEMENT.value,),
            )
            rows = cur.fetchall()

    announcements = []
    for row_id, created_at, data in rows:
        data_dict = dict(data or {})
        data_dict.setdefault("id", row_id)
        data_dict.setdefault("created_at", created_at)
        announcement = Announcement.from_dict(data_dict)

        # Skip if created before last check
        if last_checked_at and announcement.created_at <= last_checked_at:
            continue

        # Skip if expired
        if announcement.expires_at and announcement.expires_at < now:
            continue

        announcements.append(announcement)

    # Sort by created_at descending
    announcements.sort(key=lambda x: x.created_at, reverse=True)
    return announcements


def get_all_announcements(
    announcement_type: Optional[AnnouncementType] = None,
    active_only: bool = False,
) -> List[Announcement]:
    """
    Get all announcements with optional filtering.

    Args:
        announcement_type: Filter by type (changelog, feature, announcement)
        active_only: If True, only return active announcements
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            where_clauses = []
            params = []

            if announcement_type:
                where_clauses.append("data->>'type' = %s")
                params.append(announcement_type.value)

            if active_only:
                where_clauses.append("data->>'active' = 'true'")

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            cur.execute(
                f"""
                SELECT id, created_at, data
                FROM announcements
                WHERE {where_sql}
                ORDER BY created_at DESC
                """,
                params,
            )
            rows = cur.fetchall()

    announcements = []
    for row_id, created_at, data in rows:
        data_dict = dict(data or {})
        data_dict.setdefault("id", row_id)
        data_dict.setdefault("created_at", created_at)
        announcements.append(Announcement.from_dict(data_dict))

    return announcements


def create_announcement(announcement: Announcement) -> Announcement:
    """Create a new announcement.

    Promotes 'id' and stores the whole dict into the `data` JSONB column.
    """
    announcement_id = announcement.id
    announcement_data = announcement.to_dict()
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO announcements (id, data)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                    SET data = EXCLUDED.data
                """,
                (announcement_id, json.dumps(announcement_data)),
            )

    return announcement


def update_announcement(announcement_id: str, updates: dict) -> Optional[Announcement]:
    """Update an existing announcement.

    Shallow-merge updates into the existing row's data JSONB.
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE announcements
                SET data = data || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps(updates), announcement_id),
            )

    return get_announcement_by_id(announcement_id)


def delete_announcement(announcement_id: str) -> bool:
    """Delete an announcement."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM announcements
                WHERE id = %s
                """,
                (announcement_id,),
            )

    return True


def deactivate_announcement(announcement_id: str) -> bool:
    """Soft delete - set active to False."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE announcements
                SET data = data || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps({"active": False}), announcement_id),
            )

    return True


# Helper functions for version comparison
def _parse_version(version: str) -> tuple:
    """
    Parse version string into semantic tuple, build number, and has_build flag.

    Returns: (semantic_tuple, build_number, has_build)

    Examples:
    - '1.0.10' -> ((1, 0, 10), 0, False)
    - 'v1.0.10' -> ((1, 0, 10), 0, False)
    - '1.0.510+240' -> ((1, 0, 510), 240, True)
    """
    if not version:
        return ((0, 0, 0), 0, False)

    # Remove 'v' prefix if present
    version = version.lstrip("v")

    # Extract build number if present (e.g., '1.0.510+240')
    build_number = 0
    has_build = False
    if "+" in version:
        has_build = True
        version, build_str = version.split("+", 1)
        try:
            build_number = int(build_str)
        except ValueError:
            build_number = 0

    try:
        parts = version.split(".")
        version_parts = tuple(int(p) for p in parts)
        # Pad to 3 components
        while len(version_parts) < 3:
            version_parts = version_parts + (0,)
        return (version_parts[:3], build_number, has_build)
    except (ValueError, AttributeError):
        return ((0, 0, 0), 0, False)


def _version_tuple(version: str) -> tuple:
    """
    Convert version string to tuple for sorting.
    Returns full tuple including build number for proper sorting.
    """
    semantic, build, _ = _parse_version(version)
    return semantic + (build,)


def _compare_versions(v1: str, v2: str) -> int:
    """
    Two-pass version comparison.

    Pass 1: Compare semantic versions (major.minor.patch)
    Pass 2: If semantic versions are equal, compare build numbers
            BUT if either version has no build number, consider them equal

    This allows:
    - '1.0.521' to match all builds like '1.0.521+607', '1.0.521+608', etc.
    - '1.0.521+607' < '1.0.521+608' (when both have build numbers)
    - '1.0.521' < '1.0.522' (semantic comparison)

    Returns: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    sem1, build1, has_build1 = _parse_version(v1)
    sem2, build2, has_build2 = _parse_version(v2)

    # First pass: semantic version comparison
    if sem1 < sem2:
        return -1
    if sem1 > sem2:
        return 1

    # Semantic versions are equal
    # Second pass: build number comparison (only if BOTH have build numbers)
    if not has_build1 or not has_build2:
        # If either version lacks a build number, consider them equal
        # This means '1.0.521' matches '1.0.521+607'
        return 0

    # Both have build numbers, compare them
    if build1 < build2:
        return -1
    if build1 > build2:
        return 1
    return 0


# ============================================================================
# Per-user dismissal tracking
# ============================================================================


def get_dismissed_announcement_ids(uid: str) -> set:
    """Get the set of announcement IDs that a user has dismissed."""
    dismissed_ref = db.collection("users").document(uid).collection("dismissed_announcements")
    docs = dismissed_ref.stream()
    return {doc.id for doc in docs}


def dismiss_announcement(uid: str, announcement_id: str, cta_clicked: bool = False) -> bool:
    """
    Mark an announcement as dismissed for a user.
    Returns True if successful.
    """
    dismissed_ref = db.collection("users").document(uid).collection("dismissed_announcements").document(announcement_id)
    dismissed_ref.set(
        {
            "dismissed_at": datetime.now(timezone.utc),
            "cta_clicked": cta_clicked,
        }
    )
    return True


def is_announcement_dismissed(uid: str, announcement_id: str) -> bool:
    """Check if a user has dismissed a specific announcement."""
    dismissed_ref = db.collection("users").document(uid).collection("dismissed_announcements").document(announcement_id)
    doc = dismissed_ref.get()
    return doc.exists


# ============================================================================
# Unified pending announcements query
# ============================================================================


def get_pending_announcements(
    uid: str,
    app_version: str,
    platform: str,
    trigger: str,
    firmware_version: Optional[str] = None,
    device_model: Optional[str] = None,
) -> List[Announcement]:
    """
    Get all announcements that should be shown to a user.

    Filtering logic:
    1. active == True
    2. Not in user's dismissed_announcements (if show_once == True)
    3. Within time window (start_at <= now <= expires_at)
    4. Matches targeting rules (version range, device, platform)
    5. Matches trigger type
    6. Sorted by priority (descending)

    Args:
        uid: User ID for dismissal tracking
        app_version: Current app version (e.g., "1.0.522+240")
        platform: "ios" or "android"
        trigger: "app_launch", "version_upgrade", or "firmware_upgrade"
        firmware_version: Current firmware version (optional)
        device_model: Device model name (optional)

    Returns:
        List of announcements to show, sorted by priority (highest first)
    """
    now = datetime.now(timezone.utc)

    # Map trigger string to enum
    trigger_map = {
        "app_launch": TriggerType.IMMEDIATE,
        "version_upgrade": TriggerType.VERSION_UPGRADE,
        "firmware_upgrade": TriggerType.FIRMWARE_UPGRADE,
    }
    requested_trigger = trigger_map.get(trigger, TriggerType.IMMEDIATE)

    # Get user's dismissed announcements
    dismissed_ids = get_dismissed_announcement_ids(uid)

    # Query all active announcements
    announcements_ref = db.collection("announcements")
    query = announcements_ref.where(filter=FieldFilter("active", "==", True))
    docs = query.stream()

    pending = []

    for doc in docs:
        data = doc.to_dict()
        announcement = Announcement.from_dict(data)

        # Get effective targeting and display configs
        targeting = announcement.get_effective_targeting()
        display = announcement.get_effective_display()

        # 1. Check if already dismissed (and show_once is true)
        if display.show_once and announcement.id in dismissed_ids:
            continue

        # 2. Check time window
        effective_start = display.start_at
        effective_expires = display.expires_at or announcement.expires_at  # Fallback to legacy field

        if effective_start and now < effective_start:
            continue
        if effective_expires and now > effective_expires:
            continue

        # 3. Check trigger type
        if targeting.trigger != requested_trigger:
            # Special case: IMMEDIATE trigger announcements should also show on version/firmware upgrades
            if targeting.trigger != TriggerType.IMMEDIATE:
                continue

        # 4. Check platform targeting
        if targeting.platforms and platform not in targeting.platforms:
            continue

        # 5. Check app version range
        if targeting.app_version_min:
            if _compare_versions(app_version, targeting.app_version_min) < 0:
                continue
        if targeting.app_version_max:
            if _compare_versions(app_version, targeting.app_version_max) > 0:
                continue

        # 6. Check firmware version range (only if firmware_version provided)
        if targeting.firmware_version_min or targeting.firmware_version_max:
            if not firmware_version:
                continue
            if targeting.firmware_version_min:
                if _compare_versions(firmware_version, targeting.firmware_version_min) < 0:
                    continue
            if targeting.firmware_version_max:
                if _compare_versions(firmware_version, targeting.firmware_version_max) > 0:
                    continue

        # 7. Check device model targeting
        if targeting.device_models:
            if not device_model or device_model not in targeting.device_models:
                continue

        # 8. Check test_uids (if set, only those users see the announcement)
        if targeting.test_uids:
            if uid not in targeting.test_uids:
                continue

        pending.append(announcement)

    # Sort by priority (highest first)
    pending.sort(key=lambda x: x.get_effective_display().priority, reverse=True)

    return pending
