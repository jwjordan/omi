"""Regression test for _utc_date_str in utils.llm.chat.

Pydantic declares Conversation.created_at as datetime, but the Stage 2
Postgres round-trip can leave it as an ISO string depending on the
construction path. Before this fix, the three retrieve_metadata_* sites
in utils/llm/chat.py called `.astimezone(...).strftime(...)` directly on
`conversation.created_at`, which would raise
`'str' object has no attribute 'strftime'` mid-finalize.
"""

from datetime import datetime, timezone

from utils.llm.chat import _utc_date_str


def test_handles_aware_datetime():
    ts = datetime(2026, 4, 23, 15, 30, tzinfo=timezone.utc)
    assert _utc_date_str(ts) == "2026-04-23"


def test_handles_iso_string_with_z_suffix():
    assert _utc_date_str("2026-04-23T15:30:00Z") == "2026-04-23"


def test_handles_iso_string_with_offset():
    # 2026-04-23 23:30 at +08:00 is 2026-04-23 15:30 UTC
    assert _utc_date_str("2026-04-23T23:30:00+08:00") == "2026-04-23"


def test_handles_iso_string_crossing_utc_date_boundary():
    # 2026-04-23 22:00 PDT (-07:00) is 2026-04-24 05:00 UTC — must show 04-24.
    assert _utc_date_str("2026-04-23T22:00:00-07:00") == "2026-04-24"
