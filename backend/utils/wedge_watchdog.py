"""Pendant batch-mode wedge detection.

Tracks per-uid timestamps of recent /v2/sync-local-files activity so the
live-WS watchdog in transcribe.py can distinguish "user is mid-bulk-sync"
(don't intervene) from "pendant has stopped doing anything on either path"
(wedged — close the WS to force the iPhone to reconnect).
"""

import os
import time

WEDGE_TIMEOUT_S: float = float(os.getenv('PENDANT_WEDGE_TIMEOUT_S', '180'))
MAX_SELF_HEAL_ATTEMPTS: int = int(os.getenv('PENDANT_MAX_SELF_HEAL_ATTEMPTS', '2'))

_last_sync_activity: dict[str, float] = {}
_consecutive_wedge_count: dict[str, int] = {}


def record_sync_activity(uid: str) -> None:
    _last_sync_activity[uid] = time.time()


def seconds_since_last_sync_activity(uid: str) -> float:
    last = _last_sync_activity.get(uid)
    if last is None:
        return float('inf')
    return time.time() - last


def record_self_heal_attempt(uid: str) -> int:
    n = _consecutive_wedge_count.get(uid, 0) + 1
    _consecutive_wedge_count[uid] = n
    return n


def reset_wedge_count(uid: str) -> None:
    _consecutive_wedge_count.pop(uid, None)


def should_attempt_self_heal(uid: str) -> bool:
    return _consecutive_wedge_count.get(uid, 0) < MAX_SELF_HEAL_ATTEMPTS
