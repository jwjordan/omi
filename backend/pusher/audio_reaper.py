"""Daily audio-chunk retention reaper.

Runs inside pusher. Walks PENDANT_LOCAL_AUDIO_ROOT/uid/*/conv/* and
deletes conversation directories whose newest file's mtime is older
than PENDANT_AUDIO_RETENTION_DAYS.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

LOCAL_AUDIO_ROOT = os.getenv('PENDANT_LOCAL_AUDIO_ROOT')
RETENTION_DAYS = int(os.getenv('PENDANT_AUDIO_RETENTION_DAYS', '90'))
REAP_INTERVAL_S = int(os.getenv('PENDANT_REAP_INTERVAL_S', str(86400)))  # daily


def reap_once() -> int:
    """Scan once and delete stale conversation directories. Returns count deleted."""
    if not LOCAL_AUDIO_ROOT or not os.path.isdir(LOCAL_AUDIO_ROOT):
        return 0
    cutoff = time.time() - RETENTION_DAYS * 86400
    deleted = 0

    uid_root = os.path.join(LOCAL_AUDIO_ROOT, 'uid')
    if not os.path.isdir(uid_root):
        return 0

    for uid_dir in os.listdir(uid_root):
        conv_parent = os.path.join(uid_root, uid_dir, 'conv')
        if not os.path.isdir(conv_parent):
            continue
        for conv_id in os.listdir(conv_parent):
            conv_path = os.path.join(conv_parent, conv_id)
            if not os.path.isdir(conv_path):
                continue
            files = [os.path.join(conv_path, f) for f in os.listdir(conv_path)]
            if not files:
                continue
            newest_mtime = max(os.path.getmtime(f) for f in files)
            if newest_mtime < cutoff:
                try:
                    shutil.rmtree(conv_path)
                    deleted += 1
                    logger.info("reaped %s (newest mtime %.0fs old)",
                                conv_path, time.time() - newest_mtime)
                except Exception as e:
                    logger.warning("failed to reap %s: %s", conv_path, e)
    return deleted


_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def _loop():
    logger.info("audio reaper started (interval %ss, retention %d days)",
                REAP_INTERVAL_S, RETENTION_DAYS)
    while not _stop.is_set():
        try:
            reap_once()
        except Exception as e:
            logger.exception("reaper sweep failed: %s", e)
        _stop.wait(REAP_INTERVAL_S)


def start() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name='audio-reaper', daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
