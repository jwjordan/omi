"""One-shot: embed every completed conversation missing a conversation_vectors row.

Stage 3 companion script. Idempotent — safe to re-run. Logs each per-conversation
result so partial failures during the historic catch-up don't poison the whole run.

Usage (from ~/github/omi/backend):
    PYTHONPATH=. python3 scripts/backfill_vectors.py
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

# Running as a script: the backend module search path is the CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database._client import db
from database.vector_db import upsert_vector
from utils.llm.clients import embeddings


logger = logging.getLogger("backfill_vectors")


def _fetch_missing_conversations() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return (uid, conversation_id, data) for every completed, non-discarded
    conversation with no row in conversation_vectors.
    """
    sql = """
        SELECT c.uid, c.id, c.data
          FROM conversations c
     LEFT JOIN conversation_vectors cv
            ON cv.uid = c.uid AND cv.memory_id = c.id
         WHERE c.status = 'completed'
           AND NOT c.discarded
           AND cv.memory_id IS NULL
      ORDER BY c.started_at ASC
    """
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return list(cur.fetchall())


def _content_for_embedding(data: Dict[str, Any]) -> str:
    """Extract the text we embed for a conversation. Uses Claude-generated
    structured summary, same as the ingest path (process_conversation.py:580
    calls `generate_embedding(str(conversation.structured))`).
    """
    structured = (data or {}).get("structured", {}) or {}
    return str(structured)


def backfill_missing_vectors() -> Dict[str, int]:
    """Backfill every missing vector. Returns {"embedded", "failed", "skipped"}."""
    rows = _fetch_missing_conversations()
    stats = {"embedded": 0, "failed": 0, "skipped": 0}

    if not rows:
        logger.info("no conversations need backfill")
        return stats

    logger.info("backfilling %d conversations", len(rows))
    for uid, conv_id, data in rows:
        content = _content_for_embedding(data)
        if not content.strip():
            logger.warning("skip %s/%s: empty structured content", uid, conv_id)
            stats["skipped"] += 1
            continue
        try:
            vector = embeddings.embed_query(content)
            upsert_vector(uid, conv_id, vector)
            stats["embedded"] += 1
            logger.info("embedded %s/%s", uid, conv_id)
        except Exception as e:
            logger.exception("failed %s/%s: %s", uid, conv_id, e)
            stats["failed"] += 1

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    stats = backfill_missing_vectors()
    print(json.dumps(stats))
