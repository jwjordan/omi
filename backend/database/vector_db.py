"""Vector store backed by local Postgres + pgvector.

Replaces the Pinecone implementation. Two tables (see
pendant-stack/postgres-init/01-pgvector.sql):

- conversation_vectors  — ns1 equivalent (per-conversation transcript embeddings)
- memory_vectors        — ns2 equivalent (per-memory distilled-fact embeddings)

Screen-activity (ns3) functions are stubbed to no-op: the pendant use case does
not generate screenshots and no table exists for them.

Text-embedding-3-large is 3072 dims. pgvector 0.8.x caps vector indexes at 2000
(ivfflat) / 2000-for-vector or 4000-for-halfvec (HNSW), so we run sequential
scan with btree prefilter on (uid, created_at). Fine for single-user scale.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from utils.llm.clients import embeddings

logger = logging.getLogger(__name__)


# Preserved for callers that import namespace strings from this module even
# though namespacing is implicit in the Postgres table split now.
MEMORIES_NAMESPACE = "ns2"
SCREEN_ACTIVITY_NAMESPACE = "ns3"


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
def _build_pool() -> Optional[ConnectionPool]:
    """Build the module-level pool, or return None if DATABASE_URL is absent.

    We want imports to succeed in environments that don't set DATABASE_URL
    (tests, CI, early boot). Functions that need the pool raise at call time.
    """
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        return None
    try:
        pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True},
            configure=register_vector,
        )
        return pool
    except Exception as e:
        logger.warning("vector_db: failed to build Postgres pool: %s", e)
        return None


_pool: Optional[ConnectionPool] = _build_pool()


def _reset_pool_for_testing() -> None:
    """Rebuild the module-level pool. Tests swap DATABASE_URL before calling."""
    global _pool
    _pool = _build_pool()


def _connection():
    if _pool is None:
        raise RuntimeError("DATABASE_URL not configured — vector_db is disabled")
    return _pool.connection()


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# ns1 conversation_vectors
# ---------------------------------------------------------------------------
_UPSERT_CONV_SQL = """
    INSERT INTO conversation_vectors (id, uid, memory_id, created_at, embedding, metadata)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE
        SET uid = EXCLUDED.uid,
            memory_id = EXCLUDED.memory_id,
            created_at = EXCLUDED.created_at,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata
"""


def _base_metadata(uid: str, memory_id: str) -> dict:
    return {
        "uid": uid,
        "memory_id": memory_id,
        "created_at": _now_ts(),
    }


def upsert_vector(uid: str, conversation_id: str, vector: List[float]):
    """Upsert a single conversation embedding with default metadata."""
    meta = _base_metadata(uid, conversation_id)
    with _connection() as conn:
        conn.execute(
            _UPSERT_CONV_SQL,
            (
                f"{uid}-{conversation_id}",
                uid,
                conversation_id,
                meta["created_at"],
                list(vector),
                json.dumps(meta),
            ),
        )
    logger.info("upsert_vector id=%s-%s", uid, conversation_id)


def upsert_vector2(uid: str, conversation_id: str, vector: List[float], metadata: dict):
    """Upsert a conversation embedding, merging caller metadata into defaults."""
    meta = _base_metadata(uid, conversation_id)
    meta.update(metadata or {})
    with _connection() as conn:
        conn.execute(
            _UPSERT_CONV_SQL,
            (
                f"{uid}-{conversation_id}",
                uid,
                conversation_id,
                meta["created_at"],
                list(vector),
                json.dumps(meta),
            ),
        )
    logger.info("upsert_vector2 id=%s-%s keys=%s", uid, conversation_id, list(metadata or {}))


def update_vector_metadata(uid: str, conversation_id: str, metadata: dict):
    """Replace the JSONB metadata for an existing conversation row.

    The Pinecone `update(..., set_metadata=...)` API forced uid + memory_id
    into the payload. Preserve that behavior so callers that rely on it don't
    accidentally drop those fields.
    """
    metadata = dict(metadata or {})
    metadata["uid"] = uid
    metadata["memory_id"] = conversation_id
    with _connection() as conn:
        conn.execute(
            "UPDATE conversation_vectors SET metadata = %s::jsonb WHERE id = %s",
            (json.dumps(metadata), f"{uid}-{conversation_id}"),
        )


def upsert_vectors(uid: str, vectors: List[List[float]], conversation_ids: List[str]):
    """Batch-upsert conversation embeddings atomically: all rows commit or none do."""
    if not vectors:
        return
    now = _now_ts()
    with _connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for cid, vec in zip(conversation_ids, vectors):
                    meta = {"uid": uid, "memory_id": cid, "created_at": now}
                    cur.execute(
                        _UPSERT_CONV_SQL,
                        (
                            f"{uid}-{cid}",
                            uid,
                            cid,
                            now,
                            list(vec),
                            json.dumps(meta),
                        ),
                    )
    logger.info("upsert_vectors uid=%s count=%d", uid, len(vectors))


def query_vectors(
    query: str,
    uid: str,
    starts_at: int = None,
    ends_at: int = None,
    k: int = 5,
) -> List[str]:
    """Embed `query`, retrieve top-k conversation_ids by cosine distance.

    Returns conversation_ids with the `<uid>-` prefix stripped.
    """
    xq = embeddings.embed_query(query)

    sql = [
        "SELECT id, embedding <=> %s::vector AS distance",
        "FROM conversation_vectors",
        "WHERE uid = %s",
    ]
    params: list = [list(xq), uid]
    if starts_at is not None and ends_at is not None:
        sql.append("AND created_at BETWEEN %s AND %s")
        params.extend([starts_at, ends_at])
    sql.append("ORDER BY embedding <=> %s::vector")
    params.append(list(xq))
    sql.append("LIMIT %s")
    params.append(k)

    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute("\n".join(sql), tuple(params))
            rows = cur.fetchall()

    return [row[0].replace(f"{uid}-", "", 1) for row in rows]


def query_vectors_by_metadata(
    uid: str,
    vector: List[float],
    dates_filter: List[datetime],
    people: List[str],
    topics: List[str],
    entities: List[str],
    dates: List[str],
    limit: int = 5,
):
    """Filter by JSONB arrays (topics/entities/people_mentioned) + optional date range.

    Post-sorts by how many metadata keys matched (to preserve the Pinecone
    implementation's behavior). If the metadata-filter query returns nothing,
    retry once without the metadata clauses.
    """
    has_meta_filter = bool(people or topics or entities or dates)
    has_date_filter = (
        dates_filter
        and len(dates_filter) == 2
        and dates_filter[0]
        and dates_filter[1]
    )

    def _run(with_meta: bool) -> list:
        sql = [
            "SELECT id, metadata",
            "FROM conversation_vectors",
            "WHERE uid = %s",
        ]
        params: list = [uid]
        if with_meta and has_meta_filter:
            sql.append(
                "AND ("
                "metadata->'topics' ?| %s::text[] "
                "OR metadata->'entities' ?| %s::text[] "
                "OR metadata->'people_mentioned' ?| %s::text[]"
                ")"
            )
            params.extend([topics or [], entities or [], people or []])
        if has_date_filter:
            sql.append("AND created_at BETWEEN %s AND %s")
            params.extend(
                [int(dates_filter[0].timestamp()), int(dates_filter[1].timestamp())]
            )
        sql.append("ORDER BY embedding <=> %s::vector")
        params.append(list(vector))
        sql.append("LIMIT %s")
        params.append(1000)
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute("\n".join(sql), tuple(params))
                return cur.fetchall()

    rows = _run(with_meta=True)
    if not rows:
        if has_meta_filter and has_date_filter:
            logger.warning(
                "query_vectors_by_metadata retrying without structured filters"
            )
            rows = _run(with_meta=False)
        else:
            return []

    conv_match_counts = defaultdict(int)
    for row in rows:
        row_id, metadata = row[0], row[1]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        conv_id = metadata.get("memory_id") or row_id.replace(f"{uid}-", "", 1)
        for topic in topics or []:
            if topic in metadata.get("topics", []):
                conv_match_counts[conv_id] += 1
        for entity in entities or []:
            if entity in metadata.get("entities", []):
                conv_match_counts[conv_id] += 1
        for person in people or []:
            if person in metadata.get("people_mentioned", []):
                conv_match_counts[conv_id] += 1

    conversation_ids = [row[0].replace(f"{uid}-", "", 1) for row in rows]
    conversation_ids.sort(key=lambda cid: conv_match_counts[cid], reverse=True)
    return conversation_ids[:limit] if len(conversation_ids) > limit else conversation_ids


def delete_vector(uid: str, conversation_id: str):
    """Delete a conversation vector by its composed id."""
    vector_id = f"{uid}-{conversation_id}"
    with _connection() as conn:
        conn.execute(
            "DELETE FROM conversation_vectors WHERE id = %s",
            (vector_id,),
        )
    logger.info("delete_vector %s", vector_id)


# ---------------------------------------------------------------------------
# ns2 memory_vectors
# ---------------------------------------------------------------------------
_UPSERT_MEM_SQL = """
    INSERT INTO memory_vectors (id, uid, memory_id, category, created_at, embedding, metadata)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE
        SET uid = EXCLUDED.uid,
            memory_id = EXCLUDED.memory_id,
            category = EXCLUDED.category,
            created_at = EXCLUDED.created_at,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata
"""


def upsert_memory_vector(uid: str, memory_id: str, content: str, category: str):
    """Embed `content` and upsert a memory row. Returns the vector used."""
    if _pool is None:
        logger.warning("vector_db pool not initialized, skipping memory vector upsert")
        return None

    vector = embeddings.embed_query(content)
    now = _now_ts()
    meta = {
        "uid": uid,
        "memory_id": memory_id,
        "category": category,
        "created_at": now,
    }
    with _connection() as conn:
        conn.execute(
            _UPSERT_MEM_SQL,
            (
                f"{uid}-{memory_id}",
                uid,
                memory_id,
                category,
                now,
                list(vector),
                json.dumps(meta),
            ),
        )
    logger.info("upsert_memory_vector %s", memory_id)
    return vector


def upsert_memory_vectors_batch(uid: str, items: List[dict]) -> int:
    """Batch-embed + batch-upsert memories. Returns count written."""
    if _pool is None:
        logger.warning("vector_db pool not initialized, skipping memory batch upsert")
        return 0
    if not items:
        return 0

    contents = [item["content"] for item in items]
    vectors = embeddings.embed_documents(contents)
    now = _now_ts()

    with _connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for item, vec in zip(items, vectors):
                    meta = {
                        "uid": uid,
                        "memory_id": item["memory_id"],
                        "category": item["category"],
                        "created_at": now,
                    }
                    cur.execute(
                        _UPSERT_MEM_SQL,
                        (
                            f"{uid}-{item['memory_id']}",
                            uid,
                            item["memory_id"],
                            item["category"],
                            now,
                            list(vec),
                            json.dumps(meta),
                        ),
                    )

    logger.info("upsert_memory_vectors_batch count=%d", len(items))
    return len(items)


def find_similar_memories(
    uid: str, content: str, threshold: float = 0.85, limit: int = 5
) -> List[dict]:
    """Return memories with cosine similarity >= threshold, best first."""
    if _pool is None:
        logger.warning("vector_db pool not initialized, skipping memory similarity search")
        return []

    vector = embeddings.embed_query(content)
    sql = (
        "SELECT memory_id, category, embedding <=> %s::vector AS distance "
        "FROM memory_vectors "
        "WHERE uid = %s "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    )
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (list(vector), uid, list(vector), limit))
            rows = cur.fetchall()

    results = []
    for row in rows:
        memory_id, category, distance = row
        score = 1.0 - float(distance)
        if score >= threshold:
            results.append({"memory_id": memory_id, "category": category, "score": score})
    return results


def check_memory_duplicate(uid: str, content: str, threshold: float = 0.85):
    """Top-1 similar memory above threshold, or None."""
    similar = find_similar_memories(uid, content, threshold=threshold, limit=1)
    if similar:
        logger.warning("Found duplicate memory: %s", similar[0])
        return similar[0]
    return None


def search_memories_by_vector(uid: str, query: str, limit: int = 10) -> List[str]:
    """Semantic search over memory_vectors. Returns memory_ids, best first."""
    if _pool is None:
        logger.warning("vector_db pool not initialized, skipping memory search")
        return []

    vector = embeddings.embed_query(query)
    sql = (
        "SELECT memory_id "
        "FROM memory_vectors "
        "WHERE uid = %s "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    )
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (uid, list(vector), limit))
            rows = cur.fetchall()
    return [row[0] for row in rows]


def delete_memory_vector(uid: str, memory_id: str):
    """Delete a memory vector by composed id."""
    if _pool is None:
        logger.warning("vector_db pool not initialized, skipping memory vector delete")
        return
    vector_id = f"{uid}-{memory_id}"
    with _connection() as conn:
        conn.execute(
            "DELETE FROM memory_vectors WHERE id = %s",
            (vector_id,),
        )
    logger.info("delete_memory_vector %s", vector_id)


# ---------------------------------------------------------------------------
# ns3 screen_activity: stubbed (no table; pendant use case doesn't generate screenshots)
# ---------------------------------------------------------------------------
_SCREEN_STUB_MSG = "screen activity vectors not implemented in pendant-stack build"


def upsert_screen_activity_vectors(uid: str, rows: List[dict]) -> int:
    logger.warning(_SCREEN_STUB_MSG)
    return 0


def search_screen_activity_vectors(
    uid: str,
    query_vector: List[float],
    start_date: int = None,
    end_date: int = None,
    app_filter: str = None,
    k: int = 10,
) -> List[dict]:
    logger.warning(_SCREEN_STUB_MSG)
    return []


def delete_screen_activity_vectors(uid: str, ids: List[int]):
    logger.warning(_SCREEN_STUB_MSG)
    return
