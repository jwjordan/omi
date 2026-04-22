"""pgvector-backed semantic search over conversation_vectors.

Stage 3 addition. Adds a per-user, semantic search surface over the local
pgvector store. Distinct from utils/conversations/search.py, which is the
legacy Typesense implementation (kept for upstream compatibility but
unused in the self-hosted stack).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from database._client import db
from utils.llm.clients import embeddings


def _decrypt_conversation_data(data, uid):
    """Lazy-import wrapper so the module loads without opuslib at import time.

    The real implementation is in database.conversations; tests patch this
    name directly on this module, so the wrapper is only reached in
    production paths (where opuslib is available in the container).
    """
    from database.conversations import _decrypt_conversation_data as _real
    return _real(data, uid)


class EmbedServiceUnavailable(RuntimeError):
    """The embeddings backend (llm-proxy → Ollama) is down or unreachable."""


_EXCERPT_CAP = 3


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _query_terms(query: str) -> List[str]:
    """Split a query into non-empty, lowercased whitespace terms for segment grep."""
    return [t.lower() for t in query.split() if t.strip()]


def _pick_excerpts(segments: List[Dict[str, Any]], terms: List[str]) -> List[Dict[str, Any]]:
    """Return up to _EXCERPT_CAP segments whose text contains any query term.

    Match is case-insensitive substring. Preserves segment order (first
    matches first). If a segment is missing required keys, it's skipped.
    """
    if not terms or not isinstance(segments, list):
        return []
    picked: List[Dict[str, Any]] = []
    for seg in segments:
        text = seg.get("text")
        if not isinstance(text, str):
            continue
        lowered = text.lower()
        if any(term in lowered for term in terms):
            picked.append(
                {
                    "text": text,
                    "start_seconds": float(seg.get("start", 0.0)),
                    "speaker": seg.get("speaker", "SPEAKER_00"),
                }
            )
            if len(picked) >= _EXCERPT_CAP:
                break
    return picked


def semantic_search_conversations(
    uid: str,
    query: str,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Return top-N conversations semantically similar to `query` for `uid`.

    Args:
        uid: user id.
        query: natural-language search query.
        since: optional ISO 8601 lower bound on conversations.started_at.
        until: optional ISO 8601 upper bound on conversations.started_at.
        limit: max hits, capped at 20.

    Returns:
        {"query": str, "hits": [
            {"conversation_id", "started_at", "finished_at",
             "similarity", "title", "overview", "excerpts": [...]}
        ]}

    Raises:
        EmbedServiceUnavailable if the embeddings backend is unreachable.
    """
    try:
        query_vector = embeddings.embed_query(query)
    except Exception as e:
        raise EmbedServiceUnavailable(str(e)) from e

    since_dt = _parse_iso(since)
    until_dt = _parse_iso(until)

    where_clauses = [
        "cv.uid = %s",
        "c.status = 'completed'",
        "NOT c.discarded",
    ]
    params_list: list = [
        query_vector,  # SELECT similarity
        uid,           # WHERE cv.uid
    ]

    if since_dt is not None:
        where_clauses.append("c.started_at >= %s")
        params_list.append(since_dt)

    if until_dt is not None:
        where_clauses.append("c.started_at <= %s")
        params_list.append(until_dt)

    params_list.append(query_vector)   # ORDER BY
    params_list.append(min(limit, 20))  # LIMIT

    where_sql = " AND ".join(where_clauses)
    sql = f"""
        SELECT cv.memory_id,
               c.data,
               c.started_at,
               c.finished_at,
               1 - (cv.embedding <=> %s::vector) AS similarity
          FROM conversation_vectors cv
          JOIN conversations c
            ON c.id = cv.memory_id AND c.uid = cv.uid
         WHERE {where_sql}
      ORDER BY cv.embedding <=> %s::vector
         LIMIT %s
    """

    params = tuple(params_list)

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    hits: List[Dict[str, Any]] = []
    terms = _query_terms(query)
    for conv_id, data, started_at, finished_at, similarity in rows:
        structured = (data or {}).get("structured", {}) or {}
        title = structured.get("title") or ""
        overview = structured.get("overview") or ""

        excerpts: List[Dict[str, Any]] = []
        try:
            decrypted = _decrypt_conversation_data(data, uid)
        except Exception:
            # Per spec: per-hit decryption failure yields empty excerpts,
            # it does not fail the request.
            decrypted = None

        if decrypted is not None:
            segments = decrypted.get("transcript_segments", []) or []
            if isinstance(segments, list):
                excerpts = _pick_excerpts(segments, terms)

        hits.append(
            {
                "conversation_id": conv_id,
                "started_at": started_at.isoformat() if started_at else None,
                "finished_at": finished_at.isoformat() if finished_at else None,
                "similarity": float(similarity),
                "title": title,
                "overview": overview,
                "excerpts": excerpts,
            }
        )

    return {"query": query, "hits": hits}
