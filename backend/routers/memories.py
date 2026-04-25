import asyncio
import logging
from typing import List, Optional

from utils.executors import critical_executor

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

import database.memories as memories_db
from database.vector_db import (
    delete_memory_vector,
    search_memories_by_vector,
    upsert_memory_vector,
    upsert_memory_vectors_batch,
)
from models.memories import MemoryDB, Memory, MemoryCategory
from utils.apps import update_personas_async
from utils.other import endpoints as auth

logger = logging.getLogger(__name__)

router = APIRouter()

# Hard cap on memories per batch request. Keep aligned with the corresponding
# Pydantic max_length validator below and with the Swift client chunker.
MEMORIES_BATCH_MAX = 100


class BatchMemoriesRequest(BaseModel):
    memories: List[Memory] = Field(
        description="List of memories to create in a single batch request",
        max_length=MEMORIES_BATCH_MAX,
    )


class BatchMemoriesResponse(BaseModel):
    memories: List[MemoryDB]
    created_count: int


def _validate_memory(uid: str, memory_id: str) -> dict:
    memory = memories_db.get_memory(uid, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    if memory.get('is_locked', False):
        raise HTTPException(status_code=402, detail="A paid plan is required to access this memory.")

    return memory


@router.post('/v3/memories', tags=['memories'], response_model=MemoryDB)
def create_memory(memory: Memory, uid: str = Depends(auth.get_current_user_uid)):
    memory.category = MemoryCategory.manual
    memory_db = MemoryDB.from_memory(memory, uid, None, True)
    memories_db.create_memory(uid, memory_db.dict())

    upsert_memory_vector(uid, memory_db.id, memory_db.content, memory_db.category.value)

    if memory.visibility == 'public':
        critical_executor.submit(update_personas_async, uid)
    return memory_db


@router.post(
    '/v3/memories/batch',
    tags=['memories'],
    response_model=BatchMemoriesResponse,
)
async def create_memories_batch(
    request: BatchMemoriesRequest,
    uid: str = Depends(auth.with_rate_limit(auth.get_current_user_uid, "memories:batch")),
):
    """
    Create many memories in a single request.

    Solves the Cloud Armor throttling seen on onboarding: the desktop client
    used to fan out one `POST /v3/memories` per local-file memory (up to 2800
    per user), blowing through the 120 req/min per-Authorization Cloud Armor
    rule and collaterally 429-ing unrelated calls (goals, sync, chat).

    One HTTP request here = one Firestore batch write + one embeddings call +
    one Pinecone upsert, regardless of batch size.
    """
    if not request.memories:
        return BatchMemoriesResponse(memories=[], created_count=0)

    # Hardcode category to manual to match the single-create endpoint. Callers
    # that need auto-categorization should use the dev API.
    memory_dbs: List[MemoryDB] = []
    has_public = False
    for memory in request.memories:
        memory.category = MemoryCategory.manual
        memory_db = MemoryDB.from_memory(memory, uid, None, True)
        memory_dbs.append(memory_db)
        if memory.visibility == 'public':
            has_public = True

    # Firestore batch write + Pinecone batch upsert run on a worker thread so a
    # slow embeddings/Pinecone call can't starve the FastAPI sync threadpool.
    def _persist():
        memories_db.save_memories(uid, [m.dict() for m in memory_dbs])
        upsert_memory_vectors_batch(
            uid,
            [
                {
                    "memory_id": m.id,
                    "content": m.content,
                    "category": m.category.value,
                }
                for m in memory_dbs
            ],
        )

    await asyncio.to_thread(_persist)

    if has_public:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(critical_executor, update_personas_async, uid)

    return BatchMemoriesResponse(memories=memory_dbs, created_count=len(memory_dbs))


@router.get('/v3/memories', tags=['memories'], response_model=List[MemoryDB])
def get_memories(limit: int = 100, offset: int = 0, uid: str = Depends(auth.get_current_user_uid)):
    # Use high limits for the first page
    # Warn: should remove
    if offset == 0:
        limit = 5000
    memories = memories_db.get_memories(uid, limit, offset)

    valid_memories = []
    for memory in memories:
        if memory.get('is_locked', False):
            content = memory.get('content', '')
            memory['content'] = (content[:70] + '...') if len(content) > 70 else content
        try:
            valid_memories.append(MemoryDB.model_validate(memory))
        except ValidationError as e:
            missing_fields = [err['loc'][0] for err in e.errors() if err.get('loc')]
            logger.warning(
                f"Skipping invalid memory doc {memory.get('id', 'unknown')}: missing/invalid fields {missing_fields}"
            )
            continue
    return valid_memories


@router.delete('/v3/memories/{memory_id}', tags=['memories'])
def delete_memory(memory_id: str, uid: str = Depends(auth.get_current_user_uid)):
    _validate_memory(uid, memory_id)
    memories_db.delete_memory(uid, memory_id)
    delete_memory_vector(uid, memory_id)
    return {'status': 'ok'}


@router.delete('/v3/memories', tags=['memories'])
def delete_memories(uid: str = Depends(auth.get_current_user_uid)):
    memories_db.delete_all_memories(uid)
    return {'status': 'ok'}


@router.post('/v3/memories/{memory_id}/review', tags=['memories'])
def review_memory(memory_id: str, value: bool, uid: str = Depends(auth.get_current_user_uid)):
    _validate_memory(uid, memory_id)
    memories_db.review_memory(uid, memory_id, value)
    return {'status': 'ok'}


@router.patch('/v3/memories/{memory_id}', tags=['memories'])
def edit_memory(memory_id: str, value: str, uid: str = Depends(auth.get_current_user_uid)):
    _validate_memory(uid, memory_id)
    # first_word = value.split(' ')[0]
    # user_name = get_user_name(uid, use_default=False)
    # if user_name == first_word:
    #     value = value[len(first_word):].strip()

    memories_db.edit_memory(uid, memory_id, value)
    return {'status': 'ok'}


@router.patch('/v3/memories/{memory_id}/visibility', tags=['memories'])
def update_memory_visibility(memory_id: str, value: str, uid: str = Depends(auth.get_current_user_uid)):
    _validate_memory(uid, memory_id)
    if value not in ['public', 'private']:
        raise HTTPException(status_code=400, detail='Invalid visibility value')
    memories_db.change_memory_visibility(uid, memory_id, value)
    critical_executor.submit(update_personas_async, uid)
    return {'status': 'ok'}


# ---------------------------------------------------------------------------
# Edwin / pendant_memory MCP integration
# ---------------------------------------------------------------------------
# Two endpoints exposed to Edwin (NanoClaw) so it can semantically search the
# pendant-extracted memory pool and promote selected ones into its curated
# filesystem store. See docs/superpowers/specs (post Stage-3) for the
# motivating design — Omi captures opportunistically, Edwin curates
# deliberately, and these endpoints let the curated pool draw from the
# opportunistic one without merging the two stores.


_MEMORIES_SEARCH_MAX = 25


class MemoriesSearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language search query")
    limit: int = Field(default=10, ge=1, le=_MEMORIES_SEARCH_MAX)


class MemoriesSearchHit(BaseModel):
    id: str
    content: str
    category: Optional[str] = None
    headline: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    created_at: Optional[str] = None
    rank: int
    promoted_to_edwin: bool = False
    promoted_to_edwin_at: Optional[str] = None


class MemoriesSearchResponse(BaseModel):
    query: str
    hits: List[MemoriesSearchHit]


def _hit_from_memory(memory: dict, rank: int) -> MemoriesSearchHit:
    created_at = memory.get('created_at')
    if hasattr(created_at, 'isoformat'):
        created_at = created_at.isoformat()
    return MemoriesSearchHit(
        id=memory['id'],
        content=memory.get('content') or '',
        category=memory.get('category'),
        headline=memory.get('headline'),
        tags=list(memory.get('tags') or []),
        conversation_id=memory.get('conversation_id'),
        created_at=created_at,
        rank=rank,
        promoted_to_edwin=bool(memory.get('promoted_to_edwin', False)),
        promoted_to_edwin_at=memory.get('promoted_to_edwin_at'),
    )


@router.post('/v3/memories/search', tags=['memories'], response_model=MemoriesSearchResponse)
def semantic_search_memories(
    body: MemoriesSearchRequest,
    uid: str = Depends(auth.get_current_user_uid),
):
    """pgvector-backed semantic search over this user's memories.

    The vector index already exists for save-time dedup; this endpoint reuses
    it for read-side recall. Results preserve the vector ranking — the
    decryption layer in get_memories_by_ids may filter or re-order via
    user_review elsewhere, but this endpoint deliberately leaves rejected
    memories visible to the caller (Edwin) so it can decide for itself.
    """
    ids = search_memories_by_vector(uid, body.query, limit=body.limit)
    if not ids:
        return MemoriesSearchResponse(query=body.query, hits=[])

    memories = memories_db.get_memories_by_ids(uid, ids)
    by_id = {m['id']: m for m in memories}

    hits: List[MemoriesSearchHit] = []
    for rank, mid in enumerate(ids):
        m = by_id.get(mid)
        if m is None:
            continue
        hits.append(_hit_from_memory(m, rank=rank))
    return MemoriesSearchResponse(query=body.query, hits=hits)


class PromoteToEdwinRequest(BaseModel):
    note: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional Edwin-side note (e.g. why it was worth curating)",
    )


@router.post('/v3/memories/{memory_id}/promote-to-edwin', tags=['memories'])
def promote_memory_to_edwin(
    memory_id: str,
    body: PromoteToEdwinRequest,
    uid: str = Depends(auth.get_current_user_uid),
):
    """Flag an Omi memory as promoted into Edwin's curated memory store.

    Idempotent: re-calling refreshes promoted_to_edwin_at and (optionally)
    overwrites the note. Edwin is responsible for the actual filesystem
    write on its side — this endpoint only stamps the source memory so we
    can later answer "what's been promoted" queries without crawling
    Edwin's filesystem.
    """
    if memories_db.get_memory(uid, memory_id) is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    ok = memories_db.mark_memory_promoted_to_edwin(uid, memory_id, note=body.note)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {'status': 'ok', 'memory_id': memory_id}
