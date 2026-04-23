"""Voice-fingerprint matching.

Given an embedding extracted from a diarizer cluster, find the best
cosine-distance match against the user's own voice + everyone in
user_people. Returns identity as {is_user, person_id}.

Threshold 0.45 (matches Omi's existing SPEAKER_MATCH_THRESHOLD derived
from VoxCeleb EER 2.8%).
"""

from typing import Dict, List, Optional

import numpy as np

import database.users as users_db

SPEAKER_MATCH_THRESHOLD = 0.45


def _cosine_distance(a: List[float], b: List[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.dot(av, bv) / (na * nb))


def match(uid: str, embedding: List[float]) -> Dict[str, Optional[object]]:
    """Return {'is_user': bool, 'person_id': Optional[str]}.

    Only one of is_user / person_id can be non-None. If no candidate is
    below SPEAKER_MATCH_THRESHOLD, returns {is_user: False, person_id: None}.
    """
    best_distance = SPEAKER_MATCH_THRESHOLD
    best_kind: Optional[str] = None
    best_person_id: Optional[str] = None

    # Multi-sample storage preferred; fall back to legacy single-value field.
    user_embeddings: List[List[float]] = []
    if hasattr(users_db, 'get_user_speech_samples'):
        try:
            result = users_db.get_user_speech_samples(uid)
            if isinstance(result, list):
                user_embeddings = result
        except Exception:
            pass
    if not user_embeddings:
        single = users_db.get_user_speaker_embedding(uid)
        if single and isinstance(single, list):
            user_embeddings = [single]

    for user_emb in user_embeddings:
        if not user_emb or not isinstance(user_emb, list):
            continue
        d = _cosine_distance(user_emb, embedding)
        if d < best_distance:
            best_distance = d
            best_kind = 'user'
            best_person_id = None

    for person in users_db.get_people_with_embeddings(uid):
        for stored_emb in person.get('embeddings', []):
            if not stored_emb:
                continue
            d = _cosine_distance(stored_emb, embedding)
            if d < best_distance:
                best_distance = d
                best_kind = 'person'
                best_person_id = person.get('person_id')

    if best_kind == 'user':
        return {'is_user': True, 'person_id': None}
    if best_kind == 'person':
        return {'is_user': False, 'person_id': best_person_id}
    return {'is_user': False, 'person_id': None}
