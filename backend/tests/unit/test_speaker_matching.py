"""Tests for utils/stt/speaker_matching.py."""

from unittest.mock import patch

import numpy as np


def _vec(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_no_candidates_returns_no_match():
    from utils.stt.speaker_matching import match

    with patch("utils.stt.speaker_matching.users_db") as users_db:
        users_db.get_user_speaker_embedding.return_value = None
        users_db.get_people_with_embeddings.return_value = []
        result = match(uid="u1", embedding=_vec(256, 1))
        assert result == {"is_user": False, "person_id": None}


def test_matches_user_when_cosine_below_threshold():
    from utils.stt.speaker_matching import match

    user_emb = _vec(256, 42)
    with patch("utils.stt.speaker_matching.users_db") as users_db:
        users_db.get_user_speaker_embedding.return_value = user_emb
        users_db.get_people_with_embeddings.return_value = []
        # Query with a near-identical vector.
        query = user_emb.copy()
        result = match(uid="u1", embedding=query)
        assert result == {"is_user": True, "person_id": None}


def test_no_match_when_cosine_above_threshold():
    from utils.stt.speaker_matching import match

    user_emb = _vec(256, 42)
    far_emb = _vec(256, 99)  # different seed → different vector
    with patch("utils.stt.speaker_matching.users_db") as users_db:
        users_db.get_user_speaker_embedding.return_value = user_emb
        users_db.get_people_with_embeddings.return_value = []
        result = match(uid="u1", embedding=far_emb)
        assert result == {"is_user": False, "person_id": None}


def test_picks_closest_across_people():
    from utils.stt.speaker_matching import match

    a_emb = _vec(256, 10)
    b_emb = _vec(256, 20)
    with patch("utils.stt.speaker_matching.users_db") as users_db:
        users_db.get_user_speaker_embedding.return_value = None
        # Person B has an embedding identical to our query; person A is far.
        users_db.get_people_with_embeddings.return_value = [
            {"person_id": "a", "embeddings": [a_emb]},
            {"person_id": "b", "embeddings": [b_emb]},
        ]
        result = match(uid="u1", embedding=b_emb.copy())
        assert result == {"is_user": False, "person_id": "b"}


def test_user_beats_person_when_both_match():
    """If both the user's own embedding AND a person's embedding match,
    picks whichever has the closer cosine distance."""
    from utils.stt.speaker_matching import match

    user_emb = _vec(256, 7)
    # Person's embedding is slightly perturbed version of user.
    person_emb = [x + 0.01 for x in user_emb]
    with patch("utils.stt.speaker_matching.users_db") as users_db:
        users_db.get_user_speaker_embedding.return_value = user_emb
        users_db.get_people_with_embeddings.return_value = [
            {"person_id": "person1", "embeddings": [person_emb]},
        ]
        result = match(uid="u1", embedding=user_emb.copy())
        # user_emb is exact — distance ~0 — beats the perturbed person.
        assert result == {"is_user": True, "person_id": None}


def test_multiple_embeddings_per_person_uses_closest():
    from utils.stt.speaker_matching import match

    far_emb = _vec(256, 100)
    near_emb = _vec(256, 200)
    with patch("utils.stt.speaker_matching.users_db") as users_db:
        users_db.get_user_speaker_embedding.return_value = None
        users_db.get_people_with_embeddings.return_value = [
            {"person_id": "p", "embeddings": [far_emb, near_emb]},
        ]
        result = match(uid="u1", embedding=near_emb.copy())
        assert result == {"is_user": False, "person_id": "p"}
