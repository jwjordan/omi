"""Unit tests for database/users.py — Postgres impl.

Covers the major function groups of the 80-function module:
- core profile (get/set/byok/cancellation/deletion feedback)
- people CRUD + batch fetch
- speech samples + speaker embeddings
- delete_user_data (hits every user-scoped table)
- analytics/ratings
- payments + stripe customer reverse lookup
- data protection level + migration
- language / onboarding / subscription / training-data
- task integrations + default task integration
- app integrations
- transcription preferences + assistant settings + ai profile
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

# Force a fresh import of database.users (replacing any MagicMock stub that
# another test module may have left in sys.modules via `setdefault`). This
# makes the test file independent of collection order.
for _mod in ("database.users", "database._client"):
    sys.modules.pop(_mod, None)
import database.users  # noqa: E402,F401  (populates sys.modules with the real module)


def _mock_conn():
    """Returns (conn_mock, cursor_mock) with context-manager semantics."""
    cursor_mock = MagicMock()
    cursor_mock.__enter__ = MagicMock(return_value=cursor_mock)
    cursor_mock.__exit__ = MagicMock(return_value=None)

    conn_mock = MagicMock()
    conn_mock.cursor.return_value = cursor_mock
    conn_mock.__enter__ = MagicMock(return_value=conn_mock)
    conn_mock.__exit__ = MagicMock(return_value=None)

    return conn_mock, cursor_mock


def _mock_batch(db_mock, conn):
    """Wire db.batch() to return a context manager yielding conn."""
    batch_ctx = MagicMock()
    batch_ctx.__enter__ = MagicMock(return_value=conn)
    batch_ctx.__exit__ = MagicMock(return_value=None)
    db_mock.batch.return_value = batch_ctx


# ---------------------------------------------------------------------------
# Core profile
# ---------------------------------------------------------------------------


def test_is_exists_user_returns_true_when_row_found():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)

        from database.users import is_exists_user
        assert is_exists_user("u1") is True

        sql = cur.execute.call_args.args[0]
        assert "SELECT 1 FROM users" in sql
        assert "WHERE uid = %s" in sql
        assert cur.execute.call_args.args[1] == ("u1",)


def test_is_exists_user_returns_false_when_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import is_exists_user
        assert is_exists_user("ghost") is False


def test_get_user_profile_returns_data_dict():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"language": "en", "signup_platform": "mobile"},)

        from database.users import get_user_profile
        result = get_user_profile("u1")
        assert result == {"language": "en", "signup_platform": "mobile"}


def test_get_user_profile_returns_empty_dict_when_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import get_user_profile
        assert get_user_profile("ghost") == {}


def test_set_user_store_recording_permission_upserts_users_row():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_user_store_recording_permission
        set_user_store_recording_permission("u1", True)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO users" in sql
        assert "ON CONFLICT (uid) DO UPDATE" in sql
        assert "data = users.data || EXCLUDED.data" in sql
        assert params[0] == "u1"
        assert '"store_recording_permission": true' in params[1]


def test_set_user_cancellation_feedback_stores_nested_object():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_user_cancellation_feedback
        set_user_cancellation_feedback("u1", "too_expensive", "plan is too expensive")

        params = cur.execute.call_args.args[1]
        assert "cancellation_feedback" in params[1]
        assert "too_expensive" in params[1]


def test_set_user_deletion_feedback_writes_to_account_deletions_table():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_user_deletion_feedback
        set_user_deletion_feedback("u1", "not_useful", "not what I expected")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO account_deletions" in sql
        assert "ON CONFLICT (uid)" in sql
        assert params[0] == "u1"


def test_byok_state_read_returns_byok_dict():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"byok": {"active": True, "fingerprints": {"a": "b"}}},)

        from database.users import get_byok_state
        state = get_byok_state("u1")
        assert state["active"] is True
        assert state["fingerprints"] == {"a": "b"}


def test_set_byok_active_merges_payload():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_byok_active
        set_byok_active("u1", {"anthropic": "abc123"})

        params = cur.execute.call_args.args[1]
        assert '"byok"' in params[1]
        assert '"active": true' in params[1]
        assert '"anthropic": "abc123"' in params[1]


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------


def test_create_person_upserts_user_people_row():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import create_person
        data = {"id": "p1", "name": "Alice"}
        create_person("u1", data)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO user_people" in sql
        assert "ON CONFLICT (uid, person_id)" in sql
        assert params[0] == "u1"
        assert params[1] == "p1"


def test_get_person_returns_dict_with_id():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("p1", {"name": "Alice"})

        from database.users import get_person
        result = get_person("u1", "p1")
        assert result == {"id": "p1", "name": "Alice"}


def test_get_person_returns_none_when_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import get_person
        assert get_person("u1", "p404") is None


def test_get_people_returns_list_with_ids():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("p1", {"name": "Alice"}),
            ("p2", {"name": "Bob"}),
        ]

        from database.users import get_people
        result = get_people("u1")
        assert len(result) == 2
        assert result[0]["id"] == "p1"
        assert result[0]["name"] == "Alice"
        assert result[1]["id"] == "p2"


def test_get_person_by_name_filters_on_jsonb_name():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("p1", {"name": "Alice"})

        from database.users import get_person_by_name
        result = get_person_by_name("u1", "Alice")
        sql = cur.execute.call_args.args[0]
        assert "data->>'name'" in sql
        assert result == {"id": "p1", "name": "Alice"}


def test_get_people_by_ids_uses_any_array_query():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("p1", {"name": "Alice"}),
            ("p2", {"name": "Bob"}),
        ]

        from database.users import get_people_by_ids
        result = get_people_by_ids("u1", ["p1", "p2", "p3"])

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "person_id = ANY(%s::text[])" in sql
        assert params[0] == "u1"
        assert params[1] == ["p1", "p2", "p3"]
        assert len(result) == 2


def test_get_people_by_ids_empty_returns_empty_without_query():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import get_people_by_ids
        assert get_people_by_ids("u1", []) == []
        cur.execute.assert_not_called()


def test_update_person_shallow_merges_name():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import update_person
        update_person("u1", "p1", "Alicia")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE user_people" in sql
        assert "data = data || %s::jsonb" in sql
        assert '"name": "Alicia"' in params[0]
        assert params[1] == "u1"
        assert params[2] == "p1"


def test_delete_person_removes_row():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import delete_person
        delete_person("u1", "p1")

        sql = cur.execute.call_args.args[0]
        assert "DELETE FROM user_people" in sql
        assert "WHERE uid = %s AND person_id = %s" in sql


# ---------------------------------------------------------------------------
# Speech samples and speaker embeddings
# ---------------------------------------------------------------------------


def test_add_person_speech_sample_appends_to_samples_list():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = (
            {"speech_samples": ["a.wav"], "speech_sample_transcripts": ["a"]},
        )

        from database.users import add_person_speech_sample
        result = add_person_speech_sample("u1", "p1", "b.wav", transcript="b", max_samples=5)
        assert result is True

        # Last execute should be the UPDATE with merged samples
        last_call = cur.execute.call_args_list[-1]
        sql = last_call.args[0]
        params = last_call.args[1]
        assert "UPDATE user_people" in sql
        assert "b.wav" in params[0]


def test_add_person_speech_sample_enforces_max_samples():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = (
            {"speech_samples": ["a.wav", "b.wav"], "speech_sample_transcripts": ["a", "b"]},
        )

        from database.users import add_person_speech_sample
        result = add_person_speech_sample("u1", "p1", "c.wav", transcript="c", max_samples=2)
        assert result is False


def test_add_person_speech_sample_person_not_found():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.users import add_person_speech_sample
        result = add_person_speech_sample("u1", "ghost", "x.wav", transcript="x")
        assert result is False


def test_get_person_speech_samples_count_uses_jsonb_array_length():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (3,)

        from database.users import get_person_speech_samples_count
        assert get_person_speech_samples_count("u1", "p1") == 3
        assert "jsonb_array_length" in cur.execute.call_args.args[0]


def test_remove_person_speech_sample_removes_matching_index():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = (
            {
                "speech_samples": ["a.wav", "b.wav", "c.wav"],
                "speech_sample_transcripts": ["a", "b", "c"],
            },
        )

        from database.users import remove_person_speech_sample
        result = remove_person_speech_sample("u1", "p1", "b.wav")
        assert result is True

        last_call = cur.execute.call_args_list[-1]
        sql = last_call.args[0]
        params = last_call.args[1]
        assert "UPDATE user_people" in sql
        # "b.wav" and "b" must be removed; we can assert on the JSON blob
        assert "b.wav" not in params[0]
        assert "a.wav" in params[0]
        assert "c.wav" in params[0]


def test_set_user_speaker_embedding_stores_list_in_users_data():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_user_speaker_embedding
        assert set_user_speaker_embedding("u1", [0.1, 0.2, 0.3]) is True

        params = cur.execute.call_args.args[1]
        assert "speaker_embedding" in params[1]
        assert "0.1" in params[1]


def test_get_user_speaker_embedding_round_trip():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"speaker_embedding": [0.1, 0.2, 0.3]},)

        from database.users import get_user_speaker_embedding
        assert get_user_speaker_embedding("u1") == [0.1, 0.2, 0.3]


def test_get_user_speaker_embedding_none_when_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import get_user_speaker_embedding
        assert get_user_speaker_embedding("ghost") is None


def test_set_person_speaker_embedding_returns_false_when_no_rows_updated():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.users import set_person_speaker_embedding
        assert set_person_speaker_embedding("u1", "ghost", [0.1, 0.2]) is False


def test_get_person_speaker_embedding_extracts_from_jsonb():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"speaker_embedding": [0.5, 0.6]},)

        from database.users import get_person_speaker_embedding
        assert get_person_speaker_embedding("u1", "p1") == [0.5, 0.6]


def test_clear_person_speaker_embedding_removes_key():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # First SELECT returns a row so we know person exists
        cur.fetchone.return_value = (1,)

        from database.users import clear_person_speaker_embedding
        assert clear_person_speaker_embedding("u1", "p1") is True

        last_sql = cur.execute.call_args_list[-1].args[0]
        assert "data - 'speaker_embedding'" in last_sql


# ---------------------------------------------------------------------------
# delete_user_data
# ---------------------------------------------------------------------------


def test_delete_user_data_wipes_all_user_scoped_tables():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = (1,)  # user exists

        from database.users import delete_user_data, _USER_SCOPED_TABLES
        result = delete_user_data("u1")
        assert result == {"status": "ok", "message": "Account deleted successfully"}

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = "\n".join(sqls)
        # Must hit the expected user-scoped tables + the final users delete.
        for expected in [
            "user_people",
            "account_deletions",
            "user_integrations",
            "user_task_integrations",
            "conversations",
            "memories",
            "chat_sessions",
            "chat_messages",
            "phone_calls",
            "folders",
            "fair_use_state",
            "fair_use_events",
            "conversation_vectors",
            "memory_vectors",
            "import_jobs",
        ]:
            assert expected in joined, f"expected DELETE touching {expected}, saw:\n{joined}"
        assert "DELETE FROM users WHERE uid" in joined


def test_delete_user_data_returns_error_when_user_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.users import delete_user_data
        result = delete_user_data("ghost")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Analytics / Ratings
# ---------------------------------------------------------------------------


def test_set_conversation_summary_rating_score_writes_analytics():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_conversation_summary_rating_score
        set_conversation_summary_rating_score("u1", "conv1", 1)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO analytics" in sql
        assert params[1] == "memory_summary"
        assert "conv1" in params[2]


def test_get_conversation_summary_rating_score_returns_dict():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"memory_id": "conv1", "uid": "u1", "value": 1},)

        from database.users import get_conversation_summary_rating_score
        result = get_conversation_summary_rating_score("conv1")
        assert result["memory_id"] == "conv1"


def test_get_all_ratings_queries_by_type():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [({"memory_id": "a", "value": 1},)]

        from database.users import get_all_ratings
        result = get_all_ratings("memory_summary")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "WHERE type = %s" in sql
        assert params == ("memory_summary",)
        assert len(result) == 1


def test_set_chat_message_rating_score_stores_optional_fields():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_chat_message_rating_score
        set_chat_message_rating_score(
            "u1", "m1", -1, reason="too_verbose", platform="desktop", app_version="1.0"
        )

        params = cur.execute.call_args.args[1]
        assert "too_verbose" in params[2]
        assert "desktop" in params[2]
        assert "1.0" in params[2]


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------


def test_get_stripe_customer_id_returns_field_from_jsonb():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"stripe_customer_id": "cus_abc"},)

        from database.users import get_stripe_customer_id
        assert get_stripe_customer_id("u1") == "cus_abc"


def test_get_user_by_stripe_customer_id_uses_jsonb_filter():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ("u1", {"stripe_customer_id": "cus_abc", "language": "en"})

        from database.users import get_user_by_stripe_customer_id
        result = get_user_by_stripe_customer_id("cus_abc")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data->>'stripe_customer_id'" in sql
        assert params == ("cus_abc",)
        assert result["uid"] == "u1"
        assert result["language"] == "en"


def test_get_user_by_stripe_customer_id_returns_none_when_no_match():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import get_user_by_stripe_customer_id
        assert get_user_by_stripe_customer_id("cus_missing") is None


def test_update_user_subscription_strips_dynamic_fields():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import update_user_subscription
        update_user_subscription(
            "u1",
            {"plan": "basic", "status": "active", "features": ["f1"], "limits": {"x": 1}},
        )

        params = cur.execute.call_args.args[1]
        # features & limits must be dropped before storage
        assert "features" not in params[1]
        assert "limits" not in params[1]
        assert '"plan": "basic"' in params[1]


# ---------------------------------------------------------------------------
# Data protection + migration
# ---------------------------------------------------------------------------


def test_get_data_protection_level_defaults_to_enhanced():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({},)  # row exists but no field

        from database.users import get_data_protection_level
        assert get_data_protection_level("u1") == "enhanced"


def test_set_data_protection_level_rejects_invalid_value():
    import pytest
    from database.users import set_data_protection_level
    with pytest.raises(ValueError):
        set_data_protection_level("u1", "bogus")


def test_finalize_migration_removes_migration_status_atomically():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.users import finalize_migration
        finalize_migration("u1", "e2ee")

        sql = cur.execute.call_args.args[0]
        assert "INSERT INTO users" in sql
        assert "users.data - 'migration_status'" in sql


# ---------------------------------------------------------------------------
# Language / Onboarding / Subscription
# ---------------------------------------------------------------------------


def test_get_user_language_preference_defaults_empty_string():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import get_user_language_preference
        assert get_user_language_preference("ghost") == ""


def test_set_user_language_preference_writes_top_level_key():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_user_language_preference
        set_user_language_preference("u1", "en")

        params = cur.execute.call_args.args[1]
        assert '"language": "en"' in params[1]


def test_get_user_training_data_opt_in_returns_dict():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"training_data_opt_in": {"status": "approved"}},)

        from database.users import get_user_training_data_opt_in
        assert get_user_training_data_opt_in("u1") == {"status": "approved"}


# ---------------------------------------------------------------------------
# Task integrations
# ---------------------------------------------------------------------------


def test_get_task_integrations_returns_dict_keyed_by_app_key():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ("todoist", {"token": "abc"}),
            ("asana", {"token": "def"}),
        ]

        from database.users import get_task_integrations
        result = get_task_integrations("u1")
        assert result == {"todoist": {"token": "abc"}, "asana": {"token": "def"}}


def test_get_task_integration_returns_single_record():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"token": "abc"},)

        from database.users import get_task_integration
        assert get_task_integration("u1", "todoist") == {"token": "abc"}


def test_set_task_integration_upserts_with_timestamps():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # First SELECT says "no existing row"
        cur.fetchone.return_value = None

        from database.users import set_task_integration
        set_task_integration("u1", "todoist", {"token": "abc"})

        # Last call is the INSERT
        last_call = cur.execute.call_args_list[-1]
        sql = last_call.args[0]
        params = last_call.args[1]
        assert "INSERT INTO user_task_integrations" in sql
        assert "ON CONFLICT (uid, app_key)" in sql
        assert params[0] == "u1"
        assert params[1] == "todoist"
        assert "created_at" in params[2]
        assert "updated_at" in params[2]


def test_delete_task_integration_clears_default_if_matching():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # Sequence: first fetchone -> row exists for integration, second -> default matches
        cur.fetchone.side_effect = [(1,), ("todoist",)]

        from database.users import delete_task_integration
        assert delete_task_integration("u1", "todoist") is True

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = "\n".join(sqls)
        assert "DELETE FROM user_task_integrations" in joined
        assert "data - 'default_task_integration'" in joined


def test_delete_task_integration_returns_false_when_not_found():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.users import delete_task_integration
        assert delete_task_integration("u1", "todoist") is False


def test_get_default_task_integration_reads_user_data_field():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"default_task_integration": "todoist"},)

        from database.users import get_default_task_integration
        assert get_default_task_integration("u1") == "todoist"


def test_set_default_task_integration_merges_into_users_data():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.users import set_default_task_integration
        set_default_task_integration("u1", "todoist")

        params = cur.execute.call_args.args[1]
        assert '"default_task_integration": "todoist"' in params[1]


# ---------------------------------------------------------------------------
# App integrations
# ---------------------------------------------------------------------------


def test_get_integration_returns_dict_from_user_integrations_table():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"access_token": "xyz"},)

        from database.users import get_integration
        assert get_integration("u1", "google_calendar") == {"access_token": "xyz"}


def test_set_integration_upserts_row():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None  # no existing row

        from database.users import set_integration
        set_integration("u1", "google_calendar", {"access_token": "xyz"})

        last_call = cur.execute.call_args_list[-1]
        sql = last_call.args[0]
        assert "INSERT INTO user_integrations" in sql
        assert "ON CONFLICT (uid, app_key)" in sql


def test_delete_integration_returns_true_when_row_deleted():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.users import delete_integration
        assert delete_integration("u1", "google_calendar") is True


def test_delete_integration_returns_false_when_not_found():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.users import delete_integration
        assert delete_integration("u1", "missing") is False


# ---------------------------------------------------------------------------
# Transcription preferences + assistant settings + notifications
# ---------------------------------------------------------------------------


def test_get_user_transcription_preferences_defaults_when_user_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import get_user_transcription_preferences
        result = get_user_transcription_preferences("ghost")
        assert result == {"single_language_mode": False, "vocabulary": [], "language": ""}


def test_get_user_transcription_preferences_returns_stored_values():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (
            {
                "language": "en",
                "transcription_preferences": {
                    "single_language_mode": True,
                    "vocabulary": ["omi", "gpt"],
                },
            },
        )

        from database.users import get_user_transcription_preferences
        result = get_user_transcription_preferences("u1")
        assert result["single_language_mode"] is True
        assert result["vocabulary"] == ["omi", "gpt"]
        assert result["language"] == "en"


def test_set_user_transcription_preferences_truncates_vocabulary():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ({"transcription_preferences": {}},)

        from database.users import set_user_transcription_preferences
        set_user_transcription_preferences("u1", single_language_mode=True, vocabulary=list(range(150)))

        last_call = cur.execute.call_args_list[-1]
        params = last_call.args[1]
        # 100-term cap applied
        import json as _json
        stored = _json.loads(params[1])
        vocab = stored["transcription_preferences"]["vocabulary"]
        assert len(vocab) == 100
        assert stored["transcription_preferences"]["single_language_mode"] is True


def test_get_notification_settings_maps_to_wire_names():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (
            {"notifications_enabled": False, "notification_frequency": 5},
        )

        from database.users import get_notification_settings
        result = get_notification_settings("u1")
        assert result == {"enabled": False, "frequency": 5}


def test_get_notification_settings_defaults_when_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.users import get_notification_settings
        assert get_notification_settings("ghost") == {"enabled": True, "frequency": 3}


def test_get_assistant_settings_injects_update_channel_top_level():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (
            {
                "assistant_settings": {"focus": {"enabled": True}},
                "update_channel": "beta",
            },
        )

        from database.users import get_assistant_settings
        result = get_assistant_settings("u1")
        assert result["focus"] == {"enabled": True}
        assert result["update_channel"] == "beta"


def test_update_assistant_settings_deep_merges_sections():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # 1st read returns existing; subsequent INSERT is the write
        cur.fetchone.return_value = (
            {"assistant_settings": {"focus": {"enabled": True, "target": "work"}}},
        )

        from database.users import update_assistant_settings
        result = update_assistant_settings("u1", {"focus": {"target": "study"}})
        # Deep merge: enabled preserved, target overwritten
        assert result["focus"] == {"enabled": True, "target": "study"}


def test_update_assistant_settings_extracts_update_channel_to_top_level():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({"assistant_settings": {}},)

        from database.users import update_assistant_settings
        result = update_assistant_settings("u1", {"update_channel": "stable", "focus": {"enabled": False}})
        assert result["update_channel"] == "stable"
        # Last execute should have both keys in the payload
        params = cur.execute.call_args.args[1]
        assert "update_channel" in params[1]
        assert "assistant_settings" in params[1]


def test_update_ai_user_profile_partial_update_preserves_existing_fields():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # First call reads existing profile
        cur.fetchone.return_value = (
            {"ai_user_profile": {"profile_text": "old", "data_sources_used": 3}},
        )

        from database.users import update_ai_user_profile
        result = update_ai_user_profile("u1", profile_text="new")
        assert result["profile_text"] == "new"
        assert result["data_sources_used"] == 3  # preserved


def test_get_agent_vm_returns_none_when_missing():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ({},)

        from database.users import get_agent_vm
        assert get_agent_vm("u1") is None


def test_get_agent_vm_returns_stored_dict():
    with patch("database.users.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (
            {"agentVm": {"ip": "1.2.3.4", "status": "running"}},
        )

        from database.users import get_agent_vm
        assert get_agent_vm("u1") == {"ip": "1.2.3.4", "status": "running"}
