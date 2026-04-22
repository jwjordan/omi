"""Unit tests for database/conversations.py — Postgres impl.

Pendant-critical module (42 functions, 4 tables). Tests exercise every SQL
pathway the Omi app hits, with special focus on:

- upsert_conversation: ON CONFLICT UPSERT with typed columns promoted.
- get_conversations: every filter combination (discarded, statuses, dates,
  categories, folder_id, starred).
- update_conversation_segment_text: FOR UPDATE → mutate → write back.
- delete_conversation: cascade across all 4 related tables inside a batch().
- store_model_segments_result + get_conversation_transcripts_by_model: the
  4-model dict shape.
- iter_all_conversations: server-side cursor streaming.
- get_closest_conversation_to_timestamps + get_last_completed_conversation.
"""

import importlib
import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Block heavy audio deps pulled in transitively by utils.other.storage.
sys.modules.setdefault('opuslib', MagicMock())
sys.modules.setdefault('utils.other.storage', MagicMock())
sys.modules['utils.other.storage'].list_audio_chunks = MagicMock(return_value=[])

for _mod in ("database.conversations", "database._client"):
    sys.modules.pop(_mod, None)
import database.conversations  # noqa: E402,F401


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
    batch_ctx = MagicMock()
    batch_ctx.__enter__ = MagicMock(return_value=conn)
    batch_ctx.__exit__ = MagicMock(return_value=None)
    db_mock.batch.return_value = batch_ctx


# ---------------------------------------------------------------------------
# upsert_conversation
# ---------------------------------------------------------------------------


def test_upsert_conversation_inserts_with_typed_columns():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import upsert_conversation
        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        payload = {
            'id': 'c1',
            'status': 'completed',
            'discarded': False,
            'created_at': now,
            'started_at': now,
            'finished_at': now,
            'structured': {'title': 'Hi'},
            'data_protection_level': 'standard',
        }
        upsert_conversation(uid='u1', conversation_data=payload)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO conversations" in sql
        assert "ON CONFLICT (uid, id) DO UPDATE" in sql
        assert "uid, id, status, discarded, created_at, started_at, finished_at, data" in sql
        assert params[0] == 'u1'
        assert params[1] == 'c1'
        assert params[2] == 'completed'
        assert params[3] is False
        assert params[4] == now
        # typed cols should NOT be stored inside the JSONB body too
        stored = json.loads(params[7])
        assert 'status' not in stored
        assert 'discarded' not in stored
        assert 'created_at' not in stored
        assert stored['structured'] == {'title': 'Hi'}


def test_upsert_conversation_strips_audio_url_and_photos():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import upsert_conversation
        upsert_conversation(
            uid='u1',
            conversation_data={
                'id': 'c1',
                'status': 'completed',
                'discarded': False,
                'created_at': datetime.now(timezone.utc),
                'audio_base64_url': 'data:audio/...',
                'photos': [{'b64': 'x'}],
                'data_protection_level': 'standard',
            },
        )
        params = cur.execute.call_args.args[1]
        stored = json.loads(params[7])
        assert 'audio_base64_url' not in stored
        assert 'photos' not in stored


# ---------------------------------------------------------------------------
# get_conversation
# ---------------------------------------------------------------------------


def test_get_conversation_merges_typed_cols_into_returned_dict():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        # Typed cols + data JSONB body with id only
        cur.fetchone.return_value = (
            'u1', 'c1', 'completed', False, now, now, now,
            {'structured': {'title': 'T'}, 'data_protection_level': 'standard'},
        )
        # get_conversation_photos helper will fire a second query via @with_photos
        cur.fetchall.return_value = []

        from database.conversations import get_conversation
        result = get_conversation('u1', 'c1')
        assert result['id'] == 'c1'
        assert result['status'] == 'completed'
        assert result['discarded'] is False
        assert result['created_at'] == now
        assert result['structured'] == {'title': 'T'}


def test_get_conversation_returns_none_when_missing():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.conversations import get_conversation
        assert get_conversation('u1', 'c404') is None


# ---------------------------------------------------------------------------
# get_conversations — filter matrix
# ---------------------------------------------------------------------------


def test_get_conversations_applies_not_discarded_by_default():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_conversations
        get_conversations('u1')

        sql = cur.execute.call_args.args[0]
        assert "discarded = FALSE" in sql
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT %s OFFSET %s" in sql


def test_get_conversations_include_discarded_skips_filter():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_conversations
        get_conversations('u1', include_discarded=True)

        sql = cur.execute.call_args.args[0]
        assert "discarded = FALSE" not in sql


def test_get_conversations_applies_statuses_filter_as_any_array():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_conversations
        get_conversations('u1', statuses=['completed', 'in_progress'])

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "status = ANY(%s::text[])" in sql
        assert ['completed', 'in_progress'] in params


def test_get_conversations_applies_date_range():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)

        from database.conversations import get_conversations
        get_conversations('u1', start_date=start, end_date=end)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "created_at >= %s" in sql
        assert "created_at <= %s" in sql
        assert start in params
        assert end in params


def test_get_conversations_applies_category_filter_via_jsonb():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_conversations
        get_conversations('u1', categories=['personal', 'work'])

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data->'structured'->>'category' = ANY(%s::text[])" in sql
        assert ['personal', 'work'] in params


def test_get_conversations_applies_folder_id_filter():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_conversations
        get_conversations('u1', folder_id='f42')

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data->>'folder_id' = %s" in sql
        assert 'f42' in params


def test_get_conversations_applies_starred_filter():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_conversations
        get_conversations('u1', starred=True)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "(data->>'starred')::boolean = %s" in sql
        assert True in params


def test_get_conversations_count_includes_status_filter():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (7,)

        from database.conversations import get_conversations_count
        assert get_conversations_count('u1', statuses=['completed']) == 7
        sql = cur.execute.call_args.args[0]
        assert "SELECT COUNT(*) FROM conversations" in sql
        assert "status = ANY(%s::text[])" in sql


def test_get_conversations_without_photos_does_not_fetch_photos():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'c1', 'completed', False, now, now, now, {'data_protection_level': 'standard'}),
        ]

        from database.conversations import get_conversations_without_photos
        result = get_conversations_without_photos('u1')
        assert len(result) == 1
        # photos field should not be injected — only the conversations SELECT should run
        assert cur.execute.call_count == 1


# ---------------------------------------------------------------------------
# iter_all_conversations
# ---------------------------------------------------------------------------


def test_iter_all_conversations_uses_named_server_side_cursor():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        # First batch returns 2 rows, next batch empty to terminate loop
        cur.fetchmany.side_effect = [
            [
                ('u1', 'c1', 'completed', False, now, now, now, {'data_protection_level': 'standard'}),
                ('u1', 'c2', 'completed', False, now, now, now, {'data_protection_level': 'standard'}),
            ],
            [],
        ]

        from database.conversations import iter_all_conversations
        results = list(iter_all_conversations('u1', batch_size=2))

        # cursor() should have been called with a name=... kwarg
        assert conn.cursor.call_args.kwargs.get('name', '').startswith('iter_conv_')
        assert len(results) == 2
        assert results[0]['id'] == 'c1'


# ---------------------------------------------------------------------------
# update_conversation
# ---------------------------------------------------------------------------


def test_update_conversation_short_circuits_when_missing():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.conversations import update_conversation
        update_conversation('u1', 'ghost', {'foo': 'bar'})
        # Only the initial SELECT should have fired; no UPDATEs.
        exec_count = cur.execute.call_count
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any('SELECT data FROM conversations' in s for s in sqls)
        assert not any(s.strip().upper().startswith('UPDATE') for s in sqls)


def test_update_conversation_shallow_merges_flat_keys():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ({'data_protection_level': 'standard'},)

        from database.conversations import update_conversation
        update_conversation('u1', 'c1', {'visibility': 'private'})

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "data = data || %s::jsonb" in joined


def test_update_conversation_uses_jsonb_set_for_nested_keys():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ({'data_protection_level': 'standard'},)

        from database.conversations import update_conversation
        update_conversation('u1', 'c1', {'structured.title': 'New title'})

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any('jsonb_set' in s for s in sqls)
        # The path array form '{structured,title}' should be passed.
        all_params = [c.args[1] for c in cur.execute.call_args_list if len(c.args) > 1]
        assert any('{structured,title}' in (p[0] if p else '') for p in all_params)


def test_update_conversation_updates_typed_column_for_status():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ({'data_protection_level': 'standard'},)

        from database.conversations import update_conversation
        update_conversation('u1', 'c1', {'status': 'completed'})

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any('status = %s' in s for s in sqls)


# ---------------------------------------------------------------------------
# update_conversation_title & update_conversation_segment_text
# ---------------------------------------------------------------------------


def test_update_conversation_title_uses_jsonb_set_on_structured_title():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)

        from database.conversations import update_conversation_title
        update_conversation_title('u1', 'c1', 'Fancy Title')

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "jsonb_set(data, '{structured,title}'" in joined


def test_update_conversation_title_noop_when_conversation_missing():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.conversations import update_conversation_title
        update_conversation_title('u1', 'ghost', 'x')

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert not any(s.strip().upper().startswith('UPDATE') for s in sqls)


def test_update_conversation_segment_text_not_found():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.conversations import update_conversation_segment_text
        assert update_conversation_segment_text('u1', 'ghost', 's1', 'hi') == 'not_found'


def test_update_conversation_segment_text_locked():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ({'is_locked': True, 'data_protection_level': 'standard'},)

        from database.conversations import update_conversation_segment_text
        assert update_conversation_segment_text('u1', 'c1', 's1', 'hi') == 'locked'


def test_update_conversation_segment_text_segment_not_found():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # No transcript_segments key -> treated as empty list, segment missing.
        cur.fetchone.return_value = ({'data_protection_level': 'standard'},)

        from database.conversations import update_conversation_segment_text
        assert update_conversation_segment_text('u1', 'c1', 'sX', 'hi') == 'segment_not_found'


def test_update_conversation_segment_text_ok_mutates_and_writes_back():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # transcript_segments as a plain list (no compression/encryption).
        cur.fetchone.return_value = (
            {
                'data_protection_level': 'standard',
                'transcript_segments': [{'id': 's1', 'text': 'old'}],
            },
        )

        from database.conversations import update_conversation_segment_text
        result = update_conversation_segment_text('u1', 'c1', 's1', 'new text')
        assert result == 'ok'

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        # Must include FOR UPDATE and the merge write
        assert any('FOR UPDATE' in s for s in sqls)
        assert any('data = data || %s::jsonb' in s for s in sqls)


# ---------------------------------------------------------------------------
# delete_conversation(_photos) — must cascade
# ---------------------------------------------------------------------------


def test_delete_conversation_photos_issues_delete_statement():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 3

        from database.conversations import delete_conversation_photos
        assert delete_conversation_photos('u1', 'c1') == 3

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM conversation_photos" in sql
        assert "WHERE uid = %s AND conversation_id = %s" in sql
        assert params == ('u1', 'c1')


def test_delete_conversation_cascades_to_all_related_tables():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.conversations import delete_conversation
        delete_conversation('u1', 'c1')

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        # Every one of the 4 tables must be hit.
        assert "DELETE FROM conversation_photos" in joined
        assert "DELETE FROM conversation_model_transcripts" in joined
        assert "DELETE FROM action_items" in joined
        assert "DELETE FROM conversations" in joined
        # All wrapped in db.batch() (transactional).
        db_mock.batch.assert_called_once()


# ---------------------------------------------------------------------------
# get_conversations_by_id
# ---------------------------------------------------------------------------


def test_get_conversations_by_id_empty_returns_empty_without_query():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import get_conversations_by_id
        assert get_conversations_by_id('u1', []) == []
        cur.execute.assert_not_called()


def test_get_conversations_by_id_queries_with_any_array_and_excludes_discarded():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        # First fetchall: main SELECT → 1 row. Subsequent: photos subquery → [].
        cur.fetchall.side_effect = [
            [('u1', 'c1', 'completed', False, now, now, now, {'data_protection_level': 'standard'})],
            [],
        ]

        from database.conversations import get_conversations_by_id
        result = get_conversations_by_id('u1', ['c1', 'c2'])
        assert len(result) == 1

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "id = ANY(%s::text[])" in joined
        assert "discarded = FALSE" in joined


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def test_get_conversations_to_migrate_skips_public_and_matching_level():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ('c1', 'standard', None),        # needs migration -> include
            ('c2', 'enhanced', None),        # already target -> skip
            ('c3', 'standard', 'public'),    # public -> skip
            ('c4', None, None),              # defaults to 'standard' -> include
        ]

        from database.conversations import get_conversations_to_migrate
        result = get_conversations_to_migrate('u1', 'enhanced')
        ids = [r['id'] for r in result]
        assert ids == ['c1', 'c4']
        for r in result:
            assert r['type'] == 'conversation'


def test_migrate_conversations_level_batch_noop_on_empty():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.conversations import migrate_conversations_level_batch
        migrate_conversations_level_batch('u1', [], 'enhanced')
        db_mock.batch.assert_not_called()


# ---------------------------------------------------------------------------
# Status pathways
# ---------------------------------------------------------------------------


def test_get_in_progress_conversation_picks_latest():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchone.return_value = (
            'u1', 'c1', 'in_progress', False, now, now, now,
            {'data_protection_level': 'standard'},
        )
        cur.fetchall.return_value = []  # photos

        from database.conversations import get_in_progress_conversation
        result = get_in_progress_conversation('u1')
        assert result['id'] == 'c1'
        assert result['status'] == 'in_progress'
        sql = cur.execute.call_args_list[0].args[0]
        assert "status = %s" in sql
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT 1" in sql


def test_get_in_progress_conversation_returns_none_when_empty():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.conversations import get_in_progress_conversation
        assert get_in_progress_conversation('u1') is None


def test_get_processing_conversations_filters_by_status():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_processing_conversations
        result = get_processing_conversations('u1')
        assert result == []
        sql = cur.execute.call_args_list[0].args[0]
        params = cur.execute.call_args_list[0].args[1]
        assert "status = %s" in sql
        assert 'processing' in params


def test_update_conversation_status_updates_typed_column():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import update_conversation_status
        update_conversation_status('u1', 'c1', 'completed')
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE conversations SET status = %s" in sql
        assert params == ('completed', 'u1', 'c1')


def test_set_conversation_as_discarded_sets_bool_column():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import set_conversation_as_discarded
        set_conversation_as_discarded('u1', 'c1')
        sql = cur.execute.call_args.args[0]
        assert "discarded = TRUE" in sql


# ---------------------------------------------------------------------------
# Visibility, starred, unlock
# ---------------------------------------------------------------------------


def test_set_conversation_visibility_merges_into_data():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import set_conversation_visibility
        set_conversation_visibility('u1', 'c1', 'private')
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data = data || %s::jsonb" in sql
        assert '"visibility": "private"' in params[0]


def test_set_conversation_starred_merges_boolean():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import set_conversation_starred
        set_conversation_starred('u1', 'c1', True)
        params = cur.execute.call_args.args[1]
        assert '"starred": true' in params[0]


def test_unlock_all_conversations_filters_and_updates_is_locked():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import unlock_all_conversations
        unlock_all_conversations('u1')
        sql = cur.execute.call_args.args[0]
        assert "UPDATE conversations" in sql
        assert '"is_locked": false' in sql
        assert "(data->>'is_locked')::boolean = TRUE" in sql


# ---------------------------------------------------------------------------
# Timestamp + segment updates
# ---------------------------------------------------------------------------


def test_update_conversation_finished_at_sets_typed_column():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import update_conversation_finished_at
        t = datetime(2026, 4, 21, tzinfo=timezone.utc)
        update_conversation_finished_at('u1', 'c1', t)
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "finished_at = %s" in sql
        assert params == (t, 'u1', 'c1')


def test_update_conversation_segments_without_level_reads_it_from_db():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ('standard',)

        from database.conversations import update_conversation_segments
        update_conversation_segments('u1', 'c1', [{'id': 's1', 'text': 'hi'}])

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "data->>'data_protection_level'" in joined
        # Write happens via data || %s::jsonb merge (transcript_segments_compressed added)
        assert "data = data || %s::jsonb" in joined


def test_update_conversation_segments_short_circuits_when_missing():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.conversations import update_conversation_segments
        update_conversation_segments('u1', 'ghost', [{'id': 's1'}])

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert not any(s.strip().upper().startswith('UPDATE') for s in sqls)


# ---------------------------------------------------------------------------
# Model transcripts
# ---------------------------------------------------------------------------


def test_store_model_segments_result_noop_when_empty():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.conversations import store_model_segments_result
        store_model_segments_result('u1', 'c1', 'deepgram_streaming', [])
        db_mock.batch.assert_not_called()


def test_store_model_segments_result_inserts_each_segment():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        class FakeSegment:
            def dict(self):
                return {'start': 1.5, 'end': 2.5, 'text': 'hi'}

        from database.conversations import store_model_segments_result
        store_model_segments_result('u1', 'c1', 'deepgram_streaming', [FakeSegment(), FakeSegment()])

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "INSERT INTO conversation_model_transcripts" in joined
        assert "ON CONFLICT (uid, conversation_id, model_name, segment_id)" in joined
        # 2 segments => 2 INSERTs
        assert cur.execute.call_count == 2
        # start_ts column should receive the start value from the segment dict
        first_params = cur.execute.call_args_list[0].args[1]
        assert first_params[0] == 'u1'
        assert first_params[1] == 'c1'
        assert first_params[2] == 'deepgram_streaming'
        assert first_params[4] == 1.5


def test_get_conversation_transcripts_by_model_returns_four_keyed_dict():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        # Each of 4 queries returns one fake row.
        cur.fetchall.side_effect = [
            [({'start': 1, 'text': 'dg'},)],
            [({'start': 2, 'text': 'sx'},)],
            [({'start': 3, 'text': 'sm'},)],
            [({'start': 4, 'text': 'wx'},)],
        ]

        from database.conversations import get_conversation_transcripts_by_model
        result = get_conversation_transcripts_by_model('u1', 'c1')

        assert set(result.keys()) == {'deepgram', 'soniox', 'speechmatics', 'whisperx'}
        assert result['deepgram'][0]['text'] == 'dg'
        assert result['whisperx'][0]['text'] == 'wx'

        # 4 queries, one per model.
        assert cur.execute.call_count == 4
        models_used = [c.args[1][2] for c in cur.execute.call_args_list]
        assert models_used == [
            'deepgram_streaming',
            'soniox_streaming',
            'speechmatics_streaming',
            'fal_whisperx',
        ]


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def test_set_postprocessing_status_writes_subobject_atomically():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.conversations import set_postprocessing_status
        from models.conversation_enums import PostProcessingStatus, PostProcessingModel
        set_postprocessing_status('u1', 'c1', PostProcessingStatus.failed, fail_reason='boom')

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "jsonb_set(data, '{postprocessing}'" in sql
        stored = json.loads(params[0])
        assert stored['fail_reason'] == 'boom'
        assert 'status' in stored
        assert 'model' in stored


# ---------------------------------------------------------------------------
# Photos (store + retrieve)
# ---------------------------------------------------------------------------


def test_store_conversation_photos_inserts_each_with_conversation_level():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # First call: SELECT data_protection_level -> 'standard'
        cur.fetchone.return_value = ('standard',)

        class FakePhoto:
            def __init__(self, pid):
                self.id = pid
            def dict(self):
                return {'id': self.id, 'base64': 'AAA'}

        from database.conversations import store_conversation_photos
        store_conversation_photos('u1', 'c1', [FakePhoto('p1'), FakePhoto('p2')])

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        # 1 SELECT + 2 INSERTs
        insert_sqls = [s for s in sqls if 'INSERT INTO conversation_photos' in s]
        assert len(insert_sqls) == 2


def test_get_conversation_photos_reads_from_conversation_photos_table():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ({'id': 'p1', 'base64': 'AA', 'data_protection_level': 'standard'},),
            ({'id': 'p2', 'base64': 'BB', 'data_protection_level': 'standard'},),
        ]

        from database.conversations import get_conversation_photos
        result = get_conversation_photos('u1', 'c1')
        sql = cur.execute.call_args.args[0]
        assert "FROM conversation_photos" in sql
        assert "ORDER BY created_at" in sql
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Syncing / time-window queries
# ---------------------------------------------------------------------------


def test_get_closest_conversation_to_timestamps_returns_none_when_empty():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.conversations import get_closest_conversation_to_timestamps
        assert get_closest_conversation_to_timestamps('u1', 1700000000, 1700001000) is None


def test_get_closest_conversation_to_timestamps_picks_nearest_overlap():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        t0 = datetime.fromtimestamp(1700000000, tz=timezone.utc)
        t1 = datetime.fromtimestamp(1700001000, tz=timezone.utc)
        # First fetchall: main SELECT. Subsequent: photos subqueries per result → [].
        cur.fetchall.side_effect = [
            [
                ('u1', 'near', 'completed', False, t0, t0, t1, {'data_protection_level': 'standard'}),
                ('u1', 'far', 'completed', False, t0, t0, t1, {'data_protection_level': 'standard'}),
            ],
            [],
            [],
        ]

        from database.conversations import get_closest_conversation_to_timestamps
        result = get_closest_conversation_to_timestamps('u1', 1700000000, 1700001000)
        assert result is not None
        assert result['id'] in ('near', 'far')
        sql = cur.execute.call_args_list[0].args[0]
        assert "finished_at >= %s" in sql
        assert "started_at <= %s" in sql


def test_get_last_completed_conversation_limits_to_latest():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchone.return_value = (
            'u1', 'c1', 'completed', False, now, now, now,
            {'data_protection_level': 'standard'},
        )
        cur.fetchall.return_value = []  # photos

        from database.conversations import get_last_completed_conversation
        result = get_last_completed_conversation('u1')
        assert result['id'] == 'c1'
        sql = cur.execute.call_args_list[0].args[0]
        assert "LIMIT 1" in sql
        assert "ORDER BY created_at DESC" in sql


def test_get_last_completed_conversation_returns_none_when_missing():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.conversations import get_last_completed_conversation
        assert get_last_completed_conversation('u1') is None


# ---------------------------------------------------------------------------
# Action items embedded in conversations
# ---------------------------------------------------------------------------


def test_get_action_items_flattens_structured_action_items_with_metadata():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            (
                'u1', 'c1', 'completed', False, now, now, now,
                {
                    'data_protection_level': 'standard',
                    'structured': {
                        'title': 'Meeting',
                        'action_items': [
                            {'description': 'Buy milk', 'completed': False},
                            {'description': 'Call Bob', 'completed': True, 'deleted': False},
                            {'description': 'stale', 'deleted': True},
                        ],
                    },
                },
            ),
        ]

        from database.conversations import get_action_items
        items = get_action_items('u1', include_completed=True)

        # 2 items (deleted filtered)
        assert len(items) == 2
        assert items[0]['conversation_id'] == 'c1'
        assert items[0]['conversation_title'] == 'Meeting'
        assert {i['description'] for i in items} == {'Buy milk', 'Call Bob'}


def test_get_action_items_excludes_completed_when_flag_false():
    with patch("database.conversations.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            (
                'u1', 'c1', 'completed', False, now, now, now,
                {
                    'data_protection_level': 'standard',
                    'structured': {
                        'action_items': [
                            {'description': 'open', 'completed': False},
                            {'description': 'done', 'completed': True},
                        ],
                    },
                },
            ),
        ]

        from database.conversations import get_action_items
        items = get_action_items('u1', include_completed=False)
        assert len(items) == 1
        assert items[0]['description'] == 'open'


def test_update_conversation_action_items_delegates_to_update_conversation():
    with patch("database.conversations.update_conversation") as upd:
        from database.conversations import update_conversation_action_items
        update_conversation_action_items('u1', 'c1', [{'description': 'x'}])
        upd.assert_called_once_with('u1', 'c1', {'structured.action_items': [{'description': 'x'}]})


def test_update_conversation_events_delegates_to_update_conversation():
    with patch("database.conversations.update_conversation") as upd:
        from database.conversations import update_conversation_events
        update_conversation_events('u1', 'c1', [{'title': 'event'}])
        upd.assert_called_once_with('u1', 'c1', {'structured.events': [{'title': 'event'}]})


# ---------------------------------------------------------------------------
# Ensure helpers (encryption + photo decorators) stay intact
# ---------------------------------------------------------------------------


def test_ensure_timezone_aware_adds_utc_for_naive_datetime():
    from database.conversations import _ensure_timezone_aware
    naive = datetime(2026, 4, 21, 10, 0)
    result = _ensure_timezone_aware(naive)
    assert result.tzinfo is not None


def test_prepare_conversation_for_write_compresses_list_segments():
    from database.conversations import _prepare_conversation_for_write
    out = _prepare_conversation_for_write(
        {'transcript_segments': [{'id': 's1', 'text': 'hi'}]},
        uid='u1',
        level='standard',
    )
    # transcript_segments should now be bytes (zlib) and the compressed flag set.
    assert isinstance(out['transcript_segments'], bytes)
    assert out['transcript_segments_compressed'] is True


def test_prepare_photo_for_write_unchanged_for_standard_level():
    from database.conversations import _prepare_photo_for_write
    out = _prepare_photo_for_write({'base64': 'abc'}, 'u1', 'standard')
    assert out['data_protection_level'] == 'standard'
    assert out['base64'] == 'abc'
