"""Unit tests for database/memories.py — Postgres impl.

Tests exercise every SQL pathway the Omi app hits:
- create_memory: INSERT with typed columns promoted (category/scoring/is_locked)
- get_memory / get_memory not found
- get_memories: filter matrix (categories, start/end date)
- get_memories_by_ids: ANY-array batch fetch
- update_memory_fields: shallow JSONB merge + typed column updates
- review_memory / change_memory_visibility / set_memory_kg_extracted: JSONB merge
- edit_memory: FOR UPDATE, encrypts content when level=enhanced
- delete_memory / delete_memories_for_conversation / unlock_all_memories
- get_memory_ids_for_conversation: JSONB path filter
- get_memories_to_migrate / migrate_memories_level_batch
- Encryption helpers preserved
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


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
# create_memory
# ---------------------------------------------------------------------------


def test_create_memory_inserts_with_typed_columns():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.memories import create_memory
        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        payload = {
            'id': 'm1',
            'content': 'User likes dark mode.',
            'category': 'interesting',
            'scoring': 1.5,
            'is_locked': False,
            'created_at': now,
            'tags': ['ui'],
            'data_protection_level': 'standard',
        }
        create_memory(uid='u1', data=payload)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO memories" in sql
        assert "ON CONFLICT (uid, id) DO UPDATE" in sql
        # params: uid, id, category, scoring, is_locked, created_at, data_json
        assert params[0] == 'u1'
        assert params[1] == 'm1'
        assert params[2] == 'interesting'
        assert params[3] == 1.5
        assert params[4] is False
        assert params[5] == now
        stored = json.loads(params[6])
        # typed cols should NOT be duplicated inside the JSONB body
        assert 'category' not in stored
        assert 'scoring' not in stored
        assert 'is_locked' not in stored
        assert 'created_at' not in stored
        assert stored['content'] == 'User likes dark mode.'
        assert stored['tags'] == ['ui']


def test_save_memories_uses_batch_and_inserts_each():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.memories import save_memories
        now = datetime(2026, 4, 21, tzinfo=timezone.utc)
        save_memories(uid='u1', data=[
            {'id': 'm1', 'content': 'a', 'category': 'interesting', 'created_at': now,
             'data_protection_level': 'standard'},
            {'id': 'm2', 'content': 'b', 'category': 'system', 'created_at': now,
             'data_protection_level': 'standard'},
        ])

        assert cur.execute.call_count == 2
        db_mock.batch.assert_called_once()


# ---------------------------------------------------------------------------
# get_memory / get_memories
# ---------------------------------------------------------------------------


def test_get_memory_merges_typed_cols_into_returned_dict():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        cur.fetchone.return_value = (
            'm1', 'interesting', 2.0, False, now, now,
            {'content': 'hello', 'data_protection_level': 'standard'},
        )

        from database.memories import get_memory
        result = get_memory('u1', 'm1')
        assert result['id'] == 'm1'
        assert result['category'] == 'interesting'
        assert result['scoring'] == 2.0
        assert result['is_locked'] is False
        assert result['created_at'] == now
        assert result['content'] == 'hello'


def test_get_memory_returns_none_when_missing():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.memories import get_memory
        assert get_memory('u1', 'ghost') is None


def test_get_memories_applies_category_filter_as_any_array():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.memories import get_memories
        get_memories('u1', categories=['interesting', 'system'])

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "category = ANY(%s::text[])" in sql
        assert ['interesting', 'system'] in params
        assert "ORDER BY scoring DESC NULLS LAST" in sql


def test_get_memories_applies_date_range():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)

        from database.memories import get_memories
        get_memories('u1', start_date=start, end_date=end)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "created_at >= %s" in sql
        assert "created_at <= %s" in sql
        assert start in params
        assert end in params


def test_get_memories_filters_out_user_review_false():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('m1', 'interesting', 1.0, False, now, now,
             {'content': 'ok', 'data_protection_level': 'standard'}),
            ('m2', 'interesting', 1.0, False, now, now,
             {'content': 'bad', 'user_review': False, 'data_protection_level': 'standard'}),
        ]

        from database.memories import get_memories
        result = get_memories('u1')
        assert [m['id'] for m in result] == ['m1']


# ---------------------------------------------------------------------------
# get_memories_by_ids
# ---------------------------------------------------------------------------


def test_get_memories_by_ids_empty_returns_empty_without_query():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.memories import get_memories_by_ids
        assert get_memories_by_ids('u1', []) == []
        cur.execute.assert_not_called()


def test_get_memories_by_ids_uses_any_array():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('m1', 'interesting', 1.0, False, now, now,
             {'content': 'a', 'data_protection_level': 'standard'}),
        ]

        from database.memories import get_memories_by_ids
        result = get_memories_by_ids('u1', ['m1', 'm2'])
        assert len(result) == 1

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "id = ANY(%s::text[])" in sql
        assert params[0] == 'u1'
        assert params[1] == ['m1', 'm2']


# ---------------------------------------------------------------------------
# update_memory_fields / review / visibility / kg_extracted
# ---------------------------------------------------------------------------


def test_update_memory_fields_promotes_typed_cols_and_merges_jsonb():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.memories import update_memory_fields
        update_memory_fields('u1', 'm1', {
            'scoring': 3.25,
            'is_locked': True,
            'tags': ['new'],
        })

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        # JSONB merge for tags
        assert "data = data || %s::jsonb" in joined
        # Typed-column update includes scoring and is_locked
        assert "scoring = %s" in joined
        assert "is_locked = %s" in joined


def test_review_memory_merges_reviewed_and_user_review():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.memories import review_memory
        review_memory('u1', 'm1', True)
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data = data || %s::jsonb" in sql
        stored = json.loads(params[0])
        assert stored == {'reviewed': True, 'user_review': True}


def test_change_memory_visibility_merges_visibility():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.memories import change_memory_visibility
        change_memory_visibility('u1', 'm1', 'public')
        params = cur.execute.call_args.args[1]
        stored = json.loads(params[0])
        assert stored == {'visibility': 'public'}


def test_set_memory_kg_extracted_merges_flag():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.memories import set_memory_kg_extracted
        set_memory_kg_extracted('u1', 'm1')
        params = cur.execute.call_args.args[1]
        stored = json.loads(params[0])
        assert stored == {'kg_extracted': True}


# ---------------------------------------------------------------------------
# edit_memory (FOR UPDATE + encryption)
# ---------------------------------------------------------------------------


def test_edit_memory_noop_when_missing():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.memories import edit_memory
        edit_memory('u1', 'ghost', 'new content')

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        # Only the SELECT FOR UPDATE should have run; no UPDATE.
        assert any('FOR UPDATE' in s for s in sqls)
        assert not any(s.strip().upper().startswith('UPDATE') for s in sqls)


def test_edit_memory_updates_content_standard_level():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ({'data_protection_level': 'standard'},)

        from database.memories import edit_memory
        edit_memory('u1', 'm1', 'new content')

        update_call = cur.execute.call_args_list[-1]
        sql = update_call.args[0]
        params = update_call.args[1]
        assert "data = data || %s::jsonb" in sql
        stored = json.loads(params[0])
        # Standard level: content stored plain.
        assert stored['content'] == 'new content'
        assert stored['edited'] is True


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_memory_issues_delete():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.memories import delete_memory
        delete_memory('u1', 'm1')
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM memories" in sql
        assert params == ('u1', 'm1')


def test_delete_memories_for_conversation_uses_jsonb_path():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 3

        from database.memories import delete_memories_for_conversation
        delete_memories_for_conversation('u1', 'conv1')
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM memories" in sql
        assert "data->>'memory_id' = %s" in sql
        assert params == ('u1', 'conv1')


def test_get_memory_ids_for_conversation_returns_ids():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [('m1',), ('m2',)]

        from database.memories import get_memory_ids_for_conversation
        ids = get_memory_ids_for_conversation('u1', 'conv1')
        assert ids == ['m1', 'm2']
        sql = cur.execute.call_args.args[0]
        assert "data->>'memory_id' = %s" in sql


# ---------------------------------------------------------------------------
# unlock_all_memories — typed column filter
# ---------------------------------------------------------------------------


def test_unlock_all_memories_filters_and_sets_typed_column():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.memories import unlock_all_memories
        unlock_all_memories('u1')

        sql = cur.execute.call_args.args[0]
        assert "UPDATE memories" in sql
        assert "is_locked = FALSE" in sql
        assert "WHERE uid = %s AND is_locked = TRUE" in sql


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def test_get_memories_to_migrate_skips_matching_level():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ('m1', 'standard'),  # needs migration -> include
            ('m2', 'enhanced'),  # already target -> skip
            ('m3', None),        # defaults to 'standard' -> include
        ]

        from database.memories import get_memories_to_migrate
        result = get_memories_to_migrate('u1', 'enhanced')
        ids = [r['id'] for r in result]
        assert ids == ['m1', 'm3']
        for r in result:
            assert r['type'] == 'memory'


def test_migrate_memories_level_batch_noop_on_empty():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.memories import migrate_memories_level_batch
        migrate_memories_level_batch('u1', [], 'enhanced')
        db_mock.batch.assert_not_called()


def test_migrate_memories_filters_by_app_id_when_provided():
    with patch("database.memories.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchall.return_value = []

        from database.memories import migrate_memories
        result = migrate_memories('u_old', 'u_new', app_id='appX')
        # No source rows -> returns 0 (the migrate function does not insert)
        assert result == 0
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data->>'app_id' = %s" in sql
        assert 'appX' in params


# ---------------------------------------------------------------------------
# Ensure encryption helpers stay intact
# ---------------------------------------------------------------------------


def test_prepare_data_for_write_encrypts_only_on_enhanced_level():
    from database.memories import _prepare_data_for_write
    # Standard -> passthrough.
    out_std = _prepare_data_for_write({'content': 'plain'}, 'u1', 'standard')
    assert out_std['content'] == 'plain'
    # Enhanced -> content replaced with encrypted blob (just verify it changed).
    out_enh = _prepare_data_for_write({'content': 'plain'}, 'u1', 'enhanced')
    assert out_enh['content'] != 'plain'


def test_prepare_memory_for_read_passes_through_standard():
    from database.memories import _prepare_memory_for_read
    out = _prepare_memory_for_read({'content': 'hi', 'data_protection_level': 'standard'}, 'u1')
    assert out['content'] == 'hi'
