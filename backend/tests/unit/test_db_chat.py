"""Unit tests for database/chat.py — Postgres impl.

Covers all four tables the module touches (chat_messages, chat_sessions,
chat_files, and the cross-table conversations read for
`include_conversations=True`). Tests focus on:

- add_message: ON CONFLICT UPSERT with typed columns promoted; `memories`
  stripped from the JSONB body.
- get_messages: every filter permutation (plugin_id scope, session scope,
  include_conversations, include_files hydration).
- get_app_messages: plugin_id scoping + reported filter + conversation hydrate.
- iter_all_messages: server-side cursor streaming.
- Rating update / report: JSONB shallow-merge paths.
- clear_chat / batch_delete_messages: bulk DELETE.
- Files: add_multi, get with IDs, get all, ordered desc, delete_multi.
- Sessions: create, acquire, get (active / by-id), delete with cascade,
  update (returns None when missing), get_chat_sessions filter matrix.
- Migration helpers: get_chats_to_migrate, migrate_chats_level_batch.
- v2 save_message / delete_messages.
"""

import importlib
import json
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Block transitive heavy deps in case something imports them later.
sys.modules.setdefault('opuslib', MagicMock())

# Reload the module fresh so our patches bind to the current `db` symbol.
for _mod in ("database.chat", "database._client"):
    sys.modules.pop(_mod, None)
import database.chat  # noqa: E402,F401


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
# add_message
# ---------------------------------------------------------------------------


def test_add_message_inserts_with_typed_columns_and_strips_memories():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import add_message
        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        payload = {
            'id': 'm1',
            'text': 'hi',
            'created_at': now,
            'sender': 'human',
            'type': 'text',
            'app_id': 'app-1',
            'plugin_id': 'app-1',
            'chat_session_id': 's1',
            'memories': [{'id': 'c1'}],  # should be stripped
            'memories_id': ['c1'],
            'data_protection_level': 'standard',
        }
        add_message(uid='u1', message_data=payload)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO chat_messages" in sql
        assert "ON CONFLICT (uid, id) DO UPDATE" in sql
        assert "uid, id, session_id, plugin_id, created_at, data" in sql
        assert params[0] == 'u1'
        assert params[1] == 'm1'
        assert params[2] == 's1'  # session_id typed col
        assert params[3] == 'app-1'  # plugin_id typed col
        assert params[4] == now

        stored = json.loads(params[5])
        assert 'memories' not in stored
        assert stored['memories_id'] == ['c1']
        # created_at should be in the typed col, not the body
        assert 'created_at' not in stored


def test_add_message_generates_id_when_missing():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import add_message
        payload = {
            'text': 'hi',
            'sender': 'ai',
            'created_at': datetime.now(timezone.utc),
            'data_protection_level': 'standard',
        }
        add_message(uid='u1', message_data=payload)
        params = cur.execute.call_args.args[1]
        assert params[1]  # an id was generated
        assert len(params[1]) > 0


# ---------------------------------------------------------------------------
# get_app_messages
# ---------------------------------------------------------------------------


def test_get_app_messages_filters_by_plugin_id_and_orders_desc():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'm1', 's1', 'app-1', now,
             {'memories_id': [], 'data_protection_level': 'standard', 'text': 'hi'}),
        ]

        from database.chat import get_app_messages
        result = get_app_messages('u1', 'app-1', limit=5, offset=0)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM chat_messages" in sql
        assert "plugin_id IS NOT DISTINCT FROM %s" in sql
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT %s OFFSET %s" in sql
        assert params == ('u1', 'app-1', 5, 0)
        assert len(result) == 1
        assert result[0]['id'] == 'm1'


def test_get_app_messages_excludes_reported():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'm1', None, 'app', now, {'reported': True, 'data_protection_level': 'standard'}),
            ('u1', 'm2', None, 'app', now, {'data_protection_level': 'standard'}),
        ]

        from database.chat import get_app_messages
        result = get_app_messages('u1', 'app')
        assert [m['id'] for m in result] == ['m2']


def test_get_app_messages_hydrates_conversations_when_requested():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        # First SELECT: messages. Second SELECT: conversations by id.
        cur.fetchall.side_effect = [
            [
                ('u1', 'm1', None, 'app', now,
                 {'memories_id': ['c1', 'c2'], 'data_protection_level': 'standard'}),
            ],
            [('c1', {'title': 'A'}), ('c2', {'title': 'B'})],
        ]

        from database.chat import get_app_messages
        result = get_app_messages('u1', 'app', include_conversations=True)

        assert len(result) == 1
        assert {c['title'] for c in result[0]['memories']} == {'A', 'B'}
        # Second call targets conversations table with ANY(text[]).
        second_sql = cur.execute.call_args_list[1].args[0]
        assert "FROM conversations" in second_sql
        assert "id = ANY(%s::text[])" in second_sql


# ---------------------------------------------------------------------------
# get_messages — filter matrix
# ---------------------------------------------------------------------------


def test_get_messages_session_scope_skips_plugin_filter():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.chat import get_messages
        get_messages('u1', chat_session_id='s1', app_id='should-be-ignored')

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        # Session-scoped: only session filter, NOT plugin_id WHERE clause
        assert "session_id = %s" in sql
        assert "plugin_id IS NOT DISTINCT FROM" not in sql
        assert 's1' in params


def test_get_messages_app_scope_uses_plugin_is_not_distinct_from():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.chat import get_messages
        get_messages('u1', app_id=None)  # main chat

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "plugin_id IS NOT DISTINCT FROM %s" in sql
        assert None in params


def test_get_messages_include_conversations_and_files_fans_out_queries():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.side_effect = [
            # messages
            [
                ('u1', 'm1', None, 'app', now,
                 {'memories_id': ['c1'], 'files_id': ['f1'],
                  'data_protection_level': 'standard'}),
            ],
            # conversations
            [('c1', {'title': 'C'})],
            # files
            [('u1', 'f1', now, {'name': 'doc.pdf'})],
        ]

        from database.chat import get_messages
        result = get_messages('u1', app_id='app', include_conversations=True)

        assert len(result) == 1
        assert result[0]['memories'][0]['title'] == 'C'
        assert result[0]['files'][0]['name'] == 'doc.pdf'


def test_get_messages_skips_reported():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'm1', None, 'app', now, {'reported': True, 'data_protection_level': 'standard'}),
            ('u1', 'm2', None, 'app', now, {'data_protection_level': 'standard'}),
        ]

        from database.chat import get_messages
        result = get_messages('u1', app_id='app')
        assert [m['id'] for m in result] == ['m2']


# ---------------------------------------------------------------------------
# get_message_count + iter_all_messages + get_message
# ---------------------------------------------------------------------------


def test_get_message_count_returns_int():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (42,)

        from database.chat import get_message_count
        assert get_message_count('u1') == 42
        sql = cur.execute.call_args.args[0]
        assert "SELECT COUNT(*) FROM chat_messages" in sql


def test_get_message_count_returns_zero_when_no_rows():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.chat import get_message_count
        assert get_message_count('u1') == 0


def test_iter_all_messages_uses_named_server_side_cursor():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchmany.side_effect = [
            [
                ('u1', 'm1', None, 'app', now, {'data_protection_level': 'standard'}),
                ('u1', 'm2', None, 'app', now, {'data_protection_level': 'standard'}),
            ],
            [],
        ]

        from database.chat import iter_all_messages
        results = list(iter_all_messages('u1', batch_size=2))
        # cursor(name=...) should be used for server-side cursor
        assert conn.cursor.call_args.kwargs.get('name', '').startswith('iter_msgs_')
        assert len(results) == 2
        assert results[0]['id'] == 'm1'


def test_get_message_returns_none_when_missing():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.chat import get_message
        assert get_message('u1', 'ghost') is None


def test_get_message_returns_message_and_doc_id_tuple():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)
        cur.fetchone.return_value = (
            'u1', 'm1', None, 'app-1', now,
            {'text': 'hi', 'sender': 'human', 'type': 'text',
             'data_protection_level': 'standard'},
        )

        from database.chat import get_message
        result = get_message('u1', 'm1')
        assert result is not None
        msg, doc_id = result
        assert doc_id == 'm1'
        assert msg.id == 'm1'
        assert msg.text == 'hi'


# ---------------------------------------------------------------------------
# report_message + update_message_rating
# ---------------------------------------------------------------------------


def test_report_message_merges_reported_true_into_jsonb():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import report_message
        result = report_message('u1', 'm1')
        assert result == {"message": "Message reported"}
        sql = cur.execute.call_args.args[0]
        assert "UPDATE chat_messages" in sql
        assert '"reported": true' in sql


def test_update_message_rating_returns_false_when_missing():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.chat import update_message_rating
        assert update_message_rating('u1', 'ghost', 1) is False


def test_update_message_rating_updates_jsonb_on_hit():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)

        from database.chat import update_message_rating
        assert update_message_rating('u1', 'm1', 1) is True
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "UPDATE chat_messages" in joined
        assert "data = data || %s::jsonb" in joined
        # One of the calls should carry the rating JSON
        all_params = [c.args[1] for c in cur.execute.call_args_list if len(c.args) > 1]
        assert any(isinstance(p[0], str) and '"rating": 1' in p[0] for p in all_params)


# ---------------------------------------------------------------------------
# batch_delete_messages + clear_chat
# ---------------------------------------------------------------------------


def test_batch_delete_messages_filters_by_uid_app_and_session():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 3

        from database.chat import batch_delete_messages
        batch_delete_messages(uid='u1', app_id='app-1', chat_session_id='s1')

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM chat_messages" in sql
        assert "plugin_id IS NOT DISTINCT FROM %s" in sql
        assert "session_id = %s" in sql
        assert params == ('u1', 'app-1', 's1')


def test_clear_chat_returns_user_not_found_when_user_absent():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.chat import clear_chat
        assert clear_chat('u1') == {"message": "User not found"}


def test_clear_chat_deletes_messages_when_user_exists():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)
        cur.rowcount = 5

        from database.chat import clear_chat
        assert clear_chat('u1', app_id='app-1') is None
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "DELETE FROM chat_messages" in joined


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_add_multi_files_inserts_in_batch():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.chat import add_multi_files
        now = datetime.now(timezone.utc)
        add_multi_files('u1', [
            {'id': 'f1', 'name': 'a.pdf', 'created_at': now},
            {'id': 'f2', 'name': 'b.pdf', 'created_at': now},
        ])
        assert cur.execute.call_count == 2
        sql = cur.execute.call_args_list[0].args[0]
        assert "INSERT INTO chat_files" in sql
        assert "ON CONFLICT (uid, id)" in sql
        db_mock.batch.assert_called_once()


def test_add_multi_files_noop_when_empty():
    with patch("database.chat.db") as db_mock:
        from database.chat import add_multi_files
        add_multi_files('u1', [])
        db_mock.batch.assert_not_called()


def test_get_chat_files_no_filter_returns_all():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'f1', now, {'name': 'a'}),
            ('u1', 'f2', now, {'name': 'b'}),
        ]

        from database.chat import get_chat_files
        result = get_chat_files('u1')
        sql = cur.execute.call_args.args[0]
        assert "FROM chat_files" in sql
        assert "ANY" not in sql
        assert len(result) == 2


def test_get_chat_files_filters_with_any_array():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.chat import get_chat_files
        get_chat_files('u1', files_id=['f1', 'f2', 'f3'])
        sql = cur.execute.call_args.args[0]
        assert "id = ANY(%s::text[])" in sql


def test_get_chat_files_desc_orders_descending_and_limits():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.chat import get_chat_files_desc
        get_chat_files_desc('u1', files_id=['f1'], limit=5)
        sql = cur.execute.call_args.args[0]
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT %s" in sql


def test_get_chat_files_desc_without_ids_returns_most_recent():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.chat import get_chat_files_desc
        get_chat_files_desc('u1', limit=10)
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "FROM chat_files" in sql
        assert "ORDER BY created_at DESC" in sql
        assert 10 in params


def test_delete_multi_files_deletes_via_any_array():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import delete_multi_files
        delete_multi_files('u1', [{'id': 'f1'}, {'id': 'f2'}])
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM chat_files" in sql
        assert "id = ANY(%s::text[])" in sql
        assert params == ('u1', ['f1', 'f2'])


def test_delete_multi_files_noop_when_empty():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import delete_multi_files
        delete_multi_files('u1', [])
        cur.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_add_chat_session_inserts_with_typed_cols():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import add_chat_session
        now = datetime.now(timezone.utc)
        doc = {'id': 's1', 'plugin_id': 'app-1', 'created_at': now, 'title': 'Hi'}
        add_chat_session('u1', doc)
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO chat_sessions" in sql
        assert "ON CONFLICT (uid, id)" in sql
        assert params[0] == 'u1'
        assert params[1] == 's1'
        assert params[2] == 'app-1'  # plugin_id typed col


def test_get_chat_session_filters_by_plugin_id():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchone.return_value = ('u1', 's1', 'app-1', now, {'title': 'Main'})

        from database.chat import get_chat_session
        result = get_chat_session('u1', app_id='app-1')
        assert result['id'] == 's1'
        sql = cur.execute.call_args.args[0]
        assert "plugin_id IS NOT DISTINCT FROM %s" in sql
        assert "LIMIT 1" in sql


def test_get_chat_session_returns_none_when_empty():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.chat import get_chat_session
        assert get_chat_session('u1') is None


def test_get_chat_session_by_id_hits_pk():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchone.return_value = ('u1', 's1', None, now, {'title': 't'})

        from database.chat import get_chat_session_by_id
        result = get_chat_session_by_id('u1', 's1')
        assert result['id'] == 's1'
        sql = cur.execute.call_args.args[0]
        assert "uid = %s AND id = %s" in sql


def test_delete_chat_session_without_cascade_drops_only_session_row():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.chat import delete_chat_session
        delete_chat_session('u1', 's1', cascade_messages=False)

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "DELETE FROM chat_sessions" in joined
        assert "DELETE FROM chat_messages" not in joined


def test_delete_chat_session_cascade_also_deletes_messages():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = (1,)

        from database.chat import delete_chat_session
        delete_chat_session('u1', 's1', cascade_messages=True)

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "DELETE FROM chat_messages" in joined
        assert "session_id = %s" in joined
        assert "DELETE FROM chat_sessions" in joined
        db_mock.batch.assert_called_once()


def test_delete_chat_session_cascade_returns_false_when_missing():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.chat import delete_chat_session
        assert delete_chat_session('u1', 'ghost', cascade_messages=True) is False


def test_add_message_to_chat_session_appends_without_duplicates():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # Session exists with two ids already.
        cur.fetchone.return_value = ({'message_ids': ['a', 'b']},)

        from database.chat import add_message_to_chat_session
        add_message_to_chat_session('u1', 's1', 'c')

        all_params = [c.args[1] for c in cur.execute.call_args_list if len(c.args) > 1]
        write_calls = [p for p in all_params if isinstance(p[0], str) and 'a' in p[0] and 'c' in p[0]]
        assert write_calls, "expected a jsonb_set call with new id appended"
        # The appended list should include all three.
        payload = json.loads(write_calls[0][0])
        assert payload == ['a', 'b', 'c']


def test_add_message_to_chat_session_noop_on_duplicate():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = ({'message_ids': ['a', 'b']},)

        from database.chat import add_message_to_chat_session
        add_message_to_chat_session('u1', 's1', 'b')
        # Only the SELECT should have fired.
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert not any(s.strip().upper().startswith('UPDATE') for s in sqls)


def test_add_files_to_chat_session_noop_on_empty_list():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.chat import add_files_to_chat_session
        add_files_to_chat_session('u1', 's1', [])
        cur.execute.assert_not_called()


def test_update_chat_session_openai_ids_writes_both_when_provided():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import update_chat_session_openai_ids
        update_chat_session_openai_ids('u1', 's1', 'thr-1', 'asst-1')
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "UPDATE chat_sessions" in sql
        assert "data = data || %s::jsonb" in sql
        stored = json.loads(params[0])
        assert stored == {'openai_thread_id': 'thr-1', 'openai_assistant_id': 'asst-1'}


def test_update_chat_session_openai_ids_noop_when_both_none():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import update_chat_session_openai_ids
        update_chat_session_openai_ids('u1', 's1', None, None)
        cur.execute.assert_not_called()


def test_create_chat_session_writes_default_doc():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.chat import create_chat_session
        doc = create_chat_session('u1', title='Hello', app_id='app-1')
        assert doc['title'] == 'Hello'
        assert doc['starred'] is False
        assert doc['plugin_id'] == 'app-1'
        # add_chat_session should have fired the INSERT.
        sql = cur.execute.call_args.args[0]
        assert "INSERT INTO chat_sessions" in sql


def test_acquire_chat_session_reuses_existing():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        now = datetime.now(timezone.utc)
        cur.fetchone.return_value = ('u1', 's-existing', 'app', now, {})

        from database.chat import acquire_chat_session
        assert acquire_chat_session('u1', app_id='app') == 's-existing'


def test_acquire_chat_session_creates_when_absent():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.chat import acquire_chat_session
        sid = acquire_chat_session('u1', app_id='app')
        assert isinstance(sid, str) and len(sid) > 0
        # A session INSERT should have fired.
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert any("INSERT INTO chat_sessions" in s for s in sqls)


def test_get_chat_sessions_filters_by_plugin_and_starred():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.chat import get_chat_sessions
        get_chat_sessions('u1', app_id='app', starred=True)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "plugin_id IS NOT DISTINCT FROM %s" in sql
        assert "(data->>'starred')::boolean = %s" in sql
        assert "ORDER BY (data->>'updated_at') DESC" in sql
        assert True in params


def test_update_chat_session_returns_none_when_missing():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.fetchone.return_value = None

        from database.chat import update_chat_session
        assert update_chat_session('u1', 'ghost', title='x') is None


def test_update_chat_session_merges_and_returns_hydrated_row():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        now = datetime.now(timezone.utc)
        # First fetchone: existing row. Second: post-update row.
        cur.fetchone.side_effect = [
            ('u1', 's1', 'app', now, {'title': 'old'}),
            ('u1', 's1', 'app', now, {'title': 'new', 'starred': True}),
        ]

        from database.chat import update_chat_session
        result = update_chat_session('u1', 's1', title='new', starred=True)
        assert result is not None
        assert result['id'] == 's1'
        assert result['title'] == 'new'
        assert result['starred'] is True


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def test_get_chats_to_migrate_filters_to_non_target_level():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ('m1', 'standard'),   # include
            ('m2', 'enhanced'),   # skip (already target)
            ('m3', None),         # treated as 'standard' => include
        ]

        from database.chat import get_chats_to_migrate
        result = get_chats_to_migrate('u1', 'enhanced')
        ids = [r['id'] for r in result]
        assert ids == ['m1', 'm3']
        for r in result:
            assert r['type'] == 'chat'


def test_migrate_chats_level_batch_noop_on_empty():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.chat import migrate_chats_level_batch
        migrate_chats_level_batch('u1', [], 'enhanced')
        db_mock.batch.assert_not_called()


def test_migrate_chats_level_batch_skips_already_at_target():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'm1', None, None, now, {'data_protection_level': 'enhanced', 'text': 'hi'}),
        ]

        from database.chat import migrate_chats_level_batch
        migrate_chats_level_batch('u1', ['m1'], 'enhanced')
        # Only the FOR UPDATE SELECT should fire; no UPDATEs.
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert not any(s.strip().upper().startswith('UPDATE') for s in sqls)


def test_migrate_chats_level_batch_writes_level_and_text():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        now = datetime.now(timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'm1', None, None, now, {'data_protection_level': 'standard', 'text': 'hi'}),
        ]

        from database.chat import migrate_chats_level_batch
        migrate_chats_level_batch('u1', ['m1'], 'enhanced')
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "UPDATE chat_messages" in joined
        assert "data = data || %s::jsonb" in joined


# ---------------------------------------------------------------------------
# v2 save_message + delete_messages
# ---------------------------------------------------------------------------


def test_save_message_inserts_message_and_updates_session():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        # Session lookup (in FOR UPDATE) returns existing session data.
        cur.fetchone.return_value = ({'message_count': 3, 'title': 't'},)

        from database.chat import save_message
        result = save_message('u1', 'hello', 'human', app_id=None, session_id='s1')
        assert 'id' in result
        assert 'created_at' in result

        sqls = [c.args[0] for c in cur.execute.call_args_list]
        joined = '\n'.join(sqls)
        assert "INSERT INTO chat_messages" in joined
        assert "UPDATE chat_sessions" in joined


def test_delete_messages_session_scope_uses_session_col():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 4

        from database.chat import delete_messages
        assert delete_messages('u1', session_id='s1') == 4
        sql = cur.execute.call_args.args[0]
        assert "DELETE FROM chat_messages" in sql
        assert "session_id = %s" in sql


def test_delete_messages_app_scope_uses_plugin_col():
    with patch("database.chat.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 2

        from database.chat import delete_messages
        assert delete_messages('u1', app_id=None) == 2
        sql = cur.execute.call_args.args[0]
        assert "plugin_id IS NOT DISTINCT FROM %s" in sql


# ---------------------------------------------------------------------------
# Encryption helpers (preserved unchanged)
# ---------------------------------------------------------------------------


def test_encrypt_chat_data_roundtrip():
    from database.chat import _encrypt_chat_data, _decrypt_chat_data
    plain = {'text': 'hello world'}
    encrypted = _encrypt_chat_data(plain, 'u1')
    assert encrypted['text'] != 'hello world'
    decrypted = _decrypt_chat_data(encrypted, 'u1')
    assert decrypted['text'] == 'hello world'


def test_prepare_data_for_write_only_encrypts_enhanced_level():
    from database.chat import _prepare_data_for_write
    standard = _prepare_data_for_write({'text': 'hi'}, 'u1', 'standard')
    assert standard['text'] == 'hi'
    enhanced = _prepare_data_for_write({'text': 'hi'}, 'u1', 'enhanced')
    assert enhanced['text'] != 'hi'


def test_prepare_message_for_read_decrypts_when_enhanced():
    from database.chat import _encrypt_chat_data, _prepare_message_for_read
    plain = {'text': 'secret', 'data_protection_level': 'enhanced'}
    encrypted = _encrypt_chat_data(plain, 'u1')
    # encrypted dict still has enhanced level; decrypt on read.
    encrypted['data_protection_level'] = 'enhanced'
    result = _prepare_message_for_read(encrypted, 'u1')
    assert result['text'] == 'secret'


def test_prepare_message_for_read_passthrough_for_standard():
    from database.chat import _prepare_message_for_read
    msg = {'text': 'plain', 'data_protection_level': 'standard'}
    assert _prepare_message_for_read(msg, 'u1')['text'] == 'plain'


def test_prepare_message_for_read_none_returns_none():
    from database.chat import _prepare_message_for_read
    assert _prepare_message_for_read(None, 'u1') is None
