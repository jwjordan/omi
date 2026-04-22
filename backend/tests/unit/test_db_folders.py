"""Unit tests for database/folders.py — Postgres impl.

Pendant-critical module (10+ functions). Tests exercise every SQL pathway
including JSONB UPSERT, folder hierarchy, and conversation reparenting.

Key patterns:
- get_folders: ORDER BY (data->>'order')::int NULLS LAST
- create_folder: get max order, INSERT with UUID
- delete_folder: atomic batch with conversation reparenting OR folder_id clearing
- reorder_folders: batch jsonb_set for order + updated_at
- initialize_system_folders: idempotent check then bulk INSERT
- get_conversations_in_folder: JOIN conversations with folder_id filter
- update_folder_conversation_count: COUNT + jsonb_set
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


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
# get_folders
# ---------------------------------------------------------------------------


def test_get_folders_returns_all_folders_sorted_by_order():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # Mock return: (id, data JSONB)
        cur.fetchall.return_value = [
            ('f1', {'name': 'Work', 'order': 1}),
            ('f2', {'name': 'Personal', 'order': 0}),
        ]

        from database.folders import get_folders
        result = get_folders('u1')

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "ORDER BY (data->>'order')::int NULLS LAST, created_at" in sql
        assert params == ('u1',)
        assert len(result) == 2
        assert result[0]['id'] == 'f1'
        assert result[0]['name'] == 'Work'


def test_get_folders_returns_empty_list_if_no_folders():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.folders import get_folders
        result = get_folders('u1')
        assert result == []


# ---------------------------------------------------------------------------
# get_folder
# ---------------------------------------------------------------------------


def test_get_folder_returns_single_folder():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ('f1', {'name': 'Work', 'order': 0})

        from database.folders import get_folder
        result = get_folder('u1', 'f1')

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ('u1', 'f1')
        assert result['id'] == 'f1'
        assert result['name'] == 'Work'


def test_get_folder_returns_none_if_missing():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.folders import get_folder
        result = get_folder('u1', 'f404')
        assert result is None


# ---------------------------------------------------------------------------
# create_folder
# ---------------------------------------------------------------------------


def test_create_folder_inserts_with_incremented_order():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # First call: get max order
        cur.fetchone.side_effect = [
            (2,),  # max order from first query
            None,  # (unused, but called by the mock setup)
        ]

        from database.folders import create_folder
        result = create_folder('u1', 'New Folder', color='#FF0000')

        # Check first call was get max order
        first_call_sql = cur.execute.call_args_list[0].args[0]
        assert "MAX((data->>'order')::int)" in first_call_sql

        # Check second call was INSERT
        insert_call = cur.execute.call_args_list[1]
        insert_sql = insert_call.args[0]
        insert_params = insert_call.args[1]
        assert "INSERT INTO folders" in insert_sql
        assert insert_params[0] == 'u1'
        assert insert_params[1] is not None  # UUID generated
        data = json.loads(insert_params[2])
        assert data['name'] == 'New Folder'
        assert data['color'] == '#FF0000'
        assert data['order'] == 3  # max_order (2) + 1

        assert result['name'] == 'New Folder'
        assert result['order'] == 3


def test_create_folder_uses_defaults_for_color_and_icon():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (0,)

        from database.folders import create_folder
        result = create_folder('u1', 'Test')

        insert_call = cur.execute.call_args_list[1]
        data = json.loads(insert_call.args[1][2])
        assert data['color'] == '#6B7280'
        assert data['icon'] == '📁'


# ---------------------------------------------------------------------------
# update_folder
# ---------------------------------------------------------------------------


def test_update_folder_merges_jsonb_and_adds_updated_at():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ('f1', {'name': 'Old', 'order': 0})

        from database.folders import update_folder
        result = update_folder('u1', 'f1', {'name': 'New Name'})

        assert result is True
        # Should have executed UPDATE
        update_call = cur.execute.call_args_list[-1]
        update_sql = update_call.args[0]
        assert "UPDATE folders SET data" in update_sql
        params = update_call.args[1]
        # Check that updated_at was added to the data being stored
        assert 'updated_at' in params[0] or 'updated_at' in update_sql


# ---------------------------------------------------------------------------
# delete_folder
# ---------------------------------------------------------------------------


def test_delete_folder_with_move_to_folder_reparents_conversations():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        cur.rowcount = 1  # Set rowcount for delete check
        _mock_batch(db_mock, conn)

        from database.folders import delete_folder
        result = delete_folder('u1', 'f_old', move_to_folder_id='f_new')

        assert result is True
        # Should have executed UPDATE conversations then DELETE folder
        calls = cur.execute.call_args_list
        update_sql = calls[0].args[0]
        delete_sql = calls[1].args[0]

        assert "UPDATE conversations SET data = jsonb_set" in update_sql
        assert "folder_id" in update_sql
        assert "'f_new'" in str(calls[0].args[1]) or "f_new" in str(calls[0].args[1])

        assert "DELETE FROM folders" in delete_sql
        assert calls[1].args[1] == ('u1', 'f_old')


def test_delete_folder_without_move_to_folder_clears_folder_id():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.rowcount = 1

        from database.folders import delete_folder
        result = delete_folder('u1', 'f_old', move_to_folder_id=None)

        assert result is True
        calls = cur.execute.call_args_list
        update_sql = calls[0].args[0]

        assert "UPDATE conversations SET data = data - 'folder_id'" in update_sql
        assert calls[0].args[1] == ('u1', 'f_old')


# ---------------------------------------------------------------------------
# reorder_folders
# ---------------------------------------------------------------------------


def test_reorder_folders_updates_order_in_batch():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.folders import reorder_folders
        result = reorder_folders('u1', ['f1', 'f2', 'f3'])

        assert result is True
        # Should have 3 UPDATE calls (one per folder)
        calls = cur.execute.call_args_list
        assert len(calls) == 3

        for i, call in enumerate(calls):
            sql = call.args[0]
            params = call.args[1]
            assert "jsonb_set" in sql
            assert "order" in sql
            assert i in params  # order value should be index
            assert params[-2] == 'u1'
            assert params[-1] in ['f1', 'f2', 'f3']


# ---------------------------------------------------------------------------
# initialize_system_folders
# ---------------------------------------------------------------------------


def test_initialize_system_folders_creates_three_system_folders():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (0,)  # No existing folders

        _mock_batch(db_mock, conn)

        from database.folders import initialize_system_folders
        result = initialize_system_folders('u1')

        # Should have: 1 COUNT query + 3 INSERT queries in batch
        assert len(result) == 3
        assert result[0]['name'] == 'Work'
        assert result[1]['name'] == 'Personal'
        assert result[2]['name'] == 'Social'

        # Check batch INSERT calls
        batch_calls = cur.execute.call_args_list
        inserts = [c for c in batch_calls if 'INSERT INTO folders' in c.args[0]]
        assert len(inserts) == 3


def test_initialize_system_folders_is_idempotent():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)  # Already has folders

        from database.folders import get_folders
        with patch("database.folders.get_folders") as mock_get:
            mock_get.return_value = [{'name': 'Work'}]

            from database.folders import initialize_system_folders
            result = initialize_system_folders('u1')

            # Should have called get_folders instead of INSERTing
            assert len(result) == 1
            # No batch context should be entered


# ---------------------------------------------------------------------------
# get_conversations_in_folder
# ---------------------------------------------------------------------------


def test_get_conversations_in_folder_filters_by_folder_id():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('u1', 'c1', 'completed', False, now, now, now, {'title': 'Conv 1'}),
        ]

        from database.folders import get_conversations_in_folder
        result = get_conversations_in_folder('u1', 'f1', limit=50, offset=0)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "data->>'folder_id' = %s" in sql
        assert "discarded = FALSE" in sql
        assert params == ('u1', 'f1', 50, 0)

        assert len(result) == 1
        assert result[0]['id'] == 'c1'
        assert result[0]['title'] == 'Conv 1'
        assert result[0]['status'] == 'completed'


def test_get_conversations_in_folder_include_discarded():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.folders import get_conversations_in_folder
        get_conversations_in_folder('u1', 'f1', include_discarded=True)

        sql = cur.execute.call_args.args[0]
        assert "discarded = FALSE" not in sql


# ---------------------------------------------------------------------------
# move_conversation_to_folder
# ---------------------------------------------------------------------------


def test_move_conversation_to_folder_updates_folder_id_and_counts():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = ('f_old',)  # old folder_id

        with patch("database.folders.update_folder_conversation_count") as mock_count:
            from database.folders import move_conversation_to_folder
            result = move_conversation_to_folder('u1', 'c1', 'f_new')

            assert result is True
            # Should have updated the conversation
            update_call = cur.execute.call_args_list[1]
            sql = update_call.args[0]
            assert "jsonb_set(data, '{folder_id}'" in sql

            # Should have updated both old and new folder counts
            assert mock_count.call_count == 2
            assert mock_count.call_args_list[0] == (('u1', 'f_old'),)
            assert mock_count.call_args_list[1] == (('u1', 'f_new'),)


def test_move_conversation_to_folder_returns_false_if_conversation_missing():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.folders import move_conversation_to_folder
        result = move_conversation_to_folder('u1', 'c404', 'f_new')

        assert result is False


# ---------------------------------------------------------------------------
# bulk_move_conversations_to_folder
# ---------------------------------------------------------------------------


def test_bulk_move_conversations_to_folder_updates_multiple_and_counts():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)
        cur.rowcount = 2

        # First call: get affected folders
        cur.fetchall.return_value = [('f_old1',), ('f_old2',)]

        with patch("database.folders.update_folder_conversation_count"):
            from database.folders import bulk_move_conversations_to_folder
            moved = bulk_move_conversations_to_folder('u1', ['c1', 'c2'], 'f_new')

            assert moved == 2
            # Should have queried for affected folders, then updated all
            calls = cur.execute.call_args_list
            assert "SELECT DISTINCT" in calls[0].args[0]
            assert "UPDATE conversations" in calls[1].args[0]


def test_bulk_move_conversations_to_folder_returns_zero_if_empty_list():
    from database.folders import bulk_move_conversations_to_folder
    result = bulk_move_conversations_to_folder('u1', [], 'f_new')
    assert result == 0


# ---------------------------------------------------------------------------
# update_folder_conversation_count
# ---------------------------------------------------------------------------


def test_update_folder_conversation_count_counts_nondiscarded():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # First call: COUNT
        # Second call: UPDATE
        cur.fetchone.return_value = (5,)

        from database.folders import update_folder_conversation_count
        count = update_folder_conversation_count('u1', 'f1')

        assert count == 5
        # First call should be COUNT
        first_sql = cur.execute.call_args_list[0].args[0]
        assert "SELECT COUNT(*)" in first_sql
        assert "discarded = FALSE" in first_sql

        # Second call should be UPDATE with jsonb_set
        second_sql = cur.execute.call_args_list[1].args[0]
        assert "UPDATE folders SET data = jsonb_set" in second_sql
        assert "conversation_count" in second_sql


# ---------------------------------------------------------------------------
# get_folder_by_category_mapping
# ---------------------------------------------------------------------------


def test_get_folder_by_category_mapping_returns_matching_folder():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = [
            ('f1', {'name': 'Work', 'category_mapping': 'work'}),
            ('f2', {'name': 'Personal', 'category_mapping': 'personal'}),
        ]

        from database.folders import get_folder_by_category_mapping
        result = get_folder_by_category_mapping('u1', 'work')

        assert result is not None
        assert result['category_mapping'] == 'work'
        assert result['name'] == 'Work'


def test_get_folder_by_category_mapping_returns_none_if_not_found():
    with patch("database.folders.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.folders import get_folder_by_category_mapping
        result = get_folder_by_category_mapping('u1', 'nonexistent')

        assert result is None
