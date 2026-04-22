"""Unit tests for database/action_items.py — Postgres impl.

Tests exercise every SQL pathway the Omi app hits:
- create_action_item: INSERT with typed columns promoted
- get_action_item: single SELECT
- get_action_items: filtered SELECT with conversation_id, completed, dates
- update_action_item: shallow JSONB merge + typed column updates
- batch_update_action_items: loop in transaction
- delete_action_item: DELETE single
- delete_action_items_for_conversation: DELETE all for conversation
- mark_action_item_completed: UPDATE completed flag
- get_pending_apple_reminders_sync: filter by sync_requested
- batch_sync_update_action_items: batch update with exported flag
- unlock_all_action_items: unlock is_locked items
- get_daily_score: count due tasks for a date
- get_scores: daily/weekly/overall scores
"""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import uuid


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
# CREATE
# ---------------------------------------------------------------------------


def test_create_action_item_with_generated_id():
    """create_action_item generates ID if not provided."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.action_items import create_action_item

        payload = {
            'description': 'Test task',
            'sort_order': 0,
        }
        result_id = create_action_item(uid='u1', action_item_data=payload)

        # Should generate a UUID
        assert result_id
        assert len(result_id) > 0

        # Check SQL call
        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "INSERT INTO action_items" in sql
        assert "ON CONFLICT (uid, id) DO UPDATE" in sql
        assert params[0] == 'u1'
        assert params[1] == result_id  # Generated ID used
        assert params[2] is None  # conversation_id
        assert params[3] is False  # completed
        assert isinstance(params[4], datetime)  # created_at
        assert isinstance(params[5], datetime)  # updated_at


def test_create_action_item_with_conversation_id():
    """create_action_item promotes conversation_id to typed column."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.action_items import create_action_item

        payload = {
            'id': 'task1',
            'conversation_id': 'conv1',
            'description': 'Test task',
        }
        create_action_item(uid='u1', action_item_data=payload)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert params[2] == 'conv1'  # conversation_id


def test_create_action_item_as_completed():
    """create_action_item sets completed_at when created as completed."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.action_items import create_action_item

        payload = {
            'id': 'task1',
            'completed': True,
            'description': 'Done task',
        }
        create_action_item(uid='u1', action_item_data=payload)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        jsonb_data = json.loads(params[6])
        assert 'completed_at' in jsonb_data
        assert isinstance(jsonb_data['completed_at'], str)


def test_create_action_items_batch():
    """create_action_items_batch inserts multiple in a transaction."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.action_items import create_action_items_batch

        payloads = [
            {'description': 'Task 1'},
            {'description': 'Task 2'},
        ]
        result_ids = create_action_items_batch(uid='u1', action_items_data=payloads)

        assert len(result_ids) == 2
        # Check two INSERT calls
        assert cur.execute.call_count == 2


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------


def test_get_action_item_found():
    """get_action_item returns the action item if found."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchone.return_value = (
            'task1',  # id
            'conv1',  # conversation_id
            False,  # completed
            now,  # created_at
            now,  # updated_at
            {'description': 'Test task', 'sort_order': 0},  # data JSONB
        )

        from database.action_items import get_action_item

        result = get_action_item(uid='u1', action_item_id='task1')

        assert result is not None
        assert result['id'] == 'task1'
        assert result['conversation_id'] == 'conv1'
        assert result['completed'] is False
        assert result['description'] == 'Test task'
        assert result['sort_order'] == 0


def test_get_action_item_not_found():
    """get_action_item returns None if not found."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.action_items import get_action_item

        result = get_action_item(uid='u1', action_item_id='missing')
        assert result is None


def test_get_action_items_with_conversation_filter():
    """get_action_items filters by conversation_id."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('task1', 'conv1', False, now, now, {'description': 'Task 1'}),
        ]

        from database.action_items import get_action_items

        result = get_action_items(uid='u1', conversation_id='conv1')

        assert len(result) == 1
        assert result[0]['id'] == 'task1'
        assert result[0]['conversation_id'] == 'conv1'

        # Check WHERE clause
        sql = cur.execute.call_args.args[0]
        assert 'conversation_id = %s' in sql


def test_get_action_items_with_completed_filter():
    """get_action_items filters by completed status."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.action_items import get_action_items

        get_action_items(uid='u1', completed=True)

        sql = cur.execute.call_args.args[0]
        assert 'completed = %s' in sql


def test_get_action_items_with_date_filters():
    """get_action_items filters by created_at date range."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.action_items import get_action_items

        start = datetime(2026, 4, 1, tzinfo=timezone.utc)
        end = datetime(2026, 4, 30, tzinfo=timezone.utc)
        get_action_items(uid='u1', start_date=start, end_date=end)

        sql = cur.execute.call_args.args[0]
        assert 'created_at >= %s' in sql
        assert 'created_at <= %s' in sql


def test_get_action_items_with_due_date_filters():
    """get_action_items filters by due_at in JSONB."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.action_items import get_action_items

        start = datetime(2026, 4, 1, tzinfo=timezone.utc)
        end = datetime(2026, 4, 30, tzinfo=timezone.utc)
        get_action_items(uid='u1', due_start_date=start, due_end_date=end)

        sql = cur.execute.call_args.args[0]
        assert "(data->>'due_at')::timestamptz >= %s" in sql
        assert "(data->>'due_at')::timestamptz <= %s" in sql


def test_get_action_items_with_limit_offset():
    """get_action_items applies LIMIT and OFFSET."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.action_items import get_action_items

        get_action_items(uid='u1', limit=10, offset=5)

        sql = cur.execute.call_args.args[0]
        assert 'OFFSET %s' in sql
        assert 'LIMIT %s' in sql


def test_get_action_items_by_conversation():
    """get_action_items_by_conversation is convenience wrapper."""
    with patch("database.action_items.get_action_items") as mock_get:
        from database.action_items import get_action_items_by_conversation

        get_action_items_by_conversation(uid='u1', conversation_id='conv1')

        mock_get.assert_called_once_with('u1', conversation_id='conv1')


def test_get_action_items_by_ids():
    """get_action_items_by_ids returns items in input order."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('task2', None, False, now, now, {'description': 'Task 2'}),
            ('task1', None, False, now, now, {'description': 'Task 1'}),
        ]

        from database.action_items import get_action_items_by_ids

        result = get_action_items_by_ids(uid='u1', action_item_ids=['task1', 'task2'])

        # Results should be in input order
        assert len(result) == 2
        assert result[0]['id'] == 'task1'
        assert result[1]['id'] == 'task2'


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------


def test_update_action_item_exists():
    """update_action_item updates typed columns and JSONB."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # First call: check exists
        # Second call: execute UPDATE
        cur.fetchone.return_value = (1,)

        from database.action_items import update_action_item

        result = update_action_item(
            uid='u1',
            action_item_id='task1',
            update_data={'description': 'Updated', 'conversation_id': 'conv2'}
        )

        assert result is True
        # Check UPDATE call
        sql = cur.execute.call_args.args[0]
        assert 'UPDATE action_items' in sql
        assert 'data = data ||' in sql


def test_update_action_item_not_found():
    """update_action_item returns False if not found."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.action_items import update_action_item

        result = update_action_item(uid='u1', action_item_id='missing', update_data={})
        assert result is False


def test_batch_update_action_items():
    """batch_update_action_items updates multiple in transaction."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.action_items import batch_update_action_items

        class Item:
            def __init__(self, id, sort_order=None, indent_level=None):
                self.id = id
                self.sort_order = sort_order
                self.indent_level = indent_level

        items = [
            Item('task1', sort_order=0, indent_level=0),
            Item('task2', sort_order=1, indent_level=1),
        ]
        batch_update_action_items(uid='u1', items=items)

        assert cur.execute.call_count == 2


def test_mark_action_item_completed():
    """mark_action_item_completed sets completed flag and completed_at."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)

        from database.action_items import mark_action_item_completed

        result = mark_action_item_completed(uid='u1', action_item_id='task1', completed=True)

        assert result is True
        sql = cur.execute.call_args.args[0]
        assert 'UPDATE action_items' in sql


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


def test_delete_action_item_exists():
    """delete_action_item returns True if deleted."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.action_items import delete_action_item

        result = delete_action_item(uid='u1', action_item_id='task1')

        assert result is True
        sql = cur.execute.call_args.args[0]
        assert 'DELETE FROM action_items' in sql


def test_delete_action_item_not_found():
    """delete_action_item returns False if not found."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0

        from database.action_items import delete_action_item

        result = delete_action_item(uid='u1', action_item_id='missing')
        assert result is False


def test_delete_action_items_for_conversation():
    """delete_action_items_for_conversation returns rowcount."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 3

        from database.action_items import delete_action_items_for_conversation

        result = delete_action_items_for_conversation(uid='u1', conversation_id='conv1')

        assert result == 3
        sql = cur.execute.call_args.args[0]
        assert 'DELETE FROM action_items' in sql
        assert 'conversation_id = %s' in sql


# ---------------------------------------------------------------------------
# REMINDERS SYNC
# ---------------------------------------------------------------------------


def test_batch_set_sync_requested():
    """batch_set_sync_requested marks items for sync."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.action_items import batch_set_sync_requested

        batch_set_sync_requested(uid='u1', item_ids=['task1', 'task2'])

        assert cur.execute.call_count == 2
        sql = cur.execute.call_args.args[0]
        assert 'UPDATE action_items' in sql


def test_get_pending_apple_reminders_sync():
    """get_pending_apple_reminders_sync returns pending and synced items."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)

        # First call: pending items
        # Second call: synced items
        cur.fetchall.side_effect = [
            [
                ('task1', None, False, now, now, {'description': 'Pending', 'sync_requested': True}),
            ],
            [
                ('task2', None, False, now, now, {'description': 'Synced', 'exported': True}),
            ],
        ]

        from database.action_items import get_pending_apple_reminders_sync

        result = get_pending_apple_reminders_sync(uid='u1')

        assert len(result['pending_export']) == 1
        assert result['pending_export'][0]['id'] == 'task1'
        assert len(result['synced_items']) == 1
        assert result['synced_items'][0]['id'] == 'task2'


def test_batch_sync_update_action_items():
    """batch_sync_update_action_items updates with exported flag."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        _mock_batch(db_mock, conn)

        from database.action_items import batch_sync_update_action_items

        updates = [
            {'id': 'task1', 'data': {'exported': True, 'sync_requested': True}},
        ]
        batch_sync_update_action_items(uid='u1', updates=updates)

        sql = cur.execute.call_args.args[0]
        assert 'UPDATE action_items' in sql


def test_unlock_all_action_items():
    """unlock_all_action_items sets is_locked=False."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 5

        from database.action_items import unlock_all_action_items

        unlock_all_action_items(uid='u1')

        sql = cur.execute.call_args.args[0]
        assert 'UPDATE action_items' in sql
        assert "(data->>'is_locked')::boolean = true" in sql


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------


def test_get_daily_score():
    """get_daily_score computes score for a date."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (10, 7)  # 10 total, 7 completed

        from database.action_items import get_daily_score

        result = get_daily_score(uid='u1', date='2026-04-21')

        assert result['date'] == '2026-04-21'
        assert result['total_tasks'] == 10
        assert result['completed_tasks'] == 7
        assert result['score'] == 70


def test_get_daily_score_zero_tasks():
    """get_daily_score returns 0 score when no tasks."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (0, 0)

        from database.action_items import get_daily_score

        result = get_daily_score(uid='u1')

        assert result['score'] == 0
        assert result['total_tasks'] == 0


def test_get_scores():
    """get_scores computes daily/weekly/overall scores."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # Daily, Weekly, Overall
        cur.fetchone.side_effect = [
            (10, 7),  # daily: 10 tasks, 7 complete = 70%
            (50, 40),  # weekly: 50 tasks, 40 complete = 80%
            (200, 150),  # overall: 200 tasks, 150 complete = 75%
        ]

        from database.action_items import get_scores

        result = get_scores(uid='u1', date='2026-04-21')

        assert result['daily']['score'] == 70.0
        assert result['weekly']['score'] == 80.0
        assert result['overall']['score'] == 75.0
        # default_tab is 'weekly' since 80% >= 75% and 80% > 70%
        assert result['default_tab'] == 'weekly'


def test_get_scores_default_tab_weekly():
    """get_scores selects weekly as default when highest."""
    with patch("database.action_items.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchone.side_effect = [
            (0, 0),  # daily: no tasks
            (10, 9),  # weekly: 90%
            (100, 80),  # overall: 80%
        ]

        from database.action_items import get_scores

        result = get_scores(uid='u1')

        assert result['default_tab'] == 'weekly'
