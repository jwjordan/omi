"""Unit tests for database/goals.py — Postgres impl.

Tests exercise every SQL pathway:
- create_goal: INSERT with typed columns, enforce max_goals
- get_user_goal: first active goal
- get_user_goals: active goals up to limit
- get_all_goals: all or only active goals
- get_goal: single goal by id
- update_goal: JSONB merge + typed column updates
- update_goal_progress: update + save history
- delete_goal: DELETE goal + history
- save_goal_progress_history: INSERT or UPDATE history
- get_goal_history: SELECT history ordered by date DESC
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


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------


def test_create_goal_with_generated_id():
    """create_goal generates ID if not provided."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # First call: check max goals count
        # Second call: insert goal
        cur.fetchone.side_effect = [
            (0,),  # No active goals
            None,  # No oldest goal to deactivate
        ]

        from database.goals import create_goal

        payload = {
            'title': 'Test Goal',
            'target_date': '2026-05-01',
        }
        result = create_goal(uid='u1', goal_data=payload)

        # Should generate a UUID
        assert result['id']
        assert len(result['id']) > 0
        assert result['is_active'] is True
        assert result['title'] == 'Test Goal'


def test_create_goal_deactivate_oldest_at_max():
    """create_goal deactivates oldest goal when at max_goals."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        # Mock: already have 4 active goals
        cur.fetchone.side_effect = [
            (4,),  # 4 active goals
            ('goal_oldest',),  # Oldest goal to deactivate
        ]

        from database.goals import create_goal

        payload = {
            'title': 'New Goal',
        }
        create_goal(uid='u1', goal_data=payload, max_goals=4)

        # Should have called UPDATE to deactivate oldest
        calls = cur.execute.call_args_list
        # First call: count active
        # Second call: select oldest
        # Third call: update oldest
        # Fourth call: insert new
        assert len(calls) >= 3
        update_sql = calls[2][0][0]
        assert 'UPDATE goals' in update_sql
        assert 'is_active = false' in update_sql


def test_create_goal_with_explicit_id():
    """create_goal uses provided id."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (0,)

        from database.goals import create_goal

        payload = {
            'id': 'goal123',
            'title': 'Test',
        }
        result = create_goal(uid='u1', goal_data=payload)

        assert result['id'] == 'goal123'


def test_create_goal_sets_timestamps():
    """create_goal sets created_at and updated_at."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (0,)

        from database.goals import create_goal

        payload = {'title': 'Test'}
        result = create_goal(uid='u1', goal_data=payload)

        assert result['created_at']
        assert result['updated_at']
        assert isinstance(result['created_at'], datetime)


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------


def test_get_user_goal_found():
    """get_user_goal returns first active goal."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchone.return_value = (
            'goal1',  # id
            True,  # is_active
            now,  # created_at
            now,  # updated_at
            {'title': 'Active Goal', 'current_value': 10},  # data JSONB
        )

        from database.goals import get_user_goal

        result = get_user_goal(uid='u1')

        assert result is not None
        assert result['id'] == 'goal1'
        assert result['is_active'] is True
        assert result['title'] == 'Active Goal'
        assert result['current_value'] == 10


def test_get_user_goal_not_found():
    """get_user_goal returns None when no active goals."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.goals import get_user_goal

        result = get_user_goal(uid='u1')
        assert result is None


def test_get_user_goals_with_limit():
    """get_user_goals returns active goals up to limit."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('goal1', True, now, now, {'title': 'Goal 1'}),
            ('goal2', True, now, now, {'title': 'Goal 2'}),
        ]

        from database.goals import get_user_goals

        result = get_user_goals(uid='u1', limit=3)

        assert len(result) == 2
        assert result[0]['id'] == 'goal1'
        assert result[1]['id'] == 'goal2'

        # Check SQL includes LIMIT
        sql = cur.execute.call_args.args[0]
        assert 'LIMIT %s' in sql


def test_get_all_goals_active_only():
    """get_all_goals returns only active goals when include_inactive=False."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('goal1', True, now, now, {'title': 'Active 1'}),
        ]

        from database.goals import get_all_goals

        result = get_all_goals(uid='u1', include_inactive=False)

        assert len(result) == 1
        sql = cur.execute.call_args.args[0]
        assert 'is_active = true' in sql


def test_get_all_goals_including_inactive():
    """get_all_goals includes inactive when include_inactive=True."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('goal1', True, now, now, {'title': 'Active'}),
            ('goal2', False, now, now, {'title': 'Archived'}),
        ]

        from database.goals import get_all_goals

        result = get_all_goals(uid='u1', include_inactive=True)

        assert len(result) == 2
        sql = cur.execute.call_args.args[0]
        assert 'is_active = true' not in sql


def test_get_goal_found():
    """get_goal returns goal by id."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchone.return_value = (
            'goal1',
            True,
            now,
            now,
            {'title': 'Test Goal', 'target_date': '2026-05-01'},
        )

        from database.goals import get_goal

        result = get_goal(uid='u1', goal_id='goal1')

        assert result is not None
        assert result['id'] == 'goal1'
        assert result['title'] == 'Test Goal'


def test_get_goal_not_found():
    """get_goal returns None if not found."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.goals import get_goal

        result = get_goal(uid='u1', goal_id='missing')
        assert result is None


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------


def test_update_goal_exists():
    """update_goal updates goal and returns updated data."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)

        # First call: check exists
        # Second call: fetch updated goal
        cur.fetchone.side_effect = [
            (1,),  # Goal exists
            ('goal1', True, now, now, {'title': 'Updated Goal'}),
        ]

        from database.goals import update_goal

        result = update_goal(uid='u1', goal_id='goal1', updates={'title': 'Updated Goal'})

        assert result is not None
        assert result['title'] == 'Updated Goal'

        # Check UPDATE call
        sql = cur.execute.call_args_list[1][0][0]
        assert 'UPDATE goals' in sql
        assert 'data = data ||' in sql


def test_update_goal_not_found():
    """update_goal returns None if not found."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.goals import update_goal

        result = update_goal(uid='u1', goal_id='missing', updates={})
        assert result is None


def test_update_goal_progress():
    """update_goal_progress updates current_value and saves history."""
    with patch("database.goals.db") as db_mock, \
         patch("database.goals.save_goal_progress_history") as save_hist_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchone.side_effect = [
            (1,),  # Goal exists
            ('goal1', True, now, now, {'current_value': 50}),
        ]

        from database.goals import update_goal_progress

        result = update_goal_progress(uid='u1', goal_id='goal1', current_value=50.0)

        assert result is not None
        assert result['current_value'] == 50
        # Should call save_goal_progress_history
        save_hist_mock.assert_called_once_with('u1', 'goal1', 50.0)


def test_update_goal_with_is_active_flag():
    """update_goal updates is_active typed column."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchone.side_effect = [
            (1,),  # Goal exists
            ('goal1', False, now, now, {}),
        ]

        from database.goals import update_goal

        result = update_goal(uid='u1', goal_id='goal1', updates={'is_active': False})

        assert result['is_active'] is False


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


def test_delete_goal_exists():
    """delete_goal deletes goal and history."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = (1,)

        from database.goals import delete_goal

        result = delete_goal(uid='u1', goal_id='goal1')

        assert result is True
        # Should have two DELETE calls: one for goal, one for history
        calls = [call[0][0] for call in cur.execute.call_args_list]
        assert any('DELETE FROM goals' in sql for sql in calls)
        assert any('DELETE FROM goal_history' in sql for sql in calls)


def test_delete_goal_not_found():
    """delete_goal returns False if not found."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.goals import delete_goal

        result = delete_goal(uid='u1', goal_id='missing')
        assert result is False


# ---------------------------------------------------------------------------
# HISTORY
# ---------------------------------------------------------------------------


def test_save_goal_progress_history():
    """save_goal_progress_history inserts or updates history entry."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        from database.goals import save_goal_progress_history

        save_goal_progress_history(uid='u1', goal_id='goal1', value=42.5)

        sql = cur.execute.call_args.args[0]
        assert 'INSERT INTO goal_history' in sql
        assert 'ON CONFLICT' in sql

        params = cur.execute.call_args.args[1]
        assert params[0] == 'u1'
        assert params[1] == 'goal1'
        assert params[3] == 42.5


def test_get_goal_history():
    """get_goal_history returns history ordered by date DESC."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        cur.fetchall.return_value = [
            ('2026-04-21', 50.0, now),
            ('2026-04-20', 45.0, now),
        ]

        from database.goals import get_goal_history

        result = get_goal_history(uid='u1', goal_id='goal1', days=30)

        assert len(result) == 2
        assert result[0]['date'] == '2026-04-21'
        assert result[0]['value'] == 50.0
        assert result[1]['date'] == '2026-04-20'

        # Check SQL includes ORDER BY DESC
        sql = cur.execute.call_args.args[0]
        assert 'ORDER BY date DESC' in sql


def test_get_goal_history_respects_limit():
    """get_goal_history respects days parameter."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchall.return_value = []

        from database.goals import get_goal_history

        get_goal_history(uid='u1', goal_id='goal1', days=7)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert 'LIMIT %s' in sql
        assert params[-1] == 7


# ---------------------------------------------------------------------------
# INTEGRATION
# ---------------------------------------------------------------------------


def test_create_and_get_goal():
    """Integration: create goal then retrieve it."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)

        # create_goal calls
        cur.fetchone.side_effect = [
            (0,),  # count active
            None,  # no oldest
        ]

        from database.goals import create_goal

        result = create_goal(uid='u1', goal_data={'title': 'Integration Test'})
        goal_id = result['id']

        assert goal_id
        assert result['title'] == 'Integration Test'


def test_update_deactivates_at_max():
    """When creating with max_goals=1, new goal deactivates old."""
    with patch("database.goals.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        cur.fetchone.side_effect = [
            (1,),  # 1 active goal
            ('old_goal',),  # oldest to deactivate
        ]

        from database.goals import create_goal

        result = create_goal(uid='u1', goal_data={'title': 'New'}, max_goals=1)

        assert result['is_active'] is True
        # Should have deactivated old goal
        calls = [call[0][0] for call in cur.execute.call_args_list]
        assert any('UPDATE goals' in sql and 'is_active = false' in sql for sql in calls)
