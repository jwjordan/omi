"""Unit tests for database/staged_tasks.py — Postgres impl."""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call


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


def test_create_staged_task_inserts_new_row():
    """create_staged_task inserts a new task and returns dict."""
    with patch("database.staged_tasks.db") as db_mock, \
         patch("database.staged_tasks.uuid.uuid4") as uuid_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        task_id = "550e8400-e29b-41d4-a716-446655440000"
        uuid_mock.return_value = uuid.UUID(task_id)

        # Mock dedup query first (no existing tasks)
        cur.fetchall.return_value = []

        from database.staged_tasks import create_staged_task
        result = create_staged_task("user123", "Test task", priority="high")

        # Verify the INSERT was called
        calls = cur.execute.call_args_list
        insert_call = [c for c in calls if "INSERT INTO staged_tasks" in str(c)][0]
        sql = insert_call.args[0]
        params = insert_call.args[1]

        assert "INSERT INTO staged_tasks" in sql
        assert params[0] == "user123"  # uid
        assert params[1] == task_id  # id
        assert result["id"] == task_id
        assert result["description"] == "Test task"
        assert result["priority"] == "high"
        assert result["completed"] is False


def test_create_staged_task_deduplicates_by_description():
    """create_staged_task returns existing task if description matches (case-insensitive)."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        existing_id = str(uuid.uuid4())
        existing_data = {
            "id": existing_id,
            "description": "Test Task",
            "priority": "high",
            "completed": False,
        }
        # Mock dedup query returning existing row (data is dict, not JSON string)
        cur.fetchall.return_value = [(existing_id, existing_data)]

        from database.staged_tasks import create_staged_task
        result = create_staged_task("user123", "test task")  # lowercase input

        # Should not insert, just return existing
        insert_calls = [c for c in cur.execute.call_args_list if "INSERT" in str(c)]
        assert len(insert_calls) == 0
        assert result["id"] == existing_id


def test_get_staged_tasks_orders_by_relevance_and_respects_limit_offset():
    """get_staged_tasks returns uncompleted tasks ordered by relevance_score."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn

        task1_id = str(uuid.uuid4())
        task2_id = str(uuid.uuid4())
        rows = [
            (task1_id, {"description": "Task 1", "relevance_score": 0.9, "completed": False}),
            (task2_id, {"description": "Task 2", "relevance_score": 0.5, "completed": False}),
        ]
        cur.fetchall.return_value = rows

        from database.staged_tasks import get_staged_tasks
        result = get_staged_tasks("user123", limit=10, offset=0)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT id, data FROM staged_tasks" in sql
        assert "WHERE uid = %s" in sql
        assert "AND (data->>'completed')::boolean = false" in sql
        assert "ORDER BY (data->>'relevance_score')::numeric" in sql
        assert "LIMIT %s" in sql
        assert "OFFSET %s" in sql
        assert params[0] == "user123"
        assert params[1] == 10
        assert params[2] == 0

        assert len(result) == 2
        assert result[0]["id"] == task1_id
        assert result[1]["id"] == task2_id


def test_delete_staged_task_returns_true_when_deleted():
    """delete_staged_task returns True and deletes the row."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1  # Simulate 1 row deleted

        from database.staged_tasks import delete_staged_task
        result = delete_staged_task("user123", "task-id-123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM staged_tasks" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ("user123", "task-id-123")
        assert result is True


def test_delete_staged_task_returns_false_when_not_found():
    """delete_staged_task returns False if no rows deleted."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 0  # Simulate no rows deleted

        from database.staged_tasks import delete_staged_task
        result = delete_staged_task("user123", "nonexistent-id")

        assert result is False


def test_batch_update_staged_scores_filters_to_active_tasks():
    """batch_update_staged_scores only updates tasks that exist and are uncompleted."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        db_mock.batch.return_value.__enter__ = MagicMock(return_value=conn)
        db_mock.batch.return_value.__exit__ = MagicMock(return_value=None)

        # First call: fetch active IDs (from db.connection())
        task_id_1 = str(uuid.uuid4())

        # Use a side_effect to handle multiple execute calls
        execute_call_count = [0]
        def side_effect(*args, **kwargs):
            execute_call_count[0] += 1
            if "SELECT id FROM staged_tasks" in args[0]:
                cur.fetchall.return_value = [(task_id_1,)]
            return None

        cur.execute.side_effect = side_effect

        from database.staged_tasks import batch_update_staged_scores

        scores = [
            {"id": task_id_1, "relevance_score": 0.8},
            {"id": "nonexistent-id", "relevance_score": 0.5},  # Should be filtered out
        ]
        batch_update_staged_scores("user123", scores)

        # Verify only the valid ID was updated
        update_calls = [c for c in cur.execute.call_args_list if "UPDATE staged_tasks" in str(c)]
        assert len(update_calls) == 1
        update_sql = update_calls[0].args[0]
        update_params = update_calls[0].args[1]
        assert "UPDATE staged_tasks" in update_sql
        assert "jsonb_set" in update_sql
        assert task_id_1 in update_params


def test_batch_update_staged_scores_empty_list_no_op():
    """batch_update_staged_scores does nothing with empty list."""
    with patch("database.staged_tasks.db") as db_mock:
        from database.staged_tasks import batch_update_staged_scores

        batch_update_staged_scores("user123", [])

        # db.batch should not be called
        db_mock.batch.assert_not_called()


def test_promote_staged_task_moves_highest_scored_to_action_items():
    """promote_staged_task selects top task, deletes from staged, returns it."""
    with patch("database.staged_tasks.db") as db_mock, \
         patch("database.staged_tasks.action_items_db.create_action_item") as create_action_mock, \
         patch("database.staged_tasks.action_items_db.get_action_item") as get_action_mock:
        conn, cur = _mock_conn()
        db_mock.batch.return_value.__enter__ = MagicMock(return_value=conn)
        db_mock.batch.return_value.__exit__ = MagicMock(return_value=None)

        task_id = str(uuid.uuid4())
        task_data = {
            "description": "Top priority task",
            "relevance_score": 0.95,
            "priority": "high",
            "completed": False,
        }
        cur.fetchone.return_value = (task_id, task_data)

        action_id = str(uuid.uuid4())
        create_action_mock.return_value = action_id
        get_action_mock.return_value = {
            "id": action_id,
            "description": "Top priority task",
            "completed": False,
        }

        from database.staged_tasks import promote_staged_task
        result = promote_staged_task("user123")

        # Verify SELECT FOR UPDATE was called
        select_calls = [c for c in cur.execute.call_args_list if "SELECT id, data" in str(c)]
        assert len(select_calls) > 0

        # Verify DELETE was called
        delete_calls = [c for c in cur.execute.call_args_list if "DELETE FROM staged_tasks" in str(c)]
        assert len(delete_calls) == 1

        # Verify action_items.create_action_item was called
        create_action_mock.assert_called_once()

        assert result is not None
        assert result["id"] == action_id


def test_promote_staged_task_returns_none_when_no_tasks():
    """promote_staged_task returns None if no uncompleted tasks exist."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.batch.return_value.__enter__ = MagicMock(return_value=conn)
        db_mock.batch.return_value.__exit__ = MagicMock(return_value=None)

        cur.fetchone.return_value = None  # No tasks found

        from database.staged_tasks import promote_staged_task
        result = promote_staged_task("user123")

        assert result is None


def test_migrate_ai_tasks_moves_excess_tasks_atomically():
    """migrate_ai_tasks moves AI tasks beyond top 3 to staged_tasks."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        db_mock.batch.return_value.__enter__ = MagicMock(return_value=conn)
        db_mock.batch.return_value.__exit__ = MagicMock(return_value=None)

        task_ids = [str(uuid.uuid4()) for _ in range(5)]
        # 5 AI tasks, should keep top 3, move 2
        rows = [
            (task_ids[i], {
                "description": f"AI Task {i+1}",
                "source": "screenshot",
                "relevance_score": 0.8 - (i * 0.1),
                "completed": False,
            })
            for i in range(5)
        ]
        cur.fetchall.return_value = rows

        from database.staged_tasks import migrate_ai_tasks
        result = migrate_ai_tasks("user123")

        # Verify the result indicates correct counts
        assert result["moved"] == 2
        assert result["kept"] == 3


def test_migrate_conversation_items_to_staged_moves_without_source():
    """migrate_conversation_items_to_staged moves items with conversation_id but no source."""
    with patch("database.staged_tasks.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        db_mock.batch.return_value.__enter__ = MagicMock(return_value=conn)
        db_mock.batch.return_value.__exit__ = MagicMock(return_value=None)

        conv_id = str(uuid.uuid4())
        item_id = str(uuid.uuid4())
        rows = [
            (item_id, {
                "description": "From conversation",
                "conversation_id": conv_id,
                "completed": False,
            }),
        ]
        cur.fetchall.return_value = rows

        from database.staged_tasks import migrate_conversation_items_to_staged
        result = migrate_conversation_items_to_staged("user123")

        assert result["moved"] == 1
