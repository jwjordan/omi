"""Unit tests for database/phone_calls.py — Postgres impl."""

import json
import uuid
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


def test_upsert_phone_number_inserts_with_prepare():
    with patch("database.phone_calls.db") as db_mock:
        with patch("database.helpers.redis_db") as redis_mock:
            conn, cur = _mock_conn()
            db_mock.connection.return_value = conn
            redis_mock.get_user_data_protection_level.return_value = "standard"

            from database.phone_calls import upsert_phone_number
            upsert_phone_number("user123", {"id": "phone1", "phone_number": "5555551234", "is_primary": True})

            sql = cur.execute.call_args.args[0]
            params = cur.execute.call_args.args[1]
            assert "INSERT INTO phone_calls" in sql
            assert "ON CONFLICT" in sql
            assert "user123" in params
            assert "phone1" in params


def test_get_phone_numbers_returns_list():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        phone_id = str(uuid.uuid4())
        row_data = {"id": phone_id, "phone_number": "5555551234", "is_primary": True}
        cur.fetchall.return_value = [(phone_id, row_data)]

        from database.phone_calls import get_phone_numbers
        result = get_phone_numbers("user123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT" in sql
        assert "FROM phone_calls" in sql
        assert "WHERE uid = %s" in sql
        assert params == ("user123",)
        assert len(result) == 1
        assert result[0]["id"] == phone_id


def test_get_phone_number_returns_dict_when_found():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        phone_id = str(uuid.uuid4())
        row_data = {"id": phone_id, "phone_number": "5555551234", "is_primary": True}
        cur.fetchone.return_value = (phone_id, row_data)

        from database.phone_calls import get_phone_number
        result = get_phone_number("user123", phone_id)

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "SELECT" in sql
        assert "FROM phone_calls" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ("user123", phone_id)
        assert result is not None
        assert result["id"] == phone_id


def test_get_phone_number_returns_none_when_missing():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.phone_calls import get_phone_number
        result = get_phone_number("user123", "nonexistent")

        assert result is None


def test_get_phone_number_by_number_queries_by_hash():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        phone_id = str(uuid.uuid4())
        row_data = {
            "id": phone_id,
            "phone_number": "encrypted_value",
            "phone_number_hash": "abcd1234",
            "is_primary": True,
            "data_protection_level": "enhanced",
        }
        cur.fetchone.return_value = (phone_id, row_data)

        from database.phone_calls import get_phone_number_by_number
        result = get_phone_number_by_number("user123", "5555551234")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        # Should query by hash in JSONB data
        assert "data->>'phone_number_hash'" in sql
        assert "user123" in params
        assert result is not None


def test_get_phone_number_by_number_returns_none_when_missing():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None

        from database.phone_calls import get_phone_number_by_number
        result = get_phone_number_by_number("user123", "5555551234")

        assert result is None


def test_delete_phone_number_removes_row():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.rowcount = 1

        from database.phone_calls import delete_phone_number
        delete_phone_number("user123", "phone1")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        assert "DELETE FROM phone_calls" in sql
        assert "WHERE uid = %s AND id = %s" in sql
        assert params == ("user123", "phone1")


def test_get_primary_phone_number_filters_by_is_primary():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        phone_id = str(uuid.uuid4())
        row_data = {"id": phone_id, "phone_number": "5555551234", "is_primary": True}
        cur.fetchone.return_value = (phone_id, row_data)

        from database.phone_calls import get_primary_phone_number
        result = get_primary_phone_number("user123")

        sql = cur.execute.call_args.args[0]
        params = cur.execute.call_args.args[1]
        # Should filter by is_primary in JSONB data
        assert "data->>'is_primary'" in sql or "is_primary" in sql
        assert "user123" in params
        assert result is not None
        assert result["id"] == phone_id


def test_get_primary_phone_number_returns_none_when_missing():
    with patch("database.phone_calls.db") as db_mock:
        conn, cur = _mock_conn()
        db_mock.connection.return_value = conn
        cur.fetchone.return_value = None
        # Mock the fallback to get_phone_numbers
        with patch("database.phone_calls.get_phone_numbers") as get_all_mock:
            get_all_mock.return_value = []

            from database.phone_calls import get_primary_phone_number
            result = get_primary_phone_number("user123")

            assert result is None


def test_hash_phone_number_deterministic():
    with patch("database.phone_calls.encryption"):
        from database.phone_calls import _hash_phone_number

        phone = "5555551234"
        hash1 = _hash_phone_number(phone)
        hash2 = _hash_phone_number(phone)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest


def test_prepare_phone_number_for_write_with_enhanced_level():
    with patch("database.phone_calls.encryption") as enc_mock:
        enc_mock.encrypt.return_value = "encrypted_value"

        from database.phone_calls import _prepare_phone_number_for_write

        data = {"id": "phone1", "phone_number": "5555551234"}
        result = _prepare_phone_number_for_write(data, "user123", "enhanced")

        assert "phone_number_hash" in result
        assert result["phone_number"] == "encrypted_value"
        enc_mock.encrypt.assert_called_once()


def test_prepare_phone_number_for_read_with_enhanced_level():
    with patch("database.phone_calls.encryption") as enc_mock:
        enc_mock.decrypt.return_value = "5555551234"

        from database.phone_calls import _prepare_phone_number_for_read

        data = {"id": "phone1", "phone_number": "encrypted_value", "data_protection_level": "enhanced"}
        result = _prepare_phone_number_for_read(data, "user123")

        assert result["phone_number"] == "5555551234"
        enc_mock.decrypt.assert_called_once()
