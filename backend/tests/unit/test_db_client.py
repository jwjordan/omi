"""Unit tests for _client.py Postgres pool wrapper."""

import importlib
import sys
import pytest
from unittest.mock import patch, MagicMock


def _fresh_import_client():
    """Evict any pre-stubbed database._client from sys.modules (other test
    modules in this suite install a MagicMock at collection time) and
    re-import the real module."""
    sys.modules.pop("database._client", None)
    return importlib.import_module("database._client")


def test_db_exports_connection_context_manager(monkeypatch):
    """The module-level `db` must expose a .connection() context manager."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
    with patch("psycopg_pool.ConnectionPool") as MockPool:
        pool_instance = MagicMock()
        MockPool.return_value = pool_instance
        pool_instance.connection.return_value.__enter__.return_value = MagicMock()

        mod = _fresh_import_client()

        # db.connection() should return the pool's connection context manager
        cm = mod.db.connection()
        assert cm is pool_instance.connection.return_value


def test_document_id_from_seed_is_deterministic(monkeypatch):
    """Moved-but-unchanged utility; verify still works."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
    with patch("psycopg_pool.ConnectionPool"):
        mod = _fresh_import_client()
    a = mod.document_id_from_seed("abc")
    b = mod.document_id_from_seed("abc")
    c = mod.document_id_from_seed("xyz")
    assert a == b
    assert a != c
    assert len(a) == 36  # UUID string


def test_pool_configured_with_register_vector(monkeypatch):
    """The pool's per-connection configure hook must register the pgvector codec."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
    with patch("psycopg_pool.ConnectionPool") as MockPool, \
            patch("pgvector.psycopg.register_vector") as MockRegister:
        _fresh_import_client()

        call_kwargs = MockPool.call_args.kwargs
        assert "configure" in call_kwargs
        configure_fn = call_kwargs["configure"]
        # Invoking configure on a mock connection should call register_vector on it
        fake_conn = MagicMock()
        configure_fn(fake_conn)
        MockRegister.assert_called_once_with(fake_conn)


def test_autocommit_kwarg_true(monkeypatch):
    """Pool must set autocommit=True to match per-document Firestore semantics."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
    with patch("psycopg_pool.ConnectionPool") as MockPool:
        _fresh_import_client()

        call_kwargs = MockPool.call_args.kwargs
        assert call_kwargs.get("kwargs", {}).get("autocommit") is True
