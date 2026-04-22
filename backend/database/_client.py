"""Postgres connection pool shared by every backend/database/*.py module.

Replaces the Firestore client. The exported `db` object is a small adapter
that gives callers a `.connection()` context-manager yielding a psycopg
connection (autocommit) and a `.batch()` helper for explicit transactions.

Every per-connection checkout registers the pgvector codec so tables that
store vector columns (conversation_vectors, memory_vectors) work without
per-call setup.
"""

import hashlib
import os
import uuid
from typing import Iterator

from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

_DATABASE_URL = os.environ.get("DATABASE_URL")


def _configure_conn(conn):
    register_vector(conn)


if _DATABASE_URL:
    _pool: ConnectionPool = ConnectionPool(
        _DATABASE_URL,
        min_size=2,
        max_size=20,
        kwargs={"autocommit": True},
        configure=_configure_conn,
    )
else:
    _pool = None  # type: ignore[assignment]


class _Db:
    """Adapter exposing the handful of access patterns the database modules need.

    - db.connection() -> context-manager psycopg connection (autocommit=True)
    - db.batch()      -> context-manager psycopg connection wrapped in a transaction,
                         yielding the same connection so `with db.batch() as conn: conn.execute(...)`
                         commits atomically.
    """

    def connection(self):
        if _pool is None:
            raise RuntimeError("DATABASE_URL not configured; _client.py pool disabled")
        return _pool.connection()

    def batch(self):
        # Re-export of .connection() + an explicit transaction, matching the
        # Firestore "batch" mental model for multi-write atomicity.
        return _BatchContext()


class _BatchContext:
    """Context manager: yield a connection inside an explicit transaction."""

    def __enter__(self):
        self._cm = _pool.connection()
        self._conn = self._cm.__enter__()
        self._txn = self._conn.transaction()
        self._txn.__enter__()
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        try:
            self._txn.__exit__(exc_type, exc, tb)
        finally:
            return self._cm.__exit__(exc_type, exc, tb)


db = _Db()


def close_pool() -> None:
    """Drain the pool on shutdown. No-op if DATABASE_URL was unset."""
    if _pool is not None:
        _pool.close()


# ---------------------------------------------------------------------------
# Utilities preserved from the old _client.py so existing callers keep working.
# ---------------------------------------------------------------------------


def get_users_uid() -> list[str]:
    """Return the list of all user uids. Backs `backend/database/users.py::get_users_uid`."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT uid FROM users")
            return [row[0] for row in cur.fetchall()]


def document_id_from_seed(seed: str) -> str:
    """Deterministic UUID from a string seed. Kept for callers migrating
    from the firestore.Client-based implementation that used the same trick."""
    seed_hash = hashlib.sha256(seed.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=seed_hash[:16], version=4))
