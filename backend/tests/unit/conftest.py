"""Pytest configuration for unit tests - mock heavy dependencies."""

import sys
from unittest.mock import MagicMock

# Mock psycopg_pool and pgvector before any imports
sys.modules["psycopg_pool"] = MagicMock()
sys.modules["pgvector"] = MagicMock()
sys.modules["pgvector.psycopg"] = MagicMock()
sys.modules["redis"] = MagicMock()

# Also mock database.redis_db since it fails on Python 3.9 with dict | None syntax
sys.modules["database.redis_db"] = MagicMock()
