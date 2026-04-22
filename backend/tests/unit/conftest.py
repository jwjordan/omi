"""Pytest configuration for unit tests - mock heavy dependencies."""

import os
import sys
from unittest.mock import MagicMock

# Set encryption secret before any imports that depend on it
os.environ.setdefault("ENCRYPTION_SECRET", "omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv")

# Mock psycopg_pool and pgvector before any imports
sys.modules["psycopg_pool"] = MagicMock()
sys.modules["pgvector"] = MagicMock()
sys.modules["pgvector.psycopg"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["stripe"] = MagicMock()

# Also mock database.redis_db since it fails on Python 3.9 with dict | None syntax
sys.modules["database.redis_db"] = MagicMock()

# Mock utils.subscription since it uses Python 3.10+ syntax (int | None)
sys.modules["utils.subscription"] = MagicMock()
