"""Tests that verify_token() delegates to JWKS validation for normal tokens
while preserving the ADMIN_KEY bypass and LOCAL_DEVELOPMENT fallback."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level stubs: prevent Firestore/Redis init when importing endpoints.py
# (same pattern as tests/unit/test_byok_security.py)
# ---------------------------------------------------------------------------
os.environ.setdefault('OPENAI_API_KEY', 'sk-test-fake-for-unit-tests')
os.environ.setdefault('DEEPGRAM_API_KEY', 'dg-test-fake-for-unit-tests')
os.environ.setdefault('GOOGLE_API_KEY', 'goog-test-fake-for-unit-tests')
os.environ.setdefault('ANTHROPIC_API_KEY', 'ant-test-fake-for-unit-tests')
os.environ.setdefault('ENCRYPTION_SECRET', 'omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv')

# database._client is now Postgres-backed with a lazy pool (pool=None when
# DATABASE_URL is unset, which is the unit-test default). Safe to import.
# database.announcements was ported to Postgres and no longer hits Firestore
# at import time. Both stubs removed so other tests that patch them work.
sys.modules.setdefault('database.redis_db', MagicMock())
sys.modules.setdefault('database.users', MagicMock())
sys.modules.setdefault('database.user_usage', MagicMock())
sys.modules.setdefault('database.llm_usage', MagicMock())
sys.modules.setdefault('utils.other.storage', MagicMock())

from utils.other.endpoints import verify_token  # noqa: E402
from utils.other.jwks_auth import InvalidOmiTokenError  # noqa: E402


def test_admin_key_bypass_returns_uid_suffix(monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "secret-admin-key-")
    assert verify_token("secret-admin-key-my-uid") == "my-uid"


def test_admin_key_no_env_means_no_bypass(monkeypatch):
    monkeypatch.delenv("ADMIN_KEY", raising=False)
    with patch("utils.other.endpoints.verify_omi_id_token") as mock:
        mock.return_value = {"sub": "uid-123"}
        assert verify_token("some-real-jwt") == "uid-123"
        mock.assert_called_once_with("some-real-jwt")


def test_valid_token_returns_sub_claim(monkeypatch):
    monkeypatch.delenv("ADMIN_KEY", raising=False)
    with patch("utils.other.endpoints.verify_omi_id_token") as mock:
        mock.return_value = {"sub": "uid-abc", "email": "a@b.com"}
        assert verify_token("jwt.jwt.jwt") == "uid-abc"


def test_invalid_token_without_local_dev_raises(monkeypatch):
    monkeypatch.delenv("ADMIN_KEY", raising=False)
    monkeypatch.delenv("LOCAL_DEVELOPMENT", raising=False)
    with patch("utils.other.endpoints.verify_omi_id_token") as mock:
        mock.side_effect = InvalidOmiTokenError("bad")
        with pytest.raises(InvalidOmiTokenError):
            verify_token("bad.token.here")


def test_invalid_token_with_local_dev_returns_stub(monkeypatch):
    monkeypatch.delenv("ADMIN_KEY", raising=False)
    monkeypatch.setenv("LOCAL_DEVELOPMENT", "true")
    with patch("utils.other.endpoints.verify_omi_id_token") as mock:
        mock.side_effect = InvalidOmiTokenError("bad")
        assert verify_token("bad.token.here") == "123"
