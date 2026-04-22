"""Validation tests for the SemanticSearchRequest pydantic model."""

import pytest
from pydantic import ValidationError

from models.conversation import SemanticSearchRequest


def test_minimum_valid_request():
    req = SemanticSearchRequest(query="minecraft")
    assert req.query == "minecraft"
    assert req.since is None
    assert req.until is None
    assert req.limit == 5


def test_custom_limit_and_dates():
    req = SemanticSearchRequest(
        query="rameen", since="2026-04-22T00:00:00Z", until="2026-04-23T00:00:00Z", limit=10
    )
    assert req.limit == 10
    assert req.since == "2026-04-22T00:00:00Z"
    assert req.until == "2026-04-23T00:00:00Z"


def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="")


def test_limit_bounds_enforced():
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="x", limit=0)
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="x", limit=21)


def test_malformed_since_rejected():
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="x", since="not-a-date")


def test_malformed_until_rejected():
    with pytest.raises(ValidationError):
        SemanticSearchRequest(query="x", until="also-not-a-date")
