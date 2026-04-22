"""Integration tests for POST /v1/conversations/semantic-search.

Uses FastAPI's TestClient with auth dependency overrides, and mocks the
helper to isolate the route handler from the embedding/SQL path.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app
from utils.other import endpoints as auth_module


@pytest.fixture
def client():
    # Override get_current_user_uid to a fixed uid for all tests.
    async def fake_uid():
        return "test-uid"

    app.dependency_overrides[auth_module.get_current_user_uid] = fake_uid
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_valid_request_returns_hits(client):
    with patch("routers.conversations.semantic_search_conversations") as helper:
        helper.return_value = {
            "query": "minecraft",
            "hits": [
                {
                    "conversation_id": "c1",
                    "started_at": "2026-04-22T12:51:23+00:00",
                    "finished_at": "2026-04-22T13:56:44+00:00",
                    "similarity": 0.85,
                    "title": "Test",
                    "overview": "Overview",
                    "excerpts": [],
                }
            ],
        }
        resp = client.post(
            "/v1/conversations/semantic-search",
            json={"query": "minecraft", "limit": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "minecraft"
        assert len(body["hits"]) == 1
        helper.assert_called_once_with(
            uid="test-uid", query="minecraft", since=None, until=None, limit=5
        )


def test_empty_corpus_returns_200_with_empty_hits(client):
    with patch("routers.conversations.semantic_search_conversations") as helper:
        helper.return_value = {"query": "x", "hits": []}
        resp = client.post(
            "/v1/conversations/semantic-search", json={"query": "x"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"query": "x", "hits": []}


def test_invalid_date_returns_422(client):
    resp = client.post(
        "/v1/conversations/semantic-search",
        json={"query": "x", "since": "not-a-date"},
    )
    assert resp.status_code == 422


def test_empty_query_returns_422(client):
    resp = client.post("/v1/conversations/semantic-search", json={"query": ""})
    assert resp.status_code == 422


def test_limit_out_of_range_returns_422(client):
    resp = client.post(
        "/v1/conversations/semantic-search", json={"query": "x", "limit": 50}
    )
    assert resp.status_code == 422


def test_embed_service_down_returns_503(client):
    from utils.conversations.semantic_search import EmbedServiceUnavailable

    with patch("routers.conversations.semantic_search_conversations") as helper:
        helper.side_effect = EmbedServiceUnavailable("proxy 500")
        resp = client.post(
            "/v1/conversations/semantic-search", json={"query": "x"}
        )
        assert resp.status_code == 503
        assert "embedding service unavailable" in resp.text.lower()


def test_since_and_until_forwarded_to_helper(client):
    with patch("routers.conversations.semantic_search_conversations") as helper:
        helper.return_value = {"query": "x", "hits": []}
        resp = client.post(
            "/v1/conversations/semantic-search",
            json={
                "query": "x",
                "since": "2026-04-22T00:00:00Z",
                "until": "2026-04-23T00:00:00Z",
                "limit": 10,
            },
        )
        assert resp.status_code == 200
        helper.assert_called_once_with(
            uid="test-uid",
            query="x",
            since="2026-04-22T00:00:00Z",
            until="2026-04-23T00:00:00Z",
            limit=10,
        )
