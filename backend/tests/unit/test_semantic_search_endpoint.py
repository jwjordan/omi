"""Integration tests for POST /v1/conversations/semantic-search.

Uses a minimal test-local FastAPI app that includes only the conversations
router. This isolates endpoint tests from the full main.app router graph
(40+ routers with heavy ML/audio dependencies).
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app():
    import inspect
    import routers.conversations as _conv_mod
    from routers.conversations import router as conversations_router

    # Find the exact get_current_user_uid callable bound in the
    # semantic-search route at decoration time. Other test modules
    # (test_task_sharing, test_available_plans_resilience) mutate
    # sys.modules["utils.other.endpoints"].get_current_user_uid after conftest
    # runs, so we cannot rely on the current value of conv_mod.auth.get_current_user_uid.
    # The route object itself holds the original function reference.
    _get_uid_fn = None
    for route in conversations_router.routes:
        if hasattr(route, "path") and "semantic-search" in (route.path or ""):
            sig = inspect.signature(route.endpoint)
            for param in sig.parameters.values():
                if hasattr(param.default, "dependency"):
                    _dep = param.default.dependency
                    if getattr(_dep, "__name__", "") == "get_current_user_uid":
                        _get_uid_fn = _dep
                        break
            break

    if _get_uid_fn is None:
        raise RuntimeError("Could not locate get_current_user_uid dep in semantic-search route")

    app = FastAPI()
    app.include_router(conversations_router)

    async def fake_uid():
        return "test-uid"

    app.dependency_overrides[_get_uid_fn] = fake_uid
    return app


@pytest.fixture
def client():
    app = _build_app()
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
