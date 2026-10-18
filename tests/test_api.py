from __future__ import annotations

from fastapi.testclient import TestClient

from src.serving.app import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_recommend_endpoint_basic():
    client = TestClient(app)
    response = client.post("/recommend", json={"user_id": "u_000001", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "u_000001"
    assert "recommendations" in payload
    assert isinstance(payload["recommendations"], list)
    if payload["recommendations"]:
        first = payload["recommendations"][0]
        assert "item_id" in first
        assert "score" in first
        assert "reason" in first


def test_recommend_endpoint_validation():
    client = TestClient(app)

    # Missing user_id -> 422 Unprocessable Entity
    resp = client.post("/recommend", json={"top_k": 5})
    assert resp.status_code == 422

    # top_k <= 0 -> 422
    resp = client.post("/recommend", json={"user_id": "u_000001", "top_k": 0})
    assert resp.status_code == 422

    # top_k > 100 -> 422
    resp = client.post("/recommend", json={"user_id": "u_000001", "top_k": 101})
    assert resp.status_code == 422


def test_recommend_fallback_unknown_user():
    client = TestClient(app)
    # A non-existent user should still get fallback recommendations gracefully
    response = client.post("/recommend", json={"user_id": "non_existent_user_9999", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "non_existent_user_9999"
    assert isinstance(payload["recommendations"], list)
