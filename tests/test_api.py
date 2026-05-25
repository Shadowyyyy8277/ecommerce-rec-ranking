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
