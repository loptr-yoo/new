from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from building.app.main import app


def test_old_generate_returns_410() -> None:
    client = TestClient(app)
    response = client.post("/api/generate", json={"prompt": "x", "provider": "gemini", "model": "m"})
    assert response.status_code == 410
    assert response.json()["use"] == "/api/building/generate"


def test_old_generate_stream_returns_410() -> None:
    client = TestClient(app)
    response = client.post("/api/generate/stream", json={"prompt": "x", "provider": "gemini", "model": "m"})
    assert response.status_code == 410
    assert response.json()["use"] == "/api/building/generate/stream"


def test_building_generate_rejects_single_floor_before_pipeline() -> None:
    client = TestClient(app)
    response = client.post("/api/building/generate", json={"prompt": "x", "total_floors": 1})
    assert response.status_code == 422
