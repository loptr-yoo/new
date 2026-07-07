from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from building.app.main import app


def test_building_program_mock_does_not_require_provider() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/building/program",
        json={
            "prompt": "two floor residence",
            "total_floors": 2,
            "program_source": "mock",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["artifact_type"] == "stage1_result"
    assert data["source"] == "mock"
    assert data["building_program"]["total_floors"] == 2
    assert data["feasibility_reports"][0]["geometry_guaranteed"] is False

