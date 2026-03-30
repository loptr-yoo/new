import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app
from backend.settings import settings


@pytest.mark.asyncio
async def test_generate_semantics_building_branch(monkeypatch):
    from backend.api.routers import generate as generate_router
    from backend.models import BuildingAllocation, FloorAllocation, RoomAllocation

    async def mock_generate_building_semantics(_req):
        return BuildingAllocation(
            building_name="T",
            total_floors=1,
            overall_total_area=100.0,
            floors=[
                FloorAllocation(
                    floor_number=1,
                    floor_function_tag="lobby",
                    floor_total_area=100.0,
                    core_tube_area=20.0,
                    corridor_allowance_area=20.0,
                    rooms=[
                        RoomAllocation(
                            room_name="Lobby",
                            room_type="lobby",
                            target_area=60.0,
                            requires_window=True,
                            weight=5,
                            adjacency_tags=["public"],
                        )
                    ],
                )
            ],
        )

    monkeypatch.setattr(generate_router, "generate_building_semantics", mock_generate_building_semantics)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {"scene_type": "building", "user_prompt": "test", "total_area": 100.0, "total_floors": 1}
        resp = await client.post("/api/v1/generate/semantics", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["building_name"] == "T"
        assert "floors" in data


@pytest.mark.asyncio
async def test_generate_semantics_floor_branch_passthrough(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "test_key")

    from backend.api.routers import generate as generate_router

    async def mock_generate_building(*_, **__):
        from backend.models import BuildingData, ParkingLayout

        return BuildingData(blueprint=[], floors={"1F": ParkingLayout(width=10, height=10, elements=[], sceneId="building_floor_plan")})

    monkeypatch.setattr(generate_router, "generate_building", mock_generate_building)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {"scene_type": "floor", "user_prompt": "test floor"}
        resp = await client.post("/api/v1/generate/semantics", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "floors" in data
        assert "building_name" not in data

