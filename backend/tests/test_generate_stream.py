import pytest
pytest.importorskip("fastapi")
pytest.importorskip("anyio")
from httpx import AsyncClient, ASGITransport
import anyio

from backend.main import app
from backend.settings import settings

@pytest.mark.asyncio
async def test_generate_stream(monkeypatch):
    # Mock settings to avoid missing API key 400 error
    monkeypatch.setattr(settings, "gemini_api_key", "test_key")
    
    # We might also need to mock generate_building to avoid actual LLM calls
    from backend.api.routers import generate
    async def mock_generate_building(*_, **kwargs):
        from backend.models import BuildingData, ParkingLayout
        queue_on_log = kwargs.get("on_log")
        if queue_on_log:
            queue_on_log("Mocking progress...")
        # Simulate delay
        await anyio.sleep(0.1)
        return BuildingData(blueprint=[], floors={"1": ParkingLayout(width=10, height=10, elements=[])})
        
    monkeypatch.setattr(generate, "generate_building", mock_generate_building)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "prompt": "test prompt",
            "provider": "gemini",
            "model": "gemini-pro"
        }
        
        with anyio.fail_after(5):
            async with client.stream(
                "POST", 
                "/api/generate/stream",
                json=payload,
                headers={"Accept": "text/event-stream"}
            ) as response:
                assert response.status_code == 200
                
                first_chunk = True
                done_received = False
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if first_chunk:
                        assert line.startswith("data: ")
                        first_chunk = False
                        
                    if "data: [DONE]" in line:
                        done_received = True
                        break
                        
                assert done_received, "Expected to receive [DONE] event"


@pytest.mark.asyncio
@pytest.mark.parametrize("scene_id", [None, "building_floor_plan"])
async def test_execute_generation_does_not_call_apply_scene_post_process(monkeypatch, scene_id):
    from backend.core.flows import parking_flow
    from backend.core.llm.provider import LLMClient
    from backend.models import ParkingLayout

    class DummyClient(LLMClient):
        providerName = "gemini"

        async def chat(self, messages, config):
            del messages, config
            raise RuntimeError("unexpected")

    async def mock_call_llm_with_retry(*args, **kwargs):
        del args, kwargs
        return "{}"

    def mock_safe_parse_response(*args, **kwargs):
        del args, kwargs
        elements = [
            {"id": "e1", "type": "ROAD", "x": 0, "y": 0, "width": 1, "height": 1},
            {"id": "e2", "type": "WALL", "x": 1, "y": 0, "width": 1, "height": 1},
            {"id": "e3", "type": "WALL", "x": 2, "y": 0, "width": 1, "height": 1},
            {"id": "e4", "type": "GROUND", "x": 3, "y": 0, "width": 1, "height": 1},
            {"id": "e5", "type": "GROUND", "x": 4, "y": 0, "width": 1, "height": 1},
        ]
        return {"width": 10, "height": 10, "elements": elements}

    async def mock_run_iterative_fix(layout, *args, **kwargs):
        del args, kwargs
        return layout

    def mock_fill_voids_with_ground(layout):
        return layout

    def fail_apply_scene_post_process(*args, **kwargs):
        del args, kwargs
        raise AssertionError("apply_scene_post_process must not be called in execute_generation")

    def mock_validate_generated_content(layout):
        del layout
        return None

    monkeypatch.setattr(parking_flow, "call_llm_with_retry", mock_call_llm_with_retry)
    monkeypatch.setattr(parking_flow, "safe_parse_response", mock_safe_parse_response)
    monkeypatch.setattr(parking_flow, "apply_scene_type_normalization", lambda parsed, _: parsed)
    monkeypatch.setattr(parking_flow, "run_iterative_fix", mock_run_iterative_fix)
    monkeypatch.setattr(parking_flow, "fill_voids_with_ground", mock_fill_voids_with_ground)
    monkeypatch.setattr(parking_flow, "apply_scene_post_process", fail_apply_scene_post_process)
    monkeypatch.setattr(parking_flow, "validate_generated_content", mock_validate_generated_content)

    client = DummyClient()
    layout = await parking_flow.execute_generation("p", client, "k", "m", None, scene_id)
    assert isinstance(layout, ParkingLayout)
    assert len(layout.elements) >= 1
