import pytest
from httpx import AsyncClient, ASGITransport
import anyio
import json

from backend.main import app
from backend.settings import settings
from backend.models import BuildingData

@pytest.mark.asyncio
async def test_generate_stream(monkeypatch):
    # Mock settings to avoid missing API key 400 error
    monkeypatch.setattr(settings, "gemini_api_key", "test_key")
    
    # We might also need to mock generate_building to avoid actual LLM calls
    from backend.api.routers import generate
    async def mock_generate_building(*args, **kwargs):
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
