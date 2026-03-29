from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.flows.parking_flow import execute_refinement
from ...core.llm.provider import create_llm_client
from ...settings import settings

from ...models import AugmentRequest, ParkingLayout


router = APIRouter()


@router.post("/augment", response_model=ParkingLayout)
async def augment(req: AugmentRequest) -> ParkingLayout:
    api_key = None
    if req.provider == "gemini":
        api_key = settings.gemini_api_key
    elif req.provider == "deepseek":
        api_key = settings.deepseek_api_key
    elif req.provider == "openai":
        api_key = settings.openai_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail=f"Missing API key for provider: {req.provider}")

    client = create_llm_client(req.provider)
    return await execute_refinement(
        current_layout=req.layout,
        client=client,
        api_key=api_key,
        model=req.model,
        on_log=None,
        scene_id=req.sceneId,
    )
