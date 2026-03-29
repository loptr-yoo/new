from __future__ import annotations

from fastapi import APIRouter, HTTPException
from ...core.flows.parking_flow import generate_building
from ...settings import settings

from ...models import BuildingData, GenerateRequest


router = APIRouter()


@router.post("/generate", response_model=BuildingData)
async def generate(req: GenerateRequest) -> BuildingData:
    api_key = None
    if req.provider == "gemini":
        api_key = settings.gemini_api_key
    elif req.provider == "deepseek":
        api_key = settings.deepseek_api_key
    elif req.provider == "openai":
        api_key = settings.openai_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail=f"Missing API key for provider: {req.provider}")

    return await generate_building(
        prompt=req.prompt,
        provider=req.provider,
        model=req.model,
        api_key=api_key,
        on_log=None,
        scene_id=req.sceneId,
    )
