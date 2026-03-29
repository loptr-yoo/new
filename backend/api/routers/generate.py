from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..sse import sse_stream
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


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    api_key = None
    if req.provider == "gemini":
        api_key = settings.gemini_api_key
    elif req.provider == "deepseek":
        api_key = settings.deepseek_api_key
    elif req.provider == "openai":
        api_key = settings.openai_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail=f"Missing API key for provider: {req.provider}")

    queue: "asyncio.Queue[dict | None]" = asyncio.Queue()

    def on_log(msg: str) -> None:
        try:
            queue.put_nowait({"status": "progress", "msg": msg})
        except Exception:
            pass

    async def run() -> None:
        try:
            queue.put_nowait({"status": "start"})
            data = await generate_building(
                prompt=req.prompt,
                provider=req.provider,
                model=req.model,
                api_key=api_key,
                on_log=on_log,
                scene_id=req.sceneId,
            )
            queue.put_nowait({"status": "done", "data": data.model_dump()})
        except Exception as e:
            queue.put_nowait({"status": "error", "message": str(e)})
        finally:
            queue.put_nowait(None)

    asyncio.create_task(run())
    return StreamingResponse(sse_stream(queue), media_type="text/event-stream")
