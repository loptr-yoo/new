from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...core.flows.parking_flow import execute_refinement
from ...core.llm.provider import create_llm_client
from ...settings import settings
from ..sse import sse_stream

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


@router.post("/augment/stream")
async def augment_stream(req: AugmentRequest):
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
    queue: "asyncio.Queue[dict | None]" = asyncio.Queue()

    def on_log(msg: str) -> None:
        try:
            queue.put_nowait({"status": "progress", "msg": msg})
        except Exception:
            pass

    async def run() -> None:
        try:
            queue.put_nowait({"status": "start"})
            layout = await execute_refinement(
                current_layout=req.layout,
                client=client,
                api_key=api_key,
                model=req.model,
                on_log=on_log,
                scene_id=req.sceneId,
            )
            queue.put_nowait({"status": "done", "data": layout.model_dump()})
        except Exception as e:
            queue.put_nowait({"status": "error", "message": str(e)})
        finally:
            queue.put_nowait(None)

    asyncio.create_task(run())
    return StreamingResponse(sse_stream(queue), media_type="text/event-stream")
