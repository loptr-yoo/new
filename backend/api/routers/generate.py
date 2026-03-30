from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..sse import sse_stream
from ...core.flows.building_semantic_flow import generate_building_semantics
from ...core.flows.parking_flow import generate_building
from ...settings import settings

from ...models import BuildingAllocation, BuildingData, GenerateRequest, GenerateSemanticsRequest, SceneType

logger = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_MODEL_BY_PROVIDER: Dict[str, str] = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
}


def _pick_provider_model_and_key(
    provider: Optional[str],
    model: Optional[str],
) -> tuple[str, str, str]:
    """
    语义入口为旧管线（floor/parking）做“黑盒透传”时所需的 LLM 参数选择逻辑。

    注意：此处仅发生在路由层，旧业务层代码不做任何修改。
    """

    def _key_for(p: str) -> Optional[str]:
        if p == "openai":
            return settings.openai_api_key
        if p == "gemini":
            return settings.gemini_api_key
        if p == "deepseek":
            return settings.deepseek_api_key
        return None

    chosen_provider = provider
    if chosen_provider is None:
        for cand in ("openai", "gemini", "deepseek"):
            if _key_for(cand):
                chosen_provider = cand
                break
    if chosen_provider is None:
        raise HTTPException(status_code=400, detail="No available API key for any provider (openai/gemini/deepseek)")

    api_key = _key_for(chosen_provider)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"Missing API key for provider: {chosen_provider}")

    chosen_model = model or _DEFAULT_MODEL_BY_PROVIDER.get(chosen_provider)
    if not chosen_model:
        raise HTTPException(status_code=400, detail=f"Missing model for provider: {chosen_provider}")

    return chosen_provider, chosen_model, api_key


@router.post("/v1/generate/semantics", response_model=Union[BuildingAllocation, BuildingData])
async def generate_semantics_endpoint(request: GenerateSemanticsRequest):
    if request.scene_type == SceneType.BUILDING:
        return await generate_building_semantics(request)

    if request.scene_type in (SceneType.FLOOR, SceneType.PARKING):
        scene_id_map = {
            SceneType.FLOOR: "building_floor_plan",
            SceneType.PARKING: "parking_underground",
        }
        provider, model, api_key = _pick_provider_model_and_key(request.provider, request.model)
        return await generate_building(
            prompt=request.user_prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            on_log=None,
            scene_id=scene_id_map[request.scene_type],
        )

    raise HTTPException(status_code=400, detail=f"Unsupported scene_type: {request.scene_type}")


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
async def generate_stream(req: GenerateRequest, request: Request):
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
            logger.exception("SSE crash in run task")
            queue.put_nowait({"status": "error", "message": str(e)})
        finally:
            queue.put_nowait(None)

    asyncio.create_task(run())

    async def event_generator():
        try:
            async for chunk in sse_stream(queue):
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as e:
            logger.exception("SSE crash")
            yield f"data: {{\"status\": \"error\", \"message\": \"Internal Server Error\"}}\n\n"
        finally:
            yield "data: [DONE]\n\n"
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")
