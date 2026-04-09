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
        allocation, _warnings = await generate_building_semantics(request)
        return allocation

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


# ============================================================
# 新管线：Building 端到端（语义 → 几何 → 序列化）
# ============================================================

from pydantic import BaseModel as _BaseModel


class BuildingGenerateRequest(_BaseModel):
    """新管线请求体"""
    prompt: str
    total_area: Optional[float] = None
    total_floors: Optional[int] = None
    floor_width: Optional[float] = None
    floor_depth: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None


@router.post("/building/generate")
async def generate_building_layout(req: BuildingGenerateRequest):
    """
    端到端 Building 生成 API（同步返回）

    管线：LLM → BuildingAllocation → BuildingOrchestrator → 序列化 JSON
    """
    try:
        result = await asyncio.wait_for(
            _generate_building_pipeline(req),
            timeout=120.0,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out (120s)")
    except Exception as e:
        logger.exception("Building generation failed")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_building_pipeline(req: BuildingGenerateRequest) -> dict:
    """完整管线：LLM → 解析 → 几何 → 序列化"""
    import math
    from shapely.geometry import box

    from ...core.geometry.building_orchestrator import BuildingOrchestrator
    from ...core.geometry.serializers import building_result_to_dict
    from ...core.geometry.room_spec import SolverConfig

    # 1. LLM → BuildingAllocation + parse warnings
    allocation, parse_warnings = await generate_building_semantics(
        GenerateSemanticsRequest(
            scene_type=SceneType.BUILDING,
            user_prompt=req.prompt,
            total_area=req.total_area,
            total_floors=req.total_floors,
            provider=req.provider,
            model=req.model,
        )
    )

    # 2. 构建楼层边界
    if req.floor_width and req.floor_depth:
        floor_boundary = box(0, 0, req.floor_width, req.floor_depth)
    else:
        floor_area = allocation.floors[0].floor_total_area
        w = math.sqrt(floor_area * 3 / 2)
        d = floor_area / w
        floor_boundary = box(0, 0, w, d)

    # 3. 根据楼层面积动态调整参数（小楼层缩减走廊和核心筒）
    floor_area = floor_boundary.area
    if floor_area < 200:
        corridor_width = 1.2
        core_area_ratio = 0.05
    elif floor_area < 500:
        corridor_width = 1.5
        core_area_ratio = 0.06
    else:
        corridor_width = 2.0
        core_area_ratio = 0.08

    # 4. BuildingOrchestrator → BuildingResult（CPU 密集，在线程池中运行）
    orchestrator = BuildingOrchestrator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width,
        core_area_ratio=core_area_ratio,
    )
    building_result = await asyncio.to_thread(orchestrator.generate, allocation)

    # 5. 序列化 + 合并 parse_warnings
    result = building_result_to_dict(building_result, floor_boundary)
    # 合并解析阶段的 warnings
    if parse_warnings:
        if "warnings" not in result:
            result["warnings"] = []
        result["warnings"].extend(parse_warnings)
    return result
