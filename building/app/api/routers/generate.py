from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...logger import log_context
from ...models import GenerateSemanticsRequest, SceneType
from ...services.building_pipeline_service import BuildingPipelineOptions, BuildingPipelineService

logger = logging.getLogger(__name__)

_REMOVED_PARKING_SCENE_ID = "parking" + "_" + "underground"

router = APIRouter()


class BuildingGenerateRequest(BaseModel):
    """Active backend-only multi-floor building generation request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)
    total_area: Optional[float] = Field(default=None, gt=0)
    total_floors: int = Field(..., ge=2)
    floor_width: Optional[float] = Field(default=None, gt=0)
    floor_depth: Optional[float] = Field(default=None, gt=0)
    corridor_width: Optional[float] = Field(default=None, gt=0)
    core_area_ratio: Optional[float] = Field(default=None, gt=0)
    core_placement: Literal["auto", "north", "center", "south", "east", "west"] = "auto"
    topology_mode: Literal["continuous_cpsat", "grid_growth"] = "grid_growth"
    corridor_layout: Literal["door_side", "organic"] = "organic"
    provider: Optional[Literal["gemini", "deepseek", "openai"]] = None
    model: Optional[str] = None
    scene_type: Optional[str] = None
    sceneId: Optional[str] = None
    program_source: Literal["llm", "mock"] = "llm"
    use_stage1_program: bool = False

    @model_validator(mode="after")
    def _reject_legacy_modes(self) -> "BuildingGenerateRequest":
        if self.scene_type and str(self.scene_type).lower() != "building":
            raise ValueError("Only scene_type=building is supported in the active backend-only API")
        if self.sceneId and str(self.sceneId).lower() == _REMOVED_PARKING_SCENE_ID:
            raise ValueError("Removed underground parking scene is not supported in the active backend-only API")
        return self


def _to_semantics_request(req: BuildingGenerateRequest) -> GenerateSemanticsRequest:
    return GenerateSemanticsRequest(
        scene_type=SceneType.BUILDING,
        user_prompt=req.prompt,
        total_area=req.total_area,
        total_floors=req.total_floors,
        provider=req.provider,
        model=req.model,
    )


def _to_pipeline_options(req: BuildingGenerateRequest) -> BuildingPipelineOptions:
    return BuildingPipelineOptions(
        floor_width=req.floor_width,
        floor_depth=req.floor_depth,
        corridor_width=req.corridor_width,
        corridor_layout=req.corridor_layout,
        topology_mode=req.topology_mode,
        core_area_ratio=req.core_area_ratio,
        core_placement=req.core_placement,
        use_stage1_program=req.use_stage1_program,
    )



@router.post("/building/program")
async def generate_building_program(req: BuildingGenerateRequest) -> dict[str, Any]:
    session_id = f"stage1-{uuid.uuid4().hex[:8]}"
    try:
        with log_context(session_id=session_id, stage="api_building_program", topology_mode=req.topology_mode):
            result = await asyncio.wait_for(
                BuildingPipelineService().generate_stage1(
                    _to_semantics_request(req),
                    options=_to_pipeline_options(req),
                    source=req.program_source,
                ),
                timeout=120.0,
            )
        data = result.model_dump(mode="json")
        data["session_id"] = session_id
        return data
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Stage 1 program planning timed out (120s)") from exc
    except Exception as exc:
        logger.exception("[%s] Stage 1 program planning failed", session_id)
        raise HTTPException(status_code=500, detail=f"Stage 1 program planning failed: {exc}") from exc

@router.post("/building/generate")
async def generate_building_layout(req: BuildingGenerateRequest) -> dict[str, Any]:
    session_id = f"bld-{uuid.uuid4().hex[:8]}"
    try:
        with log_context(session_id=session_id, stage="api_building_generate", topology_mode=req.topology_mode):
            result = await asyncio.wait_for(
                BuildingPipelineService().generate(_to_semantics_request(req), options=_to_pipeline_options(req)),
                timeout=120.0,
            )
        data = dict(result.building_dict)
        data["session_id"] = session_id
        if result.warnings:
            data.setdefault("warnings", []).extend(result.warnings)
        return data
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Building pipeline timed out (120s)") from exc
    except Exception as exc:
        logger.exception("[%s] Building pipeline failed", session_id)
        raise HTTPException(status_code=500, detail=f"Building pipeline failed: {exc}") from exc


@router.post("/building/generate/stream")
async def generate_building_layout_stream(req: BuildingGenerateRequest) -> StreamingResponse:
    session_id = f"bld-{uuid.uuid4().hex[:8]}"

    async def event_generator():
        try:
            with log_context(session_id=session_id, stage="api_building_generate_stream", topology_mode=req.topology_mode):
                async for event in BuildingPipelineService().generate_stream(
                    _to_semantics_request(req),
                    options=_to_pipeline_options(req),
                ):
                    status = str(event.get("status") or "message")
                    idx = int(event.get("event_index") or 0)
                    floor_id = str(event.get("floor_id") or "-")
                    yield (
                        f"event: {status}\n"
                        f"id: {session_id}:{idx:06d}:{status}:{floor_id}\n"
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    )
        except Exception as exc:
            logger.exception("[%s] Building stream failed", session_id)
            event = {"status": "pipeline_failed", "session_id": session_id, "error": str(exc)}
            yield f"event: pipeline_failed\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            yield f"event: closed\ndata: {json.dumps({'status': 'closed', 'session_id': session_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/v1/generate/semantics")
async def generate_semantics_endpoint(request: GenerateSemanticsRequest):
    if request.scene_type != SceneType.BUILDING:
        raise HTTPException(status_code=400, detail="Only scene_type=building is supported")
    if request.total_floors is None or int(request.total_floors) < 2:
        raise HTTPException(status_code=400, detail="total_floors must be >= 2")
    result = await BuildingPipelineService().generate_semantics_only(request)
    return result.allocation

