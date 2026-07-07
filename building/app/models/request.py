from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_REMOVED_PARKING_SCENE_ID = "parking" + "_" + "underground"


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
    core_placement: str = "auto"
    topology_mode: Literal["continuous_cpsat", "grid_growth"] = "grid_growth"
    corridor_layout: Literal["door_side", "organic"] = "organic"
    provider: Optional[Literal["gemini", "deepseek", "openai"]] = None
    model: Optional[str] = None
    scene_type: Optional[str] = None
    sceneId: Optional[str] = None

    @model_validator(mode="after")
    def _reject_legacy_modes(self) -> "BuildingGenerateRequest":
        if self.scene_type and str(self.scene_type).lower() != "building":
            raise ValueError("Only scene_type=building is supported in the active backend-only API")
        if self.sceneId and str(self.sceneId).lower() == _REMOVED_PARKING_SCENE_ID:
            raise ValueError("Removed underground parking scene is not supported in the active backend-only API")
        return self

