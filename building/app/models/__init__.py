from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LayoutElement(BaseModel):
    id: str
    type: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation: Optional[float] = None
    label: Optional[str] = None
    subType: Optional[str] = None
    forward: Optional[Tuple[float, float, float]] = None
    polygon: Optional[list] = None
    zOrder: Optional[int] = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_forward(self) -> "LayoutElement":
        if self.forward is not None and len(self.forward) != 3:
            raise ValueError("forward must have length 3")
        return self


class ParkingLayout(BaseModel):
    """Legacy-compatible floor layout container used internally by geometry helpers."""

    width: float
    height: float
    elements: List[LayoutElement]
    sceneId: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ConstraintViolation(BaseModel):
    elementId: str
    targetId: Optional[str] = None
    type: Literal[
        "overlap",
        "out_of_bounds",
        "invalid_dimension",
        "placement_error",
        "connectivity_error",
        "width_mismatch",
    ]
    message: str


class BuildingData(BaseModel):
    blueprint: List[LayoutElement] = Field(default_factory=list)
    floors: Dict[str, ParkingLayout] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    prompt: str
    provider: Literal["gemini", "deepseek", "openai"]
    model: str
    sceneId: Optional[str] = None
    floorCount: Optional[int] = None


class AugmentRequest(BaseModel):
    layout: ParkingLayout
    provider: Literal["gemini", "deepseek", "openai"]
    model: str
    sceneId: Optional[str] = None


class SceneType(str, Enum):
    FLOOR = "floor"
    PARKING = "parking"
    BUILDING = "building"


class GenerateSemanticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_type: SceneType = Field(default=SceneType.BUILDING)
    user_prompt: str = Field(..., min_length=1)
    total_area: Optional[float] = Field(default=None, gt=0)
    total_floors: Optional[int] = Field(default=None, ge=2)
    provider: Optional[Literal["gemini", "deepseek", "openai"]] = None
    model: Optional[str] = Field(default=None, min_length=1)


class RoomAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: str = ""
    room_name: str = Field(..., min_length=1)
    room_type: str = Field(..., min_length=1)
    target_area: float = Field(..., gt=0)
    zone: str = "public"
    needs_window: bool = False
    min_width: float = Field(default=2.5, ge=0)
    aspect_ratio_range: List[float] = Field(default_factory=lambda: [0.5, 2.0])
    adjacency_required: List[str] = Field(default_factory=list)
    adjacency_preferred: List[str] = Field(default_factory=list)
    adjacency_forbidden: List[str] = Field(default_factory=list)
    size_hint: Optional[str] = None
    weight: int = Field(default=5, ge=1, le=10)
    requires_window: Optional[bool] = None
    adjacency_tags: List[str] = Field(default_factory=list)


class FloorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    floor_number: int = Field(..., ge=1)
    floor_function_tag: str = Field(..., min_length=1)
    requested_rooms_list: List[str] = Field(default_factory=list)


class BuildingEnvelopeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    building_name: str = Field(..., min_length=1)
    total_floors: int = Field(..., ge=2)
    overall_total_area: float = Field(..., gt=0)
    floors: List[FloorEnvelope] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_floor_count(self) -> "BuildingEnvelopeBase":
        if len(self.floors) != int(self.total_floors):
            raise ValueError(
                f"Envelope floor count mismatch: total_floors={self.total_floors}, floors={len(self.floors)}"
            )
        expected = list(range(1, int(self.total_floors) + 1))
        actual = [int(f.floor_number) for f in self.floors]
        if actual != expected:
            raise ValueError(f"Envelope floor_number must be continuous from 1: got {actual}")
        return self


class FloorAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    floor_number: int = Field(..., ge=1)
    floor_function_tag: str = Field(..., min_length=1)
    floor_total_area: float = Field(..., gt=0)
    core_tube_area: float = Field(..., ge=0)
    corridor_allowance_area: float = Field(..., ge=0)
    rooms: List[RoomAllocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_area_budget(self) -> "FloorAllocation":
        if self.floor_total_area - self.core_tube_area - self.corridor_allowance_area <= 0:
            raise ValueError("floor_total_area must exceed core_tube_area + corridor_allowance_area")
        if sum(float(r.target_area) for r in self.rooms) <= 0:
            raise ValueError("room target_area sum must be positive")
        return self


class BuildingAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    building_name: str = Field(..., min_length=1)
    total_floors: int = Field(..., ge=2)
    overall_total_area: float = Field(..., gt=0)
    floors: List[FloorAllocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_multifloor_count(self) -> "BuildingAllocation":
        if len(self.floors) != int(self.total_floors):
            raise ValueError(
                f"BuildingAllocation floor count mismatch: total_floors={self.total_floors}, floors={len(self.floors)}"
            )
        expected = list(range(1, int(self.total_floors) + 1))
        actual = [int(f.floor_number) for f in self.floors]
        if actual != expected:
            raise ValueError(f"BuildingAllocation floor_number must be continuous from 1: got {actual}")
        return self
