from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class LayoutElement(BaseModel):
    id: str
    type: str
    x: float
    y: float
    width: float
    height: float
    rotation: Optional[float] = None
    label: Optional[str] = None
    subType: Optional[str] = None
    forward: Optional[Tuple[float, float, float]] = None


class ParkingLayout(BaseModel):
    width: float
    height: float
    elements: List[LayoutElement]
    sceneId: Optional[str] = None


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
    blueprint: List[LayoutElement]
    floors: Dict[str, ParkingLayout]


class GenerateRequest(BaseModel):
    prompt: str
    provider: Literal["gemini", "deepseek", "openai"]
    model: str
    sceneId: Optional[str] = None
    floorCount: Optional[int] = Field(default=None)


class AugmentRequest(BaseModel):
    layout: ParkingLayout
    provider: Literal["gemini", "deepseek", "openai"]
    model: str
    sceneId: Optional[str] = None

