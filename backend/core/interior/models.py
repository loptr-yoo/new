from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FurnitureCategory(str, Enum):
    BEDDING = "床具"
    SEATING = "坐具"
    APPLIANCE = "电器"
    CABINET = "柜子"
    TABLE = "桌子"
    CHAIR = "椅子"
    HANGING = "挂件"
    DECOR = "摆件"


class FurnitureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    category: FurnitureCategory
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    priority: int = Field(1, ge=0, le=10)


class RoomBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def _check_minmax(self) -> "RoomBoundary":
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("RoomBoundary 非法：要求 x_min < x_max 且 y_min < y_max")
        return self


class Obstacle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def _check_minmax(self) -> "Obstacle":
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("Obstacle 非法：要求 x_min < x_max 且 y_min < y_max")
        return self


Rotation = Literal[0, 90, 180, 270]


class LLMCoarseLayoutItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    furniture_id: str = Field(..., min_length=1)
    cx: float
    cy: float
    rotation: Rotation


class LLMCoarseLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(..., min_length=1)
    items: List[LLMCoarseLayoutItem] = Field(default_factory=list)


class RefinedLayoutItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    furniture_id: str = Field(..., min_length=1)
    cx: float
    cy: float
    rotation: Rotation


RefinedStatus = Literal["optimal", "feasible", "fallback", "infeasible", "error"]


class RefinedLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RefinedStatus
    reasoning: str = Field(..., min_length=1)
    items: List[RefinedLayoutItem] = Field(default_factory=list)
    objective_l1: Optional[float] = None
    solver: str = Field(..., min_length=1)
    warnings: List[str] = Field(default_factory=list)
