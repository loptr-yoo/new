from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

STAGE1_SCHEMA_VERSION = "stage1.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Stage1ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    schema_version: str = STAGE1_SCHEMA_VERSION
    stage: str = "stage1"
    source: Literal["llm", "mock", "fixture"] = "mock"
    run_id: str
    generated_at: str = Field(default_factory=utc_now_iso)


class EnvelopeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gross_area: float = Field(..., gt=0)
    total_floors: int = Field(..., ge=2)
    width: Optional[float] = Field(default=None, gt=0)
    depth: Optional[float] = Field(default=None, gt=0)
    polygon: Optional[list[list[float]]] = None
    orientation: str = "north_up"
    source: str = "derived"

    @property
    def gross_area_per_floor(self) -> float:
        return float(self.gross_area) / float(self.total_floors)

    @property
    def has_concrete_geometry(self) -> bool:
        return bool((self.width and self.depth) or self.polygon)


class RawRoomDraft(BaseModel):
    model_config = ConfigDict(extra="allow")

    room_id: str = ""
    room_name: str
    room_type: str
    target_area: Optional[float] = Field(default=None, gt=0)
    zone: Optional[str] = None
    needs_window: Optional[bool] = None
    optional: Optional[bool] = None


class RawFloorDraft(BaseModel):
    model_config = ConfigDict(extra="allow")

    floor_number: int = Field(..., ge=1)
    floor_role: str = "standard"
    gross_area: Optional[float] = Field(default=None, gt=0)
    rooms: list[RawRoomDraft] = Field(default_factory=list)


class RawProgramDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    building_name: str = "Building"
    building_type: str = "residential"
    archetype: str = "residential"
    total_floors: int = Field(..., ge=2)
    envelope: EnvelopeSpec
    floors: list[RawFloorDraft] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_floor_count(self) -> "RawProgramDraft":
        if len(self.floors) != int(self.total_floors):
            raise ValueError("RawProgramDraft floors length must match total_floors")
        return self


class RoomProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str
    min_area: float
    original_target_area: float
    target_area: float
    max_area: float
    needs_window: bool = False
    wet_zone: bool = False
    privacy: str = "public"
    publicness: str = "public"
    zone: str = "public"
    optional: bool = False
    repair_status: str = "unchanged"
    preferred_adjacencies: list[str] = Field(default_factory=list)
    forbidden_adjacencies: list[str] = Field(default_factory=list)


class FloorProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    floor_number: int
    floor_role: str
    gross_area: float
    rooms: list[RoomProgram] = Field(default_factory=list)
    adjacency_preferences: list[list[str]] = Field(default_factory=list)
    forbidden_adjacencies: list[list[str]] = Field(default_factory=list)
    zones: dict[str, list[str]] = Field(default_factory=dict)


class CorePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connectivity_type: Literal["stair", "elevator", "stair_and_elevator"]
    placement_preference: str = "auto"
    selected_placement: str = "auto"
    strict_core_placement: bool = False
    core_area: float
    core_size: dict[str, float] = Field(default_factory=dict)
    core_bbox: Optional[dict[str, float]] = None
    cross_floor_alignment: str = "fixed_shared_core"
    source_of_truth: bool = True


class CorridorPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layout: str = "organic"
    reserve_ratio: float = 0.16
    wall_reserve_ratio: float = 0.04
    min_width: float = 1.2
    target_width: float = 2.0


class Stage1CoreContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core_source: str = "stage1"
    stage1_core_policy_id: str
    connectivity_type: str
    selected_placement: str
    core_area: float
    floor_count: int
    bbox: Optional[dict[str, float]] = None
    passable: bool = True
    failure_type: Optional[str] = None


class Stage1CorridorContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corridor_source: str = "stage1"
    layout: str
    reserve_ratio: float
    wall_reserve_ratio: float
    target_width: float
    passable: bool = True
    failure_type: Optional[str] = None


class FeasibilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    floor_number: int
    gross_area: float
    core_area: float
    corridor_reserve: float
    wall_reserve: float
    usable_area: float
    sum_min_room_area: float
    sum_target_room_area: float
    status: Literal["feasible", "target_scaled", "infeasible"]
    repair_action: Optional[str] = None
    scale_factor: float = 1.0
    failure_type: Optional[str] = None
    feasibility_level: str = "program"
    geometry_guaranteed: bool = False


class ProgramRepairLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[dict[str, Any]] = Field(default_factory=list)
    user_preference_overridden: bool = False


class BuildingProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_type: str = "building"
    building_name: str
    building_type: str
    archetype: str
    total_floors: int
    gross_area_per_floor: float
    envelope: EnvelopeSpec
    core_policy: CorePolicy
    corridor_policy: CorridorPolicy
    floors: list[FloorProgram]
    global_constraints: dict[str, Any] = Field(default_factory=dict)


class Stage1Result(Stage1ArtifactModel):
    artifact_type: str = "stage1_result"
    envelope: EnvelopeSpec
    building_program: BuildingProgram
    floor_programs: list[FloorProgram]
    feasibility_reports: list[FeasibilityReport]
    core_policy: CorePolicy
    corridor_policy: CorridorPolicy
    core_context: Stage1CoreContext
    corridor_context: Stage1CorridorContext
    program_repair_log: ProgramRepairLog
    policy_validation: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def can_enter_geometry(self) -> bool:
        policy_valid = bool(self.policy_validation.get("valid", False))
        feasibility_valid = all(r.status != "infeasible" for r in self.feasibility_reports)
        core_valid = self.core_context.core_source == "stage1" and bool(self.core_context.passable)
        corridor_valid = self.corridor_context.corridor_source == "stage1" and bool(self.corridor_context.passable)
        return bool(policy_valid and feasibility_valid and core_valid and corridor_valid)
