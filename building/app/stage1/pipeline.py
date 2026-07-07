from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Any, Optional

from shapely.geometry import Polygon

from ..models import BuildingAllocation, FloorAllocation, RoomAllocation
from ..pipeline_defaults import DEFAULT_CORRIDOR_LAYOUT, DEFAULT_SCENE_TYPE
from ..policies import load_policy
from ..diagnostics.failure_taxonomy import route_failure
from ..geometry.topology_generator import CoreTube
from .models import (
    BuildingProgram,
    CorePolicy,
    CorridorPolicy,
    EnvelopeSpec,
    FeasibilityReport,
    FloorProgram,
    ProgramRepairLog,
    RawFloorDraft,
    RawProgramDraft,
    RawRoomDraft,
    RoomProgram,
    Stage1CoreContext,
    Stage1CorridorContext,
    Stage1Result,
)


class Stage1ProgramInfeasibleError(RuntimeError):
    def __init__(self, result: Stage1Result):
        self.result = result
        payload = stage1_failure_payload(result)
        self.payload = payload
        super().__init__(json.dumps(payload, ensure_ascii=False))


class Stage1ContextMismatchError(RuntimeError):
    def __init__(self, failure_type: str, message: str, metadata: Optional[dict[str, Any]] = None):
        self.failure_type = failure_type
        self.metadata = dict(metadata or {})
        super().__init__(message)


def _stable_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stage1_failure_payload(result: Stage1Result) -> dict[str, Any]:
    failures = [
        r.model_dump(mode="json")
        for r in result.feasibility_reports
        if r.status == "infeasible" or r.failure_type
    ]
    first_failure = failures[0]["failure_type"] if failures else "program_infeasible"
    return {
        "result": "pipeline_failed",
        "artifact_valid": False,
        "stage": "stage1",
        "failure_type": first_failure or "program_infeasible",
        "failure_taxonomy_route": route_failure(first_failure or "program_infeasible").__dict__,
        "can_enter_geometry": result.can_enter_geometry,
        "feasibility_reports": [r.model_dump(mode="json") for r in result.feasibility_reports],
        "program_repair_log": result.program_repair_log.model_dump(mode="json"),
        "policy_validation": result.policy_validation,
        "core_context": result.core_context.model_dump(mode="json"),
        "corridor_context": result.corridor_context.model_dump(mode="json"),
    }


def _room_rule(policy: dict[str, Any], room_type: str, repair_log: ProgramRepairLog, policy_report: dict[str, Any]) -> dict[str, Any]:
    rules = policy.get("room_rules") or {}
    if room_type in rules:
        return dict(rules[room_type])
    # Transitional fallback for unknown custom rooms.
    from ..geometry.room_spec import ROOM_TYPE_DEFAULTS

    legacy = ROOM_TYPE_DEFAULTS.get(room_type) or {}
    fallback = {
        "min_area": 6.0,
        "target_area": 10.0,
        "max_area": 18.0,
        "zone": str(getattr(legacy.get("zone", "public"), "value", legacy.get("zone", "public"))),
        "needs_window": bool(legacy.get("needs_window", False)),
        "optional": True,
    }
    policy_report.setdefault("fallback_usages", []).append(
        {"key": f"room_rules.{room_type}", "fallback_source": "ROOM_TYPE_DEFAULTS", "value": fallback}
    )
    repair_log.actions.append(
        {"action": "policy_fallback", "reason": "missing_room_rule", "room_type": room_type}
    )
    return fallback


def _privacy_for_zone(zone: str) -> str:
    return "private" if zone == "private" else "public"


def _wet_zone(room_type: str) -> bool:
    return room_type in {"bathroom", "toilet", "kitchen", "laundry", "utility", "pantry"}


def _normalize_floor(
    raw_floor: RawFloorDraft,
    *,
    policy: dict[str, Any],
    repair_log: ProgramRepairLog,
    policy_report: dict[str, Any],
    gross_area_per_floor: float,
) -> FloorProgram:
    rooms: list[RoomProgram] = []
    forbidden_by_type = ((policy.get("adjacency_rules") or {}).get("forbidden") or {})
    for idx, raw_room in enumerate(raw_floor.rooms, start=1):
        room_type = str(raw_room.room_type or "room").strip().lower().replace(" ", "_")
        rule = _room_rule(policy, room_type, repair_log, policy_report)
        rid = raw_room.room_id or f"F{raw_floor.floor_number}_{room_type}_{idx}"
        original_target = float(raw_room.target_area or rule.get("target_area") or rule.get("min_area") or 6.0)
        zone = str(raw_room.zone or rule.get("zone") or "public")
        optional = bool(rule.get("optional", False) if raw_room.optional is None else raw_room.optional)
        needs_window = bool(rule.get("needs_window", False) if raw_room.needs_window is None else raw_room.needs_window)
        forbidden_types = set(forbidden_by_type.get(room_type, []) or [])
        rooms.append(
            RoomProgram(
                id=str(rid),
                name=str(raw_room.room_name),
                type=room_type,
                min_area=float(rule.get("min_area") or 1.0),
                original_target_area=original_target,
                target_area=min(original_target, float(rule.get("max_area") or original_target)),
                max_area=float(rule.get("max_area") or max(original_target, 1.0)),
                needs_window=needs_window,
                wet_zone=_wet_zone(room_type),
                privacy=_privacy_for_zone(zone),
                publicness=zone,
                zone=zone,
                optional=optional,
                forbidden_adjacencies=[
                    r.id for r in rooms if r.type in forbidden_types
                ],
            )
        )
    zones: dict[str, list[str]] = {}
    for room in rooms:
        zones.setdefault(room.zone, []).append(room.id)
    return FloorProgram(
        floor_number=int(raw_floor.floor_number),
        floor_role=str(raw_floor.floor_role or "standard"),
        gross_area=float(raw_floor.gross_area or gross_area_per_floor),
        rooms=rooms,
        zones=zones,
    )


def _corridor_policy(policy: dict[str, Any]) -> CorridorPolicy:
    rules = policy.get("corridor_rules") or {}
    return CorridorPolicy(
        layout=str(rules.get("layout") or DEFAULT_CORRIDOR_LAYOUT),
        reserve_ratio=float(rules.get("reserve_ratio", 0.16)),
        wall_reserve_ratio=float(rules.get("wall_reserve_ratio", 0.04)),
        min_width=float(rules.get("min_width", 1.2)),
        target_width=float(rules.get("target_width", 2.0)),
    )


def _core_policy(
    raw: RawProgramDraft,
    *,
    policy: dict[str, Any],
    envelope: EnvelopeSpec,
    requested_placement: str = "auto",
    strict_core_placement: Optional[bool] = None,
    repair_log: ProgramRepairLog,
) -> CorePolicy:
    rules = policy.get("vertical_core_rules") or {}
    floor_threshold = int(rules.get("elevator_floor_threshold", 4) or 4)
    connectivity = "stair_and_elevator" if int(raw.total_floors) >= floor_threshold else "stair"
    area_ratio = float(rules.get("area_ratio", 0.12))
    core_area = envelope.gross_area_per_floor * area_ratio
    side = math.sqrt(max(core_area, 0.01))
    preference = str(requested_placement or rules.get("placement_default") or "auto")
    strict = bool(rules.get("strict_core_placement", False) if strict_core_placement is None else strict_core_placement)
    candidates = list(rules.get("candidate_positions") or ["east", "north", "south", "west"])
    selected = preference if preference != "auto" else (candidates[0] if candidates else "east")
    bbox = None
    if envelope.width and envelope.depth:
        width = min(float(envelope.width) * 0.35, side)
        depth = max(core_area / max(width, 0.01), 0.01)
        if selected == "east":
            bbox = {"x": float(envelope.width) - width, "y": (float(envelope.depth) - depth) / 2.0, "width": width, "depth": depth}
        elif selected == "west":
            bbox = {"x": 0.0, "y": (float(envelope.depth) - depth) / 2.0, "width": width, "depth": depth}
        elif selected == "north":
            bbox = {"x": (float(envelope.width) - width) / 2.0, "y": float(envelope.depth) - depth, "width": width, "depth": depth}
        elif selected == "south":
            bbox = {"x": (float(envelope.width) - width) / 2.0, "y": 0.0, "width": width, "depth": depth}
        elif strict:
            repair_log.actions.append({"action": "core_policy_infeasible", "requested_placement": selected})
    return CorePolicy(
        connectivity_type=connectivity,  # type: ignore[arg-type]
        placement_preference=preference,
        selected_placement=selected,
        strict_core_placement=strict,
        core_area=float(core_area),
        core_size={"area": float(core_area), "approx_side": float(side)},
        core_bbox=bbox,
    )


def _apply_feasibility(
    floor: FloorProgram,
    *,
    core_policy: CorePolicy,
    corridor_policy: CorridorPolicy,
    repair_log: ProgramRepairLog,
) -> FeasibilityReport:
    gross = float(floor.gross_area)
    core_area = float(core_policy.core_area)
    corridor_reserve = gross * float(corridor_policy.reserve_ratio)
    wall_reserve = gross * float(corridor_policy.wall_reserve_ratio)
    usable = max(0.0, gross - core_area - corridor_reserve - wall_reserve)
    sum_min = sum(float(r.min_area) for r in floor.rooms)
    sum_target = sum(float(r.target_area) for r in floor.rooms)
    if sum_min > usable + 1e-6:
        return FeasibilityReport(
            floor_number=floor.floor_number,
            gross_area=gross,
            core_area=core_area,
            corridor_reserve=corridor_reserve,
            wall_reserve=wall_reserve,
            usable_area=usable,
            sum_min_room_area=sum_min,
            sum_target_room_area=sum_target,
            status="infeasible",
            repair_action=None,
            failure_type="min_area_exceeds_usable_area",
        )
    if sum_target <= usable + 1e-6:
        return FeasibilityReport(
            floor_number=floor.floor_number,
            gross_area=gross,
            core_area=core_area,
            corridor_reserve=corridor_reserve,
            wall_reserve=wall_reserve,
            usable_area=usable,
            sum_min_room_area=sum_min,
            sum_target_room_area=sum_target,
            status="feasible",
        )
    flexible = sum(max(0.0, float(r.target_area) - float(r.min_area)) for r in floor.rooms)
    scale = 0.0 if flexible <= 1e-6 else max(0.0, (usable - sum_min) / flexible)
    affected: list[str] = []
    for idx, room in enumerate(floor.rooms):
        old = float(room.target_area)
        new = float(room.min_area) + max(0.0, old - float(room.min_area)) * scale
        if abs(new - old) > 1e-6:
            affected.append(room.id)
            floor.rooms[idx] = room.model_copy(update={"target_area": new, "repair_status": "scaled"})
    repaired_sum = sum(float(r.target_area) for r in floor.rooms)
    repair_log.actions.append(
        {
            "action": "scale_targets",
            "reason": "target_area_exceeds_usable_area",
            "floor_number": floor.floor_number,
            "scale_factor": scale,
            "affected_rooms": affected,
            "not_scaled_below_min": True,
        }
    )
    return FeasibilityReport(
        floor_number=floor.floor_number,
        gross_area=gross,
        core_area=core_area,
        corridor_reserve=corridor_reserve,
        wall_reserve=wall_reserve,
        usable_area=usable,
        sum_min_room_area=sum_min,
        sum_target_room_area=repaired_sum,
        status="target_scaled",
        repair_action="scale_targets",
        scale_factor=scale,
        failure_type="target_area_exceeds_usable_area",
    )


def run_stage1_from_raw(
    raw: RawProgramDraft,
    *,
    source: str = "mock",
    run_id: Optional[str] = None,
    core_placement: str = "auto",
    strict_core_placement: Optional[bool] = None,
) -> Stage1Result:
    rid = run_id or f"stage1-{uuid.uuid4().hex[:8]}"
    policy, validation = load_policy(raw.archetype)
    policy_report = validation.to_dict()
    repair_log = ProgramRepairLog()
    corridor = _corridor_policy(policy)
    core = _core_policy(
        raw,
        policy=policy,
        envelope=raw.envelope,
        requested_placement=core_placement,
        strict_core_placement=strict_core_placement,
        repair_log=repair_log,
    )
    floors = [
        _normalize_floor(
            f,
            policy=policy,
            repair_log=repair_log,
            policy_report=policy_report,
            gross_area_per_floor=raw.envelope.gross_area_per_floor,
        )
        for f in raw.floors
    ]
    reports = [
        _apply_feasibility(f, core_policy=core, corridor_policy=corridor, repair_log=repair_log)
        for f in floors
    ]
    core_context = Stage1CoreContext(
        stage1_core_policy_id=_stable_hash(
            {
                "connectivity_type": core.connectivity_type,
                "selected_placement": core.selected_placement,
                "core_area": round(float(core.core_area), 6),
                "floor_count": int(raw.total_floors),
                "bbox": core.core_bbox,
            }
        ),
        connectivity_type=str(core.connectivity_type),
        selected_placement=str(core.selected_placement),
        core_area=float(core.core_area),
        floor_count=int(raw.total_floors),
        bbox=core.core_bbox,
    )
    corridor_context = Stage1CorridorContext(
        layout=str(corridor.layout),
        reserve_ratio=float(corridor.reserve_ratio),
        wall_reserve_ratio=float(corridor.wall_reserve_ratio),
        target_width=float(corridor.target_width),
    )
    building = BuildingProgram(
        scene_type=DEFAULT_SCENE_TYPE,
        building_name=raw.building_name,
        building_type=raw.building_type,
        archetype=raw.archetype,
        total_floors=raw.total_floors,
        gross_area_per_floor=raw.envelope.gross_area_per_floor,
        envelope=raw.envelope,
        core_policy=core,
        corridor_policy=corridor,
        floors=floors,
        global_constraints={"stage1_core_source_of_truth": True},
    )
    return Stage1Result(
        source=source,  # type: ignore[arg-type]
        run_id=rid,
        envelope=raw.envelope,
        building_program=building,
        floor_programs=floors,
        feasibility_reports=reports,
        core_policy=core,
        corridor_policy=corridor,
        core_context=core_context,
        corridor_context=corridor_context,
        program_repair_log=repair_log,
        policy_validation=policy_report,
    )


def allocation_to_raw_program(
    allocation: BuildingAllocation,
    *,
    source: str = "mock",
    width: Optional[float] = None,
    depth: Optional[float] = None,
) -> RawProgramDraft:
    floors: list[RawFloorDraft] = []
    for floor in allocation.floors:
        floors.append(
            RawFloorDraft(
                floor_number=int(floor.floor_number),
                floor_role=str(floor.floor_function_tag),
                gross_area=float(floor.floor_total_area),
                rooms=[
                    RawRoomDraft(
                        room_id=str(room.room_id),
                        room_name=str(room.room_name),
                        room_type=str(room.room_type),
                        target_area=float(room.target_area),
                        zone=str(room.zone),
                        needs_window=bool(room.needs_window),
                    )
                    for room in floor.rooms
                ],
            )
        )
    return RawProgramDraft(
        building_name=str(allocation.building_name),
        building_type="residential",
        archetype="residential",
        total_floors=int(allocation.total_floors),
        envelope=EnvelopeSpec(
            gross_area=float(allocation.overall_total_area),
            total_floors=int(allocation.total_floors),
            width=width,
            depth=depth,
            source=source,
        ),
        floors=floors,
    )


def run_stage1_from_allocation(
    allocation: BuildingAllocation,
    *,
    source: str = "mock",
    run_id: Optional[str] = None,
    core_placement: str = "auto",
) -> Stage1Result:
    raw = allocation_to_raw_program(allocation, source=source)
    return run_stage1_from_raw(raw, source=source, run_id=run_id, core_placement=core_placement)


def building_allocation_from_stage1(result: Stage1Result) -> BuildingAllocation:
    if not result.can_enter_geometry:
        raise Stage1ProgramInfeasibleError(result)
    floors: list[FloorAllocation] = []
    for floor, report in zip(result.floor_programs, result.feasibility_reports):
        floors.append(
            FloorAllocation(
                floor_number=int(floor.floor_number),
                floor_function_tag=str(floor.floor_role),
                floor_total_area=float(floor.gross_area),
                core_tube_area=float(report.core_area),
                corridor_allowance_area=float(report.corridor_reserve),
                rooms=[
                    RoomAllocation(
                        room_id=room.id,
                        room_name=room.name,
                        room_type=room.type,
                        target_area=float(room.target_area),
                        zone=room.zone,
                        needs_window=bool(room.needs_window),
                        min_width=1.8 if room.type in {"bathroom", "toilet"} else 2.4,
                        adjacency_required=[],
                        adjacency_preferred=list(room.preferred_adjacencies),
                        adjacency_forbidden=list(room.forbidden_adjacencies),
                        size_hint=None,
                    )
                    for room in floor.rooms
                ],
            )
        )
    return BuildingAllocation(
        building_name=result.building_program.building_name,
        total_floors=result.building_program.total_floors,
        overall_total_area=float(result.envelope.gross_area),
        floors=floors,
    )


def core_tube_from_stage1_policy(
    result: Stage1Result,
    floor_boundary: Polygon,
    *,
    require_resolved_bbox: bool = False,
) -> tuple[CoreTube, dict[str, Any]]:
    ctx = result.core_context
    if ctx.core_source != "stage1":
        raise Stage1ContextMismatchError("core_context_mismatch", "Core context source is not stage1")
    if not ctx.passable:
        raise Stage1ContextMismatchError(ctx.failure_type or "core_context_mismatch", "Core context is not passable")
    if require_resolved_bbox and ctx.bbox is None:
        raise Stage1ContextMismatchError(
            "core_policy_unresolved",
            "Stage 1 core policy has no resolved bbox and resolved core geometry is required",
            {"stage1_core_policy_id": ctx.stage1_core_policy_id},
        )
    core = CoreTube.create_for_floor(
        floor_bounds=floor_boundary.bounds,
        area_ratio=max(0.001, float(ctx.core_area) / max(float(floor_boundary.area), 1e-6)),
        position=str(ctx.selected_placement or "east"),
    )
    metadata = {
        "core_source": "stage1",
        "stage1_core_policy_id": ctx.stage1_core_policy_id,
        "connectivity_type": ctx.connectivity_type,
        "selected_placement": ctx.selected_placement,
        "core_area": ctx.core_area,
        "floor_count": ctx.floor_count,
        "bbox": ctx.bbox,
    }
    return core, metadata


def validate_stage1_core_context(result: Stage1Result, metadata: dict[str, Any]) -> None:
    ctx = result.core_context
    checks = {
        "core_source": metadata.get("core_source") == "stage1",
        "stage1_core_policy_id": metadata.get("stage1_core_policy_id") == ctx.stage1_core_policy_id,
        "selected_placement": metadata.get("selected_placement") == ctx.selected_placement,
        "floor_count": int(metadata.get("floor_count", -1)) == int(ctx.floor_count),
        "core_area": abs(float(metadata.get("core_area", -1.0)) - float(ctx.core_area)) <= 1e-6,
    }
    if not all(checks.values()):
        raise Stage1ContextMismatchError("core_context_mismatch", "Stage 1 core context mismatch", checks)


def stage2_corridor_options_from_stage1(result: Stage1Result) -> dict[str, Any]:
    ctx = result.corridor_context
    if ctx.corridor_source != "stage1" or not ctx.passable:
        raise Stage1ContextMismatchError(ctx.failure_type or "corridor_context_mismatch", "Corridor context is not passable")
    return {
        "corridor_source": "stage1",
        "corridor_layout": ctx.layout,
        "reserve_ratio": ctx.reserve_ratio,
        "wall_reserve_ratio": ctx.wall_reserve_ratio,
        "target_width": ctx.target_width,
    }


def validate_stage1_corridor_context(result: Stage1Result, options: dict[str, Any]) -> None:
    ctx = result.corridor_context
    checks = {
        "corridor_source": options.get("corridor_source") == "stage1",
        "corridor_layout": options.get("corridor_layout") == ctx.layout,
        "reserve_ratio": abs(float(options.get("reserve_ratio", -1.0)) - float(ctx.reserve_ratio)) <= 1e-6,
        "wall_reserve_ratio": abs(float(options.get("wall_reserve_ratio", -1.0)) - float(ctx.wall_reserve_ratio)) <= 1e-6,
    }
    if not all(checks.values()):
        raise Stage1ContextMismatchError("corridor_context_mismatch", "Stage 1 corridor context mismatch", checks)
