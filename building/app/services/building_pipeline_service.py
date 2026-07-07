from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, List, Optional, Tuple

from shapely.geometry import Polygon, box

from ..models import BuildingAllocation, BuildingEnvelopeBase, GenerateSemanticsRequest, RoomAllocation, SceneType
from ..logger import current_log_context, log_context, log_multiline_debug
from ..geometry.building_orchestrator import BuildingOrchestrator, BuildingResult
from ..geometry.corridor_policy import normalize_corridor_width
from ..geometry.exceptions import (
    LayoutAssignmentError,
    LayoutCoverageError,
    LayoutGeometryInvariantError,
    LayoutTopologyError,
    SemanticInvalidError,
)
from ..geometry.room_spec import SolverConfig
from ..geometry.serializers import building_result_to_dict, core_tube_to_dict, serialize_single_floor
from ..geometry.topology_generator import CoreTube
from ..geometry.topology_snapshot import BuildingAreaBudget, LayoutFailureReport, TopologySnapshot, compute_building_area_budget, repair_building_allocation_with_budget
from ..pipeline_defaults import (
    DEFAULT_CORE_PLACEMENT,
    DEFAULT_CORRIDOR_LAYOUT,
    DEFAULT_TOPOLOGY_MODE,
    LEGACY_CORE_PLACEMENT_PREFERENCE,
)
from ..semantics.generator import (
    BudgetValidationError,
    BudgetedBuildingSemanticResult,
    generate_building_envelope,
    generate_budgeted_building_semantics,
    validate_allocation_adjacency_ids,
    validate_allocation_against_budget,
)
from ..stage1 import Stage1ContextMismatchError, Stage1ProgramInfeasibleError, building_allocation_from_stage1, core_tube_from_stage1_policy, run_stage1_from_allocation, stage1_failure_payload, stage2_corridor_options_from_stage1, validate_stage1_core_context, validate_stage1_corridor_context
from ..stage1.models import Stage1Result

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildingPipelineOptions:
    floor_width: Optional[float] = None
    floor_depth: Optional[float] = None
    corridor_layout: str = DEFAULT_CORRIDOR_LAYOUT
    corridor_width: Optional[float] = None
    topology_mode: str = DEFAULT_TOPOLOGY_MODE
    core_area_ratio: Optional[float] = None
    core_placement: str = DEFAULT_CORE_PLACEMENT
    base_seed: Optional[int] = None
    config: Optional[SolverConfig] = None
    on_llm_output: Optional[Callable[[str, str], None]] = None
    use_stage1_program: bool = False


@dataclass
class BuildingPipelineResult:
    envelope: BuildingEnvelopeBase
    budget: BuildingAreaBudget
    topology_snapshot: Optional[TopologySnapshot]
    allocation: Optional[BuildingAllocation]
    building_result: Optional[BuildingResult]
    building_dict: dict
    floor_boundary: Polygon
    floor_width: float
    floor_height: float
    corridor_width: float
    core_area_ratio: float
    warnings: List[str]
    success: bool = True
    artifact_valid: bool = True
    failure: Optional[dict] = None


def _derive_floor_boundary(
    envelope: BuildingEnvelopeBase,
    options: BuildingPipelineOptions,
) -> Tuple[float, float, Polygon]:
    if options.floor_width and options.floor_depth:
        width = float(options.floor_width)
        height = float(options.floor_depth)
        return width, height, box(0.0, 0.0, width, height)

    area = float(envelope.overall_total_area) / max(1, int(envelope.total_floors))
    width = math.sqrt(area * 1.5)
    height = area / width
    return float(width), float(height), box(0.0, 0.0, float(width), float(height))


def _pick_corridor_width_and_core_ratio(floor_area: float) -> Tuple[float, float]:
    if floor_area < 80:
        return 1.5, 0.08
    if floor_area < 120:
        return 2.0, 0.12
    return 2.5, 0.12


def _floor_id_for_number(floor_number: int) -> str:
    return f"F{int(floor_number)}"


def _find_floor(allocation: BuildingAllocation, floor_number: int) -> Any:
    for floor in allocation.floors:
        if int(floor.floor_number) == int(floor_number):
            return floor
    return None


def _clone_with_repaired_floor(
    original: BuildingAllocation,
    repaired: BuildingAllocation,
    failed_floor_number: int,
) -> BuildingAllocation:
    failed_floor = _find_floor(repaired, failed_floor_number)
    if failed_floor is None:
        raise BudgetValidationError(
            "Topology Validation Error: repair response did not include the failed floor "
            f"{_floor_id_for_number(failed_floor_number)}. Return the failed floor with valid rooms. "
            "Do not modify any target_area values unless explicitly instructed."
        )

    merged = copy.deepcopy(original)
    merged.floors = []
    for old_floor in original.floors:
        if int(old_floor.floor_number) == int(failed_floor_number):
            merged.floors.append(copy.deepcopy(failed_floor))
        else:
            merged.floors.append(copy.deepcopy(old_floor))
    return merged


class BuildingPipelineService:
    """Stateless building generation service shared by API, CLI, and scripts."""

    async def generate_stage1(
        self,
        request: GenerateSemanticsRequest,
        options: Optional[BuildingPipelineOptions] = None,
        *,
        source: str = "llm",
        allocation: Optional[BuildingAllocation] = None,
    ) -> Stage1Result:
        opts = options or BuildingPipelineOptions()
        if request.scene_type != SceneType.BUILDING:
            raise ValueError("Stage 1 only supports scene_type=building")
        if request.total_floors is not None and int(request.total_floors) < 2:
            raise ValueError("total_floors must be >= 2")
        if allocation is None:
            if source == "mock":
                allocation = self._mock_stage1_allocation(request)
            else:
                semantic_result = await self.generate_semantics_only(request, opts)
                allocation = semantic_result.allocation
        return run_stage1_from_allocation(
            allocation,
            source="mock" if source == "mock" else ("fixture" if source == "fixture" else "llm"),
            core_placement=str(opts.core_placement or DEFAULT_CORE_PLACEMENT),
        )

    def _mock_stage1_allocation(self, request: GenerateSemanticsRequest) -> BuildingAllocation:
        total_floors = max(2, int(request.total_floors or 2))
        floor_area = float(request.total_area or (100.0 * total_floors)) / float(total_floors)

        def room(room_id: str, name: str, room_type: str, area: float, zone: str, needs_window: bool) -> RoomAllocation:
            return RoomAllocation(
                room_id=room_id,
                room_name=name,
                room_type=room_type,
                target_area=float(area),
                zone=zone,
                needs_window=needs_window,
                min_width=1.8 if room_type == "bathroom" else 2.4,
            )

        floors = []
        for n in range(1, total_floors + 1):
            if n == 1:
                rooms = [
                    room("F1_living", "Living Room", "living_room", 24.0, "public", True),
                    room("F1_kitchen", "Kitchen", "kitchen", 12.0, "service", True),
                    room("F1_bedroom_1", "Bedroom 1", "bedroom", 14.0, "private", True),
                    room("F1_bedroom_2", "Bedroom 2", "bedroom", 14.0, "private", True),
                    room("F1_bath", "Bathroom", "bathroom", 7.0, "service", False),
                ]
            else:
                rooms = [
                    room(f"F{n}_family", f"Floor {n} Family Room", "living_room", 18.0, "public", True),
                    room(f"F{n}_bedroom_1", f"Floor {n} Bedroom 1", "bedroom", 14.0, "private", True),
                    room(f"F{n}_bedroom_2", f"Floor {n} Bedroom 2", "bedroom", 14.0, "private", True),
                    room(f"F{n}_bath", f"Floor {n} Bathroom", "bathroom", 7.0, "service", False),
                ]
            floors.append({
                "floor_number": n,
                "floor_function_tag": "residential",
                "floor_total_area": floor_area,
                "core_tube_area": floor_area * 0.12,
                "corridor_allowance_area": floor_area * 0.16,
                "rooms": rooms,
            })
        return BuildingAllocation(
            building_name="mock_stage1_building",
            total_floors=total_floors,
            overall_total_area=floor_area * total_floors,
            floors=floors,
        )
    async def generate_semantics_only(
        self,
        request: GenerateSemanticsRequest,
        options: Optional[BuildingPipelineOptions] = None,
    ) -> BudgetedBuildingSemanticResult:
        opts = options or BuildingPipelineOptions()
        current_session = current_log_context().get("session_id")
        session_id = current_session if current_session and current_session != "-" else f"sem-{uuid.uuid4().hex[:8]}"
        with log_context(session_id=session_id, stage="semantics_only", topology_mode=opts.topology_mode):
            logger.info("[STAGE] Start semantics-only building pipeline | session=%s", session_id)
            return await self._generate_semantics_only_impl(request, opts)

    async def generate(
        self,
        request: GenerateSemanticsRequest,
        options: Optional[BuildingPipelineOptions] = None,
    ) -> BuildingPipelineResult:
        opts = options or BuildingPipelineOptions()
        current_session = current_log_context().get("session_id")
        session_id = current_session if current_session and current_session != "-" else f"bld-{uuid.uuid4().hex[:8]}"
        with log_context(session_id=session_id, stage="pipeline_start", topology_mode=opts.topology_mode):
            logger.info("[STAGE] Start building pipeline | session=%s", session_id)
            return await self._generate_impl(request, opts)

    async def _generate_semantics_only_impl(
        self,
        request: GenerateSemanticsRequest,
        options: Optional[BuildingPipelineOptions] = None,
    ) -> BudgetedBuildingSemanticResult:
        opts = options or BuildingPipelineOptions()
        if opts.use_stage1_program:
            stage1 = await self.generate_stage1(request, opts, source="llm")
            allocation = building_allocation_from_stage1(stage1)
            envelope = BuildingEnvelopeBase.model_validate({
                "building_name": allocation.building_name,
                "total_floors": allocation.total_floors,
                "overall_total_area": allocation.overall_total_area,
                "floors": [
                    {
                        "floor_number": floor.floor_number,
                        "floor_function_tag": floor.floor_function_tag,
                        "requested_rooms_list": [room.room_name for room in floor.rooms],
                    }
                    for floor in allocation.floors
                ],
            })
            _, _, floor_boundary = _derive_floor_boundary(envelope, opts)
            corridor_options = stage2_corridor_options_from_stage1(stage1)
            corridor_layout = str(corridor_options["corridor_layout"])
            corridor_width = normalize_corridor_width(float(opts.corridor_width or corridor_options["target_width"]), corridor_layout)
            fixed_core_tube, core_metadata = core_tube_from_stage1_policy(stage1, floor_boundary)
            validate_stage1_core_context(stage1, core_metadata)
            validate_stage1_corridor_context(stage1, corridor_options)
            core_area_ratio = float(stage1.core_context.core_area) / max(float(floor_boundary.area), 1e-6)
            budget = compute_building_area_budget(
                floor_boundary=floor_boundary,
                floors=allocation.floors,
                corridor_width=corridor_width,
                core_area_ratio=core_area_ratio,
                corridor_layout=corridor_layout,
                base_seed=opts.base_seed,
                fixed_core_tube=fixed_core_tube,
            )
            return BudgetedBuildingSemanticResult(
                allocation=allocation,
                warnings=["Stage 1 program adapter used"],
                envelope=envelope,
                budget=budget,
                topology_snapshot=budget.topology_snapshot,
            )
        envelope = await generate_building_envelope(request, on_llm_output=opts.on_llm_output)
        floor_w, floor_h, floor_boundary = _derive_floor_boundary(envelope, opts)
        default_cw, default_core = _pick_corridor_width_and_core_ratio(floor_w * floor_h)
        corridor_layout = str(opts.corridor_layout or "organic")
        corridor_width = normalize_corridor_width(
            float(opts.corridor_width if opts.corridor_width is not None else default_cw),
            corridor_layout,
        )
        core_area_ratio = float(opts.core_area_ratio if opts.core_area_ratio is not None else default_core)
        fixed_core_tube = self._create_fixed_core(floor_boundary, core_area_ratio, opts.core_placement)
        return await generate_budgeted_building_semantics(
            request,
            floor_boundary=floor_boundary,
            corridor_width=corridor_width,
            core_area_ratio=core_area_ratio,
            corridor_layout=corridor_layout,
            base_seed=opts.base_seed,
            fixed_core_tube=fixed_core_tube,
            envelope=envelope,
            on_llm_output=opts.on_llm_output,
        )
    async def _generate_impl(
        self,
        request: GenerateSemanticsRequest,
        options: Optional[BuildingPipelineOptions] = None,
    ) -> BudgetedBuildingSemanticResult:
        opts = options or BuildingPipelineOptions()
        if opts.use_stage1_program:
            logger.info("[STAGE] Start Stage 1 Program Pass")
            stage1 = await self.generate_stage1(request, opts, source="llm")
            try:
                allocation = building_allocation_from_stage1(stage1)
            except Stage1ProgramInfeasibleError as exc:
                envelope = BuildingEnvelopeBase.model_validate({
                    "building_name": stage1.building_program.building_name,
                    "total_floors": stage1.building_program.total_floors,
                    "overall_total_area": stage1.envelope.gross_area,
                    "floors": [
                        {
                            "floor_number": floor.floor_number,
                            "floor_function_tag": floor.floor_role,
                            "requested_rooms_list": [room.name for room in floor.rooms],
                        }
                        for floor in stage1.floor_programs
                    ],
                })
                floor_w, floor_h, floor_boundary = _derive_floor_boundary(envelope, opts)
                return self._stage1_failure_result(
                    stage1=stage1,
                    payload=exc.payload,
                    envelope=envelope,
                    floor_boundary=floor_boundary,
                    floor_w=floor_w,
                    floor_h=floor_h,
                )
            envelope = BuildingEnvelopeBase.model_validate({
                "building_name": allocation.building_name,
                "total_floors": allocation.total_floors,
                "overall_total_area": allocation.overall_total_area,
                "floors": [
                    {
                        "floor_number": floor.floor_number,
                        "floor_function_tag": floor.floor_function_tag,
                        "requested_rooms_list": [room.room_name for room in floor.rooms],
                    }
                    for floor in allocation.floors
                ],
            })
            floor_w, floor_h, floor_boundary = _derive_floor_boundary(envelope, opts)
            try:
                corridor_options = stage2_corridor_options_from_stage1(stage1)
                corridor_layout = str(corridor_options["corridor_layout"])
                corridor_width = normalize_corridor_width(
                    float(opts.corridor_width if opts.corridor_width is not None else corridor_options["target_width"]),
                    corridor_layout,
                )
                fixed_core_tube, core_metadata = core_tube_from_stage1_policy(stage1, floor_boundary)
                validate_stage1_core_context(stage1, core_metadata)
                validate_stage1_corridor_context(stage1, corridor_options)
            except Stage1ContextMismatchError as exc:
                payload = {
                    "result": "pipeline_failed",
                    "artifact_valid": False,
                    "stage": "stage1",
                    "failure_type": exc.failure_type,
                    "error": str(exc),
                    "metadata": exc.metadata,
                    "can_enter_geometry": False,
                }
                return self._stage1_failure_result(
                    stage1=stage1,
                    payload=payload,
                    envelope=envelope,
                    floor_boundary=floor_boundary,
                    floor_w=floor_w,
                    floor_h=floor_h,
                )
            core_area_ratio = float(stage1.core_context.core_area) / max(float(floor_boundary.area), 1e-6)
            budget = compute_building_area_budget(
                floor_boundary=floor_boundary,
                floors=allocation.floors,
                corridor_width=corridor_width,
                core_area_ratio=core_area_ratio,
                corridor_layout=corridor_layout,
                base_seed=opts.base_seed,
                fixed_core_tube=fixed_core_tube,
            )
            semantic_result = BudgetedBuildingSemanticResult(
                allocation=allocation,
                warnings=["Stage 1 program adapter used"],
                envelope=envelope,
                budget=budget,
                topology_snapshot=budget.topology_snapshot,
            )
        else:
            logger.info("[STAGE] Start Envelope Pass")
            envelope = await generate_building_envelope(request, on_llm_output=opts.on_llm_output)
            floor_w, floor_h, floor_boundary = _derive_floor_boundary(envelope, opts)
            default_cw, default_core = _pick_corridor_width_and_core_ratio(floor_w * floor_h)
            corridor_layout = str(opts.corridor_layout or "organic")
            corridor_width = normalize_corridor_width(
                float(opts.corridor_width if opts.corridor_width is not None else default_cw),
                corridor_layout,
            )
            core_area_ratio = float(opts.core_area_ratio if opts.core_area_ratio is not None else default_core)
            fixed_core_tube = self._create_fixed_core(floor_boundary, core_area_ratio, opts.core_placement)
            logger.info(
                "[BUDGET] Physical envelope | floors=%s | floor=%.2fx%.2f | corridor_mode=%s | corridor_width=%.2f | core_ratio=%.3f",
                envelope.total_floors,
                floor_w,
                floor_h,
                corridor_layout,
                corridor_width,
                core_area_ratio,
            )

            logger.info("[STAGE] Start Budgeted Allocation Pass")
            semantic_result = await generate_budgeted_building_semantics(
                request,
                floor_boundary=floor_boundary,
                corridor_width=corridor_width,
                core_area_ratio=core_area_ratio,
                corridor_layout=corridor_layout,
                base_seed=opts.base_seed,
                fixed_core_tube=fixed_core_tube,
                envelope=envelope,
                on_llm_output=opts.on_llm_output,
            )
        for fid, floor_budget in semantic_result.budget.floors.items():
            floor_obj = _find_floor(semantic_result.allocation, int(str(fid).lstrip("F") or 1))
            room_sum = (
                float(sum(float(r.target_area) for r in floor_obj.rooms))
                if floor_obj is not None else 0.0
            )
            logger.info(
                "[BUDGET] Floor=%s | island_area=%.2f | room_sum=%.2f | min=%.2f | max=%.2f | recommended=%.2f",
                fid,
                float(floor_budget.total_island_area),
                room_sum,
                float(floor_budget.room_sum_min),
                float(floor_budget.room_sum_max),
                float(floor_budget.room_sum_recommended),
            )

        try:
            logger.info("[STAGE] Start Geometry Generation")
            building_result, building_dict = self._run_geometry(
                allocation=semantic_result.allocation,
                topology_snapshot=semantic_result.topology_snapshot,
                floor_boundary=floor_boundary,
                corridor_width=corridor_width,
                topology_mode=opts.topology_mode,
                core_area_ratio=core_area_ratio,
                corridor_layout=corridor_layout,
                base_seed=opts.base_seed,
                config=opts.config,
                fixed_core_tube=fixed_core_tube,
            )
            logger.info("[STAGE] Geometry Complete")
            return self._result(
                semantic_result=semantic_result,
                building_result=building_result,
                building_dict=building_dict,
                floor_boundary=floor_boundary,
                floor_w=floor_w,
                floor_h=floor_h,
                corridor_width=corridor_width,
                core_area_ratio=core_area_ratio,
                warnings=list(semantic_result.warnings),
            )
        except (LayoutCoverageError, LayoutTopologyError, LayoutGeometryInvariantError) as exc:
            if getattr(exc, "semantic_repair_allowed", True) is False:
                logger.error(
                    "[GEOMETRY] Geometry failed without semantic repair | floor=%s | stage=%s | metadata=%s",
                    getattr(exc, "floor_id", None),
                    getattr(exc, "stage", None),
                    getattr(exc, "metadata", {}),
                )
                return self._typed_geometry_failure_result(
                    exc=exc,
                    semantic_result=semantic_result,
                    floor_boundary=floor_boundary,
                    floor_w=floor_w,
                    floor_h=floor_h,
                    corridor_width=corridor_width,
                    core_area_ratio=core_area_ratio,
                    topology_mode=opts.topology_mode,
                )
            failure = self._failure_report(exc, semantic_result.allocation, semantic_result.budget)
            logger.error(
                "[COVERAGE] Geometry failed | kind=%s | floor=%s | type=%s | max_gap=%.2f | metadata=%s",
                failure.failure_kind,
                failure.floor_id,
                failure.failure_type,
                float(failure.max_gap_area or 0.0),
                failure.metadata,
                exc_info=True,
            )
            failed_floor_number = int(failure.floor_id[1:]) if failure.floor_id.upper().startswith("F") else 1
            failed_floor = _find_floor(semantic_result.allocation, failed_floor_number)
            repair_prompt = repair_building_allocation_with_budget(
                failure=failure,
                budget=semantic_result.budget,
                current_rooms=list(getattr(failed_floor, "rooms", []) or []),
            )
            logger.warning(
                "[STAGE] Enter Semantic Repair | floor=%s | failure_kind=%s | failure_type=%s",
                failure.floor_id,
                failure.failure_kind,
                failure.failure_type,
            )
            log_multiline_debug(logger, "[LLM]", "Sending semantic repair prompt.", repair_prompt, "REPAIR_PROMPT")

            repaired_semantics = await generate_budgeted_building_semantics(
                request,
                floor_boundary=floor_boundary,
                corridor_width=corridor_width,
                core_area_ratio=core_area_ratio,
                corridor_layout=corridor_layout,
                base_seed=opts.base_seed,
                fixed_core_tube=fixed_core_tube,
                envelope=envelope,
                budget=semantic_result.budget,
                repair_instruction=repair_prompt,
                on_llm_output=opts.on_llm_output,
            )
            merged_allocation = _clone_with_repaired_floor(
                semantic_result.allocation,
                repaired_semantics.allocation,
                failed_floor_number,
            )
            validate_allocation_adjacency_ids(merged_allocation)
            validate_allocation_against_budget(merged_allocation, semantic_result.budget)

            logger.info("[STAGE] Start Geometry Generation After Semantic Repair")
            building_result, building_dict = self._run_geometry(
                allocation=merged_allocation,
                topology_snapshot=semantic_result.topology_snapshot,
                floor_boundary=floor_boundary,
                corridor_width=corridor_width,
                topology_mode=opts.topology_mode,
                core_area_ratio=core_area_ratio,
                corridor_layout=corridor_layout,
                base_seed=opts.base_seed,
                config=opts.config,
                fixed_core_tube=fixed_core_tube,
            )
            logger.info("[STAGE] Geometry Complete After Semantic Repair")
            repaired_semantics.allocation = merged_allocation
            repaired_semantics.budget = semantic_result.budget
            repaired_semantics.topology_snapshot = semantic_result.topology_snapshot
            return self._result(
                semantic_result=repaired_semantics,
                building_result=building_result,
                building_dict=building_dict,
                floor_boundary=floor_boundary,
                floor_w=floor_w,
                floor_h=floor_h,
                corridor_width=corridor_width,
                core_area_ratio=core_area_ratio,
                warnings=list(semantic_result.warnings) + ["Semantic repair applied"] + list(repaired_semantics.warnings),
            )

    def _create_fixed_core(self, floor_boundary: Polygon, core_area_ratio: float, core_placement: str) -> Optional[CoreTube]:
        try:
            return CoreTube.create_for_floor(
                floor_bounds=floor_boundary.bounds,
                area_ratio=float(core_area_ratio),
                position=core_placement,
            )
        except Exception as exc:
            logger.warning("core override failed: %s: %s", type(exc).__name__, exc)
            return None

    def _run_geometry(
        self,
        *,
        allocation: BuildingAllocation,
        topology_snapshot: Optional[TopologySnapshot],
        floor_boundary: Polygon,
        corridor_width: float,
        topology_mode: str,
        core_area_ratio: float,
        corridor_layout: str,
        base_seed: Optional[int],
        config: Optional[SolverConfig],
        fixed_core_tube: Optional[CoreTube],
    ) -> Tuple[BuildingResult, dict]:
        orchestrator = BuildingOrchestrator(
            floor_boundary=floor_boundary,
            corridor_width=float(corridor_width),
            core_area_ratio=float(core_area_ratio),
            corridor_layout=corridor_layout,
            topology_mode=topology_mode,
            base_seed=base_seed,
            config=config,
            topology_snapshot=topology_snapshot,
        )
        if fixed_core_tube is not None:
            orchestrator._shared_core_tube = fixed_core_tube
        building_result = orchestrator.generate(allocation, topology_snapshot=topology_snapshot)
        building_dict = building_result_to_dict(building_result, floor_boundary)
        return building_result, building_dict

    def _failure_report(
        self,
        exc: Exception,
        allocation: BuildingAllocation,
        budget: BuildingAreaBudget,
    ) -> LayoutFailureReport:
        floor_number = getattr(exc, "floor_number", None)
        if floor_number is None:
            floor_number = int(allocation.floors[0].floor_number) if allocation.floors else 1
        floor_id = getattr(exc, "floor_id", None) or _floor_id_for_number(int(floor_number))
        failed_floor = _find_floor(allocation, int(floor_number))
        budget_floor = budget.floors.get(str(floor_id))
        room_target_sum = (
            float(sum(float(r.target_area) for r in failed_floor.rooms))
            if failed_floor is not None else 0.0
        )
        return LayoutFailureReport(
            floor_id=str(floor_id),
            failure_type=type(exc).__name__,
            message=str(exc),
            failure_kind=self._failure_kind(exc),
            room_ids=[str(r.room_id) for r in getattr(failed_floor, "rooms", [])],
            room_target_sum=room_target_sum,
            island_area=float(getattr(budget_floor, "total_island_area", 0.0) or 0.0),
            max_gap_area=float(getattr(exc, "max_gap_area", 0.0) or 0.0),
            metadata=dict(getattr(exc, "metadata", {}) or {}),
        )

    def _failure_kind(self, exc: Exception) -> str:
        if isinstance(exc, LayoutAssignmentError):
            return "assignment"
        if isinstance(exc, LayoutCoverageError):
            return "coverage"
        if isinstance(exc, LayoutTopologyError):
            metadata = dict(getattr(exc, "metadata", {}) or {})
            if str(metadata.get("failure_kind", "")).lower() == "reachability":
                return "reachability"
            return "topology"
        if isinstance(exc, SemanticInvalidError):
            text = str(exc).lower()
            if "infeasible" in text:
                return "infeasible"
            if any(tok in text for tok in ("capacity", "exceed", "too small", "area")):
                return "capacity"
        text = type(exc).__name__.lower()
        if "infeasible" in text:
            return "infeasible"
        return "unknown"

    def _allocation_with_synthetic_rooms(
        self,
        allocation: BuildingAllocation,
        building_result: BuildingResult,
    ) -> BuildingAllocation:
        merged = copy.deepcopy(allocation)
        floors_by_id = {f"F{int(f.floor_number)}": f for f in merged.floors}
        layouts_obj = getattr(building_result, "floor_layouts", {}) or {}
        if isinstance(layouts_obj, dict):
            layout_items = list(layouts_obj.items())
        else:
            layout_items = [(None, layout) for layout in layouts_obj]
        for floor_key, layout in layout_items:
            records = list(getattr(layout, "synthetic_rooms", []) or [])
            if not records:
                continue
            floor_id = str(floor_key or getattr(layout, "floor_id", "") or "")
            if not floor_id:
                floor_number = getattr(layout, "floor_number", None)
                floor_id = _floor_id_for_number(int(floor_number or 1))
            floor = floors_by_id.get(floor_id)
            if floor is None:
                continue
            existing_ids = {str(r.room_id) for r in getattr(floor, "rooms", []) or []}
            for record in records:
                if not isinstance(record, dict):
                    continue
                rid = str(record.get("room_id") or "")
                if not rid or rid in existing_ids:
                    continue
                area = max(0.1, float(record.get("target_area", 0.0) or 0.0))
                floor.rooms.append(RoomAllocation(
                    room_id=rid,
                    room_name=str(record.get("room_name") or "Auto Storage"),
                    room_type="storage",
                    target_area=area,
                    zone="service",
                    needs_window=False,
                    min_width=max(0.8, min(1.5, area ** 0.5)),
                    aspect_ratio_range=[0.2, 5.0],
                    adjacency_required=[],
                    adjacency_preferred=[],
                    adjacency_forbidden=[],
                    size_hint="small",
                    weight=1,
                ))
                existing_ids.add(rid)
        return merged

    def _result(
        self,
        *,
        semantic_result: BudgetedBuildingSemanticResult,
        building_result: BuildingResult,
        building_dict: dict,
        floor_boundary: Polygon,
        floor_w: float,
        floor_h: float,
        corridor_width: float,
        core_area_ratio: float,
        warnings: List[str],
    ) -> BuildingPipelineResult:
        allocation = self._allocation_with_synthetic_rooms(
            semantic_result.allocation,
            building_result,
        )
        return BuildingPipelineResult(
            envelope=semantic_result.envelope,
            budget=semantic_result.budget,
            topology_snapshot=semantic_result.topology_snapshot,
            allocation=allocation,
            building_result=building_result,
            building_dict=building_dict,
            floor_boundary=floor_boundary,
            floor_width=float(floor_w),
            floor_height=float(floor_h),
            corridor_width=float(corridor_width),
            core_area_ratio=float(core_area_ratio),
            warnings=warnings,
        )

    def _stage1_failure_result(
        self,
        *,
        stage1: Stage1Result,
        payload: dict,
        envelope: BuildingEnvelopeBase,
        floor_boundary: Polygon,
        floor_w: float,
        floor_h: float,
    ) -> BuildingPipelineResult:
        building_dict = {
            "result": "pipeline_failed",
            "artifact_valid": False,
            "failure": payload,
            "warnings": ["Stage 1 program blocked geometry"],
        }
        logger.info(
            "[ROUTE] Stage 1 failure routed | result=pipeline_failed | failure_type=%s | can_enter_geometry=%s",
            payload.get("failure_type"),
            payload.get("can_enter_geometry"),
        )
        return BuildingPipelineResult(
            envelope=envelope,
            budget=BuildingAreaBudget(),
            topology_snapshot=None,
            allocation=None,
            building_result=None,
            building_dict=building_dict,
            floor_boundary=floor_boundary,
            floor_width=float(floor_w),
            floor_height=float(floor_h),
            corridor_width=float(stage1.corridor_context.target_width),
            core_area_ratio=float(stage1.core_context.core_area) / max(float(floor_boundary.area), 1e-6),
            warnings=["Stage 1 program blocked geometry"],
            success=False,
            artifact_valid=False,
            failure=payload,
        )

    def _typed_geometry_failure_payload(
        self,
        *,
        exc: Exception,
        topology_mode: str,
    ) -> dict:
        metadata = dict(getattr(exc, "metadata", {}) or {})
        floor_id = str(getattr(exc, "floor_id", None) or metadata.get("floor_id") or "")
        payload = {
            "result": "pipeline_failed",
            "artifact_valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stage": str(getattr(exc, "stage", None) or metadata.get("stage") or ""),
            "floor_id": floor_id,
            "topology_mode": str(metadata.get("topology_mode") or topology_mode or ""),
            "core_contract_id": metadata.get("core_contract_id"),
            "core_union_hash": metadata.get("core_union_hash"),
            "offending_objects": metadata.get("offending_objects", []),
            "overlap_area": metadata.get("overlap_area"),
            "overlap_bbox": metadata.get("overlap_bbox"),
            "threshold": metadata.get("threshold"),
            "semantic_repair_allowed": False,
            "metadata": metadata,
        }
        return payload

    def _typed_geometry_failure_result(
        self,
        *,
        exc: Exception,
        semantic_result: BudgetedBuildingSemanticResult,
        floor_boundary: Polygon,
        floor_w: float,
        floor_h: float,
        corridor_width: float,
        core_area_ratio: float,
        topology_mode: str,
    ) -> BuildingPipelineResult:
        payload = self._typed_geometry_failure_payload(exc=exc, topology_mode=topology_mode)
        building_dict = {
            "result": "pipeline_failed",
            "artifact_valid": False,
            "failure": payload,
            "warnings": list(semantic_result.warnings or []),
        }
        logger.info(
            "[ROUTE] Geometry failure routed | result=pipeline_failed | floor=%s | stage=%s | semantic_repair_allowed=False",
            payload.get("floor_id"),
            payload.get("stage"),
        )
        return BuildingPipelineResult(
            envelope=semantic_result.envelope,
            budget=semantic_result.budget,
            topology_snapshot=semantic_result.topology_snapshot,
            allocation=semantic_result.allocation,
            building_result=None,
            building_dict=building_dict,
            floor_boundary=floor_boundary,
            floor_width=float(floor_w),
            floor_height=float(floor_h),
            corridor_width=float(corridor_width),
            core_area_ratio=float(core_area_ratio),
            warnings=list(semantic_result.warnings or []),
            success=False,
            artifact_valid=False,
            failure=payload,
        )















