from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence


logger = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _is_explicit_room(room: Any) -> bool:
    if bool(getattr(room, "is_dummy", False)):
        return False
    if bool(getattr(room, "generated", False)):
        return False
    if getattr(room, "semantic_room", True) is False:
        return False
    rtype = str(getattr(room, "room_type", "") or "").lower()
    return rtype not in {"dummy", "void", "utility_dummy", "service_niche"}


@dataclass
class CoverageDebtPolicy:
    coverage_slack_ratio: float = 0.015
    coverage_slack_min: float = 0.5
    coverage_slack_max: float = 2.0
    explicit_soft_growth_ratio: float = 1.05
    area_tolerance_min_default: float = 0.85
    area_tolerance_max_default: float = 1.15
    unexpected_residual_tolerance: float = 1.5
    max_corridor_sponge_area_per_island: float = 6.0
    max_corridor_sponge_ratio_per_island: float = 0.15
    compact_filler_enabled: bool = False
    compact_filler_max_sum: float = 0.0


@dataclass
class CoverageDebtPlan:
    plan_id: str
    policy_version: str
    created_at_stage: str
    floor_id: str
    island_id: str
    topology_mode: str
    island_area: float
    assigned_room_ids: List[str]
    coverage_slack_effective: float
    raw_coverage_min: float
    assigned_explicit_min_sum: float
    assigned_explicit_target_sum: float
    assigned_explicit_soft_sum: float
    assigned_explicit_max_sum: float
    solver_feasible_upper_bound: float
    effective_solver_coverage_min: float
    raw_gap_at_target: float
    preferred_gap: float
    hard_gap: float
    planned_residual_area: float
    planned_corridor_sponge_area: float
    planned_edge_sliver_absorb_area: float
    planned_neighbor_absorb_area: float
    planned_compact_filler_area: float
    unexpected_residual_tolerance: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_coverage_debt_plan(
    *,
    floor_id: str,
    island_id: str,
    topology_mode: str,
    island_area: float,
    assigned_rooms: Sequence[Any],
    policy: CoverageDebtPolicy | None = None,
) -> CoverageDebtPlan:
    policy = policy or CoverageDebtPolicy()
    assigned_room_ids = [str(getattr(r, "room_id", "") or getattr(r, "id", "")) for r in assigned_rooms]
    explicit_rooms = [r for r in assigned_rooms if _is_explicit_room(r)]
    target_sum = float(sum(max(0.0, float(getattr(r, "target_area", 0.0) or 0.0)) for r in explicit_rooms))
    min_sum = float(
        sum(
            max(0.0, float(getattr(r, "target_area", 0.0) or 0.0))
            * float(policy.area_tolerance_min_default)
            for r in explicit_rooms
        )
    )
    max_sum = float(
        sum(
            max(0.0, float(getattr(r, "target_area", 0.0) or 0.0))
            * float(policy.area_tolerance_max_default)
            for r in explicit_rooms
        )
    )
    island_area_f = max(0.0, float(island_area))
    slack = _clamp(
        island_area_f * float(policy.coverage_slack_ratio),
        float(policy.coverage_slack_min),
        float(policy.coverage_slack_max),
    )
    raw_min = max(0.0, island_area_f - slack)
    soft_sum = target_sum * float(policy.explicit_soft_growth_ratio)
    compact_filler_max = float(policy.compact_filler_max_sum if policy.compact_filler_enabled else 0.0)
    upper = max_sum + compact_filler_max
    effective = min(raw_min, soft_sum + compact_filler_max)
    effective = min(effective, upper)
    effective = max(effective, min_sum)
    planned_residual = max(0.0, raw_min - effective)
    planned_corridor = min(
        planned_residual,
        float(policy.max_corridor_sponge_area_per_island),
        island_area_f * float(policy.max_corridor_sponge_ratio_per_island),
    )
    plan = CoverageDebtPlan(
        plan_id=f"{floor_id}:{island_id}:stage2a1_v1",
        policy_version="stage2a1_v1",
        created_at_stage="post_grid_assignment_pre_solve",
        floor_id=str(floor_id),
        island_id=str(island_id),
        topology_mode=str(topology_mode or ""),
        island_area=island_area_f,
        assigned_room_ids=assigned_room_ids,
        coverage_slack_effective=float(slack),
        raw_coverage_min=float(raw_min),
        assigned_explicit_min_sum=float(min_sum),
        assigned_explicit_target_sum=float(target_sum),
        assigned_explicit_soft_sum=float(soft_sum),
        assigned_explicit_max_sum=float(max_sum),
        solver_feasible_upper_bound=float(upper),
        effective_solver_coverage_min=float(effective),
        raw_gap_at_target=max(0.0, raw_min - target_sum),
        preferred_gap=max(0.0, raw_min - soft_sum),
        hard_gap=max(0.0, raw_min - max_sum),
        planned_residual_area=float(planned_residual),
        planned_corridor_sponge_area=float(planned_corridor),
        planned_edge_sliver_absorb_area=max(0.0, float(planned_residual) - float(planned_corridor)),
        planned_neighbor_absorb_area=0.0,
        planned_compact_filler_area=0.0,
        unexpected_residual_tolerance=float(policy.unexpected_residual_tolerance),
        diagnostics={
            "explicit_room_ids": [
                str(getattr(r, "room_id", "") or getattr(r, "id", "")) for r in explicit_rooms
            ],
            "compact_filler_enabled": bool(policy.compact_filler_enabled),
            "compact_filler_max_sum": compact_filler_max,
        },
    )
    logger.info(
        "[DEBT] Coverage debt planned | plan_id=%s | policy=%s | floor=%s | island=%s | "
        "assigned_room_ids=%s | island_area=%.2f | raw_min=%.2f | explicit_min=%.2f | "
        "explicit_target=%.2f | explicit_soft=%.2f | explicit_max=%.2f | "
        "feasible_upper=%.2f | effective_min=%.2f | planned_residual=%.2f | "
        "planned_corridor_sponge=%.2f",
        plan.plan_id,
        plan.policy_version,
        plan.floor_id,
        plan.island_id,
        plan.assigned_room_ids,
        plan.island_area,
        plan.raw_coverage_min,
        plan.assigned_explicit_min_sum,
        plan.assigned_explicit_target_sum,
        plan.assigned_explicit_soft_sum,
        plan.assigned_explicit_max_sum,
        plan.solver_feasible_upper_bound,
        plan.effective_solver_coverage_min,
        plan.planned_residual_area,
        plan.planned_corridor_sponge_area,
    )
    return plan
