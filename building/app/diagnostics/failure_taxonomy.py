from __future__ import annotations

from dataclasses import dataclass


STAGE1_FAILURES = {
    "program_semantic_missing",
    "program_capacity_shortfall",
    "program_infeasible",
    "target_area_exceeds_usable_area",
    "min_area_exceeds_usable_area",
    "core_policy_infeasible",
}

STAGE2_FAILURES = {
    "territory_semantic_source_missing",
    "island_capacity_shortfall",
    "free_space_fragmented",
    "core_overlap",
    "connectivity_failure",
    "coverage_debt",
}

STAGE3_FAILURES = {
    "door_unreachable",
    "window_shortage",
    "renderer_failure",
}


@dataclass(frozen=True)
class FailureRoute:
    failure_type: str
    stage_owner: str


def route_failure(failure_type: str) -> FailureRoute:
    key = str(failure_type or "").strip()
    if key in STAGE1_FAILURES:
        return FailureRoute(failure_type=key, stage_owner="stage1")
    if key in STAGE2_FAILURES:
        return FailureRoute(failure_type=key, stage_owner="stage2")
    if key in STAGE3_FAILURES:
        return FailureRoute(failure_type=key, stage_owner="stage3")
    return FailureRoute(failure_type=key or "unknown", stage_owner="unknown")

