"""Diagnostic-only capacity conflict analysis for topology assignment.

This module deliberately does not decide assignments for the production path.
It summarizes why the topology-assignment CP-SAT proposal failed and, when
enabled, runs bounded diagnostic-only relaxation probes.
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .topology_feasibility import (
    ClusterIslandFeasibility,
    ClusterMetrics,
    IslandMetrics,
    TopologyFeasibilityReport,
    TopologyVariant,
)


EPSILON = 1e-6
SCALE = 1000
SUMMARY_MAX = 500
HINT_MAX = 120
KNOWN_REJECTION_REASONS = [
    "cluster_min_area_over_capacity",
    "needs_window_no_facade_slot",
    "needs_corridor_access_no_access_slot",
    "large_room_slot_shortfall",
    "largest_room_area_exceeds_largest_rect",
    "largest_room_dimensions_exceed_largest_rect",
    "invalid_island",
    "core_overlap",
]
CLASSIFICATION_PRIORITY = [
    "all_variants_invalid",
    "cluster_without_feasible_island",
    "global_area_capacity_shortfall",
    "large_slot_shortfall",
    "access_slot_shortfall",
    "window_slot_shortfall",
    "island_area_capacity_shortfall",
    "solver_infeasible_after_apparent_capacity_ok",
    "unknown_capacity_conflict",
]


def _r(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _bounded(
    items: Sequence[Dict[str, Any]],
    *,
    max_items: int,
    sort_key,
) -> Dict[str, Any]:
    ordered = sorted(list(items or []), key=sort_key)
    shown = ordered[: int(max_items)]
    return {
        "items": shown,
        "truncated": len(ordered) > int(max_items),
        "shown_count": len(shown),
        "total_available_count": len(ordered),
    }


def _clip_text(text: str, limit: int) -> Tuple[str, bool]:
    value = str(text or "")
    if len(value) <= int(limit):
        return value, False
    return value[: max(0, int(limit) - 3)] + "...", True


def _valid_variants(report: TopologyFeasibilityReport) -> List[TopologyVariant]:
    return [v for v in list(report.variants or []) if bool(v.valid) and list(v.island_metrics or [])]


def _all_rows(report: TopologyFeasibilityReport) -> List[ClusterIslandFeasibility]:
    return [
        row
        for variant in list(report.variants or [])
        for row in list(variant.feasibility_matrix or [])
    ]


def _global_demand(clusters: Sequence[ClusterMetrics]) -> Dict[str, Any]:
    return {
        "target_area_sum": _r(sum(float(c.target_area_sum) for c in clusters or [])),
        "min_area_sum": _r(sum(float(c.min_area_sum) for c in clusters or [])),
        "max_area_sum": _r(sum(float(c.max_area_sum) for c in clusters or [])),
        "window_slot_demand": int(sum(int(c.needs_window_count) for c in clusters or [])),
        "access_slot_demand": int(sum(int(c.needs_corridor_access_count) for c in clusters or [])),
        "large_room_demand": int(sum(int(c.large_room_count) for c in clusters or [])),
        "medium_room_demand": int(sum(int(c.medium_room_count) for c in clusters or [])),
        "small_room_demand": int(sum(int(c.small_room_count) for c in clusters or [])),
    }


def _variant_capacity_summary(
    variants: Sequence[TopologyVariant],
    demand: Mapping[str, Any],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for variant in variants or []:
        islands = list(variant.island_metrics or [])
        effective = sum(float(i.effective_capacity_area) for i in islands)
        actual = sum(float(i.area) for i in islands)
        window = sum(int(i.window_slot_count) for i in islands)
        access = sum(int(i.corridor_door_slot_count) for i in islands)
        large = sum(int(i.slot_count_large) for i in islands)
        medium = sum(int(i.slot_count_medium) for i in islands)
        blocking: List[str] = []
        if float(demand.get("target_area_sum", 0.0) or 0.0) > effective + EPSILON:
            blocking.append("global_area_capacity_shortfall")
        if int(demand.get("window_slot_demand", 0) or 0) > window:
            blocking.append("window_slot_shortfall")
        if int(demand.get("access_slot_demand", 0) or 0) > access:
            blocking.append("access_slot_shortfall")
        if int(demand.get("large_room_demand", 0) or 0) > large:
            blocking.append("large_slot_shortfall")
        entries.append(
            {
                "variant_id": str(variant.variant_id),
                "seed": int(variant.seed),
                "valid": bool(variant.valid),
                "island_count": len(islands),
                "total_effective_capacity": _r(effective),
                "total_actual_island_area": _r(actual),
                "target_area_demand": _r(demand.get("target_area_sum", 0.0)),
                "effective_capacity_margin": _r(effective - float(demand.get("target_area_sum", 0.0) or 0.0)),
                "actual_area_margin": _r(actual - float(demand.get("target_area_sum", 0.0) or 0.0)),
                "total_window_slots": int(window),
                "total_access_slots": int(access),
                "total_large_slots": int(large),
                "total_medium_slots": int(medium),
                "window_slot_demand": int(demand.get("window_slot_demand", 0) or 0),
                "access_slot_demand": int(demand.get("access_slot_demand", 0) or 0),
                "large_room_demand": int(demand.get("large_room_demand", 0) or 0),
                "likely_blocking_constraints": blocking,
            }
        )
    return _bounded(entries, max_items=10, sort_key=lambda x: (int(x.get("seed", 0)), str(x.get("variant_id", ""))))


def _cluster_room_types(cluster: ClusterMetrics) -> List[str]:
    room_types: List[str] = []
    for room_id in list(cluster.room_ids or []):
        # ClusterMetrics does not retain every room type, so keep the largest
        # type as the most useful stable signal and preserve room ids.
        if cluster.largest_room_type and room_id == list(cluster.room_ids or [room_id])[0]:
            room_types.append(str(cluster.largest_room_type))
    return room_types


def _clusters_without_feasible_island(
    report: TopologyFeasibilityReport,
    valid_variants: Sequence[TopologyVariant],
) -> Dict[str, Any]:
    rows = _all_rows(report)
    rows_by_cluster: Dict[str, List[ClusterIslandFeasibility]] = {}
    for row in rows:
        rows_by_cluster.setdefault(str(row.cluster_id), []).append(row)
    valid_ids = {str(v.variant_id) for v in valid_variants}
    entries: List[Dict[str, Any]] = []
    for cluster in list(report.cluster_metrics or []):
        candidate_rows = [r for r in rows_by_cluster.get(str(cluster.cluster_id), []) if str(r.variant_id) in valid_ids]
        feasible_rows = [r for r in candidate_rows if bool(r.hard_feasible)]
        if feasible_rows:
            continue
        best = _best_candidate(candidate_rows)
        reason_counts: Dict[str, int] = {}
        for row in candidate_rows:
            for reason in list(row.rejection_reasons or ["unknown"]):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        entry = {
            "cluster_id": str(cluster.cluster_id),
            "room_ids": [str(x) for x in list(cluster.room_ids or [])],
            "room_types": _cluster_room_types(cluster),
            "target_area_sum": _r(cluster.target_area_sum),
            "min_area_sum": _r(cluster.min_area_sum),
            "max_area_sum": _r(cluster.max_area_sum),
            "largest_room_id": str(list(cluster.room_ids or [""])[0] if cluster.room_ids else ""),
            "largest_room_type": str(cluster.largest_room_type),
            "largest_room_area": _r(cluster.largest_room_area),
            "largest_room_min_width": _r(cluster.largest_room_min_width_estimate),
            "largest_room_min_depth": _r(cluster.largest_room_min_depth_estimate),
            "valid_variant_count_checked": len(valid_ids),
            "feasible_variant_count": len({str(r.variant_id) for r in feasible_rows}),
            "best_candidate_available": best is not None,
            "best_candidate_variant_id": str(best.variant_id) if best is not None else "",
            "best_candidate_island_id": str(best.island_id) if best is not None else "",
            "best_feasibility_score": _r(best.feasibility_score) if best is not None else 0.0,
            "failed_constraints_summary": dict(sorted(reason_counts.items())),
        }
        entries.append(entry)
    return {
        "scope": "across_all_valid_variants",
        **_bounded(
            entries,
            max_items=10,
            sort_key=lambda x: (-float(x.get("target_area_sum", 0.0)), str(x.get("cluster_id", ""))),
        ),
    }


def _best_candidate(rows: Sequence[ClusterIslandFeasibility]) -> Optional[ClusterIslandFeasibility]:
    if not rows:
        return None
    return sorted(
        list(rows),
        key=lambda r: (
            -float(r.feasibility_score),
            -float(r.capacity_margin),
            str(r.variant_id),
            str(r.island_id),
        ),
    )[0]


def _rejection_counts(report: TopologyFeasibilityReport) -> Dict[str, Any]:
    counts = {key: 0 for key in KNOWN_REJECTION_REASONS}
    unknown_keys: Dict[str, int] = {}
    hard_false = 0
    for row in _all_rows(report):
        if not bool(row.hard_feasible):
            hard_false += 1
            reasons = list(row.rejection_reasons or ["unknown"])
            for reason in reasons:
                key = str(reason)
                if key in counts:
                    counts[key] += 1
                else:
                    unknown_keys[key] = unknown_keys.get(key, 0) + 1
    return {
        "reason_vocab_version": "cluster_island_feasibility_v1",
        "reason_source": "coarse",
        "hard_feasible_false": int(hard_false),
        "known_reason_keys": list(KNOWN_REJECTION_REASONS),
        "unknown_reason_keys": sorted(unknown_keys),
        **counts,
        "unknown": int(sum(unknown_keys.values())),
    }


def _heuristic_assignment_map(heuristic_assignments: Optional[Mapping[str, Any]], primary_variant_id: str) -> Dict[str, Dict[str, str]]:
    room_to_island: Dict[str, str] = {}
    for island_id, assignment in dict(heuristic_assignments or {}).items():
        for room in list(getattr(assignment, "rooms", []) or []):
            room_to_island[str(getattr(room, "room_id", room))] = str(island_id)
    return {
        room_id: {"variant_id": str(primary_variant_id), "island_id": island_id}
        for room_id, island_id in sorted(room_to_island.items())
    }


def _diagnostic_island_loads(
    report: TopologyFeasibilityReport,
    heuristic_assignments: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    valid_variants = _valid_variants(report)
    primary = next((v for v in valid_variants if v.variant_id == report.primary_variant_id), None)
    if primary is None or not heuristic_assignments:
        return {"load_source": "none", "overloaded_islands": _bounded([], max_items=10, sort_key=lambda x: 0)}
    island_by_id = {str(i.island_id): i for i in list(primary.island_metrics or [])}
    cluster_by_room: Dict[str, ClusterMetrics] = {}
    for cluster in list(report.cluster_metrics or []):
        for room_id in list(cluster.room_ids or []):
            cluster_by_room[str(room_id)] = cluster
    loads: Dict[str, Dict[str, Any]] = {}
    for island_id, assignment in dict(heuristic_assignments or {}).items():
        island = island_by_id.get(str(island_id))
        if island is None:
            continue
        data = loads.setdefault(
            str(island_id),
            {
                "variant_id": str(primary.variant_id),
                "island_id": str(island_id),
                "assigned_or_candidate_cluster_ids": [],
                "target_area_load": 0.0,
                "effective_capacity": _r(island.effective_capacity_area),
                "window_slot_load": 0,
                "window_slot_capacity": int(island.window_slot_count),
                "access_slot_load": 0,
                "access_slot_capacity": int(island.corridor_door_slot_count),
                "large_slot_load": 0,
                "large_slot_capacity": int(island.slot_count_large),
            },
        )
        seen_clusters = set(data["assigned_or_candidate_cluster_ids"])
        for room in list(getattr(assignment, "rooms", []) or []):
            cluster = cluster_by_room.get(str(getattr(room, "room_id", room)))
            if cluster is None or cluster.cluster_id in seen_clusters:
                continue
            seen_clusters.add(cluster.cluster_id)
            data["assigned_or_candidate_cluster_ids"].append(str(cluster.cluster_id))
            data["target_area_load"] += float(cluster.target_area_sum)
            data["window_slot_load"] += int(cluster.needs_window_count)
            data["access_slot_load"] += int(cluster.needs_corridor_access_count)
            data["large_slot_load"] += int(cluster.large_room_count)
    overloaded: List[Dict[str, Any]] = []
    for data in loads.values():
        area_shortfall = float(data["target_area_load"]) - float(data["effective_capacity"])
        blocking = []
        if area_shortfall > EPSILON:
            blocking.append("area_capacity")
        if int(data["window_slot_load"]) > int(data["window_slot_capacity"]):
            blocking.append("window_slots")
        if int(data["access_slot_load"]) > int(data["access_slot_capacity"]):
            blocking.append("access_slots")
        if int(data["large_slot_load"]) > int(data["large_slot_capacity"]):
            blocking.append("large_slots")
        if not blocking:
            continue
        out = dict(data)
        out["target_area_load"] = _r(out["target_area_load"])
        out["area_shortfall"] = _r(area_shortfall)
        out["blocking_constraints"] = blocking
        overloaded.append(out)
    return {
        "load_source": "seed0_heuristic_assignment",
        "overloaded_islands": _bounded(
            overloaded,
            max_items=10,
            sort_key=lambda x: (-float(x.get("area_shortfall", 0.0)), str(x.get("island_id", ""))),
        ),
    }


def classify_capacity_conflict(summary: Mapping[str, Any]) -> Dict[str, Any]:
    detected: List[str] = []
    variant_count = int(summary.get("variant_count", 0) or 0)
    valid_variant_count = int(summary.get("valid_variant_count", 0) or 0)
    if variant_count > 0 and valid_variant_count == 0:
        detected.append("all_variants_invalid")
    clusters_without = ((summary.get("clusters_without_feasible_island") or {}).get("items")) or []
    if clusters_without:
        detected.append("cluster_without_feasible_island")
    demand = summary.get("global_demand") or {}
    variants = ((summary.get("per_variant_capacity_summary") or {}).get("items")) or []
    if variants:
        max_effective = max(float(v.get("total_effective_capacity", 0.0) or 0.0) for v in variants)
        max_window = max(int(v.get("total_window_slots", 0) or 0) for v in variants)
        max_access = max(int(v.get("total_access_slots", 0) or 0) for v in variants)
        max_large = max(int(v.get("total_large_slots", 0) or 0) for v in variants)
        if float(demand.get("target_area_sum", 0.0) or 0.0) > max_effective + EPSILON:
            detected.append("global_area_capacity_shortfall")
        if int(demand.get("large_room_demand", 0) or 0) > max_large:
            detected.append("large_slot_shortfall")
        if int(demand.get("access_slot_demand", 0) or 0) > max_access:
            detected.append("access_slot_shortfall")
        if int(demand.get("window_slot_demand", 0) or 0) > max_window:
            detected.append("window_slot_shortfall")
    overloaded = (((summary.get("diagnostic_island_loads") or {}).get("overloaded_islands") or {}).get("items")) or []
    if any("area_capacity" in list(item.get("blocking_constraints") or []) for item in overloaded):
        detected.append("island_area_capacity_shortfall")
    if not detected and str(summary.get("failure_reason") or "") in {"capacity_conflict", "solver_infeasible"}:
        detected.append("solver_infeasible_after_apparent_capacity_ok")
    if not detected:
        detected.append("unknown_capacity_conflict")
    primary = next((kind for kind in CLASSIFICATION_PRIORITY if kind in detected), "unknown_capacity_conflict")
    confidence = "low"
    if primary in {"all_variants_invalid", "cluster_without_feasible_island", "global_area_capacity_shortfall"}:
        confidence = "high"
    elif primary in {"large_slot_shortfall", "access_slot_shortfall", "window_slot_shortfall", "island_area_capacity_shortfall"}:
        confidence = "medium"
    human, hint = _human_summary_and_hint(primary, detected)
    human, human_truncated = _clip_text(human, SUMMARY_MAX)
    hint, hint_truncated = _clip_text(hint, HINT_MAX)
    return {
        "primary_conflict_type": primary,
        "all_detected_conflict_types": [kind for kind in CLASSIFICATION_PRIORITY if kind in set(detected)],
        "classification_priority_used": list(CLASSIFICATION_PRIORITY),
        "confidence": confidence,
        "human_readable_summary": human,
        "human_readable_summary_truncated": human_truncated,
        "next_action_hint": hint,
        "next_action_hint_truncated": hint_truncated,
    }


def _human_summary_and_hint(primary: str, detected: Sequence[str]) -> Tuple[str, str]:
    hint_map = {
        "all_variants_invalid": "increase_topology_variant_diversity",
        "cluster_without_feasible_island": "inspect_cluster_without_feasible_island_or_consider_cluster_split",
        "global_area_capacity_shortfall": "program_may_be_infeasible_under_fixed_core",
        "large_slot_shortfall": "consider_relaxing_large_slot_constraint_or_improve_largest_rect_estimate",
        "access_slot_shortfall": "consider_relaxing_access_slot_constraint_or_generate_more_corridor_frontage",
        "window_slot_shortfall": "consider_relaxing_window_constraint_or_generate_more_facade_frontage",
        "island_area_capacity_shortfall": "inspect_cluster_to_island_load_distribution",
        "solver_infeasible_after_apparent_capacity_ok": "inspect_solver_model_or_integrality_conflict",
        "unknown_capacity_conflict": "inspect_topology_assignment_failure_metadata",
    }
    return (
        f"Topology assignment failed primarily due to {primary}; detected={','.join(detected)}.",
        hint_map.get(primary, "inspect_topology_assignment_failure_metadata"),
    )


def build_capacity_conflict_summary(
    report: TopologyFeasibilityReport,
    *,
    failure_reason: str,
    solver_status: str = "",
    heuristic_assignments: Optional[Mapping[str, Any]] = None,
    config: Optional[Any] = None,
    floor_id: str = "",
) -> Dict[str, Any]:
    # Read-only by construction: build new lists/dicts from report attributes.
    variants = list(report.variants or [])
    valid_variants = _valid_variants(report)
    clusters = list(report.cluster_metrics or [])
    demand = _global_demand(clusters)
    summary: Dict[str, Any] = {
        "diagnostic_version": "topology_capacity_conflict_v1",
        "source": "topology_assignment_solver_failure",
        "diagnostic_generated_at_stage": "topology_assignment",
        "failure_reason": str(failure_reason or ""),
        "solver_status": str(solver_status or ""),
        "ortools_status": str(solver_status or ""),
        "analysis_only": True,
        "used_for_solver_decision": False,
        "used_for_adoption": False,
        "floor_id": str(floor_id or ""),
        "variant_count": len(variants),
        "valid_variant_count": len(valid_variants),
        "cluster_count": len(clusters),
        "total_room_count": int(sum(len(list(c.room_ids or [])) for c in clusters)),
        "global_demand": demand,
        "per_variant_capacity_summary": _variant_capacity_summary(variants, demand),
        "clusters_without_feasible_island": _clusters_without_feasible_island(report, valid_variants),
        "diagnostic_island_loads": _diagnostic_island_loads(report, heuristic_assignments),
        "hard_constraint_rejection_counts": _rejection_counts(report),
    }
    summary["diagnosis"] = classify_capacity_conflict(summary)
    summary["relaxation_ladder"] = run_relaxation_ladder(
        report,
        failure_reason=failure_reason,
        solver_status=solver_status,
        config=config,
    )
    return summary


def _ladder_disabled() -> Dict[str, Any]:
    return {
        "enabled": False,
        "levels": [],
        "summary": {
            "not_run_reason": "diagnostics_disabled",
            "total_solve_time_ms": 0.0,
            "total_wall_time_ms": 0.0,
            "stopped_due_to_time_budget": False,
        },
    }


def run_relaxation_ladder(
    report: TopologyFeasibilityReport,
    *,
    failure_reason: str,
    solver_status: str = "",
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    if not bool(getattr(config, "enable_topology_assignment_relaxation_diagnostics", False)):
        return _ladder_disabled()
    if bool(getattr(config, "force_ortools_unavailable", False)):
        return {
            "enabled": True,
            "levels": [
                {
                    "level": "L0_original",
                    "description": "original constraints",
                    "diagnostic_status": "skipped",
                    "skip_reason": "ortools_unavailable",
                    "solve_time_ms": 0.0,
                }
            ],
            "summary": {
                "not_run_reason": "ortools_unavailable",
                "total_solve_time_ms": 0.0,
                "total_wall_time_ms": 0.0,
                "stopped_due_to_time_budget": False,
                "still_infeasible_after_all_levels": True,
            },
        }
    try:
        from ortools.sat.python import cp_model  # type: ignore[import-not-found]
    except Exception:
        return {
            "enabled": True,
            "levels": [
                {
                    "level": "L0_original",
                    "description": "original constraints",
                    "diagnostic_status": "skipped",
                    "skip_reason": "ortools_unavailable",
                    "solve_time_ms": 0.0,
                }
            ],
            "summary": {
                "not_run_reason": "ortools_unavailable",
                "total_solve_time_ms": 0.0,
                "total_wall_time_ms": 0.0,
                "stopped_due_to_time_budget": False,
                "still_infeasible_after_all_levels": True,
            },
        }
    levels = [
        ("L0_original", "original constraints", 1.0, False, False, False),
        ("L1_area_slack_1_10", "area capacity slack 1.10", 1.10, False, False, False),
        ("L2_area_slack_1_15", "area capacity slack 1.15", 1.15, False, False, False),
        ("L3_soft_window_access", "window/access constraints diagnostic-soft", 1.0, True, True, False),
        ("L4_soft_large_slots", "large slot capacity diagnostic-soft", 1.0, False, False, True),
        ("L5_area_1_15_soft_slots", "area 1.15 plus diagnostic-soft slots", 1.15, True, True, True),
    ][: int(getattr(config, "topology_assignment_relaxation_max_levels", 6) or 6)]
    total_budget = float(getattr(config, "topology_assignment_relaxation_total_time_limit_seconds", 3.0) or 3.0)
    per_level = float(getattr(config, "topology_assignment_relaxation_time_limit_seconds", 0.5) or 0.5)
    workers = int(getattr(config, "topology_assignment_relaxation_num_workers", 1) or 1)
    start_wall = time.perf_counter()
    out_levels: List[Dict[str, Any]] = []
    first_feasible: Optional[Dict[str, Any]] = None
    stopped = False
    total_solve = 0.0
    for level_id, description, slack, soft_window, soft_access, soft_large in levels:
        if time.perf_counter() - start_wall > total_budget:
            stopped = True
            out_levels.append({
                "level": level_id,
                "description": description,
                "diagnostic_status": "timeout",
                "solve_time_ms": 0.0,
                "skip_reason": "total_time_budget_exceeded",
            })
            break
        if level_id == "L0_original":
            status = "infeasible" if str(failure_reason) in {"capacity_conflict", "solver_infeasible"} else "skipped"
            out_levels.append({
                "level": level_id,
                "description": description,
                "diagnostic_status": status,
                "source": "original_solver_failure",
                "solve_time_ms": 0.0,
                "solver_status": str(solver_status or ""),
            })
            continue
        probe = _solve_diagnostic_probe(
            report,
            cp_model=cp_model,
            area_slack=slack,
            soft_window=soft_window,
            soft_access=soft_access,
            soft_large=soft_large,
            time_limit=per_level,
            num_workers=workers,
        )
        total_solve += float(probe.get("solve_time_ms", 0.0) or 0.0)
        probe.update({"level": level_id, "description": description})
        out_levels.append(probe)
        if probe.get("diagnostic_status") == "feasible" and first_feasible is None:
            first_feasible = probe
    total_wall = (time.perf_counter() - start_wall) * 1000.0
    summary = {
        "total_solve_time_ms": _r(total_solve),
        "total_wall_time_ms": _r(total_wall),
        "stopped_due_to_time_budget": bool(stopped),
        "first_feasible_level": str(first_feasible.get("level")) if first_feasible else "",
        "first_feasible_selected_variant_id": str(first_feasible.get("selected_variant_id")) if first_feasible else "",
        "first_feasible_relaxation_needed": dict(first_feasible.get("relaxation_needed") or {}) if first_feasible else {},
        "still_infeasible_after_all_levels": first_feasible is None,
    }
    return {
        "enabled": True,
        "levels": out_levels,
        "summary": summary,
        "violation_units": {"area": "m2", "slots": "count"},
    }


def _solve_diagnostic_probe(
    report: TopologyFeasibilityReport,
    *,
    cp_model: Any,
    area_slack: float,
    soft_window: bool,
    soft_access: bool,
    soft_large: bool,
    time_limit: float,
    num_workers: int,
) -> Dict[str, Any]:
    try:
        variants = _valid_variants(report)
        clusters = list(report.cluster_metrics or [])
        rows = [
            row for row in _all_rows(report)
            if any(v.variant_id == row.variant_id for v in variants)
        ]
        if not variants or not clusters or not rows:
            return {"diagnostic_status": "infeasible", "solve_time_ms": 0.0, "remaining_conflict_summary": "no_valid_rows"}
        cluster_by_id = {c.cluster_id: c for c in clusters}
        island_by_key = {
            (v.variant_id, i.island_id): i
            for v in variants
            for i in list(v.island_metrics or [])
        }
        model = cp_model.CpModel()
        y = {v.variant_id: model.NewBoolVar(f"diag_y_{v.variant_id}") for v in variants}
        x = {
            (r.variant_id, r.cluster_id, r.island_id): model.NewBoolVar(f"diag_x_{r.variant_id}_{r.cluster_id}_{r.island_id}")
            for r in rows
            if r.cluster_id in cluster_by_id and (r.variant_id, r.island_id) in island_by_key
        }
        model.Add(sum(y.values()) == 1)
        for cluster in clusters:
            model.Add(sum(var for key, var in x.items() if key[1] == cluster.cluster_id) == 1)
        for key, var in x.items():
            model.Add(var <= y[key[0]])
        for variant in variants:
            for island in list(variant.island_metrics or []):
                keys = [key for key in x if key[0] == variant.variant_id and key[2] == island.island_id]
                if not keys:
                    continue
                model.Add(sum(int(round(cluster_by_id[key[1]].target_area_sum * SCALE)) * x[key] for key in keys)
                          <= int(round(island.effective_capacity_area * area_slack * SCALE)) * y[variant.variant_id])
                if not soft_window:
                    model.Add(sum(int(cluster_by_id[key[1]].needs_window_count) * x[key] for key in keys)
                              <= int(island.window_slot_count) * y[variant.variant_id])
                if not soft_access:
                    model.Add(sum(int(cluster_by_id[key[1]].needs_corridor_access_count) * x[key] for key in keys)
                              <= int(island.corridor_door_slot_count) * y[variant.variant_id])
                if not soft_large:
                    model.Add(sum(int(cluster_by_id[key[1]].large_room_count) * x[key] for key in keys)
                              <= int(island.slot_count_large) * y[variant.variant_id])
        model.Minimize(sum(int(round((1.0 - float(r.feasibility_score)) * SCALE)) * x[(r.variant_id, r.cluster_id, r.island_id)] for r in rows if (r.variant_id, r.cluster_id, r.island_id) in x))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_search_workers = int(num_workers)
        t0 = time.perf_counter()
        status = solver.Solve(model)
        solve_ms = (time.perf_counter() - t0) * 1000.0
        status_name = solver.StatusName(status)
        if status_name in {"OPTIMAL", "FEASIBLE"}:
            assignment = {
                key[1]: {"variant_id": key[0], "island_id": key[2]}
                for key, var in x.items()
                if solver.Value(var) == 1
            }
            selected_variant_id = next((vid for vid, var in y.items() if solver.Value(var) == 1), "")
            violations = _diagnostic_violations(
                assignment,
                cluster_by_id=cluster_by_id,
                island_by_key=island_by_key,
                area_slack=area_slack,
            )
            return {
                "diagnostic_status": "feasible",
                "solve_time_ms": _r(solve_ms),
                "selected_variant_id": str(selected_variant_id),
                "remaining_conflict_summary": "",
                "relaxation_needed": violations,
            }
        if status_name in {"UNKNOWN"}:
            return {"diagnostic_status": "timeout", "solve_time_ms": _r(solve_ms), "solver_status": status_name}
        return {"diagnostic_status": "infeasible", "solve_time_ms": _r(solve_ms), "solver_status": status_name}
    except Exception as exc:
        return {"diagnostic_status": "error", "solve_time_ms": 0.0, "error_type": type(exc).__name__, "message": str(exc)[:200]}


def _diagnostic_violations(
    assignment: Mapping[str, Mapping[str, str]],
    *,
    cluster_by_id: Mapping[str, ClusterMetrics],
    island_by_key: Mapping[Tuple[str, str], IslandMetrics],
    area_slack: float,
) -> Dict[str, Any]:
    loads: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cluster_id, target in dict(assignment or {}).items():
        cluster = cluster_by_id.get(cluster_id)
        if cluster is None:
            continue
        key = (str(target.get("variant_id", "")), str(target.get("island_id", "")))
        island = island_by_key.get(key)
        if island is None:
            continue
        data = loads.setdefault(
            key,
            {
                "area": 0.0,
                "window": 0,
                "access": 0,
                "large": 0,
                "area_cap": float(island.effective_capacity_area) * float(area_slack),
                "window_cap": int(island.window_slot_count),
                "access_cap": int(island.corridor_door_slot_count),
                "large_cap": int(island.slot_count_large),
            },
        )
        data["area"] += float(cluster.target_area_sum)
        data["window"] += int(cluster.needs_window_count)
        data["access"] += int(cluster.needs_corridor_access_count)
        data["large"] += int(cluster.large_room_count)
    return {
        "area_slack_used": _r(area_slack),
        "area_violation": _r(max([float(v["area"]) - float(v["area_cap"]) for v in loads.values()] + [0.0])),
        "window_slot_violation": int(max([int(v["window"]) - int(v["window_cap"]) for v in loads.values()] + [0])),
        "access_slot_violation": int(max([int(v["access"]) - int(v["access_cap"]) for v in loads.values()] + [0])),
        "large_slot_violation": int(max([int(v["large"]) - int(v["large_cap"]) for v in loads.values()] + [0])),
    }
