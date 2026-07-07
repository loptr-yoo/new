"""Dry-run topology assignment CP-SAT proposal.

This module is intentionally diagnostic-only. It proposes a variant-scoped
cluster-to-island assignment from topology feasibility metadata, compares it to
the current heuristic assignment, and never mutates the active geometry path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .topology_feasibility import (
    ClusterIslandFeasibility,
    ClusterMetrics,
    IslandMetrics,
    TopologyFeasibilityReport,
    TopologyVariant,
)
from .topology_assignment_capacity_analyzer import build_capacity_conflict_summary

logger = logging.getLogger(__name__)

EPSILON = 1e-6
SCALE = 1000


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


@dataclass
class TopologyAssignmentConfig:
    enable_topology_assignment_cp_sat: bool = True
    topology_assignment_dry_run: bool = True
    enable_topology_assignment_adoption: bool = False
    allow_topology_assignment_fallback: bool = True
    time_limit_seconds: float = 2.0
    num_workers: int = 8
    area_capacity_slack: float = 1.0
    weight_fit: float = 0.48
    weight_shape: float = 0.18
    weight_access: float = 0.14
    weight_window: float = 0.14
    weight_margin: float = 0.06
    primary_variant_tie_break: int = 1
    force_ortools_unavailable: bool = False
    enable_topology_assignment_relaxation_diagnostics: bool = False
    topology_assignment_relaxation_time_limit_seconds: float = 0.5
    topology_assignment_relaxation_total_time_limit_seconds: float = 3.0
    topology_assignment_relaxation_max_levels: int = 6
    topology_assignment_relaxation_num_workers: int = 1


@dataclass
class TopologyAssignmentFailure:
    reason: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, config: Optional[TopologyAssignmentConfig] = None) -> Dict[str, Any]:
        cfg = config or TopologyAssignmentConfig()
        return {
            "status": "failure",
            "reason": self.reason,
            "message": self.message,
            "dry_run": bool(cfg.topology_assignment_dry_run),
            "used_for_main_path": False,
            "cp_sat_enabled": bool(cfg.enable_topology_assignment_cp_sat),
            "adoption_requested": bool(cfg.enable_topology_assignment_adoption),
            "adoption_implemented": False,
            "allow_topology_assignment_fallback": bool(cfg.allow_topology_assignment_fallback),
            **dict(self.metadata),
        }


@dataclass
class TopologyAssignmentResult:
    status: str
    reason: str
    selected_variant_id: Optional[str]
    selected_seed: Optional[int]
    selected_variant_profile: Dict[str, Any]
    proposed_cluster_to_island: Dict[str, Dict[str, str]]
    heuristic_cluster_to_island: Dict[str, Dict[str, str]]
    proposal_assignment_penalty: float
    proposal_assignment_score: float
    heuristic_assignment_penalty: float
    heuristic_assignment_score: float
    score_delta: float
    island_loads: Dict[str, Dict[str, Any]]
    diff_summary: Dict[str, Any]
    failed_cluster_diagnostics: Dict[str, Any]
    solver_status: str
    objective_value: Optional[float]
    missing_feasibility_pairs: List[Dict[str, str]]
    dry_run: bool = True
    used_for_main_path: bool = False
    adoption_requested: bool = False
    adoption_implemented: bool = False
    cp_sat_enabled: bool = True
    allow_topology_assignment_fallback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "dry_run": bool(self.dry_run),
            "used_for_main_path": bool(self.used_for_main_path),
            "cp_sat_enabled": bool(self.cp_sat_enabled),
            "adoption_requested": bool(self.adoption_requested),
            "adoption_implemented": bool(self.adoption_implemented),
            "allow_topology_assignment_fallback": bool(self.allow_topology_assignment_fallback),
            "selected_variant_id": self.selected_variant_id,
            "selected_seed": self.selected_seed,
            "selected_variant_profile": dict(self.selected_variant_profile),
            "proposed_cluster_to_island": dict(self.proposed_cluster_to_island),
            "heuristic_cluster_to_island": dict(self.heuristic_cluster_to_island),
            "proposal_assignment_penalty": _r(self.proposal_assignment_penalty),
            "proposal_assignment_score": _r(self.proposal_assignment_score),
            "heuristic_assignment_penalty": _r(self.heuristic_assignment_penalty),
            "heuristic_assignment_score": _r(self.heuristic_assignment_score),
            "score_delta": _r(self.score_delta),
            "island_loads": dict(self.island_loads),
            "diff_summary": dict(self.diff_summary),
            "failed_cluster_diagnostics": dict(self.failed_cluster_diagnostics),
            "solver_status": self.solver_status,
            "objective_value": self.objective_value,
            "missing_feasibility_pairs": list(self.missing_feasibility_pairs),
        }


class TopologyAssignmentSolver:
    def __init__(self, config: Optional[TopologyAssignmentConfig] = None) -> None:
        self.config = config or TopologyAssignmentConfig()

    def solve(
        self,
        report: TopologyFeasibilityReport,
        *,
        heuristic_assignments: Optional[Mapping[str, Any]] = None,
        failed_room_id: Optional[str] = None,
        floor_id: str = "",
    ) -> Dict[str, Any]:
        if self.config.force_ortools_unavailable:
            return TopologyAssignmentFailure(
                reason="ortools_unavailable",
                message="OR-Tools is unavailable for topology assignment dry-run",
            ).to_dict(self.config)
        try:
            from ortools.sat.python import cp_model  # type: ignore[import-not-found]
        except Exception as exc:
            return TopologyAssignmentFailure(
                reason="ortools_unavailable",
                message=str(exc),
            ).to_dict(self.config)

        variants = [v for v in list(report.variants or []) if bool(v.valid) and v.island_metrics]
        clusters = list(report.cluster_metrics or [])
        if not variants:
            return self._failure(
                "no_valid_topology_variant",
                "No valid topology variant is available",
                report=report,
                heuristic_assignments=heuristic_assignments,
                floor_id=floor_id,
            )
        if not clusters:
            return self._failure(
                "cluster_without_feasible_island",
                "No clusters are available for assignment",
                report=report,
                heuristic_assignments=heuristic_assignments,
                floor_id=floor_id,
            )

        feasibility_by_key = self._feasibility_by_key(report)
        island_by_key = {
            (v.variant_id, m.island_id): m
            for v in variants
            for m in list(v.island_metrics or [])
        }
        cluster_by_id = {c.cluster_id: c for c in clusters}

        feasible_rows = [
            row for row in feasibility_by_key.values()
            if bool(row.hard_feasible)
            and any(v.variant_id == row.variant_id for v in variants)
            and (row.variant_id, row.island_id) in island_by_key
            and row.cluster_id in cluster_by_id
        ]
        missing_clusters = [
            c.cluster_id
            for c in clusters
            if not any(row.cluster_id == c.cluster_id for row in feasible_rows)
        ]
        if missing_clusters:
            return self._failure(
                "cluster_without_feasible_island",
                "At least one cluster has no hard-feasible island in any valid variant",
                {"clusters_without_feasible_island": missing_clusters},
                report=report,
                heuristic_assignments=heuristic_assignments,
                floor_id=floor_id,
            )

        model = cp_model.CpModel()
        y = {v.variant_id: model.NewBoolVar(f"y_{v.variant_id}") for v in variants}
        x = {
            (row.variant_id, row.cluster_id, row.island_id): model.NewBoolVar(
                f"x_{row.variant_id}_{row.cluster_id}_{row.island_id}"
            )
            for row in feasible_rows
        }

        model.Add(sum(y.values()) == 1)
        for cluster in clusters:
            model.Add(
                sum(var for (variant_id, cluster_id, _island_id), var in x.items() if cluster_id == cluster.cluster_id)
                == 1
            )
        for (variant_id, _cluster_id, _island_id), var in x.items():
            model.Add(var <= y[variant_id])

        for variant in variants:
            for island in list(variant.island_metrics or []):
                keys = [
                    key for key in x
                    if key[0] == variant.variant_id and key[2] == island.island_id
                ]
                if not keys:
                    continue
                target_terms = [int(round(cluster_by_id[key[1]].target_area_sum * SCALE)) * x[key] for key in keys]
                window_terms = [int(cluster_by_id[key[1]].needs_window_count) * x[key] for key in keys]
                access_terms = [int(cluster_by_id[key[1]].needs_corridor_access_count) * x[key] for key in keys]
                large_terms = [int(cluster_by_id[key[1]].large_room_count) * x[key] for key in keys]
                capacity = int(round(float(island.effective_capacity_area) * float(self.config.area_capacity_slack) * SCALE))
                model.Add(sum(target_terms) <= capacity * y[variant.variant_id])
                model.Add(sum(window_terms) <= int(island.window_slot_count) * y[variant.variant_id])
                model.Add(sum(access_terms) <= int(island.corridor_door_slot_count) * y[variant.variant_id])
                model.Add(sum(large_terms) <= int(island.slot_count_large) * y[variant.variant_id])

        objective_terms = []
        for key, var in x.items():
            penalty = self._row_penalty(feasibility_by_key[key])
            objective_terms.append(int(round(penalty * SCALE)) * var)
        for variant in variants:
            tie_break = 0 if variant.variant_id == report.primary_variant_id else int(self.config.primary_variant_tie_break)
            if tie_break:
                objective_terms.append(tie_break * y[variant.variant_id])
        model.Minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.config.time_limit_seconds)
        solver.parameters.num_search_workers = int(self.config.num_workers)
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            reason = "capacity_conflict" if status == cp_model.INFEASIBLE else "solver_infeasible"
            return self._failure(
                reason,
                "Topology assignment CP-SAT could not satisfy aggregate capacities",
                {"solver_status": status_name},
                report=report,
                heuristic_assignments=heuristic_assignments,
                floor_id=floor_id,
            )

        selected_variant_id = next((variant_id for variant_id, var in y.items() if solver.Value(var) == 1), None)
        selected_variant = next((v for v in variants if v.variant_id == selected_variant_id), None)
        proposed: Dict[str, Dict[str, str]] = {}
        selected_rows: List[ClusterIslandFeasibility] = []
        for key, var in x.items():
            if solver.Value(var) != 1:
                continue
            variant_id, cluster_id, island_id = key
            proposed[cluster_id] = {"variant_id": variant_id, "island_id": island_id}
            selected_rows.append(feasibility_by_key[key])

        heuristic = self._heuristic_cluster_to_island(
            clusters,
            heuristic_assignments or {},
            primary_variant_id=report.primary_variant_id,
        )
        proposal_penalty, proposal_score, _proposal_missing = self._assignment_quality(selected_rows)
        heuristic_rows, missing_pairs = self._rows_for_assignment(heuristic, feasibility_by_key)
        heuristic_penalty, heuristic_score, _ = self._assignment_quality(heuristic_rows, missing_count=len(missing_pairs))
        island_loads = self._island_loads(
            proposed,
            cluster_by_id=cluster_by_id,
            island_by_key=island_by_key,
        )
        diff_summary = self._diff_summary(
            report=report,
            selected_variant=selected_variant,
            heuristic=heuristic,
            proposed=proposed,
        )
        failed_cluster_diagnostics = self._failed_cluster_diagnostics(
            failed_room_id=failed_room_id,
            clusters=clusters,
            heuristic=heuristic,
            proposed=proposed,
            feasibility_by_key=feasibility_by_key,
        )
        result = TopologyAssignmentResult(
            status="success",
            reason="ok",
            selected_variant_id=selected_variant_id,
            selected_seed=int(selected_variant.seed) if selected_variant is not None else None,
            selected_variant_profile=dict(selected_variant.variant_profile) if selected_variant is not None else {},
            proposed_cluster_to_island=proposed,
            heuristic_cluster_to_island=heuristic,
            proposal_assignment_penalty=proposal_penalty,
            proposal_assignment_score=proposal_score,
            heuristic_assignment_penalty=heuristic_penalty,
            heuristic_assignment_score=heuristic_score,
            score_delta=proposal_score - heuristic_score,
            island_loads=island_loads,
            diff_summary=diff_summary,
            failed_cluster_diagnostics=failed_cluster_diagnostics,
            solver_status=status_name,
            objective_value=float(solver.ObjectiveValue()),
            missing_feasibility_pairs=missing_pairs,
            dry_run=bool(self.config.topology_assignment_dry_run),
            used_for_main_path=False,
            cp_sat_enabled=bool(self.config.enable_topology_assignment_cp_sat),
            adoption_requested=bool(self.config.enable_topology_assignment_adoption),
            adoption_implemented=False,
            allow_topology_assignment_fallback=bool(self.config.allow_topology_assignment_fallback),
        )
        return result.to_dict()

    def _failure(
        self,
        reason: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        report: Optional[TopologyFeasibilityReport] = None,
        heuristic_assignments: Optional[Mapping[str, Any]] = None,
        floor_id: str = "",
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        if report is not None and reason in {
            "capacity_conflict",
            "solver_infeasible",
            "cluster_without_feasible_island",
            "no_valid_topology_variant",
        }:
            meta["capacity_conflict_summary"] = build_capacity_conflict_summary(
                report,
                failure_reason=reason,
                solver_status=str(meta.get("solver_status") or ""),
                heuristic_assignments=heuristic_assignments,
                config=self.config,
                floor_id=floor_id,
            )
        return TopologyAssignmentFailure(reason=reason, message=message, metadata=meta).to_dict(self.config)

    @staticmethod
    def _feasibility_by_key(report: TopologyFeasibilityReport) -> Dict[Tuple[str, str, str], ClusterIslandFeasibility]:
        return {
            (row.variant_id, row.cluster_id, row.island_id): row
            for variant in list(report.variants or [])
            for row in list(variant.feasibility_matrix or [])
        }

    def _row_penalty(self, row: ClusterIslandFeasibility) -> float:
        margin_penalty = _clamp(max(0.0, -float(row.capacity_margin_ratio)))
        penalty = (
            float(self.config.weight_fit) * _clamp(1.0 - float(row.feasibility_score))
            + float(self.config.weight_shape) * _clamp(float(row.shape_penalty))
            + float(self.config.weight_access) * _clamp(float(row.access_penalty))
            + float(self.config.weight_window) * _clamp(float(row.window_penalty))
            + float(self.config.weight_margin) * margin_penalty
        )
        return _clamp(penalty)

    @staticmethod
    def find_cluster_for_room(room_id: str, clusters: Sequence[ClusterMetrics]) -> Optional[str]:
        for cluster in clusters or []:
            if room_id in set(cluster.room_ids or []):
                return cluster.cluster_id
        return None

    @staticmethod
    def _heuristic_cluster_to_island(
        clusters: Sequence[ClusterMetrics],
        heuristic_assignments: Mapping[str, Any],
        *,
        primary_variant_id: str,
    ) -> Dict[str, Dict[str, str]]:
        room_to_island: Dict[str, str] = {}
        for island_id, assignment in dict(heuristic_assignments or {}).items():
            for room in list(getattr(assignment, "rooms", []) or []):
                room_id = str(getattr(room, "room_id", room))
                room_to_island[room_id] = str(island_id)
        output: Dict[str, Dict[str, str]] = {}
        for cluster in clusters or []:
            island_ids = sorted({room_to_island.get(room_id, "") for room_id in cluster.room_ids if room_to_island.get(room_id, "")})
            if len(island_ids) == 1:
                output[cluster.cluster_id] = {"variant_id": primary_variant_id, "island_id": island_ids[0]}
            elif len(island_ids) > 1:
                output[cluster.cluster_id] = {
                    "variant_id": primary_variant_id,
                    "island_id": island_ids[0],
                    "split_island_ids": ",".join(island_ids),
                }
        return output

    def _rows_for_assignment(
        self,
        assignment: Mapping[str, Mapping[str, str]],
        feasibility_by_key: Mapping[Tuple[str, str, str], ClusterIslandFeasibility],
    ) -> Tuple[List[ClusterIslandFeasibility], List[Dict[str, str]]]:
        rows: List[ClusterIslandFeasibility] = []
        missing: List[Dict[str, str]] = []
        for cluster_id, target in dict(assignment or {}).items():
            key = (str(target.get("variant_id", "")), str(cluster_id), str(target.get("island_id", "")))
            row = feasibility_by_key.get(key)
            if row is None:
                missing.append({
                    "variant_id": key[0],
                    "cluster_id": key[1],
                    "island_id": key[2],
                })
            else:
                rows.append(row)
        return rows, missing

    def _assignment_quality(
        self,
        rows: Sequence[ClusterIslandFeasibility],
        *,
        missing_count: int = 0,
    ) -> Tuple[float, float, int]:
        penalties = [self._row_penalty(row) for row in rows or []] + [1.0] * int(missing_count)
        if not penalties:
            return 1.0, 0.0, int(missing_count)
        penalty = _clamp(sum(penalties) / float(len(penalties)))
        return penalty, _clamp(1.0 - penalty), int(missing_count)

    @staticmethod
    def _island_loads(
        assignment: Mapping[str, Mapping[str, str]],
        *,
        cluster_by_id: Mapping[str, ClusterMetrics],
        island_by_key: Mapping[Tuple[str, str], IslandMetrics],
    ) -> Dict[str, Dict[str, Any]]:
        loads: Dict[str, Dict[str, Any]] = {}
        for cluster_id, target in dict(assignment or {}).items():
            variant_id = str(target.get("variant_id", ""))
            island_id = str(target.get("island_id", ""))
            key = f"{variant_id}:{island_id}"
            cluster = cluster_by_id.get(cluster_id)
            island = island_by_key.get((variant_id, island_id))
            if cluster is None or island is None:
                continue
            data = loads.setdefault(
                key,
                {
                    "variant_id": variant_id,
                    "island_id": island_id,
                    "target_area_sum": 0.0,
                    "min_area_sum": 0.0,
                    "max_area_sum": 0.0,
                    "needs_window_count": 0,
                    "needs_corridor_access_count": 0,
                    "large_room_count": 0,
                    "effective_capacity_area": _r(island.effective_capacity_area),
                    "capacity_margin": 0.0,
                    "window_slot_count": int(island.window_slot_count),
                    "corridor_door_slot_count": int(island.corridor_door_slot_count),
                    "slot_count_large": int(island.slot_count_large),
                    "clusters": [],
                },
            )
            data["clusters"].append(cluster_id)
            data["target_area_sum"] += float(cluster.target_area_sum)
            data["min_area_sum"] += float(cluster.min_area_sum)
            data["max_area_sum"] += float(cluster.max_area_sum)
            data["needs_window_count"] += int(cluster.needs_window_count)
            data["needs_corridor_access_count"] += int(cluster.needs_corridor_access_count)
            data["large_room_count"] += int(cluster.large_room_count)
        for data in loads.values():
            data["capacity_margin"] = _r(float(data["effective_capacity_area"]) - float(data["target_area_sum"]))
            data["target_area_sum"] = _r(data["target_area_sum"])
            data["min_area_sum"] = _r(data["min_area_sum"])
            data["max_area_sum"] = _r(data["max_area_sum"])
        return loads

    @staticmethod
    def _variant_resource_summary(variant: Optional[TopologyVariant]) -> Dict[str, float]:
        if variant is None:
            return {
                "corridor_area": 0.0,
                "island_count": 0.0,
                "effective_capacity": 0.0,
                "window_slots": 0.0,
                "door_slots": 0.0,
                "largest_empty_rect": 0.0,
            }
        islands = list(variant.island_metrics or [])
        return {
            "corridor_area": float(variant.corridor_area),
            "island_count": float(len(islands)),
            "effective_capacity": float(sum(float(i.effective_capacity_area) for i in islands)),
            "window_slots": float(sum(int(i.window_slot_count) for i in islands)),
            "door_slots": float(sum(int(i.corridor_door_slot_count) for i in islands)),
            "largest_empty_rect": float(max((float(i.largest_empty_rect_estimate) for i in islands), default=0.0)),
        }

    def _diff_summary(
        self,
        *,
        report: TopologyFeasibilityReport,
        selected_variant: Optional[TopologyVariant],
        heuristic: Mapping[str, Mapping[str, str]],
        proposed: Mapping[str, Mapping[str, str]],
    ) -> Dict[str, Any]:
        primary = next((v for v in report.variants if v.variant_id == report.primary_variant_id), None)
        primary_summary = self._variant_resource_summary(primary)
        selected_summary = self._variant_resource_summary(selected_variant)
        clusters_moved = []
        for cluster_id, new_target in dict(proposed or {}).items():
            old_target = dict(heuristic.get(cluster_id, {}) or {})
            if old_target.get("variant_id") != new_target.get("variant_id") or old_target.get("island_id") != new_target.get("island_id"):
                clusters_moved.append({
                    "cluster_id": cluster_id,
                    "old_variant_id": old_target.get("variant_id"),
                    "old_island_id": old_target.get("island_id"),
                    "new_variant_id": new_target.get("variant_id"),
                    "new_island_id": new_target.get("island_id"),
                })
        selected_areas = sorted([round(float(i.area), 3) for i in list(getattr(selected_variant, "island_metrics", []) or [])])
        primary_areas = sorted([round(float(i.area), 3) for i in list(getattr(primary, "island_metrics", []) or [])])
        return {
            "topology_diff": {
                "selected_variant_id": getattr(selected_variant, "variant_id", None),
                "primary_variant_id": report.primary_variant_id,
                "selected_seed": getattr(selected_variant, "seed", None),
                "corridor_area_delta": _r(selected_summary["corridor_area"] - primary_summary["corridor_area"]),
                "island_count_delta": int(selected_summary["island_count"] - primary_summary["island_count"]),
                "island_area_distribution_delta": {
                    "primary": primary_areas,
                    "selected": selected_areas,
                },
                "effective_capacity_delta": _r(selected_summary["effective_capacity"] - primary_summary["effective_capacity"]),
                "window_slot_delta": int(selected_summary["window_slots"] - primary_summary["window_slots"]),
                "corridor_door_slot_delta": int(selected_summary["door_slots"] - primary_summary["door_slots"]),
                "largest_empty_rect_delta": _r(selected_summary["largest_empty_rect"] - primary_summary["largest_empty_rect"]),
            },
            "assignment_diff": {
                "clusters_moved": clusters_moved,
                "moved_count": len(clusters_moved),
            },
        }

    def _failed_cluster_diagnostics(
        self,
        *,
        failed_room_id: Optional[str],
        clusters: Sequence[ClusterMetrics],
        heuristic: Mapping[str, Mapping[str, str]],
        proposed: Mapping[str, Mapping[str, str]],
        feasibility_by_key: Mapping[Tuple[str, str, str], ClusterIslandFeasibility],
    ) -> Dict[str, Any]:
        if not failed_room_id:
            return {"status": "not_requested"}
        cluster_id = self.find_cluster_for_room(failed_room_id, clusters)
        if not cluster_id:
            return {"status": "room_cluster_not_found", "failed_room_id": failed_room_id}
        h = dict(heuristic.get(cluster_id, {}) or {})
        p = dict(proposed.get(cluster_id, {}) or {})
        h_row = feasibility_by_key.get((h.get("variant_id", ""), cluster_id, h.get("island_id", "")))
        p_row = feasibility_by_key.get((p.get("variant_id", ""), cluster_id, p.get("island_id", "")))
        return {
            "status": "ok",
            "failed_room_id": failed_room_id,
            "failed_cluster_id": cluster_id,
            "heuristic_variant_id": h.get("variant_id"),
            "heuristic_island_id": h.get("island_id"),
            "proposal_variant_id": p.get("variant_id"),
            "proposal_island_id": p.get("island_id"),
            "heuristic_fit_score": _r(h_row.feasibility_score) if h_row is not None else None,
            "proposal_fit_score": _r(p_row.feasibility_score) if p_row is not None else None,
        }
