"""Geometry-only fixed-allocation smoke runner.

This module intentionally bypasses the semantic/LLM pipeline.  It parses a
previously logged BuildingAllocation, runs geometry generation directly, and
summarizes enough diagnostics to verify topology-assignment adoption handoff.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import traceback
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from ..models import BuildingAllocation
from ..semantics.generator import _parse_budgeted_allocation
from .building_orchestrator import BuildingOrchestrator, BuildingResult
from .exceptions import LayoutCoverageError, LayoutGenerationError, LayoutGeometryInvariantError, LayoutTopologyError
from .postprocessor import SemanticInvalidError
from .room_spec import SolverConfig
from .territory_provenance import (
    build_circulation_contract,
    build_island_capacity_blocker_explanation,
    build_island_cluster_provenance,
)
from .topology_generator import CoreTube


DEFAULT_LLM_LOG_PATH = Path("building/out/test_gemini_east/llm_log.txt")
DEFAULT_OUTPUT_PATH = Path("building/out/test_gemini_east/geometry_smoke_fixed_allocation_result.json")
DEFAULT_ACCEPTABLE_FAILURE_STAGES = {
    "coverage_debt_fallback_infeasible",
    "core_access_failed",
    "continuous_solver_infeasible",
    "topology_assignment_infeasible",
}
CORE_OVERLAP_EPSILON = 0.01
FLOOR_SCOPE_AUDIT_VERSION = "floor_scoped_provenance_v1"
FLOOR_SCOPE_AREA_ABS_TOLERANCE = 1e-6
FLOOR_SCOPE_AREA_REL_TOLERANCE = 1e-4


@dataclass(frozen=True)
class GeometrySmokeInputs:
    floors: int = 2
    floor_width: float = 15.0
    floor_height: float = 10.0
    corridor_width: float = 1.8
    core_area_ratio: float = 0.12
    core_placement: str = "east"
    topology_mode: str = "grid_growth"
    corridor_layout: str = "organic"
    base_seed: int = 0
    enable_capacity_aware_area_allocation: bool = False
    apply_capacity_aware_area_allocation: bool = False
    enable_semantic_seeded_territory_variants: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AllocationParseResult:
    allocation: BuildingAllocation
    allocation_source_index: int
    source_kind: str
    parse_warnings: List[str]


class LLMCallForbidden(RuntimeError):
    pass


class LLMGuard(AbstractContextManager["LLMGuard"]):
    """Patch known LLM entry points so geometry-only smoke cannot call them."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.calls_attempted = 0
        self.violations: List[str] = []
        self._patches: List[Tuple[Any, str, Any]] = []

    def __enter__(self) -> "LLMGuard":
        if not self.enabled:
            return self

        def _forbidden(name: str):
            def _raise(*args: Any, **kwargs: Any) -> Any:
                self.calls_attempted += 1
                self.violations.append(name)
                raise LLMCallForbidden(f"LLM call forbidden during fixed-allocation geometry smoke: {name}")

            async def _araise(*args: Any, **kwargs: Any) -> Any:
                self.calls_attempted += 1
                self.violations.append(name)
                raise LLMCallForbidden(f"LLM call forbidden during fixed-allocation geometry smoke: {name}")

            return _araise if name.endswith(".async") else _raise

        for module_name, attr_name, is_async in [
            ("building.app.semantics.generator", "generate_building_envelope", True),
            ("building.app.semantics.generator", "generate_budgeted_building_semantics", True),
            ("building.app.semantics.generator", "generate_building_semantics", True),
            ("building.app.services.building_pipeline_service", "generate_building_envelope", True),
            ("building.app.services.building_pipeline_service", "generate_budgeted_building_semantics", True),
            ("building.app.llm.provider", "create_llm_client", False),
            ("building.app.llm.retry", "call_llm_with_retry", True),
            ("building.app.interior.coarse_layout_agent", "generate_coarse_layout", True),
        ]:
            try:
                module = __import__(module_name, fromlist=[attr_name])
                original = getattr(module, attr_name)
            except Exception:
                continue
            self._patches.append((module, attr_name, original))
            marker = f"{module_name}.{attr_name}" + (".async" if is_async else "")
            setattr(module, attr_name, _forbidden(marker))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Optional[bool]:
        for module, attr_name, original in reversed(self._patches):
            setattr(module, attr_name, original)
        self._patches.clear()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "calls_attempted": int(self.calls_attempted),
            "violations": list(self.violations),
        }


def _short_uuid() -> str:
    return uuid.uuid4().hex[:12]


def _round_float(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseGeometry):
        bounds = []
        try:
            bounds = [_round_float(v) for v in value.bounds]
        except Exception:
            bounds = []
        return {
            "geometry_type": value.geom_type,
            "area": _round_float(getattr(value, "area", 0.0)),
            "bounds": bounds,
        }
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if is_dataclass(value):
        try:
            return _json_safe(asdict(value))
        except Exception:
            return str(value)
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _extract_response_blocks(text: str) -> List[str]:
    marker = "[response]"
    starts = [m.end() for m in re.finditer(re.escape(marker), text)]
    blocks: List[str] = []
    for start in starts:
        next_messages = text.find("\n[messages]", start)
        end = next_messages if next_messages >= 0 else len(text)
        candidate = text[start:end].strip()
        if candidate:
            blocks.append(candidate)
    return blocks


def parse_latest_budgeted_allocation(llm_log_path: Path | str) -> AllocationParseResult:
    path = Path(llm_log_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _extract_response_blocks(text)
    errors: List[str] = []
    for reverse_index, block in enumerate(reversed(blocks)):
        source_index = len(blocks) - 1 - reverse_index
        try:
            allocation, warnings = _parse_budgeted_allocation(
                block,
                provider="fixed_allocation_smoke",
                model=path.name,
            )
            return AllocationParseResult(
                allocation=allocation,
                allocation_source_index=source_index,
                source_kind="budgeted_allocation_response",
                parse_warnings=list(warnings),
            )
        except Exception as exc:
            errors.append(f"response[{source_index}]: {type(exc).__name__}: {exc}")
            continue
    raise RuntimeError(
        "No valid BuildingAllocation response block found in llm_log. "
        + "; ".join(errors[-5:])
    )


def _copy_allocation(allocation: BuildingAllocation, floors: int) -> BuildingAllocation:
    if hasattr(allocation, "model_copy"):
        copied = allocation.model_copy(deep=True)
    else:
        copied = copy.deepcopy(allocation)
    if floors > 0:
        copied.floors = list(copied.floors[: int(floors)])
        copied.total_floors = len(copied.floors)
        copied.overall_total_area = float(sum(float(f.floor_total_area) for f in copied.floors))
    return copied


def _make_solver_config(mode: str, inputs: GeometrySmokeInputs) -> SolverConfig:
    config = SolverConfig()
    config.enable_topology_assignment_cp_sat = True
    config.allow_topology_assignment_fallback = True
    config.enable_topology_assignment_relaxation_diagnostics = True
    config.enable_capacity_aware_area_allocation = bool(inputs.enable_capacity_aware_area_allocation)
    config.apply_capacity_aware_area_allocation = bool(inputs.apply_capacity_aware_area_allocation)
    config.enable_semantic_seeded_territory_variants = bool(inputs.enable_semantic_seeded_territory_variants)
    config.semantic_seeded_territory_variants_dry_run = True
    if mode == "adoption":
        config.topology_assignment_dry_run = False
        config.enable_topology_assignment_adoption = True
    else:
        config.topology_assignment_dry_run = True
        config.enable_topology_assignment_adoption = False
    return config


def _make_orchestrator(inputs: GeometrySmokeInputs, config: SolverConfig) -> Tuple[BuildingOrchestrator, CoreTube]:
    floor_boundary: Polygon = box(0.0, 0.0, float(inputs.floor_width), float(inputs.floor_height))
    core = CoreTube.create_for_floor(
        floor_bounds=floor_boundary.bounds,
        area_ratio=float(inputs.core_area_ratio),
        position=str(inputs.core_placement),
    )
    orchestrator = BuildingOrchestrator(
        floor_boundary=floor_boundary,
        config=config,
        corridor_width=float(inputs.corridor_width),
        core_area_ratio=float(inputs.core_area_ratio),
        corridor_layout=str(inputs.corridor_layout),
        topology_mode=str(inputs.topology_mode),
        base_seed=int(inputs.base_seed),
    )
    orchestrator._shared_core_tube = core
    return orchestrator, core


def _meta_float(metadata: Mapping[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        if key in metadata:
            return _round_float(metadata.get(key), 6)
    return 0.0


def _find_coverage_debt(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    debt = metadata.get("coverage_debt") if isinstance(metadata, Mapping) else None
    if not isinstance(debt, Mapping):
        plan = metadata.get("coverage_debt_plan") if isinstance(metadata, Mapping) else None
        if isinstance(plan, Mapping):
            debt = {"failure_plan": plan}
        else:
            return {"available": False, "missing_reason": "coverage_debt_metadata_absent"}

    plans: Dict[str, Any] = {}
    variant_ids: List[str] = []
    island_area_sources: List[str] = []
    island_areas: Dict[str, float] = {}
    assigned_room_ids: Dict[str, List[str]] = {}
    for key, value in debt.items():
        if key in {"coverage_features", "ledger"} or not isinstance(value, Mapping):
            continue
        plan = dict(value)
        diagnostics = plan.get("diagnostics") if isinstance(plan.get("diagnostics"), Mapping) else {}
        variant_id = str(diagnostics.get("variant_id") or diagnostics.get("selected_variant_id") or "")
        if variant_id:
            variant_ids.append(variant_id)
        source = str(diagnostics.get("island_area_source") or "")
        if source:
            island_area_sources.append(source)
        island_areas[str(key)] = _round_float(plan.get("island_area", 0.0), 6)
        assigned_room_ids[str(key)] = [str(x) for x in list(plan.get("assigned_room_ids") or [])]
        plans[str(key)] = _json_safe(plan)

    if not plans:
        return {"available": False, "missing_reason": "coverage_debt_plans_absent"}
    return {
        "available": True,
        "plans": plans,
        "variant_ids": sorted(set(variant_ids)),
        "variant_id": sorted(set(variant_ids))[0] if variant_ids else "",
        "island_area_sources": sorted(set(island_area_sources)),
        "island_area_source": sorted(set(island_area_sources))[0] if island_area_sources else "",
        "island_areas": island_areas,
        "assigned_room_ids": assigned_room_ids,
    }


def _find_core_overlap(metadata: Mapping[str, Any], exc_metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    candidates: List[Mapping[str, Any]] = []
    if isinstance(metadata, Mapping):
        core_contract = metadata.get("core_contract")
        if isinstance(core_contract, Mapping):
            for key in ("exclusion_diagnostics", "diagnostics", "core_diagnostics"):
                val = core_contract.get(key)
                if isinstance(val, Mapping):
                    candidates.append(val)
        for key in ("core_diagnostics", "core_overlap_diagnostics", "diagnostics"):
            val = metadata.get(key)
            if isinstance(val, Mapping):
                candidates.append(val)
    if isinstance(exc_metadata, Mapping):
        candidates.append(exc_metadata)
        for key in ("core_diagnostics", "core_overlap_diagnostics", "diagnostics"):
            val = exc_metadata.get(key)
            if isinstance(val, Mapping):
                candidates.append(val)

    merged: Dict[str, Any] = {}
    for candidate in candidates:
        merged.update(candidate)

    room = _meta_float(merged, ["room_core_overlap_total", "room_core_overlap_area", "room_core_overlap", "overlap_area"])
    corridor = _meta_float(merged, ["corridor_core_overlap_total", "corridor_core_overlap_area", "corridor_core_overlap"])
    feature = _meta_float(merged, ["coverage_feature_core_overlap_total", "coverage_feature_core_overlap_area"])
    generated = _meta_float(merged, ["generated_room_core_overlap_total", "generated_room_core_overlap_area"])
    max_overlap = max(room, corridor, feature, generated)
    return {
        "room_core_overlap_area": room,
        "corridor_core_overlap_area": corridor,
        "coverage_feature_core_overlap_area": feature,
        "generated_room_core_overlap_area": generated,
        "max_core_overlap_area": max_overlap,
        "core_overlap_regression": bool(max_overlap > CORE_OVERLAP_EPSILON),
        "epsilon": CORE_OVERLAP_EPSILON,
    }


def _find_failed_cluster(proposal: Mapping[str, Any], failed_room_id: str = "") -> Dict[str, Any]:
    diag = proposal.get("failed_cluster_diagnostics") if isinstance(proposal, Mapping) else None
    if isinstance(diag, Mapping):
        return _json_safe(diag)
    if not failed_room_id:
        return {"status": "not_available"}
    cluster_metrics = proposal.get("cluster_metrics") if isinstance(proposal, Mapping) else None
    if isinstance(cluster_metrics, Sequence):
        for cluster in cluster_metrics:
            if isinstance(cluster, Mapping) and failed_room_id in set(str(x) for x in cluster.get("room_ids", [])):
                return {
                    "status": "ok",
                    "failed_room_id": failed_room_id,
                    "failed_cluster_id": str(cluster.get("cluster_id", "")),
                }
    return {"status": "room_cluster_not_found", "failed_room_id": failed_room_id}


def _proposal_selected_variant(proposal: Mapping[str, Any]) -> str:
    return str(proposal.get("selected_variant_id") or proposal.get("selected_variant") or "")


def _capacity_summary_from(proposal: Mapping[str, Any], adoption: Mapping[str, Any]) -> Dict[str, Any]:
    proposal_summary = proposal.get("capacity_conflict_summary") if isinstance(proposal.get("capacity_conflict_summary"), Mapping) else None
    if proposal_summary is not None:
        return dict(proposal_summary)
    gate = adoption.get("adoption_gate") if isinstance(adoption.get("adoption_gate"), Mapping) else {}
    gate_summary = gate.get("capacity_conflict_summary") if isinstance(gate.get("capacity_conflict_summary"), Mapping) else None
    return dict(gate_summary or {})


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _round_area(value: Any) -> float:
    try:
        return round(float(value or 0.0), 6)
    except Exception:
        return 0.0


def _floor_id_for_number(value: Any, fallback: str = "") -> str:
    try:
        return f"F{int(value)}"
    except Exception:
        return str(fallback or "")


def _allocation_floor_summaries(allocation: BuildingAllocation) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for idx, floor in enumerate(list(getattr(allocation, "floors", []) or []), start=1):
        floor_id = _floor_id_for_number(getattr(floor, "floor_number", idx), fallback=f"F{idx}")
        rooms = list(getattr(floor, "rooms", []) or [])
        room_ids = [str(getattr(room, "room_id", "") or f"room_{i:03d}") for i, room in enumerate(rooms, start=1)]
        target_sum = sum(float(getattr(room, "target_area", 0.0) or 0.0) for room in rooms)
        summaries[floor_id] = {
            "floor_id": floor_id,
            "floor_number": int(getattr(floor, "floor_number", idx) or idx),
            "floor_total_area": _round_area(getattr(floor, "floor_total_area", 0.0)),
            "room_count": len(rooms),
            "target_area_sum": _round_area(target_sum),
            "raw_room_ids": room_ids,
            "canonical_room_ids": [f"{floor_id}:{room_id}" for room_id in room_ids],
        }
    return summaries


def _audit_entry(
    *,
    severity: str,
    check: str,
    floor_id: str = "",
    status: str = "fail",
    expected: Any = None,
    actual: Any = None,
    detail: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    entry = {
        "severity": str(severity),
        "check": str(check),
        "status": str(status),
    }
    if floor_id:
        entry["floor_id"] = str(floor_id)
    if expected is not None:
        entry["expected"] = expected
    if actual is not None:
        entry["actual"] = actual
    if detail:
        entry["detail"] = dict(detail)
    return entry


def _add_audit_entry(audit: Dict[str, Any], entry: Mapping[str, Any]) -> None:
    severity = str(entry.get("severity") or "info")
    if severity == "error":
        audit["violations"].append(dict(entry))
    elif severity == "warning":
        audit["warnings"].append(dict(entry))
    else:
        audit["infos"].append(dict(entry))


def _area_close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= max(
        FLOOR_SCOPE_AREA_ABS_TOLERANCE,
        FLOOR_SCOPE_AREA_REL_TOLERANCE * max(abs(float(a)), abs(float(b)), 1.0),
    )


def _sum_cluster_targets(clusters: Sequence[Any]) -> Tuple[float, int, List[str], Dict[str, List[str]]]:
    target = 0.0
    room_ids: List[str] = []
    cluster_ids: List[str] = []
    rooms_by_cluster: Dict[str, List[str]] = {}
    for item in clusters:
        if not isinstance(item, Mapping):
            continue
        cluster_id = str(item.get("cluster_id") or "")
        if cluster_id:
            cluster_ids.append(cluster_id)
        raw_rooms = item.get("room_ids", item.get("rooms", []))
        rooms = [str(r) for r in _safe_list(raw_rooms)]
        if cluster_id:
            rooms_by_cluster[cluster_id] = rooms
        room_ids.extend(rooms)
        target += float(item.get("target_area_sum", item.get("target_sum", 0.0)) or 0.0)
    return _round_area(target), len(set(room_ids)), cluster_ids, rooms_by_cluster


def _compact_floor_scope_metadata(floor_id: str, metadata: Mapping[str, Any], grid: Mapping[str, Any]) -> Dict[str, Any]:
    core_contract = metadata.get("core_contract") if isinstance(metadata.get("core_contract"), Mapping) else {}
    grid_core = grid.get("core_contract") if isinstance(grid.get("core_contract"), Mapping) else {}
    clusters = _safe_list(grid.get("cluster_metrics")) or _safe_list(grid.get("clusters"))
    variants = _safe_list(grid.get("topology_variants"))
    feasibility = _safe_list(grid.get("cluster_island_feasibility"))
    actual_area = 0.0
    effective_area = 0.0
    variant_summaries: List[Dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        islands = _safe_list(variant.get("islands"))
        v_actual = sum(float((island or {}).get("area", 0.0) or 0.0) for island in islands if isinstance(island, Mapping))
        v_effective = sum(float((island or {}).get("effective_capacity_area", 0.0) or 0.0) for island in islands if isinstance(island, Mapping))
        actual_area = max(actual_area, v_actual)
        effective_area = max(effective_area, v_effective)
        variant_summaries.append(
            {
                "variant_id": str(variant.get("variant_id") or ""),
                "seed": variant.get("seed"),
                "valid": bool(variant.get("valid", False)),
                "island_count": len(islands),
                "actual_island_area_sum": _round_area(v_actual),
                "effective_capacity_sum": _round_area(v_effective),
                "core_union_hash": str(variant.get("core_union_hash") or ""),
            }
        )
    cluster_target, cluster_room_count, cluster_ids, rooms_by_cluster = _sum_cluster_targets(clusters)
    return _json_safe(
        {
            "container_floor_id": str(floor_id),
            "grid_floor_id": str(grid.get("floor_id") or ""),
            "topology_report_floor_id": str(grid.get("floor_id") or ""),
            "core_contract_floor_id": str(
                core_contract.get("floor_id")
                or grid_core.get("floor_id")
                or core_contract.get("core_contract_floor_id")
                or ""
            ),
            "core_union_hash": str(
                core_contract.get("core_union_hash")
                or grid_core.get("core_union_hash")
                or ""
            ),
            "cluster_metrics_source": "cluster_metrics" if grid.get("cluster_metrics") is not None else ("clusters" if grid.get("clusters") is not None else "missing"),
            "cluster_count": len(clusters),
            "cluster_target_area_sum": cluster_target,
            "cluster_room_count": cluster_room_count,
            "raw_cluster_ids": cluster_ids,
            "room_ids_by_cluster": rooms_by_cluster,
            "variant_count": len(variants),
            "feasibility_row_count": len(feasibility),
            "topology_variants": variant_summaries[:10],
            "actual_island_area_sum": _round_area(actual_area),
            "effective_capacity_sum": _round_area(effective_area),
            "capacity_source": "grid_growth_island_metrics" if variants else "missing",
        }
    )


def _floor_id_from_mapping(value: Mapping[str, Any]) -> str:
    return str(value.get("floor_id") or value.get("floor") or value.get("floor_name") or "")


def _check_internal_floor_id(audit: Dict[str, Any], *, source: str, floor_id: str, actual: str) -> str:
    if not actual:
        _add_audit_entry(
            audit,
            _audit_entry(
                severity="warning",
                check=f"missing_explicit_floor_provenance:{source}",
                floor_id=floor_id,
                status="missing",
                expected=floor_id,
                actual="",
            ),
        )
        return "missing"
    if actual != floor_id:
        _add_audit_entry(
            audit,
            _audit_entry(
                severity="error",
                check=f"{source}_floor_id_matches_container",
                floor_id=floor_id,
                expected=floor_id,
                actual=actual,
            ),
        )
        return "fail"
    return "pass"


def _build_floor_scope_audit(run: Mapping[str, Any], allocation: BuildingAllocation, *, mode: str) -> Dict[str, Any]:
    allocation_by_floor = _allocation_floor_summaries(allocation)
    floors = run.get("floors") if isinstance(run.get("floors"), Mapping) else {}
    audit: Dict[str, Any] = {
        "audit_version": FLOOR_SCOPE_AUDIT_VERSION,
        "source": "fixed_allocation_smoke",
        "mode": str(mode),
        "analysis_only": True,
        "used_for_solver_decision": False,
        "used_for_adoption": False,
        "area_consistency_tolerance": {
            "abs": FLOOR_SCOPE_AREA_ABS_TOLERANCE,
            "rel": FLOOR_SCOPE_AREA_REL_TOLERANCE,
        },
        "count_tolerance": 0,
        "floor_scope_consistent": True,
        "violations": [],
        "warnings": [],
        "infos": [],
        "allocation_by_floor": allocation_by_floor,
        "raw_room_ids_by_floor": {fid: data.get("raw_room_ids", []) for fid, data in allocation_by_floor.items()},
        "canonical_room_ids_by_floor": {fid: data.get("canonical_room_ids", []) for fid, data in allocation_by_floor.items()},
        "raw_cluster_ids_by_floor": {},
        "canonical_cluster_ids_by_floor": {},
        "room_ids_by_cluster": {},
        "provenance_chain_by_floor": {},
        "area_consistency_by_floor": {},
        "room_count_consistency_by_floor": {},
        "capacity_provenance_by_floor": {},
        "core_provenance": {
            "shared_core_expected": True,
            "same_hash_allowed": True,
            "core_union_hash_by_floor": {},
            "core_scope_consistent": True,
        },
        "failure_attribution": {},
        "f2_capacity_conflict_provenance": {
            "present": False,
            "verdict": "insufficient_metadata",
            "evidence": {},
            "blocking_evidence_missing": [],
        },
    }

    raw_room_owner: Dict[str, List[str]] = {}
    canonical_rooms: Dict[str, List[str]] = {}
    for floor_id, data in allocation_by_floor.items():
        for room_id in data.get("raw_room_ids", []):
            raw_room_owner.setdefault(str(room_id), []).append(floor_id)
            canonical_rooms.setdefault(f"{floor_id}:{room_id}", []).append(floor_id)
    for room_id, owners in sorted(raw_room_owner.items()):
        if len(set(owners)) > 1:
            _add_audit_entry(
                audit,
                _audit_entry(
                    severity="warning",
                    check="duplicate_raw_room_id_across_floors",
                    status="warning",
                    actual=room_id,
                    detail={"floor_ids": sorted(set(owners))},
                ),
            )
    for canonical_id, owners in sorted(canonical_rooms.items()):
        if len(owners) > 1:
            _add_audit_entry(
                audit,
                _audit_entry(
                    severity="error",
                    check="duplicate_canonical_room_id",
                    status="fail",
                    actual=canonical_id,
                    detail={"floor_ids": owners},
                ),
            )

    per_floor_failed = [str(fid) for fid, diag in floors.items() if isinstance(diag, Mapping) and str(diag.get("stage") or "")]
    top_floor = str(run.get("floor_id") or "")
    first_floor = str(run.get("floor_id") or (per_floor_failed[0] if per_floor_failed else ""))
    if not top_floor and per_floor_failed:
        attribution_status = "missing"
    elif top_floor and top_floor in per_floor_failed:
        attribution_status = "consistent"
    elif top_floor and per_floor_failed:
        attribution_status = "mismatch"
        _add_audit_entry(
            audit,
            _audit_entry(
                severity="error",
                check="top_level_failure_floor_matches_per_floor_failure",
                expected=per_floor_failed,
                actual=top_floor,
            ),
        )
    else:
        attribution_status = "consistent"
    audit["failure_attribution"] = {
        "first_failed_floor_id": first_floor,
        "top_level_failure_floor_id": top_floor,
        "per_floor_failed_floor_ids": per_floor_failed,
        "attribution_status": attribution_status,
    }

    raw_cluster_owner: Dict[str, List[str]] = {}
    canonical_clusters: Dict[str, List[str]] = {}
    floor_ids = sorted(set(allocation_by_floor) | {str(fid) for fid in floors.keys()})
    for floor_id in floor_ids:
        diag = floors.get(floor_id) if isinstance(floors, Mapping) else {}
        diag = diag if isinstance(diag, Mapping) else {}
        proposal = diag.get("topology_assignment_proposal") if isinstance(diag.get("topology_assignment_proposal"), Mapping) else {}
        adoption = diag.get("topology_assignment_adoption") if isinstance(diag.get("topology_assignment_adoption"), Mapping) else {}
        gate = adoption.get("adoption_gate") if isinstance(adoption.get("adoption_gate"), Mapping) else {}
        capacity = diag.get("capacity_conflict_summary") if isinstance(diag.get("capacity_conflict_summary"), Mapping) else {}
        floor_scope_meta = diag.get("floor_scope_metadata") if isinstance(diag.get("floor_scope_metadata"), Mapping) else {}
        allocation_data = allocation_by_floor.get(floor_id, {})

        proposal_floor = _floor_id_from_mapping(proposal)
        capacity_floor = _floor_id_from_mapping(capacity)
        gate_floor = _floor_id_from_mapping(gate)
        gate_capacity = gate.get("capacity_conflict_summary") if isinstance(gate.get("capacity_conflict_summary"), Mapping) else {}
        gate_capacity_floor = _floor_id_from_mapping(gate_capacity)
        report_floor = str(floor_scope_meta.get("topology_report_floor_id") or "")
        cluster_ids = [str(x) for x in _safe_list(floor_scope_meta.get("raw_cluster_ids"))]
        audit["raw_cluster_ids_by_floor"][floor_id] = cluster_ids
        audit["canonical_cluster_ids_by_floor"][floor_id] = [f"{floor_id}:{cluster_id}" for cluster_id in cluster_ids]
        audit["room_ids_by_cluster"][floor_id] = dict(floor_scope_meta.get("room_ids_by_cluster") or {})
        for cluster_id in cluster_ids:
            raw_cluster_owner.setdefault(cluster_id, []).append(floor_id)
            canonical_clusters.setdefault(f"{floor_id}:{cluster_id}", []).append(floor_id)

        proposal_status = str(proposal.get("status") or "")
        proposal_reason = str(proposal.get("reason") or "")
        primary_conflict = str(
            diag.get("primary_conflict_type")
            or ((capacity.get("diagnosis") if isinstance(capacity.get("diagnosis"), Mapping) else {}) or {}).get("primary_conflict_type")
            or ""
        )
        chain = {
            "allocation": {
                "floor_id": allocation_data.get("floor_id", ""),
                "room_count": allocation_data.get("room_count"),
                "target_area_sum": allocation_data.get("target_area_sum"),
                "status": "pass" if allocation_data else "missing",
            },
            "clusters": {
                "floor_id": floor_id if cluster_ids else "",
                "cluster_count": int(floor_scope_meta.get("cluster_count", 0) or 0),
                "room_count": int(floor_scope_meta.get("cluster_room_count", 0) or 0),
                "target_area_sum": floor_scope_meta.get("cluster_target_area_sum", 0.0),
                "status": "pass" if cluster_ids else "missing",
            },
            "feasibility_report": {
                "floor_id": report_floor,
                "variant_count": int(floor_scope_meta.get("variant_count", 0) or 0),
                "cluster_count": int(floor_scope_meta.get("cluster_count", 0) or 0),
                "feasibility_row_count": int(floor_scope_meta.get("feasibility_row_count", 0) or 0),
                "status": "pass" if report_floor == floor_id else ("missing" if not report_floor else "fail"),
            },
            "proposal": {
                "floor_id": proposal_floor,
                "status": proposal_status,
                "reason": proposal_reason,
                "floor_id_check": _check_internal_floor_id(audit, source="proposal", floor_id=floor_id, actual=proposal_floor)
                if proposal
                else "missing",
            },
            "capacity_summary": {
                "floor_id": capacity_floor,
                "primary_conflict_type": primary_conflict,
                "target_area_sum": ((capacity.get("global_demand") if isinstance(capacity.get("global_demand"), Mapping) else {}) or {}).get("target_area_sum"),
                "floor_id_check": _check_internal_floor_id(audit, source="capacity_summary", floor_id=floor_id, actual=capacity_floor)
                if capacity
                else "missing",
            },
            "adoption_gate": {
                "floor_id": gate_floor,
                "gate_block_reason": str(gate.get("gate_block_reason") or ""),
                "gate_capacity_summary_floor_id": gate_capacity_floor,
                "floor_id_check": _check_internal_floor_id(audit, source="adoption_gate", floor_id=floor_id, actual=gate_floor)
                if gate and gate_floor
                else ("missing" if gate else "not_applicable"),
            },
        }
        if gate_capacity:
            _check_internal_floor_id(audit, source="adoption_gate_capacity_summary", floor_id=floor_id, actual=gate_capacity_floor)
        if report_floor:
            _check_internal_floor_id(audit, source="topology_report", floor_id=floor_id, actual=report_floor)
        elif floor_scope_meta:
            _add_audit_entry(
                audit,
                _audit_entry(
                    severity="warning",
                    check="missing_explicit_floor_provenance:topology_report",
                    floor_id=floor_id,
                    status="missing",
                    expected=floor_id,
                    actual="",
                ),
            )
        chain["chain_consistent"] = not any(
            part.get("floor_id_check") == "fail" or part.get("status") == "fail"
            for part in chain.values()
            if isinstance(part, Mapping)
        )
        audit["provenance_chain_by_floor"][floor_id] = chain

        alloc_target = float(allocation_data.get("target_area_sum", 0.0) or 0.0)
        cluster_target = float(floor_scope_meta.get("cluster_target_area_sum", 0.0) or 0.0)
        demand = capacity.get("global_demand") if isinstance(capacity.get("global_demand"), Mapping) else {}
        summary_target = float((demand or {}).get("target_area_sum", 0.0) or 0.0)
        area_status = "missing"
        if allocation_data and cluster_ids and capacity:
            allocation_cluster_close = _area_close(alloc_target, cluster_target)
            cluster_summary_close = _area_close(cluster_target, summary_target)
            if allocation_cluster_close and cluster_summary_close:
                area_status = "pass"
            elif cluster_summary_close:
                area_status = "pass_with_allocation_transform"
                _add_audit_entry(
                    audit,
                    _audit_entry(
                        severity="warning",
                        check="allocation_target_area_differs_from_cluster_target_area",
                        floor_id=floor_id,
                        status="warning",
                        expected={"allocation": alloc_target},
                        actual={"cluster": cluster_target, "summary": summary_target},
                        detail={"interpretation": "likely_budget_or_area_reconciliation_before_geometry"},
                    ),
                )
            else:
                area_status = "fail"
                _add_audit_entry(
                    audit,
                    _audit_entry(
                        severity="error",
                        check="cluster_summary_target_area_consistency",
                        floor_id=floor_id,
                        expected={"cluster": cluster_target},
                        actual={"summary": summary_target},
                    ),
                )
        audit["area_consistency_by_floor"][floor_id] = {
            "status": area_status,
            "allocation_target_area_sum": _round_area(alloc_target),
            "cluster_target_area_sum": _round_area(cluster_target),
            "summary_target_area_sum": _round_area(summary_target),
            "delta_allocation_vs_cluster": _round_area(alloc_target - cluster_target),
            "delta_cluster_vs_summary": _round_area(cluster_target - summary_target),
        }

        alloc_count = int(allocation_data.get("room_count", 0) or 0)
        cluster_count = int(floor_scope_meta.get("cluster_room_count", 0) or 0)
        summary_count = int(capacity.get("total_room_count", 0) or 0)
        count_status = "missing"
        if allocation_data and cluster_ids and capacity:
            count_status = "pass" if alloc_count == cluster_count == summary_count else "fail"
            if count_status == "fail":
                _add_audit_entry(
                    audit,
                    _audit_entry(
                        severity="error",
                        check="allocation_cluster_summary_room_count_consistency",
                        floor_id=floor_id,
                        expected={"allocation": alloc_count, "cluster": cluster_count},
                        actual={"summary": summary_count},
                    ),
                )
        audit["room_count_consistency_by_floor"][floor_id] = {
            "status": count_status,
            "allocation_room_count": alloc_count,
            "cluster_room_count": cluster_count,
            "summary_total_room_count": summary_count,
        }

        actual_capacity = float(floor_scope_meta.get("actual_island_area_sum", 0.0) or 0.0)
        effective_capacity = float(floor_scope_meta.get("effective_capacity_sum", 0.0) or 0.0)
        if capacity:
            variants = ((capacity.get("per_variant_capacity_summary") or {}).get("items") or []) if isinstance(capacity.get("per_variant_capacity_summary"), Mapping) else []
            valid_variants = [v for v in variants if isinstance(v, Mapping) and bool(v.get("valid"))]
            if valid_variants:
                actual_capacity = max(float(v.get("total_actual_island_area", 0.0) or 0.0) for v in valid_variants)
                effective_capacity = max(float(v.get("total_effective_capacity", 0.0) or 0.0) for v in valid_variants)
        audit["capacity_provenance_by_floor"][floor_id] = {
            "actual_island_area_sum": _round_area(actual_capacity),
            "effective_capacity_sum": _round_area(effective_capacity),
            "capacity_source": "topology_assignment_capacity_summary" if capacity else str(floor_scope_meta.get("capacity_source") or "missing"),
        }

        core_hash = str(floor_scope_meta.get("core_union_hash") or "")
        audit["core_provenance"]["core_union_hash_by_floor"][floor_id] = core_hash
        if floor_scope_meta and not core_hash:
            _add_audit_entry(
                audit,
                _audit_entry(
                    severity="warning",
                    check="missing_core_union_hash_provenance",
                    floor_id=floor_id,
                    status="missing",
                ),
            )

    for cluster_id, owners in sorted(raw_cluster_owner.items()):
        if len(set(owners)) > 1:
            _add_audit_entry(
                audit,
                _audit_entry(
                    severity="info",
                    check="duplicate_raw_cluster_id_across_floors",
                    status="info",
                    actual=cluster_id,
                    detail={"floor_ids": sorted(set(owners))},
                ),
            )
    for canonical_id, owners in sorted(canonical_clusters.items()):
        if len(owners) > 1:
            _add_audit_entry(
                audit,
                _audit_entry(
                    severity="error",
                    check="duplicate_canonical_cluster_id",
                    status="fail",
                    actual=canonical_id,
                    detail={"floor_ids": owners},
                ),
            )

    f2_diag = floors.get("F2") if isinstance(floors, Mapping) else {}
    f2_capacity = f2_diag.get("capacity_conflict_summary") if isinstance(f2_diag, Mapping) and isinstance(f2_diag.get("capacity_conflict_summary"), Mapping) else {}
    f2_present = bool(f2_capacity)
    f2_area = audit["area_consistency_by_floor"].get("F2", {})
    f2_count = audit["room_count_consistency_by_floor"].get("F2", {})
    f2_chain = audit["provenance_chain_by_floor"].get("F2", {})
    f2_adoption = f2_diag.get("topology_assignment_adoption") if isinstance(f2_diag, Mapping) else {}
    f2_adoption = f2_adoption if isinstance(f2_adoption, Mapping) else {}
    f2_gate = f2_adoption.get("adoption_gate") if isinstance(f2_adoption.get("adoption_gate"), Mapping) else {}
    f2_summary_floor = str(f2_capacity.get("floor_id") or "")
    f2_gate_summary = f2_gate.get("capacity_conflict_summary") if isinstance(f2_gate, Mapping) and isinstance(f2_gate.get("capacity_conflict_summary"), Mapping) else {}
    proposal_summary_gate_consistent = bool(
        f2_capacity
        and (not f2_gate_summary or str(f2_gate_summary.get("floor_id") or "") == f2_summary_floor)
        and (
            not f2_gate_summary
            or ((f2_gate_summary.get("diagnosis") or {}).get("primary_conflict_type") if isinstance(f2_gate_summary.get("diagnosis"), Mapping) else "")
            == ((f2_capacity.get("diagnosis") or {}).get("primary_conflict_type") if isinstance(f2_capacity.get("diagnosis"), Mapping) else "")
        )
    )
    missing_evidence: List[str] = []
    if not f2_summary_floor:
        missing_evidence.append("capacity_summary.floor_id")
    if f2_area.get("status") == "missing":
        missing_evidence.append("area_consistency")
    if f2_count.get("status") == "missing":
        missing_evidence.append("room_count_consistency")
    evidence = {
        "floor_id_matches": bool(f2_summary_floor == "F2"),
        "room_count_matches_f2_allocation": bool(f2_count.get("status") == "pass"),
        "area_sum_matches_f2_allocation": bool(f2_area.get("status") in {"pass", "pass_with_allocation_transform"}),
        "allocation_target_area_transform_detected": bool(f2_area.get("status") == "pass_with_allocation_transform"),
        "cluster_scope_consistent": not any(
            v.get("severity") == "error" and str(v.get("check", "")).startswith("duplicate_canonical_cluster")
            for v in audit["violations"]
        ),
        "proposal_summary_gate_consistent": proposal_summary_gate_consistent,
        "sufficient_metadata": not missing_evidence,
    }
    required_f2_evidence = [
        "floor_id_matches",
        "room_count_matches_f2_allocation",
        "area_sum_matches_f2_allocation",
        "cluster_scope_consistent",
        "proposal_summary_gate_consistent",
        "sufficient_metadata",
    ]
    if not f2_present:
        verdict = "insufficient_metadata"
    elif any(v.get("severity") == "error" and v.get("floor_id") == "F2" for v in audit["violations"]):
        verdict = "suspect_floor_scope"
    elif all(bool(evidence.get(key)) for key in required_f2_evidence):
        verdict = "confirmed_f2"
    elif not evidence["sufficient_metadata"]:
        verdict = "insufficient_metadata"
    else:
        verdict = "suspect_floor_scope"
    audit["f2_capacity_conflict_provenance"] = {
        "present": f2_present,
        "verdict": verdict,
        "evidence": evidence,
        "blocking_evidence_missing": missing_evidence,
        "primary_conflict_type": (
            ((f2_capacity.get("diagnosis") or {}).get("primary_conflict_type") if isinstance(f2_capacity.get("diagnosis"), Mapping) else "")
            if f2_capacity
            else ""
        ),
        "chain_consistent": bool(f2_chain.get("chain_consistent", False)) if f2_chain else False,
    }
    audit["floor_scope_consistent"] = not any(v.get("severity") == "error" for v in audit["violations"])
    return _json_safe(audit)


def _floor_scope_audit_comparison(dry_run: Mapping[str, Any], adoption: Mapping[str, Any]) -> Dict[str, Any]:
    dry = dry_run.get("floor_scope_audit") if isinstance(dry_run.get("floor_scope_audit"), Mapping) else {}
    adopt = adoption.get("floor_scope_audit") if isinstance(adoption.get("floor_scope_audit"), Mapping) else {}
    dry_attr = dry.get("failure_attribution") if isinstance(dry.get("failure_attribution"), Mapping) else {}
    adopt_attr = adopt.get("failure_attribution") if isinstance(adopt.get("failure_attribution"), Mapping) else {}
    dry_floor = str(dry_attr.get("first_failed_floor_id") or "")
    adopt_floor = str(adopt_attr.get("first_failed_floor_id") or "")
    return _json_safe(
        {
            "audit_version": FLOOR_SCOPE_AUDIT_VERSION,
            "analysis_only": True,
            "dry_run_floor_scope_consistent": bool(dry.get("floor_scope_consistent", True)),
            "adoption_floor_scope_consistent": bool(adopt.get("floor_scope_consistent", True)),
            "failure_floor_changed": bool(dry_floor and adopt_floor and dry_floor != adopt_floor),
            "dry_run_first_failed_floor_id": dry_floor,
            "adoption_first_failed_floor_id": adopt_floor,
            "dry_run_f2_capacity_conflict_verdict": str(((dry.get("f2_capacity_conflict_provenance") or {}).get("verdict")) or ""),
            "adoption_f2_capacity_conflict_verdict": str(((adopt.get("f2_capacity_conflict_provenance") or {}).get("verdict")) or ""),
            "dry_run_error_count": len(list(dry.get("violations") or [])),
            "adoption_error_count": len(list(adopt.get("violations") or [])),
        }
    )


def _adoption_invariants_for_floor(
    *,
    mode: str,
    proposal: Mapping[str, Any],
    adoption: Mapping[str, Any],
    runtime_variant_id: str,
    handoff_variant_id: str,
    coverage_debt: Mapping[str, Any],
) -> Dict[str, Any]:
    violations: List[str] = []
    proposal_status = str(proposal.get("status", "") or "")
    selected_variant_id = _proposal_selected_variant(proposal)
    gate = adoption.get("adoption_gate") if isinstance(adoption.get("adoption_gate"), Mapping) else {}
    gate_opened = bool(gate.get("gate_opened")) if gate else False
    gate_block_reason = str(gate.get("gate_block_reason") or "") if gate else ""

    if mode == "dry_run":
        if bool(proposal.get("used_for_main_path")):
            violations.append("dry_run proposal used_for_main_path=True")
        if bool(adoption.get("applied")):
            violations.append("dry_run adoption applied=True")
        if gate and gate_opened:
            violations.append("dry_run adoption gate opened")
    elif mode == "adoption" and gate_opened:
        if not bool(proposal.get("used_for_main_path")):
            violations.append("adoption proposal used_for_main_path=False")
        if not bool(adoption.get("used_for_main_path")):
            violations.append("adoption record used_for_main_path=False")
        if selected_variant_id and runtime_variant_id != selected_variant_id:
            violations.append("runtime_topology_variant_id != proposal.selected_variant_id")
        if selected_variant_id and handoff_variant_id and handoff_variant_id != selected_variant_id:
            violations.append("handoff_variant_id != proposal.selected_variant_id")
        if bool(adoption.get("applied")) and not bool(adoption.get("object_identity_verified")):
            violations.append("object_identity_verified is false")
        if bool(coverage_debt.get("available")) and selected_variant_id:
            variant_ids = set(str(x) for x in coverage_debt.get("variant_ids", []) if str(x))
            if variant_ids and selected_variant_id not in variant_ids:
                violations.append("coverage_debt variant_id does not match selected variant")
            sources = set(str(x) for x in coverage_debt.get("island_area_sources", []) if str(x))
            if sources and "adopted_variant_polygon" not in sources:
                violations.append("coverage_debt island_area_source is not adopted_variant_polygon")
    elif mode == "adoption" and proposal_status == "success" and not gate:
        violations.append("adoption gate missing for successful proposal")

    if gate and not gate_opened and not gate_block_reason:
        violations.append("adoption gate closed without gate_block_reason")

    return {"passed": not violations, "violations": violations}


def _extract_floor_diagnostics(
    floor_id: str,
    layout_or_metadata: Any,
    *,
    mode: str,
    stage: str = "",
    error_type: str = "",
    exc_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata: Mapping[str, Any]
    if isinstance(layout_or_metadata, Mapping):
        metadata = layout_or_metadata
    else:
        metadata = getattr(layout_or_metadata, "solver_metadata", {}) or {}

    grid = metadata.get("grid_growth") if isinstance(metadata.get("grid_growth"), Mapping) else {}
    proposal = grid.get("topology_assignment_proposal") if isinstance(grid.get("topology_assignment_proposal"), Mapping) else {}
    adoption = grid.get("topology_assignment_adoption") if isinstance(grid.get("topology_assignment_adoption"), Mapping) else {}
    runtime_variant_id = str(grid.get("runtime_topology_variant_id") or adoption.get("runtime_topology_variant_id") or "")
    handoff_variant_id = str(grid.get("handoff_variant_id") or runtime_variant_id or "")
    if not handoff_variant_id:
        handoff = grid.get("handoff")
        if isinstance(handoff, Sequence) and handoff:
            first = handoff[0]
            if isinstance(first, Mapping):
                handoff_variant_id = str(first.get("variant_id") or "")
    floor_scope_metadata = _compact_floor_scope_metadata(floor_id, metadata, grid)
    coverage_debt = _find_coverage_debt(metadata)
    failed_room_id = ""
    failed_cluster_id = ""
    if isinstance(exc_metadata, Mapping):
        failed_room_id = str(exc_metadata.get("failed_room_id") or exc_metadata.get("room_id") or "")
        failed_cluster_id = str(exc_metadata.get("failed_cluster_id") or exc_metadata.get("cluster_id") or "")
    failed_cluster_diag = _find_failed_cluster(proposal, failed_room_id)
    if not failed_cluster_id and isinstance(failed_cluster_diag, Mapping):
        failed_cluster_id = str(failed_cluster_diag.get("failed_cluster_id") or "")

    core_overlap = _find_core_overlap(metadata, exc_metadata)
    capacity_summary = _capacity_summary_from(proposal, adoption)
    capacity_area_allocation = (
        grid.get("capacity_aware_area_allocation")
        if isinstance(grid.get("capacity_aware_area_allocation"), Mapping)
        else {}
    )
    capacity_diagnosis = capacity_summary.get("diagnosis") if isinstance(capacity_summary.get("diagnosis"), Mapping) else {}
    relaxation_ladder = capacity_summary.get("relaxation_ladder") if isinstance(capacity_summary.get("relaxation_ladder"), Mapping) else {}
    circulation_contract = build_circulation_contract(
        floor_id=floor_id,
        grid=grid,
        core_overlap_diagnostics=core_overlap,
    )
    island_provenance = build_island_cluster_provenance(
        floor_id=floor_id,
        grid=grid,
        capacity_summary=capacity_summary,
    )
    blocker_explanation = build_island_capacity_blocker_explanation(
        floor_id=floor_id,
        primary_conflict_type=str(capacity_diagnosis.get("primary_conflict_type") or ""),
        provenance=island_provenance,
        capacity_summary=capacity_summary,
    )
    semantic_variants = grid.get("semantic_seeded_topology_variants")
    semantic_provenance: Dict[str, Any] = {}
    semantic_blocker: Dict[str, Any] = {}
    if isinstance(semantic_variants, Sequence) and not isinstance(semantic_variants, (str, bytes, bytearray)) and semantic_variants:
        semantic_grid = dict(grid)
        semantic_grid["topology_variants"] = list(semantic_variants)
        if isinstance(grid.get("semantic_seeded_candidate_island_metadata"), Sequence):
            semantic_grid["candidate_island_metadata"] = list(grid.get("semantic_seeded_candidate_island_metadata") or [])
        if isinstance(grid.get("semantic_seeded_cluster_feasibility_summary"), Mapping):
            semantic_grid["cluster_feasibility_summary"] = dict(grid.get("semantic_seeded_cluster_feasibility_summary") or {})
        if isinstance(grid.get("semantic_seeded_assignment_proposal"), Mapping):
            semantic_grid["topology_assignment_proposal"] = dict(grid.get("semantic_seeded_assignment_proposal") or {})
        semantic_provenance = build_island_cluster_provenance(
            floor_id=floor_id,
            grid=semantic_grid,
            capacity_summary=capacity_summary,
            analysis_target_kind="semantic_seeded_dry_run",
        )
        semantic_blocker = build_island_capacity_blocker_explanation(
            floor_id=floor_id,
            primary_conflict_type=str(capacity_diagnosis.get("primary_conflict_type") or ""),
            provenance=semantic_provenance,
            capacity_summary=capacity_summary,
        )
    invariants = _adoption_invariants_for_floor(
        mode=mode,
        proposal=proposal,
        adoption=adoption,
        runtime_variant_id=runtime_variant_id,
        handoff_variant_id=handoff_variant_id,
        coverage_debt=coverage_debt,
    )
    return _json_safe(
        {
            "stage": stage,
            "error_type": error_type,
            "topology_assignment_proposal": proposal,
            "topology_assignment_proposal_before_area_allocation": grid.get("topology_assignment_proposal_before_area_allocation")
            if isinstance(grid.get("topology_assignment_proposal_before_area_allocation"), Mapping)
            else {},
            "topology_assignment_proposal_after_area_allocation": grid.get("topology_assignment_proposal_after_area_allocation")
            if isinstance(grid.get("topology_assignment_proposal_after_area_allocation"), Mapping)
            else {},
            "topology_assignment_adoption": adoption,
            "runtime_topology_variant_id": runtime_variant_id,
            "handoff_variant_id": handoff_variant_id,
            "coverage_debt_variant_id": coverage_debt.get("variant_id", ""),
            "coverage_debt": coverage_debt,
            "failed_room_id": failed_room_id,
            "failed_cluster_id": failed_cluster_id,
            "failed_cluster_diagnostics": failed_cluster_diag,
            "core_overlap_diagnostics": core_overlap,
            "capacity_conflict_summary": capacity_summary,
            "primary_conflict_type": str(capacity_diagnosis.get("primary_conflict_type") or ""),
            "capacity_conflict_next_action_hint": str(capacity_diagnosis.get("next_action_hint") or ""),
            "relaxation_ladder": relaxation_ladder,
            "capacity_aware_area_allocation": capacity_area_allocation,
            "area_allocation_status": str(capacity_area_allocation.get("status") or ""),
            "area_compression_applied": bool(capacity_area_allocation.get("area_compression_applied", False)),
            "geometry_target_area_sum": capacity_area_allocation.get("geometry_target_area_sum"),
            "preferred_target_area_sum": capacity_area_allocation.get("preferred_target_area_sum"),
            "capacity_budget": capacity_area_allocation.get("allocation_capacity_budget_effective"),
            "active_proposal_source": str(grid.get("active_proposal_source") or ""),
            "active_target_hash": str(grid.get("active_target_hash") or ""),
            "adoption_invariants": invariants,
            "floor_scope_metadata": floor_scope_metadata,
            "circulation_territory_contract": circulation_contract,
            "island_cluster_provenance": island_provenance,
            "island_capacity_blocker_explanation": blocker_explanation,
            "semantic_seeded_territory_diagnostics": grid.get("semantic_seeded_territory_diagnostics")
            if isinstance(grid.get("semantic_seeded_territory_diagnostics"), Mapping)
            else {},
            "semantic_seeded_assignment_proposal": grid.get("semantic_seeded_assignment_proposal")
            if isinstance(grid.get("semantic_seeded_assignment_proposal"), Mapping)
            else {},
            "semantic_seeded_comparison": grid.get("semantic_seeded_comparison")
            if isinstance(grid.get("semantic_seeded_comparison"), Mapping)
            else {},
            "semantic_seeded_island_cluster_provenance": semantic_provenance,
            "semantic_seeded_island_capacity_blocker_explanation": semantic_blocker,
        }
    )


def _failure_stage(exc: BaseException) -> str:
    stage = str(getattr(exc, "stage", "") or "")
    metadata = getattr(exc, "metadata", None)
    if not stage and isinstance(metadata, Mapping):
        stage = str(metadata.get("stage") or "")
    return stage or type(exc).__name__


def _failure_floor(exc: BaseException) -> str:
    floor_id = str(getattr(exc, "floor_id", "") or "")
    metadata = getattr(exc, "metadata", None)
    if not floor_id and isinstance(metadata, Mapping):
        floor_id = str(metadata.get("floor_id") or "")
    return floor_id


def _is_typed_geometry_failure(exc: BaseException) -> bool:
    return isinstance(exc, (LayoutCoverageError, LayoutTopologyError, LayoutGeometryInvariantError))


def _semantic_repair_allowed(exc: BaseException) -> bool:
    metadata = getattr(exc, "metadata", None)
    if isinstance(metadata, Mapping) and "semantic_repair_allowed" in metadata:
        return bool(metadata.get("semantic_repair_allowed"))
    return bool(getattr(exc, "semantic_repair_allowed", False))


def _run_mode(
    *,
    mode: str,
    allocation: BuildingAllocation,
    inputs: GeometrySmokeInputs,
    forbid_llm: bool,
    allocation_source: Mapping[str, Any],
) -> Dict[str, Any]:
    mode_run_id = _short_uuid()
    context_ids = {
        "mode_run_id": mode_run_id,
        "allocation_context_id": _short_uuid(),
        "solver_config_context_id": _short_uuid(),
        "orchestrator_context_id": _short_uuid(),
        "core_context_id": _short_uuid(),
    }
    copied_allocation = _copy_allocation(allocation, inputs.floors)
    config = _make_solver_config(mode, inputs)
    orchestrator, core = _make_orchestrator(inputs, config)

    llm_guard = LLMGuard(enabled=forbid_llm)
    base: Dict[str, Any] = {
        "mode": mode,
        **context_ids,
        "allocation_source": dict(allocation_source),
        "result": "unhandled_exception",
        "error_type": "",
        "stage": "",
        "floor_id": "",
        "semantic_repair_allowed": False,
        "floors": {},
        "llm_guard": {},
        "acceptable": False,
        "core_bounds": [_round_float(v) for v in getattr(core, "polygon", box(0, 0, 0, 0)).bounds],
    }
    try:
        with llm_guard:
            building_result: BuildingResult = orchestrator.generate(copied_allocation)
        floors: Dict[str, Any] = {}
        for fid, layout in (building_result.floor_layouts or {}).items():
            floors[str(fid)] = _extract_floor_diagnostics(str(fid), layout, mode=mode)
        base.update(
            {
                "result": "success",
                "error_type": "",
                "stage": "",
                "floor_id": "",
                "semantic_repair_allowed": False,
                "floors": floors,
                "warnings": list(getattr(building_result, "warnings", []) or []),
            }
        )
    except BaseException as exc:
        if isinstance(exc, LLMCallForbidden):
            base.update(
                {
                    "result": "unhandled_exception",
                    "error_type": type(exc).__name__,
                    "stage": "llm_call_forbidden",
                    "traceback_summary": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                }
            )
        elif _is_typed_geometry_failure(exc):
            exc_metadata = getattr(exc, "metadata", {}) if isinstance(getattr(exc, "metadata", {}), Mapping) else {}
            floor_id = _failure_floor(exc)
            floor_diag = _extract_floor_diagnostics(
                floor_id or "unknown",
                exc_metadata.get("solver_metadata") if isinstance(exc_metadata.get("solver_metadata"), Mapping) else exc_metadata,
                mode=mode,
                stage=_failure_stage(exc),
                error_type=type(exc).__name__,
                exc_metadata=exc_metadata,
            )
            base.update(
                {
                    "result": "typed_failure",
                    "error_type": type(exc).__name__,
                    "stage": _failure_stage(exc),
                    "floor_id": floor_id,
                    "semantic_repair_allowed": _semantic_repair_allowed(exc),
                    "floors": {floor_id or "unknown": floor_diag},
                    "error_message": str(exc),
                }
            )
        else:
            base.update(
                {
                    "result": "unhandled_exception",
                    "error_type": type(exc).__name__,
                    "stage": type(exc).__name__,
                    "traceback_summary": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-8:]),
                    "error_message": str(exc),
                }
            )
    finally:
        base["llm_guard"] = llm_guard.to_dict()

    base["floor_scope_audit"] = _build_floor_scope_audit(base, copied_allocation, mode=mode)
    _annotate_acceptability(base, DEFAULT_ACCEPTABLE_FAILURE_STAGES)
    return _json_safe(base)


def _iter_floor_diagnostics(run: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    floors = run.get("floors")
    if isinstance(floors, Mapping):
        for diag in floors.values():
            if isinstance(diag, Mapping):
                yield diag


def _summary_for_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    first_failure_stage = str(run.get("stage") or "")
    first_failed_floor_id = str(run.get("floor_id") or "")
    core_regression = any(
        bool(((diag.get("core_overlap_diagnostics") or {}).get("core_overlap_regression")))
        for diag in _iter_floor_diagnostics(run)
    )
    invariant_violations = [
        v
        for diag in _iter_floor_diagnostics(run)
        for v in list(((diag.get("adoption_invariants") or {}).get("violations")) or [])
    ]
    llm_calls = int(((run.get("llm_guard") or {}).get("calls_attempted")) or 0)
    return {
        "first_failure_stage": first_failure_stage,
        "first_failed_floor_id": first_failed_floor_id,
        "whole_building_valid": str(run.get("result")) == "success",
        "adoption_invariant_pass": not invariant_violations,
        "adoption_invariant_violation_count": len(invariant_violations),
        "core_overlap_regression": bool(core_regression),
        "llm_calls_attempted": llm_calls,
    }


def _building_adoption_gate_for_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    floor_gates: Dict[str, Dict[str, Any]] = {}
    blocking_floor_ids: List[str] = []
    floors = run.get("floors")
    if isinstance(floors, Mapping):
        for floor_id, diag in floors.items():
            if not isinstance(diag, Mapping):
                continue
            adoption = diag.get("topology_assignment_adoption") if isinstance(diag.get("topology_assignment_adoption"), Mapping) else {}
            gate = adoption.get("adoption_gate") if isinstance(adoption.get("adoption_gate"), Mapping) else {}
            opened = bool(gate.get("gate_opened")) if gate else False
            reason = str(gate.get("gate_block_reason") or "") if gate else "adoption_gate_missing"
            floor_gates[str(floor_id)] = {
                "adoption_gate_opened": opened,
                "gate_block_reason": reason,
            }
            if not opened:
                blocking_floor_ids.append(str(floor_id))
    all_opened = bool(floor_gates) and not blocking_floor_ids
    first_reason = ""
    if blocking_floor_ids:
        first_reason = str(floor_gates.get(blocking_floor_ids[0], {}).get("gate_block_reason") or "")
    return {
        "per_floor_gate_enforced": True,
        "building_gate_enforced": False,
        "policy": "all_or_nothing",
        "diagnostic_only": True,
        "all_floors_gate_opened": bool(all_opened),
        "blocking_floor_ids": blocking_floor_ids,
        "building_gate_block_reason": first_reason,
        "floors": floor_gates,
    }


def _annotate_acceptability(
    run: Dict[str, Any],
    acceptable_failure_stages: Iterable[str],
    *,
    strict_floor_scope: bool = False,
    strict_circulation_contract: bool = False,
) -> None:
    acceptable_stages = {str(s) for s in acceptable_failure_stages}
    result = str(run.get("result") or "")
    stage = str(run.get("stage") or "")
    summary = _summary_for_run(run)
    unacceptable_reasons: List[str] = []
    if result == "unhandled_exception":
        unacceptable_reasons.append("unhandled_exception")
    if int(summary.get("llm_calls_attempted", 0) or 0) > 0:
        unacceptable_reasons.append("llm_call_attempted")
    if bool(summary.get("core_overlap_regression")):
        unacceptable_reasons.append("core_overlap_regression")
    if int(summary.get("adoption_invariant_violation_count", 0) or 0) > 0:
        unacceptable_reasons.append("adoption_invariant_violation")
    audit = run.get("floor_scope_audit") if isinstance(run.get("floor_scope_audit"), Mapping) else {}
    if strict_floor_scope and audit and not bool(audit.get("floor_scope_consistent", True)):
        unacceptable_reasons.append("floor_scope_audit_error")
    circulation_errors = [
        violation
        for diag in _iter_floor_diagnostics(run)
        for violation in list(((diag.get("circulation_territory_contract") or {}).get("violations")) or [])
        if isinstance(violation, Mapping) and str(violation.get("severity") or "") == "error"
    ]
    if strict_circulation_contract and circulation_errors:
        unacceptable_reasons.append("circulation_contract_error")
    if result == "typed_failure" and stage not in acceptable_stages:
        unacceptable_reasons.append(f"unacceptable_failure_stage:{stage}")
    summary["floor_scope_consistent"] = bool(audit.get("floor_scope_consistent", True)) if audit else True
    summary["floor_scope_error_count"] = len(list(audit.get("violations") or [])) if audit else 0
    summary["floor_scope_strict_mode"] = bool(strict_floor_scope)
    summary["floor_scope_affects_exit_code"] = bool(strict_floor_scope)
    summary["circulation_contract_error_count"] = len(circulation_errors)
    summary["circulation_contract_strict_mode"] = bool(strict_circulation_contract)
    summary["circulation_contract_affects_exit_code"] = bool(strict_circulation_contract)
    run["summary"] = summary
    run["building_adoption_gate"] = _building_adoption_gate_for_run(run)
    run["acceptable"] = not unacceptable_reasons and result in {"success", "typed_failure"}
    run["unacceptable_reasons"] = unacceptable_reasons


def _first_floor_diag(run: Mapping[str, Any]) -> Mapping[str, Any]:
    for diag in _iter_floor_diagnostics(run):
        return diag
    return {}


def _recommended_next_phase(
    recommendation_by_floor: Mapping[str, str],
    likely_cause_by_floor: Mapping[str, str],
    contract_pass_by_floor: Mapping[str, Any],
) -> Dict[str, str]:
    for floor_id, passed in sorted(contract_pass_by_floor.items()):
        if passed is False:
            return {
                "recommended_next_phase": "circulation_routing_repair",
                "recommendation_confidence": "high",
                "recommendation_reason": f"{floor_id} circulation contract has explicit error evidence",
            }
    priority = [
        "cluster_split_diagnostic",
        "semantic_seeded_territory_variants",
        "circulation_routing_repair",
        "add_missing_metadata",
    ]
    for candidate in priority:
        floors = sorted(fid for fid, rec in recommendation_by_floor.items() if rec == candidate)
        if floors:
            cause = str(likely_cause_by_floor.get(floors[0]) or "")
            confidence = "medium" if candidate != "add_missing_metadata" else "low"
            if candidate == "cluster_split_diagnostic":
                confidence = "high"
            return {
                "recommended_next_phase": candidate,
                "recommendation_confidence": confidence,
                "recommendation_reason": f"{candidate} suggested by {floors[0]} blocker cause {cause or 'unknown'}",
            }
    return {
        "recommended_next_phase": "no_action",
        "recommendation_confidence": "low",
        "recommendation_reason": "no circulation or island capacity blocker evidence available",
    }


def _compare_runs(dry_run: Mapping[str, Any], adoption: Mapping[str, Any]) -> Dict[str, Any]:
    dry_floor = _first_floor_diag(dry_run)
    adopt_floor = _first_floor_diag(adoption)
    dry_stage = str(dry_run.get("stage") or dry_floor.get("stage") or "")
    adoption_stage = str(adoption.get("stage") or adopt_floor.get("stage") or "")
    proposal = _mapping_or_empty(adopt_floor.get("topology_assignment_proposal"))
    adoption_record = _mapping_or_empty(adopt_floor.get("topology_assignment_adoption"))
    adoption_gate = _mapping_or_empty(adoption_record.get("adoption_gate"))
    selected_variant_id = _proposal_selected_variant(proposal)
    coverage_debt = _mapping_or_empty(adopt_floor.get("coverage_debt"))
    coverage_match = True
    if selected_variant_id and isinstance(coverage_debt, Mapping) and coverage_debt.get("available"):
        ids = set(str(x) for x in coverage_debt.get("variant_ids", []) if str(x))
        coverage_match = not ids or selected_variant_id in ids
    capacity_floors: List[str] = []
    capacity_types: Dict[str, str] = {}
    capacity_hints: Dict[str, str] = {}
    capacity_aware_by_floor: Dict[str, Any] = {}
    target_delta_by_floor: Dict[str, Any] = {}
    capacity_aware_effect: Dict[str, Any] = {}
    circulation_pass_by_floor: Dict[str, Any] = {}
    circulation_confidence_by_floor: Dict[str, str] = {}
    provenance_counts_by_floor: Dict[str, Any] = {}
    semantic_match_by_floor: Dict[str, Any] = {}
    blocker_cause_by_floor: Dict[str, str] = {}
    blocking_clusters_by_floor: Dict[str, Any] = {}
    next_phase_by_floor: Dict[str, str] = {}
    semantic_seeded_comparison_by_floor: Dict[str, Any] = {}
    semantic_seeded_recommendation_by_floor: Dict[str, str] = {}
    semantic_seeded_source_islands_by_floor: Dict[str, int] = {}
    floors = adoption.get("floors")
    if isinstance(floors, Mapping):
        for floor_id, diag in floors.items():
            if not isinstance(diag, Mapping):
                continue
            fid = str(floor_id)
            contract = _mapping_or_empty(diag.get("circulation_territory_contract"))
            provenance = _mapping_or_empty(diag.get("island_cluster_provenance"))
            blocker = _mapping_or_empty(diag.get("island_capacity_blocker_explanation"))
            if contract:
                circulation_pass_by_floor[fid] = contract.get("contract_pass")
                circulation_confidence_by_floor[fid] = str(contract.get("contract_confidence") or "")
            if provenance:
                provenance_counts_by_floor[fid] = dict(_mapping_or_empty(provenance.get("provenance_type_counts")))
                semantic_match_by_floor[fid] = _mapping_or_empty(provenance.get("semantic_source_match_rate"))
            if blocker:
                blocker_cause_by_floor[fid] = str(blocker.get("primary_likely_cause") or "")
                blocking_clusters_by_floor[fid] = list(blocker.get("blocking_clusters") or [])
                next_phase_by_floor[fid] = str(blocker.get("next_phase_recommendation") or "")
            semantic_comparison = _mapping_or_empty(diag.get("semantic_seeded_comparison"))
            if semantic_comparison:
                semantic_seeded_comparison_by_floor[fid] = semantic_comparison
                semantic_seeded_recommendation_by_floor[fid] = str(semantic_comparison.get("recommended_next_phase") or "")
                semantic_seeded_source_islands_by_floor[fid] = int(semantic_comparison.get("semantic_source_island_count") or 0)
            summary = _mapping_or_empty(diag.get("capacity_conflict_summary"))
            diagnosis = _mapping_or_empty(summary.get("diagnosis"))
            primary = str(diag.get("primary_conflict_type") or diagnosis.get("primary_conflict_type") or "")
            hint = str(diag.get("capacity_conflict_next_action_hint") or diagnosis.get("next_action_hint") or "")
            if summary:
                capacity_floors.append(str(floor_id))
            if primary:
                capacity_types[str(floor_id)] = primary
            if hint:
                capacity_hints[str(floor_id)] = hint
            allocation = _mapping_or_empty(diag.get("capacity_aware_area_allocation"))
            if allocation:
                before = _mapping_or_empty(diag.get("topology_assignment_proposal_before_area_allocation"))
                after = _mapping_or_empty(diag.get("topology_assignment_proposal_after_area_allocation"))
                before_summary = _mapping_or_empty(before.get("capacity_conflict_summary"))
                before_diag = _mapping_or_empty(before_summary.get("diagnosis"))
                after_summary = _mapping_or_empty(after.get("capacity_conflict_summary"))
                after_diag = _mapping_or_empty(after_summary.get("diagnosis"))
                fid = str(floor_id)
                before_conflict = str(before_diag.get("primary_conflict_type") or "")
                after_conflict = str(after_diag.get("primary_conflict_type") or "")
                global_shortfall_before = before_conflict == "global_area_capacity_shortfall"
                global_shortfall_after = after_conflict == "global_area_capacity_shortfall"
                allocation_status = str(allocation.get("status") or "")
                if global_shortfall_before:
                    global_shortfall_result = "resolved" if not global_shortfall_after else "still_present"
                elif allocation_status == "preferred_within_capacity":
                    global_shortfall_result = "not_applicable_preferred_within_capacity"
                else:
                    global_shortfall_result = "not_applicable_no_global_shortfall_before"
                capacity_aware_by_floor[fid] = {
                    "status": allocation_status,
                    "applied": allocation.get("applied"),
                    "area_compression_applied": allocation.get("area_compression_applied"),
                    "preferred_target_area_sum": allocation.get("preferred_target_area_sum"),
                    "geometry_target_area_sum": allocation.get("geometry_target_area_sum"),
                    "capacity_budget": allocation.get("allocation_capacity_budget_effective"),
                    "area_target_hash_before": allocation.get("area_target_hash_before"),
                    "area_target_hash_after": allocation.get("area_target_hash_after"),
                    "active_proposal_source": diag.get("active_proposal_source"),
                    "active_target_hash": diag.get("active_target_hash"),
                }
                target_delta_by_floor[fid] = {
                    "preferred_minus_geometry": round(float(allocation.get("preferred_target_area_sum", 0.0) or 0.0) - float(allocation.get("geometry_target_area_sum", 0.0) or 0.0), 4),
                    "raw_minus_geometry": round(float(allocation.get("raw_allocation_target_area_sum", 0.0) or 0.0) - float(allocation.get("geometry_target_area_sum", 0.0) or 0.0), 4),
                }
                capacity_aware_effect[fid] = {
                    "before_proposal_status": before.get("status"),
                    "after_proposal_status": after.get("status"),
                    "before_primary_conflict_type": before_conflict,
                    "after_primary_conflict_type": after_conflict,
                    "global_area_capacity_shortfall_present_before": global_shortfall_before,
                    "global_area_capacity_shortfall_present_after": global_shortfall_after,
                    "global_area_capacity_shortfall_resolved": global_shortfall_before and not global_shortfall_after,
                    "global_area_capacity_shortfall_result": global_shortfall_result,
                    "allocation_status": allocation_status,
                    "target_allocation_applied": bool(allocation.get("applied")),
                    "target_allocation_explanation": allocation.get("message") or allocation.get("status"),
                    "new_blocker_if_any": after_conflict or before_conflict,
                    "gate_changed": bool(_mapping_or_empty(diag.get("adoption_invariants")).get("passed", True))
                    and bool(adoption_gate.get("gate_opened")),
                    "failure_stage_changed": dry_stage != adoption_stage,
                }
    recommendation = _recommended_next_phase(next_phase_by_floor, blocker_cause_by_floor, circulation_pass_by_floor)
    semantic_recs = {
        fid: rec
        for fid, rec in semantic_seeded_recommendation_by_floor.items()
        if rec
    }
    if semantic_recs:
        first_floor = sorted(semantic_recs)[0]
        comparison = _mapping_or_empty(semantic_seeded_comparison_by_floor.get(first_floor))
        recommendation = {
            "recommended_next_phase": semantic_recs[first_floor],
            "recommendation_confidence": str(comparison.get("recommendation_confidence") or "medium"),
            "recommendation_reason": str(comparison.get("recommendation_reason") or f"semantic seeded comparison available for {first_floor}"),
        }
    return _json_safe(
        {
            "adoption_changed_stage": dry_stage != adoption_stage,
            "dry_run_stage": dry_stage,
            "adoption_stage": adoption_stage,
            "dry_run_failed_room_id": str(dry_floor.get("failed_room_id") or ""),
            "adoption_failed_room_id": str(adopt_floor.get("failed_room_id") or ""),
            "dry_run_failed_cluster_id": str(dry_floor.get("failed_cluster_id") or ""),
            "adoption_failed_cluster_id": str(adopt_floor.get("failed_cluster_id") or ""),
            "runtime_variant_changed": str(dry_floor.get("runtime_topology_variant_id") or "")
            != str(adopt_floor.get("runtime_topology_variant_id") or ""),
            "selected_variant_id": selected_variant_id,
            "adoption_used_for_main_path": bool(adoption_record.get("used_for_main_path")),
            "adoption_gate_opened": bool(adoption_gate.get("gate_opened")) if adoption_gate else False,
            "adoption_gate_block_reason": str(adoption_gate.get("gate_block_reason") or "") if adoption_gate else "",
            "building_adoption_gate_opened": bool(((adoption.get("building_adoption_gate") or {}).get("all_floors_gate_opened"))),
            "building_gate_block_reason": str(((adoption.get("building_adoption_gate") or {}).get("building_gate_block_reason")) or ""),
            "blocking_floor_ids": list(((adoption.get("building_adoption_gate") or {}).get("blocking_floor_ids")) or []),
            "floors": dict(((adoption.get("building_adoption_gate") or {}).get("floors")) or {}),
            "capacity_conflict_floors": capacity_floors,
            "capacity_conflict_primary_types_by_floor": capacity_types,
            "capacity_conflict_next_action_hints": capacity_hints,
            "capacity_aware_area_allocation_by_floor": capacity_aware_by_floor,
            "target_sum_delta_by_floor": target_delta_by_floor,
            "capacity_aware_effect": capacity_aware_effect,
            "circulation_contract_pass_by_floor": circulation_pass_by_floor,
            "contract_confidence_by_floor": circulation_confidence_by_floor,
            "island_provenance_type_counts_by_floor": provenance_counts_by_floor,
            "semantic_source_match_rate_by_floor": semantic_match_by_floor,
            "island_capacity_blocker_likely_cause_by_floor": blocker_cause_by_floor,
            "blocking_clusters_by_floor": blocking_clusters_by_floor,
            "next_phase_recommendation_by_floor": next_phase_by_floor,
            "semantic_seeded_comparison_by_floor": semantic_seeded_comparison_by_floor,
            "semantic_seeded_recommendation_by_floor": semantic_seeded_recommendation_by_floor,
            "semantic_seeded_source_islands_by_floor": semantic_seeded_source_islands_by_floor,
            **recommendation,
            "core_overlap_regression": bool(
                (dry_run.get("summary") or {}).get("core_overlap_regression")
                or (adoption.get("summary") or {}).get("core_overlap_regression")
            ),
            "coverage_debt_variant_matches_adoption": bool(coverage_match),
            "floor_scope_audit_comparison": _floor_scope_audit_comparison(dry_run, adoption),
        }
    )


def _mode_isolation_verified(dry_run: Mapping[str, Any], adoption: Mapping[str, Any]) -> bool:
    keys = [
        "mode_run_id",
        "allocation_context_id",
        "solver_config_context_id",
        "orchestrator_context_id",
        "core_context_id",
    ]
    return all(str(dry_run.get(k) or "") and str(dry_run.get(k)) != str(adoption.get(k)) for k in keys)


def run_fixed_allocation_geometry_smoke(
    *,
    llm_log_path: Path | str = DEFAULT_LLM_LOG_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    mode: str = "both",
    geometry_inputs: Optional[GeometrySmokeInputs] = None,
    acceptable_failure_stages: Optional[Iterable[str]] = None,
    forbid_llm: bool = True,
    write_mode_snapshots: bool = False,
    strict_floor_scope: bool = False,
    strict_circulation_contract: bool = False,
) -> Dict[str, Any]:
    mode = str(mode).replace("-", "_")
    if mode not in {"dry_run", "adoption", "both"}:
        raise ValueError(f"Unsupported smoke mode: {mode}")
    inputs = geometry_inputs or GeometrySmokeInputs()
    acceptable = set(acceptable_failure_stages or DEFAULT_ACCEPTABLE_FAILURE_STAGES)
    parse_result = parse_latest_budgeted_allocation(llm_log_path)
    allocation_source = {
        "allocation_source_index": parse_result.allocation_source_index,
        "source_kind": parse_result.source_kind,
        "parse_warnings": list(parse_result.parse_warnings),
    }

    def _single(which: str) -> Dict[str, Any]:
        run = _run_mode(
            mode=which,
            allocation=parse_result.allocation,
            inputs=inputs,
            forbid_llm=forbid_llm,
            allocation_source=allocation_source,
        )
        _annotate_acceptability(
            run,
            acceptable,
            strict_floor_scope=bool(strict_floor_scope),
            strict_circulation_contract=bool(strict_circulation_contract),
        )
        return run

    result: Dict[str, Any]
    if mode == "both":
        dry = _single("dry_run")
        adoption = _single("adoption")
        comparison = _compare_runs(dry, adoption)
        mode_isolated = _mode_isolation_verified(dry, adoption)
        summary = {
            "dry_run": dry.get("summary", {}),
            "adoption": adoption.get("summary", {}),
            "first_failure_stage": adoption.get("summary", {}).get("first_failure_stage")
            or dry.get("summary", {}).get("first_failure_stage", ""),
            "first_failed_floor_id": adoption.get("summary", {}).get("first_failed_floor_id")
            or dry.get("summary", {}).get("first_failed_floor_id", ""),
            "whole_building_valid": bool(dry.get("summary", {}).get("whole_building_valid"))
            and bool(adoption.get("summary", {}).get("whole_building_valid")),
            "adoption_invariant_pass": bool(dry.get("summary", {}).get("adoption_invariant_pass", True))
            and bool(adoption.get("summary", {}).get("adoption_invariant_pass", True)),
            "adoption_invariant_violation_count": int(dry.get("summary", {}).get("adoption_invariant_violation_count", 0) or 0)
            + int(adoption.get("summary", {}).get("adoption_invariant_violation_count", 0) or 0),
            "core_overlap_regression": bool(comparison.get("core_overlap_regression")),
            "llm_calls_attempted": int(dry.get("summary", {}).get("llm_calls_attempted", 0) or 0)
            + int(adoption.get("summary", {}).get("llm_calls_attempted", 0) or 0),
            "floor_scope_consistent": bool(dry.get("summary", {}).get("floor_scope_consistent", True))
            and bool(adoption.get("summary", {}).get("floor_scope_consistent", True)),
            "floor_scope_error_count": int(dry.get("summary", {}).get("floor_scope_error_count", 0) or 0)
            + int(adoption.get("summary", {}).get("floor_scope_error_count", 0) or 0),
            "floor_scope_strict_mode": bool(strict_floor_scope),
            "floor_scope_affects_exit_code": bool(strict_floor_scope),
            "circulation_contract_error_count": int(dry.get("summary", {}).get("circulation_contract_error_count", 0) or 0)
            + int(adoption.get("summary", {}).get("circulation_contract_error_count", 0) or 0),
            "circulation_contract_strict_mode": bool(strict_circulation_contract),
            "circulation_contract_affects_exit_code": bool(strict_circulation_contract),
        }
        acceptable_result = bool(dry.get("acceptable")) and bool(adoption.get("acceptable")) and mode_isolated
        if str(dry.get("result")) == "unhandled_exception" or str(adoption.get("result")) == "unhandled_exception":
            top_result = "unhandled_exception"
        elif str(dry.get("result")) == "typed_failure" or str(adoption.get("result")) == "typed_failure":
            top_result = "typed_failure"
        elif bool(dry.get("summary", {}).get("whole_building_valid")) and bool(adoption.get("summary", {}).get("whole_building_valid")):
            top_result = "success"
        else:
            top_result = "typed_failure"
        primary_run = adoption if adoption.get("result") != "success" else dry
        result = {
            "result": top_result if acceptable_result else "unacceptable_failure",
            "error_type": str(primary_run.get("error_type") or dry.get("error_type") or ""),
            "stage": str(summary.get("first_failure_stage") or ""),
            "floor_id": str(summary.get("first_failed_floor_id") or ""),
            "semantic_repair_allowed": bool(primary_run.get("semantic_repair_allowed", False)),
            "summary": summary,
            "geometry_inputs": inputs.to_dict(),
            "mode_isolation_verified": mode_isolated,
            "floors": primary_run.get("floors", {}),
            "dry_run": dry,
            "adoption": adoption,
            "comparison": comparison,
            "acceptable": acceptable_result,
            "acceptable_failure_stages": sorted(acceptable),
        }
    else:
        single = _single(mode)
        result = {
            **single,
            "geometry_inputs": inputs.to_dict(),
            "mode_isolation_verified": True,
            "acceptable_failure_stages": sorted(acceptable),
        }

    result = _json_safe(result)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        result["acceptable"] = False
        result["result"] = "unhandled_exception"
        result["json_write_error"] = f"{type(exc).__name__}: {exc}"
        raise

    if write_mode_snapshots and mode == "both":
        stem = out.with_suffix("")
        (stem.parent / f"{stem.name}_dry_run.json").write_text(
            json.dumps(result.get("dry_run", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (stem.parent / f"{stem.name}_adoption.json").write_text(
            json.dumps(result.get("adoption", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (stem.parent / f"{stem.name}_comparison.json").write_text(
            json.dumps(result.get("comparison", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-allocation geometry-only smoke.")
    parser.add_argument("--mode", choices=["dry-run", "adoption", "both"], default="both")
    parser.add_argument("--llm-log", default=str(DEFAULT_LLM_LOG_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--floor-width", type=float, default=15.0)
    parser.add_argument("--floor-height", type=float, default=10.0)
    parser.add_argument("--floors", type=int, default=2)
    parser.add_argument("--corridor-width", type=float, default=1.8)
    parser.add_argument("--core-area-ratio", type=float, default=0.12)
    parser.add_argument("--core-placement", default="east")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--acceptable-failure-stage", action="append", default=None)
    parser.add_argument("--allow-llm", action="store_true", help="Disable the default no-LLM guard.")
    parser.add_argument("--write-mode-snapshots", action="store_true")
    parser.add_argument("--strict-floor-scope", action="store_true", help="Fail the smoke when floor provenance audit has error-level violations.")
    parser.add_argument(
        "--strict-circulation-contract",
        action="store_true",
        help="Fail the smoke only when circulation contract diagnostics contain explicit error evidence.",
    )
    parser.add_argument("--enable-capacity-aware-area-allocation", action="store_true")
    parser.add_argument("--apply-capacity-aware-area-allocation", action="store_true")
    parser.add_argument("--enable-semantic-seeded-territory-variants", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    inputs = GeometrySmokeInputs(
        floors=int(args.floors),
        floor_width=float(args.floor_width),
        floor_height=float(args.floor_height),
        corridor_width=float(args.corridor_width),
        core_area_ratio=float(args.core_area_ratio),
        core_placement=str(args.core_placement),
        base_seed=int(args.base_seed),
        enable_capacity_aware_area_allocation=bool(args.enable_capacity_aware_area_allocation),
        apply_capacity_aware_area_allocation=bool(args.apply_capacity_aware_area_allocation),
        enable_semantic_seeded_territory_variants=bool(args.enable_semantic_seeded_territory_variants),
    )
    try:
        result = run_fixed_allocation_geometry_smoke(
            llm_log_path=args.llm_log,
            out_path=args.out,
            mode=str(args.mode).replace("-", "_"),
            geometry_inputs=inputs,
            acceptable_failure_stages=args.acceptable_failure_stage,
            forbid_llm=not bool(args.allow_llm),
            write_mode_snapshots=bool(args.write_mode_snapshots),
            strict_floor_scope=bool(args.strict_floor_scope),
            strict_circulation_contract=bool(args.strict_circulation_contract),
        )
    except Exception as exc:
        print(json.dumps({"result": "unhandled_exception", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "result": result.get("result"),
                "acceptable": result.get("acceptable"),
                "summary": result.get("summary"),
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    return 0 if bool(result.get("acceptable", False)) else 1


__all__ = [
    "GeometrySmokeInputs",
    "LLMGuard",
    "parse_latest_budgeted_allocation",
    "run_fixed_allocation_geometry_smoke",
    "main",
]

