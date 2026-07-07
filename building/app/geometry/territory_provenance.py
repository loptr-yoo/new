"""Diagnostic-only circulation and island provenance evidence.

This module intentionally works on JSON-like metadata instead of geometry
objects.  The reports are evidence for smoke tests and benchmarks; they must
not influence solver decisions or adoption.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MAX_ITEMS = 20
CORE_OVERLAP_EPSILON = 0.01


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
        return round(value, 6)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in list(value)]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            return str(value)
    return str(value)


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        try:
            return dict(asdict(value))
        except Exception:
            return {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            return dict(result) if isinstance(result, Mapping) else {}
        except Exception:
            return {}
    return {}


def _list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return float(default)


def _round(value: Any) -> float:
    return round(_float(value), 6)


def _ids(value: Any) -> List[str]:
    return sorted({str(v) for v in _list(value) if str(v)})


def _bounded(items: Iterable[Any], *, max_items: int = MAX_ITEMS) -> Dict[str, Any]:
    materialized = list(items)
    return {
        "items": materialized[:max_items],
        "shown_count": min(len(materialized), max_items),
        "total_count": len(materialized),
        "truncated": len(materialized) > max_items,
    }


def _base_report(version: str) -> Dict[str, Any]:
    return {
        "report_version": version,
        "analysis_only": True,
        "used_for_solver_decision": False,
        "used_for_adoption": False,
    }


def _floor_role(floor_id: str) -> str:
    text = str(floor_id or "").strip().upper()
    return "ground_floor" if text in {"F1", "1", "GROUND", "GROUND_FLOOR"} else "upper_floor"


def _status(status: str, evidence_source: str, **extra: Any) -> Dict[str, Any]:
    out = {"status": status, "evidence_source": evidence_source}
    out.update(extra)
    return out


def _corridor_metadata(grid: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = grid.get("corridor_evidence")
    if isinstance(evidence, Mapping):
        return dict(evidence)
    corridor = grid.get("corridor")
    if isinstance(corridor, Mapping):
        return dict(corridor)
    corridors = _list(grid.get("corridors") or grid.get("corridor_skeleton"))
    if corridors:
        area = 0.0
        ids: List[str] = []
        for item in corridors:
            row = _mapping(item)
            if not row:
                continue
            ids.append(str(row.get("id") or row.get("corridor_id") or "corridor"))
            area += _float(row.get("area"))
        return {"ids": ids, "area": area}
    return {}


def _cluster_rows(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = _list(grid.get("cluster_metrics")) or _list(grid.get("clusters"))
    return [_mapping(row) for row in rows if _mapping(row)]


def _cluster_feasibility_summary(grid: Mapping[str, Any]) -> Dict[str, Any]:
    summary = grid.get("cluster_feasibility_summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _cluster_feasibility_items(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    summary = _cluster_feasibility_summary(grid)
    return [_mapping(row) for row in _list(summary.get("items")) if _mapping(row)]


def _handoff_rows(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [_mapping(row) for row in _list(grid.get("handoff")) if _mapping(row)]


def _cluster_id(row: Mapping[str, Any]) -> str:
    return str(row.get("cluster_id") or row.get("id") or "")


def _cluster_room_ids(row: Mapping[str, Any]) -> List[str]:
    return _ids(row.get("room_ids", row.get("rooms", [])))


def _cluster_target(row: Mapping[str, Any]) -> float:
    return _round(row.get("target_area_sum", row.get("target_sum", row.get("target_area", 0.0))))


def _cluster_min(row: Mapping[str, Any]) -> float:
    return _round(row.get("min_area_sum", row.get("min_sum", row.get("min_area", 0.0))))


def _required_clusters(grid: Mapping[str, Any], handoff: Sequence[Mapping[str, Any]]) -> Tuple[List[str], str]:
    clusters = [_cluster_id(row) for row in _cluster_rows(grid) if _cluster_id(row)]
    if clusters:
        return sorted(set(clusters)), "cluster_metrics" if grid.get("cluster_metrics") is not None else "clusters"
    feasibility = _list(grid.get("cluster_island_feasibility"))
    feasibility_ids = [str(_mapping(row).get("cluster_id") or "") for row in feasibility]
    feasibility_ids = [cid for cid in feasibility_ids if cid]
    if feasibility_ids:
        return sorted(set(feasibility_ids)), "feasibility_report"
    assigned: List[str] = []
    for row in handoff:
        assigned.extend(_ids(row.get("clusters", row.get("cluster_ids", []))))
    if assigned:
        return sorted(set(assigned)), "assigned_fallback"
    return [], "missing"


def _truthy_metadata_check(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return _status("pass" if bool(value) else "fail", "metadata")


def build_circulation_contract(
    floor_id: str,
    grid: Mapping[str, Any],
    *,
    core_overlap_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a minimal circulation contract report for one floor.

    Missing graph/anchor/width data lowers confidence but is not an error.
    Explicit overlap or explicit graph contradiction is an error.
    """

    grid = grid if isinstance(grid, Mapping) else {}
    floor_id = str(floor_id or grid.get("floor_id") or "")
    role = _floor_role(floor_id)
    requires_core = True
    requires_exterior = role == "ground_floor"
    corridor = _corridor_metadata(grid)
    handoff = _handoff_rows(grid)
    warnings: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    corridor_area = _round(corridor.get("corridor_area", corridor.get("area", grid.get("corridor_area", 0.0))))
    corridor_present = bool(corridor or corridor_area > 0.0 or grid.get("corridor_present"))

    core_overlap = _round(
        (core_overlap_diagnostics or {}).get(
            "corridor_core_overlap_area",
        grid.get(
            "corridor_core_overlap_area",
            corridor.get("corridor_core_overlap_area", corridor.get("core_overlap_after", corridor.get("core_overlap_area", 0.0))),
        ),
        )
    )
    if core_overlap > CORE_OVERLAP_EPSILON:
        violations.append(
            {
                "severity": "error",
                "check": "corridor_core_overlap",
                "status": "fail",
                "actual": core_overlap,
                "expected": f"<= {CORE_OVERLAP_EPSILON}",
            }
        )

    evidence_source = str(corridor.get("evidence_source") or "metadata")
    touches_core = _truthy_metadata_check(corridor.get("touches_core", grid.get("corridor_touches_core")))
    if touches_core is None:
        touches_core = _status("missing", "missing")
        warnings.append({"severity": "warning", "check": "missing_core_touch_metadata", "status": "missing"})
    elif touches_core.get("evidence_source") == "metadata":
        touches_core["evidence_source"] = evidence_source
    elif requires_core and touches_core["status"] == "fail":
        violations.append({"severity": "error", "check": "corridor_touches_core", "status": "fail"})

    touches_exterior = _truthy_metadata_check(corridor.get("touches_exterior", grid.get("corridor_touches_exterior")))
    if touches_exterior is None:
        touches_exterior = _status("missing", "missing" if requires_exterior else "not_applicable")
        if requires_exterior:
            warnings.append({"severity": "warning", "check": "missing_exterior_touch_metadata", "status": "missing"})
    elif requires_exterior and touches_exterior["status"] == "fail":
        violations.append({"severity": "error", "check": "corridor_touches_exterior", "status": "fail"})
    elif touches_exterior.get("evidence_source") == "metadata":
        touches_exterior["evidence_source"] = evidence_source
    if not requires_exterior and touches_exterior["status"] == "missing":
        touches_exterior = _status("not_applicable", "not_applicable")

    corridor_connected = corridor.get("graph_connected", grid.get("corridor_graph_connected"))
    graph_disconnected = corridor_connected is False
    if graph_disconnected:
        violations.append({"severity": "error", "check": "corridor_graph_connected", "status": "fail", "evidence_source": "graph"})

    assigned_island_ids = sorted({str(row.get("island_id") or "") for row in handoff if str(row.get("island_id") or "")})
    if not assigned_island_ids:
        assigned_island_ids = _ids(corridor.get("connected_island_ids", []))
    explicit_unconnected = _ids(corridor.get("unconnected_island_ids", grid.get("unconnected_island_ids", [])))
    explicit_connected = _ids(corridor.get("connected_island_ids", grid.get("connected_island_ids", [])))
    if not assigned_island_ids:
        connected_check = _status("missing", "missing")
        connected_island_ids: List[str] = []
        unconnected_island_ids: List[str] = []
        warnings.append({"severity": "warning", "check": "assigned_island_connectivity_missing_assignment", "status": "missing"})
    elif explicit_unconnected:
        connected_check = _status("fail", "metadata")
        connected_island_ids = [iid for iid in assigned_island_ids if iid not in set(explicit_unconnected)]
        unconnected_island_ids = explicit_unconnected
        violations.append({"severity": "error", "check": "assigned_islands_unconnected", "status": "fail", "actual": explicit_unconnected})
    elif explicit_connected:
        missing = [iid for iid in assigned_island_ids if iid not in set(explicit_connected)]
        connected_check = _status("fail" if missing else "pass", "metadata")
        connected_island_ids = [iid for iid in assigned_island_ids if iid in set(explicit_connected)]
        unconnected_island_ids = missing
        if missing:
            violations.append({"severity": "error", "check": "assigned_islands_unconnected", "status": "fail", "actual": missing})
    elif corridor_present:
        connected_check = _status("pass", "metadata")
        connected_island_ids = list(assigned_island_ids)
        unconnected_island_ids = []
        warnings.append({"severity": "warning", "check": "assigned_island_connectivity_uses_metadata_assumption", "status": "missing"})
    else:
        connected_check = _status("missing", "missing")
        connected_island_ids = []
        unconnected_island_ids = list(assigned_island_ids)
        warnings.append({"severity": "warning", "check": "corridor_missing_for_assignment_connectivity", "status": "missing"})

    required_cluster_ids, required_source = _required_clusters(grid, handoff)
    confidence = "high"
    if any(item.get("status") == "missing" for item in [touches_core, touches_exterior, connected_check]):
        confidence = "low"
    elif warnings or any(item.get("evidence_source") == "metadata" for item in [touches_core, touches_exterior, connected_check]):
        confidence = "medium"
    if graph_disconnected:
        confidence = "low"

    report = {
        **_base_report("circulation_territory_contract_v1"),
        "floor_id": floor_id,
        "floor_role": role,
        "requires_core": requires_core,
        "requires_exterior": requires_exterior,
        "corridor_present": corridor_present,
        "touches_core": touches_core,
        "touches_exterior": touches_exterior,
        "connects_to_assigned_islands": connected_check,
        "connected_island_ids": connected_island_ids,
        "unconnected_island_ids": unconnected_island_ids,
        "required_cluster_ids": required_cluster_ids,
        "required_cluster_source": required_source,
        "core_overlap_area": core_overlap,
        "corridor_area": corridor_area,
        "contract_pass": not any(v.get("severity") == "error" for v in violations),
        "contract_confidence": confidence,
        "violations": violations,
        "warnings": warnings,
    }
    return _json_safe(report)


def _variant_rows(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    variants = [_mapping(row) for row in _list(grid.get("topology_variants")) if _mapping(row)]
    if variants:
        return variants
    island_metrics = _list(grid.get("island_metrics") or grid.get("candidate_island_metrics"))
    if island_metrics:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in island_metrics:
            row = _mapping(item)
            variant_id = str(row.get("variant_id") or grid.get("runtime_topology_variant_id") or grid.get("handoff_variant_id") or "primary")
            grouped[variant_id].append(row)
        return [{"variant_id": variant_id, "islands": islands, "valid": True} for variant_id, islands in sorted(grouped.items())]
    candidate_rows = _candidate_island_metadata_rows(grid)
    if candidate_rows:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            grouped[str(row.get("variant_id") or "primary")].append(row)
        return [{"variant_id": variant_id, "islands": islands, "valid": True} for variant_id, islands in sorted(grouped.items())]
    return []


def _candidate_island_metadata_rows(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = grid.get("candidate_island_metadata")
    return [_mapping(row) for row in _list(rows) if _mapping(row)]


def _candidate_island_metadata_by_key(grid: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in _candidate_island_metadata_rows(grid):
        variant_id = str(row.get("variant_id") or "")
        island_id = str(row.get("island_id") or row.get("id") or "")
        if variant_id and island_id:
            out[(variant_id, island_id)] = row
    return out


def _island_rows(variant: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = _list(variant.get("islands") or variant.get("candidate_islands") or variant.get("island_metrics"))
    return [_mapping(row) for row in rows if _mapping(row)]


def _island_id(row: Mapping[str, Any]) -> str:
    return str(row.get("island_id") or row.get("id") or "")


def _island_effective_capacity(row: Mapping[str, Any]) -> float:
    return _round(row.get("effective_capacity_area", row.get("capacity_area", row.get("area", 0.0))))


def _source_cluster_ids(row: Mapping[str, Any]) -> List[str]:
    raw = row.get("source_cluster_ids")
    if raw is None:
        raw = row.get("generated_from_cluster_ids")
    ids = _ids(raw)
    seed = str(row.get("seed_cluster_id") or "")
    if seed:
        ids.append(seed)
    return sorted(set(ids))


def _provenance_type(row: Mapping[str, Any], source_cluster_ids: Sequence[str]) -> str:
    explicit = str(row.get("provenance_type") or row.get("source_type") or "").strip()
    allowed = {"semantic_cluster_grown", "corridor_partitioned", "residual_region", "core_cut_residual", "unknown"}
    if explicit in allowed:
        return explicit
    if source_cluster_ids:
        return "semantic_cluster_grown"
    generation_source = str(row.get("island_generation_source") or row.get("generation_source") or "").strip().lower()
    if generation_source in {"usable_polygon_minus_corridor_core", "corridor_partitioned", "corridor_partition"}:
        return "corridor_partitioned"
    if generation_source in {"residual_region", "residual", "gap", "filler"}:
        return "residual_region"
    if generation_source in {"core_cut_residual", "core_cut", "core_exclusion_residual"}:
        return "core_cut_residual"
    text = " ".join(str(row.get(k) or "").lower() for k in ("kind", "origin", "reason", "source"))
    if any(token in text for token in ("residual", "gap", "filler")):
        return "residual_region"
    if "core" in text and any(token in text for token in ("cut", "residual", "exclusion")):
        return "core_cut_residual"
    if "corridor_partitioned" in text or "corridor_partition" in text:
        return "corridor_partitioned"
    return "unknown"


def _cluster_map(grid: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _cluster_rows(grid):
        cid = _cluster_id(row)
        if cid:
            out[cid] = {
                "cluster_id": cid,
                "room_ids": _cluster_room_ids(row),
                "target_sum": _cluster_target(row),
                "min_sum": _cluster_min(row),
            }
    for row in _cluster_feasibility_items(grid):
        cid = str(row.get("cluster_id") or "")
        if not cid or cid in out:
            continue
        out[cid] = {
            "cluster_id": cid,
            "room_ids": _ids(row.get("room_ids", [])),
            "target_sum": _round(row.get("target_sum", row.get("target_area_sum", 0.0))),
            "min_sum": _round(row.get("min_sum", row.get("min_area_sum", 0.0))),
        }
    return out


def _load_row_from_item(item: Mapping[str, Any], *, default_variant: str = "", default_source: str = "") -> Optional[Dict[str, Any]]:
    island_id = str(item.get("island_id") or item.get("candidate_island_id") or "")
    if not island_id:
        return None
    variant_id = str(item.get("variant_id") or default_variant)
    clusters = _ids(
        item.get(
            "assigned_cluster_ids",
            item.get("assigned_or_candidate_cluster_ids", item.get("clusters", item.get("cluster_ids", []))),
        )
    )
    rooms = _ids(item.get("assigned_room_ids", item.get("rooms", item.get("room_ids", []))))
    target = _round(
        item.get(
            "assigned_target_sum",
            item.get("target_area_load", item.get("target_area", item.get("target_area_sum", item.get("total_area", 0.0)))),
        )
    )
    min_sum = _round(item.get("assigned_min_sum", item.get("min_area_load", item.get("min_area", item.get("min_area_sum", 0.0)))))
    effective = _round(item.get("effective_capacity", item.get("effective_capacity_area", item.get("capacity_area", 0.0))))
    margin = item.get("capacity_margin")
    if margin is None and effective > 0.0:
        margin = effective - target
    overload = item.get("overload_area")
    if overload is None:
        overload = item.get("area_shortfall")
    if overload is None and margin is not None:
        overload = max(0.0, -_float(margin))
    return {
        "variant_id": variant_id,
        "island_id": island_id,
        "assigned_cluster_ids": clusters,
        "assigned_room_ids": rooms,
        "assigned_target_sum": target,
        "assigned_min_sum": min_sum,
        "effective_capacity": effective,
        "capacity_margin": _round(margin),
        "overload_area": _round(overload),
        "load_source": str(item.get("load_source") or default_source or "diagnostic_best_effort"),
    }


def _handoff_by_variant_and_island(
    grid: Mapping[str, Any],
    *,
    capacity_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    runtime_variant = str(grid.get("runtime_topology_variant_id") or grid.get("handoff_variant_id") or "")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    load_meta = grid.get("assigned_load_by_island")
    if isinstance(load_meta, Mapping):
        default_source = str(load_meta.get("load_source") or "diagnostic_best_effort")
        for item in _list(load_meta.get("items")):
            row = _load_row_from_item(_mapping(item), default_variant=runtime_variant, default_source=default_source)
            if row is not None:
                out[(row["variant_id"], row["island_id"])] = row
    if out:
        return out
    for row in _handoff_rows(grid):
        island_id = str(row.get("island_id") or "")
        if not island_id:
            continue
        variant_id = str(row.get("variant_id") or runtime_variant)
        clusters = _ids(row.get("clusters", row.get("cluster_ids", [])))
        rooms = _ids(row.get("rooms", row.get("room_ids", row.get("assigned_room_ids", []))))
        out[(variant_id, island_id)] = {
            "variant_id": variant_id,
            "island_id": island_id,
            "assigned_cluster_ids": clusters,
            "assigned_room_ids": rooms,
            "assigned_target_sum": _round(row.get("target_area", row.get("target_area_sum", row.get("total_area", 0.0)))),
            "assigned_min_sum": _round(row.get("min_area", row.get("min_area_sum", 0.0))),
            "load_source": "actual_runtime_assignment" if runtime_variant else "heuristic_assignment",
        }
    if out:
        return out
    proposal = grid.get("topology_assignment_proposal")
    if isinstance(proposal, Mapping):
        default_variant = str(proposal.get("selected_variant_id") or runtime_variant)
        for item in _list(proposal.get("island_loads")):
            row = _load_row_from_item(_mapping(item), default_variant=default_variant, default_source="topology_assignment_proposal")
            if row is not None:
                out[(row["variant_id"], row["island_id"])] = row
    if out:
        return out
    summary = capacity_summary if isinstance(capacity_summary, Mapping) else {}
    diagnostic_loads = summary.get("diagnostic_island_loads")
    if isinstance(diagnostic_loads, Mapping):
        default_source = str(diagnostic_loads.get("load_source") or "diagnostic_best_effort")
        overloaded = diagnostic_loads.get("overloaded_islands")
        if isinstance(overloaded, Mapping):
            for item in _list(overloaded.get("items")):
                row = _load_row_from_item(_mapping(item), default_variant=runtime_variant, default_source=default_source)
                if row is not None:
                    out[(row["variant_id"], row["island_id"])] = row
    return out


def build_island_cluster_provenance(
    floor_id: str,
    grid: Mapping[str, Any],
    *,
    capacity_summary: Optional[Mapping[str, Any]] = None,
    analysis_target_kind: str = "all_variants_summary",
) -> Dict[str, Any]:
    """Build bounded, variant-scoped island provenance metadata."""

    grid = grid if isinstance(grid, Mapping) else {}
    floor_id = str(floor_id or grid.get("floor_id") or "")
    active_target_hash = str(grid.get("active_target_hash") or "")
    allocation = _mapping(grid.get("capacity_aware_area_allocation"))
    area_allocation_id = str(grid.get("active_area_allocation_id") or allocation.get("area_allocation_id") or "")
    target_sum_source = "geometry_target" if allocation.get("applied") else ("preferred_target" if allocation else "unknown")
    cluster_by_id = _cluster_map(grid)
    handoff_by_key = _handoff_by_variant_and_island(grid, capacity_summary=capacity_summary)
    variants = _variant_rows(grid)
    candidate_meta_by_key = _candidate_island_metadata_by_key(grid)
    cluster_feasibility = _cluster_feasibility_summary(grid)

    per_island: List[Dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    islands_with_source = 0
    semantic_pairs_total = 0
    semantic_pairs_match = 0
    actual_sum = 0.0
    effective_sum = 0.0

    for variant in sorted(variants, key=lambda row: (str(row.get("seed", "")), str(row.get("variant_id") or ""))):
        variant_id = str(variant.get("variant_id") or variant.get("id") or "primary")
        for island in sorted(_island_rows(variant), key=lambda row: _island_id(row)):
            island_id = _island_id(island)
            if not island_id:
                continue
            meta = candidate_meta_by_key.get((variant_id, island_id), {})
            if meta:
                island = {**island, **meta}
            source_ids = _source_cluster_ids(island)
            ptype = _provenance_type(island, source_ids)
            type_counts[ptype] += 1
            if source_ids:
                islands_with_source += 1
            handoff = handoff_by_key.get((variant_id, island_id), {})
            assigned_clusters = _ids(handoff.get("assigned_cluster_ids", []))
            assigned_rooms = _ids(handoff.get("assigned_room_ids", []))
            if assigned_clusters and not assigned_rooms:
                for cid in assigned_clusters:
                    assigned_rooms.extend(cluster_by_id.get(cid, {}).get("room_ids", []))
                assigned_rooms = sorted(set(assigned_rooms))
            assigned_target = _float(handoff.get("assigned_target_sum", 0.0))
            if assigned_clusters and assigned_target <= 0.0:
                assigned_target = sum(_float(cluster_by_id.get(cid, {}).get("target_sum", 0.0)) for cid in assigned_clusters)
            effective = _island_effective_capacity(island)
            handoff_effective = _float(handoff.get("effective_capacity", 0.0))
            if effective <= 0.0 and handoff_effective > 0.0:
                effective = handoff_effective
            actual = _round(island.get("area", effective))
            actual_sum += actual
            effective_sum += effective
            if "capacity_margin" in handoff:
                margin = _round(handoff.get("capacity_margin"))
            else:
                margin = _round(effective - assigned_target)
            if "overload_area" in handoff:
                overload = _round(handoff.get("overload_area"))
            else:
                overload = _round(max(0.0, -margin))
            if source_ids and assigned_clusters:
                semantic_pairs_total += len(assigned_clusters)
                semantic_pairs_match += sum(1 for cid in assigned_clusters if cid in set(source_ids))
            per_island.append(
                {
                    "floor_id": floor_id,
                    "variant_id": variant_id,
                    "canonical_island_id": f"{floor_id}:{variant_id}:{island_id}",
                    "island_id": island_id,
                    "provenance_type": ptype,
                    "source_cluster_ids": source_ids,
                    "assigned_cluster_ids": assigned_clusters,
                    "assigned_room_ids": assigned_rooms,
                    "effective_capacity": effective,
                    "actual_area": actual,
                    "assigned_target_sum": _round(assigned_target),
                    "target_sum_source": target_sum_source,
                    "active_target_hash": active_target_hash,
                    "area_allocation_id": area_allocation_id,
                    "capacity_margin": margin,
                    "overload_area": overload,
                    "load_source": str(handoff.get("load_source") or "unavailable"),
                }
            )

    if islands_with_source <= 0:
        match_rate = {"value": None, "status": "not_applicable_no_source_islands", "denominator": 0}
    elif semantic_pairs_total <= 0:
        match_rate = {"value": None, "status": "not_applicable_no_assigned_pairs_with_source_islands", "denominator": 0}
    else:
        match_rate = {
            "value": round(float(semantic_pairs_match) / float(semantic_pairs_total), 6),
            "status": "ok",
            "numerator": semantic_pairs_match,
            "denominator": semantic_pairs_total,
        }

    cluster_items = [
        {
            "cluster_id": cid,
            "room_ids": row.get("room_ids", []),
            "target_sum": _round(row.get("target_sum", 0.0)),
            "min_sum": _round(row.get("min_sum", 0.0)),
        }
        for cid, row in sorted(cluster_by_id.items())
    ]
    report = {
        **_base_report("island_cluster_provenance_v1"),
        "floor_id": floor_id,
        "analysis_target_kind": analysis_target_kind,
        "active_target_hash": active_target_hash,
        "area_allocation_id": area_allocation_id,
        "target_sum_source": target_sum_source,
        "variant_count": len(variants),
        "island_count": len(per_island),
        "cluster_count": len(cluster_items),
        "islands_with_source_cluster_ids": islands_with_source,
        "islands_without_source_cluster_ids": max(0, len(per_island) - islands_with_source),
        "provenance_type_counts": dict(sorted(type_counts.items())),
        "actual_area_sum": _round(actual_sum),
        "effective_capacity_sum": _round(effective_sum),
        "semantic_source_match_rate": match_rate,
        "candidate_island_metadata_available": bool(candidate_meta_by_key),
        "assigned_load_by_island_available": bool(handoff_by_key),
        "cluster_feasibility_summary": cluster_feasibility,
        "clusters": _bounded(cluster_items),
        "per_island": _bounded(sorted(per_island, key=lambda row: row["canonical_island_id"])),
    }
    return _json_safe(report)


def _capacity_primary_type(capacity_summary: Optional[Mapping[str, Any]]) -> str:
    summary = capacity_summary if isinstance(capacity_summary, Mapping) else {}
    diagnosis = summary.get("diagnosis") if isinstance(summary.get("diagnosis"), Mapping) else {}
    return str(diagnosis.get("primary_conflict_type") or "")


def _cluster_too_large(provenance: Mapping[str, Any], max_effective: float) -> List[Dict[str, Any]]:
    feasibility = provenance.get("cluster_feasibility_summary")
    if isinstance(feasibility, Mapping):
        out: List[Dict[str, Any]] = []
        for row in _list(feasibility.get("items")):
            cluster = _mapping(row)
            target_too_large = bool(cluster.get("target_too_large_for_any_island"))
            min_too_large = bool(cluster.get("min_too_large_for_any_island"))
            if not (target_too_large or min_too_large):
                continue
            out.append(
                {
                    "cluster_id": str(cluster.get("cluster_id") or ""),
                    "cluster_target_sum": _round(cluster.get("target_sum", cluster.get("target_area_sum", 0.0))),
                    "cluster_min_sum": _round(cluster.get("min_sum", cluster.get("min_area_sum", 0.0))),
                    "max_island_effective_capacity": _round(cluster.get("max_island_effective_capacity", max_effective)),
                    "target_too_large_for_any_island": target_too_large,
                    "min_too_large_for_any_island": min_too_large,
                    "feasible_island_count": int(cluster.get("feasible_island_count") or 0),
                    "best_candidate_variant_id": str(cluster.get("best_candidate_variant_id") or ""),
                    "best_candidate_island_id": str(cluster.get("best_candidate_island_id") or ""),
                    "failed_constraint_counts": dict(cluster.get("failed_constraint_counts") or {}),
                }
            )
        if out:
            return sorted(out, key=lambda row: (-float(row["cluster_target_sum"]), str(row["cluster_id"])))
    clusters = ((_mapping(provenance.get("clusters")).get("items")) or []) if isinstance(provenance.get("clusters"), Mapping) else []
    out: List[Dict[str, Any]] = []
    for row in clusters:
        cluster = _mapping(row)
        target = _float(cluster.get("target_sum"))
        min_sum = _float(cluster.get("min_sum"))
        target_too_large = bool(max_effective > 0.0 and target > max_effective + 1e-6)
        min_too_large = bool(max_effective > 0.0 and min_sum > max_effective + 1e-6)
        if target_too_large or min_too_large:
            out.append(
                {
                    "cluster_id": str(cluster.get("cluster_id") or ""),
                    "cluster_target_sum": _round(target),
                    "cluster_min_sum": _round(min_sum),
                    "max_island_effective_capacity": _round(max_effective),
                    "target_too_large_for_any_island": target_too_large,
                    "min_too_large_for_any_island": min_too_large,
                }
            )
    return sorted(out, key=lambda row: (-float(row["cluster_target_sum"]), str(row["cluster_id"])))


def build_island_capacity_blocker_explanation(
    floor_id: str,
    primary_conflict_type: str,
    provenance: Mapping[str, Any],
    *,
    capacity_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Explain island capacity shortfall with conservative MVP rules."""

    floor_id = str(floor_id or provenance.get("floor_id") or "")
    primary_conflict_type = str(primary_conflict_type or _capacity_primary_type(capacity_summary))
    items = []
    if isinstance(provenance.get("per_island"), Mapping):
        items = [_mapping(item) for item in _list(provenance.get("per_island", {}).get("items"))]
    overloaded = [
        {
            "canonical_island_id": str(item.get("canonical_island_id") or ""),
            "variant_id": str(item.get("variant_id") or ""),
            "island_id": str(item.get("island_id") or ""),
            "overload_area": _round(item.get("overload_area", 0.0)),
            "assigned_cluster_ids": _ids(item.get("assigned_cluster_ids", [])),
            "assigned_target_sum": _round(item.get("assigned_target_sum", 0.0)),
            "effective_capacity": _round(item.get("effective_capacity", 0.0)),
            "load_source": str(item.get("load_source") or "unavailable"),
        }
        for item in items
        if _float(item.get("overload_area")) > 1e-6
    ]
    summary = capacity_summary if isinstance(capacity_summary, Mapping) else {}
    diagnostic_loads = summary.get("diagnostic_island_loads")
    if isinstance(diagnostic_loads, Mapping):
        load_source = str(diagnostic_loads.get("load_source") or "diagnostic_best_effort")
        overloaded_block = diagnostic_loads.get("overloaded_islands")
        if isinstance(overloaded_block, Mapping):
            for raw_item in _list(overloaded_block.get("items")):
                row = _mapping(raw_item)
                overload = _round(row.get("overload_area", row.get("area_shortfall", 0.0)))
                if overload <= 1e-6:
                    continue
                variant_id = str(row.get("variant_id") or "")
                island_id = str(row.get("island_id") or "")
                overloaded.append(
                    {
                        "canonical_island_id": str(row.get("canonical_island_id") or f"{floor_id}:{variant_id}:{island_id}"),
                        "variant_id": variant_id,
                        "island_id": island_id,
                        "overload_area": overload,
                        "assigned_cluster_ids": _ids(
                            row.get(
                                "assigned_cluster_ids",
                                row.get("assigned_or_candidate_cluster_ids", row.get("clusters", row.get("cluster_ids", []))),
                            )
                        ),
                        "assigned_target_sum": _round(row.get("assigned_target_sum", row.get("target_area_load", 0.0))),
                        "effective_capacity": _round(row.get("effective_capacity", row.get("effective_capacity_area", 0.0))),
                        "load_source": load_source,
                    }
                )
    deduped_overloaded: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
    for row in overloaded:
        key = (
            str(row.get("variant_id") or ""),
            str(row.get("island_id") or ""),
            tuple(_ids(row.get("assigned_cluster_ids", []))),
        )
        current = deduped_overloaded.get(key)
        if current is None or _float(row.get("overload_area")) > _float(current.get("overload_area")):
            deduped_overloaded[key] = row
    overloaded = list(deduped_overloaded.values())
    overloaded = sorted(overloaded, key=lambda row: (-float(row["overload_area"]), row["canonical_island_id"]))
    blocking_clusters = sorted({cid for row in overloaded for cid in row.get("assigned_cluster_ids", [])})
    max_effective = max((_float(item.get("effective_capacity")) for item in items), default=0.0)
    cluster_large = _cluster_too_large(provenance, max_effective)
    blocking_clusters = sorted(set(blocking_clusters) | {str(row.get("cluster_id") or "") for row in cluster_large if str(row.get("cluster_id") or "")})
    clusters_without_feasible: List[Dict[str, Any]] = []
    without_feasible = summary.get("clusters_without_feasible_island")
    if isinstance(without_feasible, Mapping):
        for raw_item in _list(without_feasible.get("items")):
            row = _mapping(raw_item)
            cid = str(row.get("cluster_id") or "")
            if cid:
                clusters_without_feasible.append(
                    {
                        "cluster_id": cid,
                        "target_sum": _round(row.get("target_sum", row.get("target_area_sum", 0.0))),
                        "min_sum": _round(row.get("min_sum", row.get("min_area_sum", 0.0))),
                        "feasible_variant_count": int(row.get("feasible_variant_count") or row.get("feasible_island_count") or 0),
                        "best_candidate_variant_id": str(row.get("best_candidate_variant_id") or ""),
                        "best_candidate_island_id": str(row.get("best_candidate_island_id") or ""),
                    }
                )
        blocking_clusters = sorted(set(blocking_clusters) | {row["cluster_id"] for row in clusters_without_feasible})
    no_source_islands = int(provenance.get("islands_with_source_cluster_ids", 0) or 0) == 0 and int(provenance.get("island_count", 0) or 0) > 0
    global_semantic_source_missing = bool(no_source_islands)
    blocking_cluster_source_missing = bool(no_source_islands and blocking_clusters)

    detected: List[str] = []
    if cluster_large:
        detected.append("cluster_too_large_for_generated_islands")
    if blocking_cluster_source_missing:
        detected.append("semantic_source_missing")
    actual_sum = _float(provenance.get("actual_area_sum", 0.0))
    effective_sum = _float(provenance.get("effective_capacity_sum", 0.0))
    fragmentation_evidence = {
        "actual_area_sum": _round(actual_sum),
        "effective_capacity_sum": _round(effective_sum),
        "actual_to_effective_ratio": _round(actual_sum / effective_sum) if effective_sum > 1e-6 else None,
        "evidence_status": "partial" if actual_sum > 0.0 and effective_sum > 0.0 else "insufficient",
    }
    if overloaded and actual_sum > effective_sum * 1.10 and effective_sum > 0.0:
        detected.append("island_fragmentation")
    # v1 only claims assignment distribution when explicit alternative evidence is present.
    alternative_evidence = bool(_mapping(capacity_summary or {}).get("assignment_distribution_alternatives"))
    if alternative_evidence:
        detected.append("assignment_distribution_issue")
    if primary_conflict_type != "island_area_capacity_shortfall" and not detected:
        detected = []
    if primary_conflict_type == "island_area_capacity_shortfall" and not detected:
        detected.append("insufficient_metadata")

    priority = [
        "cluster_too_large_for_generated_islands",
        "semantic_source_missing",
        "island_fragmentation",
        "assignment_distribution_issue",
        "insufficient_metadata",
    ]
    primary = next((cause for cause in priority if cause in detected), "no_action")
    confidence = "low"
    if primary in {"cluster_too_large_for_generated_islands", "semantic_source_missing"}:
        confidence = "medium"
    if primary == "cluster_too_large_for_generated_islands" and any(item.get("min_too_large_for_any_island") for item in cluster_large):
        confidence = "high"
    if primary == "no_action":
        confidence = "low"
    next_phase = {
        "cluster_too_large_for_generated_islands": "cluster_split_diagnostic",
        "semantic_source_missing": "semantic_seeded_territory_variants",
        "island_fragmentation": "semantic_seeded_territory_variants",
        "assignment_distribution_issue": "semantic_seeded_territory_variants",
        "insufficient_metadata": "add_missing_metadata",
        "no_action": "no_action",
    }.get(primary, "add_missing_metadata")

    report = {
        **_base_report("island_capacity_blocker_explanation_v1"),
        "floor_id": floor_id,
        "primary_conflict_type": primary_conflict_type,
        "detected_causes": detected,
        "primary_likely_cause": primary,
        "confidence": confidence,
        "classification_priority_used": priority,
        "global_semantic_source_missing": global_semantic_source_missing,
        "blocking_cluster_source_missing": blocking_cluster_source_missing,
        "cluster_too_large": _bounded(cluster_large),
        "target_sum_source": str(provenance.get("target_sum_source") or "unknown"),
        "active_target_hash": str(provenance.get("active_target_hash") or ""),
        "area_allocation_id": str(provenance.get("area_allocation_id") or ""),
        "blocking_clusters": blocking_clusters[:MAX_ITEMS],
        "overloaded_islands": _bounded(overloaded),
        "clusters_without_feasible_island": _bounded(clusters_without_feasible),
        "cluster_feasibility_summary_available": bool(isinstance(provenance.get("cluster_feasibility_summary"), Mapping) and provenance.get("cluster_feasibility_summary", {}).get("available")),
        "fragmentation_evidence": fragmentation_evidence,
        "assignment_distribution_alternative_evidence": alternative_evidence,
        "next_phase_recommendation": next_phase,
    }
    return _json_safe(report)


__all__ = [
    "build_circulation_contract",
    "build_island_cluster_provenance",
    "build_island_capacity_blocker_explanation",
]

