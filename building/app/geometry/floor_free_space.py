from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .core_contracts import (
    CORE_OVERLAP_EPSILON_AREA,
    CoreFootprintContract,
    build_core_footprint_contract,
    collect_core_geometry_diagnostics,
)
from .exceptions import LayoutGeometryInvariantError
from .topology_generator import CoreTube


STAGE2A_REPORT_SCHEMA_VERSION = "stage2a.v1"


def classify_free_space_geometry(geom: BaseGeometry) -> str:
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return "empty"
    if not bool(getattr(geom, "is_valid", True)):
        return "invalid"
    if isinstance(geom, Polygon):
        return "polygon_with_holes" if len(list(geom.interiors)) > 0 else "polygon"
    if isinstance(geom, MultiPolygon):
        return "multipolygon"
    if isinstance(geom, GeometryCollection):
        return "geometry_collection"
    return type(geom).__name__.lower()


@dataclass(frozen=True)
class FloorFreeSpace:
    floor_number: int
    floor_id: str
    envelope_geometry: Polygon
    stage1_core_reference: Dict[str, Any]
    stage1_core_tube: CoreTube
    core_contract: CoreFootprintContract
    free_space_geometry: Polygon
    geometry_kind: str
    source: str
    corridor_context_reference: Dict[str, Any]

    def to_report(self) -> Dict[str, Any]:
        return {
            "stage": "stage2a",
            "schema_version": STAGE2A_REPORT_SCHEMA_VERSION,
            "floor_id": self.floor_id,
            "floor_number": self.floor_number,
            "source": self.source,
            "core_source": self.stage1_core_reference.get("core_source"),
            "corridor_source": self.corridor_context_reference.get("corridor_source"),
            "topology_mode": self.corridor_context_reference.get("topology_mode"),
            "corridor_layout": self.corridor_context_reference.get("corridor_layout"),
            "envelope_status": "resolved",
            "core_status": "resolved",
            "floor_free_space_constructed": True,
            "free_space_geometry_kind": self.geometry_kind,
            "free_space_area": float(self.free_space_geometry.area),
            "core_area": float(getattr(self.stage1_core_tube.polygon, "area", 0.0) or 0.0),
            "core_contract_id": self.core_contract.core_contract_id,
            "core_union_hash": self.core_contract.core_union_hash,
            "core_positive_overlap_area": 0.0,
            "coverage_fallback_touched_core": False,
            "serializer_core_contract_id": self.core_contract.core_contract_id,
            "stage1_core_policy_id": self.stage1_core_reference.get("stage1_core_policy_id"),
        }


def _stage2a_failure(
    failure_type: str,
    message: str,
    *,
    floor_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> LayoutGeometryInvariantError:
    meta = dict(metadata or {})
    meta.update(
        {
            "stage": "stage2a_preflight_failed",
            "failure_type": failure_type,
            "stage2a_failure": True,
            "semantic_repair_allowed": False,
        }
    )
    return LayoutGeometryInvariantError(
        message,
        floor_id=floor_id,
        stage="stage2a_preflight_failed",
        metadata=meta,
    )


def _resolve_free_space_failure(kind: str) -> str:
    if kind == "polygon_with_holes":
        return "free_space_has_holes"
    if kind == "multipolygon":
        return "free_space_fragmented"
    return "free_space_invalid"


def build_floor_free_space(
    *,
    floor_number: int,
    floor_boundary: Polygon,
    stage1_core_tube: CoreTube,
    core_metadata: Mapping[str, Any],
    corridor_options: Mapping[str, Any],
    topology_mode: str,
    corridor_width: Optional[float] = None,
) -> FloorFreeSpace:
    floor_id = f"F{int(floor_number)}"
    if floor_boundary is None or floor_boundary.is_empty or not floor_boundary.is_valid:
        raise _stage2a_failure(
            "envelope_unresolved",
            "Stage 2A requires a resolved, valid envelope before geometry",
            floor_id=floor_id,
        )
    if core_metadata.get("core_source") != "stage1":
        raise _stage2a_failure(
            "core_context_mismatch",
            "Stage 2A requires Stage 1 as the only core source",
            floor_id=floor_id,
            metadata={"core_source": core_metadata.get("core_source")},
        )
    if not core_metadata.get("bbox"):
        raise _stage2a_failure(
            "core_policy_unresolved",
            "Stage 2A requires a resolved Stage 1 core bbox",
            floor_id=floor_id,
            metadata={"stage1_core_policy_id": core_metadata.get("stage1_core_policy_id")},
        )
    core_tube = copy.deepcopy(stage1_core_tube)
    core_poly = getattr(core_tube, "polygon", None)
    if core_poly is None or core_poly.is_empty or not core_poly.is_valid:
        raise _stage2a_failure(
            "core_policy_unresolved",
            "Stage 1 CoreTube has no valid resolved geometry",
            floor_id=floor_id,
            metadata={"stage1_core_policy_id": core_metadata.get("stage1_core_policy_id")},
        )
    if not floor_boundary.buffer(1e-7).covers(core_poly):
        raise _stage2a_failure(
            "core_outside_envelope",
            "Stage 1 core must be inside the resolved envelope",
            floor_id=floor_id,
            metadata={
                "core_bounds": tuple(float(v) for v in core_poly.bounds),
                "floor_bounds": tuple(float(v) for v in floor_boundary.bounds),
            },
        )

    try:
        free_space = floor_boundary.difference(core_poly).buffer(0)
    except Exception as exc:
        raise _stage2a_failure(
            "free_space_invalid",
            "Failed to subtract Stage 1 core from envelope",
            floor_id=floor_id,
            metadata={"error_type": type(exc).__name__, "error": str(exc)},
        ) from exc
    kind = classify_free_space_geometry(free_space)
    if kind != "polygon":
        raise _stage2a_failure(
            _resolve_free_space_failure(kind),
            f"Stage 2A MVP only supports simple polygon free_space, got {kind}",
            floor_id=floor_id,
            metadata={"free_space_geometry_kind": kind},
        )

    core_contract = build_core_footprint_contract(
        core_tube,
        floor_id=floor_id,
        topology_mode=str(topology_mode or ""),
        created_from="stage2a_floor_free_space",
    )
    setattr(core_tube, "core_source", "stage1")
    setattr(core_tube, "stage1_core_policy_id", core_metadata.get("stage1_core_policy_id"))

    stage1_core_reference = {
        "core_source": "stage1",
        "stage1_core_policy_id": core_metadata.get("stage1_core_policy_id"),
        "connectivity_type": core_metadata.get("connectivity_type"),
        "selected_placement": core_metadata.get("selected_placement"),
        "core_area": core_metadata.get("core_area"),
        "floor_count": core_metadata.get("floor_count"),
        "bbox": core_metadata.get("bbox"),
        "core_union_hash": core_contract.core_union_hash,
        "core_contract_id": core_contract.core_contract_id,
    }
    corridor_context_reference = {
        "corridor_source": corridor_options.get("corridor_source"),
        "topology_mode": str(topology_mode or ""),
        "corridor_layout": corridor_options.get("corridor_layout"),
        "reserve_ratio": corridor_options.get("reserve_ratio"),
        "wall_reserve_ratio": corridor_options.get("wall_reserve_ratio"),
        "target_width": corridor_options.get("target_width"),
        "corridor_width": corridor_width,
    }
    if corridor_context_reference["corridor_source"] != "stage1":
        raise _stage2a_failure(
            "corridor_context_mismatch",
            "Stage 2A requires Stage 1 as the corridor context source",
            floor_id=floor_id,
            metadata=corridor_context_reference,
        )
    return FloorFreeSpace(
        floor_number=int(floor_number),
        floor_id=floor_id,
        envelope_geometry=floor_boundary,
        stage1_core_reference=stage1_core_reference,
        stage1_core_tube=core_tube,
        core_contract=core_contract,
        free_space_geometry=free_space,
        geometry_kind=kind,
        source="stage1",
        corridor_context_reference=corridor_context_reference,
    )


def build_floor_free_spaces_for_allocation(
    *,
    floor_numbers: Iterable[int],
    floor_boundary: Polygon,
    stage1_core_tube: CoreTube,
    core_metadata: Mapping[str, Any],
    corridor_options: Mapping[str, Any],
    topology_mode: str,
    corridor_width: Optional[float] = None,
) -> Dict[str, FloorFreeSpace]:
    return {
        f"F{int(n)}": build_floor_free_space(
            floor_number=int(n),
            floor_boundary=floor_boundary,
            stage1_core_tube=stage1_core_tube,
            core_metadata=core_metadata,
            corridor_options=corridor_options,
            topology_mode=topology_mode,
            corridor_width=corridor_width,
        )
        for n in floor_numbers
    }


def positive_core_overlap_area(
    *,
    floor_id: str,
    topology_mode: str,
    core_contract: CoreFootprintContract,
    rooms: Optional[list[Any]] = None,
    corridors: Optional[list[Any]] = None,
    coverage_features: Optional[list[Any]] = None,
) -> float:
    diagnostics = collect_core_geometry_diagnostics(
        floor_id=floor_id,
        topology_mode=topology_mode,
        core_contract=core_contract,
        rooms=rooms or [],
        corridors=corridors or [],
        coverage_features=coverage_features or [],
        epsilon_area=CORE_OVERLAP_EPSILON_AREA,
    )
    return float(
        diagnostics.get("room_core_overlap_total", 0.0)
        + diagnostics.get("corridor_core_overlap_total", 0.0)
        + diagnostics.get("coverage_feature_core_overlap_total", 0.0)
    )


def build_stage2a_report(floor_free_spaces: Mapping[str, FloorFreeSpace]) -> Dict[str, Any]:
    floors = {fid: ffs.to_report() for fid, ffs in sorted(floor_free_spaces.items())}
    first = next(iter(floors.values()), {})
    return {
        "stage": "stage2a",
        "schema_version": STAGE2A_REPORT_SCHEMA_VERSION,
        "core_source": first.get("core_source"),
        "corridor_source": first.get("corridor_source"),
        "topology_mode": first.get("topology_mode"),
        "corridor_layout": first.get("corridor_layout"),
        "envelope_status": first.get("envelope_status"),
        "core_status": first.get("core_status"),
        "floor_free_space_constructed": all(bool(v.get("floor_free_space_constructed")) for v in floors.values()),
        "free_space_geometry_kind": first.get("free_space_geometry_kind"),
        "core_union_hash": first.get("core_union_hash"),
        "core_positive_overlap_area": max(
            [float(v.get("core_positive_overlap_area", 0.0) or 0.0) for v in floors.values()] or [0.0]
        ),
        "coverage_fallback_touched_core": any(bool(v.get("coverage_fallback_touched_core")) for v in floors.values()),
        "serializer_core_contract_id": first.get("serializer_core_contract_id"),
        "floors": floors,
    }
