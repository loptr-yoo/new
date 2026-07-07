from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .exceptions import LayoutGeometryInvariantError

logger = logging.getLogger(__name__)

CORE_OVERLAP_EPSILON_AREA = 0.01
MIN_CORE_DOOR_WIDTH = 0.8
CORE_BUDGET_AREA_MISMATCH_ABS = 0.5
CORE_BUDGET_AREA_MISMATCH_RATIO = 0.05

PUBLIC_CORE_HALL_TYPES = {"elevator_hall", "staircase_hall"}
CORE_SHAFT_TYPES = {"elevator_shaft", "staircase_shaft"}


def _as_polygon(poly: Any) -> Optional[Polygon]:
    if isinstance(poly, Polygon) and not poly.is_empty:
        return poly
    return None


def _polygon_from_object(obj: Any) -> Optional[Polygon]:
    if isinstance(obj, Polygon):
        return _as_polygon(obj)
    if isinstance(obj, dict):
        return _as_polygon(obj.get("polygon"))
    return _as_polygon(getattr(obj, "polygon", None))


def _object_id(obj: Any, default: str) -> str:
    if isinstance(obj, dict):
        return str(obj.get("id") or obj.get("room_id") or obj.get("feature_id") or default)
    return str(
        getattr(obj, "id", None)
        or getattr(obj, "room_id", None)
        or getattr(obj, "feature_id", None)
        or default
    )


def _object_type(obj: Any, fallback: str) -> str:
    if isinstance(obj, dict):
        return str(obj.get("room_type") or obj.get("type") or obj.get("coverage_role") or fallback).lower()
    return str(
        getattr(obj, "room_type", None)
        or getattr(obj, "type", None)
        or getattr(obj, "coverage_role", None)
        or fallback
    ).lower()


def _union_polygons(polys: Iterable[Polygon]) -> BaseGeometry:
    valid = [p for p in polys if isinstance(p, Polygon) and (not p.is_empty)]
    if not valid:
        return GeometryCollection()
    try:
        return unary_union(valid)
    except Exception:
        repaired = []
        for p in valid:
            try:
                repaired.append(p.buffer(0))
            except Exception:
                repaired.append(p)
        return unary_union(repaired) if repaired else GeometryCollection()


def _iter_polygons(geom: Any) -> List[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    pieces: List[Polygon] = []
    if hasattr(geom, "geoms"):
        for part in getattr(geom, "geoms", []):
            pieces.extend(_iter_polygons(part))
    return pieces


def core_union_hash(geom: BaseGeometry, *, precision: int = 4) -> str:
    """Stable hash for equivalent core footprints across ring/component order."""
    try:
        fixed = geom.buffer(0)
    except Exception:
        fixed = geom
    polys = []
    for poly in _iter_polygons(fixed):
        try:
            exterior = [(round(float(x), precision), round(float(y), precision)) for x, y in poly.exterior.coords]
            if exterior and exterior[0] == exterior[-1]:
                exterior = exterior[:-1]
            # Make ring order stable enough for axis-aligned core rectangles.
            if exterior:
                min_idx = min(range(len(exterior)), key=lambda i: exterior[i])
                exterior = exterior[min_idx:] + exterior[:min_idx]
                rev = list(reversed(exterior))
                if tuple(rev) < tuple(exterior):
                    exterior = rev
            holes = []
            for ring in poly.interiors:
                coords = [(round(float(x), precision), round(float(y), precision)) for x, y in ring.coords]
                if coords and coords[0] == coords[-1]:
                    coords = coords[:-1]
                holes.append(tuple(sorted(coords)))
            minx, miny, maxx, maxy = (round(float(v), precision) for v in poly.bounds)
            polys.append((minx, miny, maxx, maxy, round(float(poly.area), precision), tuple(exterior), tuple(sorted(holes))))
        except Exception:
            try:
                minx, miny, maxx, maxy = (round(float(v), precision) for v in poly.bounds)
                polys.append((minx, miny, maxx, maxy, round(float(poly.area), precision)))
            except Exception:
                continue
    payload = repr(sorted(polys)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _stable_contract_id(
    *,
    floor_id: Optional[str],
    topology_mode: Optional[str],
    core_union: BaseGeometry,
) -> str:
    try:
        minx, miny, maxx, maxy = core_union.bounds
        payload = (
            f"{floor_id or 'floor'}|{topology_mode or 'unknown'}|"
            f"{float(core_union.area):.6f}|{minx:.4f},{miny:.4f},{maxx:.4f},{maxy:.4f}|"
            f"{core_union_hash(core_union)}"
        )
    except Exception:
        payload = f"{floor_id or 'floor'}|{topology_mode or 'unknown'}|empty"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"core_contract_{floor_id or 'floor'}_{digest}"


@dataclass
class CoreComponent:
    component_id: str
    component_type: str
    polygon: Polygon

    @property
    def area(self) -> float:
        return float(self.polygon.area)


@dataclass
class CoreFootprintContract:
    core_contract_id: str
    version: str
    created_from: str
    floor_id: str
    topology_mode: str
    core_public_halls: List[CoreComponent] = field(default_factory=list)
    core_shafts: List[CoreComponent] = field(default_factory=list)
    core_union: BaseGeometry = field(default_factory=GeometryCollection)
    core_public_union: BaseGeometry = field(default_factory=GeometryCollection)
    core_union_hash: str = ""
    core_union_area: float = 0.0
    core_union_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def public_hall_ids(self) -> List[str]:
        return [c.component_id for c in self.core_public_halls]

    @property
    def shaft_ids(self) -> List[str]:
        return [c.component_id for c in self.core_shafts]

    @property
    def core_zone_ids(self) -> List[str]:
        return self.public_hall_ids + self.shaft_ids

    def validate(self, *, epsilon_area: float = CORE_OVERLAP_EPSILON_AREA) -> "CoreFootprintContract":
        missing = []
        if not self.core_contract_id:
            missing.append("core_contract_id")
        if not self.floor_id:
            missing.append("floor_id")
        if self.core_union is None or getattr(self.core_union, "is_empty", True):
            missing.append("core_union")
        if missing:
            raise LayoutGeometryInvariantError(
                "Core footprint contract is incomplete",
                floor_id=self.floor_id or None,
                stage="core_footprint_invalid",
                metadata={
                    "core_contract_id": self.core_contract_id,
                    "missing_fields": missing,
                    "semantic_repair_allowed": False,
                },
            )

        if not self.core_union_hash:
            self.core_union_hash = core_union_hash(self.core_union)
        try:
            self.core_union_area = float(getattr(self.core_union, "area", 0.0) or 0.0)
            self.core_union_bounds = tuple(float(v) for v in self.core_union.bounds)  # type: ignore[assignment]
        except Exception:
            self.core_union_area = 0.0
            self.core_union_bounds = (0.0, 0.0, 0.0, 0.0)

        components = list(self.core_public_halls) + list(self.core_shafts)
        invalid_components = [
            c.component_id for c in components
            if c.polygon is None or c.polygon.is_empty or not bool(getattr(c.polygon, "is_valid", True))
        ]
        if invalid_components:
            raise LayoutGeometryInvariantError(
                "Core footprint contains invalid components",
                floor_id=self.floor_id,
                stage="core_footprint_invalid",
                metadata={
                    "core_contract_id": self.core_contract_id,
                    "offending_objects": invalid_components,
                    "semantic_repair_allowed": False,
                },
            )

        overlapping_components = []
        for a, b in combinations(components, 2):
            try:
                overlap = float(a.polygon.intersection(b.polygon).area)
            except Exception:
                overlap = float("inf")
            if overlap > float(epsilon_area):
                overlapping_components.append({
                    "a": a.component_id,
                    "b": b.component_id,
                    "overlap_area": overlap,
                })
        if overlapping_components:
            raise LayoutGeometryInvariantError(
                "Core footprint components overlap",
                floor_id=self.floor_id,
                stage="core_footprint_invalid",
                metadata={
                    "core_contract_id": self.core_contract_id,
                    "offending_objects": overlapping_components,
                    "threshold": float(epsilon_area),
                    "semantic_repair_allowed": False,
                },
            )

        try:
            public_outside = float(self.core_public_union.difference(self.core_union.buffer(1e-8)).area)
        except Exception:
            public_outside = float("inf")
        if public_outside > float(epsilon_area):
            raise LayoutGeometryInvariantError(
                "Core public hall union is outside core union",
                floor_id=self.floor_id,
                stage="core_footprint_invalid",
                metadata={
                    "core_contract_id": self.core_contract_id,
                    "overlap_area": public_outside,
                    "threshold": float(epsilon_area),
                    "semantic_repair_allowed": False,
                },
            )
        return self


@dataclass
class CorePortalSpec:
    portal_id: str
    core_contract_id: str
    floor_id: str
    corridor_id: str
    core_public_hall_id: str
    portal_side: Optional[str] = None
    normal: Optional[Tuple[float, float]] = None
    candidate_edge: Optional[Any] = None
    door_center: Optional[Tuple[float, float]] = None
    min_width: float = MIN_CORE_DOOR_WIDTH
    max_width: float = 1.2
    source: str = "core_access_planner"


def build_core_footprint_contract(
    core_tube: Any,
    *,
    floor_id: Optional[str] = None,
    topology_mode: Optional[str] = None,
    created_from: str = "core_tube",
) -> Optional[CoreFootprintContract]:
    if core_tube is None:
        return None

    public_specs = [
        ("core_staircase_hall", "staircase_hall", getattr(core_tube, "staircase_hall", None)),
        ("core_staircase_hall_b", "staircase_hall", getattr(core_tube, "staircase_hall_b", None)),
        ("core_elevator_hall", "elevator_hall", getattr(core_tube, "elevator_hall", None)),
        ("core_elevator_hall_b", "elevator_hall", getattr(core_tube, "elevator_hall_b", None)),
    ]
    shaft_specs = [
        ("core_staircase_shaft", "staircase_shaft", getattr(core_tube, "staircase_shaft", None)),
        ("core_elevator_shaft", "elevator_shaft", getattr(core_tube, "elevator_shaft", None)),
    ]
    public_halls = [
        CoreComponent(cid, ctype, poly)
        for cid, ctype, poly in public_specs
        if isinstance(poly, Polygon) and not poly.is_empty
    ]
    shafts = [
        CoreComponent(cid, ctype, poly)
        for cid, ctype, poly in shaft_specs
        if isinstance(poly, Polygon) and not poly.is_empty
    ]
    components = public_halls + shafts
    if components:
        core_union = _union_polygons(c.polygon for c in components)
    else:
        poly = getattr(core_tube, "polygon", None)
        if not isinstance(poly, Polygon) or poly.is_empty:
            return None
        core_union = poly
    public_union = _union_polygons(c.polygon for c in public_halls)
    fid = str(floor_id or getattr(core_tube, "floor_id", None) or "F?")
    mode = str(topology_mode or getattr(core_tube, "topology_mode", None) or "unknown")
    existing_floor = getattr(core_tube, "core_contract_floor_id", None)
    existing_id = getattr(core_tube, "core_contract_id", None)
    contract_id = str(existing_id) if existing_id and str(existing_floor or fid) == str(fid) else _stable_contract_id(
        floor_id=fid,
        topology_mode=mode,
        core_union=core_union,
    )
    setattr(core_tube, "core_contract_id", contract_id)
    setattr(core_tube, "core_contract_version", "stage2_core_v1")
    setattr(core_tube, "core_contract_floor_id", fid)
    core_union_bounds = None
    if not getattr(core_union, "is_empty", True):
        minx, miny, maxx, maxy = core_union.bounds
        core_union_bounds = (float(minx), float(miny), float(maxx), float(maxy))

    diagnostics = {
        "core_union_area": float(getattr(core_union, "area", 0.0) or 0.0),
        "core_union_hash": core_union_hash(core_union),
        "core_union_bounds": core_union_bounds,
        "core_public_union_area": float(getattr(public_union, "area", 0.0) or 0.0),
        "component_count": len(components),
        "public_hall_ids": [c.component_id for c in public_halls],
        "shaft_ids": [c.component_id for c in shafts],
    }
    return CoreFootprintContract(
        core_contract_id=contract_id,
        version="stage2_core_v1",
        created_from=created_from,
        floor_id=fid,
        topology_mode=mode,
        core_public_halls=public_halls,
        core_shafts=shafts,
        core_union=core_union,
        core_public_union=public_union,
        core_union_hash=diagnostics["core_union_hash"],
        core_union_area=float(diagnostics["core_union_area"]),
        core_union_bounds=core_union_bounds or (0.0, 0.0, 0.0, 0.0),
        diagnostics=diagnostics,
    ).validate()


def _overlap_record(
    *,
    obj: Any,
    object_type: str,
    core_contract: CoreFootprintContract,
    epsilon_area: float,
    default_id: str,
) -> Optional[Dict[str, Any]]:
    poly = _polygon_from_object(obj)
    if poly is None:
        return None
    try:
        hit = poly.intersection(core_contract.core_union)
        area = float(getattr(hit, "area", 0.0) or 0.0)
    except Exception as exc:
        return {
            "object_id": _object_id(obj, default_id),
            "object_type": object_type,
            "object_type_pair": f"{object_type}:core",
            "overlap_area": float("inf"),
            "overlap_bbox": None,
            "failure_reason": f"intersection_failed:{type(exc).__name__}",
        }
    if area <= float(epsilon_area):
        return None
    try:
        bbox = tuple(round(float(v), 4) for v in hit.bounds)
    except Exception:
        bbox = None
    return {
        "object_id": _object_id(obj, default_id),
        "object_type": _object_type(obj, object_type),
        "object_type_pair": f"{object_type}:core",
        "overlap_area": area,
        "overlap_bbox": bbox,
        "threshold": float(epsilon_area),
    }


def collect_core_geometry_diagnostics(
    *,
    floor_id: str,
    topology_mode: str,
    core_contract: Optional[CoreFootprintContract],
    rooms: Optional[Sequence[Any]] = None,
    generated_rooms: Optional[Sequence[Any]] = None,
    coverage_features: Optional[Sequence[Any]] = None,
    corridors: Optional[Sequence[Any]] = None,
    doors: Optional[Sequence[Any]] = None,
    walls: Optional[Sequence[Any]] = None,
    serializer_overlap_area: float = 0.0,
    epsilon_area: float = CORE_OVERLAP_EPSILON_AREA,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "floor_id": floor_id,
        "topology_mode": topology_mode,
        "core_contract_id": getattr(core_contract, "core_contract_id", None),
        "core_union_hash": getattr(core_contract, "core_union_hash", None),
        "core_union_bounds": getattr(core_contract, "core_union_bounds", None),
        "core_union_area": float(getattr(getattr(core_contract, "core_union", None), "area", 0.0) or 0.0),
        "room_core_overlap_total": 0.0,
        "generated_room_core_overlap_total": 0.0,
        "coverage_feature_core_overlap_total": 0.0,
        "corridor_core_overlap_total": 0.0,
        "core_access_doors": [],
        "core_internal_doors": [],
        "room_core_doors": [],
        "wall_core_2d_fallback_count": 0,
        "tiny_wall_count": 0,
        "outside_wall_area": 0.0,
        "serializer_overlap_area": float(serializer_overlap_area or 0.0),
        "valid_core_geometry": True,
        "overlap_records": [],
    }
    if core_contract is None:
        diag["valid_core_geometry"] = False
        diag["failure_reason"] = "missing_core_contract"
        return diag

    groups = [
        ("room", rooms or [], "room_core_overlap_total"),
        ("generated_room", generated_rooms or [], "generated_room_core_overlap_total"),
        ("coverage_feature", coverage_features or [], "coverage_feature_core_overlap_total"),
        ("corridor", corridors or [], "corridor_core_overlap_total"),
    ]
    for object_type, items, total_key in groups:
        total = 0.0
        for idx, item in enumerate(items):
            rec = _overlap_record(
                obj=item,
                object_type=object_type,
                core_contract=core_contract,
                epsilon_area=epsilon_area,
                default_id=f"{object_type}_{idx}",
            )
            if rec is not None:
                diag["overlap_records"].append(rec)
                total += float(rec.get("overlap_area", 0.0) or 0.0)
        diag[total_key] = total

    if diag["overlap_records"]:
        diag["valid_core_geometry"] = False

    door_diag = classify_core_doors(doors or [], zone_types={}, core_contract=core_contract)
    diag.update({
        "core_access_doors": door_diag["core_access_doors"],
        "core_internal_doors": door_diag["core_internal_doors"],
        "room_core_doors": door_diag["room_core_doors"],
    })
    return diag


def validate_core_exclusion(
    *,
    floor_id: str,
    topology_mode: str,
    core_contract: Optional[CoreFootprintContract],
    rooms: Optional[Sequence[Any]] = None,
    generated_rooms: Optional[Sequence[Any]] = None,
    coverage_features: Optional[Sequence[Any]] = None,
    corridors: Optional[Sequence[Any]] = None,
    epsilon_area: float = CORE_OVERLAP_EPSILON_AREA,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    diagnostics = collect_core_geometry_diagnostics(
        floor_id=floor_id,
        topology_mode=topology_mode,
        core_contract=core_contract,
        rooms=rooms,
        generated_rooms=generated_rooms,
        coverage_features=coverage_features,
        corridors=corridors,
        epsilon_area=epsilon_area,
    )
    logger.info(
        "[CORE] Diagnostics | floor=%s | contract=%s | room_core_overlap=%.4f | corridor_core_overlap=%.4f | coverage_feature_core_overlap=%.4f | core_access_doors=%d",
        floor_id,
        diagnostics.get("core_contract_id"),
        float(diagnostics.get("room_core_overlap_total", 0.0) or 0.0),
        float(diagnostics.get("corridor_core_overlap_total", 0.0) or 0.0),
        float(diagnostics.get("coverage_feature_core_overlap_total", 0.0) or 0.0),
        len(diagnostics.get("core_access_doors", []) or []),
    )
    overlaps = list(diagnostics.get("overlap_records", []) or [])
    if overlaps and hard_fail:
        max_rec = max(overlaps, key=lambda r: float(r.get("overlap_area", 0.0) or 0.0))
        raise LayoutGeometryInvariantError(
            "Core exclusion failed: non-core geometry overlaps core footprint",
            floor_id=floor_id,
            stage="core_exclusion_failed",
            metadata={
                "floor_id": floor_id,
                "topology_mode": topology_mode,
                "core_contract_id": diagnostics.get("core_contract_id"),
                "core_union_hash": diagnostics.get("core_union_hash"),
                "object_type_pair": max_rec.get("object_type_pair"),
                "offending_objects": overlaps,
                "overlap_area": float(max_rec.get("overlap_area", 0.0) or 0.0),
                "overlap_bbox": max_rec.get("overlap_bbox"),
                "threshold": float(epsilon_area),
                "semantic_repair_allowed": False,
            },
        )
    if overlaps:
        logger.warning(
            "[CORE] Exclusion warning | floor=%s | contract=%s | overlaps=%d",
            floor_id,
            diagnostics.get("core_contract_id"),
            len(overlaps),
        )
    else:
        logger.info("[CORE] Exclusion pass | floor=%s | contract=%s", floor_id, diagnostics.get("core_contract_id"))
    return diagnostics


def _norm_zone_type(zone_id: str, zone_types: Dict[str, str]) -> str:
    raw = str(zone_types.get(str(zone_id), "") or "").lower()
    zid = str(zone_id).lower()
    if raw:
        return raw
    if "corridor" in zid:
        return "corridor"
    if "elevator_hall" in zid:
        return "elevator_hall"
    if "staircase_hall" in zid:
        return "staircase_hall"
    if "elevator_shaft" in zid:
        return "elevator_shaft"
    if "staircase_shaft" in zid:
        return "staircase_shaft"
    return "room"


def _is_public_corridor(zone_id: str, zone_types: Dict[str, str]) -> bool:
    zt = _norm_zone_type(zone_id, zone_types)
    return zt in {"corridor", "main_corridor", "public_corridor", "corridor_grid_main"} or "corridor" in str(zone_id).lower()


def classify_core_doors(
    doors: Sequence[Any],
    *,
    zone_types: Optional[Dict[str, str]],
    core_contract: Optional[CoreFootprintContract],
    min_width: float = MIN_CORE_DOOR_WIDTH,
) -> Dict[str, List[Dict[str, Any]]]:
    out = {
        "core_access_doors": [],
        "core_internal_doors": [],
        "room_core_doors": [],
        "corridor_to_core_shaft_doors": [],
        "missing_portal_binding_doors": [],
    }
    if core_contract is None:
        return out
    zt = dict(zone_types or {})
    public_ids = set(core_contract.public_hall_ids)
    shaft_ids = set(core_contract.shaft_ids)
    core_ids = public_ids | shaft_ids
    for index, door in enumerate(doors or []):
        con = list(getattr(door, "connects", []) or (door.get("connects", []) if isinstance(door, dict) else []))
        if len(con) != 2:
            continue
        a, b = str(con[0]), str(con[1])
        width = float(getattr(door, "width", None) or (door.get("width", 0.0) if isinstance(door, dict) else 0.0) or 0.0)
        portal_id = getattr(door, "source_portal_spec_id", None)
        if isinstance(door, dict):
            portal_id = door.get("source_portal_spec_id", portal_id)
        rec = {"door_index": index, "connects": [a, b], "width": width, "source_portal_spec_id": portal_id}
        a_core = a in core_ids
        b_core = b in core_ids
        if not a_core and not b_core:
            continue
        if a_core and b_core:
            out["core_internal_doors"].append(rec)
            continue
        core_id = a if a_core else b
        other_id = b if a_core else a
        if core_id in shaft_ids:
            out["corridor_to_core_shaft_doors"].append(rec)
            continue
        if _is_public_corridor(other_id, zt) and width >= float(min_width):
            out["core_access_doors"].append(rec)
            if not portal_id:
                out["missing_portal_binding_doors"].append(rec)
            continue
        out["room_core_doors"].append(rec)
    return out


def validate_core_access(
    *,
    floor_id: str,
    topology_mode: str,
    core_contract: Optional[CoreFootprintContract],
    doors: Sequence[Any],
    zone_types: Dict[str, str],
    min_width: float = MIN_CORE_DOOR_WIDTH,
    hard_fail: bool = True,
    require_portal_binding: bool = False,
) -> Dict[str, Any]:
    door_diag = classify_core_doors(
        doors,
        zone_types=zone_types,
        core_contract=core_contract,
        min_width=min_width,
    )
    access = list(door_diag.get("core_access_doors", []) or [])
    core_contract_id = getattr(core_contract, "core_contract_id", None)
    if access and (not require_portal_binding or not door_diag.get("missing_portal_binding_doors")):
        logger.info(
            "[CORE] Access pass | floor=%s | contract=%s | access_doors=%d",
            floor_id,
            core_contract_id,
            len(access),
        )
        return {
            "floor_id": floor_id,
            "topology_mode": topology_mode,
            "core_contract_id": core_contract_id,
            "valid_core_access": True,
            **door_diag,
        }

    message = "Core access failed: no public corridor door to core public hall"
    if access and require_portal_binding:
        message = "Core access failed: core door missing portal binding"
    metadata = {
        "floor_id": floor_id,
        "topology_mode": topology_mode,
        "core_contract_id": core_contract_id,
        "valid_core_access": False,
        "core_access_doors": access,
        "core_internal_doors": door_diag.get("core_internal_doors", []),
        "room_core_doors": door_diag.get("room_core_doors", []),
        "corridor_to_core_shaft_doors": door_diag.get("corridor_to_core_shaft_doors", []),
        "missing_portal_binding_doors": door_diag.get("missing_portal_binding_doors", []),
        "threshold": float(min_width),
        "semantic_repair_allowed": False,
    }
    if hard_fail:
        raise LayoutGeometryInvariantError(
            message,
            floor_id=floor_id,
            stage="core_access_failed",
            metadata=metadata,
        )
    logger.warning(
        "[CORE] Access warning | floor=%s | contract=%s | access_doors=%d | room_core_doors=%d",
        floor_id,
        core_contract_id,
        len(access),
        len(door_diag.get("room_core_doors", []) or []),
    )
    return metadata


def reconcile_core_area_for_budget(
    *,
    floor_id: str,
    topology_mode: str,
    core_contract: Optional[CoreFootprintContract],
    core_tube_area: Optional[float],
    hard_fail: bool = False,
) -> Dict[str, Any]:
    area = float(getattr(getattr(core_contract, "core_union", None), "area", 0.0) or 0.0)
    budget_area = float(core_tube_area or 0.0)
    diff = abs(area - budget_area)
    threshold = max(CORE_BUDGET_AREA_MISMATCH_ABS, budget_area * CORE_BUDGET_AREA_MISMATCH_RATIO)
    meta = {
        "floor_id": floor_id,
        "topology_mode": topology_mode,
        "core_contract_id": getattr(core_contract, "core_contract_id", None),
        "core_union_hash": getattr(core_contract, "core_union_hash", None),
        "core_union_bounds": getattr(core_contract, "core_union_bounds", None),
        "core_union_area": area,
        "budget_core_tube_area": budget_area,
        "area_delta": diff,
        "threshold": threshold,
        "reconciled_core_area": area,
    }
    if diff > threshold:
        msg = "Core footprint area differs from physical budget core area"
        if hard_fail:
            raise LayoutGeometryInvariantError(
                msg,
                floor_id=floor_id,
                stage="core_footprint_invalid",
                metadata={**meta, "semantic_repair_allowed": False},
            )
        logger.warning("[CORE] Budget mismatch | %s", meta)
    else:
        logger.info(
            "[CORE] Budget reconciled | floor=%s | contract=%s | core_union_area=%.3f | budget_core_area=%.3f",
            floor_id,
            meta["core_contract_id"],
            area,
            budget_area,
        )
    return meta


def validate_wall_mesh_qa(
    *,
    floor_id: str,
    topology_mode: str,
    walls: Sequence[Any],
    floor_boundary: Polygon,
    wall_thickness: float = 0.12,
    max_tiny_wall_count: int = 4,
    outside_wall_epsilon_area: float = 0.005,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    tiny_threshold = max(0.15, float(wall_thickness) * 1.25)
    tiny_walls: List[Dict[str, Any]] = []
    outside_area = 0.0
    outside_walls: List[Dict[str, Any]] = []
    for idx, wall in enumerate(walls or []):
        geom = getattr(wall, "geometry", None)
        if geom is None or getattr(geom, "is_empty", True):
            continue
        try:
            minx, miny, maxx, maxy = geom.bounds
            width = abs(float(maxx) - float(minx))
            height = abs(float(maxy) - float(miny))
            if width <= tiny_threshold and height <= tiny_threshold:
                tiny_walls.append({
                    "wall_index": idx,
                    "room_ids": list(getattr(wall, "room_ids", []) or []),
                    "bbox": (round(float(minx), 4), round(float(miny), 4), round(float(maxx), 4), round(float(maxy), 4)),
                })
        except Exception:
            pass
        try:
            buffered = geom.buffer(
                float(getattr(wall, "thickness", wall_thickness) or wall_thickness) / 2.0,
                cap_style=2,
                join_style=2,
            )
            outside = buffered.difference(floor_boundary)
            area = float(getattr(outside, "area", 0.0) or 0.0)
        except Exception:
            area = 0.0
        if area > float(outside_wall_epsilon_area):
            outside_area += area
            try:
                bbox = tuple(round(float(v), 4) for v in buffered.bounds) if buffered is not None else None
            except Exception:
                bbox = None
            outside_walls.append({
                "wall_index": idx,
                "room_ids": list(getattr(wall, "room_ids", []) or []),
                "outside_area": area,
                "bbox": bbox,
            })

    diagnostics = {
        "floor_id": floor_id,
        "topology_mode": topology_mode,
        "tiny_wall_count": len(tiny_walls),
        "max_tiny_wall_count": int(max_tiny_wall_count),
        "tiny_wall_size_threshold": float(tiny_threshold),
        "outside_wall_area": float(outside_area),
        "outside_wall_epsilon_area": float(outside_wall_epsilon_area),
        "tiny_walls": tiny_walls[:20],
        "outside_walls": outside_walls[:20],
        "valid_wall_mesh": not (len(tiny_walls) > int(max_tiny_wall_count) or outside_area > float(outside_wall_epsilon_area)),
    }
    logger.info(
        "[WALL_QA] Checked | floor=%s | tiny=%d/%d | outside_area=%.4f | valid=%s",
        floor_id,
        len(tiny_walls),
        int(max_tiny_wall_count),
        float(outside_area),
        diagnostics["valid_wall_mesh"],
    )
    if hard_fail and not diagnostics["valid_wall_mesh"]:
        stage_reason = "outside_wall" if outside_area > float(outside_wall_epsilon_area) else "tiny_wall_explosion"
        raise LayoutGeometryInvariantError(
            "Wall mesh QA failed",
            floor_id=floor_id,
            stage="wall_mesh_qa_failed",
            metadata={
                **diagnostics,
                "failure_reason": stage_reason,
                "threshold": float(outside_wall_epsilon_area)
                if stage_reason == "outside_wall"
                else int(max_tiny_wall_count),
                "semantic_repair_allowed": False,
            },
        )
    return diagnostics


