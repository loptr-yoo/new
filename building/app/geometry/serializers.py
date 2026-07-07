"""
serializers.py

将几何引擎输出（Shapely 对象）序列化为 JSON 可传输格式。
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .building_orchestrator import BuildingResult
from .core_contracts import (
    CORE_OVERLAP_EPSILON_AREA,
    build_core_footprint_contract,
    validate_core_exclusion,
    validate_wall_mesh_qa,
)
from .coord_transform import math_to_screen_element, to_screen_forward, to_screen_point, to_screen_polygon
from .layout_generator import LayoutResultV2, RoomResult
from .postprocessor import (
    LayoutCoverageError,
    LayoutTopologyError,
    PostprocessResult,
    door_to_dict,
    generate_wall_mesh,
    postprocess_floor,
    safe_snap_polygon,
    wall_to_dict,
    window_to_dict,
)
from .topology_generator import CoreTube, Corridor

logger = logging.getLogger(__name__)

try:
    from shapely.validation import make_valid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore[assignment]


def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and (not g.is_empty)]
        if not polys:
            return None
        return max(polys, key=lambda g: float(g.area))
    return None


def _clip_to_slab(geom, slab: Polygon) -> Optional[Polygon]:
    try:
        clipped = geom.intersection(slab)
    except Exception:
        return None
    try:
        if (not clipped.is_empty) and (not clipped.is_valid) and make_valid is not None:
            clipped = make_valid(clipped)
    except Exception:
        pass
    return _largest_polygon(clipped)


def _round_polygon_coords(poly: Polygon, ndigits: int) -> Polygon:
    ext = [(round(float(x), ndigits), round(float(y), ndigits)) for x, y in poly.exterior.coords]
    holes = [
        [(round(float(x), ndigits), round(float(y), ndigits)) for x, y in ring.coords]
        for ring in poly.interiors
    ]
    return Polygon(ext, holes)


def _snap_for_export(poly: Optional[BaseGeometry], tol: float = 0.01) -> Optional[Polygon]:
    if poly is None:
        return None
    if getattr(poly, "is_empty", True):
        return None
    snapped = safe_snap_polygon(poly, float(tol))
    if snapped is None or snapped.is_empty:
        return None
    return snapped


def room_result_to_dict(room: RoomResult, floor_id: str, *, floor_bounds: Tuple[float, float, float, float]) -> dict:
    """RoomResult (Shapely) → 可序列化 dict"""
    if room.polygon.is_empty:
        return {
            "room_id": room.id,
            "room_type": room.room_type,
            "floor_id": floor_id,
            "polygon": [],
            "center": [0, 0],
            "area": 0,
            "width": 0,
            "depth": 0,
            "target_area": round(room.target_area, 2),
            "has_window": room.has_window,
            "is_dummy": bool(getattr(room, "is_dummy", False)),
        }

    slab = box(*floor_bounds)
    clipped = _clip_to_slab(room.polygon, slab) or room.polygon
    snapped = _snap_for_export(clipped, tol=0.01)
    if snapped is None:
        return {
            "room_id": room.id,
            "room_type": room.room_type,
            "floor_id": floor_id,
            "polygon": [],
            "center": [0, 0],
            "area": 0,
            "width": 0,
            "depth": 0,
            "target_area": round(room.target_area, 2),
            "has_window": room.has_window,
            "is_dummy": bool(getattr(room, "is_dummy", False)),
        }
    clipped = _round_polygon_coords(snapped, 4)
    coords = list(clipped.exterior.coords)
    minx, miny, maxx, maxy = clipped.bounds
    poly = to_screen_polygon(coords, bounds=floor_bounds)
    centroid = clipped.centroid
    cx, cy = to_screen_point(float(centroid.x), float(centroid.y), bounds=floor_bounds)
    actual_area = float(clipped.area)
    is_dummy = bool(getattr(room, "is_dummy", False))
    solid = bool(is_dummy and str(room.room_type).lower() in ("utility", "void"))
    tar_raw = getattr(room, "target_area_raw", None)
    return {
        "room_id": room.id,
        "room_type": room.room_type,
        "floor_id": floor_id,
        "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
        "center": [round(float(cx), 4), round(float(cy), 4)],
        "area": round(actual_area, 2),
        "actual_area": round(actual_area, 4),
        "width": round(float(maxx - minx), 4),
        "depth": round(float(maxy - miny), 4),
        "target_area": round(room.target_area, 2),
        "has_window": room.has_window,
        "is_dummy": is_dummy,
        "target_area_raw": round(float(tar_raw), 2) if (is_dummy and tar_raw is not None) else None,
        "no_wall_inset": solid,
    }


def core_tube_to_dict(core_tube: Any, *, floor_bounds: Tuple[float, float, float, float]) -> dict:
    """CoreTube → 可序列化 dict"""
    if not hasattr(core_tube, "polygon"):
        return {}

    slab = box(*floor_bounds)
    core_poly = _clip_to_slab(core_tube.polygon, slab) or core_tube.polygon
    ccx, ccy = to_screen_point(float(core_tube.center[0]), float(core_tube.center[1]), bounds=floor_bounds)
    snapped_core = _snap_for_export(core_poly, tol=0.01)
    if snapped_core is None:
        result: dict = {
            "boundary": [],
            "center": [round(float(ccx), 4), round(float(ccy), 4)],
            "area": 0,
            "actual_area": 0,
        }
    else:
        core_poly = _round_polygon_coords(snapped_core, 4)
        coords = list(core_poly.exterior.coords)
        boundary = to_screen_polygon(coords, bounds=floor_bounds)
        result = {
            "boundary": [[round(float(x), 4), round(float(y), 4)] for x, y in boundary],
            "center": [round(float(ccx), 4), round(float(ccy), 4)],
            "area": round(core_poly.area, 2),
            "actual_area": round(core_poly.area, 4),
        }
    if getattr(core_tube, "core_contract_id", None):
        result["core_contract_id"] = str(getattr(core_tube, "core_contract_id"))
    if getattr(core_tube, "core_contract_version", None):
        result["core_contract_version"] = str(getattr(core_tube, "core_contract_version"))

    if hasattr(core_tube, "elevator") and core_tube.elevator is not None:
        g = _clip_to_slab(core_tube.elevator, slab) or core_tube.elevator
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["elevator"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }

    if hasattr(core_tube, "staircase") and core_tube.staircase is not None:
        g = _clip_to_slab(core_tube.staircase, slab) or core_tube.staircase
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["staircase"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }

    if hasattr(core_tube, "staircase_hall") and core_tube.staircase_hall is not None:
        g = _clip_to_slab(core_tube.staircase_hall, slab) or core_tube.staircase_hall
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["staircase_hall"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }
    if hasattr(core_tube, "staircase_hall_b") and getattr(core_tube, "staircase_hall_b", None) is not None:
        g = _clip_to_slab(core_tube.staircase_hall_b, slab) or core_tube.staircase_hall_b
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["staircase_hall_b"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }

    if hasattr(core_tube, "staircase_shaft") and core_tube.staircase_shaft is not None:
        g = _clip_to_slab(core_tube.staircase_shaft, slab) or core_tube.staircase_shaft
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["staircase_shaft"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }

    if hasattr(core_tube, "elevator_hall") and core_tube.elevator_hall is not None:
        g = _clip_to_slab(core_tube.elevator_hall, slab) or core_tube.elevator_hall
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["elevator_hall"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }
    if hasattr(core_tube, "elevator_hall_b") and getattr(core_tube, "elevator_hall_b", None) is not None:
        g = _clip_to_slab(core_tube.elevator_hall_b, slab) or core_tube.elevator_hall_b
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["elevator_hall_b"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }

    if hasattr(core_tube, "elevator_shaft") and core_tube.elevator_shaft is not None:
        g = _clip_to_slab(core_tube.elevator_shaft, slab) or core_tube.elevator_shaft
        snapped = _snap_for_export(g, tol=0.01)
        if snapped is not None:
            g2 = _round_polygon_coords(snapped, 4)
            poly = to_screen_polygon(list(g2.exterior.coords), bounds=floor_bounds)
            result["elevator_shaft"] = {
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "area": round(g2.area, 2),
                "actual_area": round(g2.area, 4),
            }

    return result


def corridor_to_dict(corridor: Corridor, *, floor_bounds: Tuple[float, float, float, float]) -> dict:
    """Corridor → 可序列化 dict"""
    slab = box(*floor_bounds)
    clipped = _clip_to_slab(corridor.polygon, slab) or corridor.polygon
    snapped = _snap_for_export(clipped, tol=0.01)
    if snapped is None:
        return {
            "id": corridor.id,
            "type": "corridor",
            "polygon": [],
            "center": [0.0, 0.0],
            "width": 0,
            "depth": 0,
            "area": 0,
            "actual_area": 0,
            "orientation": corridor.orientation,
        }
    clipped = _round_polygon_coords(snapped, 4)
    coords = list(clipped.exterior.coords)
    minx, miny, maxx, maxy = clipped.bounds
    poly = to_screen_polygon(coords, bounds=floor_bounds)
    ccx, ccy = to_screen_point((minx + maxx) / 2.0, (miny + maxy) / 2.0, bounds=floor_bounds)
    return {
        "id": corridor.id,
        "type": "corridor",
        "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
        "center": [
            round(float(ccx), 4),
            round(float(ccy), 4),
        ],
        "width": round(float(maxx - minx), 4),
        "depth": round(float(maxy - miny), 4),
        "area": round(float(clipped.area), 2),
        "actual_area": round(float(clipped.area), 4),
        "orientation": corridor.orientation,
    }


def floor_slab_to_dict(floor_boundary: Polygon, *, floor_bounds: Tuple[float, float, float, float]) -> dict:
    """楼层边界 → floor_slab 元素 dict"""
    snapped = _snap_for_export(floor_boundary, tol=0.01)
    if snapped is None:
        return {
            "id": "floor_slab",
            "type": "floor_slab",
            "polygon": [],
            "center": [0.0, 0.0],
            "width": 0,
            "depth": 0,
            "area": 0,
            "actual_area": 0,
        }
    floor_boundary = _round_polygon_coords(snapped, 4)
    coords = list(floor_boundary.exterior.coords)
    minx, miny, maxx, maxy = floor_boundary.bounds
    poly = to_screen_polygon(coords, bounds=floor_bounds)
    ccx, ccy = to_screen_point((minx + maxx) / 2.0, (miny + maxy) / 2.0, bounds=floor_bounds)
    return {
        "id": "floor_slab",
        "type": "floor_slab",
        "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
        "center": [round(float(ccx), 4), round(float(ccy), 4)],
        "width": round(float(maxx - minx), 4),
        "depth": round(float(maxy - miny), 4),
        "area": round(float(floor_boundary.area), 2),
        "actual_area": round(float(floor_boundary.area), 4),
    }


def _postprocess_to_dict(pp: PostprocessResult, *, floor_bounds: Tuple[float, float, float, float]) -> dict:
    """PostprocessResult → 可序列化 dict"""
    walls = []
    for w in pp.walls:
        d = wall_to_dict(w)
        d = math_to_screen_element(d, bounds=floor_bounds)
        poly = d.get("polygon")
        if isinstance(poly, list):
            d["polygon"] = [[round(float(x), 2), round(float(y), 2)] for x, y in poly]
        coords = d.get("coords")
        if isinstance(coords, list):
            d["coords"] = [[round(float(x), 2), round(float(y), 2)] for x, y in coords]
        walls.append(d)

    doors = []
    for d0 in pp.doors:
        d = door_to_dict(d0)
        d["type"] = "door"
        d = math_to_screen_element(d, bounds=floor_bounds)
        pos = d.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            d["position"] = [round(float(pos[0]), 2), round(float(pos[1]), 2)]
        doors.append(d)

    windows = []
    for w0 in pp.windows:
        d = window_to_dict(w0)
        d["type"] = "window"
        d = math_to_screen_element(d, bounds=floor_bounds)
        pos = d.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            d["position"] = [round(float(pos[0]), 2), round(float(pos[1]), 2)]
        windows.append(d)

    return {"walls": walls, "doors": doors, "windows": windows}


def _floor_number_from_id(fid: str) -> int:
    m = re.match(r"^f(\d+)$", (fid or "").strip().lower())
    return int(m.group(1)) if m else 10**9


def _center_from_poly(poly: Any) -> Optional[Tuple[float, float]]:
    try:
        if poly is None or poly.is_empty:
            return None
        c = poly.centroid
        return (float(c.x), float(c.y))
    except Exception:
        return None


def _calculate_forward(p1: Tuple[float, float], p2: Tuple[float, float]) -> List[float]:
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    length = float(math.hypot(dx, dy))
    if length < 1e-3:
        return [1.0, 0.0, 0.0]
    return [float(dx / length), float(dy / length), 0.0]


def serialize_single_floor(
    *,
    floor_id: str,
    floor_number: int,
    layout: LayoutResultV2,
    floor_boundary: Polygon,
    wall_graph: Any = None,
    wall_mesh: Any = None,
    render_assets: Any = None,
    topology_meta: Optional[dict] = None,
    ground_floor_id: Optional[str] = None,
    core_forward: Optional[Dict[str, List[float]]] = None,
    diagnostics: Optional[dict] = None,
) -> dict:
    """Serialize one floor using the same schema as building_result_to_dict()."""
    del wall_graph, wall_mesh, render_assets  # reserved for future external injection
    logger.info("[SERIALIZER] Serializing floor=%s", floor_id)
    b = floor_boundary.bounds
    floor_bounds: Tuple[float, float, float, float] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    floor_warnings = list(getattr(layout, "warnings", []) or [])

    rooms_in = list(getattr(layout, "room_layouts", []) or [])
    corridors_in = list(getattr(layout, "corridors", []) or [])

    if str(getattr(layout, "corridor_layout", "") or "").lower() == "organic":
        try:
            from .postprocessor import fuse_dummy_to_corridor
            rooms_in, corridors_in, fuse_warnings = fuse_dummy_to_corridor(
                rooms=rooms_in,
                corridors=corridors_in,
            )
            if fuse_warnings:
                floor_warnings.extend(list(fuse_warnings))
                logger.warning(
                    "[SERIALIZER] Dummy fuse warnings | floor=%s | warnings=%s",
                    floor_id,
                    list(fuse_warnings),
                )
        except Exception:
            logger.warning("[SERIALIZER] Dummy fuse failed | floor=%s", floor_id, exc_info=True)

    rooms = [room_result_to_dict(r, floor_id, floor_bounds=floor_bounds) for r in rooms_in]

    corridors = []
    for c in corridors_in:
        try:
            corridors.append(corridor_to_dict(c, floor_bounds=floor_bounds))
        except Exception:
            pass

    slab = floor_slab_to_dict(floor_boundary, floor_bounds=floor_bounds)
    room_rects: Dict[str, tuple] = {}
    zone_types: Dict[str, str] = {}

    if layout.core_tube is not None and hasattr(layout.core_tube, "polygon") and not layout.core_tube.polygon.is_empty:
        ct = layout.core_tube
        subzones = [
            ("core_staircase_hall", "staircase_hall", getattr(ct, "staircase_hall", None)),
            ("core_staircase_hall_b", "staircase_hall", getattr(ct, "staircase_hall_b", None)),
            ("core_staircase_shaft", "staircase_shaft", getattr(ct, "staircase_shaft", None)),
            ("core_elevator_hall", "elevator_hall", getattr(ct, "elevator_hall", None)),
            ("core_elevator_hall_b", "elevator_hall", getattr(ct, "elevator_hall_b", None)),
            ("core_elevator_shaft", "elevator_shaft", getattr(ct, "elevator_shaft", None)),
        ]
        required = {
            "core_staircase_hall",
            "core_staircase_shaft",
            "core_elevator_hall",
            "core_elevator_shaft",
        }
        has_subzones = True
        for zid, _zt, poly in subzones:
            if zid not in required:
                continue
            if poly is None or (hasattr(poly, "is_empty") and poly.is_empty):
                has_subzones = False
                break
        if has_subzones:
            for zid, zt, poly in subzones:
                if poly is None or poly.is_empty:
                    continue
                minx, miny, maxx, maxy = poly.bounds
                room_rects[zid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
                zone_types[zid] = zt
        else:
            minx, miny, maxx, maxy = ct.polygon.bounds
            room_rects["core_tube"] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
            zone_types["core_tube"] = "core"

    for c in corridors_in:
        if hasattr(c, "polygon") and not c.polygon.is_empty:
            minx, miny, maxx, maxy = c.polygon.bounds
            room_rects[c.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
            zone_types[c.id] = "corridor"

    for r in rooms_in:
        if r.polygon.is_empty:
            continue
        if str(getattr(r, "room_type", "") or "").lower() == "void" or bool(getattr(r, "skip_solver", False)):
            continue
        minx, miny, maxx, maxy = r.polygon.bounds
        room_rects[r.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
        zone_types[r.id] = str(getattr(r, "room_type", "") or "room").lower()

    solver_metadata = dict(getattr(layout, "solver_metadata", {}) or {})
    topology_mode_l = str(
        (topology_meta or {}).get("topology_mode")
        or (solver_metadata.get("core_contract", {}) or {}).get("topology_mode")
        or ("grid_growth" if "grid_growth" in solver_metadata else "")
        or ""
    ).lower()
    core_contract = None
    if layout.core_tube is not None:
        try:
            core_contract = build_core_footprint_contract(
                layout.core_tube,
                floor_id=floor_id,
                topology_mode=topology_mode_l or "unknown",
                created_from="serializer",
            )
            core_exclusion = validate_core_exclusion(
                floor_id=floor_id,
                topology_mode=topology_mode_l or "unknown",
                core_contract=core_contract,
                rooms=rooms_in,
                corridors=corridors_in,
                coverage_features=[],
                epsilon_area=CORE_OVERLAP_EPSILON_AREA,
                hard_fail=(topology_mode_l == "grid_growth"),
            )
            solver_metadata.setdefault("core_contract", {})["serializer_exclusion_diagnostics"] = core_exclusion
            if "floor_free_space" in solver_metadata:
                solver_metadata["floor_free_space"]["serializer_core_contract_id"] = core_contract.core_contract_id
        except (LayoutCoverageError, LayoutTopologyError):
            raise
        except Exception:
            if topology_mode_l == "grid_growth":
                raise
            logger.warning("[SERIALIZER] Core exclusion diagnostics failed | floor=%s", floor_id, exc_info=True)

    try:
        floor_area = float(floor_boundary.area)
        space_area = 0.0
        if layout.core_tube is not None and hasattr(layout.core_tube, "polygon") and not layout.core_tube.polygon.is_empty:
            space_area += float(layout.core_tube.polygon.area)
        for c in corridors_in:
            if hasattr(c, "polygon") and c.polygon is not None and not c.polygon.is_empty:
                space_area += float(c.polygon.area)
        for r in rooms_in:
            if hasattr(r, "polygon") and not r.polygon.is_empty:
                space_area += float(r.polygon.area)
        gap_area = floor_area - space_area
        if gap_area > 0.05:
            floor_warnings.append(f"Coverage gap area={gap_area:.3f}m^2 (floor - sum(spaces))")
            logger.warning(
                "[SERIALIZER] Coverage approximation gap | floor=%s | gap=%.3fm2 | floor_area=%.3fm2 | space_area=%.3fm2",
                floor_id,
                gap_area,
                floor_area,
                space_area,
            )
        elif gap_area < -0.05:
            floor_warnings.append(f"Coverage overlap area={(-gap_area):.3f}m^2 (sum(spaces) - floor)")
            logger.warning(
                "[SERIALIZER] Coverage approximation overlap | floor=%s | overlap=%.3fm2 | floor_area=%.3fm2 | space_area=%.3fm2",
                floor_id,
                -gap_area,
                floor_area,
                space_area,
            )
            if topology_mode_l == "grid_growth" and core_contract is not None and (-gap_area) > 0.05:
                raise LayoutCoverageError(
                    "Serializer geometry overlap exceeds grid_growth artifact tolerance",
                    floor_id=floor_id,
                    metadata={
                        "failure_kind": "geometry_invariant",
                        "stage": "serializer_geometry_overlap_failed",
                        "topology_mode": topology_mode_l,
                        "core_contract_id": getattr(core_contract, "core_contract_id", None),
                        "overlap_area": float(-gap_area),
                        "threshold": 0.05,
                        "semantic_repair_allowed": False,
                    },
                    stage="serializer_geometry_overlap_failed",
                    semantic_repair_allowed=False,
                )
    except (LayoutCoverageError, LayoutTopologyError):
        raise
    except Exception:
        logger.debug("[SERIALIZER] Coverage approximation failed | floor=%s", floor_id, exc_info=True)

    try:
        logger.info("[SERIALIZER] Generate wall mesh | floor=%s | rooms=%d | corridors=%d", floor_id, len(rooms_in), len(corridors_in))
        pp_walls = generate_wall_mesh(
            rooms=rooms_in,
            corridors=corridors_in,
            core_tube=layout.core_tube,
            floor_boundary=floor_boundary,
        )
        wall_qa = validate_wall_mesh_qa(
            floor_id=floor_id,
            topology_mode=topology_mode_l or "unknown",
            walls=pp_walls,
            floor_boundary=floor_boundary,
            hard_fail=(topology_mode_l == "grid_growth"),
        )
        solver_metadata.setdefault("core_contract", {})["wall_mesh_qa"] = wall_qa
    except (LayoutCoverageError, LayoutTopologyError) as e:
        if hasattr(e, "with_floor"):
            e.with_floor(int(floor_number), floor_id)
        raise

    rooms_needing_window = set()
    for room in rooms_in:
        room_id = getattr(room, "id", getattr(room, "room_id", "?"))
        if str(getattr(room, "room_type", "") or "").lower() == "void" or bool(getattr(room, "skip_solver", False)):
            continue
        has_window = getattr(room, "has_window", False) or getattr(room, "needs_window", False)
        if has_window:
            rooms_needing_window.add(room_id)

    is_ground_floor = bool(ground_floor_id and floor_id == ground_floor_id)
    try:
        logger.info("[SERIALIZER] Postprocess floor | floor=%s | walls=%d", floor_id, len(pp_walls))
        pp = postprocess_floor(
            rooms=rooms_in,
            floor_boundary=floor_boundary,
            corridors=corridors_in,
            core_tube=layout.core_tube,
            floor_id=floor_id,
            topology_mode=topology_mode_l,
            is_ground_floor=is_ground_floor,
            walls=pp_walls,
            zone_types=zone_types,
            zone_rects=room_rects,
            required_adjacency=getattr(layout, "required_adjacency", None),
            rooms_needing_window=rooms_needing_window,
            floor_bounds=floor_boundary.bounds,
        )
    except (LayoutCoverageError, LayoutTopologyError) as e:
        if hasattr(e, "with_floor"):
            e.with_floor(int(floor_number), floor_id)
        raise
    pp_screen = _postprocess_to_dict(pp, floor_bounds=floor_bounds)
    logger.info(
        "[SERIALIZER] Postprocess complete | floor=%s | walls=%d | doors=%d | windows=%d",
        floor_id,
        len(pp.walls),
        len(pp.doors),
        len(pp.windows),
    )

    if is_ground_floor and core_forward is not None and not core_forward and layout.core_tube is not None:
        corridor_centers: List[Tuple[float, float]] = []
        if corridors_in:
            for c in corridors_in:
                try:
                    poly = getattr(c, "polygon", None)
                    if poly is None or poly.is_empty:
                        continue
                    minx, miny, maxx, maxy = poly.bounds
                    corridor_centers.append((float((minx + maxx) / 2), float((miny + maxy) / 2)))
                except Exception:
                    continue
        default_dir = [1.0, 0.0, 0.0]
        ct = layout.core_tube
        ch = _center_from_poly(getattr(ct, "staircase_hall", None))
        cs = _center_from_poly(getattr(ct, "staircase_shaft", None))
        eh = _center_from_poly(getattr(ct, "elevator_hall", None))
        es = _center_from_poly(getattr(ct, "elevator_shaft", None))

        def _nearest_corr(p: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
            if p is None or not corridor_centers:
                return None
            best = None
            best_d = 1e18
            for cc in corridor_centers:
                d = (cc[0] - p[0]) ** 2 + (cc[1] - p[1]) ** 2
                if d < best_d:
                    best_d = d
                    best = cc
            return best

        if ch is not None:
            tgt = _nearest_corr(ch)
            core_forward["staircase_hall"] = _calculate_forward(ch, tgt) if tgt else default_dir
        if cs is not None:
            tgt = ch or _nearest_corr(cs)
            core_forward["staircase_shaft"] = _calculate_forward(cs, tgt) if tgt else default_dir
        if eh is not None:
            tgt = _nearest_corr(eh)
            core_forward["elevator_hall"] = _calculate_forward(eh, tgt) if tgt else default_dir
        if es is not None:
            tgt = eh or _nearest_corr(es)
            core_forward["elevator_shaft"] = _calculate_forward(es, tgt) if tgt else default_dir
        if "elevator_hall" in core_forward:
            core_forward["elevator"] = list(core_forward["elevator_hall"])
        if "staircase_hall" in core_forward:
            core_forward["staircase"] = list(core_forward["staircase_hall"])

    merged_diagnostics = dict(diagnostics or {})
    merged_diagnostics.setdefault("schema_version", "stage2a-floor-v1")
    merged_diagnostics.setdefault("serialization_source", "building_result")
    merged_diagnostics.setdefault("solver_metadata", solver_metadata)
    if topology_meta:
        merged_diagnostics.setdefault("topology_meta", topology_meta)

    return {
        "floor_name": floor_id,
        "floor_slab": slab,
        "corridors": corridors,
        "rooms": rooms,
        "walls": pp_screen["walls"],
        "doors": pp_screen["doors"],
        "windows": pp_screen["windows"],
        "generation_time_ms": round(layout.generation_time_ms, 1),
        "warnings": floor_warnings,
        "diagnostics": merged_diagnostics,
    }


def building_result_to_dict(
    result: BuildingResult,
    floor_boundary: Polygon,
) -> dict:
    """BuildingResult → 完整 API 响应格式"""
    logger.info("[SERIALIZER] Start building_result_to_dict | floors=%d", len(result.floor_layouts))
    b = floor_boundary.bounds
    floor_bounds: Tuple[float, float, float, float] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    minx, miny, maxx, maxy = floor_bounds

    floors: Dict[str, dict] = {}
    floor_ids = list(result.floor_layouts.keys())

    def _floor_number(fid: str) -> int:
        m = re.match(r"^f(\d+)$", (fid or "").strip().lower())
        return int(m.group(1)) if m else 10**9

    ground_floor_id = "F1" if "F1" in floor_ids else None
    if ground_floor_id is None and floor_ids:
        min_num = min((_floor_number(fid) for fid in floor_ids), default=10**9)
        for fid in floor_ids:
            if _floor_number(fid) == min_num:
                ground_floor_id = fid
                break

    core_forward: Dict[str, List[float]] = {}

    def _center_from_poly(poly: Any) -> Optional[Tuple[float, float]]:
        try:
            if poly is None or poly.is_empty:
                return None
            c = poly.centroid
            return (float(c.x), float(c.y))
        except Exception:
            return None

    def _calculate_forward(p1: Tuple[float, float], p2: Tuple[float, float]) -> List[float]:
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        length = float(math.hypot(dx, dy))
        if length < 1e-3:
            return [1.0, 0.0, 0.0]
        return [float(dx / length), float(dy / length), 0.0]

    for floor_id, layout in result.floor_layouts.items():
        floors[floor_id] = serialize_single_floor(
            floor_id=floor_id,
            floor_number=_floor_number(floor_id),
            layout=layout,
            floor_boundary=floor_boundary,
            ground_floor_id=ground_floor_id,
            core_forward=core_forward,
            diagnostics={"serialization_source": "building_result"},
        )
        continue
        logger.info("[SERIALIZER] Serializing floor=%s", floor_id)
        floor_warnings = list(getattr(layout, "warnings", []) or [])

        rooms_in = list(getattr(layout, "room_layouts", []) or [])
        corridors_in = list(getattr(layout, "corridors", []) or [])

        did_fuse = False
        if str(getattr(layout, "corridor_layout", "") or "").lower() == "organic":
            try:
                from .postprocessor import fuse_dummy_to_corridor
                rooms_in, corridors_in, fuse_warnings = fuse_dummy_to_corridor(
                    rooms=rooms_in,
                    corridors=corridors_in,
                )
                if fuse_warnings:
                    did_fuse = True
                    floor_warnings.extend(list(fuse_warnings))
                    logger.warning(
                        "[SERIALIZER] Dummy fuse warnings | floor=%s | warnings=%s",
                        floor_id,
                        list(fuse_warnings),
                    )
            except Exception:
                logger.warning("[SERIALIZER] Dummy fuse failed | floor=%s", floor_id, exc_info=True)
                pass

        rooms = [room_result_to_dict(r, floor_id, floor_bounds=floor_bounds) for r in rooms_in]

        corridors = []
        for c in corridors_in:
            try:
                corridors.append(corridor_to_dict(c, floor_bounds=floor_bounds))
            except Exception:
                pass

        # 楼板背景
        slab = floor_slab_to_dict(floor_boundary, floor_bounds=floor_bounds)

        room_rects: Dict[str, tuple] = {}
        zone_types: Dict[str, str] = {}

        if layout.core_tube is not None and hasattr(layout.core_tube, "polygon") and not layout.core_tube.polygon.is_empty:
            ct = layout.core_tube
            subzones = [
                ("core_staircase_hall", "staircase_hall", getattr(ct, "staircase_hall", None)),
                ("core_staircase_hall_b", "staircase_hall", getattr(ct, "staircase_hall_b", None)),
                ("core_staircase_shaft", "staircase_shaft", getattr(ct, "staircase_shaft", None)),
                ("core_elevator_hall", "elevator_hall", getattr(ct, "elevator_hall", None)),
                ("core_elevator_hall_b", "elevator_hall", getattr(ct, "elevator_hall_b", None)),
                ("core_elevator_shaft", "elevator_shaft", getattr(ct, "elevator_shaft", None)),
            ]
            required = {
                "core_staircase_hall",
                "core_staircase_shaft",
                "core_elevator_hall",
                "core_elevator_shaft",
            }
            has_subzones = True
            for zid, _zt, poly in subzones:
                if zid not in required:
                    continue
                if poly is None or (hasattr(poly, "is_empty") and poly.is_empty):
                    has_subzones = False
                    break
            if has_subzones:
                for zid, zt, poly in subzones:
                    if poly is None or poly.is_empty:
                        continue
                    minx, miny, maxx, maxy = poly.bounds
                    room_rects[zid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
                    zone_types[zid] = zt
            else:
                minx, miny, maxx, maxy = ct.polygon.bounds
                room_rects["core_tube"] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
                zone_types["core_tube"] = "core"

        for c in corridors_in:
            if hasattr(c, "polygon") and not c.polygon.is_empty:
                minx, miny, maxx, maxy = c.polygon.bounds
                room_rects[c.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
                zone_types[c.id] = "corridor"

        for r in rooms_in:
            if r.polygon.is_empty:
                continue
            if str(getattr(r, "room_type", "") or "").lower() == "void" or bool(getattr(r, "skip_solver", False)):
                continue
            minx, miny, maxx, maxy = r.polygon.bounds
            room_rects[r.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
            zone_types[r.id] = str(getattr(r, "room_type", "") or "room").lower()
        try:
            floor_area = float(floor_boundary.area)
            space_area = 0.0
            if layout.core_tube is not None and hasattr(layout.core_tube, "polygon") and not layout.core_tube.polygon.is_empty:
                space_area += float(layout.core_tube.polygon.area)
            for c in corridors_in:
                if hasattr(c, "polygon") and c.polygon is not None and not c.polygon.is_empty:
                    space_area += float(c.polygon.area)
            for r in rooms_in:
                if hasattr(r, "polygon") and not r.polygon.is_empty:
                    space_area += float(r.polygon.area)
            gap_area = floor_area - space_area
            if gap_area > 0.05:
                floor_warnings.append(f"Coverage gap area={gap_area:.3f}m^2 (floor - sum(spaces))")
                logger.warning(
                    "[SERIALIZER] Coverage approximation gap | floor=%s | gap=%.3fm2 | floor_area=%.3fm2 | space_area=%.3fm2",
                    floor_id,
                    gap_area,
                    floor_area,
                    space_area,
                )
            elif gap_area < -0.05:
                floor_warnings.append(f"Coverage overlap area={(-gap_area):.3f}m^2 (sum(spaces) - floor)")
                logger.warning(
                    "[SERIALIZER] Coverage approximation overlap | floor=%s | overlap=%.3fm2 | floor_area=%.3fm2 | space_area=%.3fm2",
                    floor_id,
                    -gap_area,
                    floor_area,
                    space_area,
                )
        except Exception:
            logger.debug("[SERIALIZER] Coverage approximation failed | floor=%s", floor_id, exc_info=True)
            pass

        try:
            logger.info("[SERIALIZER] Generate wall mesh | floor=%s | rooms=%d | corridors=%d", floor_id, len(rooms_in), len(corridors_in))
            pp_walls = generate_wall_mesh(
                rooms=rooms_in,
                corridors=corridors_in,
                core_tube=layout.core_tube,
                floor_boundary=floor_boundary,
            )
        except (LayoutCoverageError, LayoutTopologyError) as e:
            if hasattr(e, "with_floor"):
                e.with_floor(_floor_number(floor_id), floor_id)
            raise

        rooms_needing_window = set()
        for room in rooms_in:
            room_id = getattr(room, "id", getattr(room, "room_id", "?"))
            if str(getattr(room, "room_type", "") or "").lower() == "void" or bool(getattr(room, "skip_solver", False)):
                continue
            has_window = getattr(room, "has_window", False) or getattr(room, "needs_window", False)
            if has_window:
                rooms_needing_window.add(room_id)

        is_ground_floor = bool(ground_floor_id and floor_id == ground_floor_id)
        try:
            logger.info("[SERIALIZER] Postprocess floor | floor=%s | walls=%d", floor_id, len(pp_walls))
            pp = postprocess_floor(
                rooms=rooms_in,
                floor_boundary=floor_boundary,
                corridors=corridors_in,
                core_tube=layout.core_tube,
                is_ground_floor=is_ground_floor,
                walls=pp_walls,
                zone_types=zone_types,
                zone_rects=room_rects,
                required_adjacency=getattr(layout, "required_adjacency", None),
                rooms_needing_window=rooms_needing_window,
                floor_bounds=floor_boundary.bounds,
            )
        except (LayoutCoverageError, LayoutTopologyError) as e:
            if hasattr(e, "with_floor"):
                e.with_floor(_floor_number(floor_id), floor_id)
            raise
        pp_screen = _postprocess_to_dict(pp, floor_bounds=floor_bounds)
        logger.info(
            "[SERIALIZER] Postprocess complete | floor=%s | walls=%d | doors=%d | windows=%d",
            floor_id,
            len(pp.walls),
            len(pp.doors),
            len(pp.windows),
        )

        if is_ground_floor and not core_forward and layout.core_tube is not None:
            corridor_centers: List[Tuple[float, float]] = []
            if corridors_in:
                for c in corridors_in:
                    try:
                        poly = getattr(c, "polygon", None)
                        if poly is None or poly.is_empty:
                            continue
                        minx, miny, maxx, maxy = poly.bounds
                        corridor_centers.append((float((minx + maxx) / 2), float((miny + maxy) / 2)))
                    except Exception:
                        continue
            default_dir = [1.0, 0.0, 0.0]
            ct = layout.core_tube
            ch = _center_from_poly(getattr(ct, "staircase_hall", None))
            cs = _center_from_poly(getattr(ct, "staircase_shaft", None))
            eh = _center_from_poly(getattr(ct, "elevator_hall", None))
            es = _center_from_poly(getattr(ct, "elevator_shaft", None))

            def _nearest_corr(p: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
                if p is None or not corridor_centers:
                    return None
                best = None
                best_d = 1e18
                for cc in corridor_centers:
                    d = (cc[0] - p[0]) ** 2 + (cc[1] - p[1]) ** 2
                    if d < best_d:
                        best_d = d
                        best = cc
                return best

            if ch is not None:
                tgt = _nearest_corr(ch)
                core_forward["staircase_hall"] = _calculate_forward(ch, tgt) if tgt else default_dir
            if cs is not None:
                tgt = ch or _nearest_corr(cs)
                core_forward["staircase_shaft"] = _calculate_forward(cs, tgt) if tgt else default_dir
            if eh is not None:
                tgt = _nearest_corr(eh)
                core_forward["elevator_hall"] = _calculate_forward(eh, tgt) if tgt else default_dir
            if es is not None:
                tgt = eh or _nearest_corr(es)
                core_forward["elevator_shaft"] = _calculate_forward(es, tgt) if tgt else default_dir
            if "elevator_hall" in core_forward:
                core_forward["elevator"] = list(core_forward["elevator_hall"])
            if "staircase_hall" in core_forward:
                core_forward["staircase"] = list(core_forward["staircase_hall"])

        floors[floor_id] = {
            "floor_name": floor_id,
            "floor_slab": slab,
            "corridors": corridors,
            "rooms": rooms,
            "walls": pp_screen["walls"],
            "doors": pp_screen["doors"],
            "windows": pp_screen["windows"],
            "generation_time_ms": round(layout.generation_time_ms, 1),
            "warnings": floor_warnings,
        }

    # 降级摘要
    degradation_summary = _build_degradation_summary(result.warnings)
    logger.info("[SERIALIZER] Complete building_result_to_dict | floors=%d | warnings=%d", len(floors), len(result.warnings))

    ct_dict = core_tube_to_dict(result.core_tube, floor_bounds=floor_bounds) if result.core_tube else None
    if isinstance(ct_dict, dict) and core_forward:
        for k, fwd in core_forward.items():
            info = ct_dict.get(k)
            if isinstance(info, dict):
                try:
                    fx, fy = float(fwd[0]), float(fwd[1])
                    sfx, sfy, _ = to_screen_forward(fx, fy)
                    info["forward"] = [float(sfx), float(sfy), 0.0]
                except Exception:
                    info["forward"] = fwd

    return {
        "building": {
            "width": round(maxx - minx, 2),
            "depth": round(maxy - miny, 2),
            "floors": floors,
        },
        "core_tube": ct_dict,
        "warnings": result.warnings,
        "degradation_summary": degradation_summary,
    }


def _build_degradation_summary(warnings: List[str]) -> dict:
    """从 warnings 列表构建结构化降级摘要"""
    summary = {
        "total_degradations": 0,
        "skipped_rooms": [],
        "miqp_fallback_floors": [],
        "adjacency_dropped": 0,
        "unreachable_rooms": [],
        "parse_fixes": 0,
    }

    for w in warnings:
        w_lower = w.lower()
        if "skip" in w_lower and "room" in w_lower:
            summary["skipped_rooms"].append(w)
            summary["total_degradations"] += 1
        elif "miqp" in w_lower and "fallback" in w_lower:
            summary["miqp_fallback_floors"].append(w)
            summary["total_degradations"] += 1
        elif "adjacen" in w_lower and ("drop" in w_lower or "移除" in w or "removed" in w_lower):
            summary["adjacency_dropped"] += 1
            summary["total_degradations"] += 1
        elif "unreachable" in w_lower or "不可达" in w:
            summary["unreachable_rooms"].append(w)
            summary["total_degradations"] += 1
        elif "修正" in w or "fix" in w_lower or "repair" in w_lower or "已修正" in w:
            summary["parse_fixes"] += 1
            summary["total_degradations"] += 1

    return summary
