"""
serializers.py

将几何引擎输出（Shapely 对象）序列化为 JSON 可传输格式。
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .building_orchestrator import BuildingResult
from .coord_transform import to_screen_forward, to_screen_point, to_screen_polygon
from .layout_generator import LayoutResultV2, RoomResult
from .postprocessor import (
    PostprocessResult,
    door_to_dict,
    generate_walls_from_topology,
    generate_wall_mesh,
    postprocess_floor,
    wall_to_dict,
    window_to_dict,
)
from .topology_generator import CoreTube, Corridor

try:
    from shapely.validation import make_valid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore[assignment]


def _largest_polygon(geom: BaseGeometry) -> Optional[Polygon]:
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


def _clip_to_slab(geom: BaseGeometry, slab: Polygon) -> Optional[Polygon]:
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
    clipped = _round_polygon_coords(clipped, 4)
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
    core_poly = _round_polygon_coords(core_poly, 4)
    coords = list(core_poly.exterior.coords)
    boundary = to_screen_polygon(coords, bounds=floor_bounds)
    ccx, ccy = to_screen_point(float(core_tube.center[0]), float(core_tube.center[1]), bounds=floor_bounds)
    result: dict = {
        "boundary": [[round(float(x), 4), round(float(y), 4)] for x, y in boundary],
        "center": [round(float(ccx), 4), round(float(ccy), 4)],
        "area": round(core_poly.area, 2),
        "actual_area": round(core_poly.area, 4),
    }

    if hasattr(core_tube, "elevator") and core_tube.elevator is not None:
        g = _clip_to_slab(core_tube.elevator, slab) or core_tube.elevator
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["elevator"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }

    if hasattr(core_tube, "staircase") and core_tube.staircase is not None:
        g = _clip_to_slab(core_tube.staircase, slab) or core_tube.staircase
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["staircase"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }

    if hasattr(core_tube, "staircase_hall") and core_tube.staircase_hall is not None:
        g = _clip_to_slab(core_tube.staircase_hall, slab) or core_tube.staircase_hall
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["staircase_hall"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }
    if hasattr(core_tube, "staircase_hall_b") and getattr(core_tube, "staircase_hall_b", None) is not None:
        g = _clip_to_slab(core_tube.staircase_hall_b, slab) or core_tube.staircase_hall_b
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["staircase_hall_b"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }

    if hasattr(core_tube, "staircase_shaft") and core_tube.staircase_shaft is not None:
        g = _clip_to_slab(core_tube.staircase_shaft, slab) or core_tube.staircase_shaft
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["staircase_shaft"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }

    if hasattr(core_tube, "elevator_hall") and core_tube.elevator_hall is not None:
        g = _clip_to_slab(core_tube.elevator_hall, slab) or core_tube.elevator_hall
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["elevator_hall"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }
    if hasattr(core_tube, "elevator_hall_b") and getattr(core_tube, "elevator_hall_b", None) is not None:
        g = _clip_to_slab(core_tube.elevator_hall_b, slab) or core_tube.elevator_hall_b
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["elevator_hall_b"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }

    if hasattr(core_tube, "elevator_shaft") and core_tube.elevator_shaft is not None:
        g = _clip_to_slab(core_tube.elevator_shaft, slab) or core_tube.elevator_shaft
        g = _round_polygon_coords(g, 4)
        poly = to_screen_polygon(list(g.exterior.coords), bounds=floor_bounds)
        result["elevator_shaft"] = {
            "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
            "area": round(g.area, 2),
            "actual_area": round(g.area, 4),
        }

    return result


def corridor_to_dict(corridor: Corridor, *, floor_bounds: Tuple[float, float, float, float]) -> dict:
    """Corridor → 可序列化 dict"""
    slab = box(*floor_bounds)
    clipped = _clip_to_slab(corridor.polygon, slab) or corridor.polygon
    clipped = _round_polygon_coords(clipped, 4)
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
        poly = d.get("polygon")
        if isinstance(poly, list) and len(poly) >= 3:
            d["polygon"] = [[round(x, 2), round(y, 2)] for x, y in to_screen_polygon(poly, bounds=floor_bounds)]
        walls.append(d)

    doors = []
    for d0 in pp.doors:
        d = door_to_dict(d0)
        pos = d.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            sx, sy = to_screen_point(float(pos[0]), float(pos[1]), bounds=floor_bounds)
            d["position"] = [round(sx, 2), round(sy, 2)]
        fwd = d.get("forward")
        if isinstance(fwd, (list, tuple)) and len(fwd) == 3:
            fx, fy = float(fwd[0]), float(fwd[1])
            d["forward"] = [float(to_screen_forward(fx, fy)[0]), float(to_screen_forward(fx, fy)[1]), 0.0]
        doors.append(d)

    windows = []
    for w0 in pp.windows:
        d = window_to_dict(w0)
        pos = d.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            sx, sy = to_screen_point(float(pos[0]), float(pos[1]), bounds=floor_bounds)
            d["position"] = [round(sx, 2), round(sy, 2)]
        fwd = d.get("forward")
        if isinstance(fwd, (list, tuple)) and len(fwd) == 3:
            fx, fy = float(fwd[0]), float(fwd[1])
            d["forward"] = [float(to_screen_forward(fx, fy)[0]), float(to_screen_forward(fx, fy)[1]), 0.0]
        windows.append(d)

    return {"walls": walls, "doors": doors, "windows": windows}


def building_result_to_dict(
    result: BuildingResult,
    floor_boundary: Polygon,
) -> dict:
    """BuildingResult → 完整 API 响应格式"""
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
            except Exception:
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
            has_subzones = all(p is not None and hasattr(p, "is_empty") and not p.is_empty for _, _, p in subzones)
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
            zone_types[r.id] = "room"
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
            elif gap_area < -0.05:
                floor_warnings.append(f"Coverage overlap area={(-gap_area):.3f}m^2 (sum(spaces) - floor)")
        except Exception:
            pass

        edge_set = getattr(layout, "edge_set", None)
        if did_fuse:
            pp_walls = generate_wall_mesh(
                rooms=rooms_in,
                corridors=corridors_in,
                core_tube=layout.core_tube,
                floor_boundary=floor_boundary,
            )
        elif room_rects:
            try:
                pp_walls = generate_walls_from_topology(
                    room_rects=room_rects,
                    edge_set=(edge_set or {}),
                    floor_bounds=floor_boundary.bounds,
                    zone_types=zone_types,
                )
            except Exception:
                pp_walls = generate_wall_mesh(
                    rooms=rooms_in,
                    corridors=corridors_in,
                    core_tube=layout.core_tube,
                    floor_boundary=floor_boundary,
                )
        else:
            pp_walls = generate_wall_mesh(
                rooms=rooms_in,
                corridors=corridors_in,
                core_tube=layout.core_tube,
                floor_boundary=floor_boundary,
            )

        rooms_needing_window = set()
        for room in rooms_in:
            room_id = getattr(room, "id", getattr(room, "room_id", "?"))
            if str(getattr(room, "room_type", "") or "").lower() == "void" or bool(getattr(room, "skip_solver", False)):
                continue
            has_window = getattr(room, "has_window", False) or getattr(room, "needs_window", False)
            if has_window:
                rooms_needing_window.add(room_id)

        is_ground_floor = bool(ground_floor_id and floor_id == ground_floor_id)
        pp = postprocess_floor(
            rooms=rooms_in,
            floor_boundary=floor_boundary,
            corridors=corridors_in,
            is_ground_floor=is_ground_floor,
            walls=pp_walls,
            zone_types=zone_types,
            zone_rects=room_rects,
            rooms_needing_window=rooms_needing_window,
            floor_bounds=floor_boundary.bounds,
        )
        pp_screen = _postprocess_to_dict(pp, floor_bounds=floor_bounds)

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
