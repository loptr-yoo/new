"""
serializers.py

将几何引擎输出（Shapely 对象）序列化为 JSON 可传输格式。
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from .building_orchestrator import BuildingResult
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


def room_result_to_dict(room: RoomResult, floor_id: str) -> dict:
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
        }

    coords = list(room.polygon.exterior.coords)
    minx, miny, maxx, maxy = room.polygon.bounds
    return {
        "room_id": room.id,
        "room_type": room.room_type,
        "floor_id": floor_id,
        "polygon": [[round(x, 2), round(y, 2)] for x, y in coords],
        "center": [round(room.centroid[0], 2), round(room.centroid[1], 2)],
        "area": round(room.area, 2),
        "width": round(maxx - minx, 2),
        "depth": round(maxy - miny, 2),
        "target_area": round(room.target_area, 2),
        "has_window": room.has_window,
    }


def core_tube_to_dict(core_tube: Any) -> dict:
    """CoreTube → 可序列化 dict"""
    if not hasattr(core_tube, "polygon"):
        return {}

    coords = list(core_tube.polygon.exterior.coords)
    result: dict = {
        "boundary": [[round(x, 2), round(y, 2)] for x, y in coords],
        "center": [round(core_tube.center[0], 2), round(core_tube.center[1], 2)],
        "area": round(core_tube.polygon.area, 2),
    }

    if hasattr(core_tube, "elevator") and core_tube.elevator is not None:
        result["elevator"] = {
            "polygon": [
                [round(x, 2), round(y, 2)]
                for x, y in core_tube.elevator.exterior.coords
            ],
            "area": round(core_tube.elevator_area, 2),
        }

    if hasattr(core_tube, "staircase") and core_tube.staircase is not None:
        result["staircase"] = {
            "polygon": [
                [round(x, 2), round(y, 2)]
                for x, y in core_tube.staircase.exterior.coords
            ],
            "area": round(core_tube.staircase_area, 2),
        }

    if hasattr(core_tube, "staircase_hall") and core_tube.staircase_hall is not None:
        result["staircase_hall"] = {
            "polygon": [
                [round(x, 2), round(y, 2)]
                for x, y in core_tube.staircase_hall.exterior.coords
            ],
            "area": round(getattr(core_tube, "staircase_hall_area", 0.0), 2),
        }

    if hasattr(core_tube, "staircase_shaft") and core_tube.staircase_shaft is not None:
        result["staircase_shaft"] = {
            "polygon": [
                [round(x, 2), round(y, 2)]
                for x, y in core_tube.staircase_shaft.exterior.coords
            ],
            "area": round(getattr(core_tube, "staircase_shaft_area", 0.0), 2),
        }

    if hasattr(core_tube, "elevator_hall") and core_tube.elevator_hall is not None:
        result["elevator_hall"] = {
            "polygon": [
                [round(x, 2), round(y, 2)]
                for x, y in core_tube.elevator_hall.exterior.coords
            ],
            "area": round(getattr(core_tube, "elevator_hall_area", 0.0), 2),
        }

    if hasattr(core_tube, "elevator_shaft") and core_tube.elevator_shaft is not None:
        result["elevator_shaft"] = {
            "polygon": [
                [round(x, 2), round(y, 2)]
                for x, y in core_tube.elevator_shaft.exterior.coords
            ],
            "area": round(getattr(core_tube, "elevator_shaft_area", 0.0), 2),
        }

    return result


def corridor_to_dict(corridor: Corridor) -> dict:
    """Corridor → 可序列化 dict"""
    coords = list(corridor.polygon.exterior.coords)
    minx, miny, maxx, maxy = corridor.polygon.bounds
    return {
        "id": corridor.id,
        "type": "corridor",
        "polygon": [[round(x, 2), round(y, 2)] for x, y in coords],
        "center": [
            round((minx + maxx) / 2, 2),
            round((miny + maxy) / 2, 2),
        ],
        "width": round(maxx - minx, 2),
        "depth": round(maxy - miny, 2),
        "area": round(corridor.polygon.area, 2),
        "orientation": corridor.orientation,
    }


def floor_slab_to_dict(floor_boundary: Polygon) -> dict:
    """楼层边界 → floor_slab 元素 dict"""
    coords = list(floor_boundary.exterior.coords)
    minx, miny, maxx, maxy = floor_boundary.bounds
    return {
        "id": "floor_slab",
        "type": "floor_slab",
        "polygon": [[round(x, 2), round(y, 2)] for x, y in coords],
        "center": [round((minx + maxx) / 2, 2), round((miny + maxy) / 2, 2)],
        "width": round(maxx - minx, 2),
        "depth": round(maxy - miny, 2),
        "area": round(floor_boundary.area, 2),
    }


def _postprocess_to_dict(pp: PostprocessResult) -> dict:
    """PostprocessResult → 可序列化 dict"""
    return {
        "walls": [wall_to_dict(w) for w in pp.walls],
        "doors": [door_to_dict(d) for d in pp.doors],
        "windows": [window_to_dict(w) for w in pp.windows],
    }


def building_result_to_dict(
    result: BuildingResult,
    floor_boundary: Polygon,
) -> dict:
    """BuildingResult → 完整 API 响应格式"""
    minx, miny, maxx, maxy = floor_boundary.bounds

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
        dz = float(p2[1]) - float(p1[1])
        length = float(math.hypot(dx, dz))
        if length < 1e-3:
            return [1.0, 0.0, 0.0]
        return [float(dx / length), 0.0, float(dz / length)]

    for floor_id, layout in result.floor_layouts.items():
        rooms = [room_result_to_dict(r, floor_id) for r in layout.room_layouts]

        # 走廊序列化
        corridors = []
        if hasattr(layout, "corridors") and layout.corridors:
            for c in layout.corridors:
                try:
                    corridors.append(corridor_to_dict(c))
                except Exception:
                    pass

        # 楼板背景
        slab = floor_slab_to_dict(floor_boundary)

        room_rects: Dict[str, tuple] = {}
        zone_types: Dict[str, str] = {}

        if layout.core_tube is not None and hasattr(layout.core_tube, "polygon") and not layout.core_tube.polygon.is_empty:
            ct = layout.core_tube
            subzones = [
                ("core_staircase_hall", "staircase_hall", getattr(ct, "staircase_hall", None)),
                ("core_staircase_shaft", "staircase_shaft", getattr(ct, "staircase_shaft", None)),
                ("core_elevator_hall", "elevator_hall", getattr(ct, "elevator_hall", None)),
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

        if hasattr(layout, "corridors") and layout.corridors:
            for c in layout.corridors:
                if hasattr(c, "polygon") and not c.polygon.is_empty:
                    minx, miny, maxx, maxy = c.polygon.bounds
                    room_rects[c.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
                    zone_types[c.id] = "corridor"

        for r in layout.room_layouts:
            if r.polygon.is_empty:
                continue
            minx, miny, maxx, maxy = r.polygon.bounds
            room_rects[r.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
            zone_types[r.id] = "room"

        floor_warnings = list(getattr(layout, "warnings", []) or [])
        try:
            floor_area = float(floor_boundary.area)
            space_area = 0.0
            if layout.core_tube is not None and hasattr(layout.core_tube, "polygon") and not layout.core_tube.polygon.is_empty:
                space_area += float(layout.core_tube.polygon.area)
            if hasattr(layout, "corridors") and layout.corridors:
                for c in layout.corridors:
                    if hasattr(c, "polygon") and c.polygon is not None and not c.polygon.is_empty:
                        space_area += float(c.polygon.area)
            for r in layout.room_layouts:
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
        if edge_set and room_rects:
            pp_walls = generate_walls_from_topology(
                room_rects=room_rects,
                edge_set=edge_set,
                floor_bounds=floor_boundary.bounds,
                zone_types=zone_types,
            )
        else:
            pp_walls = generate_wall_mesh(
                rooms=layout.room_layouts,
                corridors=layout.corridors if hasattr(layout, 'corridors') else [],
                core_tube=layout.core_tube,
                floor_boundary=floor_boundary,
            )

        rooms_needing_window = set()
        for room in layout.room_layouts:
            room_id = getattr(room, "id", getattr(room, "room_id", "?"))
            has_window = getattr(room, "has_window", False) or getattr(room, "needs_window", False)
            if has_window:
                rooms_needing_window.add(room_id)

        is_ground_floor = bool(ground_floor_id and floor_id == ground_floor_id)
        pp = postprocess_floor(
            rooms=layout.room_layouts,
            floor_boundary=floor_boundary,
            corridors=layout.corridors if hasattr(layout, "corridors") else [],
            is_ground_floor=is_ground_floor,
            walls=pp_walls,
            zone_types=zone_types,
            zone_rects=room_rects,
            rooms_needing_window=rooms_needing_window,
            floor_bounds=floor_boundary.bounds,
        )

        if is_ground_floor and not core_forward and layout.core_tube is not None:
            corridor_centers: List[Tuple[float, float]] = []
            if hasattr(layout, "corridors") and layout.corridors:
                for c in layout.corridors:
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
            "walls": [wall_to_dict(w) for w in pp.walls],
            "doors": [door_to_dict(d) for d in pp.doors],
            "windows": [window_to_dict(w) for w in pp.windows],
            "generation_time_ms": round(layout.generation_time_ms, 1),
            "warnings": floor_warnings,
        }

    # 降级摘要
    degradation_summary = _build_degradation_summary(result.warnings)

    ct_dict = core_tube_to_dict(result.core_tube) if result.core_tube else None
    if isinstance(ct_dict, dict) and core_forward:
        for k, fwd in core_forward.items():
            info = ct_dict.get(k)
            if isinstance(info, dict):
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
