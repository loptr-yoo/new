"""
serializers.py

将几何引擎输出（Shapely 对象）序列化为 JSON 可传输格式。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from shapely.geometry import Polygon

from .building_orchestrator import BuildingResult
from .layout_generator import LayoutResultV2, RoomResult
from .postprocessor import (
    PostprocessResult,
    door_to_dict,
    postprocess_floor,
    wall_to_dict,
    window_to_dict,
)
from .topology_generator import CoreTube


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

    return result


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
    for floor_id, layout in result.floor_layouts.items():
        rooms = [room_result_to_dict(r, floor_id) for r in layout.room_layouts]

        # 后处理：自动生成墙/门/窗
        pp = postprocess_floor(layout.room_layouts, floor_boundary)

        floors[floor_id] = {
            "floor_name": floor_id,
            "rooms": rooms,
            **_postprocess_to_dict(pp),
            "generation_time_ms": round(layout.generation_time_ms, 1),
            "warnings": layout.warnings,
        }

    # 降级摘要
    degradation_summary = _build_degradation_summary(result.warnings)

    return {
        "building": {
            "width": round(maxx - minx, 2),
            "depth": round(maxy - miny, 2),
            "floors": floors,
        },
        "core_tube": core_tube_to_dict(result.core_tube) if result.core_tube else None,
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
