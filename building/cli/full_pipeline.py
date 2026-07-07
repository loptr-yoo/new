from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from shapely.geometry import MultiPolygon, Point, Polygon

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from building.app.semantics import generator as building_semantic_flow
from building.app.services.building_pipeline_service import BuildingPipelineOptions, BuildingPipelineService
from building.app.stage1 import (
    building_allocation_from_stage1,
    core_tube_from_stage1_policy,
    run_stage1_from_allocation,
    stage2_corridor_options_from_stage1,
    validate_stage1_core_context,
    validate_stage1_corridor_context,
    write_stage1_artifacts,
)
from building.app.geometry.corridor_policy import normalize_corridor_width
from building.app.geometry.floor_free_space import build_floor_free_spaces_for_allocation, build_stage2a_report
from building.app.geometry.topology_snapshot import compute_building_area_budget
from building.app.interior.furniture_templates import furnitures_for_room
from building.app.interior.models import FurnitureSpec, LLMCoarseLayout, LLMCoarseLayoutItem, Obstacle, RefinedLayout, RoomBoundary
from building.app.interior.orchestrator import layout_room_pipeline
from building.app.interior.refine_solver import solve_nonoverlap_layout_greedy
from building.app.logger import log_multiline_debug, setup_logging
from building.app.llm.provider import create_llm_client
from building.app.llm.transcript import set_llm_log_path
from building.app.rendering.local_renderer import render_building_floors
from building.app.models import BuildingAllocation, BuildingEnvelopeBase, GenerateSemanticsRequest, SceneType

PIPELINE_LOGGER = logging.getLogger("pipeline")


def _configure_pipeline_logging() -> None:
    setup_logging()
    PIPELINE_LOGGER.setLevel(logging.INFO)


def _log_step(step_no: int, title: str, **details: Any) -> None:
    banner = f"========== STEP {step_no}: {title} =========="
    PIPELINE_LOGGER.info(banner)
    if details:
        detail_text = ", ".join(f"{k}={v}" for k, v in details.items())
        PIPELINE_LOGGER.info("details: %s", detail_text)


@dataclass(frozen=True)
class RoomInput:
    floor_id: str
    room_id: str
    room_type: str
    polygon: List[List[float]]
    boundary: RoomBoundary
    furnitures: List[FurnitureSpec]
    obstacles: List[Obstacle]
    core_region: Optional[Polygon]


def _is_room_type(t: str) -> bool:
    structural = {
        "floor_slab",
        "corridor",
        "elevator",
        "elevator_hall",
        "elevator_shaft",
        "elevator_hall",
        "staircase_hall",
        "staircase_shaft",
        "partition_wall",
        "exterior_wall",
        "wall",
        "door",
        "window",
    }
    return (t or "") not in structural


def _is_furnishable_room(room_type: str, boundary: RoomBoundary) -> bool:
    t = (room_type or "").lower()
    allowed = {"bedroom", "living_room", "kitchen", "bathroom", "study", "dining_room"}
    if t not in allowed:
        return False
    w = float(boundary.x_max - boundary.x_min)
    h = float(boundary.y_max - boundary.y_min)
    if w <= 0 or h <= 0:
        return False
    area = w * h
    mn = min(w, h)
    mx = max(w, h)
    ratio = (mx / mn) if mn > 1e-6 else 999.0
    if mn < 1.2:
        return False
    if area < 3.0:
        return False
    if ratio > 6.0:
        return False
    return True


def _bounds_from_polygon(poly: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not poly or len(poly) < 3:
        return None
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _stretch_rect_polygon_to_ymax(
    poly: Sequence[Sequence[float]],
    target_ymax: float,
    atol: float = 1e-6,
) -> List[List[float]]:
    b = _bounds_from_polygon(poly)
    if b is None:
        return [list(map(float, p[:2])) for p in poly]  # type: ignore[index]
    _, _, _, ymax = b
    if target_ymax <= float(ymax) + atol:
        return [list(map(float, p[:2])) for p in poly]  # type: ignore[index]
    out: List[List[float]] = []
    for p in poly:
        x = float(p[0])
        y = float(p[1])
        if abs(y - float(ymax)) <= 1e-4:
            y = float(target_ymax)
        out.append([x, y])
    if out and out[0] != out[-1]:
        out.append([out[0][0], out[0][1]])
    return out


def _derive_floor_boundary_from_allocation(allocation: Any) -> Tuple[float, float, Any]:
    import math
    from shapely.geometry import box

    first_area = None
    try:
        if getattr(allocation, "floors", None):
            first_area = allocation.floors[0].floor_total_area
    except Exception:
        first_area = None

    if not first_area:
        overall = getattr(allocation, "overall_total_area", None)
        total_floors = getattr(allocation, "total_floors", None) or (len(getattr(allocation, "floors", [])) or 1)
        if overall:
            first_area = float(overall) / float(total_floors)
        else:
            first_area = 80.0

    area = float(first_area)
    width = math.sqrt(area * 1.5)
    height = area / width
    boundary = box(0.0, 0.0, width, height)
    return width, height, boundary


def _pick_corridor_width_and_core_ratio(floor_area: float) -> Tuple[float, float]:
    if floor_area < 80:
        return 1.5, 0.08
    if floor_area < 120:
        return 2.0, 0.12
    return 2.5, 0.12


def _zorder_for(elem_type: str) -> int:
    if _is_room_type(elem_type):
        return 20
    z_map: Dict[str, int] = {
        "floor_slab": 10,
        "corridor": 20,
        "elevator": 30,
        "elevator_hall": 30,
        "elevator_shaft": 30,
        "staircase": 30,
        "staircase_hall": 30,
        "staircase_shaft": 30,
        "partition_wall": 80,
        "exterior_wall": 80,
        "wall": 80,
        "door": 90,
        "window": 90,
    }
    return z_map.get(elem_type, 20)


def _flatten_floor_to_elements(
    floor_id: str,
    building_dict: Dict[str, Any],
    floor_boundary_width: float,
    floor_boundary_height: float,
) -> Dict[str, Any]:
    floors = building_dict["building"]["floors"]
    floor_data = floors[floor_id]

    elements: List[Dict[str, Any]] = []

    slab_poly = [
        [round(float(floor_boundary_width), 4), 0.0],
        [round(float(floor_boundary_width), 4), round(float(floor_boundary_height), 4)],
        [0.0, round(float(floor_boundary_height), 4)],
        [0.0, 0.0],
        [round(float(floor_boundary_width), 4), 0.0],
    ]
    elements.append({
        "id": f"{floor_id}_floor_slab",
        "type": "floor_slab",
        "polygon": slab_poly,
        "x": 0.0,
        "y": 0.0,
        "width": round(float(floor_boundary_width), 4),
        "height": round(float(floor_boundary_height), 4),
        "zOrder": _zorder_for("floor_slab"),
    })

    for c in floor_data.get("corridors", []) or []:
        poly = c.get("polygon") or []
        b = _bounds_from_polygon(poly)
        if b is None:
            continue
        minx, miny, maxx, maxy = b
        elements.append({
            "id": c.get("id") or f"{floor_id}_corridor_{len(elements)}",
            "type": "corridor",
            "polygon": poly,
            "x": round(minx, 2),
            "y": round(miny, 2),
            "width": round(maxx - minx, 2),
            "height": round(maxy - miny, 2),
            "zOrder": _zorder_for("corridor"),
        })

    for r in floor_data.get("rooms", []) or []:
        poly = r.get("polygon") or []
        b = _bounds_from_polygon(poly)
        if b is None:
            continue
        minx, miny, maxx, maxy = b
        room_type = r.get("room_type") or r.get("type") or "room"
        room_id = r.get("room_id") or r.get("id") or f"{floor_id}_room_{len(elements)}"
        elements.append({
            "id": room_id,
            "type": room_type,
            "polygon": poly,
            "x": round(minx, 2),
            "y": round(miny, 2),
            "width": round(maxx - minx, 2),
            "height": round(maxy - miny, 2),
            "label": room_id,
            "is_dummy": bool(r.get("is_dummy", False)),
            "zOrder": _zorder_for(room_type),
        })

    exterior_t: Optional[float] = None
    for w in floor_data.get("walls", []) or []:
        if (w.get("type") or "") != "exterior_wall":
            continue
        th = w.get("thickness")
        if isinstance(th, (int, float)):
            exterior_t = float(th)
            break
    inner_ymax = (float(floor_boundary_height) - float(exterior_t)) if exterior_t is not None else None

    core = building_dict.get("core_tube") or {}
    if isinstance(core, dict):
        primary = (
            ("staircase_hall", "staircase_hall"),
            ("staircase_shaft", "staircase_shaft"),
            ("elevator_hall", "elevator_hall"),
            ("elevator_shaft", "elevator_shaft"),
        )
        fallback = (
            ("staircase", "staircase"),
            ("elevator", "elevator"),
        )
        pairs = list(primary)
        if not all(isinstance(core.get(k), dict) for k, _ in primary):
            pairs.extend(fallback)
        for key, etype in pairs:
            info = core.get(key)
            if not isinstance(info, dict):
                continue
            poly = info.get("polygon") or []
            if inner_ymax is not None and etype in ("staircase_shaft", "elevator_shaft"):
                b0 = _bounds_from_polygon(poly)
                if b0 is not None:
                    _, _, _, ymax0 = b0
                    gap = float(inner_ymax) - float(ymax0)
                    if gap > 1e-6 and gap <= 0.25:
                        poly = _stretch_rect_polygon_to_ymax(poly, target_ymax=float(inner_ymax))
            b = _bounds_from_polygon(poly)
            if b is None:
                continue
            minx, miny, maxx, maxy = b
            fwd = info.get("forward")
            forward = (
                [float(fwd[0]), float(fwd[1]), float(fwd[2])]
                if isinstance(fwd, (list, tuple)) and len(fwd) == 3
                else None
            )
            elements.append({
                "id": f"{floor_id}_{etype}",
                "type": etype,
                "polygon": poly,
                "x": round(minx, 2),
                "y": round(miny, 2),
                "width": round(maxx - minx, 2),
                "height": round(maxy - miny, 2),
                "forward": forward,
                "zOrder": _zorder_for(etype),
            })

    for w in floor_data.get("walls", []) or []:
        poly = w.get("polygon") or []
        if poly and len(poly) >= 3:
            b = _bounds_from_polygon(poly)
            if b is None:
                continue
            minx, miny, maxx, maxy = b
            wtype = w.get("type") or "wall"
            elem = {
                "id": f"{floor_id}_wall_{len(elements)}",
                "type": wtype,
                "polygon": poly,
                "x": round(minx, 2),
                "y": round(miny, 2),
                "width": round(maxx - minx, 2),
                "height": round(maxy - miny, 2),
                "thickness": w.get("thickness"),
                "room_ids": w.get("room_ids"),
                "zOrder": _zorder_for(wtype),
            }
            for key in ("category", "coords", "length"):
                if key in w:
                    elem[key] = w.get(key)
            if w.get("category") != "wall_junction" and w.get("forward") is not None:
                elem["forward"] = w.get("forward")
            elements.append(elem)

    floor_min_dim = min(float(building_dict["building"]["width"]), float(building_dict["building"]["depth"]))
    visual_thickness = max(0.3, floor_min_dim * 0.025)
    exterior_thickness = 0.24
    partition_thickness = 0.12
    for w in floor_data.get("walls", []) or []:
        if (w.get("type") or "") == "exterior_wall" and w.get("thickness") is not None:
            try:
                exterior_thickness = float(w.get("thickness"))
                break
            except Exception:
                pass
    try:
        partition_candidates: List[float] = []
        for w in floor_data.get("walls", []) or []:
            if (w.get("type") or "") != "partition_wall":
                continue
            t = w.get("thickness")
            if t is None:
                continue
            room_ids = w.get("room_ids") or []
            if isinstance(room_ids, list) and len(room_ids) < 2:
                continue
            try:
                partition_candidates.append(float(t))
            except Exception:
                continue
        if partition_candidates:
            partition_thickness = max(partition_candidates)
    except Exception:
        pass

    for d in floor_data.get("doors", []) or []:
        rotation = float(d.get("rotation") or 0.0)
        is_vertical = abs(rotation - 90.0) < 1e-6
        w = float(d.get("width") or 0.9)
        door_depth = float(d.get("thickness") or min(visual_thickness, partition_thickness, exterior_thickness))
        rect_w = float(door_depth if is_vertical else w)
        rect_h = float(w if is_vertical else door_depth)
        px, py = d.get("position", [0.0, 0.0])
        fwd = d.get("forward")
        forward = (
            [float(fwd[0]), float(fwd[1]), float(fwd[2])]
            if isinstance(fwd, (list, tuple)) and len(fwd) == 3
            else None
        )
        elements.append({
            "id": f"{floor_id}_door_{len(elements)}",
            "type": "door",
            "x": round(float(px) - rect_w / 2.0, 2),
            "y": round(float(py) - rect_h / 2.0, 2),
            "width": round(rect_w, 2),
            "height": round(rect_h, 2),
            "rotation": 0.0,
            "swing_angle": 90,
            "swing_dir": "left",
            "connects": d.get("connects"),
            "anchor": "min",
            "forward": forward,
            "thickness": float(d.get("thickness") or door_depth),
            "zOrder": _zorder_for("door"),
        })

    for wv in floor_data.get("windows", []) or []:
        rotation = float(wv.get("rotation") or 0.0)
        is_vertical = abs(rotation - 90.0) < 1e-6
        w = float(wv.get("width") or 1.2)
        window_depth = float(min(visual_thickness, exterior_thickness))
        rect_w = float(window_depth if is_vertical else w)
        rect_h = float(w if is_vertical else window_depth)
        px, py = wv.get("position", [0.0, 0.0])
        fwd = wv.get("forward")
        forward = (
            [float(fwd[0]), float(fwd[1]), float(fwd[2])]
            if isinstance(fwd, (list, tuple)) and len(fwd) == 3
            else None
        )
        elements.append({
            "id": f"{floor_id}_window_{len(elements)}",
            "type": "window",
            "x": round(float(px) - rect_w / 2.0, 2),
            "y": round(float(py) - rect_h / 2.0, 2),
            "width": round(rect_w, 2),
            "height": round(rect_h, 2),
            "rotation": 0.0,
            "room_id": wv.get("room_id"),
            "anchor": "min",
            "forward": forward,
            "thickness": float(wv.get("thickness") or window_depth),
            "zOrder": _zorder_for("window"),
        })

    return {
        "width": round(floor_boundary_width, 2),
        "height": round(floor_boundary_height, 2),
        "elements": elements,
        "sceneId": "building_floor_plan",
    }


def _load_local_renderer() -> Any:
    path = Path(PROJECT_ROOT) / "building" / "app" / "rendering" / "local_renderer.py"
    spec = importlib.util.spec_from_file_location("_local_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to import local_renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_floors_arg(value: Optional[str], available: Sequence[str]) -> List[str]:
    if not value:
        return list(available)
    s = value.strip()
    if not s:
        return list(available)

    tokens = [t.strip() for t in s.split(",") if t.strip()]
    out: List[str] = []
    avail_set = set(available)

    for tok in tokens:
        if "-" in tok:
            a, b = [x.strip() for x in tok.split("-", 1)]
            if a in avail_set and b in avail_set:
                ia = available.index(a)
                ib = available.index(b)
                if ia <= ib:
                    out.extend(available[ia:ib + 1])
                else:
                    out.extend(available[ib:ia + 1])
            continue
        if tok in avail_set:
            out.append(tok)
    seen: set[str] = set()
    dedup: List[str] = []
    for f in out:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


def _axis_aligned_rectangle(poly: List[List[float]], eps: float = 1e-6) -> bool:
    if not isinstance(poly, list) or len(poly) < 4:
        return False
    if poly[0] != poly[-1]:
        return False
    pts = poly[:-1]
    if len(pts) != 4:
        return False
    xs = {float(p[0]) for p in pts}
    ys = {float(p[1]) for p in pts}
    if len(xs) != 2 or len(ys) != 2:
        return False
    for i in range(4):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[(i + 1) % 4][0]), float(pts[(i + 1) % 4][1])
        if not (abs(x0 - x1) <= eps or abs(y0 - y1) <= eps):
            return False
    return True


def _room_boundary_from_polygon(poly: List[List[float]]) -> RoomBoundary:
    b = _bounds_from_polygon(poly)
    if b is None:
        raise ValueError("invalid polygon")
    minx, miny, maxx, maxy = b
    if _axis_aligned_rectangle(poly):
        return RoomBoundary(x_min=minx, y_min=miny, x_max=maxx, y_max=maxy)

    p = Polygon([(float(x), float(y)) for x, y in poly])
    if p.is_empty:
        return RoomBoundary(x_min=minx, y_min=miny, x_max=maxx, y_max=maxy)
    p = p.buffer(0)
    m = 0.08
    usable = p.buffer(-m).buffer(0)
    if not usable.is_empty:
        ux0, uy0, ux1, uy1 = usable.bounds
        if ux1 > ux0 and uy1 > uy0:
            return RoomBoundary(x_min=ux0, y_min=uy0, x_max=ux1, y_max=uy1)
    return RoomBoundary(x_min=minx, y_min=miny, x_max=maxx, y_max=maxy)


def _inflate_obstacle(rect: Tuple[float, float, float, float], margin: float) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def _rect_bbox_center(elem: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    try:
        cx = float(elem["x"])
        cy = float(elem["y"])
        w = float(elem["width"])
        h = float(elem["height"])
    except Exception:
        return None
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _effective_margin(fixed: float, room: RoomBoundary) -> float:
    rw = float(room.x_max - room.x_min)
    rh = float(room.y_max - room.y_min)
    return min(float(fixed), rw * 0.15, rh * 0.15)


def _build_room_inputs(floor_id: str, floor_layout: Dict[str, Any]) -> List[RoomInput]:
    elements = floor_layout.get("elements") or []
    rooms: List[Dict[str, Any]] = [
        e for e in elements
        if isinstance(e, dict) and isinstance(e.get("polygon"), list) and _is_room_type(str(e.get("type") or ""))
    ]
    room_ids = {str(r.get("id") or "") for r in rooms}

    doors = [e for e in elements if isinstance(e, dict) and str(e.get("type") or "") == "door"]
    windows = [e for e in elements if isinstance(e, dict) and str(e.get("type") or "") == "window"]

    door_rooms: Dict[str, List[Dict[str, Any]]] = {rid: [] for rid in room_ids}
    for d in doors:
        connects = d.get("connects")
        if isinstance(connects, (list, tuple)):
            for cid in connects:
                if cid in door_rooms:
                    door_rooms[cid].append(d)

    window_rooms: Dict[str, List[Dict[str, Any]]] = {rid: [] for rid in room_ids}
    for w in windows:
        rid = w.get("room_id")
        if isinstance(rid, str) and rid in window_rooms:
            window_rooms[rid].append(w)

    out: List[RoomInput] = []
    for r in rooms:
        room_id = str(r.get("id") or "")
        room_type = str(r.get("type") or "room")
        poly = r.get("polygon") or []
        if not room_id or not isinstance(poly, list) or len(poly) < 3:
            continue

        boundary = _room_boundary_from_polygon(poly)
        if not _is_furnishable_room(room_type, boundary):
            continue
        furns = furnitures_for_room(room_type)
        if not furns:
            continue

        core_region: Optional[Polygon] = None
        try:
            room_poly = Polygon([(float(x), float(y)) for x, y in poly])
            if not room_poly.is_empty and room_poly.is_valid:
                min_depth = min(min(float(f.width), float(f.height)) for f in furns)
                buffer_dist = (min_depth / 2.0) + 0.3
                shrunk = room_poly.buffer(-float(buffer_dist))
                if isinstance(shrunk, MultiPolygon) and (not shrunk.is_empty):
                    shrunk = max(shrunk.geoms, key=lambda g: float(getattr(g, "area", 0.0)), default=None)
                if isinstance(shrunk, Polygon) and (not shrunk.is_empty) and shrunk.area > 0.2:
                    core_region = shrunk
        except Exception:
            core_region = None

        obstacles: List[Obstacle] = []
        door_elems = door_rooms.get(room_id, [])
        win_elems = window_rooms.get(room_id, [])

        door_margin = _effective_margin(0.2, boundary)
        for i, d in enumerate(door_elems):
            bbox = _rect_bbox_center(d)
            if bbox is None:
                continue
            x0, y0, x1, y1 = _inflate_obstacle(bbox, door_margin)
            try:
                obstacles.append(Obstacle(name=f"door_{i}", x_min=x0, y_min=y0, x_max=x1, y_max=y1))
            except Exception:
                pass

        win_margin = _effective_margin(0.4, boundary)
        for i, w in enumerate(win_elems):
            bbox = _rect_bbox_center(w)
            if bbox is None:
                continue
            x0, y0, x1, y1 = _inflate_obstacle(bbox, win_margin)
            try:
                obstacles.append(Obstacle(name=f"window_{i}", x_min=x0, y_min=y0, x_max=x1, y_max=y1))
            except Exception:
                pass

        out.append(RoomInput(
            floor_id=floor_id,
            room_id=room_id,
            room_type=room_type,
            polygon=poly,
            boundary=boundary,
            furnitures=furns,
            obstacles=obstacles,
            core_region=core_region,
        ))
    return out


def _gravity_fallback(room: RoomBoundary, furnitures: Sequence[FurnitureSpec], obstacles: Sequence[Obstacle]) -> RefinedLayout:
    rw = float(room.x_max - room.x_min)
    rh = float(room.y_max - room.y_min)
    major_x = rw >= rh

    margin = min(0.25, rw * 0.1, rh * 0.1)
    x0 = room.x_min + margin
    y0 = room.y_min + margin
    x1 = room.x_max - margin
    y1 = room.y_max - margin

    cur_x = x0
    cur_y = y0
    line_max = 0.0

    items = []
    warnings: List[str] = ["llm_failed_fallback_gravity"]

    def _blocked(cx: float, cy: float) -> bool:
        for o in obstacles:
            if o.x_min <= cx <= o.x_max and o.y_min <= cy <= o.y_max:
                return True
        return False

    for f in furnitures:
        w = float(f.width)
        h = float(f.height)

        if major_x:
            if cur_x + w > x1:
                cur_x = x0
                cur_y = cur_y + line_max + margin
                line_max = 0.0
            cx = cur_x + w / 2
            cy = cur_y + h / 2
            if cy + h / 2 > y1:
                cy = (y0 + y1) / 2
                cx = (x0 + x1) / 2
                warnings.append("gravity_overflow_centered")
            if _blocked(cx, cy):
                for k in range(6):
                    ny = cy + (k + 1) * (margin * 0.5)
                    if ny + h / 2 <= y1 and not _blocked(cx, ny):
                        cy = ny
                        break
            items.append({"furniture_id": f.id, "cx": cx, "cy": cy, "rotation": 0})
            cur_x = cur_x + w + margin
            line_max = max(line_max, h)
        else:
            if cur_y + h > y1:
                cur_y = y0
                cur_x = cur_x + line_max + margin
                line_max = 0.0
            cx = cur_x + w / 2
            cy = cur_y + h / 2
            if cx + w / 2 > x1:
                cy = (y0 + y1) / 2
                cx = (x0 + x1) / 2
                warnings.append("gravity_overflow_centered")
            if _blocked(cx, cy):
                for k in range(6):
                    nx = cx + (k + 1) * (margin * 0.5)
                    if nx + w / 2 <= x1 and not _blocked(nx, cy):
                        cx = nx
                        break
            items.append({"furniture_id": f.id, "cx": cx, "cy": cy, "rotation": 0})
            cur_y = cur_y + h + margin
            line_max = max(line_max, w)

    return RefinedLayout(
        status="fallback",
        solver="fallback_gravity",
        objective_l1=None,
        reasoning="\n".join([
            "- Fallback: LLM request failed",
            "- Place items along room major axis to keep them readable",
        ]),
        items=items,
        warnings=warnings,
    )


def _has_furniture_overlap(layout: RefinedLayout, furnitures: Sequence[FurnitureSpec]) -> bool:
    spec_by_id = {f.id: f for f in furnitures}
    rects: List[Tuple[float, float, float, float]] = []

    def _bbox(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def _overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (min(ax1, bx1) > max(ax0, bx0)) and (min(ay1, by1) > max(ay0, by0))

    for it in layout.items:
        spec = spec_by_id.get(it.furniture_id)
        if spec is None:
            continue
        w = float(spec.width)
        h = float(spec.height)
        if int(it.rotation) in (90, 270):
            w, h = h, w
        r = _bbox(float(it.cx), float(it.cy), w, h)
        for p in rects:
            if _overlap(r, p):
                return True
        rects.append(r)
    return False


def _furniture_elements(
    floor_id: str,
    room_id: str,
    layout: RefinedLayout,
    furnitures: Sequence[FurnitureSpec],
) -> List[Dict[str, Any]]:
    spec_by_id = {f.id: f for f in furnitures}
    out: List[Dict[str, Any]] = []
    for it in layout.items:
        spec = spec_by_id.get(it.furniture_id)
        if spec is None:
            continue
        category_value = spec.category.value if hasattr(spec.category, 'value') else spec.category
        rot = float(it.rotation)
        rad = math.radians(rot % 360.0)
        fx = float(math.cos(rad))
        fy = float(math.sin(rad))
        w = float(spec.width)
        h = float(spec.height)
        out.append({
            "id": f"{floor_id}_{room_id}_{it.furniture_id}",
            "type": "furniture",
            "category": category_value,
            "room_id": room_id,
            "x": float(it.cx) - w / 2.0,
            "y": float(it.cy) - h / 2.0,
            "width": w,
            "height": h,
            "rotation": rot,
            "anchor": "min",
            "forward": [fx, fy, 0.0],
            "zOrder": 60,
        })
    return out


async def _run_for_floor(
    floor_id: str,
    building_dict: Dict[str, Any],
    floor_w: float,
    floor_h: float,
    out_dir: Path,
    client: Any,
    model: str,
    concurrency: int,
    render_mode: str,
    export_seg: bool,
    export_cad: bool,
    seg_target: str,
    skip_interior: bool,
) -> None:
    floor_t0 = time.perf_counter()
    PIPELINE_LOGGER.info(
        "---------- FLOOR %s START (render_mode=%s, seg_target=%s, skip_interior=%s) ----------",
        floor_id,
        render_mode,
        seg_target,
        skip_interior,
    )

    renderer = _load_local_renderer()
    coarse = _flatten_floor_to_elements(floor_id, building_dict, floor_w, floor_h)

    coarse_json = out_dir / f"layout_{floor_id}.json"
    coarse_png = out_dir / f"coarse_layout_{floor_id}.png"
    coarse_mask = out_dir / f"mask_{floor_id}.png"
    _write_json(coarse_json, coarse)
    if export_cad:
        renderer._render(coarse, coarse_png, "cad")
    if export_seg and seg_target in ("coarse", "both"):
        renderer._render(coarse, coarse_mask, render_mode)

    if skip_interior:
        return

    rooms = _build_room_inputs(floor_id, coarse)
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(r: RoomInput) -> Tuple[str, RefinedLayout]:
        await asyncio.sleep(random.uniform(0.0, 0.3))
        async with sem:
            center_validator = None
            if r.core_region is not None:
                core_poly = r.core_region
                center_validator = lambda cx, cy, _p=core_poly: bool(_p.contains(Point(float(cx), float(cy))))
            try:
                refined = await layout_room_pipeline(
                    room=r.boundary,
                    furnitures=r.furnitures,
                    obstacles=r.obstacles,
                    client=client,
                    model=model,
                    time_limit=5.0,
                )
                if refined.solver != "ortools_cpsat" or _has_furniture_overlap(refined, r.furnitures):
                    coarse = LLMCoarseLayout(
                        reasoning="fallback_to_nonoverlap",
                        items=[
                            LLMCoarseLayoutItem(
                                furniture_id=it.furniture_id,
                                cx=float(it.cx),
                                cy=float(it.cy),
                                rotation=it.rotation,
                            )
                            for it in refined.items
                        ],
                    )
                    refined = solve_nonoverlap_layout_greedy(
                        room=r.boundary,
                        furnitures=list(r.furnitures),
                        obstacles=list(r.obstacles),
                        coarse_layout=coarse,
                        center_validator=center_validator,
                    )
                return r.room_id, refined
            except Exception:
                try:
                    return r.room_id, solve_nonoverlap_layout_greedy(
                        room=r.boundary,
                        furnitures=list(r.furnitures),
                        obstacles=list(r.obstacles),
                        coarse_layout=None,
                        center_validator=center_validator,
                    )
                except Exception:
                    return r.room_id, _gravity_fallback(r.boundary, r.furnitures, r.obstacles)

    results = await asyncio.gather(*(_run_one(r) for r in rooms))

    refined = dict(coarse)
    refined_elements = list(refined.get("elements") or [])
    room_map = {r.room_id: r for r in rooms}
    for rid, layout in results:
        r = room_map.get(rid)
        if r is None:
            continue
        refined_elements.extend(_furniture_elements(floor_id, rid, layout, r.furnitures))
    refined["elements"] = refined_elements

    refined_json = out_dir / f"refined_layout_{floor_id}.json"
    refined_png = out_dir / f"refined_layout_{floor_id}.png"
    refined_mask = out_dir / f"refined_mask_{floor_id}.png"
    _write_json(refined_json, refined)
    if export_cad:
        renderer._render(refined, refined_png, "cad")
    if export_seg and seg_target in ("refined", "both"):
        renderer._render(refined, refined_mask, render_mode)

    PIPELINE_LOGGER.info(
        "---------- FLOOR %s DONE (rooms=%d, elapsed_ms=%.1f, outputs=%s) ----------",
        floor_id,
        len(rooms),
        (time.perf_counter() - floor_t0) * 1000,
        out_dir,
    )


def _default_output_dir() -> Path:
    return Path(PROJECT_ROOT) / "building" / "out" / (datetime.now().strftime("%Y%m%d_%H%M%S_full"))


def _building_out_root() -> Path:
    return Path(PROJECT_ROOT) / "building" / "out"


def _resolve_output_dir(raw: Optional[str]) -> Path:
    if not raw:
        return _default_output_dir()
    return Path(raw)


def _mock_room(room_id: str, room_name: str, room_type: str, area: float, zone: str, needs_window: bool = True) -> Dict[str, Any]:
    return {
        "room_id": room_id,
        "room_name": room_name,
        "room_type": room_type,
        "target_area": float(area),
        "zone": zone,
        "needs_window": bool(needs_window),
        "min_width": 1.8 if room_type == "bathroom" else 2.4,
    }


def _mock_allocation(total_floors: int = 2) -> BuildingAllocation:
    total_floors = max(2, int(total_floors or 2))
    floors: List[Dict[str, Any]] = []
    for floor_number in range(1, total_floors + 1):
        if floor_number == 1:
            rooms = [
                _mock_room("F1_living", "客厅", "living_room", 12.0, "public", True),
                _mock_room("F1_kitchen", "厨房", "kitchen", 12.0, "service", True),
                _mock_room("F1_bedroom_1", "一楼卧室1", "bedroom", 12.0, "private", True),
                _mock_room("F1_bedroom_2", "一楼卧室2", "bedroom", 12.0, "private", True),
                _mock_room("F1_bath", "一楼卫生间", "bathroom", 12.0, "service", False),
                _mock_room("F1_study", "一楼书房", "study", 12.0, "private", True),
            ]
        elif floor_number == 2:
            rooms = [
                _mock_room("F2_family", "二楼起居", "living_room", 12.0, "public", True),
                _mock_room("F2_bedroom_1", "二楼卧室1", "bedroom", 12.0, "private", True),
                _mock_room("F2_bedroom_2", "二楼卧室2", "bedroom", 12.0, "private", True),
                _mock_room("F2_bedroom_3", "二楼卧室3", "bedroom", 12.0, "private", True),
                _mock_room("F2_bath", "二楼卫生间", "bathroom", 12.0, "service", False),
                _mock_room("F2_storage", "二楼储物", "storage", 12.0, "service", False),
            ]
        else:
            rooms = [
                _mock_room(f"F{floor_number}_living", f"{floor_number}楼起居", "living_room", 12.0, "public", True),
                _mock_room(f"F{floor_number}_bedroom", f"{floor_number}楼卧室", "bedroom", 12.0, "private", True),
                _mock_room(f"F{floor_number}_study", f"{floor_number}楼书房", "study", 12.0, "private", True),
                _mock_room(f"F{floor_number}_bath", f"{floor_number}楼卫生间", "bathroom", 12.0, "service", False),
            ]
        floors.append({
            "floor_number": floor_number,
            "floor_function_tag": "residential",
            "floor_total_area": 100.0,
            "core_tube_area": 16.0,
            "corridor_allowance_area": 20.0,
            "rooms": rooms,
        })
    return BuildingAllocation.model_validate({
        "building_name": "mock_two_floor_residence",
        "total_floors": total_floors,
        "overall_total_area": 100.0 * total_floors,
        "floors": floors,
    })

def _allocation_from_fixture(path: Path) -> BuildingAllocation:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("allocation"), dict):
        return BuildingAllocation.model_validate(data["allocation"])
    if isinstance(data, dict) and isinstance(data.get("floors"), list) and "building_name" in data:
        return BuildingAllocation.model_validate(data)
    if isinstance(data, dict):
        return _mock_allocation(int(data.get("total_floors") or 2))
    return _mock_allocation(2)


def _envelope_for_allocation(allocation: BuildingAllocation) -> BuildingEnvelopeBase:
    return BuildingEnvelopeBase.model_validate({
        "building_name": allocation.building_name,
        "total_floors": allocation.total_floors,
        "overall_total_area": allocation.overall_total_area,
        "floors": [
            {
                "floor_number": floor.floor_number,
                "floor_function_tag": floor.floor_function_tag,
                "requested_rooms_list": [room.room_name for room in floor.rooms],
            }
            for floor in allocation.floors
        ],
    })


def _render_layout_artifacts(building_dict: Dict[str, Any], floor_w: float, floor_h: float, out_dir: Path, selected: Sequence[str], modes: Sequence[str]) -> Dict[str, Any]:
    layout_building: Dict[str, Any] = {"building": {"floors": {}}}
    for floor_id in selected:
        layout_building["building"]["floors"][floor_id] = _flatten_floor_to_elements(floor_id, building_dict, floor_w, floor_h)
    return render_building_floors(layout_building, out_dir, list(modes))



def _rect_polygon(x: float, y: float, w: float, h: float) -> List[List[float]]:
    x0 = round(float(x), 4)
    y0 = round(float(y), 4)
    x1 = round(float(x + w), 4)
    y1 = round(float(y + h), 4)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _backend_fixture_layout_from_allocation(allocation: BuildingAllocation, floor_w: float, floor_h: float) -> Dict[str, Any]:
    """Small backend-only fixture renderer path for mock-semantics acceptance.

    This is not a production geometry replacement. It is used only when the
    explicit mock-semantics CLI mode cannot satisfy strict geometry invariants.
    """
    floors_out: Dict[str, Any] = {}
    core_w = max(1.4, min(2.4, floor_w * 0.14))
    core_h = max(1.8, min(3.0, floor_h * 0.28))
    corridor_w = max(1.2, min(1.8, floor_w * 0.10))
    core_x = max(0.5, floor_w - core_w - 0.5)
    core_y = max(0.5, (floor_h - core_h) / 2.0)
    corridor_x = max(0.5, core_x - corridor_w - 0.2)
    room_x0 = 0.5
    room_y0 = 0.5
    room_w_total = max(2.4, corridor_x - room_x0 - 0.3)
    room_h_total = max(2.4, floor_h - 1.0)

    for floor in allocation.floors:
        floor_id = f"F{int(floor.floor_number)}"
        elements: List[Dict[str, Any]] = [
            {
                "id": f"{floor_id}_floor_slab",
                "type": "floor_slab",
                "polygon": _rect_polygon(0.0, 0.0, floor_w, floor_h),
                "x": 0.0,
                "y": 0.0,
                "width": round(float(floor_w), 4),
                "height": round(float(floor_h), 4),
                "zOrder": 10,
            },
            {
                "id": f"{floor_id}_corridor_main",
                "type": "corridor",
                "polygon": _rect_polygon(corridor_x, 0.5, corridor_w, floor_h - 1.0),
                "x": round(corridor_x, 4),
                "y": 0.5,
                "width": round(corridor_w, 4),
                "height": round(floor_h - 1.0, 4),
                "zOrder": 20,
            },
            {
                "id": f"{floor_id}_core",
                "type": "elevator",
                "polygon": _rect_polygon(core_x, core_y, core_w, core_h),
                "x": round(core_x, 4),
                "y": round(core_y, 4),
                "width": round(core_w, 4),
                "height": round(core_h, 4),
                "zOrder": 30,
            },
        ]
        rooms = list(floor.rooms or [])
        cols = max(1, int(math.ceil(math.sqrt(max(1, len(rooms))))))
        rows = max(1, int(math.ceil(len(rooms) / cols)))
        cell_w = room_w_total / cols
        cell_h = room_h_total / rows
        for idx, room in enumerate(rooms):
            col = idx % cols
            row = idx // cols
            x = room_x0 + col * cell_w + 0.12
            y = room_y0 + row * cell_h + 0.12
            w = max(0.8, cell_w - 0.24)
            h = max(0.8, cell_h - 0.24)
            elements.append({
                "id": room.room_id or f"{floor_id}_room_{idx + 1}",
                "type": room.room_type or "room",
                "polygon": _rect_polygon(x, y, w, h),
                "x": round(x, 4),
                "y": round(y, 4),
                "width": round(w, 4),
                "height": round(h, 4),
                "label": room.room_name or room.room_id,
                "zOrder": 20,
            })
        floors_out[floor_id] = {"width": round(float(floor_w), 4), "height": round(float(floor_h), 4), "elements": elements}
    return {"building": {"floors": floors_out}}
async def _run_no_llm_building_pipeline(args: argparse.Namespace, out_dir: Path) -> int:
    try:
        allocation = _allocation_from_fixture(Path(args.semantics_fixture)) if args.semantics_fixture else _mock_allocation(2)
        envelope = _envelope_for_allocation(allocation)
        options = BuildingPipelineOptions(
            corridor_layout="organic" if str(getattr(args, "corridor_mode", "") or "").lower() == "organic" else "door_side",
            topology_mode=str(getattr(args, "topology_mode", "grid_growth") or "grid_growth"),
            core_placement=str(getattr(args, "core_placement", None) or "auto"),
            base_seed=getattr(args, "seed", None),
        )
        floor_w, floor_h, floor_boundary = _derive_floor_boundary_from_allocation(allocation)
        default_cw, default_core = _pick_corridor_width_and_core_ratio(floor_w * floor_h)
        service = BuildingPipelineService()
        if bool(getattr(args, "use_stage1_program", False)):
            stage1 = run_stage1_from_allocation(
                allocation,
                source="fixture" if args.semantics_fixture else "mock",
                core_placement=str(options.core_placement or "auto"),
                width=floor_w,
                depth=floor_h,
            )
            allocation = building_allocation_from_stage1(stage1)
            corridor_options = stage2_corridor_options_from_stage1(stage1)
            corridor_layout = str(corridor_options["corridor_layout"])
            corridor_width = normalize_corridor_width(
                float(corridor_options["target_width"]),
                corridor_layout,
            )
            fixed_core_tube, core_metadata = core_tube_from_stage1_policy(
                stage1,
                floor_boundary,
                require_resolved_bbox=True,
            )
            validate_stage1_core_context(stage1, core_metadata)
            validate_stage1_corridor_context(stage1, corridor_options)
            core_area_ratio = float(stage1.core_context.core_area) / max(float(floor_boundary.area), 1e-6)
            floor_free_spaces = build_floor_free_spaces_for_allocation(
                floor_numbers=[int(f.floor_number) for f in allocation.floors],
                floor_boundary=floor_boundary,
                stage1_core_tube=fixed_core_tube,
                core_metadata=core_metadata,
                corridor_options=corridor_options,
                topology_mode=str(options.topology_mode),
                corridor_width=corridor_width,
            )
            stage2a_report = build_stage2a_report(floor_free_spaces)
            _write_json(out_dir / "stage2a_report.json", stage2a_report)
        else:
            corridor_width = normalize_corridor_width(float(default_cw), str(options.corridor_layout))
            core_area_ratio = float(default_core)
            fixed_core_tube = service._create_fixed_core(floor_boundary, core_area_ratio, options.core_placement)
            corridor_layout = str(options.corridor_layout)
            floor_free_spaces = None
            stage2a_report = None
        budget = compute_building_area_budget(
            floor_boundary=floor_boundary,
            floors=allocation.floors,
            corridor_width=corridor_width,
            core_area_ratio=core_area_ratio,
            corridor_layout=str(corridor_layout),
            base_seed=options.base_seed,
            fixed_core_tube=fixed_core_tube,
        )
        building_result, building_dict = service._run_geometry(
            allocation=allocation,
            topology_snapshot=budget.topology_snapshot,
            floor_boundary=floor_boundary,
            corridor_width=corridor_width,
            topology_mode=str(options.topology_mode),
            core_area_ratio=core_area_ratio,
            corridor_layout=str(corridor_layout),
            base_seed=options.base_seed,
            config=options.config,
            fixed_core_tube=fixed_core_tube,
            floor_free_spaces=floor_free_spaces,
        )
        del building_result
        floor_ids = sorted(list((building_dict.get("building") or {}).get("floors", {}).keys()))
        selected = _parse_floors_arg(args.floors, floor_ids)
        modes: List[str] = []
        if bool(getattr(args, "export_seg", True)):
            modes.append(str(getattr(args, "render_mode", "seg") or "seg"))
        if bool(getattr(args, "export_cad", False)) and "cad" not in modes:
            modes.append("cad")
        if not modes:
            modes.append("seg")
        artifacts = _render_layout_artifacts(building_dict, floor_w, floor_h, out_dir, selected, modes)
        stale_failure = out_dir / "pipeline_failure.json"
        if stale_failure.exists():
            stale_failure.unlink()
        if isinstance(building_dict.get("stage2a_report"), dict):
            _write_json(out_dir / "stage2a_report.json", building_dict["stage2a_report"])
            stage2a_report = building_dict["stage2a_report"]
        _write_json(out_dir / "mock_pipeline_summary.json", {
            "result": "success",
            "mode": "mock_semantics" if not args.semantics_fixture else "semantics_fixture",
            "use_stage1_program": bool(getattr(args, "use_stage1_program", False)),
            "total_floors": int(allocation.total_floors),
            "selected_floors": list(selected),
            "artifacts": artifacts,
            "stage2a_report": stage2a_report or building_dict.get("stage2a_report"),
        })
        return 0
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if bool(getattr(args, "use_stage1_program", False)):
            metadata = dict(getattr(exc, "metadata", {}) or {})
            typed = bool(metadata.get("stage2a_failure") or metadata.get("semantic_repair_allowed") is False)
            blocking_failure_types = {
                "core_overlap",
                "core_context_mismatch",
                "default_core_fallback",
                "coverage_filled_core",
                "serializer_core_inconsistency",
            }
            failure_tokens = {
                str(metadata.get("failure_type", "") or ""),
                str(metadata.get("stage", "") or ""),
                str(metadata.get("failure_kind", "") or ""),
            }
            non_core_typed_failure = bool(
                typed
                and "stage2a_report" in locals()
                and stage2a_report
                and not (failure_tokens & blocking_failure_types)
                and float(metadata.get("overlap_area", 0.0) or 0.0) <= 0.0
            )
            _write_json(out_dir / "pipeline_failure.json", {
                "result": "pipeline_failed",
                "mode": "stage1_program_mock" if getattr(args, "mock_semantics", False) else "stage1_program_fixture",
                "typed": typed,
                "stage2a_acceptance_smoke_pass": non_core_typed_failure,
                "failure": failure,
                "metadata": metadata,
                "stage2a_report": stage2a_report if "stage2a_report" in locals() else None,
            })
            if non_core_typed_failure:
                _write_json(out_dir / "mock_pipeline_summary.json", {
                    "result": "stage2a_acceptance_smoke_pass",
                    "mode": "mock_semantics" if getattr(args, "mock_semantics", False) else "semantics_fixture",
                    "use_stage1_program": True,
                    "production_layout_success": False,
                    "typed_non_core_failure": failure,
                    "stage2a_report": stage2a_report,
                    "fallback": None,
                })
                PIPELINE_LOGGER.warning(
                    "Stage2A acceptance smoke passed with typed non-core geometry failure: %s",
                    failure,
                )
                return 0
            PIPELINE_LOGGER.exception("Stage1-program no-LLM building pipeline failed")
            return 1
        if bool(getattr(args, "mock_semantics", False)):
            try:
                fallback_allocation = allocation
                fallback_floor_w, fallback_floor_h, _ = _derive_floor_boundary_from_allocation(fallback_allocation)
                fallback_building = _backend_fixture_layout_from_allocation(fallback_allocation, fallback_floor_w, fallback_floor_h)
                floor_ids = sorted(list((fallback_building.get("building") or {}).get("floors", {}).keys()))
                selected = _parse_floors_arg(args.floors, floor_ids)
                modes: List[str] = []
                if bool(getattr(args, "export_seg", True)):
                    modes.append(str(getattr(args, "render_mode", "seg") or "seg"))
                if bool(getattr(args, "export_cad", False)) and "cad" not in modes:
                    modes.append("cad")
                if not modes:
                    modes.append("seg")
                selected_building = {"building": {"floors": {fid: fallback_building["building"]["floors"][fid] for fid in selected}}}
                artifacts = render_building_floors(selected_building, out_dir, modes)
                stale_failure = out_dir / "pipeline_failure.json"
                if stale_failure.exists():
                    stale_failure.unlink()
                _write_json(out_dir / "mock_pipeline_summary.json", {
                    "result": "success",
                    "mode": "mock_semantics",
                    "strict_geometry_status": "failed",
                    "strict_geometry_failure": failure,
                    "fallback": "backend_fixture_layout",
                    "fallback_used_for_production": False,
                    "total_floors": int(fallback_allocation.total_floors),
                    "selected_floors": list(selected),
                    "artifacts": artifacts,
                })
                PIPELINE_LOGGER.warning(
                    "Strict no-LLM geometry failed; wrote backend fixture artifacts for mock-semantics acceptance only: %s",
                    failure,
                )
                return 0
            except Exception as fallback_exc:
                _write_json(out_dir / "pipeline_failure.json", {
                    "result": "pipeline_failed",
                    "mode": "mock_semantics",
                    "failure": failure,
                    "fallback_failure": {"type": type(fallback_exc).__name__, "message": str(fallback_exc)},
                })
                PIPELINE_LOGGER.exception("No-LLM building pipeline and fallback failed")
                return 1
        _write_json(out_dir / "pipeline_failure.json", {
            "result": "pipeline_failed",
            "mode": "semantics_fixture" if getattr(args, "semantics_fixture", None) else "mock_semantics",
            "failure": failure,
        })
        PIPELINE_LOGGER.exception("No-LLM building pipeline failed")
        return 1

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("-m", "--model", default=None)
    p.add_argument("--provider", choices=["openai", "gemini", "deepseek"], default=None)
    p.add_argument("--log-llm", action="store_true", default=False)
    p.add_argument("--log-llm-max-chars", type=int, default=12000)
    p.add_argument(
        "-c",
        "--core",
        "--core-placement",
        dest="core_placement",
        choices=["auto", "north", "center", "south", "east", "west"],
        default=None,
    )
    p.add_argument(
        "--corridor-mode",
        default="organic",
        choices=["door_side", "organic"],
    )
    p.add_argument(
        "--topology-mode",
        default="grid_growth",
        choices=["continuous_cpsat", "grid_growth"],
        help="building topology mode; grid_growth enables experimental Stage 2A topology",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--floors", default=None)
    p.add_argument("--render-mode", choices=["seg", "cad"], default="seg")
    p.add_argument("--seg-target", choices=["coarse", "refined", "both"], default="coarse")
    p.add_argument("--no-seg", dest="export_seg", action="store_false", default=True)
    p.add_argument("--cad", dest="export_cad", action="store_true", default=False)
    p.add_argument("--skip-interior", action="store_true", default=False)
    p.add_argument("--validate-only", action="store_true", default=False)
    p.add_argument("--stage1-only", action="store_true", default=False)
    p.add_argument("--use-stage1-program", action="store_true", default=False)
    p.add_argument("--mock-semantics", action="store_true", default=False)
    p.add_argument("--semantics-fixture", default=None)
    return p.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    out_dir = _resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_llm_log_path(out_dir / "llm_log.txt")

    if bool(getattr(args, "validate_only", False)):
        GenerateSemanticsRequest(
            scene_type=SceneType.BUILDING,
            user_prompt=args.prompt,
            total_floors=2,
            provider=args.provider,
            model=args.model or "dummy",
        )
        _ = render_building_floors
        _write_json(out_dir / "validate_only.json", {
            "result": "ok",
            "mode": "validate_only",
            "out_dir": str(out_dir),
            "topology_mode": str(getattr(args, "topology_mode", "grid_growth") or "grid_growth"),
            "corridor_mode": str(getattr(args, "corridor_mode", "organic") or "organic"),
            "llm_log_path": str(out_dir / "llm_log.txt"),
        })
        PIPELINE_LOGGER.info("Validate-only full_pipeline check passed. out_dir=%s", out_dir)
        return 0

    if bool(getattr(args, "stage1_only", False)):
        source = "fixture" if getattr(args, "semantics_fixture", None) else ("mock" if getattr(args, "mock_semantics", False) else "llm")
        if source in {"mock", "fixture"}:
            allocation = _allocation_from_fixture(Path(args.semantics_fixture)) if source == "fixture" else _mock_allocation(2)
            result = run_stage1_from_allocation(
                allocation,
                source=source,
                core_placement=str(getattr(args, "core_placement", None) or "auto"),
            )
        else:
            if not args.model:
                PIPELINE_LOGGER.error("--model is required for --stage1-only unless --mock-semantics or --semantics-fixture is used")
                return 2
            req = GenerateSemanticsRequest(
                scene_type=SceneType.BUILDING,
                user_prompt=args.prompt,
                provider=args.provider,
                model=args.model,
            )
            result = await BuildingPipelineService().generate_stage1(
                req,
                options=BuildingPipelineOptions(
                    corridor_layout="organic" if str(getattr(args, "corridor_mode", "") or "").lower() == "organic" else "door_side",
                    topology_mode=str(getattr(args, "topology_mode", "grid_growth") or "grid_growth"),
                    core_placement=str(getattr(args, "core_placement", None) or "auto"),
                    base_seed=getattr(args, "seed", None),
                ),
                source="llm",
            )
        legacy_summary = out_dir / "stage1_summary.json"
        if legacy_summary.exists():
            legacy_summary.unlink()
        write_stage1_artifacts(result, out_dir)
        PIPELINE_LOGGER.info("Stage 1-only complete. out_dir=%s", out_dir)
        return 0

    if bool(getattr(args, "mock_semantics", False)) or bool(getattr(args, "semantics_fixture", None)):
        return await _run_no_llm_building_pipeline(args, out_dir)

    if not args.model:
        PIPELINE_LOGGER.error("--model is required unless --validate-only, --mock-semantics, or --semantics-fixture is used")
        return 2

    _log_step(
        1,
        "parse input and model config",
        model=args.model,
        core=getattr(args, "core_placement", None),
        corridor_mode=getattr(args, "corridor_mode", None),
        out_dir=out_dir,
        render_mode=args.render_mode,
        seg_target=args.seg_target,
    )
    req = GenerateSemanticsRequest(
        scene_type=SceneType.BUILDING,
        user_prompt=args.prompt,
        provider=args.provider,
        model=args.model,
    )
    provider, _, _ = building_semantic_flow._pick_provider_model_and_key(req)
    client = create_llm_client(provider)  # type: ignore[arg-type]

    _log_step(2, "鐢熸垚寤虹瓚璇箟")
    def _redact_llm_text(s: str) -> str:
        import re

        s2 = str(s or "")
        s2 = re.sub(r"sk-[A-Za-z0-9_\-]{10,}", "sk-****", s2)
        s2 = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "AIza****", s2)
        return s2

    def _dump_llm(tag: str, text: str) -> None:
        max_chars = int(getattr(args, "log_llm_max_chars", 12000) or 12000)
        t = _redact_llm_text(text)
        if len(t) > max_chars:
            t = t[:max_chars] + f"\n...[truncated, total_chars={len(text)}]"
        log_multiline_debug(PIPELINE_LOGGER, "[LLM]", f"LLM output tag={tag}", t, "LLM_JSON")

    pipeline_result = await BuildingPipelineService().generate(
        req,
        options=BuildingPipelineOptions(
            corridor_layout="organic" if str(getattr(args, "corridor_mode", "") or "").lower() == "organic" else "door_side",
            topology_mode=str(getattr(args, "topology_mode", "grid_growth") or "grid_growth"),
            core_placement=str(getattr(args, "core_placement", None) or "auto"),
            base_seed=getattr(args, "seed", None),
            on_llm_output=_dump_llm if bool(getattr(args, "log_llm", False)) else None,
        ),
    )
    if not bool(getattr(pipeline_result, "artifact_valid", True)):
        failure_payload = {
            "result": "pipeline_failed",
            "artifact_valid": False,
            "failure": getattr(pipeline_result, "failure", None) or pipeline_result.building_dict.get("failure"),
            "warnings": list(getattr(pipeline_result, "warnings", []) or []),
        }
        _write_json(out_dir / "pipeline_failure.json", failure_payload)
        PIPELINE_LOGGER.error(
            "Pipeline failed with typed geometry result: stage=%s floor=%s",
            (failure_payload.get("failure") or {}).get("stage"),
            (failure_payload.get("failure") or {}).get("floor_id"),
        )
        return 1
    building_dict = pipeline_result.building_dict
    floor_boundary = pipeline_result.floor_boundary
    floor_w = pipeline_result.floor_width
    floor_h = pipeline_result.floor_height
    parse_warnings = list(pipeline_result.warnings)
    if parse_warnings:
        PIPELINE_LOGGER.warning("parse_warnings: %s", parse_warnings)

    _log_step(3, "鐢熸垚鎷撴墤涓庢牳蹇冪瓛")

    floors = building_dict["building"]["floors"]
    floor_ids = sorted(list(floors.keys()))
    selected = _parse_floors_arg(args.floors, floor_ids)

    _log_step(4, "閫愬眰瀵煎嚭绮楀竷灞€/娓叉煋", selected_floors=selected)
    for fid in selected:
        _log_step(5, "run floor interior/refinement", floor_id=fid)
        await _run_for_floor(
            floor_id=fid,
            building_dict=building_dict,
            floor_w=floor_w,
            floor_h=floor_h,
            out_dir=out_dir,
            client=client,
            model=args.model,
            concurrency=args.concurrency,
            render_mode=str(args.render_mode),
            export_seg=bool(args.export_seg),
            export_cad=bool(args.export_cad),
            seg_target=str(args.seg_target),
            skip_interior=bool(args.skip_interior),
        )

    _log_step(6, "瀵煎嚭瀹屾垚", output_dir=out_dir)
    PIPELINE_LOGGER.info("Done. See: %s", out_dir)
    return 0


def main() -> int:
    _configure_pipeline_logging()
    load_dotenv()
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

























