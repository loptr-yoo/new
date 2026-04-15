from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from shapely.geometry import Polygon

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.flows import building_semantic_flow
from backend.core.geometry.building_orchestrator import BuildingOrchestrator
from backend.core.geometry.serializers import building_result_to_dict
from backend.core.geometry.topology_generator import CoreTube
from backend.core.interior.furniture_templates import furnitures_for_room
from backend.core.interior.models import FurnitureSpec, Obstacle, RefinedLayout, RoomBoundary
from backend.core.interior.orchestrator import layout_room_pipeline
from backend.core.llm.provider import create_llm_client
from backend.models import GenerateSemanticsRequest, SceneType


@dataclass(frozen=True)
class RoomInput:
    floor_id: str
    room_id: str
    room_type: str
    polygon: List[List[float]]
    boundary: RoomBoundary
    furnitures: List[FurnitureSpec]
    obstacles: List[Obstacle]


def _is_room_type(t: str) -> bool:
    structural = {
        "floor_slab",
        "corridor",
        "elevator",
        "staircase",
        "partition_wall",
        "exterior_wall",
        "wall",
        "door",
        "window",
    }
    return (t or "") not in structural


def _bounds_from_polygon(poly: Sequence[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not poly or len(poly) < 3:
        return None
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


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
        "staircase": 30,
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
        [round(floor_boundary_width, 2), 0.0],
        [round(floor_boundary_width, 2), round(floor_boundary_height, 2)],
        [0.0, round(floor_boundary_height, 2)],
        [0.0, 0.0],
        [round(floor_boundary_width, 2), 0.0],
    ]
    elements.append({
        "id": f"{floor_id}_floor_slab",
        "type": "floor_slab",
        "polygon": slab_poly,
        "x": 0.0,
        "y": 0.0,
        "width": round(floor_boundary_width, 2),
        "height": round(floor_boundary_height, 2),
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
            "zOrder": _zorder_for(room_type),
        })

    core = building_dict.get("core_tube") or {}
    if isinstance(core, dict):
        for key, etype in (("elevator", "elevator"), ("staircase", "staircase")):
            info = core.get(key)
            if not isinstance(info, dict):
                continue
            poly = info.get("polygon") or []
            b = _bounds_from_polygon(poly)
            if b is None:
                continue
            minx, miny, maxx, maxy = b
            elements.append({
                "id": f"{floor_id}_{etype}",
                "type": etype,
                "polygon": poly,
                "x": round(minx, 2),
                "y": round(miny, 2),
                "width": round(maxx - minx, 2),
                "height": round(maxy - miny, 2),
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
            elements.append({
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
            })

    floor_min_dim = min(float(building_dict["building"]["width"]), float(building_dict["building"]["depth"]))
    visual_thickness = max(0.3, floor_min_dim * 0.025)
    exterior_thickness = 0.24
    for w in floor_data.get("walls", []) or []:
        if (w.get("type") or "") == "exterior_wall" and w.get("thickness") is not None:
            try:
                exterior_thickness = float(w.get("thickness"))
                break
            except Exception:
                pass

    for d in floor_data.get("doors", []) or []:
        rotation = float(d.get("rotation") or 0.0)
        is_vertical = abs(rotation - 90.0) < 1e-6
        w = float(d.get("width") or 0.9)
        rect_w = float(visual_thickness if is_vertical else w)
        rect_h = float(w if is_vertical else visual_thickness)
        px, py = d.get("position", [0.0, 0.0])
        elements.append({
            "id": f"{floor_id}_door_{len(elements)}",
            "type": "door",
            "x": round(float(px), 2),
            "y": round(float(py), 2),
            "width": round(rect_w, 2),
            "height": round(rect_h, 2),
            "rotation": 0.0,
            "swing_angle": 90,
            "swing_dir": "left",
            "connects": d.get("connects"),
            "anchor": "center",
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
        elements.append({
            "id": f"{floor_id}_window_{len(elements)}",
            "type": "window",
            "x": round(float(px), 2),
            "y": round(float(py), 2),
            "width": round(rect_w, 2),
            "height": round(rect_h, 2),
            "rotation": 0.0,
            "room_id": wv.get("room_id"),
            "anchor": "center",
            "zOrder": _zorder_for("window"),
        })

    return {
        "width": round(floor_boundary_width, 2),
        "height": round(floor_boundary_height, 2),
        "elements": elements,
        "sceneId": "building_floor_plan",
    }


def _load_local_renderer() -> Any:
    path = Path(PROJECT_ROOT) / "scripts" / "local_renderer.py"
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
        furns = furnitures_for_room(room_type)
        if not furns:
            continue

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
        out.append({
            "id": f"{floor_id}_{room_id}_{it.furniture_id}",
            "type": "furniture",
            "room_id": room_id,
            "x": float(it.cx),
            "y": float(it.cy),
            "width": float(spec.width),
            "height": float(spec.height),
            "rotation": float(it.rotation),
            "anchor": "center",
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
) -> None:
    renderer = _load_local_renderer()
    coarse = _flatten_floor_to_elements(floor_id, building_dict, floor_w, floor_h)

    coarse_json = out_dir / f"coarse_layout_{floor_id}.json"
    coarse_png = out_dir / f"coarse_layout_{floor_id}.png"
    _write_json(coarse_json, coarse)
    renderer._render(coarse, coarse_png)

    rooms = _build_room_inputs(floor_id, coarse)
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(r: RoomInput) -> Tuple[str, RefinedLayout]:
        await asyncio.sleep(random.uniform(0.0, 0.3))
        async with sem:
            try:
                refined = await layout_room_pipeline(
                    room=r.boundary,
                    furnitures=r.furnitures,
                    obstacles=r.obstacles,
                    client=client,
                    model=model,
                    time_limit=5.0,
                )
                return r.room_id, refined
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
    _write_json(refined_json, refined)
    renderer._render(refined, refined_png)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--provider", choices=["openai", "gemini", "deepseek"], default=None)
    p.add_argument("-c", "--core", choices=["north", "center", "south"], default="north")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--floors", default=None)
    return p.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    req = GenerateSemanticsRequest(
        scene_type=SceneType.BUILDING,
        user_prompt=args.prompt,
        provider=args.provider,
        model=args.model,
    )
    provider, _, _ = building_semantic_flow._pick_provider_model_and_key(req)
    client = create_llm_client(provider)  # type: ignore[arg-type]

    allocation, parse_warnings = await building_semantic_flow.generate_building_semantics(req)
    if parse_warnings:
        print(f"[full_pipeline] parse_warnings: {parse_warnings}", file=sys.stderr)

    floor_w, floor_h, floor_boundary = _derive_floor_boundary_from_allocation(allocation)
    corridor_width, core_area_ratio = _pick_corridor_width_and_core_ratio(floor_w * floor_h)

    orchestrator = BuildingOrchestrator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width,
        core_area_ratio=core_area_ratio,
        corridor_layout="door_side",
    )
    try:
        orchestrator._shared_core_tube = CoreTube.create_for_floor(
            floor_bounds=floor_boundary.bounds,
            area_ratio=core_area_ratio,
            position=args.core,
        )
    except Exception as e:
        print(f"[full_pipeline] core override failed: {type(e).__name__}: {e}", file=sys.stderr)

    building_result = orchestrator.generate(allocation)
    building_dict = building_result_to_dict(building_result, floor_boundary)

    floors = building_dict["building"]["floors"]
    floor_ids = sorted(list(floors.keys()))
    selected = _parse_floors_arg(args.floors, floor_ids)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(PROJECT_ROOT) / "out" / (datetime.now().strftime("%Y%m%d_%H%M%S_full")))
    out_dir.mkdir(parents=True, exist_ok=True)

    for fid in selected:
        await _run_for_floor(
            floor_id=fid,
            building_dict=building_dict,
            floor_w=floor_w,
            floor_h=floor_h,
            out_dir=out_dir,
            client=client,
            model=args.model,
            concurrency=args.concurrency,
        )

    print(f"[full_pipeline] Done. See: {out_dir}")
    return 0


def main() -> int:
    load_dotenv()
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
