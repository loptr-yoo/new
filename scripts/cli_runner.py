#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cli_runner.py

纯本地/无头（headless）生成触发器：
1) 调用 Building 语义规划（LLM）得到 BuildingAllocation
2) 调用 BuildingOrchestrator 生成几何（Rooms/Walls/Doors/Windows）
3) 扁平化为单层 layout.json（width/height/elements[]），用于本地 Matplotlib 渲染回归

重要约束（为避免常见坑）：
- 必须 load_dotenv() 加载 .env，否则可能出现 API Key Missing
- generate_building_semantics 是异步函数，必须 asyncio.run(...) 启动
- floor_boundary 固定原点为 (0,0)，避免出现负坐标导致渲染器额外复杂
- 离散型实体（door/window）统一用中心点锚点：x,y 表示几何中心点 (cx,cy)，rotation 围绕中心旋转
- 输出时强制注入 zOrder，渲染端只读，不猜
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from shapely.geometry import box

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.flows.building_semantic_flow import generate_building_semantics
from backend.core.geometry.building_orchestrator import BuildingOrchestrator
from backend.core.geometry.serializers import building_result_to_dict
from backend.core.geometry.topology_generator import CoreTube
from backend.models import GenerateSemanticsRequest, SceneType


Z_ORDER_MAP: Dict[str, int] = {
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


def _is_room_type(t: str) -> bool:
    """
    约定：除了结构类型外，其它都视为房间/功能块，走同一 zOrder 档位。
    """
    structural = {
        "floor_slab",
        "corridor",
        "elevator",
        "elevator_hall",
        "elevator_shaft",
        "staircase",
        "staircase_hall",
        "staircase_shaft",
        "partition_wall",
        "exterior_wall",
        "wall",
        "door",
        "window",
    }
    return t not in structural


def _zorder_for(elem_type: str) -> int:
    if _is_room_type(elem_type):
        return 20
    return Z_ORDER_MAP.get(elem_type, 20)


def _bounds_from_polygon(poly: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    if not poly or len(poly) < 3:
        return None
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _pick_first_floor_id(floors: Dict[str, Any]) -> str:
    if "F1" in floors:
        return "F1"
    return sorted(floors.keys())[0]


def _derive_floor_boundary_from_allocation(allocation: Any) -> Tuple[float, float, Any]:
    """
    用显式数学逻辑从首层面积推导 width/height：
      width = sqrt(area * 1.5)
      height = area / width
      floor_boundary = box(0,0,width,height)
    """
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
    """
    与 API 端相同口径：
    - corridor_width: <80 => 1.5, <120 => 2.0, else 2.5
    - core_area_ratio: <80 => 0.08, else 0.12
    """
    if floor_area < 80:
        return 1.5, 0.08
    if floor_area < 120:
        return 2.0, 0.12
    return 2.5, 0.12


def _flatten_floor_to_elements(
    floor_id: str,
    building_dict: Dict[str, Any],
    floor_boundary_width: float,
    floor_boundary_height: float,
) -> Dict[str, Any]:
    """
    将 building_result_to_dict 的 floors[floor_id] 扁平化为：
      {width,height,elements:[...],sceneId}

    注意：本地渲染只需要几何表达，无需引入前端 SVG 细节。
    """
    floors = building_dict["building"]["floors"]
    floor_data = floors[floor_id]

    elements: List[Dict[str, Any]] = []

    # Layer 1: floor slab
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

    # Layer 2: corridors
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

    # Layer 3: rooms
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

    # Layer 4: core tube sub-areas (optional, keep as polygons if present)
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

    # Layer 5: walls (polygon preferred)
    for w in floor_data.get("walls", []) or []:
        poly = w.get("polygon") or []
        if poly and len(poly) >= 3:
            b = _bounds_from_polygon(poly)
            if b is None:
                continue
            minx, miny, maxx, maxy = b
            elements.append({
                "id": f"{floor_id}_wall_{len(elements)}",
                "type": w.get("type") or "wall",
                "polygon": poly,
                "x": round(minx, 2),
                "y": round(miny, 2),
                "width": round(maxx - minx, 2),
                "height": round(maxy - miny, 2),
                "thickness": w.get("thickness"),
                "room_ids": w.get("room_ids"),
                "zOrder": _zorder_for(w.get("type") or "wall"),
            })

    # Layer 6: doors/windows (rect, anchor=center)
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
    partition_thickness = 0.12
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
            "forward": d.get("forward"),
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
            "forward": wv.get("forward"),
            "thickness": float(wv.get("thickness") or window_depth),
            "zOrder": _zorder_for("window"),
        })

    return {
        "width": round(floor_boundary_width, 2),
        "height": round(floor_boundary_height, 2),
        "elements": elements,
        "sceneId": "building_floor_plan",
    }


async def _run_async(args: argparse.Namespace) -> int:
    req = GenerateSemanticsRequest(
        scene_type=SceneType.BUILDING,
        user_prompt=args.prompt,
        model=args.model,
    )

    allocation, parse_warnings = await generate_building_semantics(req)
    if parse_warnings:
        print(f"[cli_runner] parse_warnings: {parse_warnings}", file=sys.stderr)

    width, height, floor_boundary = _derive_floor_boundary_from_allocation(allocation)
    corridor_width, core_area_ratio = _pick_corridor_width_and_core_ratio(width * height)

    corridor_layout = "door_side"
    if str(getattr(args, "corridor_mode", "") or "").lower() == "organic":
        corridor_layout = "organic"

    orchestrator = BuildingOrchestrator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width,
        core_area_ratio=core_area_ratio,
        corridor_layout=corridor_layout,
        base_seed=getattr(args, "seed", None),
    )

    try:
        orchestrator._shared_core_tube = CoreTube.create_for_floor(
            floor_bounds=floor_boundary.bounds,
            area_ratio=core_area_ratio,
            position=args.core_placement,
        )
    except Exception as e:
        print(f"[cli_runner] core override failed: {type(e).__name__}: {e}", file=sys.stderr)

    building_result = orchestrator.generate(allocation)
    building_dict = building_result_to_dict(building_result, floor_boundary)

    floors = building_dict["building"]["floors"]
    if not floors:
        print("[cli_runner] No floors generated.", file=sys.stderr)
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 遍历 building_dict 里的所有楼层
    for floor_id in floors.keys():
        # 分别把每一层的数据拍扁
        layout = _flatten_floor_to_elements(floor_id, building_dict, width, height)
        
        # 动态生成文件名，例如：out/layout.json 变成 out/layout_F1.json, out/layout_F2.json
        floor_file_name = f"{out_path.stem}_{floor_id}{out_path.suffix}"
        floor_out_path = out_path.parent / floor_file_name
        
        # 保存该层的 JSON
        floor_out_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[cli_runner] Wrote Floor {floor_id}: {floor_out_path}")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Building V2 headless generator (LLM -> geometry -> layout.json)")
    p.add_argument("-p", "--prompt", required=True, help="提示词 / 用户需求描述")
    p.add_argument("-m", "--model", required=True, help="模型名称（透传到后端 LLM 配置）")
    p.add_argument(
        "-c",
        "--core",
        "--core-placement",
        dest="core_placement",
        required=True,
        choices=["north", "center", "south", "east", "west"],
        help="核心筒位置",
    )
    p.add_argument(
        "--corridor-mode",
        default="door_side",
        choices=["door_side", "organic"],
        help="走廊模式",
    )
    p.add_argument(
        "--seed",
        default=None,
        type=int,
        help="随机种子（用于标准层分组/端头退让；不传则使用默认值）",
    )
    p.add_argument("-o", "--output", required=True, help="输出 JSON 路径（layout.json）")
    return p.parse_args(argv)


def main() -> int:
    load_dotenv()
    args = _parse_args()
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
