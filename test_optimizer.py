"""
test_cma_optimizer.py  (已迁移至 Treemap + MIQP 流程)

用法:  cd new-main && python test_cma_optimizer.py

输出:  artifacts/optimized_layout.svg
       artifacts/optimized_layout_detail.svg  (带标注)
"""
import os
from pathlib import Path
from typing import List, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from backend.core.geometry.island_partition_solver import (
    RoomResult,
    partition_island,
    partition_island_semantic,
)
from backend.core.geometry.room_spec import (
    RoomSpec as SemanticRoomSpec,
    RoomSpec as OldRoomSpec,
    SolverConfig,
    ZoneType,
)


# ============================================================
# SVG 导出（复用 test_power_diagram.py 的思路，增加矩形标注）
# ============================================================

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#17becf", "#bcbd22", "#aec7e8", "#ffbb78",
]


def _fmt(v: float) -> str:
    s = f"{float(v):.3f}"
    s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _as_polygons(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_as_polygons(g))
        return out
    return []


def _ring_to_path(coords) -> str:
    pts = list(coords)
    if len(pts) < 3:
        return ""
    parts = [f"M {_fmt(pts[0][0])} {_fmt(pts[0][1])}"]
    for x, y in pts[1:]:
        parts.append(f"L {_fmt(x)} {_fmt(y)}")
    parts.append("Z")
    return " ".join(parts)


def _polygon_to_path(p: Polygon) -> str:
    d = _ring_to_path(p.exterior.coords)
    for hole in p.interiors:
        d2 = _ring_to_path(hole.coords)
        if d2:
            d = f"{d} {d2}"
    return d


def export_layout_svg(
    boundary: Polygon,
    results: List[RoomResult],
    filepath: str,
    show_labels: bool = True,
) -> None:
    """将 MIQP 分房结果导出为 SVG"""
    # 计算 viewBox
    minx, miny, maxx, maxy = boundary.bounds
    pad = max(1.0, 0.03 * max(maxx - minx, maxy - miny))
    vx, vy = minx - pad, miny - pad
    vw, vh = (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad

    # Y 轴翻转参数
    translate_y = miny + maxy

    elems: List[str] = []

    # 画房间（填色矩形）
    for i, room in enumerate(results):
        color = PALETTE[i % len(PALETTE)]
        for p in _as_polygons(room.polygon):
            d = _polygon_to_path(p)
            if d:
                elems.append(
                    f'<path d="{d}" fill="{color}" fill-opacity="0.85" '
                    f'stroke="#333333" stroke-width="0.3" />'
                )

    # 画边界轮廓
    for p in _as_polygons(boundary):
        d = _polygon_to_path(p)
        if d:
            elems.append(
                f'<path d="{d}" fill="none" stroke="#000000" stroke-width="0.6" />'
            )

    # 画标注（房间名 + 面积）
    label_elems: List[str] = []
    if show_labels:
        font_size = max(1.2, min(3.0, vw / 25))
        for i, room in enumerate(results):
            cx, cy = room.center
            # 翻转 Y
            cy_flipped = translate_y - cy
            area_str = f"{room.actual_area:.1f}sqm"
            label_elems.append(
                f'<text x="{_fmt(cx)}" y="{_fmt(cy_flipped - font_size * 0.3)}" '
                f'font-size="{_fmt(font_size)}" text-anchor="middle" fill="#ffffff" '
                f'font-family="monospace" font-weight="bold">{room.room_id}</text>'
            )
            label_elems.append(
                f'<text x="{_fmt(cx)}" y="{_fmt(cy_flipped + font_size * 0.9)}" '
                f'font-size="{_fmt(font_size * 0.7)}" text-anchor="middle" fill="#eeeeee" '
                f'font-family="monospace">{area_str}</text>'
            )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{_fmt(vx)} {_fmt(vy)} {_fmt(vw)} {_fmt(vh)}">\n'
        f'  <rect x="{_fmt(vx)}" y="{_fmt(vy)}" width="{_fmt(vw)}" height="{_fmt(vh)}" fill="#1e1e2e" />\n'
        f'  <g transform="translate(0,{_fmt(translate_y)}) scale(1,-1)" shape-rendering="crispEdges">\n'
        f"    {''.join(elems)}\n"
        "  </g>\n"
        f"  {''.join(label_elems)}\n"
        "</svg>\n"
    )

    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")


# ============================================================
# 主测试
# ============================================================


def main() -> None:
    config = SolverConfig(time_limit=15.0, num_workers=4)

    # ---- L 型边界 (语义版) ----
    boundary = Polygon([
        (0.0, 0.0),
        (40.0, 0.0),
        (40.0, 40.0),
        (0.0, 40.0),
        (0.0, 25.0),
        (15.0, 25.0),
        (15.0, 15.0),
        (0.0, 15.0),
    ])

    total_usable_area = float(boundary.area)
    print(f"island area: {total_usable_area:.1f} sqm")

    ratios = [1.0, 2.0, 4.0, 8.0]
    total_ratio = sum(ratios)

    rooms = [
        SemanticRoomSpec(
            room_id="R1", room_type="bathroom",
            target_area=(ratios[0] / total_ratio) * total_usable_area,
            zone=ZoneType.PRIVATE, needs_window=False,
            min_width=2.0, min_depth=2.0,
            adjacency_required=["R2"],
        ),
        SemanticRoomSpec(
            room_id="R2", room_type="bedroom",
            target_area=(ratios[1] / total_ratio) * total_usable_area,
            zone=ZoneType.PRIVATE, needs_window=True,
            min_width=2.0, min_depth=2.0,
            adjacency_required=["R1"],
        ),
        SemanticRoomSpec(
            room_id="R3", room_type="kitchen",
            target_area=(ratios[2] / total_ratio) * total_usable_area,
            zone=ZoneType.PUBLIC, needs_window=True,
            min_width=3.0, min_depth=3.0,
            adjacency_required=["R4"],
        ),
        SemanticRoomSpec(
            room_id="R4", room_type="living_room",
            target_area=(ratios[3] / total_ratio) * total_usable_area,
            zone=ZoneType.PUBLIC, needs_window=True,
            min_width=4.0, min_depth=4.0,
            adjacency_required=["R3"],
        ),
    ]
    adj_graph = {
        "R1": ["R2"], "R2": ["R1"],
        "R3": ["R4"], "R4": ["R3"],
    }

    for r in rooms:
        print(f"  {r.room_id} ({r.zone.value}): target={r.target_area:.1f} sqm, window={r.needs_window}")

    print("\n>>> Semantic Treemap + MIQP ...")
    results = partition_island_semantic(
        island_polygon=boundary,
        rooms=rooms,
        adjacency_graph=adj_graph,
        exterior_walls=["north", "south", "east", "west"],
        config=config,
    )

    print(f"\n>>> Done, {len(results)} rooms:")
    for r in results:
        target = next((s.target_area for s in rooms if s.room_id == r.room_id), 0)
        error = abs(r.actual_area - target) / target * 100 if target > 0 else 0
        ar = r.width / r.depth if r.depth > 0 else float("inf")
        print(f"  {r.room_id}: ({r.x:.1f}, {r.y:.1f}) {r.width:.1f}x{r.depth:.1f} "
              f"= {r.actual_area:.1f} sqm (target {target:.1f}, err {error:.1f}%, AR {ar:.2f})")

    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)

    svg_path = os.path.join(output_dir, "optimized_layout.svg")
    export_layout_svg(boundary, results, svg_path, show_labels=True)
    print(f"\n>>> SVG: {os.path.abspath(svg_path)}")

    # ---- 8 房间矩形岛屿 ----
    print("\n" + "=" * 50)
    print(">>> 8-room rectangular island (semantic)...")

    boundary2 = Polygon([(0, 0), (50, 0), (50, 30), (0, 30)])
    # 总面积 1500 sqm，按比例分配使房间填满岛屿
    base_areas_8 = [50, 70, 90, 110, 130, 150, 170, 190]
    scale_8 = 1500.0 / sum(base_areas_8)  # = 1.5625
    room_types_8 = ["living_room", "dining_room", "kitchen", "bedroom",
                     "bedroom", "bathroom", "bathroom", "storage"]
    zones_8 = [ZoneType.PUBLIC, ZoneType.PUBLIC, ZoneType.PUBLIC, ZoneType.PRIVATE,
               ZoneType.PRIVATE, ZoneType.PRIVATE, ZoneType.PRIVATE, ZoneType.SERVICE]
    rooms2 = [
        SemanticRoomSpec(
            room_id=f"Room{i+1}", room_type=room_types_8[i],
            target_area=base_areas_8[i] * scale_8, zone=zones_8[i],
            needs_window=(room_types_8[i] in ("living_room", "bedroom", "kitchen")),
            min_width=3, min_depth=3,
            adjacency_required=(
                [f"Room{i+2}"] if i < 2 else
                [f"Room{i}"] if i == 2 else
                []
            ),
        )
        for i in range(8)
    ]
    adj2 = {
        "Room1": ["Room2"], "Room2": ["Room1", "Room3"], "Room3": ["Room2"],
    }

    results2 = partition_island_semantic(
        boundary2, rooms2, adj2,
        ["north", "south", "east", "west"], config,
    )

    print(f">>> {len(results2)} rooms:")
    for r in results2:
        ar = r.width / r.depth if r.depth > 0 else float("inf")
        print(f"  {r.room_id}: {r.width:.1f}x{r.depth:.1f} = {r.actual_area:.1f} sqm (AR {ar:.2f})")

    svg_path2 = os.path.join(output_dir, "optimized_layout_8rooms.svg")
    export_layout_svg(boundary2, results2, svg_path2, show_labels=True)
    print(f">>> SVG: {os.path.abspath(svg_path2)}")

    # ---- 住宅户型 ----
    print("\n" + "=" * 50)
    print(">>> Residential layout (5 rooms, semantic)...")

    boundary3 = Polygon([(0, 0), (20, 0), (20, 15), (0, 15)])
    # 总面积 300 sqm，按比例分配
    rooms3 = [
        SemanticRoomSpec(
            room_id="living", room_type="living_room", target_area=90,
            zone=ZoneType.PUBLIC, needs_window=True,
            min_width=4, min_depth=4,
            adjacency_required=["dining"],
        ),
        SemanticRoomSpec(
            room_id="master", room_type="bedroom", target_area=70,
            zone=ZoneType.PRIVATE, needs_window=True,
            min_width=3, min_depth=3,
            adjacency_required=["bath"],
        ),
        SemanticRoomSpec(
            room_id="bed2", room_type="bedroom", target_area=55,
            zone=ZoneType.PRIVATE, needs_window=True,
            min_width=3, min_depth=3,
        ),
        SemanticRoomSpec(
            room_id="dining", room_type="kitchen", target_area=50,
            zone=ZoneType.PUBLIC, needs_window=True,
            min_width=2.5, min_depth=2,
            adjacency_required=["living"],
        ),
        SemanticRoomSpec(
            room_id="bath", room_type="bathroom", target_area=35,
            zone=ZoneType.PRIVATE, needs_window=False,
            min_width=2, min_depth=2,
            adjacency_required=["master"],
        ),
    ]
    adj3 = {
        "living": ["dining"], "dining": ["living"],
        "master": ["bath"], "bath": ["master"],
    }

    results3 = partition_island_semantic(
        boundary3, rooms3, adj3,
        ["north", "south", "east", "west"], config,
    )

    print(f">>> {len(results3)} rooms:")
    for r in results3:
        ar = r.width / r.depth if r.depth > 0 else float("inf")
        print(f"  {r.room_id}: {r.width:.1f}x{r.depth:.1f} = {r.actual_area:.1f} sqm (AR {ar:.2f})")

    svg_path3 = os.path.join(output_dir, "optimized_layout_residential.svg")
    export_layout_svg(boundary3, results3, svg_path3, show_labels=True)
    print(f">>> SVG: {os.path.abspath(svg_path3)}")


if __name__ == "__main__":
    main()
