from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Tuple, cast

from shapely.affinity import scale as scale_geom
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import unary_union
import shapely.ops as ops

from ...models import FloorAllocation
from .building_types import FloorSkeleton

MIN_ROOM_AREA = 2.0        # 可用岛屿的最小有效面积（平方米，过滤噪点）
MIN_CORRIDOR_WIDTH = 1.2   # 满足消防疏散的最小走廊宽度
MAX_CORRIDOR_WIDTH = 4.0   # 避免走廊过度浪费的最大宽度
CORRIDOR_SHRINK_TOLERANCE = 1e-6 # 容差值


def _as_polygons(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_as_polygons(g))
        return out
    return []


def _as_lines(geom) -> List[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[LineString] = []
        for g in geom.geoms:
            out.extend(_as_lines(g))
        return out
    return []


def _repair_polygon(poly: Polygon) -> Polygon:
    if poly.is_empty:
        return poly
    if poly.is_valid:
        return poly
    repaired = poly.buffer(0)
    if isinstance(repaired, Polygon) and not repaired.is_empty and repaired.is_valid:
        return repaired
    if isinstance(repaired, MultiPolygon) and len(repaired.geoms) > 0:
        largest = max(repaired.geoms, key=lambda p: p.area)
        if isinstance(largest, Polygon) and largest.is_valid:
            return largest
    return poly


def _validate_outline_points(outline_points: Optional[List[Tuple[float, float]]]) -> bool:
    if not outline_points:
        return False
    if len(outline_points) < 3:
        return False
    unique = {(float(x), float(y)) for x, y in outline_points}
    if len(unique) < 3:
        return False
    return True


def _default_rect_boundary(area: float) -> Polygon:
    if area <= 0:
        raise ValueError("floor_total_area must be > 0")
    w = math.sqrt(area * 3.0 / 2.0)
    h = math.sqrt(area * 2.0 / 3.0)
    return Polygon([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])


def _align_area_to_target(boundary: Polygon, target_area: float) -> Polygon:
    boundary = _repair_polygon(boundary)
    if boundary.is_empty:
        raise ValueError("boundary polygon is empty")
    if target_area <= 0:
        raise ValueError("target_area must be > 0")
    real_area = float(boundary.area)
    if real_area <= 0:
        raise ValueError("boundary polygon area must be > 0")
    scale_factor = math.sqrt(target_area / real_area)
    scaled = scale_geom(boundary, xfact=scale_factor, yfact=scale_factor, origin=(0.0, 0.0))
    scaled = _repair_polygon(scaled)
    if scaled.is_empty or scaled.area <= 0:
        raise ValueError("failed to scale boundary polygon to target area")
    return scaled


def _pick_anchor_point(boundary: Polygon) -> Point:
    boundary = _repair_polygon(boundary)
    try:
        polylabel_fn = getattr(ops, "polylabel", None)
        if callable(polylabel_fn):
            p = polylabel_fn(boundary, tolerance=1.0)
        else:
            p = None
        if isinstance(p, Point) and p.within(boundary):
            return p
    except Exception:
        pass
    p2 = boundary.representative_point()
    if not p2.within(boundary):
        return Point(boundary.bounds[0], boundary.bounds[1])
    return p2


def _square_at_center(area: float, center: Point) -> Polygon:
    if area <= 0:
        raise ValueError("core_tube_area must be > 0")
    side = math.sqrt(area)
    half = side / 2.0
    cx = float(center.x)
    cy = float(center.y)
    return Polygon([(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)])


def _shrink_to_fit(core: Polygon, boundary: Polygon) -> Polygon:
    core = _repair_polygon(core)
    boundary = _repair_polygon(boundary)
    if core.is_empty:
        raise ValueError("core polygon is empty")

    eps = 1e-6
    if core.within(boundary.buffer(-eps)):
        return core

    candidate = core
    # 每次缩小 10%，最多尝试 30 次，使用 origin='center' 保证原地中心缩小
    for _ in range(30):
        candidate = scale_geom(candidate, xfact=0.9, yfact=0.9, origin='center')
        candidate = _repair_polygon(candidate)
        if candidate.is_empty:
            continue
        if candidate.within(boundary.buffer(-eps)):
            return candidate

    raise ValueError("core_tube_polygon cannot fit within boundary after shrinking")


def _corridor_centerlines(boundary: Polygon, anchor: Point) -> Tuple[LineString, LineString]:
    minx, miny, maxx, maxy = boundary.bounds
    cx = float(anchor.x)
    cy = float(anchor.y)
    h = LineString([(minx, cy), (maxx, cy)])
    v = LineString([(cx, miny), (cx, maxy)])
    return h, v


def _corridor_width_from_allowance(
    boundary: Polygon,
    lines: Iterable[LineString],
    corridor_allowance_area: float,
) -> float:
    if corridor_allowance_area <= 0:
        return 1.5
    inside = []
    for ln in lines:
        inside.append(ln.intersection(boundary))
    length = sum(seg.length for g in inside for seg in _as_lines(g))
    if length <= 1e-6:
        return 1.5
    width = corridor_allowance_area / length
    if width < MIN_CORRIDOR_WIDTH:
        return MIN_CORRIDOR_WIDTH
    if width > MAX_CORRIDOR_WIDTH:
        return MAX_CORRIDOR_WIDTH
    return float(width)


def _keep_components_connected_to_core(corridor, core: Polygon):
    if corridor.is_empty:
        return corridor
    polys = _as_polygons(corridor)
    if not polys:
        return corridor
    eps = 1e-6
    core_probe = core.buffer(eps)
    kept = [p for p in polys if p.intersects(core_probe)]
    if not kept:
        return corridor
    if len(kept) == 1:
        return kept[0]
    return unary_union(kept)


def _connect_core_to_corridor_if_needed(corridor, core: Polygon, width: float, boundary: Polygon):
    if corridor.is_empty:
        corridor = Polygon()
    if corridor.intersects(core) or corridor.touches(core):
        return corridor

    try:
        core_anchor = core.representative_point()
        nearest_points_fn = getattr(ops, "nearest_points", None)
        if callable(nearest_points_fn):
            typed_nearest_points = cast(Callable[[object, object], Tuple[Point, Point]], nearest_points_fn)
            _, p_on_corridor = typed_nearest_points(core_anchor, corridor)
        else:
            p_on_corridor = corridor.representative_point()
        connector = LineString([core_anchor, p_on_corridor])
        patch = (
            connector.buffer(width / 2.0, cap_style="flat", join_style="mitre")
            .intersection(boundary)
            .difference(core)
        )
        return unary_union([corridor, patch])
    except Exception:
        return corridor

#走廊膨胀+核心筒环路
def generate_floor_skeleton(
    floor_data: FloorAllocation,
    outline_points: Optional[List[Tuple[float, float]]] = None,
) -> FloorSkeleton:
    if _validate_outline_points(outline_points):
        assert outline_points is not None
        boundary = Polygon([(float(x), float(y)) for x, y in outline_points])
        boundary = _repair_polygon(boundary)
        if boundary.is_empty or boundary.area <= 0:
            boundary = _default_rect_boundary(float(floor_data.floor_total_area))
        else:
            boundary = _align_area_to_target(boundary, float(floor_data.floor_total_area))
    else:
        boundary = _default_rect_boundary(float(floor_data.floor_total_area))

    boundary = _repair_polygon(boundary)
    if boundary.is_empty or boundary.area <= 0:
        raise ValueError("failed to build a valid boundary polygon")

    anchor = _pick_anchor_point(boundary)

    core = _square_at_center(float(floor_data.core_tube_area), anchor)
    core = _shrink_to_fit(core, boundary)

    h_line, v_line = _corridor_centerlines(boundary, anchor)
    width = _corridor_width_from_allowance(boundary, [h_line, v_line], float(floor_data.corridor_allowance_area))


    centerline = unary_union([h_line, v_line])
    cross_corridor = centerline.buffer(width / 2.0, cap_style='flat', join_style='mitre')
    
    core_ring = core.buffer(width, join_style='mitre')
    
    full_corridor = unary_union([cross_corridor, core_ring])

    corridor = full_corridor.intersection(boundary)
    corridor = corridor.difference(core)
    corridor = _keep_components_connected_to_core(corridor, core)
    corridor = _connect_core_to_corridor_if_needed(corridor, core, width, boundary)

    corridor = corridor.intersection(boundary)
    corridor = corridor.difference(core)
    corridor = _keep_components_connected_to_core(corridor, core)

    usable_area = boundary.difference(unary_union([core, corridor]))
    islands = []
    for p in _as_polygons(usable_area):
        if p.is_empty:
            continue
        if p.area < MIN_ROOM_AREA:
            continue
        if not p.intersects(corridor):
            continue
        islands.append(_repair_polygon(p))

    islands.sort(key=lambda p: p.area, reverse=True)

    corridor_geom = unary_union(_as_polygons(corridor))
    if isinstance(corridor_geom, (Polygon, MultiPolygon)):
        corridor_poly = corridor_geom
    else:
        corridor_poly = MultiPolygon() # 兜底机制

    return FloorSkeleton(
        boundary_polygon=boundary,
        core_tube_polygon=core,
        corridor_polygon=corridor_poly,
        usable_islands=islands,
    )
