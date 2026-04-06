from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

MIN_ORTHO_AREA = 0.1  # sq.m - polygons smaller than this are treated as slivers
_SEED_MATCH_RADIUS = 0.5  # meters - tolerant seed claiming buffer


# ---------------------------------------------------------------------------
# Step 1-2: Extract internal skeleton + simplify
# ---------------------------------------------------------------------------

def _extract_internal_skeleton(boundary: Polygon, cells: List[Polygon]) -> BaseGeometry:
    """Extract and simplify the internal wall skeleton from cell boundaries."""
    boundaries = [c.boundary for c in cells if not c.is_empty]
    if not boundaries:
        return LineString()

    all_edges = unary_union(boundaries)
    # Use buffered exterior to avoid float-precision artifacts (hairline slivers)
    exterior_zone = boundary.exterior.buffer(1e-3)
    internal = all_edges.difference(exterior_zone)

    if internal.is_empty:
        return internal

    internal = internal.simplify(1.0, preserve_topology=True)
    return internal


# ---------------------------------------------------------------------------
# Step 3: Force orthogonal segments
# ---------------------------------------------------------------------------

def _extract_lines(geometry: BaseGeometry) -> List[LineString]:
    """Extract LineStrings from mixed geometries (including GeometryCollection)."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        out: List[LineString] = []
        for part in geometry.geoms:
            out.extend(_extract_lines(part))
        return out
    return []

def _force_orthogonal_segments(skeleton) -> List[LineString]:
    """
    通过 '连续端点追踪 (Continuous Endpoint Tracking)' 强制正交化。
    保证内部绝对连通，不新增额外顶点，并通过轻微延长确保 T 型路口完美闭合。
    """
    ortho_lines: List[LineString] = []
    extend_dist = 1.0  # 延长线段以形成完美的十字交叉路口

    for line in _extract_lines(skeleton):
        pts = list(line.coords)
        if len(pts) < 2:
            continue
            
        new_pts = [pts[0]]
        # 1. 连续追踪对齐：保证同一条线内部绝不断裂
        for i in range(1, len(pts)):
            p_prev = new_pts[-1]  # ⚠️ 核心：以上一个已经修改过的点为基准！
            p_orig = pts[i]
            
            dx = p_orig[0] - p_prev[0]
            dy = p_orig[1] - p_prev[1]
            
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
                
            if abs(dx) > abs(dy):
                # 趋于水平 → 强制 Y 坐标等于上一个点的 Y (完美对接)
                new_pts.append((p_orig[0], p_prev[1]))
            else:
                # 趋于垂直 → 强制 X 坐标等于上一个点的 X (完美对接)
                new_pts.append((p_prev[0], p_orig[1]))
                
        # 2. 延长线段：解决不同墙壁交汇时的微小缝隙
        for i in range(len(new_pts) - 1):
            p1 = new_pts[i]
            p2 = new_pts[i+1]
            length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            if length < 1e-6: 
                continue
                
            ux = (p2[0] - p1[0]) / length
            uy = (p2[1] - p1[1]) / length
            
            # 向两头分别延长 1.0 米
            ex1 = (p1[0] - ux * extend_dist, p1[1] - uy * extend_dist)
            ex2 = (p2[0] + ux * extend_dist, p2[1] + uy * extend_dist)
            
            ortho_lines.append(LineString([ex1, ex2]))

    return ortho_lines
# ---------------------------------------------------------------------------
# Step 4-5: Polygonize reassembly + seed claiming
# ---------------------------------------------------------------------------

def _as_polygons(geom: BaseGeometry) -> List[Polygon]:
    """Extract all Polygon instances from an arbitrary geometry."""
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


def _reassemble_and_claim(
    boundary: Polygon,
    ortho_lines: List[LineString],
    seeds: List[Tuple[float, float]],
    original_cells: List[Polygon],
) -> List[Polygon]:
    """Polygonize orthogonal lines + boundary exterior, then assign to seeds."""
    n = len(seeds)
    if not ortho_lines:
        return list(original_cells)

    # Step 4: Build line soup & polygonize
    line_geoms: List[BaseGeometry] = list(ortho_lines)
    line_geoms.append(LineString(boundary.exterior.coords))
    blueprint = unary_union(line_geoms)
    raw_polys = list(polygonize(blueprint))

    if not raw_polys:
        return list(original_cells)

    # Filter slivers and clip to boundary
    valid_polys: List[Polygon] = []
    for poly in raw_polys:
        clipped_parts = _as_polygons(poly.intersection(boundary))
        for cp in clipped_parts:
            if cp.area >= MIN_ORTHO_AREA:
                valid_polys.append(cp)

    if not valid_polys:
        return list(original_cells)

    # Step 5: Seed claiming
    final_cells: List[Polygon] = [Polygon()] * n
    claimed: List[bool] = [False] * len(valid_polys)  # track which polygons are claimed

    for si, (sx, sy) in enumerate(seeds):
        seed_buf = Point(sx, sy).buffer(_SEED_MATCH_RADIUS)
        candidates: List[Tuple[int, float]] = []  # (poly_index, distance_to_centroid)
        for pi, poly in enumerate(valid_polys):
            if poly.intersects(seed_buf):
                dist = Point(sx, sy).distance(poly.centroid)
                candidates.append((pi, dist))

        if candidates:
            # Pick closest centroid match
            candidates.sort(key=lambda t: t[1])
            best_pi = candidates[0][0]
            if final_cells[si].is_empty:
                final_cells[si] = valid_polys[best_pi]
                claimed[best_pi] = True
        else:
            # Fallback: keep original cell
            if si < len(original_cells):
                final_cells[si] = original_cells[si] if isinstance(original_cells[si], Polygon) else Polygon()

    # Merge unclaimed polygons into nearest seed (by distance)
    for pi, poly in enumerate(valid_polys):
        if claimed[pi]:
            continue
        best_idx = 0
        min_dist = float("inf")
        for si, (sx, sy) in enumerate(seeds):
            d = poly.distance(Point(sx, sy))
            if d < min_dist:
                min_dist = d
                best_idx = si
        # Buffer-weld to prevent flyover (MultiPolygon)
        welded = unary_union([final_cells[best_idx].buffer(1e-3), poly.buffer(1e-3)])
        polys = _as_polygons(welded)
        if polys:
            final_cells[best_idx] = max(polys, key=lambda p: p.area)
        claimed[pi] = True

    # Thorough gap filling: distribute ALL uncovered boundary area
    claimed_union = unary_union([c for c in final_cells if not c.is_empty])
    leftover_area = boundary.difference(claimed_union)
    if not leftover_area.is_empty:
        leftover_polys = list(getattr(leftover_area, 'geoms', [leftover_area]))
        for piece in leftover_polys:
            if piece.area < MIN_ORTHO_AREA:
                continue
            best_idx = 0
            min_dist = float("inf")
            for si, (sx, sy) in enumerate(seeds):
                d = piece.distance(Point(sx, sy))
                if d < min_dist:
                    min_dist = d
                    best_idx = si
            # Buffer-weld then shrink back to prevent flyover
            welded = unary_union([final_cells[best_idx].buffer(1e-3), piece.buffer(1e-3)])
            polys = _as_polygons(welded)
            if polys:
                final_cells[best_idx] = max(polys, key=lambda p: p.area)

    # Final clip to boundary
    for si in range(n):
        if not final_cells[si].is_empty:
            clipped = final_cells[si].intersection(boundary)
            polys = _as_polygons(clipped)
            final_cells[si] = max(polys, key=lambda p: p.area) if polys else Polygon()

    return final_cells


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def orthogonalize_layout(
    boundary: Polygon,
    cells: List[Polygon],
    seeds: List[Tuple[float, float]],
) -> List[Polygon]:
    """Post-process rasterized room cells into orthogonal (axis-aligned) rooms.

    Implements the SPO-LGP algorithm:
    1. Extract internal wall skeleton
    2. Simplify rasterization noise
    3. Force all segments to horizontal/vertical
    4. Polygonize with original boundary exterior
    5. Reclaim polygons by seed proximity
    """
    if not cells or all(c.is_empty for c in cells):
        return list(cells)
    if boundary.is_empty:
        return list(cells)
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    if boundary.is_empty:
        return list(cells)

    try:
        # Step 1-2: 提取内墙骨架并简化
        skeleton = _extract_internal_skeleton(boundary, cells)
        if skeleton.is_empty:
            logger.warning("Orthogonalization: skeleton is empty, returning original cells")
            return list(cells)
        logger.info("Skeleton extracted: %s, length=%.2f", skeleton.geom_type, skeleton.length)

        # Step 3: 强制正交化
        ortho_lines = _force_orthogonal_segments(skeleton)
        if not ortho_lines:
            logger.warning("Orthogonalization: no orthogonal lines generated, returning original cells")
            return list(cells)
        logger.info("Orthogonal segments: %d lines generated", len(ortho_lines))

        # Step 4-5: 重构多边形 + 种子认领
        result = _reassemble_and_claim(boundary, ortho_lines, seeds, cells)
        logger.info("Reassembly complete: %d cells returned", len(result))
        return result
    except Exception as exc:
        logger.error("Orthogonalization failed, returning original cells: %s", exc, exc_info=True)
        return list(cells)
