"""
postprocessor.py

后处理：根据房间 polygon 自动生成墙体、门、窗户。

借鉴 Co-Layout（AAAI 2026）思路：
求解器只负责房间分区，墙/门/窗由后处理启发式规则自动放置。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import (
    CAP_STYLE,
    GeometryCollection,
    JOIN_STYLE,
    LineString,
    LinearRing,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


from .core_contracts import (
    build_core_footprint_contract,
    validate_core_access,
)
from .exceptions import LayoutAssignmentError, LayoutCoverageError, LayoutTopologyError, SemanticInvalidError

try:
    from shapely.validation import make_valid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore[assignment]

try:
    from shapely import set_precision
except Exception:  # pragma: no cover
    set_precision = None


def _largest_polygon(g: BaseGeometry) -> Optional[Polygon]:
    if g is None or g.is_empty:
        return None
    if isinstance(g, Polygon):
        return g
    if isinstance(g, MultiPolygon):
        polys = [p for p in g.geoms if isinstance(p, Polygon) and (not p.is_empty)]
        return max(polys, key=lambda p: float(p.area)) if polys else None
    if isinstance(g, GeometryCollection):
        polys = [p for p in g.geoms if isinstance(p, Polygon) and (not p.is_empty)]
        return max(polys, key=lambda p: float(p.area)) if polys else None
    return None


def _round_polygon(poly: Polygon, round_digits: int) -> Optional[Polygon]:
    if poly is None or poly.is_empty:
        return None
    try:
        coords = [(round(float(x), round_digits), round(float(y), round_digits)) for x, y in poly.exterior.coords]
        p2 = Polygon(coords)
        if (not p2.is_valid) and make_valid is not None:
            fixed = make_valid(p2)
            best = _largest_polygon(fixed) if fixed is not None else None
            if best is not None:
                p2 = best
        return p2 if (p2 is not None and (not p2.is_empty)) else None
    except Exception:
        return poly


def safe_snap_polygon(
    geom: BaseGeometry,
    tol: float,
    *,
    min_area: float = 1e-4,
) -> Optional[Polygon]:
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    if set_precision is None:
        def _q(v: float) -> float:
            return round(float(v) / float(tol)) * float(tol)

        def _dedup_adjacent(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            out: List[Tuple[float, float]] = []
            for x, y in coords:
                p = (float(x), float(y))
                if not out or p != out[-1]:
                    out.append(p)
            return out

        def _drop_short_edges(coords: List[Tuple[float, float]], min_edge: float = 0.01) -> List[Tuple[float, float]]:
            if len(coords) < 4:
                return coords
            changed = True
            while changed and len(coords) >= 4:
                changed = False
                if coords[0] != coords[-1]:
                    coords = list(coords) + [coords[0]]
                    changed = True
                i = 0
                while i < len(coords) - 1 and len(coords) >= 4:
                    x0, y0 = coords[i]
                    x1, y1 = coords[i + 1]
                    if (float(x1) - float(x0)) ** 2 + (float(y1) - float(y0)) ** 2 < float(min_edge) ** 2:
                        coords.pop(i + 1)
                        changed = True
                        if i + 1 == len(coords) - 1:
                            if coords and coords[0] != coords[-1]:
                                coords[-1] = coords[0]
                        continue
                    i += 1
            if coords and coords[0] != coords[-1]:
                coords = list(coords) + [coords[0]]
            return coords

        def _snap_ring(ring: LinearRing) -> Optional[List[Tuple[float, float]]]:
            try:
                coords = [(float(_q(x)), float(_q(y))) for x, y in ring.coords]
            except Exception:
                return None
            coords = _dedup_adjacent(coords)
            coords = _drop_short_edges(coords, min_edge=0.01)
            if len(coords) < 4:
                return None
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            if len(coords) < 4:
                return None
            return coords

        try:
            poly0 = geom if isinstance(geom, Polygon) else _largest_polygon(geom)
        except Exception:
            poly0 = None
        if poly0 is None or poly0.is_empty:
            return None
        ext = _snap_ring(poly0.exterior)
        if ext is None:
            return None
        holes: List[List[Tuple[float, float]]] = []
        for r in poly0.interiors:
            h = _snap_ring(r)
            if h is not None and len(h) >= 4:
                holes.append(h)
        try:
            snapped = Polygon(ext, holes)
        except Exception:
            return None
        g2: BaseGeometry = snapped
    else:
        try:
            g2 = set_precision(geom, float(tol))
            try:
                g2 = set_precision(g2, 0.0)
            except Exception:
                pass
        except Exception:
            g2 = geom

    if g2 is None:
        return None
    try:
        if g2.is_empty:
            return None
    except Exception:
        return None

    poly = _largest_polygon(g2) if not isinstance(g2, Polygon) else g2
    if poly is None or poly.is_empty:
        return None

    if not bool(getattr(poly, "is_valid", True)):
        fixed: BaseGeometry = poly
        if make_valid is not None:
            try:
                fixed = make_valid(poly)
            except Exception:
                fixed = poly
        if fixed is poly:
            try:
                fixed = poly.buffer(0)
            except Exception:
                fixed = poly
        poly2 = _largest_polygon(fixed) if not isinstance(fixed, Polygon) else fixed
        if poly2 is None or poly2.is_empty:
            return None
        poly = poly2

    if float(poly.area) < float(min_area):
        return None

    return poly


def _denoise_corridor_polygon(poly: Polygon, *, r: float = 0.15) -> Polygon:
    p0 = poly
    if p0 is None or getattr(p0, "is_empty", True):
        return p0

    def _snap(p: BaseGeometry) -> Optional[Polygon]:
        out = safe_snap_polygon(p, 0.01)
        return out if out is not None and (not out.is_empty) else None

    cur = _snap(p0) or p0

    try:
        closed = cur.buffer(float(r), join_style=JOIN_STYLE.mitre).buffer(-float(r), join_style=JOIN_STYLE.mitre)
        closed2 = _largest_polygon(closed) or cur
        closed3 = _snap(closed2) or closed2
        cur = closed3
    except Exception:
        pass

    rk = float(r)
    for _ in range(3):
        try:
            eroded = cur.buffer(-rk, join_style=JOIN_STYLE.mitre)
        except Exception:
            break

        if eroded.is_empty or isinstance(eroded, MultiPolygon):
            logger.warning("开运算导致走廊断裂，降低半径重试: r=%.3f", rk)
            rk *= 0.5
            continue

        try:
            opened = eroded.buffer(rk, join_style=JOIN_STYLE.mitre)
        except Exception:
            break

        opened2 = _largest_polygon(opened) or cur
        opened3 = _snap(opened2)
        if opened3 is not None:
            cur = opened3
        else:
            cur = opened2
        break

    return cur


def fuse_dummy_to_corridor(
    *,
    rooms: List[Any],
    corridors: List[Any],
    eps: float = 1e-4,
    round_digits: int = 4,
) -> Tuple[List[Any], List[Any], List[str]]:
    if not rooms or not corridors:
        return list(rooms), list(corridors), []

    corridor_copies: List[Any] = []
    for c in corridors:
        try:
            c2 = c.__class__(
                id=getattr(c, "id"),
                centerline=getattr(c, "centerline"),
                width=getattr(c, "width"),
                orientation=getattr(c, "orientation"),
            )
            c2.polygon = getattr(c, "polygon")
            corridor_copies.append(c2)
        except Exception:
            corridor_copies.append(c)

    warnings: List[str] = []
    remaining_rooms: List[Any] = list(rooms)

    def _iter_dummy_ids() -> List[str]:
        out = []
        for r in remaining_rooms:
            if bool(getattr(r, "is_dummy", False)) and (not bool(getattr(r, "skip_solver", False))):
                rid = getattr(r, "id", getattr(r, "room_id", None))
                if rid:
                    out.append(str(rid))
        return out

    while True:
        fused_any = False
        to_fuse: Dict[str, List[Any]] = {}
        for r in list(remaining_rooms):
            if not bool(getattr(r, "is_dummy", False)):
                continue
            if bool(getattr(r, "skip_solver", False)):
                continue
            poly = getattr(r, "polygon", None)
            if poly is None or poly.is_empty:
                continue

            best_idx = None
            best_score = 0.0
            for i, c in enumerate(corridor_copies):
                cpoly = getattr(c, "polygon", None)
                if cpoly is None or cpoly.is_empty:
                    continue
                try:
                    if not poly.intersects(cpoly.buffer(float(eps))):
                        continue
                    hit = poly.intersection(cpoly.buffer(float(eps)))
                    score = float(getattr(hit, "area", 0.0)) + float(getattr(hit, "length", 0.0)) * float(eps)
                except Exception:
                    continue
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is None:
                continue

            cid = str(getattr(corridor_copies[best_idx], "id", best_idx))
            to_fuse.setdefault(cid, []).append(r)

        if not to_fuse:
            break

        fused_ids: Set[str] = set()
        for c in corridor_copies:
            cid = str(getattr(c, "id", ""))
            if cid not in to_fuse:
                continue
            cpoly = getattr(c, "polygon", None)
            if cpoly is None or cpoly.is_empty:
                continue
            g: BaseGeometry = cpoly
            for r in to_fuse[cid]:
                poly = getattr(r, "polygon", None)
                if poly is None or poly.is_empty:
                    continue
                try:
                    g = g.union(poly)
                except Exception:
                    continue
                rid = getattr(r, "id", getattr(r, "room_id", None))
                if rid:
                    fused_ids.add(str(rid))

            p = _largest_polygon(g)
            if p is None:
                continue
            try:
                p = p.buffer(float(eps)).buffer(-float(eps)).buffer(0)
            except Exception:
                pass
            p2 = safe_snap_polygon(p, 0.01)
            if p2 is None:
                p2 = _round_polygon(p, int(round_digits))
            if p2 is not None and (not p2.is_empty):
                c.polygon = p2

        if fused_ids:
            fused_any = True
            warnings.append(f"Fused dummy rooms into corridor: {sorted(fused_ids)}")
            remaining_rooms = [
                r for r in remaining_rooms
                if str(getattr(r, "id", getattr(r, "room_id", ""))) not in fused_ids
            ]

        if not fused_any:
            break

    for c in corridor_copies:
        poly = getattr(c, "polygon", None)
        if poly is None or getattr(poly, "is_empty", True):
            continue
        if not isinstance(poly, Polygon):
            continue
        try:
            denoised = _denoise_corridor_polygon(poly, r=0.15)
        except Exception:
            continue
        if denoised is not None and (not denoised.is_empty):
            c.polygon = denoised

    return remaining_rooms, corridor_copies, warnings


# ============================================================
# 数据结构
# ============================================================

@dataclass
class WallSegment:
    """墙体段"""
    type: str  # "exterior_wall" | "partition_wall"
    geometry: BaseGeometry  # LineString / MultiLineString
    thickness: float  # 米
    room_ids: List[str]
    forward: Optional[Tuple[float, float, float]] = None
    category: Optional[str] = None
    graph: bool = False

    @property
    def length(self) -> float:
        return self.geometry.length if self.geometry and not self.geometry.is_empty else 0.0


@dataclass
class DoorPlacement:
    """门的放置"""
    position: Tuple[float, float]
    width: float  # 米
    connects: List[str]  # 连接的两个 room_id
    wall_type: str  # 所在墙的类型
    rotation: float = 0.0  # 0=水平墙, 90=垂直墙
    thickness: float = 0.12  # 米
    forward: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    source_portal_spec_id: Optional[str] = None


@dataclass
class WindowPlacement:
    """窗户的放置"""
    position: Tuple[float, float]
    width: float  # 米
    room_id: str
    wall_length: float  # 所在墙段长度
    rotation: float = 0.0  # 0=水平墙, 90=垂直墙
    thickness: float = 0.24  # 米
    forward: Tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass
class PostprocessResult:
    """后处理结果"""
    walls: List[WallSegment] = field(default_factory=list)
    doors: List[DoorPlacement] = field(default_factory=list)
    windows: List[WindowPlacement] = field(default_factory=list)


# ============================================================
# 墙体生成
# ============================================================

def generate_walls(
    rooms: list,
    floor_boundary: Polygon,
    exterior_thickness: float = 0.24,
    partition_thickness: float = 0.12,
    min_wall_length: float = 0.3,
) -> List[WallSegment]:
    """
    根据房间 polygon 共享边自动生成墙体。

    Args:
        rooms: 有 id/room_id, polygon 属性的 RoomResult 列表
        floor_boundary: 楼层外轮廓
        exterior_thickness: 外墙厚度 (m)
        partition_thickness: 隔墙厚度 (m)
        min_wall_length: 最小墙段长度 (m)

    Returns:
        WallSegment 列表
    """
    walls: List[WallSegment] = []

    # 外墙：房间 polygon 与楼层边界的共享边
    for room in rooms:
        room_id = getattr(room, "id", getattr(room, "room_id", "?"))
        poly = room.polygon
        if poly.is_empty:
            continue

        try:
            shared = poly.boundary.intersection(floor_boundary.boundary)
            if not shared.is_empty and shared.length > min_wall_length:
                walls.append(WallSegment(
                    type="exterior_wall",
                    geometry=shared,
                    thickness=exterior_thickness,
                    room_ids=[room_id],
                ))
        except Exception as e:
            logger.debug(f"Exterior wall calc failed for {room_id}: {e}")

    # 内墙：相邻房间 polygon 的共享边
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]
            if a.polygon.is_empty or b.polygon.is_empty:
                continue

            aid = getattr(a, "id", getattr(a, "room_id", "?"))
            bid = getattr(b, "id", getattr(b, "room_id", "?"))

            try:
                shared = a.polygon.boundary.intersection(b.polygon.boundary)
                if not shared.is_empty and shared.length > min_wall_length:
                    walls.append(WallSegment(
                        type="partition_wall",
                        geometry=shared,
                        thickness=partition_thickness,
                        room_ids=[aid, bid],
                    ))
            except Exception as e:
                logger.debug(f"Partition wall calc failed for {aid}-{bid}: {e}")

    return walls


def generate_walls_from_topology(
    room_rects: Dict[str, Tuple[float, float, float, float]],
    edge_set: Dict[FrozenSet[str], str],
    floor_bounds: Tuple[float, float, float, float],
    zone_types: Dict[str, str],
    wall_thickness: float = 0.12,
    exterior_thickness: float = 0.24,
    min_wall_length: float = 0.3,
) -> List[WallSegment]:
    """
    从拓扑边集合直接构造墙体（零 Shapely 求交）。

    Args:
        room_rects: {zone_id: (x, y, w, h)}，包含走廊与核心筒
        edge_set: {frozenset({id_a, id_b}): 'vertical'|'horizontal'}（天然去重）
        floor_bounds: (minx, miny, maxx, maxy)
    """
    walls: List[WallSegment] = []
    fminx, fminy, fmaxx, fmaxy = floor_bounds
    snap_tolerance = 0.2

    floor_poly = box(fminx, fminy, fmaxx, fmaxy)

    orig_room_rects: Dict[str, Tuple[float, float, float, float]] = {
        k: (float(v[0]), float(v[1]), float(v[2]), float(v[3])) for k, v in room_rects.items()
    }
    working_rects: Dict[str, Tuple[float, float, float, float]] = {
        k: (float(v[0]), float(v[1]), float(v[2]), float(v[3])) for k, v in room_rects.items()
    }

    align_tolerance = 0.4

    anchor_x: Set[float] = {float(fminx), float(fmaxx)}
    anchor_y: Set[float] = {float(fminy), float(fmaxy)}
    for zid, (x, y, w, h) in working_rects.items():
        zt = str(zone_types.get(zid) or "")
        if ("elevator" in zt) or ("staircase" in zt):
            anchor_x.update([float(x), float(x + w)])
            anchor_y.update([float(y), float(y + h)])

    all_x: List[float] = []
    all_y: List[float] = []
    for x, y, w, h in working_rects.values():
        all_x.extend([float(x), float(x + w)])
        all_y.extend([float(y), float(y + h)])

    def _build_grid(coords: List[float], anchors: Set[float]) -> List[float]:
        if not coords:
            return sorted(float(a) for a in anchors)
        sorted_c = sorted(float(c) for c in coords)
        grid: List[float] = []
        cluster_start = float(sorted_c[0])
        current_cluster: List[float] = [float(sorted_c[0])]

        def _finalize(cluster: List[float]) -> None:
            if not cluster:
                return
            avg = float(sum(cluster) / len(cluster))
            nearest_anchor = None
            nearest_d = 1e18
            for a in anchors:
                d = abs(float(a) - avg)
                if d < nearest_d:
                    nearest_d = d
                    nearest_anchor = float(a)
            if nearest_anchor is not None and nearest_d <= float(align_tolerance):
                grid.append(float(nearest_anchor))
            else:
                grid.append(avg)

        for c in sorted_c[1:]:
            c = float(c)
            if c - cluster_start <= float(align_tolerance):
                current_cluster.append(c)
            else:
                _finalize(current_cluster)
                cluster_start = c
                current_cluster = [c]
        _finalize(current_cluster)

        for a in anchors:
            if not any(abs(float(g) - float(a)) < 1e-4 for g in grid):
                grid.append(float(a))
        grid = sorted(grid)
        return grid

    def _snap_to_grid(val: float, grid: List[float]) -> float:
        if not grid:
            return float(val)
        nearest = min(grid, key=lambda g: abs(float(g) - float(val)))
        if abs(float(nearest) - float(val)) <= float(align_tolerance):
            return float(nearest)
        return float(val)

    grid_x = _build_grid(all_x, anchors=anchor_x)
    grid_y = _build_grid(all_y, anchors=anchor_y)

    for zid, (x, y, w, h) in list(working_rects.items()):
        zt = str(zone_types.get(zid) or "")
        if ("elevator" in zt) or ("staircase" in zt):
            continue
        new_x1 = _snap_to_grid(float(x), grid_x)
        new_x2 = _snap_to_grid(float(x + w), grid_x)
        new_y1 = _snap_to_grid(float(y), grid_y)
        new_y2 = _snap_to_grid(float(y + h), grid_y)
        if (new_x2 - new_x1) > 0.1 and (new_y2 - new_y1) > 0.1:
            working_rects[zid] = (float(new_x1), float(new_y1), float(new_x2 - new_x1), float(new_y2 - new_y1))

    should_rollback = False
    for zid, (nx, ny, nw, nh) in working_rects.items():
        o = orig_room_rects.get(zid)
        if o is None:
            should_rollback = True
            break
        ox, oy, ow, oh = (float(o[0]), float(o[1]), float(o[2]), float(o[3]))
        if float(nw) <= 0.1 or float(nh) <= 0.1:
            should_rollback = True
            break
        orig_area = float(ow) * float(oh)
        new_area = float(nw) * float(nh)
        if orig_area <= 1e-9:
            should_rollback = True
            break
        if abs(float(new_area) - float(orig_area)) / float(orig_area) > 0.15:
            should_rollback = True
            break
        max_edge_move = max(
            abs(float(nx) - float(ox)),
            abs(float(ny) - float(oy)),
            abs(float(nx + nw) - float(ox + ow)),
            abs(float(ny + nh) - float(oy + oh)),
        )
        if float(max_edge_move) > float(align_tolerance):
            should_rollback = True
            break

    if should_rollback:
        room_rects = {k: (float(v[0]), float(v[1]), float(v[2]), float(v[3])) for k, v in orig_room_rects.items()}
    else:
        room_rects = working_rects
    edge_set_aug: Dict[FrozenSet[str], str] = dict(edge_set)
    forced_edges: Set[FrozenSet[str]] = set()

    def _augment_edge_set_from_rects(
        rects: Dict[str, Tuple[float, float, float, float]],
        edges: Dict[FrozenSet[str], str],
        tol: float,
    ) -> None:
        ids = list(rects.keys())
        for i in range(len(ids)):
            ida = ids[i]
            ax, ay, aw, ah = rects[ida]
            for j in range(i + 1, len(ids)):
                idb = ids[j]
                key = frozenset({ida, idb})
                if key in edges:
                    continue
                bx, by, bw, bh = rects[idb]
                overlap_y = min(ay + ah, by + bh) - max(ay, by)
                overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
                near_vertical = min(
                    abs((ax + aw) - bx),
                    abs(ax - (bx + bw)),
                ) <= tol
                near_horizontal = min(
                    abs((ay + ah) - by),
                    abs(ay - (by + bh)),
                ) <= tol
                if near_vertical and overlap_y > min_wall_length:
                    edges[key] = "vertical"
                elif near_horizontal and overlap_x > min_wall_length:
                    edges[key] = "horizontal"

    _augment_edge_set_from_rects(room_rects, edge_set_aug, tol=snap_tolerance)

    def _force_core_edges(
        rects: Dict[str, Tuple[float, float, float, float]],
        edges: Dict[FrozenSet[str], str],
        tol: float,
    ) -> None:
        def _zt(zid: str) -> str:
            return str(zone_types.get(zid) or "")

        def _is_room(zt: str) -> bool:
            return zt == "room"

        def _is_corridor(zt: str) -> bool:
            return zt == "corridor"

        def _is_elevator_any(zt: str) -> bool:
            return zt in ("elevator_hall", "elevator_shaft")

        def _is_stair_any(zt: str) -> bool:
            return zt in ("staircase_hall", "staircase_shaft")

        ids = list(rects.keys())
        for i in range(len(ids)):
            ida = ids[i]
            ta = _zt(ida)
            ax, ay, aw, ah = rects[ida]
            for j in range(i + 1, len(ids)):
                idb = ids[j]
                tb = _zt(idb)
                bx, by, bw, bh = rects[idb]

                if not ta or not tb:
                    continue

                if (ta == "elevator_hall" and tb == "elevator_shaft") or (ta == "elevator_shaft" and tb == "elevator_hall"):
                    continue
                if (ta == "staircase_hall" and tb == "staircase_shaft") or (ta == "staircase_shaft" and tb == "staircase_hall"):
                    continue

                must = False
                if (_is_elevator_any(ta) and _is_room(tb)) or (_is_elevator_any(tb) and _is_room(ta)):
                    must = True
                elif (_is_stair_any(ta) and _is_room(tb)) or (_is_stair_any(tb) and _is_room(ta)):
                    must = True
                elif (ta == "elevator_hall" and tb == "staircase_hall") or (tb == "elevator_hall" and ta == "staircase_hall"):
                    must = True
                elif ((ta in ("elevator_hall", "staircase_hall") and _is_corridor(tb)) or (tb in ("elevator_hall", "staircase_hall") and _is_corridor(ta))):
                    must = True
                elif (ta == "staircase_shaft" and tb == "elevator_hall") or (tb == "staircase_shaft" and ta == "elevator_hall"):
                    must = True
                if not must:
                    continue

                key = frozenset({ida, idb})
                forced_edges.add(key)

                overlap_y = min(ay + ah, by + bh) - max(ay, by)
                overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
                near_vertical = min(
                    abs((ax + aw) - bx),
                    abs(ax - (bx + bw)),
                ) <= tol
                near_horizontal = min(
                    abs((ay + ah) - by),
                    abs(ay - (by + bh)),
                ) <= tol
                if near_vertical and overlap_y > 0.05:
                    edges[key] = "vertical"
                elif near_horizontal and overlap_x > 0.05:
                    edges[key] = "horizontal"

    _force_core_edges(room_rects, edge_set_aug, tol=snap_tolerance)

    def _clip_line_to_floor(line: LineString) -> LineString:
        try:
            clipped = line.intersection(floor_poly)
        except Exception:
            return line
        candidates = _extract_linestrings(clipped)
        if not candidates:
            return line
        return max(candidates, key=lambda s: s.length)

    for edge_key, orientation in edge_set_aug.items():
        id_a, id_b = tuple(edge_key)
        if id_a not in room_rects or id_b not in room_rects:
            continue
        t1 = zone_types.get(id_a)
        t2 = zone_types.get(id_b)
        force_keep = edge_key in forced_edges
        
        # 物理抹除：如果一边是电梯井，一边是电梯厅，跳过，不生成任何墙体
        if (t1 == "elevator_shaft" and t2 == "elevator_hall") or \
           (t1 == "elevator_hall" and t2 == "elevator_shaft"):
            continue

        # 楼梯井与楼梯厅之间不应有墙体（保持连续平地）
        if (t1 == "staircase_shaft" and t2 == "staircase_hall") or \
           (t1 == "staircase_hall" and t2 == "staircase_shaft"):
            continue
        ax, ay, aw, ah = room_rects[id_a]
        bx, by, bw, bh = room_rects[id_b]

        if orientation == "vertical":
            dist1 = abs((ax + aw) - bx)
            dist2 = abs(ax - (bx + bw))
            if dist1 < dist2:
                shared_x = ((ax + aw) + bx) / 2.0
            else:
                shared_x = (ax + (bx + bw)) / 2.0
            y0 = max(ay, by)
            y1 = min(ay + ah, by + bh)

            min_len = 0.05 if (("corridor" in str(t1 or "")) or ("corridor" in str(t2 or ""))) else float(min_wall_length)
            if (y1 - y0) > min_len:
                if not force_keep and (("shaft" in str(t1 or "")) or ("shaft" in str(t2 or ""))):
                    if abs(float(shared_x) - float(fminx)) <= 0.05 or abs(float(shared_x) - float(fmaxx)) <= 0.05:
                        continue
                line = LineString([(shared_x, y0), (shared_x, y1)])
                line = _clip_line_to_floor(line)
                walls.append(WallSegment(
                    type="partition_wall",
                    geometry=line,
                    thickness=wall_thickness,
                    room_ids=[id_a, id_b],
                ))
        else:
            dist1 = abs((ay + ah) - by)
            dist2 = abs(ay - (by + bh))
            if dist1 < dist2:
                shared_y = ((ay + ah) + by) / 2.0
            else:
                shared_y = (ay + (by + bh)) / 2.0
            x0 = max(ax, bx)
            x1 = min(ax + aw, bx + bw)

            min_len = 0.05 if (("corridor" in str(t1 or "")) or ("corridor" in str(t2 or ""))) else float(min_wall_length)
            if (x1 - x0) > min_len:
                if not force_keep and (("shaft" in str(t1 or "")) or ("shaft" in str(t2 or ""))):
                    if abs(float(shared_y) - float(fminy)) <= 0.05 or abs(float(shared_y) - float(fmaxy)) <= 0.05:
                        continue
                line = LineString([(x0, shared_y), (x1, shared_y)])
                line = _clip_line_to_floor(line)
                walls.append(WallSegment(
                    type="partition_wall",
                    geometry=line,
                    thickness=wall_thickness,
                    room_ids=[id_a, id_b],
                ))

    inner_ymax = fmaxy - float(exterior_thickness)
    neighbor_tol = 0.06
    core_zone_ids = [
        zid for zid, zt in zone_types.items()
        if zt in ("staircase", "elevator_hall", "elevator_shaft") and zid in room_rects
    ]

    def _has_neighbor_below(top_y: float, x0: float, x1: float, self_id: str) -> bool:
        for oid, (ox, oy, ow, _) in room_rects.items():
            if oid == self_id:
                continue
            if abs(float(oy) - float(top_y)) > neighbor_tol:
                continue
            overlap = min(float(x1), float(ox + ow)) - max(float(x0), float(ox))
            if overlap > min_wall_length:
                return True
        return False

    for zid in core_zone_ids:
        x, y, w, h = room_rects[zid]
        top_y = float(y + h)
        if top_y > (inner_ymax - 0.02):
            continue
        zt0 = str(zone_types.get(zid) or "")
        gap = max(0.0, inner_ymax - top_y)
        if "shaft" in zt0 and gap <= 0.25:
            continue
        if _has_neighbor_below(top_y=top_y, x0=float(x), x1=float(x + w), self_id=zid):
            continue
        if float(w) <= min_wall_length:
            continue
        safe_thickness = min(float(wall_thickness), max(0.02, gap - 0.01))
        line = LineString([(float(x), top_y), (float(x + w), top_y)])
        line = _clip_line_to_floor(line)
        walls.append(WallSegment(
            type="partition_wall",
            geometry=line,
            thickness=safe_thickness,
            room_ids=[zid],
        ))

    walls = _apply_wall_graph(
        walls,
        floor_boundary=floor_poly,
        wall_thickness=wall_thickness,
        exterior_thickness=exterior_thickness,
        min_wall_length=min_wall_length,
    )
    walls.extend(_generate_exterior_wall_pieces(floor_poly, exterior_thickness))

    return _dedup_walls(walls)


# ============================================================
# 全局墙网生成（替代旧 generate_walls）
# ============================================================

def _extract_linestrings(geom) -> List[LineString]:
    """从几何对象提取所有 LineString，拆成逐段 2 点直线段。

    多点 LineString（如 L 形折线）会被拆成独立直线段，
    避免 buffer 后产生 L 形厚多边形（Y 字灰块的根因）。
    """
    if geom.is_empty:
        return []
    if isinstance(geom, (LineString, LinearRing)):
        coords = list(geom.coords)
        if len(coords) <= 2:
            line = LineString(coords)
            return [line] if line.length > 0.05 else []
        # 多点 LineString → 逐段 2 点直线段
        segments: List[LineString] = []
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            if seg.length > 0.05:  # 5cm 容差（snap_to_grid 偏移约 5cm）
                segments.append(seg)
        return segments
    if isinstance(geom, MultiLineString):
        result: List[LineString] = []
        for line in geom.geoms:
            result.extend(_extract_linestrings(line))
        return result
    if isinstance(geom, GeometryCollection) or hasattr(geom, "geoms"):
        result = []
        for g in geom.geoms:
            result.extend(_extract_linestrings(g))
        return result
    return []


def merge_collinear_segments(lines: List[LineString], *, tol: float = 0.02) -> List[LineString]:
    if not lines:
        return []

    horizontals: Dict[float, List[Tuple[float, float]]] = {}
    verticals: Dict[float, List[Tuple[float, float]]] = {}
    others: List[LineString] = []

    def _key(v: float) -> float:
        return round(float(v) / float(tol)) * float(tol)

    for ln in lines:
        try:
            if ln.is_empty:
                continue
            coords = list(ln.coords)
            if len(coords) < 2:
                continue
            (x0, y0), (x1, y1) = coords[0], coords[-1]
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        except Exception:
            continue

        if abs(y1 - y0) <= tol and abs(x1 - x0) > tol:
            yk = _key((y0 + y1) / 2.0)
            a, b = (x0, x1) if x0 <= x1 else (x1, x0)
            horizontals.setdefault(yk, []).append((a, b))
        elif abs(x1 - x0) <= tol and abs(y1 - y0) > tol:
            xk = _key((x0 + x1) / 2.0)
            a, b = (y0, y1) if y0 <= y1 else (y1, y0)
            verticals.setdefault(xk, []).append((a, b))
        else:
            others.append(ln)

    merged: List[LineString] = []

    for yk, intervals in horizontals.items():
        intervals.sort(key=lambda t: (float(t[0]), float(t[1])))
        cur0, cur1 = intervals[0]
        for a, b in intervals[1:]:
            if float(a) <= float(cur1) + float(tol):
                cur1 = max(float(cur1), float(b))
            else:
                merged.append(LineString([(float(cur0), float(yk)), (float(cur1), float(yk))]))
                cur0, cur1 = a, b
        merged.append(LineString([(float(cur0), float(yk)), (float(cur1), float(yk))]))

    for xk, intervals in verticals.items():
        intervals.sort(key=lambda t: (float(t[0]), float(t[1])))
        cur0, cur1 = intervals[0]
        for a, b in intervals[1:]:
            if float(a) <= float(cur1) + float(tol):
                cur1 = max(float(cur1), float(b))
            else:
                merged.append(LineString([(float(xk), float(cur0)), (float(xk), float(cur1))]))
                cur0, cur1 = a, b
        merged.append(LineString([(float(xk), float(cur0)), (float(xk), float(cur1))]))

    return merged + others


HSeg = Tuple[float, float, float, int]
VSeg = Tuple[float, float, float, int]


def _axis_aligned_segments_from_polygon(poly: Polygon, *, tol: float = 1e-6) -> Tuple[List[HSeg], List[VSeg]]:
    p = orient(poly, sign=1.0)

    h: List[HSeg] = []
    v: List[VSeg] = []

    rings: List[LinearRing] = []
    try:
        rings.append(p.exterior)
        rings.extend(list(p.interiors))
    except Exception:
        return (h, v)

    for ring in rings:
        try:
            coords = list(ring.coords)
            if len(coords) < 2:
                continue
            ring_ccw = bool(LinearRing(coords).is_ccw)
        except Exception:
            continue

        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
            dx = x1 - x0
            dy = y1 - y0
            if abs(dy) <= float(tol) and abs(dx) > float(tol):
                left_side = 1 if dx > 0.0 else -1
                interior_side = left_side if ring_ccw else -left_side
                sign = 1 if interior_side > 0 else -1
                xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
                h.append((float(y0), float(xa), float(xb), int(sign)))
            elif abs(dx) <= float(tol) and abs(dy) > float(tol):
                left_side = -1 if dy > 0.0 else 1
                interior_side = left_side if ring_ccw else -left_side
                sign = 1 if interior_side > 0 else -1
                ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
                v.append((float(x0), float(ya), float(yb), int(sign)))

    return (h, v)


def _extend_line(line: LineString, extension_dist: float) -> LineString:
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    p0 = np.array(coords[0], dtype=float)
    p1 = np.array(coords[-1], dtype=float)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        return line
    unit_dir = v / length
    new_p0 = p0 - unit_dir * float(extension_dist)
    new_p1 = p1 + unit_dir * float(extension_dist)
    return LineString([(float(new_p0[0]), float(new_p0[1])), (float(new_p1[0]), float(new_p1[1]))])


def _clean_geom(geom: BaseGeometry) -> BaseGeometry:
    try:
        return geom.buffer(0)
    except Exception:
        return geom


def _extract_polygons(geom: BaseGeometry, min_area: float = 1e-4) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > min_area else []
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if p.area > min_area]
    geoms = getattr(geom, "geoms", None)
    if geoms is not None:
        out: List[Polygon] = []
        for g in geoms:
            out.extend(_extract_polygons(g, min_area=min_area))
        return out
    return []


def _generate_exterior_wall_pieces(floor_poly: Polygon, thickness: float) -> List[WallSegment]:
    if thickness <= 0:
        return []
    outer = floor_poly
    try:
        inner = outer.buffer(-thickness, join_style=JOIN_STYLE.mitre)
    except Exception:
        inner = Polygon()
    try:
        ring = outer if inner.is_empty else outer.difference(inner)
    except Exception:
        ring = outer
    ring = _clean_geom(ring)

    minx, miny, maxx, maxy = outer.bounds
    t = float(thickness)
    strips = [
        box(minx, maxy - t, maxx, maxy),
        box(minx, miny, maxx, miny + t),
        box(minx, miny + t, minx + t, maxy - t),
        box(maxx - t, miny + t, maxx, maxy - t),
    ]

    out_walls: List[WallSegment] = []
    for strip in strips:
        try:
            piece = ring.intersection(strip)
        except Exception:
            continue
        piece = _clean_geom(piece)
        for poly in _extract_polygons(piece, min_area=1e-4):
            out_walls.append(WallSegment(
                type="exterior_wall",
                geometry=poly,
                thickness=thickness,
                room_ids=["__exterior__"],
            ))
    return out_walls


def _apply_wall_graph(
    walls: List[WallSegment],
    *,
    floor_boundary: Polygon,
    wall_thickness: float,
    exterior_thickness: float,
    min_wall_length: float,
    node_tol: float = 0.03,
) -> List[WallSegment]:
    if floor_boundary is None or floor_boundary.is_empty:
        return walls

    fminx, fminy, fmaxx, fmaxy = (float(v) for v in floor_boundary.bounds)
    line_items: List[Dict[str, Any]] = []
    preserved: List[WallSegment] = []

    def _axis_line(line: LineString) -> Optional[Tuple[str, float, float, float]]:
        coords = list(line.coords)
        if len(coords) < 2:
            return None
        x0, y0 = float(coords[0][0]), float(coords[0][1])
        x1, y1 = float(coords[-1][0]), float(coords[-1][1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dy <= node_tol and dx > node_tol:
            a, b = (x0, x1) if x0 <= x1 else (x1, x0)
            return ("h", (y0 + y1) / 2.0, float(a), float(b))
        if dx <= node_tol and dy > node_tol:
            a, b = (y0, y1) if y0 <= y1 else (y1, y0)
            return ("v", (x0 + x1) / 2.0, float(a), float(b))
        return None

    for wall in walls:
        if (
            wall.type == "partition_wall"
            and isinstance(wall.geometry, LineString)
            and getattr(wall, "category", None) != "wall_junction"
        ):
            axis = _axis_line(wall.geometry)
            if axis is None:
                preserved.append(wall)
                continue
            orient, const, a0, a1 = axis
            if a1 - a0 < 0.02:
                continue
            line_items.append({
                "orient": orient,
                "const": float(const),
                "a0": float(a0),
                "a1": float(a1),
                "wall": wall,
            })
        else:
            preserved.append(wall)

    if not line_items:
        return walls

    def _node_key(x: float, y: float) -> Tuple[int, int]:
        return (int(round(float(x) / float(node_tol))), int(round(float(y) / float(node_tol))))

    def _point_from_axis(item: Dict[str, Any], a: float) -> Tuple[float, float]:
        if item["orient"] == "h":
            return (float(a), float(item["const"]))
        return (float(item["const"]), float(a))

    points_by_item: List[List[Tuple[float, float]]] = []
    for item in line_items:
        p0 = _point_from_axis(item, item["a0"])
        p1 = _point_from_axis(item, item["a1"])
        points_by_item.append([p0, p1])

    for i, a in enumerate(line_items):
        for j in range(i + 1, len(line_items)):
            b = line_items[j]
            if a["orient"] == b["orient"]:
                continue
            h = a if a["orient"] == "h" else b
            v = a if a["orient"] == "v" else b
            x = float(v["const"])
            y = float(h["const"])
            if (
                x >= float(h["a0"]) - node_tol
                and x <= float(h["a1"]) + node_tol
                and y >= float(v["a0"]) - node_tol
                and y <= float(v["a1"]) + node_tol
            ):
                p = (float(x), float(y))
                points_by_item[i].append(p)
                points_by_item[j].append(p)

    node_acc: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for pts in points_by_item:
        for x, y in pts:
            node_acc.setdefault(_node_key(x, y), []).append((float(x), float(y)))

    node_pos: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for key, pts in node_acc.items():
        node_pos[key] = (
            round(float(sum(p[0] for p in pts) / len(pts)), 4),
            round(float(sum(p[1] for p in pts) / len(pts)), 4),
        )

    def _snap_node(x: float, y: float) -> Tuple[float, float]:
        return node_pos.get(_node_key(x, y), (round(float(x), 4), round(float(y), 4)))

    raw_segments: List[WallSegment] = []
    incident: Dict[Tuple[float, float], List[WallSegment]] = {}

    def _append_incident(pt: Tuple[float, float], wall: WallSegment) -> None:
        incident.setdefault((round(pt[0], 4), round(pt[1], 4)), []).append(wall)

    for item, pts0 in zip(line_items, points_by_item):
        orient = str(item["orient"])
        unique: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for x, y in pts0:
            sx, sy = _snap_node(x, y)
            unique[_node_key(sx, sy)] = (float(sx), float(sy))
        pts = list(unique.values())
        if orient == "h":
            pts.sort(key=lambda p: float(p[0]))
        else:
            pts.sort(key=lambda p: float(p[1]))
        parent: WallSegment = item["wall"]
        for p0, p1 in zip(pts, pts[1:]):
            length = abs(float(p1[0] - p0[0])) + abs(float(p1[1] - p0[1]))
            if length < 0.02:
                continue
            line = LineString([p0, p1])
            seg = WallSegment(
                type="partition_wall",
                geometry=line,
                thickness=float(parent.thickness),
                room_ids=list(parent.room_ids),
                forward=parent.forward,
                category=getattr(parent, "category", None),
                graph=True,
            )
            raw_segments.append(seg)
            _append_incident(p0, seg)
            _append_incident(p1, seg)

    if not raw_segments:
        return preserved

    def _nearest_side_distance(x: float, y: float) -> Tuple[str, float]:
        candidates = [
            ("left", abs(float(x) - fminx)),
            ("right", abs(fmaxx - float(x))),
            ("bottom", abs(float(y) - fminy)),
            ("top", abs(fmaxy - float(y))),
        ]
        return min(candidates, key=lambda t: float(t[1]))

    def _extend_degree_one_endpoint(seg: WallSegment) -> WallSegment:
        coords = list(seg.geometry.coords)
        if len(coords) < 2:
            return seg
        pts = [(float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))]
        changed = False
        for idx, pt in enumerate(list(pts)):
            key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
            if len(incident.get(key, [])) != 1:
                continue
            side, dist = _nearest_side_distance(pt[0], pt[1])
            if dist > float(exterior_thickness) + 0.05:
                continue
            other = pts[1 - idx]
            if abs(float(pt[0]) - float(other[0])) <= node_tol:
                if side == "bottom":
                    pts[idx] = (pt[0], fminy)
                    changed = True
                elif side == "top":
                    pts[idx] = (pt[0], fmaxy)
                    changed = True
            elif abs(float(pt[1]) - float(other[1])) <= node_tol:
                if side == "left":
                    pts[idx] = (fminx, pt[1])
                    changed = True
                elif side == "right":
                    pts[idx] = (fmaxx, pt[1])
                    changed = True
        if not changed:
            return seg
        line = LineString(pts)
        try:
            clipped = line.intersection(floor_boundary)
            candidates = _extract_linestrings(clipped)
            if candidates:
                line = max(candidates, key=lambda s: s.length)
        except Exception:
            pass
        return WallSegment(
            type=seg.type,
            geometry=line,
            thickness=seg.thickness,
            room_ids=list(seg.room_ids),
            forward=seg.forward,
            category=seg.category,
            graph=True,
        )

    line_segments = [_extend_degree_one_endpoint(seg) for seg in raw_segments]

    junctions: List[WallSegment] = []
    half = max(0.01, float(wall_thickness) / 2.0)
    for node, segs in incident.items():
        if len(segs) < 2:
            continue
        x, y = float(node[0]), float(node[1])
        patch = box(x - half, y - half, x + half, y + half)
        try:
            patch = patch.intersection(floor_boundary)
        except Exception:
            continue
        if patch.is_empty:
            continue
        ids = sorted({str(rid) for s in segs for rid in (s.room_ids or [])})
        junctions.append(WallSegment(
            type="partition_wall",
            geometry=patch,
            thickness=float(wall_thickness),
            room_ids=ids,
            category="wall_junction",
            graph=True,
        ))

    return _dedup_walls(line_segments) + junctions + preserved


def generate_wall_mesh(
    rooms: list,
    corridors: Optional[list] = None,
    core_tube=None,
    floor_boundary: Optional[Polygon] = None,
    wall_thickness: float = 0.12,
    exterior_thickness: float = 0.24,
    boundary_tolerance: float = 0.002,
    min_wall_length: float = 0.3,
) -> List[WallSegment]:
    """
    全局墙网生成（替代旧 generate_walls）。

    所有区域（房间+走廊+核心筒）一起参与边界提取。
    用 boundary buffer 容差求交，直接提取正交 LineString，禁用 MRR。

    Args:
        rooms: RoomResult 列表
        corridors: Corridor 列表
        core_tube: CoreTube 对象
        floor_boundary: 楼层外轮廓
        wall_thickness: 隔墙厚度 (m)
        exterior_thickness: 外墙厚度 (m)
        boundary_tolerance: 浮点容差 (m)
        min_wall_length: 最小墙段长度 (m)
    """
    if floor_boundary is None:
        return []

    snapped_floor = safe_snap_polygon(floor_boundary, 0.05)
    if snapped_floor is not None and (not snapped_floor.is_empty):
        floor_boundary = snapped_floor

    # 收集所有区域
    all_zones = []
    for room in rooms:
        rid = getattr(room, "id", getattr(room, "room_id", "?"))
        rtype = str(getattr(room, "room_type", getattr(room, "type", "")) or "").lower()
        if hasattr(room, "polygon") and not room.polygon.is_empty:
            p = safe_snap_polygon(room.polygon, 0.05)
            if p is not None and (not p.is_empty):
                kind = "void" if rtype == "void" or bool(getattr(room, "skip_solver", False)) else "room"
                all_zones.append((kind, rid, p))
    for corridor in (corridors or []):
        if hasattr(corridor, "polygon") and not corridor.polygon.is_empty:
            p = safe_snap_polygon(corridor.polygon, 0.05)
            if p is not None and (not p.is_empty):
                all_zones.append(("corridor", corridor.id, p))
    if core_tube and hasattr(core_tube, "polygon") and not core_tube.polygon.is_empty:
        subzones = [
            ("core", "core_staircase_hall", getattr(core_tube, "staircase_hall", None)),
            ("core", "core_staircase_hall_b", getattr(core_tube, "staircase_hall_b", None)),
            ("core", "core_staircase_shaft", getattr(core_tube, "staircase_shaft", None)),
            ("core", "core_elevator_hall", getattr(core_tube, "elevator_hall", None)),
            ("core", "core_elevator_hall_b", getattr(core_tube, "elevator_hall_b", None)),
            ("core", "core_elevator_shaft", getattr(core_tube, "elevator_shaft", None)),
        ]
        used = False
        for _, zid, poly in subzones:
            if poly is None or getattr(poly, "is_empty", True):
                continue
            p = safe_snap_polygon(poly, 0.05)
            if p is None or p.is_empty:
                continue
            all_zones.append(("core", str(zid), p))
            used = True
        if not used:
            p = safe_snap_polygon(core_tube.polygon, 0.05)
            if p is not None and (not p.is_empty):
                all_zones.append(("core", "core_tube", p))

    def _shared_edge_len(a: Polygon, b: Polygon) -> float:
        try:
            shared = a.boundary.intersection(b.boundary.buffer(max(0.01, float(boundary_tolerance) * 4.0)))
            return float(getattr(shared, "length", 0.0))
        except Exception:
            return 0.0

    def _fill_rate(poly: Polygon) -> float:
        try:
            mrr = poly.minimum_rotated_rectangle
            area = float(getattr(mrr, "area", 0.0))
            return float(poly.area) / area if area > 1e-9 else 0.0
        except Exception:
            return 0.0

    def _storage_id(parent_id: Optional[str], idx: int) -> str:
        if parent_id:
            return f"__storage_parent__{str(parent_id)}__{idx}"
        return f"__storage__{idx}"

    occupied_polys = [poly for _kind, _zid, poly in all_zones if isinstance(poly, Polygon) and not poly.is_empty]
    auto_void_count = 0
    auto_void_area = 0.0
    storage_count = 0
    if occupied_polys:
        try:
            inner_floor_geom: BaseGeometry = floor_boundary.buffer(
                -float(exterior_thickness),
                join_style=JOIN_STYLE.mitre,
            )
            if (
                inner_floor_geom.is_empty
                or float(getattr(inner_floor_geom, "area", 0.0)) < max(0.1, float(floor_boundary.area) * 0.10)
            ):
                inner_floor_geom = floor_boundary
        except Exception:
            inner_floor_geom = floor_boundary

        try:
            occupied = unary_union(occupied_polys)
            gap_geom = inner_floor_geom.difference(occupied)
        except Exception:
            gap_geom = GeometryCollection()

        for poly in _extract_polygons(gap_geom, min_area=0.02):
            p = safe_snap_polygon(poly, 0.05, min_area=0.02)
            if p is None or p.is_empty or float(p.area) <= 0.02:
                continue
            area = float(p.area)
            if area > 1.5:
                raise LayoutCoverageError(f"Macro auto-void remains: area={area:.2f}m2")

            candidates: List[Tuple[float, int, str, Any, Polygon]] = []
            for idx, (kind, zid, zpoly) in enumerate(all_zones):
                if str(kind) not in ("corridor", "room"):
                    continue
                if not isinstance(zpoly, Polygon) or zpoly.is_empty:
                    continue
                shared_len = _shared_edge_len(p, zpoly)
                if shared_len > max(0.05, float(min_wall_length) * 0.25):
                    candidates.append((shared_len, idx, str(kind), zid, zpoly))
            candidates.sort(key=lambda item: (float(item[0]), 1 if item[2] == "corridor" else 0), reverse=True)

            merged = False
            if candidates:
                corridor_candidates = [c for c in candidates if c[2] == "corridor"]
                if area <= 0.2 and corridor_candidates:
                    _shared, idx, kind, zid, zpoly = corridor_candidates[0]
                    all_zones[idx] = (kind, zid, safe_snap_polygon(zpoly.union(p), 0.05) or zpoly.union(p))
                    merged = True
                elif area <= 0.2:
                    _shared, idx, kind, zid, zpoly = candidates[0]
                    raw = zpoly.union(p)
                    if kind == "room" and _fill_rate(raw) > 0.85:
                        all_zones[idx] = (kind, zid, safe_snap_polygon(raw, 0.05) or raw)
                        merged = True
                    elif kind == "corridor":
                        all_zones[idx] = (kind, zid, safe_snap_polygon(raw, 0.05) or raw)
                        merged = True
                else:
                    _shared, idx, kind, zid, zpoly = candidates[0]
                    raw = zpoly.union(p)
                    all_zones[idx] = (kind, zid, safe_snap_polygon(raw, 0.05) or raw)
                    merged = True

            if not merged:
                if area > 0.2:
                    raise LayoutCoverageError(f"Unabsorbable medium auto-void remains: area={area:.2f}m2")
                parent = str(candidates[0][3]) if candidates and candidates[0][2] == "room" else None
                sid = _storage_id(parent, storage_count)
                storage_count += 1
                all_zones.append(("storage", sid, p))
            auto_void_count += 1
            auto_void_area += area
    if auto_void_count:
        logger.info(
            "Auto-void resolved: count=%d, area=%.3fm2, storage_fallback=%d",
            auto_void_count,
            auto_void_area,
            storage_count,
        )

    zone_kind_by_id = {str(zid): str(kind) for kind, zid, _poly in all_zones}

    walls: List[WallSegment] = []
    walls.extend(_generate_exterior_wall_pieces(floor_boundary, exterior_thickness))

    axis_tol = max(float(boundary_tolerance) * 2.0, 1e-3)

    seg_index: Dict[str, Tuple[Dict[float, List[HSeg]], Dict[float, List[VSeg]]]] = {}
    seg_pool: Dict[str, Tuple[Dict[float, List[Tuple[float, float, int]]], Dict[float, List[Tuple[float, float, int]]]]] = {}

    def _bucket(val: float) -> float:
        return round(float(val) / float(axis_tol)) * float(axis_tol)

    def _index_for(poly: Polygon, zid: str) -> Tuple[Dict[float, List[HSeg]], Dict[float, List[VSeg]]]:
        cached = seg_index.get(str(zid))
        if cached is not None:
            return cached
        hs, vs = _axis_aligned_segments_from_polygon(poly, tol=axis_tol)
        hm: Dict[float, List[HSeg]] = {}
        vm: Dict[float, List[VSeg]] = {}
        pool_h: Dict[float, List[Tuple[float, float, int]]] = {}
        pool_v: Dict[float, List[Tuple[float, float, int]]] = {}
        for y, x0, x1, sign in hs:
            yk = _bucket(float(y))
            hm.setdefault(yk, []).append((float(y), float(x0), float(x1), int(sign)))
            pool_h.setdefault(yk, []).append((float(x0), float(x1), int(sign)))
        for x, y0, y1, sign in vs:
            xk = _bucket(float(x))
            vm.setdefault(xk, []).append((float(x), float(y0), float(y1), int(sign)))
            pool_v.setdefault(xk, []).append((float(y0), float(y1), int(sign)))
        seg_index[str(zid)] = (hm, vm)
        seg_pool[str(zid)] = (pool_h, pool_v)
        return (hm, vm)

    def _outward_from_h(sign: int) -> Tuple[float, float, float]:
        return (0.0, -float(sign), 0.0)

    def _outward_from_v(sign: int) -> Tuple[float, float, float]:
        return (-float(sign), 0.0, 0.0)

    def _prefer_kind(kind: str) -> bool:
        return kind in ("corridor", "core")

    def _segment_room_ids(kind_a: str, id_a: Any, kind_b: str, id_b: Any) -> List[str]:
        if str(kind_a) == "void" and str(kind_b) != "void":
            return [str(id_b), str(id_a)]
        if str(kind_b) == "void" and str(kind_a) != "void":
            return [str(id_a), str(id_b)]
        return [str(id_a), str(id_b)]

    def _consume_1d_interval(
        intervals: List[Tuple[float, float, int]],
        sub0: float,
        sub1: float,
        sign: int,
    ) -> List[Tuple[float, float, int]]:
        out: List[Tuple[float, float, int]] = []
        a0 = float(min(sub0, sub1))
        a1 = float(max(sub0, sub1))
        for x0, x1, s in intervals:
            if int(s) != int(sign):
                out.append((float(x0), float(x1), int(s)))
                continue
            b0 = float(min(x0, x1))
            b1 = float(max(x0, x1))
            if a1 <= b0 + 1e-9 or b1 <= a0 + 1e-9:
                out.append((float(b0), float(b1), int(s)))
                continue
            if b0 < a0 - 1e-9:
                out.append((float(b0), float(a0), int(s)))
            if a1 < b1 - 1e-9:
                out.append((float(a1), float(b1), int(s)))
        return out

    def consume_overlaps_h(zid: str, yk: float, x0: float, x1: float, sign: int) -> None:
        pools = seg_pool.get(str(zid))
        if pools is None:
            return
        ph, _ = pools
        items = ph.get(float(yk))
        if not items:
            return
        ph[float(yk)] = _consume_1d_interval(items, float(x0), float(x1), int(sign))

    def consume_overlaps_v(zid: str, xk: float, y0: float, y1: float, sign: int) -> None:
        pools = seg_pool.get(str(zid))
        if pools is None:
            return
        _, pv = pools
        items = pv.get(float(xk))
        if not items:
            return
        pv[float(xk)] = _consume_1d_interval(items, float(y0), float(y1), int(sign))

    blacklist_pairs = {
        frozenset({"core_elevator_hall", "core_elevator_shaft"}),
        frozenset({"core_staircase_hall", "core_staircase_shaft"}),
    }

    for _, zid, poly in all_zones:
        try:
            _index_for(poly, str(zid))
        except Exception:
            pass

    # 内墙：所有区域两两比较（1D 线段重叠为主，2D 面交集为兜底）
    for i in range(len(all_zones)):
        kind_a, id_a, poly_a = all_zones[i]
        for j in range(i + 1, len(all_zones)):
            kind_b, id_b, poly_b = all_zones[j]
            try:
                try:
                    if float(poly_a.distance(poly_b)) > float(boundary_tolerance) * 2.0 + 1e-6:
                        continue
                except Exception:
                    pass

                hm_a, vm_a = _index_for(poly_a, str(id_a))
                hm_b, vm_b = _index_for(poly_b, str(id_b))

                is_blacklisted = frozenset({str(id_a), str(id_b)}) in blacklist_pairs
                found_any = False
                raw_lines_by_forward: Dict[Tuple[float, float, float], List[LineString]] = {}

                for yk, segs_a in hm_a.items():
                    segs_b = hm_b.get(yk)
                    if not segs_b:
                        continue
                    for y_a, x0_a, x1_a, s_a in segs_a:
                        for y_b, x0_b, x1_b, s_b in segs_b:
                            if int(s_a) == int(s_b):
                                continue
                            a0 = max(float(x0_a), float(x0_b))
                            a1 = min(float(x1_a), float(x1_b))
                            if float(a1 - a0) < float(min_wall_length):
                                continue
                            consume_overlaps_h(str(id_a), float(yk), float(a0), float(a1), int(s_a))
                            consume_overlaps_h(str(id_b), float(yk), float(a0), float(a1), int(s_b))
                            if is_blacklisted:
                                found_any = True
                                continue
                            y_line = (float(y_a) + float(y_b)) / 2.0
                            line = LineString([(float(a0), float(y_line)), (float(a1), float(y_line))])
                            forward = _outward_from_h(int(s_a))
                            if _prefer_kind(str(kind_b)) and (not _prefer_kind(str(kind_a))):
                                forward = _outward_from_h(int(s_b))
                            raw_lines_by_forward.setdefault(forward, []).append(line)
                            found_any = True

                for xk, segs_a in vm_a.items():
                    segs_b = vm_b.get(xk)
                    if not segs_b:
                        continue
                    for x_a, y0_a, y1_a, s_a in segs_a:
                        for x_b, y0_b, y1_b, s_b in segs_b:
                            if int(s_a) == int(s_b):
                                continue
                            a0 = max(float(y0_a), float(y0_b))
                            a1 = min(float(y1_a), float(y1_b))
                            if float(a1 - a0) < float(min_wall_length):
                                continue
                            consume_overlaps_v(str(id_a), float(xk), float(a0), float(a1), int(s_a))
                            consume_overlaps_v(str(id_b), float(xk), float(a0), float(a1), int(s_b))
                            if is_blacklisted:
                                found_any = True
                                continue
                            x_line = (float(x_a) + float(x_b)) / 2.0
                            line = LineString([(float(x_line), float(a0)), (float(x_line), float(a1))])
                            forward = _outward_from_v(int(s_a))
                            if _prefer_kind(str(kind_b)) and (not _prefer_kind(str(kind_a))):
                                forward = _outward_from_v(int(s_b))
                            raw_lines_by_forward.setdefault(forward, []).append(line)
                            found_any = True

                if is_blacklisted:
                    continue

                if found_any:
                    for forward, lines0 in raw_lines_by_forward.items():
                        merged_lines = merge_collinear_segments(lines0, tol=0.02)
                        for line in merged_lines:
                            if line.length < float(min_wall_length):
                                continue
                            walls.append(WallSegment(
                                type="partition_wall",
                                geometry=line,
                                thickness=wall_thickness,
                                room_ids=_segment_room_ids(str(kind_a), id_a, str(kind_b), id_b),
                                forward=forward,
                            ))
                    continue

                bound_a = poly_a.boundary
                bound_b = poly_b.boundary.buffer(boundary_tolerance)
                shared = bound_a.intersection(bound_b)
                if shared.is_empty:
                    continue
                logger.info("1D extraction failed, falling back to 2D intersection for zones: %s-%s", str(id_a), str(id_b))
                lines2 = merge_collinear_segments(_extract_linestrings(shared), tol=0.02)
                for line in lines2:
                    if line.length <= min_wall_length:
                        continue
                    try:
                        clipped = line.intersection(floor_boundary)
                    except Exception:
                        clipped = line
                    candidates = _extract_linestrings(clipped)
                    if candidates:
                        line = max(candidates, key=lambda s: s.length)
                    walls.append(WallSegment(
                        type="partition_wall",
                        geometry=line,
                        thickness=wall_thickness,
                        room_ids=_segment_room_ids(str(kind_a), id_a, str(kind_b), id_b),
                    ))
            except Exception as e:
                logger.debug(f"Partition wall failed for {id_a}-{id_b}: {e}")

    try:
        pool_lines: List[LineString] = []
        for w in walls:
            g = getattr(w, "geometry", None)
            if g is None or getattr(g, "is_empty", True):
                continue
            if isinstance(g, LineString):
                pool_lines.append(g)
            else:
                try:
                    pool_lines.extend(_extract_linestrings(g.boundary))
                except Exception:
                    pass
        walls_spatial = unary_union(pool_lines) if pool_lines else GeometryCollection()
    except Exception:
        walls_spatial = GeometryCollection()
    try:
        walls_buffer = walls_spatial.buffer(
            0.02,
            cap_style=CAP_STYLE.flat,
            join_style=JOIN_STYLE.mitre,
        ) if not walls_spatial.is_empty else GeometryCollection()
    except Exception:
        walls_buffer = GeometryCollection()

    orphan_kept = 0
    orphan_skipped = 0

    def _covered_by_existing(line: LineString) -> bool:
        if walls_buffer.is_empty or line.length <= 1e-9:
            return False
        try:
            covered = line.intersection(walls_buffer)
            coverage_ratio = float(getattr(covered, "length", 0.0)) / float(line.length)
        except Exception:
            return False
        return coverage_ratio >= 0.8

    for zid, (ph, pv) in seg_pool.items():
        if zone_kind_by_id.get(str(zid)) == "void":
            continue
        for yk, intervals in ph.items():
            for a0, a1, sign in intervals:
                if float(abs(float(a1) - float(a0))) < float(min_wall_length):
                    continue
                line = LineString([(float(min(a0, a1)), float(yk)), (float(max(a0, a1)), float(yk))])
                try:
                    if float(line.distance(floor_boundary.exterior)) < 0.05:
                        continue
                except Exception:
                    pass
                if _covered_by_existing(line):
                    orphan_skipped += 1
                    continue
                walls.append(WallSegment(
                    type="partition_wall",
                    geometry=line,
                    thickness=wall_thickness,
                    room_ids=[str(zid)],
                    forward=_outward_from_h(int(sign)),
                ))
                orphan_kept += 1
        for xk, intervals in pv.items():
            for a0, a1, sign in intervals:
                if float(abs(float(a1) - float(a0))) < float(min_wall_length):
                    continue
                line = LineString([(float(xk), float(min(a0, a1))), (float(xk), float(max(a0, a1)))])
                try:
                    if float(line.distance(floor_boundary.exterior)) < 0.05:
                        continue
                except Exception:
                    pass
                if _covered_by_existing(line):
                    orphan_skipped += 1
                    continue
                walls.append(WallSegment(
                    type="partition_wall",
                    geometry=line,
                    thickness=wall_thickness,
                    room_ids=[str(zid)],
                    forward=_outward_from_v(int(sign)),
                ))
                orphan_kept += 1

    if orphan_kept or orphan_skipped:
        logger.info("Orphan wall recovery: kept=%d, skipped_as_covered=%d", orphan_kept, orphan_skipped)

    return _dedup_walls(_apply_wall_graph(
        walls,
        floor_boundary=floor_boundary,
        wall_thickness=wall_thickness,
        exterior_thickness=exterior_thickness,
        min_wall_length=min_wall_length,
    ))


def _dedup_walls(walls: List[WallSegment]) -> List[WallSegment]:
    """按端点坐标去重，避免重复绘制导致颜色叠加变深（叠影）"""
    seen: Dict[Tuple[Any, ...], WallSegment] = {}
    result: List[WallSegment] = []
    for w in walls:
        try:
            if isinstance(w.geometry, LineString):
                coords = list(w.geometry.coords)
            else:
                result.append(w)  # 不认识的几何类型不去重，直接保留
                continue
        except Exception:
            result.append(w)
            continue

        if len(coords) < 2:
            result.append(w)
            continue

        a = (round(coords[0][0], 2), round(coords[0][1], 2))
        b = (round(coords[-1][0], 2), round(coords[-1][1], 2))
        p1, p2 = (a, b) if a <= b else (b, a)
        key = (w.type, round(float(w.thickness), 3), p1, p2)
        existing = seen.get(key)
        if existing is None:
            seen[key] = w
            result.append(w)
            continue

        existing_ids = [str(rid) for rid in (existing.room_ids or [])]
        for rid in (w.room_ids or []):
            rid_s = str(rid)
            if rid_s not in existing_ids:
                existing_ids.append(rid_s)
        existing.room_ids = existing_ids
        if existing.forward is None and w.forward is not None:
            existing.forward = w.forward
        if existing.category is None and w.category is not None:
            existing.category = w.category
        existing.graph = bool(getattr(existing, "graph", False) or getattr(w, "graph", False))
    return result


def _normalize_walls(
    walls: List[WallSegment],
    *,
    floor_bounds: Tuple[float, float, float, float],
    zone_rects: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> List[WallSegment]:
    grid_x: List[float] = [float(floor_bounds[0]), float(floor_bounds[2])]
    grid_y: List[float] = [float(floor_bounds[1]), float(floor_bounds[3])]
    if zone_rects:
        for x, y, w, h in zone_rects.values():
            grid_x.extend([float(x), float(x + w)])
            grid_y.extend([float(y), float(y + h)])
    grid_x = sorted({round(float(v), 4) for v in grid_x})
    grid_y = sorted({round(float(v), 4) for v in grid_y})

    def _snap_axis(val: float, grid: List[float], tol: float) -> float:
        if not grid:
            return float(val)
        nearest = min(grid, key=lambda g: abs(float(g) - float(val)))
        return float(nearest) if abs(float(nearest) - float(val)) <= float(tol) else float(val)

    def _axis_align(line: LineString) -> LineString:
        coords = list(line.coords)
        if len(coords) < 2:
            return line
        (x1, y1), (x2, y2) = coords[0], coords[-1]
        dx = abs(float(x2) - float(x1))
        dy = abs(float(y2) - float(y1))
        eps = 0.05
        if dx <= dy and dx < eps:
            sx = _snap_axis((float(x1) + float(x2)) / 2.0, grid_x, tol=0.12)
            return LineString([(sx, float(y1)), (sx, float(y2))])
        if dy < dx and dy < eps:
            sy = _snap_axis((float(y1) + float(y2)) / 2.0, grid_y, tol=0.12)
            return LineString([(float(x1), sy), (float(x2), sy)])
        return line

    lines: List[WallSegment] = []
    others: List[WallSegment] = []
    for w in walls:
        if isinstance(w.geometry, LineString):
            try:
                g = _axis_align(w.geometry)
                lines.append(WallSegment(
                    type=w.type,
                    geometry=g,
                    thickness=w.thickness,
                    room_ids=list(w.room_ids),
                    forward=w.forward,
                    category=w.category,
                    graph=getattr(w, "graph", False),
                ))
            except Exception:
                lines.append(w)
        else:
            others.append(w)

    if any(getattr(w, "graph", False) for w in lines + others):
        return _dedup_walls(lines + others)

    snap_tol = 0.05
    pts: List[Tuple[float, float]] = []
    for w in lines:
        coords = list(w.geometry.coords)
        if len(coords) >= 2:
            pts.append((float(coords[0][0]), float(coords[0][1])))
            pts.append((float(coords[-1][0]), float(coords[-1][1])))

    clusters: List[List[Tuple[float, float]]] = []
    for x, y in pts:
        placed = False
        for c in clusters:
            cx, cy = c[0]
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= snap_tol * snap_tol:
                c.append((x, y))
                placed = True
                break
        if not placed:
            clusters.append([(x, y)])

    mapping: Dict[Tuple[float, float], Tuple[float, float]] = {}
    for c in clusters:
        mx = sum(p[0] for p in c) / len(c)
        my = sum(p[1] for p in c) / len(c)
        mx = _snap_axis(mx, grid_x, tol=snap_tol)
        my = _snap_axis(my, grid_y, tol=snap_tol)
        mx = round(float(mx), 4)
        my = round(float(my), 4)
        for p in c:
            mapping[(round(float(p[0]), 4), round(float(p[1]), 4))] = (mx, my)

    snapped: List[WallSegment] = []
    for w in lines:
        coords = list(w.geometry.coords)
        if len(coords) < 2:
            snapped.append(w)
            continue
        a = (round(float(coords[0][0]), 4), round(float(coords[0][1]), 4))
        b = (round(float(coords[-1][0]), 4), round(float(coords[-1][1]), 4))
        a2 = mapping.get(a, a)
        b2 = mapping.get(b, b)
        g = LineString([a2, b2])
        snapped.append(WallSegment(
            type=w.type,
            geometry=g,
            thickness=w.thickness,
            room_ids=list(w.room_ids),
            forward=w.forward,
            category=w.category,
            graph=getattr(w, "graph", False),
        ))

    merged: List[WallSegment] = []
    buckets: Dict[Tuple, List[WallSegment]] = {}
    for w in snapped:
        if not isinstance(w.geometry, LineString):
            merged.append(w)
            continue
        coords = list(w.geometry.coords)
        if len(coords) < 2:
            merged.append(w)
            continue
        (x1, y1), (x2, y2) = coords[0], coords[-1]
        dx = abs(float(x2) - float(x1))
        dy = abs(float(y2) - float(y1))
        if dx <= dy:
            key = (w.type, round(float(w.thickness), 3), frozenset(w.room_ids), "v", round(float(x1), 4))
        else:
            key = (w.type, round(float(w.thickness), 3), frozenset(w.room_ids), "h", round(float(y1), 4))
        buckets.setdefault(key, []).append(w)

    for key, segs in buckets.items():
        _, thick, room_ids, orient, cst = key
        if orient == "v":
            intervals = []
            for w in segs:
                y0 = float(min(w.geometry.coords[0][1], w.geometry.coords[-1][1]))
                y1 = float(max(w.geometry.coords[0][1], w.geometry.coords[-1][1]))
                intervals.append((y0, y1))
            intervals.sort()
            out = []
            cur0, cur1 = intervals[0]
            for a0, a1 in intervals[1:]:
                if a0 <= cur1 + 1e-3:
                    cur1 = max(cur1, a1)
                else:
                    out.append((cur0, cur1))
                    cur0, cur1 = a0, a1
            out.append((cur0, cur1))
            for a0, a1 in out:
                g = LineString([(float(cst), float(a0)), (float(cst), float(a1))])
                if g.length > 0.02:
                    merged.append(WallSegment(type=key[0], geometry=g, thickness=float(thick), room_ids=list(room_ids)))
        else:
            intervals = []
            for w in segs:
                x0 = float(min(w.geometry.coords[0][0], w.geometry.coords[-1][0]))
                x1 = float(max(w.geometry.coords[0][0], w.geometry.coords[-1][0]))
                intervals.append((x0, x1))
            intervals.sort()
            out = []
            cur0, cur1 = intervals[0]
            for a0, a1 in intervals[1:]:
                if a0 <= cur1 + 1e-3:
                    cur1 = max(cur1, a1)
                else:
                    out.append((cur0, cur1))
                    cur0, cur1 = a0, a1
            out.append((cur0, cur1))
            for a0, a1 in out:
                g = LineString([(float(a0), float(cst)), (float(a1), float(cst))])
                if g.length > 0.02:
                    merged.append(WallSegment(type=key[0], geometry=g, thickness=float(thick), room_ids=list(room_ids)))

    merged = _dedup_walls(merged) + others

    degrees: Dict[Tuple[float, float], int] = {}
    for w in merged:
        if not isinstance(w.geometry, LineString) or w.type != "partition_wall":
            continue
        coords = list(w.geometry.coords)
        if len(coords) < 2:
            continue
        a = (round(float(coords[0][0]), 4), round(float(coords[0][1]), 4))
        b = (round(float(coords[-1][0]), 4), round(float(coords[-1][1]), 4))
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1

    pruned: List[WallSegment] = []
    for w in merged:
        if not isinstance(w.geometry, LineString) or w.type != "partition_wall":
            pruned.append(w)
            continue
        coords = list(w.geometry.coords)
        if len(coords) < 2:
            pruned.append(w)
            continue
        a = (round(float(coords[0][0]), 4), round(float(coords[0][1]), 4))
        b = (round(float(coords[-1][0]), 4), round(float(coords[-1][1]), 4))
        if degrees.get(a, 0) <= 1 and degrees.get(b, 0) <= 1 and float(w.geometry.length) < 0.2:
            continue
        pruned.append(w)

    final_walls = _dedup_walls(pruned)
    node_count = 0
    edge_count = 0
    junction_count = 0
    for wall in final_walls:
        if getattr(wall, "category", None) == "wall_junction":
            junction_count += 1
        if isinstance(wall.geometry, LineString):
            edge_count += 1
            coords = list(wall.geometry.coords)
            node_count += min(2, len(coords))
    logger.info(
        "[GRAPH] Wall Graph generated | walls=%d | line_edges=%d | approx_nodes=%d | junctions=%d",
        len(final_walls),
        edge_count,
        node_count,
        junction_count,
    )
    return final_walls


# ============================================================
# 墙体方向判断
# ============================================================

def _wall_rotation(wall: WallSegment) -> float:
    """判断墙体方向：水平=0, 垂直=90（用 bounds 而非首两个 coords）"""
    minx, miny, maxx, maxy = wall.geometry.bounds
    dx = maxx - minx
    dy = maxy - miny
    return 0.0 if dx >= dy else 90.0


def _normalize_2d(dx: float, dy: float) -> Tuple[float, float]:
    n = float((dx * dx + dy * dy) ** 0.5)
    if n <= 1e-9:
        return (0.0, 1.0)
    return (dx / n, dy / n)


def storage_parent_id(zid: str) -> Optional[str]:
    z = str(zid)
    prefix = "__storage_parent__"
    if not z.startswith(prefix):
        return None
    rest = z[len(prefix):]
    if "__" not in rest:
        return None
    return rest.rsplit("__", 1)[0] or None


def normalize_room_meta_type(raw: Any, zid: Optional[str] = None) -> str:
    text = f"{raw or ''} {zid or ''}".strip().lower()
    if not text:
        return "room"
    if (
        text.startswith("__storage")
        or text.startswith("__auto_void")
        or "storage" in text
        or "utility" in text
        or "closet" in text
        or "wardrobe" in text
        or "衣帽" in text
        or "储藏" in text
        or "设备" in text
    ):
        return "storage"
    if "void" in text or "unassigned" in text:
        return "storage"
    if "corridor" in text or "hallway" in text or "passage" in text or "走廊" in text:
        return "corridor"
    if "lobby" in text or "entrance" in text or "门厅" in text:
        return "hall"
    if "living" in text or "客厅" in text:
        return "living"
    if "dining" in text or "餐厅" in text:
        return "dining"
    if "bedroom" in text or "bed" in text or "主卧" in text or "次卧" in text or "卧室" in text:
        return "bedroom"
    if "bath" in text or "toilet" in text or "卫生" in text or "洗手" in text:
        return "bathroom"
    if "kitchen" in text or "厨房" in text:
        return "kitchen"
    if "staircase_hall" in text:
        return "staircase_hall"
    if "elevator_hall" in text:
        return "elevator_hall"
    if "staircase" in text:
        return "staircase"
    if "elevator_shaft" in text:
        return "elevator_shaft"
    if "core" in text or "shaft" in text:
        return "core"
    return "room"


PUBLIC_ACCESS_METAS = {"corridor", "hall", "living", "dining", "elevator_hall", "staircase_hall"}
HABITABLE_METAS = {"room", "bedroom", "bathroom", "kitchen", "living", "dining"}


def is_valid_bedroom_leaf(parent_id: str, child_id: str, zone_types: Dict[str, str], graph: Dict[str, Set[str]]) -> bool:
    child_meta = normalize_room_meta_type(zone_types.get(child_id), child_id)
    if child_meta == "storage":
        return storage_parent_id(child_id) == str(parent_id)
    if child_meta == "bathroom":
        private_tokens = ("ensuite", "master_bath", "private_bath", "套卫", "主卫")
        raw = f"{zone_types.get(child_id, '')} {child_id}".lower()
        return any(t in raw for t in private_tokens) or len(graph.get(child_id, set())) <= 1
    if child_meta == "balcony":
        raw = f"{zone_types.get(child_id, '')} {child_id}".lower()
        return "private" in raw or "私人" in raw
    return False


def build_access_graph(doors: List["DoorPlacement"], zone_types: Optional[Dict[str, str]] = None) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for door in doors:
        con = list(getattr(door, "connects", []) or [])
        if len(con) != 2:
            continue
        a, b = str(con[0]), str(con[1])
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def update_access_graph_with_door(graph: Dict[str, Set[str]], door: "DoorPlacement") -> Tuple[Optional[str], Optional[str]]:
    con = list(getattr(door, "connects", []) or [])
    if len(con) != 2:
        return (None, None)
    a, b = str(con[0]), str(con[1])
    graph.setdefault(a, set()).add(b)
    graph.setdefault(b, set()).add(a)
    return (a, b)


def _connected_component(graph: Dict[str, Set[str]], start: str) -> Set[str]:
    seen: Set[str] = set()
    queue = [str(start)]
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        for nb in graph.get(node, set()):
            if nb not in seen:
                queue.append(nb)
    return seen


def _reachability_state(
    graph: Dict[str, Set[str]],
    zone_types: Dict[str, str],
    *,
    allow_core_hall_starts: bool = True,
) -> Tuple[Dict[str, str], Set[str], Set[str], List[str]]:
    metas = {zid: normalize_room_meta_type(zt, zid) for zid, zt in zone_types.items()}
    for zid in graph:
        metas.setdefault(zid, normalize_room_meta_type(zone_types.get(zid), zid))

    start_metas = {"corridor", "hall", "living"}
    if allow_core_hall_starts:
        start_metas.update({"elevator_hall", "staircase_hall"})
    starts = {
        zid for zid, meta in metas.items()
        if meta in start_metas
    }
    if "__exterior__" in graph:
        starts.add("__exterior__")

    reachable: Set[str] = set()
    queue: List[str] = list(starts)
    while queue:
        node = queue.pop(0)
        if node in reachable:
            continue
        reachable.add(node)
        node_meta = metas.get(node, normalize_room_meta_type(None, node))
        if node_meta in ("storage", "bathroom"):
            continue
        for nb in graph.get(node, set()):
            if nb in reachable:
                continue
            nb_meta = metas.get(nb, normalize_room_meta_type(None, nb))
            if node_meta == "bedroom" and not is_valid_bedroom_leaf(node, nb, zone_types, graph):
                continue
            if node_meta not in ("corridor", "hall", "living", "elevator_hall", "staircase_hall", "core", "room", "bedroom", "kitchen", "dining", "__exterior__") and node != "__exterior__":
                continue
            queue.append(nb)

    required = [
        zid for zid, meta in metas.items()
        if meta in HABITABLE_METAS
    ]
    unreachable = sorted(zid for zid in required if zid not in reachable)
    return metas, starts, reachable, unreachable


def validate_reachability(
    doors: List["DoorPlacement"],
    zone_types: Dict[str, str],
    *,
    allow_core_hall_starts: bool = True,
) -> None:
    graph = build_access_graph(doors, zone_types)

    if not graph:
        return

    metas, starts, reachable, unreachable = _reachability_state(
        graph,
        zone_types,
        allow_core_hall_starts=allow_core_hall_starts,
    )
    if not starts:
        logger.error("[VALIDATION] Rule: REACHABILITY | Result: FAIL | Reason=no_circulation_or_exterior_start")
        raise LayoutTopologyError(
            "No circulation/exterior start node for reachability check",
            metadata={"failure_kind": "reachability", "reason": "no_circulation_or_exterior_start"},
        )

    if unreachable:
        logger.error(
            "[VALIDATION] Rule: REACHABILITY | Result: FAIL | Unreachable=%s | Starts=%s",
            unreachable,
            sorted(starts),
        )
        raise LayoutTopologyError(
            f"Unreachable habitable rooms: {', '.join(unreachable)}",
            metadata={
                "failure_kind": "reachability",
                "unreachable_rooms": list(unreachable),
                "start_nodes": sorted(starts),
            },
        )

    for sid, meta in metas.items():
        if meta == "storage" and len(graph.get(sid, set())) > 1:
            logger.error(
                "[VALIDATION] Rule: STORAGE_LEAF | Result: FAIL | Storage=%s | Degree=%d | Neighbors=%s",
                sid,
                len(graph.get(sid, set())),
                sorted(graph.get(sid, set())),
            )
            raise LayoutTopologyError(
                f"Storage node has multiple doors: {sid}",
                metadata={
                    "failure_kind": "reachability",
                    "reason": "storage_not_leaf",
                    "storage_id": sid,
                    "neighbors": sorted(graph.get(sid, set())),
                },
            )
    required_count = sum(1 for _zid, meta in metas.items() if meta in HABITABLE_METAS)
    logger.info("[VALIDATION] Rule: REACHABILITY | Result: PASS | Required=%d | Reachable=%d", required_count, len(reachable))


def _required_neighbors(required_adjacency: Optional[Dict[str, List[str]]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for rid, vals in (required_adjacency or {}).items():
        a = str(rid)
        out.setdefault(a, set())
        for val in vals or []:
            b = str(val)
            if not b:
                continue
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


def _axis_overlap_length_for_wall_polys(a: Polygon, b: Polygon, wall: WallSegment) -> float:
    if wall.geometry is None or wall.geometry.is_empty or not hasattr(wall.geometry, "coords"):
        return 0.0
    coords = list(wall.geometry.coords)
    if len(coords) < 2:
        return 0.0
    x1, y1 = float(coords[0][0]), float(coords[0][1])
    x2, y2 = float(coords[-1][0]), float(coords[-1][1])
    aminx, aminy, amaxx, amaxy = (float(v) for v in a.bounds)
    bminx, bminy, bmaxx, bmaxy = (float(v) for v in b.bounds)
    if abs(x2 - x1) >= abs(y2 - y1):
        lo = max(min(x1, x2), aminx, bminx)
        hi = min(max(x1, x2), amaxx, bmaxx)
    else:
        lo = max(min(y1, y2), aminy, bminy)
        hi = min(max(y1, y2), amaxy, bmaxy)
    return max(0.0, float(hi - lo))


def _effective_shared_wall_length_for_ids(
    wall: WallSegment,
    a_id: str,
    b_id: str,
    zone_polys: Optional[Dict[str, Polygon]],
    *,
    shared_wall_tolerance: float = 1e-4,
) -> float:
    if zone_polys is None:
        return float(wall.length)
    pa = zone_polys.get(str(a_id))
    pb = zone_polys.get(str(b_id))
    if pa is None or pb is None or pa.is_empty or pb.is_empty:
        return float(wall.length)
    try:
        shared_len = float(pa.boundary.intersection(pb.boundary).length)
    except Exception:
        shared_len = 0.0
    try:
        close_enough = (
            float(pa.boundary.distance(pb.boundary)) <= shared_wall_tolerance
            and float(wall.geometry.distance(pa.boundary)) <= shared_wall_tolerance
            and float(wall.geometry.distance(pb.boundary)) <= shared_wall_tolerance
        )
    except Exception:
        close_enough = False
    if close_enough:
        shared_len = max(shared_len, _axis_overlap_length_for_wall_polys(pa, pb, wall))
    return min(float(wall.length), float(shared_len))


def _door_projection_interval(wall: WallSegment, position: Tuple[float, float], width: float) -> Optional[Tuple[float, float]]:
    if not isinstance(wall.geometry, LineString) or wall.geometry.is_empty:
        return None
    try:
        s = float(wall.geometry.project(Point(float(position[0]), float(position[1]))))
        half = max(0.0, float(width) / 2.0)
        return (s - half, s + half)
    except Exception:
        return None


def _door_collides_on_wall(
    wall: WallSegment,
    position: Tuple[float, float],
    width: float,
    doors: List[DoorPlacement],
    *,
    clearance: float = 0.15,
) -> Optional[DoorPlacement]:
    new_span = _door_projection_interval(wall, position, width)
    if new_span is None:
        return None
    for door in doors:
        try:
            p = Point(float(door.position[0]), float(door.position[1]))
            if float(wall.geometry.distance(p)) > 0.05:
                continue
            old_span = _door_projection_interval(wall, (float(door.position[0]), float(door.position[1])), float(door.width))
            if old_span is None:
                continue
            if new_span[0] <= old_span[1] + float(clearance) and old_span[0] <= new_span[1] + float(clearance):
                return door
        except Exception:
            continue
    return None


def _candidate_door_points(wall: WallSegment, width: float, *, clearance: float = 0.05) -> List[Tuple[float, float]]:
    if not isinstance(wall.geometry, LineString) or wall.geometry.is_empty:
        return []
    length = float(wall.length)
    margin = float(width) / 2.0 + float(clearance)
    if length < 2.0 * margin:
        return []
    points: List[Tuple[float, float]] = []
    seen: Set[Tuple[float, float]] = set()
    for frac in (0.5, 0.35, 0.65, 0.2, 0.8):
        dist = min(max(length * float(frac), margin), length - margin)
        try:
            p = wall.geometry.interpolate(dist)
            key = (round(float(p.x), 4), round(float(p.y), 4))
            if key in seen:
                continue
            seen.add(key)
            points.append((float(p.x), float(p.y)))
        except Exception:
            continue
    return points


def _simple_wall_forward(wall: WallSegment) -> Tuple[float, float, float]:
    rot = _wall_rotation(wall)
    if abs(float(rot) - 90.0) < 1e-6:
        return (1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0)


def _is_suite_fallback_target(
    child_id: str,
    parent_id: str,
    zone_types: Dict[str, str],
    required_neighbors: Dict[str, Set[str]],
) -> bool:
    if str(parent_id) not in required_neighbors.get(str(child_id), set()):
        return False
    child_meta = normalize_room_meta_type(zone_types.get(str(child_id)), str(child_id))
    parent_meta = normalize_room_meta_type(zone_types.get(str(parent_id)), str(parent_id))
    if child_meta in ("bathroom", "storage", "balcony"):
        return parent_meta in ("bedroom", "room", "living")
    return False


def repair_unreachable_doors(
    doors: List[DoorPlacement],
    walls: List[WallSegment],
    zone_types: Dict[str, str],
    *,
    zone_polys: Optional[Dict[str, Polygon]] = None,
    required_adjacency: Optional[Dict[str, List[str]]] = None,
    door_width: float = 0.9,
    min_door_width: float = 0.8,
    allow_core_targets: bool = True,
    allow_core_hall_starts: bool = True,
) -> Dict[str, Any]:
    graph = build_access_graph(doors, zone_types)
    required_map = _required_neighbors(required_adjacency)
    attempts: List[Dict[str, Any]] = []
    collision_rejects: List[Dict[str, Any]] = []
    suite_candidates: List[Dict[str, Any]] = []
    added = 0

    walls_for_doors = [
        w for w in walls
        if w.type == "partition_wall"
        and isinstance(w.geometry, LineString)
        and getattr(w, "category", None) != "wall_junction"
        and len(getattr(w, "room_ids", []) or []) == 2
    ]

    max_iterations = max(1, len(zone_types) + 1)
    for _iteration in range(max_iterations):
        _metas, starts, reachable, unreachable = _reachability_state(
            graph,
            zone_types,
            allow_core_hall_starts=allow_core_hall_starts,
        )
        if not unreachable:
            break
        progress = False
        for rid in list(unreachable):
            rid = str(rid)
            rid_meta = normalize_room_meta_type(zone_types.get(rid), rid)
            candidates: List[Tuple[int, float, WallSegment, str, str]] = []
            for wall in walls_for_doors:
                ids = [str(x) for x in (wall.room_ids or [])]
                if rid not in ids:
                    continue
                other = ids[1] if ids[0] == rid else ids[0]
                if other not in reachable:
                    continue
                other_meta = normalize_room_meta_type(zone_types.get(other), other)
                reason = ""
                priority = 0
                if other_meta in PUBLIC_ACCESS_METAS:
                    if (not allow_core_targets) and other_meta in ("elevator_hall", "staircase_hall"):
                        continue
                    reason = "public_access"
                    priority = 100
                elif _is_suite_fallback_target(rid, other, zone_types, required_map):
                    reason = "suite_required_parent"
                    priority = 70
                    suite_candidates.append({"room": rid, "parent": other, "room_meta": rid_meta})
                else:
                    continue
                shared_len = _effective_shared_wall_length_for_ids(wall, rid, other, zone_polys)
                if shared_len < float(min_door_width):
                    attempts.append({
                        "room": rid,
                        "target": other,
                        "reason": reason,
                        "result": "shared_wall_too_short",
                        "shared_len": round(float(shared_len), 4),
                    })
                    continue
                candidates.append((priority, float(shared_len), wall, other, reason))
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

            for priority, shared_len, wall, other, reason in candidates:
                door_w = min(float(door_width), float(shared_len), float(wall.length))
                if door_w < float(min_door_width):
                    continue
                chosen_point: Optional[Tuple[float, float]] = None
                collided_with: Optional[DoorPlacement] = None
                for p in _candidate_door_points(wall, door_w, clearance=0.0):
                    collided_with = _door_collides_on_wall(wall, p, door_w, doors)
                    if collided_with is None:
                        chosen_point = p
                        break
                if chosen_point is None:
                    collision_rejects.append({
                        "room": rid,
                        "target": other,
                        "reason": reason,
                        "existing_connects": list(getattr(collided_with, "connects", []) or []) if collided_with else [],
                    })
                    logger.warning(
                        "[DOOR] skip_reason=door_collision | room=%s | target=%s | reason=%s",
                        rid,
                        other,
                        reason,
                    )
                    continue
                rot = _wall_rotation(wall)
                new_door = DoorPlacement(
                    position=(round(float(chosen_point[0]), 2), round(float(chosen_point[1]), 2)),
                    width=round(float(door_w), 2),
                    connects=list(wall.room_ids),
                    wall_type=wall.type,
                    rotation=rot,
                    thickness=float(wall.thickness),
                    forward=_simple_wall_forward(wall),
                )
                doors.append(new_door)
                update_access_graph_with_door(graph, new_door)
                component = _connected_component(graph, rid)
                added += 1
                progress = True
                attempts.append({
                    "room": rid,
                    "target": other,
                    "reason": reason,
                    "result": "door_added",
                    "shared_len": round(float(shared_len), 4),
                    "component_size": len(component),
                })
                logger.warning(
                    "[DOOR] Reachability fallback door generated | room=%s | target=%s | reason=%s | width=%.2fm | component_size=%d",
                    rid,
                    other,
                    reason,
                    float(door_w),
                    len(component),
                )
                break
        if not progress:
            break

    _metas, starts, reachable, unreachable = _reachability_state(
        graph,
        zone_types,
        allow_core_hall_starts=allow_core_hall_starts,
    )
    return {
        "added_doors": added,
        "unreachable_after": list(unreachable),
        "start_nodes": sorted(starts),
        "fallback_attempts": attempts[-80:],
        "suite_candidates": suite_candidates[-40:],
        "collision_rejects": collision_rejects[-40:],
    }


# ============================================================
# 门的放置
# ============================================================

def generate_doors(
    walls: List[WallSegment],
    zone_types: Optional[Dict[str, str]] = None,
    zone_rects: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    zone_polys: Optional[Dict[str, Polygon]] = None,
    door_width: float = 0.9,
    door_eps: float = 0.2,
) -> List[DoorPlacement]:
    """
    在内墙上放置门。

    规则：每对相邻房间的共享内墙中点放一扇门。

    Args:
        walls: WallSegment 列表
        door_width: 门宽 (m)

    Returns:
        DoorPlacement 列表
    """
    doors: List[DoorPlacement] = []

    def _is_door_space(zt: str) -> bool:
        mt = normalize_room_meta_type(zt)
        return mt in (
            "room",
            "bedroom",
            "bathroom",
            "kitchen",
            "living",
            "dining",
            "storage",
            "staircase",
            "staircase_hall",
            "elevator_hall",
        )

    def get_type(zid: str) -> str:
        zt_lower = str(zid).lower()
        if zt_lower.startswith("room_dummy_") or zt_lower.startswith("__storage") or zt_lower.startswith("__auto_void"):
            return "storage"
        if zone_types is not None and zid in zone_types:
            return normalize_room_meta_type(zone_types.get(zid, "room"), zid)
        if "corridor" in zt_lower:
            return "corridor"
        if zid == "core_tube" or zt_lower == "core_tube":
            return "core"
        return normalize_room_meta_type(zt_lower, zid)

    def _is_void_id(zid: str) -> bool:
        return get_type(str(zid)) == "storage"

    def _is_storage(zid: str) -> bool:
        return get_type(str(zid)) == "storage"

    def _is_circulation(zid: str) -> bool:
        return get_type(str(zid)) in ("corridor", "hall", "living")

    def _storage_door_priority(storage_id: str, other_id: str, wall: WallSegment) -> Optional[Tuple[int, float]]:
        parent = storage_parent_id(storage_id)
        other_type = get_type(other_id)
        if parent:
            return (100, float(wall.length)) if str(other_id) == str(parent) else None
        if other_type in ("corridor", "hall", "living"):
            return (90, float(wall.length))
        if other_type in ("bedroom", "kitchen"):
            return (50, float(wall.length))
        return None

    def _legal_wall_pair(a: str, b: str, wall: WallSegment) -> bool:
        ta, tb = get_type(a), get_type(b)
        if ta == "storage" and tb == "storage":
            return False
        if ta == "storage":
            return _storage_door_priority(a, b, wall) is not None
        if tb == "storage":
            return _storage_door_priority(b, a, wall) is not None
        if ta == "bedroom" and tb == "bedroom":
            return False
        if ta == "bathroom" and tb == "bathroom":
            return False
        if {ta, tb} == {"kitchen", "bedroom"}:
            return False
        if (ta == "staircase_hall" and tb == "elevator_hall") or (tb == "staircase_hall" and ta == "elevator_hall"):
            return True
        if (ta == "staircase" and tb == "elevator_hall") or (tb == "staircase" and ta == "elevator_hall"):
            return True
        if (ta == "corridor" and tb == "core") or (tb == "corridor" and ta == "core"):
            return True
        if "elevator_shaft" in (ta, tb):
            return False
        if "staircase" in (ta, tb):
            return False
        if ta in ("corridor", "hall", "living") or tb in ("corridor", "hall", "living"):
            return True
        if {ta, tb} == {"bedroom", "bathroom"}:
            return True
        if {ta, tb} & {"living", "dining"} and {ta, tb} & {"kitchen"}:
            return True
        return False

    margin = 0.2
    min_storage_door_width = max(0.8, min(float(door_width), 0.9))
    shared_wall_tolerance = 1e-4

    def _axis_overlap_length_for_wall(a: Polygon, b: Polygon, wall: WallSegment) -> float:
        if wall.geometry is None or wall.geometry.is_empty or not hasattr(wall.geometry, "coords"):
            return 0.0
        coords = list(wall.geometry.coords)
        if len(coords) < 2:
            return 0.0
        x1, y1 = float(coords[0][0]), float(coords[0][1])
        x2, y2 = float(coords[-1][0]), float(coords[-1][1])
        aminx, aminy, amaxx, amaxy = (float(v) for v in a.bounds)
        bminx, bminy, bmaxx, bmaxy = (float(v) for v in b.bounds)
        if abs(x2 - x1) >= abs(y2 - y1):
            lo = max(min(x1, x2), aminx, bminx)
            hi = min(max(x1, x2), amaxx, bmaxx)
        else:
            lo = max(min(y1, y2), aminy, bminy)
            hi = min(max(y1, y2), amaxy, bmaxy)
        return max(0.0, float(hi - lo))

    def _effective_shared_wall_length(wall: WallSegment, a_id: str, b_id: str) -> float:
        if zone_polys is None:
            return float(wall.length)
        pa = zone_polys.get(str(a_id))
        pb = zone_polys.get(str(b_id))
        if pa is None or pb is None or pa.is_empty or pb.is_empty:
            return float(wall.length)
        try:
            shared_len = float(pa.boundary.intersection(pb.boundary).length)
        except Exception:
            shared_len = 0.0
        if shared_len >= min_storage_door_width:
            return min(float(wall.length), shared_len)
        try:
            close_enough = (
                float(pa.boundary.distance(pb.boundary)) <= shared_wall_tolerance
                and float(wall.geometry.distance(pa.boundary)) <= shared_wall_tolerance
                and float(wall.geometry.distance(pb.boundary)) <= shared_wall_tolerance
            )
        except Exception:
            close_enough = False
        if not close_enough:
            return shared_len
        return min(float(wall.length), _axis_overlap_length_for_wall(pa, pb, wall))

    def _zone_center(zid: str) -> Optional[Tuple[float, float]]:
        if zone_rects is None:
            return None
        rect = zone_rects.get(zid)
        if rect is None:
            return None
        x, y, w, h = rect
        return (float(x + w / 2), float(y + h / 2))

    def _default_forward(rotation: float) -> Tuple[float, float, float]:
        if abs(float(rotation) - 90.0) < 1e-6:
            return (1.0, 0.0, 0.0)
        return (0.0, 1.0, 0.0)

    def _door_forward_rect(_pos: Tuple[float, float], connects: List[str], rotation: float) -> Tuple[float, float, float]:
        if zone_types is None or zone_rects is None or len(connects) != 2:
            return _default_forward(rotation)
        a, b = connects[0], connects[1]
        ta = get_type(a)
        tb = get_type(b)
        from_id: Optional[str] = None
        to_id: Optional[str] = None
        if ta == "corridor" and tb != "corridor":
            from_id, to_id = a, b
        elif tb == "corridor" and ta != "corridor":
            from_id, to_id = b, a
        elif ta == "core" and tb != "core":
            from_id, to_id = a, b
        elif tb == "core" and ta != "core":
            from_id, to_id = b, a
        else:
            s = sorted([a, b])
            from_id, to_id = s[0], s[1]
        c_from = _zone_center(from_id) if from_id else None
        c_to = _zone_center(to_id) if to_id else None
        if c_from is None or c_to is None:
            return _default_forward(rotation)
        dx, dy = _normalize_2d(float(c_to[0]) - float(c_from[0]), float(c_to[1]) - float(c_from[1]))
        return (float(dx), float(dy), 0.0)

    def _seg_dist_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        vx = float(bx) - float(ax)
        vy = float(by) - float(ay)
        denom = vx * vx + vy * vy
        if denom <= 1e-12:
            dx = float(px) - float(ax)
            dy = float(py) - float(ay)
            return dx * dx + dy * dy
        t = ((float(px) - float(ax)) * vx + (float(py) - float(ay)) * vy) / denom
        if t <= 0.0:
            cx, cy = float(ax), float(ay)
        elif t >= 1.0:
            cx, cy = float(bx), float(by)
        else:
            cx = float(ax) + t * vx
            cy = float(ay) + t * vy
        dx = float(px) - cx
        dy = float(py) - cy
        return dx * dx + dy * dy

    def _wall_local_tangent(wall: WallSegment, pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if not isinstance(wall.geometry, LineString):
            return None
        coords = list(wall.geometry.coords)
        if len(coords) < 2:
            return None
        if len(coords) == 2:
            (x0, y0), (x1, y1) = coords[0], coords[-1]
            dx, dy = _normalize_2d(float(x1) - float(x0), float(y1) - float(y0))
            return (dx, dy)
        px, py = float(pos[0]), float(pos[1])
        best = None
        best_d = float("inf")
        for i in range(len(coords) - 1):
            ax, ay = coords[i]
            bx, by = coords[i + 1]
            d = _seg_dist_sq(px, py, float(ax), float(ay), float(bx), float(by))
            if d < best_d:
                best_d = d
                best = (float(ax), float(ay), float(bx), float(by))
        if best is None:
            return None
        ax, ay, bx, by = best
        dx, dy = _normalize_2d(float(bx) - float(ax), float(by) - float(ay))
        return (dx, dy)

    def _door_forward_from_wall(wall: WallSegment, _pos: Tuple[float, float], connects: List[str], rotation: float) -> Tuple[float, float, float]:
        if zone_polys is None or zone_types is None or len(connects) != 2:
            return _door_forward_rect(_pos, connects, rotation)
        t = _wall_local_tangent(wall, _pos)
        if t is None:
            return _door_forward_rect(_pos, connects, rotation)
        tx, ty = t
        n1 = (-float(ty), float(tx))
        n2 = (float(ty), -float(tx))

        a, b = connects[0], connects[1]
        ta = get_type(a)
        tb = get_type(b)
        from_id: Optional[str] = None
        to_id: Optional[str] = None
        if ta == "corridor" and tb != "corridor":
            from_id, to_id = a, b
        elif tb == "corridor" and ta != "corridor":
            from_id, to_id = b, a
        elif ta == "core" and tb != "core":
            from_id, to_id = a, b
        elif tb == "core" and ta != "core":
            from_id, to_id = b, a
        else:
            s = sorted([a, b])
            from_id, to_id = s[0], s[1]

        poly = zone_polys.get(to_id) if to_id else None
        if poly is None or getattr(poly, "is_empty", True):
            return _door_forward_rect(_pos, connects, rotation)

        eps = float(door_eps)
        p1 = Point(float(_pos[0]) + eps * float(n1[0]), float(_pos[1]) + eps * float(n1[1]))
        p2 = Point(float(_pos[0]) + eps * float(n2[0]), float(_pos[1]) + eps * float(n2[1]))

        def _hit(p: Point) -> bool:
            try:
                return bool(poly.contains(p))
            except Exception:
                return False

        def _hit2(p: Point) -> bool:
            try:
                return bool(poly.intersects(p))
            except Exception:
                return False

        h1 = _hit(p1) or _hit2(p1)
        h2 = _hit(p2) or _hit2(p2)
        if h1 and (not h2):
            return (float(n1[0]), float(n1[1]), 0.0)
        if h2 and (not h1):
            return (float(n2[0]), float(n2[1]), 0.0)

        try:
            c = poly.centroid
            vx = float(c.x) - float(_pos[0])
            vy = float(c.y) - float(_pos[1])
            d1 = vx * float(n1[0]) + vy * float(n1[1])
            d2 = vx * float(n2[0]) + vy * float(n2[1])
            return (float(n1[0]), float(n1[1]), 0.0) if d1 >= d2 else (float(n2[0]), float(n2[1]), 0.0)
        except Exception:
            return _door_forward_rect(_pos, connects, rotation)

    def _door_point_with_clearance(wall: WallSegment, w: float, clearance: float) -> Optional[Tuple[float, float]]:
        if wall.length < (float(w) + 2 * float(clearance)):
            return None
        try:
            if isinstance(wall.geometry, LineString):
                p = wall.geometry.interpolate(0.5, normalized=True)
                return (float(p.x), float(p.y))
        except Exception:
            pass
        coords = list(getattr(wall.geometry, "coords", []))
        if len(coords) < 2:
            return None
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        return ((float(x0) + float(x1)) / 2, (float(y0) + float(y1)) / 2)

    def _door_point_with_width(wall: WallSegment, w: float) -> Optional[Tuple[float, float]]:
        return _door_point_with_clearance(wall, w, margin)

    def _door_point(wall: WallSegment) -> Optional[Tuple[float, float]]:
        return _door_point_with_width(wall, float(door_width))

    def _core_hall_door_width(wall: WallSegment) -> Optional[float]:
        usable = float(wall.length) - 2 * float(margin) - 0.02
        w = min(0.8, usable)
        if w < 0.6:
            return None
        return float(w)

    def _door_point_staircase_hall_elevator_hall(
        wall: WallSegment,
        staircase_hall_zid: str,
        elevator_hall_zid: str,
        w: float,
    ) -> Optional[Tuple[float, float]]:
        if zone_rects is None:
            return _door_point_with_width(wall, w)
        stair = zone_rects.get(staircase_hall_zid)
        hall = zone_rects.get(elevator_hall_zid)
        if stair is None or hall is None:
            return _door_point_with_width(wall, w)
        sx, sy, sw, sh = stair
        hx, hy, hw, hh = hall

        rot = _wall_rotation(wall)
        x0, y0 = wall.geometry.coords[0]
        x1, y1 = wall.geometry.coords[-1]

        clear_margin = float(w) / 2 + 0.05

        if abs(float(rot) - 90.0) < 1e-6:
            y_line0 = float(min(y0, y1))
            y_line1 = float(max(y0, y1))
            y_hall0 = float(hy)
            y_hall1 = float(hy + hh)
            y_stair0 = float(sy)
            y_stair1 = float(sy + sh)
            seg0 = max(y_line0, y_hall0, y_stair0)
            seg1 = min(y_line1, y_hall1, y_stair1)
            if seg1 <= seg0:
                return _door_point_with_width(wall, w)
            lo = seg0 + clear_margin
            hi = seg1 - clear_margin
            if hi < lo:
                return _door_point_with_width(wall, w)
            return (float(x0), float(lo))

        x_line0 = float(min(x0, x1))
        x_line1 = float(max(x0, x1))
        x_hall0 = float(hx)
        x_hall1 = float(hx + hw)
        x_stair0 = float(sx)
        x_stair1 = float(sx + sw)
        seg0 = max(x_line0, x_hall0, x_stair0)
        seg1 = min(x_line1, x_hall1, x_stair1)
        if seg1 <= seg0:
            return _door_point_with_width(wall, w)
        lo = seg0 + clear_margin
        hi = seg1 - clear_margin
        if hi < lo:
            return _door_point_with_width(wall, w)
        return (float(lo), float(y0))

    def _door_point_staircase_elevator_hall(
        wall: WallSegment,
        staircase_zid: str,
        elevator_hall_zid: str,
    ) -> Optional[Tuple[float, float]]:
        if zone_rects is None:
            return _door_point(wall)
        stair = zone_rects.get(staircase_zid)
        hall = zone_rects.get(elevator_hall_zid)
        if stair is None or hall is None:
            return _door_point(wall)
        sx, sy, sw, sh = stair
        hx, hy, hw, hh = hall

        rot = _wall_rotation(wall)
        x0, y0 = wall.geometry.coords[0]
        x1, y1 = wall.geometry.coords[-1]

        stair_minx = float(sx)
        stair_miny = float(sy)
        stair_maxx = float(sx + sw)
        stair_maxy = float(sy + sh)

        stair_cx = float(sx + sw / 2)
        stair_cy = float(sy + sh / 2)
        hall_cx = float(hx + hw / 2)
        hall_cy = float(hy + hh / 2)
        dx = float(hall_cx) - stair_cx
        dy = float(hall_cy) - stair_cy
        axis = "y" if float(sh) >= float(sw) else "x"
        side = "max" if ((dy > 0) if axis == "y" else (dx > 0)) else "min"

        landing_min_ratio = 0.20
        landing_max_ratio = 0.30
        landing_x0 = stair_minx
        landing_x1 = stair_maxx
        landing_y0 = stair_miny
        landing_y1 = stair_maxy
        if axis == "x":
            w = stair_maxx - stair_minx
            target_len = float(door_width)
            lw = max(w * landing_min_ratio, min(w * landing_max_ratio, target_len))
            if side == "min":
                landing_x0, landing_x1 = (stair_minx, stair_minx + lw)
            else:
                landing_x0, landing_x1 = (stair_maxx - lw, stair_maxx)
        else:
            h = stair_maxy - stair_miny
            target_len = float(door_width)
            lh = max(h * landing_min_ratio, min(h * landing_max_ratio, target_len))
            if side == "min":
                landing_y0, landing_y1 = (stair_miny, stair_miny + lh)
            else:
                landing_y0, landing_y1 = (stair_maxy - lh, stair_maxy)

        clear_margin = float(door_width) / 2 + 0.05

        if abs(float(rot) - 90.0) < 1e-6:
            y_line0 = float(min(y0, y1))
            y_line1 = float(max(y0, y1))
            y_hall0 = float(hy)
            y_hall1 = float(hy + hh)
            seg0 = max(y_line0, y_hall0)
            seg1 = min(y_line1, y_hall1)
            if seg1 <= seg0:
                seg0, seg1 = (y_line0, y_line1)
            x_const = float(x0)
            if axis == "y":
                seg0 = max(seg0, landing_y0)
                seg1 = min(seg1, landing_y1)
            else:
                if not (landing_x0 - 1e-6 <= x_const <= landing_x1 + 1e-6):
                    return _door_point(wall)
            if seg1 <= seg0:
                return _door_point(wall)
            lo = seg0 + clear_margin
            hi = seg1 - clear_margin
            if hi < lo:
                lo = seg0
                hi = seg1
            if side == "min":
                y = lo
            else:
                y = hi
            return (x_const, float(y))

        x_line0 = float(min(x0, x1))
        x_line1 = float(max(x0, x1))
        x_hall0 = float(hx)
        x_hall1 = float(hx + hw)
        seg0 = max(x_line0, x_hall0)
        seg1 = min(x_line1, x_hall1)
        if seg1 <= seg0:
            seg0, seg1 = (x_line0, x_line1)
        y_const = float(y0)
        if axis == "x":
            seg0 = max(seg0, landing_x0)
            seg1 = min(seg1, landing_x1)
        else:
            if not (landing_y0 - 1e-6 <= y_const <= landing_y1 + 1e-6):
                return _door_point(wall)
        if seg1 <= seg0:
            return _door_point(wall)
        lo = seg0 + clear_margin
        hi = seg1 - clear_margin
        if hi < lo:
            lo = seg0
            hi = seg1
        if side == "min":
            x = lo
        else:
            x = hi
        return (float(x), y_const)

    def _wall_key(w: WallSegment) -> Tuple:
        coords = list(w.geometry.coords)
        a = (round(coords[0][0], 2), round(coords[0][1], 2))
        b = (round(coords[-1][0], 2), round(coords[-1][1], 2))
        p1, p2 = (a, b) if a <= b else (b, a)
        return (w.type, round(float(w.thickness), 3), p1, p2)

    candidates: List[WallSegment] = []
    hall_hall_walls: List[WallSegment] = []
    walls_for_doors = [
        w for w in walls
        if isinstance(w.geometry, LineString)
        and getattr(w, "category", None) != "wall_junction"
    ]

    for wall in walls_for_doors:
        if wall.type != "partition_wall":
            continue
        if len(wall.room_ids) != 2:
            continue

        if zone_types is not None:
            a, b = wall.room_ids[0], wall.room_ids[1]
            type_a = get_type(a)
            type_b = get_type(b)
            if (type_a == "staircase_hall" and type_b == "elevator_hall") or (type_b == "staircase_hall" and type_a == "elevator_hall"):
                if wall.length >= 0.6:
                    hall_hall_walls.append(wall)
                continue
            if (type_a == "staircase_shaft" and type_b == "elevator_hall") or (type_b == "staircase_shaft" and type_a == "elevator_hall"):
                continue
            if "elevator_shaft" in (type_a, type_b):
                continue
            if wall.length < door_width:
                continue
            if type_a == "room" and type_b == "room" and zone_rects is not None:
                ra = zone_rects.get(a)
                rb = zone_rects.get(b)
                if ra is not None and rb is not None:
                    ax0, ay0, aw0, ah0 = ra
                    bx0, by0, bw0, bh0 = rb
                    overlap_y0 = max(0.0, min(float(ay0 + ah0), float(by0 + bh0)) - max(float(ay0), float(by0)))
                    overlap_x0 = max(0.0, min(float(ax0 + aw0), float(bx0 + bw0)) - max(float(ax0), float(bx0)))
                    gap_x0 = 0.0
                    if float(ax0 + aw0) <= float(bx0):
                        gap_x0 = float(bx0) - float(ax0 + aw0)
                    elif float(bx0 + bw0) <= float(ax0):
                        gap_x0 = float(ax0) - float(bx0 + bw0)
                    gap_y0 = 0.0
                    if float(ay0 + ah0) <= float(by0):
                        gap_y0 = float(by0) - float(ay0 + ah0)
                    elif float(by0 + bh0) <= float(ay0):
                        gap_y0 = float(ay0) - float(by0 + bh0)
                    if (gap_x0 > 0.01 and overlap_y0 >= 0.5) or (gap_y0 > 0.01 and overlap_x0 >= 0.5):
                        continue
            if not _legal_wall_pair(a, b, wall):
                continue
            if "core" in (type_a, type_b) and zone_rects is not None:
                core_id = a if type_a == "core" else (b if type_b == "core" else None)
                if core_id and core_id in zone_rects:
                    core_south = float(zone_rects[core_id][1])
                    x0, y0 = wall.geometry.coords[0]
                    x1, y1 = wall.geometry.coords[-1]
                    if abs(y0 - y1) < abs(x0 - x1):
                        if abs(y0 - core_south) > 0.02:
                            continue
                    else:
                        continue
        else:
            if wall.length < door_width:
                continue

        candidates.append(wall)

    used: set = set()
    if zone_types is not None and hall_hall_walls:
        chosen = max(hall_hall_walls, key=lambda w: w.length)
        door_w = min(0.8, float(chosen.length) - 0.10)
        if door_w >= 0.6:
            try:
                center_pt = chosen.geometry.interpolate(0.5, normalized=True)
                coords = list(chosen.geometry.coords)
                dx = float(coords[-1][0]) - float(coords[0][0])
                dy = float(coords[-1][1]) - float(coords[0][1])
                angle_rad = float(math.atan2(dy, dx))
                rot_deg = float(math.degrees(angle_rad))
                rot = 90.0 if abs(abs(rot_deg) - 90.0) <= 1e-3 else 0.0
                nx, ny = (-dy, dx)
                nlen = float(math.hypot(nx, ny))
                forward = (nx / nlen, ny / nlen, 0.0) if nlen > 1e-3 else (1.0, 0.0, 0.0)
                key = _wall_key(chosen)
                used.add(key)
                doors.append(DoorPlacement(
                    position=(round(float(center_pt.x), 2), round(float(center_pt.y), 2)),
                    width=round(float(door_w), 2),
                    connects=list(chosen.room_ids),
                    wall_type=chosen.type,
                    rotation=round(float(rot), 2),
                    thickness=float(chosen.thickness),
                    forward=(float(forward[0]), float(forward[1]), 0.0),
                ))
            except Exception:
                pass

    if zone_types is None:
        for wall in candidates:
            p = _door_point(wall)
            if p is None:
                continue
            key = _wall_key(wall)
            if key in used:
                continue
            used.add(key)
            rot = _wall_rotation(wall)
            connects = list(wall.room_ids)
            doors.append(DoorPlacement(
                position=(round(p[0], 2), round(p[1], 2)),
                width=door_width,
                connects=connects,
                wall_type=wall.type,
                rotation=rot,
                thickness=float(wall.thickness),
                forward=_door_forward_rect(p, connects, rot),
            ))
        return doors

    storage_with_door: Set[str] = set()
    storage_id_set = {rid for w in candidates for rid in (w.room_ids or []) if _is_storage(str(rid))}
    if zone_types is not None:
        storage_id_set.update({zid for zid, zt in zone_types.items() if get_type(str(zid)) == "storage"})
    storage_ids = sorted(storage_id_set)
    for sid in storage_ids:
        storage_candidates: List[Tuple[int, float, WallSegment, str]] = []
        for w in candidates:
            if sid not in (w.room_ids or []):
                continue
            if len(w.room_ids) != 2:
                continue
            other = w.room_ids[1] if w.room_ids[0] == sid else w.room_ids[0]
            priority = _storage_door_priority(str(sid), str(other), w)
            if priority is None:
                continue
            storage_candidates.append((int(priority[0]), float(priority[1]), w, str(other)))
        storage_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _prio, _length, chosen, _other in storage_candidates:
            p = _door_point(chosen)
            if p is None:
                continue
            key = _wall_key(chosen)
            if key in used:
                continue
            used.add(key)
            storage_with_door.add(str(sid))
            rot = _wall_rotation(chosen)
            connects = list(chosen.room_ids)
            doors.append(DoorPlacement(
                position=(round(p[0], 2), round(p[1], 2)),
                width=door_width,
                connects=connects,
                wall_type=chosen.type,
                rotation=rot,
                thickness=float(chosen.thickness),
                forward=_door_forward_from_wall(chosen, p, connects, rot),
            ))
            break

    def _is_strict_storage_id(zid: str) -> bool:
        raw = str(zid)
        return raw.startswith("room_storage_") or raw.startswith("__auto_storage") or raw.startswith("__storage")

    for sid in storage_ids:
        sid = str(sid)
        if sid in storage_with_door:
            continue
        fallback_candidates: List[Tuple[int, float, WallSegment, str]] = []
        for wall in walls_for_doors:
            if wall.type != "partition_wall":
                continue
            if len(wall.room_ids) != 2 or sid not in (wall.room_ids or []):
                continue
            if getattr(wall, "category", None) == "wall_junction":
                continue
            other = wall.room_ids[1] if wall.room_ids[0] == sid else wall.room_ids[0]
            priority = _storage_door_priority(sid, str(other), wall)
            if priority is None:
                continue
            if _wall_key(wall) in used:
                continue
            effective_shared_len = _effective_shared_wall_length(wall, sid, str(other))
            # Fallback can tolerate tiny floating-point offsets, but still needs a real door-width overlap.
            if effective_shared_len < min_storage_door_width:
                logger.warning(
                    "[DOOR] Storage door candidate rejected | Storage=%s | Other=%s | SharedLen=%.3fm | Min=%.3fm",
                    sid,
                    other,
                    float(effective_shared_len),
                    float(min_storage_door_width),
                )
                continue
            fallback_candidates.append((int(priority[0]), float(effective_shared_len), wall, str(other)))
        fallback_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _prio, _length, chosen, _other in fallback_candidates:
            door_w = min(float(door_width), float(_length), float(chosen.length))
            if door_w < min_storage_door_width:
                logger.warning(
                    "[DOOR] Storage fallback door too narrow | Storage=%s | DoorWidth=%.3fm | Min=%.3fm",
                    sid,
                    float(door_w),
                    float(min_storage_door_width),
                )
                continue
            p = _door_point_with_clearance(chosen, door_w, 0.0)
            if p is None:
                continue
            key = _wall_key(chosen)
            if key in used:
                continue
            used.add(key)
            storage_with_door.add(sid)
            rot = _wall_rotation(chosen)
            connects = list(chosen.room_ids)
            doors.append(DoorPlacement(
                position=(round(p[0], 2), round(p[1], 2)),
                width=round(float(door_w), 2),
                connects=connects,
                wall_type=chosen.type,
                rotation=rot,
                thickness=float(chosen.thickness),
                forward=_door_forward_from_wall(chosen, p, connects, rot),
            ))
            logger.warning(
                "[DOOR] Storage fallback door generated | Storage=%s | Connects=%s | Width=%.2fm",
                sid,
                connects,
                float(door_w),
            )
            break
        if sid not in storage_with_door and _is_strict_storage_id(sid):
            logger.error("[DOOR] Synthetic storage has no legal door | Storage=%s", sid)
            raise LayoutTopologyError(f"Synthetic storage has no legal door: {sid}")

    room_ids = [
        zid for zid, zt in zone_types.items()
        if _is_door_space(zt) and not _is_storage(str(zid))
    ]
    for rid in room_ids:
        room_candidates = [
            w for w in candidates
            if rid in w.room_ids and not any(_is_storage(str(x)) for x in (w.room_ids or []))
        ]
        if not room_candidates:
            continue

        corridor_candidates = []
        other_candidates = []
        for w in room_candidates:
            a, b = w.room_ids[0], w.room_ids[1]
            other = b if a == rid else a
            if get_type(other) == "corridor":
                corridor_candidates.append(w)
            else:
                other_candidates.append(w)

        chosen = None
        if corridor_candidates:
            chosen = max(corridor_candidates, key=lambda w: w.length)
        elif other_candidates:
            chosen = max(other_candidates, key=lambda w: w.length)
        if chosen is None:
            continue
        a, b = chosen.room_ids[0], chosen.room_ids[1]
        ta = get_type(a)
        tb = get_type(b)
        p = None
        if (ta == "staircase_hall" and tb == "elevator_hall") or (ta == "elevator_hall" and tb == "staircase_hall"):
            stair_id = a if ta == "staircase_hall" else b
            hall_id = a if ta == "elevator_hall" else b
            w = _core_hall_door_width(chosen)
            if w is None:
                continue
            p = _door_point_staircase_hall_elevator_hall(chosen, stair_id, hall_id, w)
            if p is None:
                continue
            key = _wall_key(chosen)
            if key in used:
                continue
            used.add(key)
            rot = _wall_rotation(chosen)
            connects = list(chosen.room_ids)
            doors.append(DoorPlacement(
                position=(round(p[0], 2), round(p[1], 2)),
                width=float(w),
                connects=connects,
                wall_type=chosen.type,
                rotation=rot,
                thickness=float(chosen.thickness),
                forward=_door_forward_from_wall(chosen, p, connects, rot),
            ))
            continue
        if (ta == "staircase" and tb == "elevator_hall") or (ta == "elevator_hall" and tb == "staircase"):
            stair_id = a if ta == "staircase" else b
            hall_id = a if ta == "elevator_hall" else b
            p = _door_point_staircase_elevator_hall(chosen, stair_id, hall_id)
        else:
            p = _door_point(chosen)
        if p is None:
            continue
        key = _wall_key(chosen)
        if key in used:
            continue
        used.add(key)
        rot = _wall_rotation(chosen)
        connects = list(chosen.room_ids)
        doors.append(DoorPlacement(
            position=(round(p[0], 2), round(p[1], 2)),
            width=door_width,
            connects=connects,
            wall_type=chosen.type,
            rotation=rot,
            thickness=float(chosen.thickness),
            forward=_door_forward_from_wall(chosen, p, connects, rot),
        ))

    core_candidates = []
    for w in candidates:
        a, b = w.room_ids[0], w.room_ids[1]
        type_a = get_type(a)
        type_b = get_type(b)
        if (type_a == "corridor" and type_b == "core") or (type_b == "corridor" and type_a == "core"):
            core_candidates.append(w)
    if core_candidates:
        chosen = max(core_candidates, key=lambda w: w.length)
        p = _door_point(chosen)
        if p is not None:
            key = _wall_key(chosen)
            if key not in used:
                used.add(key)
                rot = _wall_rotation(chosen)
                connects = list(chosen.room_ids)
                doors.append(DoorPlacement(
                    position=(round(p[0], 2), round(p[1], 2)),
                    width=door_width,
                    connects=connects,
                    wall_type=chosen.type,
                    rotation=rot,
                    thickness=float(chosen.thickness),
                    forward=_door_forward_from_wall(chosen, p, connects, rot),
                ))

    staircase_id = None
    elevator_hall_id = None
    for zid, zt in zone_types.items():
        if zt in ("staircase_hall", "staircase") and staircase_id is None:
            staircase_id = zid
        elif zt == "elevator_hall" and elevator_hall_id is None:
            elevator_hall_id = zid

    if staircase_id and elevator_hall_id:
        forced = []
        for w in candidates:
            a, b = w.room_ids[0], w.room_ids[1]
            if {a, b} == {staircase_id, elevator_hall_id}:
                forced.append(w)
        if forced:
            chosen = max(forced, key=lambda w: w.length)
            st = get_type(staircase_id)
            if st == "staircase_hall":
                w = _core_hall_door_width(chosen)
                if w is not None:
                    p = _door_point_staircase_hall_elevator_hall(chosen, staircase_id, elevator_hall_id, w)
                    if p is not None:
                        key = _wall_key(chosen)
                        if key not in used:
                            used.add(key)
                            rot = _wall_rotation(chosen)
                            connects = list(chosen.room_ids)
                            doors.append(DoorPlacement(
                                position=(round(p[0], 2), round(p[1], 2)),
                                width=float(w),
                                connects=connects,
                                wall_type=chosen.type,
                                rotation=rot,
                                thickness=float(chosen.thickness),
                                forward=_door_forward_from_wall(chosen, p, connects, rot),
                            ))
            else:
                p = _door_point_staircase_elevator_hall(chosen, staircase_id, elevator_hall_id)
                if p is not None:
                    key = _wall_key(chosen)
                    if key not in used:
                        used.add(key)
                        rot = _wall_rotation(chosen)
                        connects = list(chosen.room_ids)
                        doors.append(DoorPlacement(
                            position=(round(p[0], 2), round(p[1], 2)),
                            width=door_width,
                            connects=connects,
                            wall_type=chosen.type,
                            rotation=rot,
                            thickness=float(chosen.thickness),
                            forward=_door_forward_from_wall(chosen, p, connects, rot),
                        ))

    return doors


# ============================================================
# 窗户放置
# ============================================================

def generate_windows(
    walls: List[WallSegment],
    rooms: list,
    window_width: float = 1.2,
    window_spacing: float = 2.0,
) -> List[WindowPlacement]:
    """
    在外墙上为 needs_window=True 的房间放置窗户。

    规则：沿外墙每 window_spacing 米放一个窗，至少一个。

    Args:
        walls: WallSegment 列表
        rooms: 有 id/room_id, has_window/needs_window 属性的 RoomResult 列表
        window_width: 窗宽 (m)
        window_spacing: 窗间距 (m)

    Returns:
        WindowPlacement 列表
    """
    # 构建需要窗户的房间集合
    window_rooms: set = set()
    for room in rooms:
        room_id = getattr(room, "id", getattr(room, "room_id", "?"))
        has_window = getattr(room, "has_window", False) or getattr(room, "needs_window", False)
        if has_window:
            window_rooms.add(room_id)

    windows: List[WindowPlacement] = []
    room_center: Dict[str, Tuple[float, float]] = {}
    for r in rooms:
        rid = getattr(r, "id", getattr(r, "room_id", "?"))
        try:
            if hasattr(r, "polygon") and r.polygon is not None and (not r.polygon.is_empty):
                c = r.polygon.centroid
                room_center[rid] = (float(c.x), float(c.y))
        except Exception:
            continue

    for wall in walls:
        if wall.type != "exterior_wall":
            continue
        if len(wall.room_ids) != 1:
            continue

        room_id = wall.room_ids[0]
        if room_id not in window_rooms:
            continue

        wall_length = wall.length
        if wall_length < window_width:
            continue

        num_windows = max(1, int(wall_length / window_spacing))

        for k in range(num_windows):
            try:
                pos = wall.geometry.interpolate((k + 0.5) / num_windows, normalized=True)
                windows.append(WindowPlacement(
                    position=(round(pos.x, 2), round(pos.y, 2)),
                    width=window_width,
                    room_id=room_id,
                    wall_length=round(wall_length, 2),
                    rotation=_wall_rotation(wall),
                ))
            except Exception as e:
                logger.debug(f"Window placement failed: {e}")

    return windows


def generate_windows_from_exterior_walls(
    exterior_walls: List[WallSegment],
    rooms_needing_window: Set[str],
    window_width: float = 1.2,
    window_spacing: float = 2.0,
) -> List[WindowPlacement]:
    windows: List[WindowPlacement] = []

    for wall in exterior_walls:
        if wall.type != "exterior_wall":
            continue
        if len(wall.room_ids) != 1:
            continue

        room_id = wall.room_ids[0]
        if room_id not in rooms_needing_window:
            continue

        coords = list(wall.geometry.coords)
        if len(coords) < 2:
            continue
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        wall_len = wall.length
        if wall_len < window_width:
            continue

        num_windows = max(1, int(wall_len / window_spacing))
        for k in range(num_windows):
            t = (k + 0.5) / num_windows
            wx = x0 + t * (x1 - x0)
            wy = y0 + t * (y1 - y0)
            rot = _wall_rotation(wall)
            if abs(float(rot) - 90.0) < 1e-6:
                fx, fy = (1.0, 0.0)
            else:
                fx, fy = (0.0, 1.0)
            windows.append(WindowPlacement(
                position=(round(wx, 2), round(wy, 2)),
                width=window_width,
                room_id=room_id,
                wall_length=round(wall_len, 2),
                rotation=rot,
                thickness=float(wall.thickness),
                forward=(float(fx), float(fy), 0.0),
            ))

    return windows


def generate_windows_from_floor_boundary(
    room_rects: Dict[str, Tuple[float, float, float, float]],
    zone_types: Dict[str, str],
    rooms_needing_window: Set[str],
    floor_bounds: Tuple[float, float, float, float],
    exterior_thickness: float = 0.24,
    window_width: float = 1.2,
    window_spacing: float = 2.0,
) -> List[WindowPlacement]:
    fminx, fminy, fmaxx, fmaxy = floor_bounds
    proximity = max(0.05, float(exterior_thickness))

    windows: List[WindowPlacement] = []
    for rid in rooms_needing_window:
        if normalize_room_meta_type(zone_types.get(rid), rid) not in ("room", "bedroom", "bathroom", "kitchen", "living", "dining"):
            continue
        rect = room_rects.get(rid)
        if rect is None:
            continue
        rx, ry, rw, rh = rect
        if rw <= 0 or rh <= 0:
            continue

        left_gap = rx - fminx
        right_gap = fmaxx - (rx + rw)
        bottom_gap = ry - fminy
        top_gap = fmaxy - (ry + rh)

        gaps = {
            "left": left_gap,
            "right": right_gap,
            "bottom": bottom_gap,
            "top": top_gap,
        }
        side = min(gaps.keys(), key=lambda k: gaps[k])
        if gaps[side] > proximity:
            continue

        t = float(exterior_thickness)
        if side in ("left", "right"):
            x = (fminx + t / 2) if side == "left" else (fmaxx - t / 2)
            y0 = max(fminy, ry)
            y1 = min(fmaxy, ry + rh)
            wall_len = y1 - y0
            if wall_len < window_width:
                continue
            num = max(1, int(wall_len / window_spacing))
            for k in range(num):
                t = (k + 0.5) / num
                wy = y0 + t * (y1 - y0)
                cx = float(rx + rw / 2)
                cy = float(ry + rh / 2)
                fx, fy = _normalize_2d(float(cx) - float(x), float(cy) - float(wy))
                windows.append(WindowPlacement(
                    position=(round(float(x), 2), round(float(wy), 2)),
                    width=window_width,
                    room_id=rid,
                    wall_length=round(float(wall_len), 2),
                    rotation=90.0,
                    thickness=float(exterior_thickness),
                    forward=(float(fx), float(fy), 0.0),
                ))
        else:
            y = (fminy + t / 2) if side == "bottom" else (fmaxy - t / 2)
            x0 = max(fminx, rx)
            x1 = min(fmaxx, rx + rw)
            wall_len = x1 - x0
            if wall_len < window_width:
                continue
            num = max(1, int(wall_len / window_spacing))
            for k in range(num):
                t = (k + 0.5) / num
                wx = x0 + t * (x1 - x0)
                cx = float(rx + rw / 2)
                cy = float(ry + rh / 2)
                fx, fy = _normalize_2d(float(cx) - float(wx), float(cy) - float(y))
                windows.append(WindowPlacement(
                    position=(round(float(wx), 2), round(float(y), 2)),
                    width=window_width,
                    room_id=rid,
                    wall_length=round(float(wall_len), 2),
                    rotation=0.0,
                    thickness=float(exterior_thickness),
                    forward=(float(fx), float(fy), 0.0),
                ))

    return windows


# ============================================================
# 一站式后处理
# ============================================================

def postprocess_floor(
    rooms: list,
    floor_boundary: Polygon,
    corridors: Optional[list] = None,
    core_tube=None,
    floor_id: Optional[str] = None,
    topology_mode: Optional[str] = None,
    is_ground_floor: bool = False,
    walls: Optional[List[WallSegment]] = None,
    zone_types: Optional[Dict[str, str]] = None,
    zone_rects: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    required_adjacency: Optional[Dict[str, List[str]]] = None,
    rooms_needing_window: Optional[Set[str]] = None,
    floor_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> PostprocessResult:
    """
    对单层布局执行完整后处理。

    Args:
        rooms: RoomResult 列表
        floor_boundary: 楼层外轮廓
        corridors: Corridor 列表（可选，参与内墙生成）

    Returns:
        PostprocessResult
    """
    if walls is None:
        walls = generate_wall_mesh(
            rooms=rooms,
            corridors=corridors or [],
            core_tube=core_tube,
            floor_boundary=floor_boundary,
        )

    rr: Dict[str, Tuple[float, float, float, float]] = dict(zone_rects or {})
    zt: Dict[str, str] = dict(zone_types or {})
    rnw: Set[str] = set(rooms_needing_window or set())
    if zone_rects is None or zone_types is None or rooms_needing_window is None:
        for r in rooms:
            rid = getattr(r, "id", getattr(r, "room_id", "?"))
            rtype = str(getattr(r, "room_type", getattr(r, "type", "")) or "").lower()
            if rtype == "void" or bool(getattr(r, "skip_solver", False)):
                continue
            if not hasattr(r, "polygon") or r.polygon.is_empty:
                continue
            minx, miny, maxx, maxy = r.polygon.bounds
            rr[rid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
            zt[rid] = zt.get(rid, rtype or "room")
            has_window = getattr(r, "has_window", False) or getattr(r, "needs_window", False)
            if has_window:
                rnw.add(rid)
        if corridors:
            for c in corridors:
                cid = getattr(c, "id", None)
                poly = getattr(c, "polygon", None)
                if cid and poly is not None and hasattr(poly, "is_empty") and (not poly.is_empty):
                    minx, miny, maxx, maxy = poly.bounds
                    rr[cid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
                    zt[cid] = zt.get(cid, "corridor")

    if core_tube is not None and hasattr(core_tube, "polygon") and (not getattr(core_tube.polygon, "is_empty", True)):
        core_zones = [
            ("core_staircase_hall", "staircase_hall", getattr(core_tube, "staircase_hall", None)),
            ("core_staircase_hall_b", "staircase_hall", getattr(core_tube, "staircase_hall_b", None)),
            ("core_elevator_hall", "elevator_hall", getattr(core_tube, "elevator_hall", None)),
            ("core_elevator_hall_b", "elevator_hall", getattr(core_tube, "elevator_hall_b", None)),
        ]
        for zid, ztype, poly in core_zones:
            if zid in rr and zid in zt:
                continue
            if poly is None or getattr(poly, "is_empty", True):
                continue
            minx, miny, maxx, maxy = poly.bounds
            rr[zid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
            zt[zid] = zt.get(zid, ztype)

    for r in rooms:
        rid = getattr(r, "id", getattr(r, "room_id", "?"))
        rtype = str(getattr(r, "room_type", getattr(r, "type", "")) or "").lower()
        if rtype == "void" or bool(getattr(r, "skip_solver", False)):
            rr.pop(rid, None)
            zt.pop(rid, None)
            if rid in rnw:
                rnw.remove(rid)

    try:
        fminx, fminy, fmaxx, fmaxy = (floor_bounds or floor_boundary.bounds)
    except Exception:
        fminx, fminy, fmaxx, fmaxy = floor_boundary.bounds

    walls = _normalize_walls(walls, floor_bounds=(fminx, fminy, fmaxx, fmaxy), zone_rects=rr)

    zone_polys: Dict[str, Polygon] = {}
    for r in rooms:
        rid = getattr(r, "id", getattr(r, "room_id", "?"))
        rtype = str(getattr(r, "room_type", getattr(r, "type", "")) or "").lower()
        if rtype == "void" or bool(getattr(r, "skip_solver", False)):
            continue
        poly = getattr(r, "polygon", None)
        if isinstance(poly, Polygon) and (not poly.is_empty):
            zone_polys[str(rid)] = poly
    for c in (corridors or []):
        cid = getattr(c, "id", None)
        poly = getattr(c, "polygon", None)
        if cid is None or poly is None or getattr(poly, "is_empty", True):
            continue
        if isinstance(poly, Polygon):
            zone_polys[str(cid)] = poly
    if core_tube is not None:
        used_core = False
        core_keys = [
            ("core_staircase_hall", "staircase_hall"),
            ("core_staircase_hall_b", "staircase_hall_b"),
            ("core_elevator_hall", "elevator_hall"),
            ("core_elevator_hall_b", "elevator_hall_b"),
        ]
        for zid, attr in core_keys:
            poly = getattr(core_tube, attr, None)
            if poly is None or getattr(poly, "is_empty", True):
                continue
            if isinstance(poly, Polygon):
                zone_polys[str(zid)] = poly
                used_core = True
        core_poly = getattr(core_tube, "polygon", None)
        if (not used_core) and isinstance(core_poly, Polygon) and (not core_poly.is_empty):
            zone_polys["core_tube"] = core_poly

    doors = generate_doors(walls, zone_types=zt or None, zone_rects=rr or None, zone_polys=zone_polys or None)
    exterior_thickness = next((w.thickness for w in walls if w.type == "exterior_wall"), 0.24)
    windows = generate_windows_from_floor_boundary(
        room_rects=rr,
        zone_types=zt,
        rooms_needing_window=rnw,
        floor_bounds=floor_bounds or floor_boundary.bounds,
        exterior_thickness=float(exterior_thickness),
    )

    if is_ground_floor and zt and rr:
        fminx, fminy, fmaxx, fmaxy = (floor_bounds or floor_boundary.bounds)
        
        
        ext_thick = float(exterior_thickness)
        door_w = 1.0
        
        if ext_thick > 1e-6:
            cm = door_w / 2 + 0.05
            proximity = 0.25

            entrance_ids = {
                str(getattr(r, "id", getattr(r, "room_id", "")))
                for r in rooms
                if str(getattr(r, "room_type", getattr(r, "type", "")) or "").lower() == "entrance"
            }
            corridor_ids = [
                rid for rid, t in zt.items()
                if (t == "corridor" or str(rid) in entrance_ids) and rid in rr
            ]

            chosen: Optional[Tuple[str, str, float, float, float, float]] = None
            corridor_poly_by_id: Dict[str, Polygon] = {}
            if corridors:
                for c in corridors:
                    cid0 = getattr(c, "id", None) if not isinstance(c, dict) else c.get("id")
                    poly0 = getattr(c, "polygon", None) if not isinstance(c, dict) else c.get("polygon")
                    if not cid0 or poly0 is None:
                        continue
                    try:
                        if isinstance(poly0, Polygon):
                            poly = poly0
                        else:
                            poly = Polygon(poly0)
                        if not poly.is_empty:
                            corridor_poly_by_id[str(cid0)] = poly
                    except Exception:
                        continue
            for r in rooms:
                rid0 = str(getattr(r, "id", getattr(r, "room_id", "")))
                if rid0 not in entrance_ids:
                    continue
                poly0 = getattr(r, "polygon", None)
                if poly0 is None:
                    continue
                try:
                    if isinstance(poly0, Polygon):
                        poly = poly0
                    else:
                        poly = Polygon(poly0)
                    if not poly.is_empty:
                        corridor_poly_by_id[str(rid0)] = poly
                except Exception:
                    continue

            if corridor_poly_by_id and corridor_ids:
                wall_lines = {
                    "right": LineString([(float(fmaxx), float(fminy)), (float(fmaxx), float(fmaxy))]),
                    "left": LineString([(float(fminx), float(fminy)), (float(fminx), float(fmaxy))]),
                    "top": LineString([(float(fminx), float(fmaxy)), (float(fmaxx), float(fmaxy))]),
                    "bottom": LineString([(float(fminx), float(fminy)), (float(fmaxx), float(fminy))]),
                }
                best_len = 0.0
                best_tuple: Optional[Tuple[str, str, float, float, float, float]] = None
                for cid in corridor_ids:
                    poly = corridor_poly_by_id.get(str(cid))
                    if poly is None or poly.is_empty:
                        continue
                    for side, wl in wall_lines.items():
                        try:
                            hit = poly.intersection(wl.buffer(float(proximity)))
                            hit_len = float(getattr(hit, "length", 0.0))
                        except Exception:
                            continue
                        if hit_len > best_len + 1e-6:
                            rx, ry, rw, rh = rr[cid]
                            try:
                                hminx, hminy, hmaxx, hmaxy = (float(v) for v in hit.bounds)
                                if side in ("left", "right"):
                                    ry = float(hminy)
                                    rh = float(hmaxy - hminy)
                                else:
                                    rx = float(hminx)
                                    rw = float(hmaxx - hminx)
                            except Exception:
                                pass
                            best_len = hit_len
                            best_tuple = (side, str(cid), float(rx), float(ry), float(rw), float(rh))
                if best_tuple is not None and best_len >= float(door_w) + 0.1:
                    chosen = best_tuple

            if chosen is None:
                right_candidates: List[Tuple[float, float, str, float, float, float, float]] = []
                for cid in corridor_ids:
                    rx, ry, rw, rh = rr[cid]
                    right_gap = float(fmaxx) - float(rx + rw)
                    if right_gap <= proximity:
                        right_candidates.append((float(rx + rw), float(rh), cid, float(rx), float(ry), float(rw), float(rh)))

                right_candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
                if right_candidates:
                    _, _, cid, rx, ry, rw, rh = right_candidates[0]
                    chosen = ("right", cid, rx, ry, rw, rh)
                else:
                    other_candidates: List[Tuple[float, float, float, str, str, float, float, float, float]] = []
                    for cid in corridor_ids:
                        rx, ry, rw, rh = rr[cid]
                        left_gap = float(rx) - float(fminx)
                        right_gap = float(fmaxx) - float(rx + rw)
                        bottom_gap = float(ry) - float(fminy)
                        top_gap = float(fmaxy) - float(ry + rh)
                        gaps = [
                            ("right", right_gap),
                            ("top", top_gap),
                            ("bottom", bottom_gap),
                            ("left", left_gap),
                        ]
                        side, gap = min(gaps, key=lambda t: t[1])
                        if float(gap) > proximity:
                            continue
                        side_score = {"right": 4.0, "top": 3.0, "bottom": 2.0, "left": 1.0}.get(side, 0.0)
                        other_candidates.append((side_score, -float(gap), float(rw * rh), cid, side, float(rx), float(ry), float(rw), float(rh)))

                    other_candidates.sort(reverse=True)
                    if other_candidates:
                        _, _, _, cid, side, rx, ry, rw, rh = other_candidates[0]
                        chosen = (side, cid, rx, ry, rw, rh)

            if chosen is not None:
                side, corridor_id, rx, ry, rw, rh = chosen
                if side in ("left", "right"):
                    wall_len = float(rh)
                    if wall_len >= (door_w + 0.1):
                        # --- 修复 2：计算门在墙内的 X 轴中心点时，必须使用墙厚 (ext_thick) ---
                        x = float(fminx) + ext_thick / 2 if side == "left" else float(fmaxx) - ext_thick / 2
                        y_min = float(ry) + cm
                        y_max = float(ry + rh) - cm
                        if y_max >= y_min:
                            y_mid = float(ry + rh / 2)
                            y = min(max(y_mid, y_min), y_max)
                            doors.append(DoorPlacement(
                                position=(round(float(x), 2), round(float(y), 2)),
                                width=door_w,  # --- 修复 3：传入真正的门宽 1.2 ---
                                connects=[corridor_id, "__exterior__"],
                                wall_type="exterior_wall",
                                rotation=90.0,
                                thickness=ext_thick,
                                forward=(1.0, 0.0, 0.0),
                            ))
                else:
                    wall_len = float(rw)
                    if wall_len >= (door_w + 0.1):
                        # --- 修复 4：计算门在墙内的 Y 轴中心点时，必须使用墙厚 (ext_thick) ---
                        y = float(fminy) + ext_thick / 2 if side == "bottom" else float(fmaxy) - ext_thick / 2
                        x_min = float(rx) + cm
                        x_max = float(rx + rw) - cm
                        if x_max >= x_min:
                            x_mid = float(rx + rw / 2)
                            x = min(max(x_mid, x_min), x_max)
                            doors.append(DoorPlacement(
                                position=(round(float(x), 2), round(float(y), 2)),
                                width=door_w,  # --- 修复 5：传入真正的门宽 1.2 ---
                                connects=[corridor_id, "__exterior__"],
                                wall_type="exterior_wall",
                                rotation=0.0,
                                thickness=ext_thick,
                                forward=(0.0, 1.0, 0.0),
                            ))

    topology_mode_l = str(topology_mode or getattr(core_tube, "topology_mode", "") or "").lower()
    core_access_passed = True
    if core_tube is not None:
        try:
            core_contract = build_core_footprint_contract(
                core_tube,
                floor_id=floor_id or getattr(core_tube, "core_contract_floor_id", None) or "F?",
                topology_mode=topology_mode_l or "unknown",
                created_from="postprocess_floor",
            )
            core_access_meta = validate_core_access(
                floor_id=floor_id or str(getattr(core_tube, "core_contract_floor_id", None) or "F?"),
                topology_mode=topology_mode_l or "unknown",
                core_contract=core_contract,
                doors=doors,
                zone_types=zt,
                min_width=0.8,
                hard_fail=(topology_mode_l == "grid_growth"),
                require_portal_binding=False,
            )
            core_access_passed = bool(core_access_meta.get("valid_core_access", False))
            if core_access_meta.get("missing_portal_binding_doors"):
                logger.warning(
                    "[CORE] Access portal binding missing | floor=%s | contract=%s | doors=%s",
                    floor_id,
                    core_access_meta.get("core_contract_id"),
                    core_access_meta.get("missing_portal_binding_doors"),
                )
        except LayoutTopologyError:
            raise
        except Exception:
            if topology_mode_l == "grid_growth":
                raise
            core_access_passed = False
            logger.warning("[CORE] Access diagnostics failed | floor=%s", floor_id, exc_info=True)

    if zt:
        repair_meta = repair_unreachable_doors(
            doors,
            walls,
            zt,
            zone_polys=zone_polys or None,
            required_adjacency=required_adjacency,
            door_width=0.9,
            min_door_width=0.8,
            allow_core_targets=(topology_mode_l != "grid_growth"),
            allow_core_hall_starts=bool(core_access_passed or topology_mode_l != "grid_growth"),
        )
        if int(repair_meta.get("added_doors", 0) or 0) > 0:
            logger.warning(
                "[VALIDATION] Rule: REACHABILITY_REPAIR | Result: APPLIED | AddedDoors=%d | Remaining=%s",
                int(repair_meta.get("added_doors", 0) or 0),
                repair_meta.get("unreachable_after", []),
            )
        try:
            validate_reachability(
                doors,
                zt,
                allow_core_hall_starts=bool(core_access_passed or topology_mode_l != "grid_growth"),
            )
        except LayoutTopologyError as exc:
            metadata = dict(getattr(exc, "metadata", {}) or {})
            metadata.setdefault("failure_kind", "reachability")
            metadata["door_fallback"] = repair_meta
            exc.metadata = metadata
            raise

    logger.info("[DOOR] Door generation complete | doors=%d | windows=%d", len(doors), len(windows))
    return PostprocessResult(walls=walls, doors=doors, windows=windows)


# ============================================================
# 序列化辅助
# ============================================================

def wall_to_dict(wall: WallSegment) -> dict:
    """WallSegment → 可序列化 dict"""
    coords = []
    if isinstance(wall.geometry, LineString):
        coords = [[round(x, 2), round(y, 2)] for x, y in wall.geometry.coords]
    elif isinstance(wall.geometry, MultiLineString):
        for line in wall.geometry.geoms:
            coords.extend([[round(x, 2), round(y, 2)] for x, y in line.coords])

    wall_polygon = []
    try:
        poly: Optional[Polygon] = None
        if isinstance(wall.geometry, Polygon) and not wall.geometry.is_empty:
            poly = wall.geometry
        elif isinstance(wall.geometry, MultiPolygon) and not wall.geometry.is_empty:
            poly = max(wall.geometry.geoms, key=lambda p: p.area, default=None)
        else:
            buffered = wall.geometry.buffer(
                wall.thickness / 2,
                cap_style=CAP_STYLE.flat,
                join_style=JOIN_STYLE.mitre,
            )
            if isinstance(buffered, Polygon) and not buffered.is_empty:
                poly = buffered
            elif isinstance(buffered, MultiPolygon) and not buffered.is_empty:
                poly = max(buffered.geoms, key=lambda p: p.area, default=None)
        if poly is not None and not poly.is_empty:
            wall_polygon = [[round(x, 2), round(y, 2)] for x, y in poly.exterior.coords]
    except Exception:
        pass

    return {
        "type": wall.type,
        "coords": coords,
        "polygon": wall_polygon,
        "thickness": wall.thickness,
        "length": round(wall.length, 2),
        "room_ids": wall.room_ids,
        "forward": list(wall.forward) if wall.forward is not None else None,
        "category": wall.category,
    }


def door_to_dict(door: DoorPlacement) -> dict:
    """DoorPlacement → 可序列化 dict"""
    out = {
        "position": list(door.position),
        "width": door.width,
        "connects": door.connects,
        "rotation": door.rotation,
        "wall_type": door.wall_type,
        "thickness": door.thickness,
        "forward": list(door.forward),
    }
    if getattr(door, "source_portal_spec_id", None):
        out["source_portal_spec_id"] = door.source_portal_spec_id
    return out


def window_to_dict(window: WindowPlacement) -> dict:
    """WindowPlacement → 可序列化 dict"""
    return {
        "position": list(window.position),
        "width": window.width,
        "room_id": window.room_id,
        "rotation": window.rotation,
        "thickness": window.thickness,
        "forward": list(window.forward),
    }
