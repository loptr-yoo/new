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

logger = logging.getLogger(__name__)

try:
    from shapely.validation import make_valid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore[assignment]


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
                line = _extend_line(LineString([(shared_x, y0), (shared_x, y1)]), wall_thickness / 2)
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
                line = _extend_line(LineString([(x0, shared_y), (x1, shared_y)]), wall_thickness / 2)
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
        line = _extend_line(LineString([(float(x), top_y), (float(x + w), top_y)]), safe_thickness / 2)
        line = _clip_line_to_floor(line)
        walls.append(WallSegment(
            type="partition_wall",
            geometry=line,
            thickness=safe_thickness,
            room_ids=[zid],
        ))

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

    # 收集所有区域
    all_zones = []
    for room in rooms:
        rid = getattr(room, "id", getattr(room, "room_id", "?"))
        rtype = str(getattr(room, "room_type", getattr(room, "type", "")) or "").lower()
        if rtype == "void" or bool(getattr(room, "skip_solver", False)):
            continue
        if hasattr(room, "polygon") and not room.polygon.is_empty:
            all_zones.append(("room", rid, room.polygon))
    for corridor in (corridors or []):
        if hasattr(corridor, "polygon") and not corridor.polygon.is_empty:
            all_zones.append(("corridor", corridor.id, corridor.polygon))
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
            all_zones.append(("core", str(zid), poly))
            used = True
        if not used:
            all_zones.append(("core", "core_tube", core_tube.polygon))

    walls: List[WallSegment] = []
    walls.extend(_generate_exterior_wall_pieces(floor_boundary, exterior_thickness))

    # 内墙：所有区域两两比较
    for i in range(len(all_zones)):
        _, id_a, poly_a = all_zones[i]
        bound_a = poly_a.boundary
        for j in range(i + 1, len(all_zones)):
            _, id_b, poly_b = all_zones[j]
            try:
                bound_b = poly_b.boundary.buffer(boundary_tolerance)
                shared = bound_a.intersection(bound_b)
                if shared.is_empty:
                    continue
                for line in _extract_linestrings(shared):
                    if line.length > min_wall_length:
                        ext = _extend_line(line, wall_thickness / 2)
                        try:
                            clipped = ext.intersection(floor_boundary)
                        except Exception:
                            clipped = ext
                        candidates = _extract_linestrings(clipped)
                        if candidates:
                            ext = max(candidates, key=lambda s: s.length)
                        walls.append(WallSegment(
                            type="partition_wall",
                            geometry=ext,
                            thickness=wall_thickness,
                            room_ids=[id_a, id_b],
                        ))
            except Exception as e:
                logger.debug(f"Partition wall failed for {id_a}-{id_b}: {e}")

    return _dedup_walls(walls)


def _dedup_walls(walls: List[WallSegment]) -> List[WallSegment]:
    """按端点坐标去重，避免重复绘制导致颜色叠加变深（叠影）"""
    seen: set = set()
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
        if key not in seen:
            seen.add(key)
            result.append(w)
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
                lines.append(WallSegment(type=w.type, geometry=g, thickness=w.thickness, room_ids=list(w.room_ids)))
            except Exception:
                lines.append(w)
        else:
            others.append(w)

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
        snapped.append(WallSegment(type=w.type, geometry=g, thickness=w.thickness, room_ids=list(w.room_ids)))

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

    return _dedup_walls(pruned)


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


# ============================================================
# 门的放置
# ============================================================

def generate_doors(
    walls: List[WallSegment],
    zone_types: Optional[Dict[str, str]] = None,
    zone_rects: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    door_width: float = 0.9,
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

    dummy_prefix = "room_dummy_"

    def _is_door_space(zt: str) -> bool:
        return zt in ("room", "staircase", "staircase_hall", "elevator_hall")

    def get_type(zid: str) -> str:
        zt_lower = zid.lower()
        if zt_lower.startswith(dummy_prefix):
            return "dummy"
        if zone_types is not None and zid in zone_types:
            return zone_types.get(zid, "room")
        if "corridor" in zt_lower:
            return "corridor"
        if zid == "core_tube" or zt_lower == "core_tube":
            return "core"
        return "room"

    margin = 0.2

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

    def _door_forward(_pos: Tuple[float, float], connects: List[str], rotation: float) -> Tuple[float, float, float]:
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

    def _door_point_with_width(wall: WallSegment, w: float) -> Optional[Tuple[float, float]]:
        if wall.length < (float(w) + 2 * margin):
            return None
        x0, y0 = wall.geometry.coords[0]
        x1, y1 = wall.geometry.coords[1]
        return ((x0 + x1) / 2, (y0 + y1) / 2)

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
        x1, y1 = wall.geometry.coords[1]

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
        x1, y1 = wall.geometry.coords[1]

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

    for wall in walls:
        if wall.type != "partition_wall":
            continue
        if len(wall.room_ids) != 2:
            continue

        if zone_types is not None:
            a, b = wall.room_ids[0], wall.room_ids[1]
            type_a = get_type(a)
            type_b = get_type(b)
            if "dummy" in (type_a, type_b):
                other = type_b if type_a == "dummy" else type_a
                if other != "corridor":
                    continue
                min_len = max(1.0, float(door_width))
                if wall.length < min_len:
                    continue
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
            legal = (
                ((type_a in ("room", "elevator_hall", "dummy") and type_b == "corridor")) or
                ((type_b in ("room", "elevator_hall", "dummy") and type_a == "corridor")) or
                (type_a == "staircase_hall" and type_b == "elevator_hall") or
                (type_b == "staircase_hall" and type_a == "elevator_hall") or
                (type_a == "staircase" and type_b == "elevator_hall") or
                (type_b == "staircase" and type_a == "elevator_hall") or
                (type_a == "corridor" and type_b == "core") or
                (type_b == "corridor" and type_a == "core") or
                (type_a == "room" and type_b == "room")
            )
            if not legal:
                continue
            if "core" in (type_a, type_b) and zone_rects is not None:
                core_id = a if type_a == "core" else (b if type_b == "core" else None)
                if core_id and core_id in zone_rects:
                    core_south = float(zone_rects[core_id][1])
                    x0, y0 = wall.geometry.coords[0]
                    x1, y1 = wall.geometry.coords[1]
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
                forward=_door_forward(p, connects, rot),
            ))
        return doors

    room_ids = [zid for zid, zt in zone_types.items() if _is_door_space(zt)]
    for rid in room_ids:
        room_candidates = [w for w in candidates if rid in w.room_ids]
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
                forward=_door_forward(p, connects, rot),
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
            forward=_door_forward(p, connects, rot),
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
                    forward=_door_forward(p, connects, rot),
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
                                forward=_door_forward(p, connects, rot),
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
                            forward=_door_forward(p, connects, rot),
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
        if zone_types.get(rid) != "room":
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
    is_ground_floor: bool = False,
    walls: Optional[List[WallSegment]] = None,
    zone_types: Optional[Dict[str, str]] = None,
    zone_rects: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
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
            core_tube=None,
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
            zt[rid] = zt.get(rid, "room")
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

    partition_thickness = next((float(w.thickness) for w in walls if w.type == "partition_wall"), 0.12)
    void_walls: List[WallSegment] = []
    for r in rooms:
        rid = getattr(r, "id", getattr(r, "room_id", "?"))
        rtype = str(getattr(r, "room_type", getattr(r, "type", "")) or "").lower()
        if rtype != "void" and (not bool(getattr(r, "skip_solver", False))):
            continue
        poly = getattr(r, "polygon", None)
        if poly is None or poly.is_empty:
            continue
        try:
            ring = poly.exterior
        except Exception:
            continue
        for seg in _extract_linestrings(ring):
            minx, miny, maxx, maxy = seg.bounds
            if (abs(float(minx) - float(maxx)) <= 1e-6) and (
                abs(float(minx) - float(fminx)) <= 0.02 or abs(float(minx) - float(fmaxx)) <= 0.02
            ):
                continue
            if (abs(float(miny) - float(maxy)) <= 1e-6) and (
                abs(float(miny) - float(fminy)) <= 0.02 or abs(float(miny) - float(fmaxy)) <= 0.02
            ):
                continue
            if seg.length <= 0.05:
                continue
            void_walls.append(WallSegment(
                type="partition_wall",
                geometry=seg,
                thickness=partition_thickness,
                room_ids=[str(rid)],
            ))
    if void_walls:
        walls = list(walls) + void_walls
    walls = _normalize_walls(walls, floor_bounds=(fminx, fminy, fmaxx, fmaxy), zone_rects=rr)

    doors = generate_doors(walls, zone_types=zt or None, zone_rects=rr or None)
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

            corridor_ids = [rid for rid, t in zt.items() if t == "corridor" and rid in rr]

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

    if corridors:
        corridor_cover_thickness = next((float(w.thickness) for w in walls if w.type == "partition_wall"), 0.12)

        def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            if not intervals:
                return []
            items = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
            out: List[Tuple[float, float]] = []
            cur_a, cur_b = items[0]
            for a, b in items[1:]:
                if a <= cur_b + 1e-3:
                    cur_b = max(cur_b, b)
                else:
                    out.append((cur_a, cur_b))
                    cur_a, cur_b = a, b
            out.append((cur_a, cur_b))
            return out

        def _append_projected_intervals(geom: BaseGeometry, orientation: str, a0: float, a1: float, out: List[Tuple[float, float]]) -> None:
            if geom.is_empty:
                return
            if isinstance(geom, (GeometryCollection, MultiPolygon, MultiLineString)):
                for g in geom.geoms:
                    _append_projected_intervals(g, orientation, a0, a1, out)
                return
            try:
                minx, miny, maxx, maxy = geom.bounds
            except Exception:
                return
            if orientation == "horizontal":
                a = max(float(a0), float(minx))
                b = min(float(a1), float(maxx))
            else:
                a = max(float(a0), float(miny))
                b = min(float(a1), float(maxy))
            if b - a > 1e-3:
                out.append((a, b))

        def _wall_surface_geometry(wall: WallSegment) -> Optional[BaseGeometry]:
            geom = wall.geometry
            if geom is None or geom.is_empty:
                return None
            if isinstance(geom, (Polygon, MultiPolygon)):
                return geom
            try:
                return geom.buffer(
                    float(wall.thickness) / 2.0,
                    cap_style=CAP_STYLE.flat,
                    join_style=JOIN_STYLE.mitre,
                )
            except Exception:
                return None

        def _cover_intervals_horizontal(corridor_id: str, edge_line: LineString, x0: float, x1: float, thickness: float) -> List[Tuple[float, float]]:
            intervals: List[Tuple[float, float]] = []
            eps = max(1e-3, float(thickness) * 0.1)
            try:
                edge_buffer = edge_line.buffer(float(thickness) / 2.0 + eps, cap_style=CAP_STYLE.flat)
            except Exception:
                return intervals
            for w in walls:
                if w.type not in ("partition_wall", "exterior_wall"):
                    continue
                if w.type == "partition_wall" and corridor_id not in (w.room_ids or []):
                    continue
                surf = _wall_surface_geometry(w)
                if surf is None or surf.is_empty:
                    continue
                try:
                    hit = surf.intersection(edge_buffer)
                except Exception:
                    continue
                _append_projected_intervals(hit, "horizontal", x0, x1, intervals)
            return _merge_intervals(intervals)

        def _cover_intervals_vertical(corridor_id: str, edge_line: LineString, y0: float, y1: float, thickness: float) -> List[Tuple[float, float]]:
            intervals: List[Tuple[float, float]] = []
            eps = max(1e-3, float(thickness) * 0.1)
            try:
                edge_buffer = edge_line.buffer(float(thickness) / 2.0 + eps, cap_style=CAP_STYLE.flat)
            except Exception:
                return intervals
            for w in walls:
                if w.type not in ("partition_wall", "exterior_wall"):
                    continue
                if w.type == "partition_wall" and corridor_id not in (w.room_ids or []):
                    continue
                surf = _wall_surface_geometry(w)
                if surf is None or surf.is_empty:
                    continue
                try:
                    hit = surf.intersection(edge_buffer)
                except Exception:
                    continue
                _append_projected_intervals(hit, "vertical", y0, y1, intervals)
            return _merge_intervals(intervals)

        def _fill_gaps_1d_horizontal(corridor_id: str, edge_line: LineString, y_line: float, x0: float, x1: float, thickness: float) -> None:
            covered = _cover_intervals_horizontal(corridor_id, edge_line, x0, x1, thickness)
            cur = float(x0)
            for a, b in covered:
                if a > cur + 1e-3:
                    if float(a - cur) > 0.1:
                        walls.append(WallSegment(
                            type="partition_wall",
                            geometry=LineString([(float(cur), float(y_line)), (float(a), float(y_line))]),
                            thickness=float(thickness),
                            room_ids=[str(corridor_id)],
                        ))
                cur = max(cur, b)
            if float(x1 - cur) > 0.1:
                walls.append(WallSegment(
                    type="partition_wall",
                    geometry=LineString([(float(cur), float(y_line)), (float(x1), float(y_line))]),
                    thickness=float(thickness),
                    room_ids=[str(corridor_id)],
                ))

        def _fill_gaps_1d_vertical(corridor_id: str, edge_line: LineString, x_line: float, y0: float, y1: float, thickness: float) -> None:
            covered = _cover_intervals_vertical(corridor_id, edge_line, y0, y1, thickness)
            cur = float(y0)
            for a, b in covered:
                if a > cur + 1e-3:
                    if float(a - cur) > 0.1:
                        walls.append(WallSegment(
                            type="partition_wall",
                            geometry=LineString([(float(x_line), float(cur)), (float(x_line), float(a))]),
                            thickness=float(thickness),
                            room_ids=[str(corridor_id)],
                        ))
                cur = max(cur, b)
            if float(y1 - cur) > 0.1:
                walls.append(WallSegment(
                    type="partition_wall",
                    geometry=LineString([(float(x_line), float(cur)), (float(x_line), float(y1))]),
                    thickness=float(thickness),
                    room_ids=[str(corridor_id)],
                ))

        def _outward_normal_for_axis_edge(edge_line: LineString, corridor_poly: Polygon, thickness: float) -> Optional[Tuple[float, float]]:
            coords = list(edge_line.coords)
            if len(coords) < 2:
                return None
            (x0, y0), (x1, y1) = coords[0], coords[-1]
            dx = abs(float(x1) - float(x0))
            dy = abs(float(y1) - float(y0))
            probe_dist = max(float(thickness) * 0.1, 0.005)
            mid_pt = edge_line.interpolate(0.5, normalized=True)
            if dy < 1e-3:
                candidates = [(0.0, 1.0), (0.0, -1.0)]
            elif dx < 1e-3:
                candidates = [(1.0, 0.0), (-1.0, 0.0)]
            else:
                logger.warning("Skipping non-axis-aligned corridor boundary segment: %s", edge_line)
                return None

            probes = [
                Point(float(mid_pt.x) + nx * probe_dist, float(mid_pt.y) + ny * probe_dist)
                for nx, ny in candidates
            ]
            inside = [bool(corridor_poly.contains(p)) for p in probes]
            if inside[0] != inside[1]:
                return candidates[1] if inside[0] else candidates[0]

            try:
                inner = corridor_poly.buffer(-probe_dist * 0.5)
            except Exception:
                inner = None
            if inner is not None and not inner.is_empty:
                inner_inside = [bool(inner.contains(p)) for p in probes]
                if inner_inside[0] != inner_inside[1]:
                    return candidates[1] if inner_inside[0] else candidates[0]

            logger.warning("Could not determine outward normal for corridor boundary segment: %s", edge_line)
            return None

        for c in corridors or []:
            corridor_id = getattr(c, "id", None) if not isinstance(c, dict) else c.get("id")
            poly0 = getattr(c, "polygon", None) if not isinstance(c, dict) else c.get("polygon")
            if not corridor_id:
                continue
            try:
                if isinstance(poly0, Polygon):
                    corridor_poly = poly0
                elif isinstance(poly0, list):
                    corridor_poly = Polygon(poly0)
                else:
                    continue
                if corridor_poly.is_empty:
                    continue
            except Exception:
                continue

            coords = list(corridor_poly.exterior.coords)
            for p0, p1 in zip(coords, coords[1:]):
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                edge_line = LineString([(x0, y0), (x1, y1)])
                normal = _outward_normal_for_axis_edge(edge_line, corridor_poly, float(corridor_cover_thickness))
                if normal is None:
                    continue
                nx, ny = normal
                if dy < 1e-3:
                    x_start, x_end = sorted((x0, x1))
                    if x_end - x_start <= 1e-3:
                        continue
                    y_line = y0 + ny * float(corridor_cover_thickness) / 2.0
                    _fill_gaps_1d_horizontal(
                        str(corridor_id),
                        edge_line,
                        float(y_line),
                        float(x_start),
                        float(x_end),
                        float(corridor_cover_thickness),
                    )
                elif dx < 1e-3:
                    y_start, y_end = sorted((y0, y1))
                    if y_end - y_start <= 1e-3:
                        continue
                    x_line = x0 + nx * float(corridor_cover_thickness) / 2.0
                    _fill_gaps_1d_vertical(
                        str(corridor_id),
                        edge_line,
                        float(x_line),
                        float(y_start),
                        float(y_end),
                        float(corridor_cover_thickness),
                    )
                else:
                    logger.warning("Skipping non-axis-aligned corridor boundary segment: %s", edge_line)

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
    }


def door_to_dict(door: DoorPlacement) -> dict:
    """DoorPlacement → 可序列化 dict"""
    return {
        "position": list(door.position),
        "width": door.width,
        "connects": door.connects,
        "rotation": door.rotation,
        "wall_type": door.wall_type,
        "thickness": door.thickness,
        "forward": list(door.forward),
    }


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
