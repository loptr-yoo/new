"""
postprocessor.py

后处理：根据房间 polygon 自动生成墙体、门、窗户。

借鉴 Co-Layout（AAAI 2026）思路：
求解器只负责房间分区，墙/门/窗由后处理启发式规则自动放置。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import (
    CAP_STYLE,
    GeometryCollection,
    JOIN_STYLE,
    LineString,
    LinearRing,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import snap

logger = logging.getLogger(__name__)


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
    forward: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class WindowPlacement:
    """窗户的放置"""
    position: Tuple[float, float]
    width: float  # 米
    room_id: str
    wall_length: float  # 所在墙段长度
    rotation: float = 0.0  # 0=水平墙, 90=垂直墙
    thickness: float = 0.24  # 米
    forward: Tuple[float, float, float] = (0.0, 0.0, 1.0)


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

    def _snap_room_rects(
        rects: Dict[str, Tuple[float, float, float, float]],
        tolerance: float,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        if tolerance <= 0 or len(rects) < 2:
            return rects
        poly_map: Dict[str, Polygon] = {}
        for zid, (x, y, w, h) in rects.items():
            if w <= 0 or h <= 0:
                continue
            poly_map[zid] = box(float(x), float(y), float(x + w), float(y + h))
        ids = list(poly_map.keys())
        for i in range(len(ids)):
            ida = ids[i]
            pa = poly_map[ida]
            for j in range(i + 1, len(ids)):
                idb = ids[j]
                pb = poly_map[idb]
                try:
                    pa = snap(pa, pb, tolerance)
                    pb = snap(pb, pa, tolerance)
                    if isinstance(pa, Polygon) and not pa.is_empty:
                        poly_map[ida] = pa
                    if isinstance(pb, Polygon) and not pb.is_empty:
                        poly_map[idb] = pb
                except Exception:
                    continue
        out: Dict[str, Tuple[float, float, float, float]] = dict(rects)
        for zid, p in poly_map.items():
            minx, miny, maxx, maxy = p.bounds
            out[zid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
        return out

    room_rects = _snap_room_rects(room_rects, tolerance=snap_tolerance)
    edge_set_aug: Dict[FrozenSet[str], str] = dict(edge_set)

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
        
        # 物理抹除：如果一边是电梯井，一边是电梯厅，跳过，不生成任何墙体
        if (t1 == "elevator_shaft" and t2 == "elevator_hall") or \
           (t1 == "elevator_hall" and t2 == "elevator_shaft"):
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
            if (y1 - y0) > min_wall_length:
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
            if (x1 - x0) > min_wall_length:
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
        for oid, (ox, oy, ow, oh) in room_rects.items():
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
        if _has_neighbor_below(top_y=top_y, x0=float(x), x1=float(x + w), self_id=zid):
            continue
        if float(w) <= min_wall_length:
            continue
        gap = max(0.0, inner_ymax - top_y)
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
        if hasattr(room, "polygon") and not room.polygon.is_empty:
            all_zones.append(("room", rid, room.polygon))
    for corridor in (corridors or []):
        if hasattr(corridor, "polygon") and not corridor.polygon.is_empty:
            all_zones.append(("corridor", corridor.id, corridor.polygon))
    if core_tube and hasattr(core_tube, "polygon") and not core_tube.polygon.is_empty:
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

    def _is_door_space(zt: str) -> bool:
        return zt in ("room", "staircase", "elevator_hall")

    def get_type(zid: str) -> str:
        zt_lower = zid.lower()
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
        return (0.0, 0.0, 1.0)

    def _door_forward(pos: Tuple[float, float], connects: List[str], rotation: float) -> Tuple[float, float, float]:
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
        return (float(dx), 0.0, float(dy))

    def _door_point(wall: WallSegment) -> Optional[Tuple[float, float]]:
        if wall.length < (door_width + 2 * margin):
            return None
        x0, y0 = wall.geometry.coords[0]
        x1, y1 = wall.geometry.coords[1]
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def _door_point_staircase_elevator_hall(
        wall: WallSegment,
        staircase_zid: str,
        elevator_hall_zid: str,
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        if zone_rects is None:
            p = _door_point(wall)
            return (p, float(door_width)) if p is not None else None
        stair = zone_rects.get(staircase_zid)
        hall = zone_rects.get(elevator_hall_zid)
        if stair is None or hall is None:
            p = _door_point(wall)
            return (p, float(door_width)) if p is not None else None
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

        half = float(door_width) / 2

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
                    p = _door_point(wall)
                    return (p, float(door_width)) if p is not None else None
            if seg1 <= seg0:
                p = _door_point(wall)
                return (p, float(door_width)) if p is not None else None
            avail = float(seg1 - seg0)
            door_w = float(min(float(door_width), avail))
            lo = float(seg0) + door_w / 2
            hi = float(seg1) - door_w / 2
            if hi < lo:
                p = _door_point(wall)
                return (p, float(door_width)) if p is not None else None
            y = lo if side == "min" else hi
            return ((x_const, float(y)), door_w)

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
                p = _door_point(wall)
                return (p, float(door_width)) if p is not None else None
        if seg1 <= seg0:
            p = _door_point(wall)
            return (p, float(door_width)) if p is not None else None
        avail = float(seg1 - seg0)
        door_w = float(min(float(door_width), avail))
        lo = float(seg0) + door_w / 2
        hi = float(seg1) - door_w / 2
        if hi < lo:
            p = _door_point(wall)
            return (p, float(door_width)) if p is not None else None
        x = lo if side == "min" else hi
        return ((float(x), y_const), door_w)

    def _wall_key(w: WallSegment) -> Tuple:
        coords = list(w.geometry.coords)
        a = (round(coords[0][0], 2), round(coords[0][1], 2))
        b = (round(coords[-1][0], 2), round(coords[-1][1], 2))
        p1, p2 = (a, b) if a <= b else (b, a)
        return (w.type, round(float(w.thickness), 3), p1, p2)

    candidates: List[WallSegment] = []

    for wall in walls:
        if wall.type != "partition_wall":
            continue
        if len(wall.room_ids) != 2:
            continue
        if wall.length < door_width:
            continue

        if zone_types is not None:
            a, b = wall.room_ids[0], wall.room_ids[1]
            type_a = get_type(a)
            type_b = get_type(b)
            if "elevator_shaft" in (type_a, type_b):
                continue
            legal = (
                ((type_a in ("room", "elevator_hall") and type_b == "corridor")) or
                ((type_b in ("room", "elevator_hall") and type_a == "corridor")) or
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
                    _cx, cy, _cw, _ch = zone_rects[core_id]
                    core_south = cy
                    x0, y0 = wall.geometry.coords[0]
                    x1, y1 = wall.geometry.coords[1]
                    if abs(y0 - y1) < abs(x0 - x1):
                        if abs(y0 - core_south) > 0.02:
                            continue
                    else:
                        continue

        candidates.append(wall)

    used: set = set()

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
        w_override: Optional[float] = None
        if (ta == "staircase" and tb == "elevator_hall") or (ta == "elevator_hall" and tb == "staircase"):
            stair_id = a if ta == "staircase" else b
            hall_id = a if ta == "elevator_hall" else b
            pack = _door_point_staircase_elevator_hall(chosen, stair_id, hall_id)
            if pack is not None:
                p, w_override = pack
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
            width=float(w_override) if w_override is not None else door_width,
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
        if zt == "staircase" and staircase_id is None:
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
            pack = _door_point_staircase_elevator_hall(chosen, staircase_id, elevator_hall_id)
            if pack is not None:
                p, w_override = pack
                key = _wall_key(chosen)
                if key not in used:
                    used.add(key)
                    rot = _wall_rotation(chosen)
                    connects = list(chosen.room_ids)
                    doors.append(DoorPlacement(
                        position=(round(p[0], 2), round(p[1], 2)),
                        width=float(w_override) if w_override is not None else door_width,
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
                forward=(float(fx), 0.0, float(fy)),
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
                    forward=(float(fx), 0.0, float(fy)),
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
                    forward=(float(fx), 0.0, float(fy)),
                ))

    return windows


# ============================================================
# 一站式后处理
# ============================================================

def postprocess_floor(
    rooms: list,
    floor_boundary: Polygon,
    corridors: Optional[list] = None,
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
    walls = generate_wall_mesh(
        rooms=rooms,
        corridors=corridors or [],
        core_tube=None,
        floor_boundary=floor_boundary,
    )
    doors = generate_doors(walls)
    room_rects: Dict[str, Tuple[float, float, float, float]] = {}
    zone_types: Dict[str, str] = {}
    rooms_needing_window: Set[str] = set()
    for r in rooms:
        rid = getattr(r, "id", getattr(r, "room_id", "?"))
        if not hasattr(r, "polygon") or r.polygon.is_empty:
            continue
        minx, miny, maxx, maxy = r.polygon.bounds
        room_rects[rid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
        zone_types[rid] = "room"
        has_window = getattr(r, "has_window", False) or getattr(r, "needs_window", False)
        if has_window:
            rooms_needing_window.add(rid)
    exterior_thickness = next((w.thickness for w in walls if w.type == "exterior_wall"), 0.24)
    windows = generate_windows_from_floor_boundary(
        room_rects=room_rects,
        zone_types=zone_types,
        rooms_needing_window=rooms_needing_window,
        floor_bounds=floor_boundary.bounds,
        exterior_thickness=float(exterior_thickness),
    )

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
