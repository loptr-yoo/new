from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union

from .room_spec import ZoneType

logger = logging.getLogger(__name__)

try:
    from shapely.validation import make_valid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore[assignment]

try:
    from shapely import set_precision  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    set_precision = None  # type: ignore[assignment]


def _largest_polygon(geom: BaseGeometry) -> Optional[Polygon]:
    try:
        polys = _as_polygons(geom)
    except Exception:
        return None
    if not polys:
        return None
    return max(polys, key=lambda p: float(p.area), default=None)


def _is_axis_aligned_polygon(poly: Polygon, *, tol: float = 1e-6) -> bool:
    try:
        coords = list(poly.exterior.coords)
    except Exception:
        return False
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        dx = abs(float(x1) - float(x0))
        dy = abs(float(y1) - float(y0))
        if dx > float(tol) and dy > float(tol):
            return False
    return True


def _orthogonalize_polygon(poly: Polygon, *, tol: float = 1e-6) -> Polygon:
    if poly is None or poly.is_empty:
        return poly
    if _is_axis_aligned_polygon(poly, tol=tol):
        return poly

    src_area = float(poly.area)

    def _build(use_first: bool) -> Optional[Polygon]:
        try:
            coords = list(poly.exterior.coords)
        except Exception:
            return None
        if not coords:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        out: List[Tuple[float, float]] = [(float(coords[0][0]), float(coords[0][1]))]
        changed = False
        for cur in coords[1:]:
            prev = out[-1]
            x0, y0 = float(prev[0]), float(prev[1])
            x1, y1 = float(cur[0]), float(cur[1])
            if abs(x1 - x0) > float(tol) and abs(y1 - y0) > float(tol):
                changed = True
                corner = (x0, y1) if use_first else (x1, y0)
                if abs(float(corner[0]) - x0) > 1e-12 or abs(float(corner[1]) - y0) > 1e-12:
                    out.append((float(corner[0]), float(corner[1])))
            out.append((x1, y1))

        if not changed:
            return None

        dedup: List[Tuple[float, float]] = []
        for p in out:
            if not dedup or (abs(dedup[-1][0] - p[0]) > 1e-12 or abs(dedup[-1][1] - p[1]) > 1e-12):
                dedup.append(p)
        if len(dedup) >= 2 and dedup[0] != dedup[-1]:
            dedup.append(dedup[0])
        if len(dedup) < 4:
            return None

        cand: BaseGeometry = Polygon(dedup)
        if (not getattr(cand, "is_valid", True)) and make_valid is not None:
            try:
                cand = make_valid(cand)  # type: ignore[assignment]
            except Exception:
                pass
        if not isinstance(cand, Polygon):
            cand2 = _largest_polygon(cand)
            if cand2 is None:
                return None
            cand = cand2
        try:
            cand = cand.buffer(0)
        except Exception:
            pass
        if not isinstance(cand, Polygon) or cand.is_empty:
            return None
        if not _is_axis_aligned_polygon(cand, tol=tol):
            return None
        return cand

    a = _build(True)
    b = _build(False)
    options = [p for p in [a, b] if p is not None and (not p.is_empty)]
    if not options:
        return poly
    return min(options, key=lambda p: abs(float(p.area) - src_area))


def _safe_snap_polygon_like(geom: BaseGeometry, *, tol: float) -> Optional[Polygon]:
    """
    将几何做轻量级“量子化 + 修复”，用于入口并入走廊后的正交清理。

    设计意图（中文说明）：
    - union() 之后常出现 1e-12 量级的毛刺/针眼；
    - 下游墙/门提取对轴对齐非常敏感，这里优先把边界坐标吸附到 tol 网格上。
    """
    raw_poly = geom if isinstance(geom, Polygon) else _largest_polygon(geom)
    raw_axis_ok = bool(raw_poly is not None and (not raw_poly.is_empty) and _is_axis_aligned_polygon(raw_poly))

    g2: BaseGeometry = geom
    if set_precision is not None:
        try:
            g2 = set_precision(g2, float(tol))
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

    poly = g2 if isinstance(g2, Polygon) else _largest_polygon(g2)
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
        poly2 = fixed if isinstance(fixed, Polygon) else _largest_polygon(fixed)
        if poly2 is None or poly2.is_empty:
            return None
        poly = poly2

    if float(poly.area) < 1e-4:
        return None

    if (not _is_axis_aligned_polygon(poly)) and raw_axis_ok and raw_poly is not None:
        return raw_poly

    if not _is_axis_aligned_polygon(poly):
        try:
            coords = [(round(float(x) / float(tol)) * float(tol), round(float(y) / float(tol)) * float(tol)) for x, y in poly.exterior.coords]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            q = Polygon(coords)
            if (not q.is_valid) and make_valid is not None:
                try:
                    q2 = make_valid(q)
                except Exception:
                    q2 = q
                q = q2 if isinstance(q2, Polygon) else (_largest_polygon(q2) or q)
            if q is not None and (not q.is_empty) and _is_axis_aligned_polygon(q):
                return q
        except Exception:
            pass

    return poly

def _as_polygons(geom) -> List[Polygon]:
    """从几何对象中提取所有 Polygon"""
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




# ═══════════════════════════════════════════════════════════════════════════
# 矩形拓扑生成器
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CoreTube:
    """
    核心筒定义

    设计原则：
    - 紧凑矩形，包含电梯、楼梯、设备间
    - 占楼层面积 5-10%
    - 位置靠近中心或入口
    - 自动拆分为 elevator (60-65%) + staircase (35-40%) 两个紧邻子矩形
    """
    polygon: Polygon  # 整体外轮廓（用于拓扑扣除）
    center: Tuple[float, float]
    width: float
    depth: float
    # 子区域
    elevator: Optional[object] = None
    staircase: Optional[Polygon] = None
    staircase_hall: Optional[Polygon] = None
    staircase_shaft: Optional[Polygon] = None
    elevator_hall: Optional[Polygon] = None
    elevator_shaft: Optional[Polygon] = None
    elevator_hall_b: Optional[Polygon] = None
    staircase_hall_b: Optional[Polygon] = None
    opening_sides: List[str] = field(default_factory=lambda: ["south"])
    elevator_area: float = 0.0
    staircase_area: float = 0.0
    staircase_hall_area: float = 0.0
    staircase_shaft_area: float = 0.0
    elevator_hall_area: float = 0.0
    elevator_shaft_area: float = 0.0

    def set_opening_sides(self, opening_sides: List[str]) -> None:
        sides = [str(s).lower() for s in (opening_sides or []) if str(s).strip()]
        if not sides:
            sides = ["south"]
        seen = set()
        ordered: List[str] = []
        for s in sides:
            if s in ("north", "south", "east", "west") and s not in seen:
                seen.add(s)
                ordered.append(s)
        if not ordered:
            ordered = ["south"]
        self.opening_sides = ordered
        self.build_subzones_from_bounds()

    def build_subzones_from_bounds(self) -> None:
        minx, miny, maxx, maxy = self.polygon.bounds
        w = maxx - minx
        h = maxy - miny
        if w <= 0 or h <= 0:
            self.staircase = None
            self.staircase_hall = None
            self.staircase_shaft = None
            self.elevator_hall = None
            self.elevator_shaft = None
            self.elevator_hall_b = None
            self.staircase_hall_b = None
            self.elevator = None
            self.staircase_area = 0.0
            self.staircase_hall_area = 0.0
            self.staircase_shaft_area = 0.0
            self.elevator_hall_area = 0.0
            self.elevator_shaft_area = 0.0
            self.elevator_area = 0.0
            return

        self.elevator_hall_b = None
        self.staircase_hall_b = None

        sides = [str(s).lower() for s in (self.opening_sides or ["south"])]
        if len(sides) >= 2:
            sset = set(sides[:2])
        else:
            sset = set(sides)

        split_x = minx + 0.4 * w
        stair_split_y = miny + 0.3 * h

        if sset == {"west", "east"}:
            band = 0.25 * w
            band = max(0.8, min(band, 0.45 * w))
            cx0 = minx + band
            cx1 = maxx - band
            if cx1 <= cx0 + 0.2:
                sset = {sides[0]}
            else:
                stair_hall_w = box(minx, miny, minx + band, stair_split_y)
                elev_hall_w = box(minx, stair_split_y, minx + band, maxy)
                stair_hall_e = box(maxx - band, miny, maxx, stair_split_y)
                elev_hall_e = box(maxx - band, stair_split_y, maxx, maxy)

                staircase_hall = stair_hall_w
                self.staircase_hall_b = stair_hall_e
                elevator_hall = elev_hall_w
                self.elevator_hall_b = elev_hall_e

                staircase_shaft = box(cx0, miny, cx1, stair_split_y)
                elevator_shaft = box(cx0, stair_split_y, cx1, maxy)

                try:
                    staircase = unary_union([staircase_hall, staircase_shaft])
                except Exception:
                    staircase = staircase_hall
                try:
                    elevator = unary_union([elevator_hall, self.elevator_hall_b, elevator_shaft])
                except Exception:
                    elevator = elevator_hall

                self.staircase = staircase if isinstance(staircase, Polygon) else staircase_hall
                self.staircase_hall = staircase_hall
                self.staircase_shaft = staircase_shaft
                self.elevator_hall = elevator_hall
                self.elevator_shaft = elevator_shaft
                self.staircase_area = float(self.staircase.area) if self.staircase is not None else 0.0
                self.staircase_hall_area = float(staircase_hall.area)
                self.staircase_shaft_area = float(staircase_shaft.area)
                self.elevator_hall_area = float(elevator_hall.area) + float(self.elevator_hall_b.area)
                self.elevator_shaft_area = float(elevator_shaft.area)
                self.elevator = elevator
                try:
                    self.elevator_area = float(elevator.area)
                except Exception:
                    self.elevator_area = float(elevator_hall.area)
                return

        if sset == {"south", "north"}:
            band = 0.25 * h
            band = max(0.8, min(band, 0.45 * h))
            cy0 = miny + band
            cy1 = maxy - band
            if cy1 <= cy0 + 0.2:
                sset = {sides[0]}
            else:
                stair_hall_s = box(minx, miny, split_x, miny + band)
                elev_hall_s = box(split_x, miny, maxx, miny + band)
                stair_hall_n = box(minx, maxy - band, split_x, maxy)
                elev_hall_n = box(split_x, maxy - band, maxx, maxy)

                staircase_hall = stair_hall_s
                self.staircase_hall_b = stair_hall_n
                elevator_hall = elev_hall_s
                self.elevator_hall_b = elev_hall_n

                staircase_shaft = box(minx, cy0, split_x, cy1)
                elevator_shaft = box(split_x, cy0, maxx, cy1)

                try:
                    staircase = unary_union([staircase_hall, self.staircase_hall_b, staircase_shaft])
                except Exception:
                    staircase = staircase_hall
                try:
                    elevator = unary_union([elevator_hall, self.elevator_hall_b, elevator_shaft])
                except Exception:
                    elevator = elevator_hall

                self.staircase = staircase if isinstance(staircase, Polygon) else staircase_hall
                self.staircase_hall = staircase_hall
                self.staircase_shaft = staircase_shaft
                self.elevator_hall = elevator_hall
                self.elevator_shaft = elevator_shaft
                self.staircase_area = float(self.staircase.area) if self.staircase is not None else 0.0
                self.staircase_hall_area = float(staircase_hall.area) + float(self.staircase_hall_b.area)
                self.staircase_shaft_area = float(staircase_shaft.area)
                self.elevator_hall_area = float(elevator_hall.area) + float(self.elevator_hall_b.area)
                self.elevator_shaft_area = float(elevator_shaft.area)
                self.elevator = elevator
                try:
                    self.elevator_area = float(elevator.area)
                except Exception:
                    self.elevator_area = float(elevator_hall.area)
                return

        side = next(iter(sset or {"south"}))
        if side in ("south", "north"):
            band = 0.35 * h
            band = max(0.8, min(band, 0.6 * h))
            if side == "south":
                hall_y0, hall_y1 = miny, miny + band
                shaft_y0, shaft_y1 = miny + band, maxy
            else:
                hall_y0, hall_y1 = maxy - band, maxy
                shaft_y0, shaft_y1 = miny, maxy - band

            staircase_hall = box(minx, hall_y0, split_x, hall_y1)
            elevator_hall = box(split_x, hall_y0, maxx, hall_y1)
            staircase_shaft = box(minx, shaft_y0, split_x, shaft_y1)
            elevator_shaft = box(split_x, shaft_y0, maxx, shaft_y1)
        else:
            band = 0.35 * w
            band = max(0.8, min(band, 0.6 * w))
            if side == "west":
                hall_x0, hall_x1 = minx, minx + band
                shaft_x0, shaft_x1 = minx + band, maxx
            else:
                hall_x0, hall_x1 = maxx - band, maxx
                shaft_x0, shaft_x1 = minx, maxx - band

            staircase_hall = box(hall_x0, miny, hall_x1, stair_split_y)
            elevator_hall = box(hall_x0, stair_split_y, hall_x1, maxy)
            staircase_shaft = box(shaft_x0, miny, shaft_x1, stair_split_y)
            elevator_shaft = box(shaft_x0, stair_split_y, shaft_x1, maxy)

        try:
            merged_s = unary_union([staircase_hall, staircase_shaft])
            staircase = merged_s if isinstance(merged_s, Polygon) else staircase_hall
        except Exception:
            staircase = staircase_hall

        self.staircase = staircase
        self.staircase_hall = staircase_hall
        self.staircase_shaft = staircase_shaft
        self.elevator_hall = elevator_hall
        self.elevator_shaft = elevator_shaft
        self.staircase_area = float(staircase.area)
        self.staircase_hall_area = float(staircase_hall.area)
        self.staircase_shaft_area = float(staircase_shaft.area)
        self.elevator_hall_area = float(elevator_hall.area)
        self.elevator_shaft_area = float(elevator_shaft.area)
        try:
            merged = unary_union([elevator_hall, elevator_shaft])
            self.elevator = merged
            self.elevator_area = float(merged.area)
        except Exception:
            self.elevator = elevator_hall
            self.elevator_area = float(elevator_hall.area)

    def translate(self, dx: float = 0.0, dy: float = 0.0) -> None:
        if abs(float(dx)) <= 1e-9 and abs(float(dy)) <= 1e-9:
            return
        self.polygon = affinity.translate(self.polygon, xoff=float(dx), yoff=float(dy))
        cx, cy = self.center
        self.center = (float(cx) + float(dx), float(cy) + float(dy))
        self.build_subzones_from_bounds()

    @classmethod
    def create(
        cls,
        center: Tuple[float, float],
        width: float,
        depth: float,
        elevator_ratio: float = 0.62,
    ) -> CoreTube:
        """创建矩形核心筒，自动拆分 staircase + elevator_hall + elevator_shaft"""
        cx, cy = center
        polygon = box(
            cx - width / 2, cy - depth / 2,
            cx + width / 2, cy + depth / 2,
        )
        core = cls(polygon=polygon, center=center, width=width, depth=depth)
        core.build_subzones_from_bounds()
        return core

    @classmethod
    def create_for_floor(
        cls,
        floor_bounds: Tuple[float, float, float, float],
        area_ratio: float = 0.08,
        aspect_ratio: float = 1.0,
        position: str = "north",
        grid_alignment: float = 0.5,
    ) -> CoreTube:
        """根据楼层自动创建核心筒"""
        x_min, y_min, x_max, y_max = floor_bounds
        floor_width = x_max - x_min
        floor_depth = y_max - y_min
        floor_area = (x_max - x_min) * (y_max - y_min)

        core_area = floor_area * area_ratio
        width = np.sqrt(core_area * aspect_ratio)
        depth = core_area / width

        width = max(grid_alignment, round(width / grid_alignment) * grid_alignment)
        depth = max(grid_alignment, round(depth / grid_alignment) * grid_alignment)

        max_width = max(grid_alignment, floor_width - 2 * grid_alignment)
        max_depth = max(grid_alignment, floor_depth - 2 * grid_alignment)
        width = min(width, max_width)
        depth = min(depth, max_depth)

        pos = str(position or "north").lower()

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        if pos == "north":
            cy = y_max - depth / 2
        elif pos == "south":
            cy = y_min + depth / 2
        elif pos == "east":
            cx = x_max - width / 2
        elif pos == "west":
            cx = x_min + width / 2
        elif pos == "center":
            pass
        elif pos == "entrance":
            cy = y_min + depth / 2 + 3
        else:
            cy = y_max - depth / 2

        cx = round(cx / grid_alignment) * grid_alignment
        cx = min(max(cx, x_min + width / 2), x_max - width / 2)

        if pos in ("north",):
            cy = np.floor(cy / grid_alignment) * grid_alignment
        elif pos in ("south",):
            cy = np.ceil(cy / grid_alignment) * grid_alignment
        else:
            cy = round(cy / grid_alignment) * grid_alignment
        cy = min(max(cy, y_min + depth / 2), y_max - depth / 2)

        core = cls.create((cx, cy), width, depth)
        try:
            _, y0, _, y1 = (float(v) for v in floor_bounds)
            _, core_miny, _, core_maxy = (float(v) for v in core.polygon.bounds)
            x0, _, x1, _ = (float(v) for v in floor_bounds)
            core_minx, _, core_maxx, _ = (float(v) for v in core.polygon.bounds)
            exterior_thickness = 0.24
            max_snap = float(grid_alignment) + 0.05
            if pos == "north":
                dy = (y1 - exterior_thickness) - core_maxy
                if (abs(float(dy)) > 1e-6) and (abs(float(dy)) <= max_snap):
                    core.translate(dy=float(dy))
            elif pos == "south":
                dy = (y0 + exterior_thickness) - core_miny
                if (abs(float(dy)) > 1e-6) and (abs(float(dy)) <= max_snap):
                    core.translate(dy=float(dy))
            elif pos == "east":
                dx = (x1 - exterior_thickness) - core_maxx
                if (abs(float(dx)) > 1e-6) and (abs(float(dx)) <= max_snap):
                    core.translate(dx=float(dx))
            elif pos == "west":
                dx = (x0 + exterior_thickness) - core_minx
                if (abs(float(dx)) > 1e-6) and (abs(float(dx)) <= max_snap):
                    core.translate(dx=float(dx))
        except Exception:
            pass

        # 边界安全：确保子区域不超出楼层
        floor_poly = box(x_min, y_min, x_max, y_max)
        if core.staircase is not None and not floor_poly.contains(core.staircase):
            logger.warning("Staircase extends beyond floor boundary, shrinking")
            clipped = core.staircase.intersection(floor_poly)
            if not clipped.is_empty and isinstance(clipped, Polygon):
                core.staircase = clipped
                core.staircase_area = float(clipped.area)
            else:
                core.staircase = None
                core.staircase_area = 0.0
        if core.staircase_hall is not None and not floor_poly.contains(core.staircase_hall):
            logger.warning("Staircase hall extends beyond floor boundary, shrinking")
            clipped = core.staircase_hall.intersection(floor_poly)
            if not clipped.is_empty and isinstance(clipped, Polygon):
                core.staircase_hall = clipped
                core.staircase_hall_area = float(clipped.area)
            else:
                core.staircase_hall = None
                core.staircase_hall_area = 0.0
        if core.staircase_shaft is not None and not floor_poly.contains(core.staircase_shaft):
            logger.warning("Staircase shaft extends beyond floor boundary, shrinking")
            clipped = core.staircase_shaft.intersection(floor_poly)
            if not clipped.is_empty and isinstance(clipped, Polygon):
                core.staircase_shaft = clipped
                core.staircase_shaft_area = float(clipped.area)
            else:
                core.staircase_shaft = None
                core.staircase_shaft_area = 0.0
        if core.elevator_hall is not None and not floor_poly.contains(core.elevator_hall):
            logger.warning("Elevator hall extends beyond floor boundary, shrinking")
            clipped = core.elevator_hall.intersection(floor_poly)
            if not clipped.is_empty and isinstance(clipped, Polygon):
                core.elevator_hall = clipped
                core.elevator_hall_area = float(clipped.area)
            else:
                core.elevator_hall = None
                core.elevator_hall_area = 0.0
        if core.elevator_shaft is not None and not floor_poly.contains(core.elevator_shaft):
            logger.warning("Elevator shaft extends beyond floor boundary, shrinking")
            clipped = core.elevator_shaft.intersection(floor_poly)
            if not clipped.is_empty and isinstance(clipped, Polygon):
                core.elevator_shaft = clipped
                core.elevator_shaft_area = float(clipped.area)
            else:
                core.elevator_shaft = None
                core.elevator_shaft_area = 0.0
        try:
            if core.elevator_hall is not None and core.elevator_shaft is not None:
                merged = unary_union([core.elevator_hall, core.elevator_shaft])
                core.elevator = merged
                core.elevator_area = float(merged.area)
        except Exception:
            pass

        try:
            parts = [p for p in (core.staircase_hall, core.staircase_shaft) if p is not None and not p.is_empty]
            if parts:
                merged_s = unary_union(parts)
                if isinstance(merged_s, Polygon) and not merged_s.is_empty:
                    core.staircase = merged_s
                    core.staircase_area = float(merged_s.area)
        except Exception:
            pass

        return core


@dataclass
class Corridor:
    """走廊定义"""
    id: str
    centerline: LineString
    width: float
    orientation: str  # 'horizontal' | 'vertical'
    polygon: BaseGeometry = field(init=False)

    def __post_init__(self):
        self.polygon = self.centerline.buffer(
            self.width / 2,
            cap_style="flat",
            join_style="mitre",
        )


def _unit_axis_from_delta(dx: float, dy: float) -> Optional[Tuple[float, float]]:
    if abs(float(dx)) <= 1e-12 and abs(float(dy)) <= 1e-12:
        return None
    if abs(float(dx)) >= abs(float(dy)):
        return (1.0, 0.0) if float(dx) >= 0.0 else (-1.0, 0.0)
    return (0.0, 1.0) if float(dy) >= 0.0 else (0.0, -1.0)


@dataclass
class Island:
    """
    岛屿定义

    属性：
    - 几何：矩形多边形
    - 语义：外墙方向、推荐分区、到入口/核心筒距离
    """
    id: str
    polygon: Polygon

    # 几何属性（自动计算）
    area: float = field(init=False)
    bounds: Tuple[float, float, float, float] = field(init=False)
    width: float = field(init=False)
    depth: float = field(init=False)

    # 语义属性
    has_exterior_wall: bool = False
    exterior_walls: List[str] = field(default_factory=list)
    corridor_edges: List[str] = field(default_factory=list)  # 哪些边接触走廊
    distance_to_entrance: float = 0.0
    distance_to_core: float = 0.0
    suggested_zone: ZoneType = ZoneType.PUBLIC

    # 容量跟踪（用于房间分配）
    assigned_rooms: List[str] = field(default_factory=list)
    remaining_capacity: float = field(init=False)

    def __post_init__(self):
        self.area = self.polygon.area
        self.bounds = self.polygon.bounds
        self.width = self.bounds[2] - self.bounds[0]
        self.depth = self.bounds[3] - self.bounds[1]
        self.remaining_capacity = self.area

    @property
    def is_rectangular(self) -> bool:
        """检查是否为矩形"""
        bbox_area = self.width * self.depth
        if bbox_area <= 0:
            return False
        return self.area / bbox_area > 0.99

    @property
    def centroid(self) -> Tuple[float, float]:
        c = self.polygon.centroid
        return (c.x, c.y)


class RectangularTopologyGenerator:
    """
    矩形拓扑生成器

    生成策略：
    1. 计算核心筒位置和尺寸
    2. 生成正交走廊网格（经过核心筒）
    3. 用核心筒和走廊切割楼层
    4. 提取矩形岛屿
    5. 计算岛屿语义属性
    """

    def __init__(
        self,
        floor_boundary: Polygon,
        corridor_width: float = 2.0,
        min_island_area: float = 0.0,  # 0 = auto
        grid_alignment: float = 0.5,
    ):
        self.floor = floor_boundary
        self.corridor_width = corridor_width
        # 动态 min_island_area：楼层面积的 2%，下限 4m²
        if min_island_area <= 0:
            self.min_island_area = max(4.0, floor_boundary.area * 0.02)
        else:
            self.min_island_area = min_island_area
        self.grid_alignment = grid_alignment

        self.bounds = floor_boundary.bounds
        self.x_min, self.y_min, self.x_max, self.y_max = self.bounds
        self.floor_width = self.x_max - self.x_min
        self.floor_depth = self.y_max - self.y_min

    def _orthogonal_polyline(
        self,
        p0: Point,
        p1: Point,
        *,
        obstacles: List[Polygon],
    ) -> LineString:
        eps = 0.02
        cw = float(self.corridor_width)
        margin = cw / 2.0 + 0.25

        def _line(coords: List[Tuple[float, float]]) -> LineString:
            clean: List[Tuple[float, float]] = []
            for x, y in coords:
                pt = (float(x), float(y))
                if not clean or (abs(clean[-1][0] - pt[0]) > 1e-9 or abs(clean[-1][1] - pt[1]) > 1e-9):
                    clean.append(pt)
            if len(clean) < 2:
                clean = [clean[0], clean[0]] if clean else [(float(p0.x), float(p0.y)), (float(p1.x), float(p1.y))]
            return LineString(clean)

        def _score(ls: LineString) -> Tuple[float, float]:
            try:
                tube = ls.buffer(cw / 2.0 + eps, cap_style="flat", join_style="mitre")
            except Exception:
                tube = ls
            hits = 0.0
            for obs in obstacles:
                if obs is None or getattr(obs, "is_empty", True):
                    continue
                try:
                    if tube.intersects(obs.buffer(eps)):
                        hits += 1.0
                except Exception:
                    continue
            return (hits, float(ls.length))

        x0, y0 = float(p0.x), float(p0.y)
        x1, y1 = float(p1.x), float(p1.y)

        candidates: List[LineString] = []
        candidates.append(_line([(x0, y0), (x1, y0), (x1, y1)]))
        candidates.append(_line([(x0, y0), (x0, y1), (x1, y1)]))

        obs_union = unary_union([o for o in obstacles if o is not None and not getattr(o, "is_empty", True)]) if obstacles else Polygon()
        try:
            ominx, ominy, omaxx, omaxy = (float(v) for v in obs_union.bounds)
            if not np.isfinite([ominx, ominy, omaxx, omaxy]).all():
                raise ValueError("bad bounds")
        except Exception:
            ominx, ominy, omaxx, omaxy = (float(self.x_min), float(self.y_min), float(self.x_max), float(self.y_max))

        bx_left = float(ominx) - margin
        bx_right = float(omaxx) + margin
        by_bot = float(ominy) - margin
        by_top = float(omaxy) + margin

        bx_left = min(max(bx_left, float(self.x_min) + cw / 2.0), float(self.x_max) - cw / 2.0)
        bx_right = min(max(bx_right, float(self.x_min) + cw / 2.0), float(self.x_max) - cw / 2.0)
        by_bot = min(max(by_bot, float(self.y_min) + cw / 2.0), float(self.y_max) - cw / 2.0)
        by_top = min(max(by_top, float(self.y_min) + cw / 2.0), float(self.y_max) - cw / 2.0)

        candidates.append(_line([(x0, y0), (bx_left, y0), (bx_left, y1), (x1, y1)]))
        candidates.append(_line([(x0, y0), (bx_right, y0), (bx_right, y1), (x1, y1)]))
        candidates.append(_line([(x0, y0), (x0, by_bot), (x1, by_bot), (x1, y1)]))
        candidates.append(_line([(x0, y0), (x0, by_top), (x1, by_top), (x1, y1)]))

        best = min(candidates, key=_score)
        return best

    def generate(
        self,
        core_tube: Optional[CoreTube] = None,
        corridor_layout: str = "door_side",
        entrance_position: Optional[Tuple[float, float]] = None,
        group_seed: Optional[int] = None,
        force_corridor_boundary_contact: bool = False,
    ) -> Tuple[CoreTube, List[Corridor], List[Island]]:
        """
        生成矩形拓扑

        参数:
            core_tube: 核心筒（如果为 None 则自动创建）
            corridor_layout: 走廊布局类型 ('door_side' | 'cross' | 'H' | 'grid')
            entrance_position: 入口位置

        返回:
            (核心筒, 走廊列表, 岛屿列表)
        """
        # Step 1: 创建核心筒
        if core_tube is None:
            core_tube = CoreTube.create_for_floor(self.bounds, grid_alignment=self.grid_alignment)

        if entrance_position is None:
            entrance_position = (
                (self.x_min + self.x_max) / 2,
                self.y_min,
            )
        else:
            entrance_position = self._project_entrance_to_boundary(entrance_position)

        # Step 2: 生成走廊
        if corridor_layout in ("door_side", "cross"):
            corridors = self._generate_plug_corridors(core_tube, group_seed=group_seed)
        elif corridor_layout == "H":
            corridors = self._generate_h_corridors(core_tube)
        elif corridor_layout == "grid":
            corridors = self._generate_grid_corridors(core_tube)
        elif corridor_layout == "organic":
            trunc = 0.0 if force_corridor_boundary_contact else None
            corridors = self._generate_organic_corridors(core_tube, group_seed=group_seed, truncation_override=trunc)
        else:
            corridors = self._generate_plug_corridors(core_tube, group_seed=group_seed)

        # Step 2.5: 核心筒对齐走廊交叉区（保持 core_placement 稳定性：不再平移核心筒）
        core_tube = core_tube

        # Step 2.6: 走廊裁剪核心筒（避免几何重叠；保持 corridor 与 hall 的“贴边”不被 buffer 缝隙吞没）
        core_poly_for_cut = core_tube.polygon
        try:
            if (not core_poly_for_cut.is_valid) and make_valid is not None:
                core_poly_for_cut = make_valid(core_poly_for_cut)
        except Exception:
            pass
        cut_corridors: List[Corridor] = []
        for corridor in corridors:
            try:
                diff = corridor.polygon.difference(core_poly_for_cut)
                try:
                    diff = diff.buffer(0)
                except Exception:
                    pass
            except Exception:
                diff = corridor.polygon

            polys = _as_polygons(diff)
            kept = [p for p in polys if p.area > 0.1]
            if not kept:
                continue

            for k, poly in enumerate(kept):
                cid = corridor.id if k == 0 else f"{corridor.id}_p{k}"
                c = Corridor(
                    id=cid,
                    centerline=corridor.centerline,
                    width=corridor.width,
                    orientation=corridor.orientation,
                )
                c.polygon = _safe_snap_polygon_like(poly, tol=0.01) or poly
                cut_corridors.append(c)
        corridors = cut_corridors
        if not corridors:
            try:
                fallback = self._generate_cross_corridors(core_tube)
                recut: List[Corridor] = []
                for corridor in fallback:
                    try:
                        diff = corridor.polygon.difference(core_poly_for_cut)
                        try:
                            diff = diff.buffer(0)
                        except Exception:
                            pass
                    except Exception:
                        diff = corridor.polygon
                    polys = _as_polygons(diff)
                    kept = [p for p in polys if p.area > 0.1]
                    for k, poly in enumerate(kept):
                        cid = corridor.id if k == 0 else f"{corridor.id}_p{k}"
                        c = Corridor(
                            id=cid,
                            centerline=corridor.centerline,
                            width=corridor.width,
                            orientation=corridor.orientation,
                        )
                        c.polygon = _safe_snap_polygon_like(poly, tol=0.01) or poly
                        recut.append(c)
                corridors = recut or fallback
            except Exception:
                corridors = []

        # Step 2.7: 计算排除区域（核心筒+走廊），存为实例变量供后续方法使用
        self._subtract_union = unary_union(
            [core_tube.polygon] + [c.polygon for c in corridors]
        )

        if force_corridor_boundary_contact:
            try:
                corridor_union = unary_union([c.polygon for c in corridors]) if corridors else Polygon()
                shared_len = float(corridor_union.intersection(self.floor.exterior).length)
            except Exception:
                shared_len = 0.0

            if shared_len < 0.05:
                try:
                    hall_a = getattr(core_tube, "elevator_hall", None)
                    hall_b = getattr(core_tube, "elevator_hall_b", None)
                    halls: List[Polygon] = []
                    if hall_a is not None and (not getattr(hall_a, "is_empty", True)):
                        halls.append(hall_a)
                    if hall_b is not None and (not getattr(hall_b, "is_empty", True)):
                        halls.append(hall_b)

                    corridor_targets = [
                        c.polygon
                        for c in corridors
                        if c.polygon is not None and not getattr(c.polygon, "is_empty", True)
                    ]
                    if corridor_targets:
                        targets: List[BaseGeometry] = corridor_targets
                    elif halls:
                        targets = list(halls)
                    else:
                        targets = [core_tube.polygon]
                    target_union = unary_union(targets)

                    p_ext = Point(entrance_position)
                    p_in = nearest_points(p_ext, target_union)[1]

                    orth = self._orthogonal_polyline(p_ext, p_in, obstacles=[core_tube.polygon] + halls)
                    snap_tol = 0.01
                    try:
                        orth_coords = [
                            (
                                round(float(x) / float(snap_tol)) * float(snap_tol),
                                round(float(y) / float(snap_tol)) * float(snap_tol),
                            )
                            for x, y in orth.coords
                        ]
                        orth = LineString(orth_coords)
                    except Exception:
                        pass
                    cw = float(self.corridor_width)
                    entrance_poly0 = orth.buffer(cw / 2.0, cap_style="flat", join_style="mitre")
                    try:
                        entrance_poly0 = entrance_poly0.intersection(self.floor)
                    except Exception:
                        pass
                    entrance_poly0 = _largest_polygon(entrance_poly0) or entrance_poly0

                    best_corridor: Optional[Corridor] = None
                    best_area = 0.0
                    for c in corridors:
                        try:
                            a = float(c.polygon.intersection(entrance_poly0).area)
                        except Exception:
                            a = 0.0
                        if a > best_area + 1e-9:
                            best_area = a
                            best_corridor = c
                    if best_corridor is None and corridors:
                        try:
                            best_corridor = min(corridors, key=lambda cc: float(cc.polygon.distance(entrance_poly0)))
                        except Exception:
                            best_corridor = corridors[0]

                    deep_insert = 0.2
                    orth_deep_coords = list(orth.coords)
                    if best_corridor is not None and len(orth_deep_coords) >= 2:
                        prev = orth_deep_coords[-2]
                        last = orth_deep_coords[-1]
                        dx = float(last[0]) - float(prev[0])
                        dy = float(last[1]) - float(prev[1])
                        seg_dir = _unit_axis_from_delta(dx, dy) if abs(dx) + abs(dy) >= 1e-6 else None

                        probe = min(0.05, float(deep_insert) * 0.25)
                        try_dirs: List[Tuple[float, float]] = []
                        if seg_dir is not None:
                            try_dirs.append(seg_dir)
                        for d in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]:
                            if d not in try_dirs:
                                try_dirs.append(d)

                        def _contains(pt: Tuple[float, float]) -> bool:
                            try:
                                return bool(best_corridor.polygon.contains(Point(float(pt[0]), float(pt[1]))))
                            except Exception:
                                return False

                        chosen: Optional[Tuple[float, float]] = None
                        for d in try_dirs:
                            if _contains((float(last[0]) + float(d[0]) * probe, float(last[1]) + float(d[1]) * probe)):
                                chosen = d
                                break

                        if chosen is not None:
                            p_deep = (
                                round((float(last[0]) + float(chosen[0]) * float(deep_insert)) / float(snap_tol)) * float(snap_tol),
                                round((float(last[1]) + float(chosen[1]) * float(deep_insert)) / float(snap_tol)) * float(snap_tol),
                            )
                            if seg_dir is not None and abs(chosen[0] - seg_dir[0]) <= 1e-9 and abs(chosen[1] - seg_dir[1]) <= 1e-9:
                                orth_deep_coords[-1] = p_deep
                            else:
                                orth_deep_coords.append(p_deep)

                    try:
                        orth_deep_coords = [
                            (
                                round(float(x) / float(snap_tol)) * float(snap_tol),
                                round(float(y) / float(snap_tol)) * float(snap_tol),
                            )
                            for x, y in orth_deep_coords
                        ]
                    except Exception:
                        pass
                    orth_deep = LineString(orth_deep_coords)
                    entrance_poly = orth_deep.buffer(cw / 2.0, cap_style="flat", join_style="mitre")
                    try:
                        entrance_poly = entrance_poly.intersection(self.floor)
                    except Exception:
                        pass
                    entrance_poly = _largest_polygon(entrance_poly) or entrance_poly

                    if best_corridor is None:
                        c = Corridor(
                            id="corridor_main_entry",
                            centerline=orth_deep,
                            width=cw,
                            orientation="entry",
                        )
                        c.polygon = _largest_polygon(entrance_poly) or entrance_poly0
                        corridors = list(corridors) + [c]
                    else:
                        raw_merged = best_corridor.polygon.union(entrance_poly)
                        try:
                            raw_merged = raw_merged.intersection(self.floor)
                        except Exception:
                            pass
                        merged = _safe_snap_polygon_like(raw_merged, tol=0.01) or _largest_polygon(raw_merged)
                        if merged is not None and not merged.is_empty:
                            best_corridor.polygon = _orthogonalize_polygon(merged, tol=1e-6)

                    self._subtract_union = unary_union([core_tube.polygon] + [c.polygon for c in corridors])
                    try:
                        corridor_union2 = unary_union([c.polygon for c in corridors]) if corridors else Polygon()
                        shared_len2 = float(corridor_union2.intersection(self.floor.exterior).length)
                    except Exception:
                        shared_len2 = 0.0
                    if float(shared_len2) < 0.05:
                        logger.warning(
                            "force_corridor_boundary_contact ineffective: shared_len=%.4f, entrance=%s, p_ext=(%.3f,%.3f), p_in=(%.3f,%.3f), best=%s",
                            float(shared_len2),
                            str(entrance_position),
                            float(p_ext.x),
                            float(p_ext.y),
                            float(p_in.x),
                            float(p_in.y),
                            str(getattr(best_corridor, "id", None)),
                        )
                except Exception as e:
                    logger.warning("Failed to force corridor boundary contact: %s", str(e)[:200])

        # Step 3: 生成岛屿（网格切片：保证轴对齐矩形）
        island_polys, cell_map, edge_set_islands = self._generate_perfect_rectangular_islands(
            self._subtract_union
        )
        self._edge_set_islands = edge_set_islands
        islands = [Island(id=f"island_{i}", polygon=p) for i, p in enumerate(island_polys)]

        # Step 4: 解决矩形化后的重叠（理论上不应有重叠；保留防御）
        islands = self._resolve_overlaps(islands)

        # Step 5: 计算语义属性（入口位置投影+幻觉检测）
        self._compute_island_semantics(islands, core_tube, entrance_position, corridors)

        # Step 6: 验证
        self._validate(islands)

        return core_tube, corridors, islands

    def _project_entrance_to_boundary(
        self, entrance: Tuple[float, float],
    ) -> Tuple[float, float]:
        """将入口位置投影到楼层边界最近点。

        距离 > 5m 判定为 LLM 幻觉，回退默认入口（底边中点）。
        """
        point = Point(entrance)
        distance_to_boundary = self.floor.exterior.distance(point)

        if distance_to_boundary > 5.0:
            logger.warning(
                "Entrance %s is %.1fm from boundary (likely LLM hallucination), "
                "using default entrance",
                entrance, distance_to_boundary,
            )
            return ((self.x_min + self.x_max) / 2, self.y_min)

        projected = self.floor.exterior.interpolate(
            self.floor.exterior.project(point)
        )
        return (projected.x, projected.y)

    def _generate_perfect_rectangular_islands(
        self, subtract_union
    ) -> Tuple[List[Polygon], dict, dict]:
        """
        X/Y 轴对齐网格切片法，保证输出 100% 轴对齐矩形岛屿（以网格合并后的矩形为单位）。

        Returns:
            (island_polys, cell_map, edge_set_islands)
            - island_polys: List[Polygon]（每个为 box(...) 生成并做微缩）
            - cell_map: {(i, j): island_id} 网格 cell → 合并后的岛屿 id
            - edge_set_islands: {frozenset({id_a, id_b}): 'vertical'|'horizontal'}
        """

        fminx, fminy, fmaxx, fmaxy = self.floor.bounds

        xs = {round(fminx, 2), round(fmaxx, 2)}
        ys = {round(fminy, 2), round(fmaxy, 2)}

        def collect_coords(g) -> None:
            if g.is_empty:
                return
            if isinstance(g, Polygon):
                for x0, y0 in g.exterior.coords:
                    xs.add(round(x0, 2))
                    ys.add(round(y0, 2))
                return
            if isinstance(g, (MultiPolygon, GeometryCollection)):
                for gg in g.geoms:
                    collect_coords(gg)

        collect_coords(subtract_union)

        xs_list = sorted(xs)
        ys_list = sorted(ys)

        if len(xs_list) < 2 or len(ys_list) < 2:
            return [], {}, {}

        free = [[False for _ in range(len(ys_list) - 1)] for _ in range(len(xs_list) - 1)]

        for i in range(len(xs_list) - 1):
            x0, x1 = xs_list[i], xs_list[i + 1]
            if (x1 - x0) < 0.05:
                continue
            for j in range(len(ys_list) - 1):
                y0, y1 = ys_list[j], ys_list[j + 1]
                if (y1 - y0) < 0.05:
                    continue

                cell = box(x0, y0, x1, y1)
                try:
                    cell2 = cell.intersection(self.floor)
                except Exception:
                    continue
                if cell2.is_empty or not isinstance(cell2, Polygon):
                    continue
                if cell2.area <= 0:
                    continue
                if cell2.area / cell.area < 0.99:
                    continue

                try:
                    overlap = cell2.intersection(subtract_union)
                    overlap_ratio = overlap.area / cell2.area if cell2.area > 0 else 1.0
                except Exception:
                    overlap_ratio = 1.0
                if overlap_ratio > 0.05:
                    continue

                free[i][j] = True

        spans_by_row = []
        for j in range(len(ys_list) - 1):
            spans = []
            i = 0
            while i < len(xs_list) - 1:
                if not free[i][j]:
                    i += 1
                    continue
                start = i
                while i < len(xs_list) - 1 and free[i][j]:
                    i += 1
                spans.append((start, i))
            spans_by_row.append(spans)

        active = {}
        rects = []
        for j, spans in enumerate(spans_by_row):
            spans_set = set(spans)
            next_active = {}
            for span in spans:
                if span in active:
                    j0, _j1 = active[span]
                    next_active[span] = (j0, j)
                else:
                    next_active[span] = (j, j)
            for span, (j0, j1) in active.items():
                if span not in spans_set:
                    i0, i1 = span
                    rects.append((i0, i1, j0, j1))
            active = next_active

        for span, (j0, j1) in active.items():
            i0, i1 = span
            rects.append((i0, i1, j0, j1))

        island_polys: List[Polygon] = []
        cell_map: dict = {}
        for i0, i1, j0, j1 in rects:
            x0, x1 = xs_list[i0], xs_list[i1]
            y0, y1 = ys_list[j0], ys_list[j1 + 1]
            poly = box(x0, y0, x1, y1)
            shrunk = poly.buffer(-0.001, join_style="mitre")
            if shrunk.is_empty or not isinstance(shrunk, Polygon):
                continue
            if shrunk.area < self.min_island_area:
                continue
            island_id = f"island_{len(island_polys)}"
            island_polys.append(shrunk)
            for ii in range(i0, i1):
                for jj in range(j0, j1 + 1):
                    cell_map[(ii, jj)] = island_id

        edge_set_islands: dict = {}
        for (i, j), id_a in cell_map.items():
            id_b = cell_map.get((i + 1, j))
            if id_b and id_b != id_a:
                edge_set_islands[frozenset({id_a, id_b})] = "vertical"
            id_b = cell_map.get((i, j + 1))
            if id_b and id_b != id_a:
                edge_set_islands[frozenset({id_a, id_b})] = "horizontal"

        return island_polys, cell_map, edge_set_islands

    def _generate_cross_corridors(self, core: CoreTube) -> List[Corridor]:
        """单条水平走廊，止步于核心筒南边缘（开门面）。

        核心筒靠北墙时，走廊在核心筒南侧水平贯穿楼层，
        所有房间通过走廊到达核心筒的电梯/楼梯。
        """
        _cx, cy = core.center
        south_edge_y = cy - core.depth / 2
        center_y = south_edge_y - self.corridor_width / 2
        center_y = min(
            max(center_y, self.y_min + self.corridor_width / 2),
            self.y_max - self.corridor_width / 2,
        )

        h_corridor = Corridor(
            id="corridor_h",
            centerline=LineString([(self.x_min, center_y), (self.x_max, center_y)]),
            width=self.corridor_width,
            orientation="horizontal",
        )

        return [h_corridor]

    def _generate_plug_corridors(self, core: CoreTube, *, group_seed: Optional[int]) -> List[Corridor]:
        cw = float(self.corridor_width)
        eps = 0.05
        door_min = 0.9 + 2.0 * 0.12 + eps

        opening_sides = [str(s).lower() for s in (getattr(core, "opening_sides", None) or ["south"])]
        if not opening_sides:
            opening_sides = ["south"]

        hall_a = getattr(core, "elevator_hall", None)
        hall_b = getattr(core, "elevator_hall_b", None)

        def _clamp(v: float, lo: float, hi: float) -> float:
            return float(min(max(float(v), float(lo)), float(hi)))

        def _align(v: float) -> float:
            return round(float(v) / float(self.grid_alignment)) * float(self.grid_alignment)

        def _shared_len_for_bounds(side: str, cpoly: BaseGeometry, hpoly: BaseGeometry) -> float:
            if cpoly is None or hpoly is None or cpoly.is_empty or hpoly.is_empty:
                return 0.0
            cminx, cminy, cmaxx, cmaxy = (float(v) for v in cpoly.bounds)
            hminx, hminy, hmaxx, hmaxy = (float(v) for v in hpoly.bounds)
            tol = 0.06
            if side == "west" and abs(cmaxx - hminx) <= tol:
                return max(0.0, min(cmaxy, hmaxy) - max(cminy, hminy))
            if side == "east" and abs(cminx - hmaxx) <= tol:
                return max(0.0, min(cmaxy, hmaxy) - max(cminy, hminy))
            if side == "south" and abs(cmaxy - hminy) <= tol:
                return max(0.0, min(cmaxx, hmaxx) - max(cminx, hminx))
            if side == "north" and abs(cminy - hmaxy) <= tol:
                return max(0.0, min(cmaxx, hmaxx) - max(cminx, hminx))
            return 0.0

        def _best_anchor_for_side(side: str, hpoly: Polygon) -> Optional[Tuple[float, float]]:
            if hpoly is None or hpoly.is_empty:
                return None
            hminx, hminy, hmaxx, hmaxy = (float(v) for v in hpoly.bounds)
            margin = max(0.2, cw / 2.0 + 0.1)
            if side in ("north", "south"):
                xs = [(hminx + hmaxx) / 2.0, hminx + margin, hmaxx - margin]
                best = None
                best_len = -1.0
                for x0 in xs:
                    x = _align(_clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0))
                    if side == "south":
                        y0 = float(hminy)
                        y1 = float(self.y_min) + cw / 2.0
                    else:
                        y0 = float(hmaxy)
                        y1 = float(self.y_max) - cw / 2.0
                    y0 = _clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                    y1 = _clamp(y1, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                    c = Corridor(id="tmp", centerline=LineString([(x, y0), (x, y1)]), width=cw, orientation="vertical")
                    L = _shared_len_for_bounds(side, c.polygon, hpoly)
                    if L > best_len + 1e-6:
                        best_len = L
                        best = (x, L)
                return best
            ys = [(hminy + hmaxy) / 2.0, hminy + margin, hmaxy - margin]
            best = None
            best_len = -1.0
            for y0 in ys:
                y = _align(_clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0))
                if side == "west":
                    x0 = float(hminx)
                    x1 = float(self.x_min) + cw / 2.0
                else:
                    x0 = float(hmaxx)
                    x1 = float(self.x_max) - cw / 2.0
                x0 = _clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                x1 = _clamp(x1, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                c = Corridor(id="tmp", centerline=LineString([(x0, y), (x1, y)]), width=cw, orientation="horizontal")
                L = _shared_len_for_bounds(side, c.polygon, hpoly)
                if L > best_len + 1e-6:
                    best_len = L
                    best = (y, L)
            return best

        def _build_corridor_for_side(side: str, hpoly: Polygon, cid: str) -> Optional[Corridor]:
            pick = _best_anchor_for_side(side, hpoly)
            if pick is None:
                return None
            val, L = pick
            if float(L) + 1e-6 < float(door_min):
                return None
            hminx, hminy, hmaxx, hmaxy = (float(v) for v in hpoly.bounds)
            if side in ("north", "south"):
                x = float(val)
                if side == "south":
                    y0, y1 = float(hminy), float(self.y_min) + cw / 2.0
                else:
                    y0, y1 = float(hmaxy), float(self.y_max) - cw / 2.0
                y0 = _clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                y1 = _clamp(y1, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                return Corridor(id=cid, centerline=LineString([(x, y0), (x, y1)]), width=cw, orientation="vertical")
            y = float(val)
            if side == "west":
                x0, x1 = float(hminx), float(self.x_min) + cw / 2.0
            else:
                x0, x1 = float(hmaxx), float(self.x_max) - cw / 2.0
            x0 = _clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
            x1 = _clamp(x1, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
            return Corridor(id=cid, centerline=LineString([(x0, y), (x1, y)]), width=cw, orientation="horizontal")

        corridors: List[Corridor] = []
        for side in opening_sides:
            if side in ("west", "east") and hall_b is not None and set(opening_sides) == {"west", "east"}:
                hpoly = hall_a if side == "west" else hall_b
            elif side in ("south", "north") and hall_b is not None and set(opening_sides) == {"south", "north"}:
                hpoly = hall_a if side == "south" else hall_b
            else:
                hpoly = hall_a
            if hpoly is None or hpoly.is_empty:
                continue
            c = _build_corridor_for_side(side, hpoly, cid=f"corridor_main_{side}")
            if c is not None:
                corridors.append(c)

        if corridors:
            return corridors

        try:
            corridors = self._generate_organic_corridors(core, group_seed=group_seed)
            if corridors:
                return corridors
        except Exception:
            pass

        try:
            return list(self._generate_cross_corridors(core))
        except Exception:
            return []

    def _generate_h_corridors(self, core: CoreTube) -> List[Corridor]:
        """H 型走廊布局（适合长条形楼层）"""
        cx, cy = core.center

        h_corridor = Corridor(
            id="corridor_h",
            centerline=LineString([(self.x_min, cy), (self.x_max, cy)]),
            width=self.corridor_width,
            orientation="horizontal",
        )

        v_left = Corridor(
            id="corridor_v_left",
            centerline=LineString([
                (self.x_min + self.floor_width * 0.15, self.y_min),
                (self.x_min + self.floor_width * 0.15, self.y_max),
            ]),
            width=self.corridor_width,
            orientation="vertical",
        )

        v_right = Corridor(
            id="corridor_v_right",
            centerline=LineString([
                (self.x_max - self.floor_width * 0.15, self.y_min),
                (self.x_max - self.floor_width * 0.15, self.y_max),
            ]),
            width=self.corridor_width,
            orientation="vertical",
        )

        return [h_corridor, v_left, v_right]

    def _generate_grid_corridors(self, core: CoreTube) -> List[Corridor]:
        """网格走廊布局（适合大型楼层）"""
        corridors = list(self._generate_cross_corridors(core))
        cx, cy = core.center
        max_island_dim = 15.0

        left_width = cx - self.corridor_width / 2 - self.x_min
        right_width = self.x_max - (cx + self.corridor_width / 2)

        if left_width > max_island_dim:
            x_pos = self._align((self.x_min + cx) / 2)
            corridors.append(Corridor(
                id="corridor_v_extra_left",
                centerline=LineString([(x_pos, self.y_min), (x_pos, self.y_max)]),
                width=self.corridor_width,
                orientation="vertical",
            ))

        if right_width > max_island_dim:
            x_pos = self._align((self.x_max + cx) / 2)
            corridors.append(Corridor(
                id="corridor_v_extra_right",
                centerline=LineString([(x_pos, self.y_min), (x_pos, self.y_max)]),
                width=self.corridor_width,
                orientation="vertical",
            ))

        top_depth = self.y_max - (cy + self.corridor_width / 2)
        bottom_depth = cy - self.corridor_width / 2 - self.y_min

        if top_depth > max_island_dim:
            y_pos = self._align((self.y_max + cy) / 2)
            corridors.append(Corridor(
                id="corridor_h_extra_top",
                centerline=LineString([(self.x_min, y_pos), (self.x_max, y_pos)]),
                width=self.corridor_width,
                orientation="horizontal",
            ))

        if bottom_depth > max_island_dim:
            y_pos = self._align((self.y_min + cy) / 2)
            corridors.append(Corridor(
                id="corridor_h_extra_bottom",
                centerline=LineString([(self.x_min, y_pos), (self.x_max, y_pos)]),
                width=self.corridor_width,
                orientation="horizontal",
            ))

        return corridors

    def _infer_core_placement(self, core: CoreTube) -> str:
        try:
            cminx, cminy, cmaxx, cmaxy = (float(v) for v in core.polygon.bounds)
        except Exception:
            return "center"
        tol = 0.6
        if abs(float(cmaxy) - float(self.y_max)) <= tol:
            return "north"
        if abs(float(cminy) - float(self.y_min)) <= tol:
            return "south"
        if abs(float(cmaxx) - float(self.x_max)) <= tol:
            return "east"
        if abs(float(cminx) - float(self.x_min)) <= tol:
            return "west"
        return "center"

    def _sample_truncation(self, group_seed: Optional[int]) -> float:
        if group_seed is None:
            return 5.0
        rng = np.random.default_rng(int(group_seed))
        return float([3.0, 5.0, 8.0][int(rng.integers(0, 3))])

    def _generate_organic_corridors(
        self,
        core: CoreTube,
        *,
        group_seed: Optional[int],
        truncation_override: Optional[float] = None,
    ) -> List[Corridor]:
        t = float(truncation_override) if truncation_override is not None else self._sample_truncation(group_seed)
        cw = float(self.corridor_width)
        eps = 0.05
        door_min = 0.9 + 2.0 * 0.12 + eps
        min_len = max(float(cw), 1.2)

        opening_sides = [str(s).lower() for s in (getattr(core, "opening_sides", None) or ["south"])]
        if not opening_sides:
            opening_sides = ["south"]

        corridors: List[Corridor] = []

        hall_a = getattr(core, "elevator_hall", None)
        hall_b = getattr(core, "elevator_hall_b", None)

        def _clamp(v: float, lo: float, hi: float) -> float:
            return float(min(max(float(v), float(lo)), float(hi)))

        def _align(v: float) -> float:
            return round(float(v) / float(self.grid_alignment)) * float(self.grid_alignment)

        def _shared_len_for_bounds(side: str, cpoly: BaseGeometry, hpoly: BaseGeometry) -> float:
            if cpoly is None or hpoly is None or cpoly.is_empty or hpoly.is_empty:
                return 0.0
            cminx, cminy, cmaxx, cmaxy = (float(v) for v in cpoly.bounds)
            hminx, hminy, hmaxx, hmaxy = (float(v) for v in hpoly.bounds)
            tol = 0.06
            if side == "west" and abs(cmaxx - hminx) <= tol:
                return max(0.0, min(cmaxy, hmaxy) - max(cminy, hminy))
            if side == "east" and abs(cminx - hmaxx) <= tol:
                return max(0.0, min(cmaxy, hmaxy) - max(cminy, hminy))
            if side == "south" and abs(cmaxy - hminy) <= tol:
                return max(0.0, min(cmaxx, hmaxx) - max(cminx, hminx))
            if side == "north" and abs(cminy - hmaxy) <= tol:
                return max(0.0, min(cmaxx, hmaxx) - max(cminx, hminx))
            return 0.0

        def _best_anchor_for_side(side: str, hpoly: Polygon) -> Optional[Tuple[float, float]]:
            if hpoly is None or hpoly.is_empty:
                return None
            hminx, hminy, hmaxx, hmaxy = (float(v) for v in hpoly.bounds)
            margin = max(0.2, cw / 2.0 + 0.1)
            if side in ("north", "south"):
                xs = [
                    (hminx + hmaxx) / 2.0,
                    hminx + margin,
                    hmaxx - margin,
                ]
                best = None
                best_len = -1.0
                for x0 in xs:
                    x = _align(_clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0))
                    if side == "south":
                        y0 = float(hminy)
                        target = float(self.y_min) + float(t) + cw / 2.0
                        y1 = min(target, float(y0) - float(min_len))
                        y0 = _clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                        y1 = _clamp(y1, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                        c = Corridor(id="tmp", centerline=LineString([(x, y0), (x, y1)]), width=cw, orientation="vertical")
                    else:
                        y0 = float(hmaxy)
                        target = float(self.y_max) - float(t) - cw / 2.0
                        y1 = max(target, float(y0) + float(min_len))
                        y0 = _clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                        y1 = _clamp(y1, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                        c = Corridor(id="tmp", centerline=LineString([(x, y0), (x, y1)]), width=cw, orientation="vertical")
                    L = _shared_len_for_bounds(side, c.polygon, hpoly)
                    if L > best_len + 1e-6:
                        best_len = L
                        best = (x, L)
                if best is None:
                    return None
                return (best[0], best[1])
            else:
                ys = [
                    (hminy + hmaxy) / 2.0,
                    hminy + margin,
                    hmaxy - margin,
                ]
                best = None
                best_len = -1.0
                for y0 in ys:
                    y = _align(_clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0))
                    if side == "west":
                        x0 = float(hminx)
                        target = float(self.x_min) + float(t) + cw / 2.0
                        x1 = min(target, float(x0) - float(min_len))
                        x0 = _clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                        x1 = _clamp(x1, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                        c = Corridor(id="tmp", centerline=LineString([(x0, y), (x1, y)]), width=cw, orientation="horizontal")
                    else:
                        x0 = float(hmaxx)
                        target = float(self.x_max) - float(t) - cw / 2.0
                        x1 = max(target, float(x0) + float(min_len))
                        x0 = _clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                        x1 = _clamp(x1, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                        c = Corridor(id="tmp", centerline=LineString([(x0, y), (x1, y)]), width=cw, orientation="horizontal")
                    L = _shared_len_for_bounds(side, c.polygon, hpoly)
                    if L > best_len + 1e-6:
                        best_len = L
                        best = (y, L)
                if best is None:
                    return None
                return (best[0], best[1])

        def _build_corridor_for_side(side: str, hpoly: Polygon, cid: str) -> Optional[Corridor]:
            pick = _best_anchor_for_side(side, hpoly)
            if pick is None:
                return None
            val, L = pick
            if float(L) + 1e-6 < float(door_min):
                return None
            hminx, hminy, hmaxx, hmaxy = (float(v) for v in hpoly.bounds)
            if side in ("north", "south"):
                x = float(val)
                if side == "south":
                    y0 = float(hminy)
                    target = float(self.y_min) + float(t) + cw / 2.0
                    y1 = min(target, float(y0) - float(min_len))
                else:
                    y0 = float(hmaxy)
                    target = float(self.y_max) - float(t) - cw / 2.0
                    y1 = max(target, float(y0) + float(min_len))
                y0 = _clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                y1 = _clamp(y1, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                return Corridor(id=cid, centerline=LineString([(x, y0), (x, y1)]), width=cw, orientation="vertical")
            y = float(val)
            if side == "west":
                x0 = float(hminx)
                target = float(self.x_min) + float(t) + cw / 2.0
                x1 = min(target, float(x0) - float(min_len))
            else:
                x0 = float(hmaxx)
                target = float(self.x_max) - float(t) - cw / 2.0
                x1 = max(target, float(x0) + float(min_len))
            x0 = _clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
            x1 = _clamp(x1, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
            return Corridor(id=cid, centerline=LineString([(x0, y), (x1, y)]), width=cw, orientation="horizontal")

        for side in opening_sides:
            if side in ("west", "east") and hall_b is not None and core.elevator_hall_b is not None and set(opening_sides) == {"west", "east"}:
                hpoly = hall_a if side == "west" else hall_b
            elif side in ("south", "north") and hall_b is not None and core.elevator_hall_b is not None and set(opening_sides) == {"south", "north"}:
                hpoly = hall_a if side == "south" else hall_b
            else:
                hpoly = hall_a
            if hpoly is None or hpoly.is_empty:
                continue
            c = _build_corridor_for_side(side, hpoly, cid=f"corridor_main_{side}")
            if c is not None:
                corridors.append(c)

        if not corridors:
            placement = self._infer_core_placement(core)
            hall = hall_a if (hall_a is not None and (not getattr(hall_a, "is_empty", True))) else core.polygon
            hminx, hminy, hmaxx, hmaxy = (float(v) for v in hall.bounds)
            if placement in ("north", "south", "center"):
                x = _align((hminx + hmaxx) / 2.0)
                x = _clamp(x, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                if placement == "north":
                    y0 = float(hminy)
                    target = float(self.y_min) + float(t) + cw / 2.0
                    y1 = min(target, float(y0) - float(min_len))
                elif placement == "south":
                    y0 = float(hmaxy)
                    target = float(self.y_max) - float(t) - cw / 2.0
                    y1 = max(target, float(y0) + float(min_len))
                else:
                    y0 = float(self.y_min) + float(t) + cw / 2.0
                    y1 = float(self.y_max) - float(t) - cw / 2.0
                    if y1 < y0 + float(min_len):
                        y1 = y0 + float(min_len)
                y0 = _clamp(y0, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                y1 = _clamp(y1, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                corridors.append(Corridor(id="corridor_v_main", centerline=LineString([(x, y0), (x, y1)]), width=cw, orientation="vertical"))
            else:
                y = _align((hminy + hmaxy) / 2.0)
                y = _clamp(y, float(self.y_min) + cw / 2.0, float(self.y_max) - cw / 2.0)
                if placement == "east":
                    x0 = float(hminx)
                    target = float(self.x_min) + float(t) + cw / 2.0
                    x1 = min(target, float(x0) - float(min_len))
                else:
                    x0 = float(hmaxx)
                    target = float(self.x_max) - float(t) - cw / 2.0
                    x1 = max(target, float(x0) + float(min_len))
                x0 = _clamp(x0, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                x1 = _clamp(x1, float(self.x_min) + cw / 2.0, float(self.x_max) - cw / 2.0)
                corridors.append(Corridor(id="corridor_h_main", centerline=LineString([(x0, y), (x1, y)]), width=cw, orientation="horizontal"))

        if group_seed is not None and corridors:
            rng = np.random.default_rng(int(group_seed) + 17)
            use_t = bool(int(rng.integers(0, 2)))
        else:
            use_t = False

        if use_t and corridors:
            main = corridors[0]
            try:
                if main.polygon is None or main.polygon.is_empty:
                    return corridors
                minx, miny, maxx, maxy = (float(v) for v in main.polygon.bounds)
                if not (np.isfinite([minx, miny, maxx, maxy]).all()):
                    return corridors
            except Exception:
                return corridors
            if main.orientation == "vertical":
                yb = round(((miny + maxy) / 2.0) / float(self.grid_alignment)) * float(self.grid_alignment)
                left = float(self.x_min) + float(t) + cw / 2.0
                right = float(self.x_max) - float(t) - cw / 2.0
                if right - left >= 4.0:
                    corridors.append(Corridor(
                        id="corridor_h_branch",
                        centerline=LineString([(left, yb), (right, yb)]),
                        width=cw,
                        orientation="horizontal",
                    ))
            elif main.orientation == "horizontal":
                xb = round(((minx + maxx) / 2.0) / float(self.grid_alignment)) * float(self.grid_alignment)
                bot = float(self.y_min) + float(t) + cw / 2.0
                top = float(self.y_max) - float(t) - cw / 2.0
                if top - bot >= 4.0:
                    corridors.append(Corridor(
                        id="corridor_v_branch",
                        centerline=LineString([(xb, bot), (xb, top)]),
                        width=cw,
                        orientation="vertical",
                    ))

        return corridors

    def _align_core_to_corridors(self, core: CoreTube, corridors: List[Corridor]) -> CoreTube:
        """确保核心筒至少覆盖走廊交叉区域，避免产生非矩形碎片。

        如果核心筒比走廊交叉区小，扩展到覆盖交叉区。
        """
        cx, cy = core.center
        half_cw = self.corridor_width / 2

        # 核心筒最小范围 = 走廊交叉区域
        min_left = cx - half_cw
        min_right = cx + half_cw
        min_bottom = cy - half_cw
        min_top = cy + half_cw

        # 当前核心筒范围
        c_left = cx - core.width / 2
        c_right = cx + core.width / 2
        c_bottom = cy - core.depth / 2
        c_top = cy + core.depth / 2

        # 只在需要扩展时才重建
        if c_left <= min_left and c_right >= min_right and c_bottom <= min_bottom and c_top >= min_top:
            return core

        new_left = min(c_left, min_left)
        new_right = max(c_right, min_right)
        new_bottom = min(c_bottom, min_bottom)
        new_top = max(c_top, min_top)

        new_width = new_right - new_left
        new_depth = new_top - new_bottom

        logger.info(
            "Aligning core tube to corridor intersection: "
            "%.1fx%.1f → %.1fx%.1f",
            core.width, core.depth, new_width, new_depth,
        )
        aligned_width = round(new_width / self.grid_alignment) * self.grid_alignment
        aligned_depth = round(new_depth / self.grid_alignment) * self.grid_alignment

        max_width = max(self.grid_alignment, self.floor_width - 2 * self.grid_alignment)
        max_depth = max(self.grid_alignment, self.floor_depth - 2 * self.grid_alignment)
        aligned_width = min(aligned_width, max_width)
        aligned_depth = min(aligned_depth, max_depth)

        cx2, cy2 = core.center
        cx2 = min(max(cx2, self.x_min + aligned_width / 2), self.x_max - aligned_width / 2)
        cy2 = min(max(cy2, self.y_min + aligned_depth / 2), self.y_max - aligned_depth / 2)
        return CoreTube.create((cx2, cy2), aligned_width, aligned_depth)

    def _align(self, value: float) -> float:
        """对齐到网格"""
        return round(value / self.grid_alignment) * self.grid_alignment

    @staticmethod
    def _is_rectangular(poly: Polygon, tolerance: float = 0.99) -> bool:
        """检查多边形是否接近矩形"""
        if poly.is_empty:
            return False
        minx, miny, maxx, maxy = poly.bounds
        bbox_area = (maxx - minx) * (maxy - miny)
        if bbox_area <= 0:
            return False
        return poly.area / bbox_area > tolerance

    def _max_inscribed_rectangle(self, poly: Polygon) -> Optional[Polygon]:
        """找多边形的最大内接矩形（网格采样法）"""
        minx, miny, maxx, maxy = poly.bounds
        w = maxx - minx
        h = maxy - miny
        step = max(w / 10, h / 10, 1.0)

        best = None
        best_area = 0.0

        for x in np.arange(minx + step / 2, maxx - step / 2, step):
            for y in np.arange(miny + step / 2, maxy - step / 2, step):
                if not poly.contains(Point(x, y)):
                    continue

                # 向四个方向扩展
                left = x
                while left - step > minx and poly.contains(Point(left - step, y)):
                    left -= step
                right = x
                while right + step < maxx and poly.contains(Point(right + step, y)):
                    right += step
                down = y
                while down - step > miny and poly.contains(Point(x, down - step)):
                    down -= step
                up = y
                while up + step < maxy and poly.contains(Point(x, up + step)):
                    up += step

                rect = box(left, down, right, up)
                if rect.area > best_area and poly.contains(rect):
                    best = rect
                    best_area = rect.area

        return best

    def _split_to_rectangles(
        self,
        poly: Polygon,
        depth: int = 0,
        max_depth: int = 5,
    ) -> List[Polygon]:
        """将非矩形多边形递归分割为矩形集合"""
        MIN_AREA = 4.0

        if depth >= max_depth:
            clipped = box(*poly.bounds).intersection(poly)
            if isinstance(clipped, Polygon) and clipped.area >= MIN_AREA:
                return [clipped]
            return []

        if poly.area < MIN_AREA:
            return []

        mir = self._max_inscribed_rectangle(poly)

        if mir is None or mir.area < MIN_AREA:
            clipped = box(*poly.bounds).intersection(poly)
            if isinstance(clipped, Polygon) and clipped.area >= MIN_AREA:
                return [clipped]
            return []

        result = [mir]
        remainder = poly.difference(mir)
        for sub in _as_polygons(remainder):
            if sub.area >= MIN_AREA:
                result.extend(self._split_to_rectangles(sub, depth + 1, max_depth))

        return result

    def _generate_islands(
        self,
        core: CoreTube,
        corridors: List[Corridor],
    ) -> List[Island]:
        """生成矩形岛屿"""
        # 使用 generate() 中已计算的排除区域
        remaining = self.floor.difference(self._subtract_union)

        # 提取多边形
        if remaining.is_empty:
            return []

        polygons = _as_polygons(remaining)

        # 创建岛屿
        islands = []
        idx = 0
        for i, poly in enumerate(polygons):
            if poly.area < self.min_island_area:
                continue

            if self._is_rectangular(poly):
                # 已经是矩形，直接使用
                islands.append(Island(id=f"island_{idx}", polygon=poly))
                idx += 1
            else:
                # 非矩形：尝试分割为多个矩形子岛
                sub_rects = self._split_to_rectangles(poly)
                for j, rect in enumerate(sub_rects):
                    if rect.area < self.min_island_area:
                        continue
                    # 确保结果是矩形：裁剪到可用区域（排除核心筒+走廊）
                    rect = rect.intersection(remaining)
                    if not isinstance(rect, Polygon) or rect.is_empty:
                        continue
                    if not self._is_rectangular(rect):
                        rect = box(*rect.bounds).intersection(remaining)
                    if isinstance(rect, Polygon) and not rect.is_empty and rect.area >= self.min_island_area:
                        islands.append(Island(id=f"island_{idx}", polygon=rect))
                        idx += 1

        return islands

    def _resolve_overlaps(self, islands: List[Island]) -> List[Island]:
        """解决矩形化后的岛屿重叠。

        策略：按面积降序排列，大岛屿优先保留。
        后续岛屿从中减去所有已接受岛屿的区域，然后裁剪到 remaining。
        """
        remaining = self.floor.difference(self._subtract_union)

        # 按面积降序
        islands.sort(key=lambda i: i.area, reverse=True)

        accepted: List[Island] = []
        for island in islands:
            if not accepted:
                # 首个（最大）岛屿也要裁剪到 remaining
                poly = island.polygon.intersection(remaining)
                if isinstance(poly, Polygon) and not poly.is_empty and poly.area >= self.min_island_area:
                    new_isl = Island(id=island.id, polygon=poly)
                    new_isl.has_exterior_wall = island.has_exterior_wall
                    new_isl.exterior_walls = list(island.exterior_walls)
                    new_isl.suggested_zone = island.suggested_zone
                    accepted.append(new_isl)
                continue

            # 从当前岛屿减去所有已接受的岛屿
            poly = island.polygon
            for prev in accepted:
                if poly.intersects(prev.polygon):
                    poly = poly.difference(prev.polygon)
                    if poly.is_empty:
                        break

            if poly.is_empty:
                continue

            # 取最大连通区域，矩形化
            candidates = _as_polygons(poly)
            if not candidates:
                continue

            best = max(candidates, key=lambda p: p.area)
            if best.area < self.min_island_area:
                continue

            # 如果不是矩形，取 bounding box 但裁剪到 remaining
            if not self._is_rectangular(best):
                clipped = box(*best.bounds).intersection(remaining)
                if isinstance(clipped, Polygon) and clipped.area >= self.min_island_area:
                    best = clipped
                else:
                    continue

            # 最终裁剪到 remaining（硬约束）
            best = best.intersection(remaining)
            if not isinstance(best, Polygon) or best.is_empty or best.area < self.min_island_area:
                continue

            # 保留原 island 的语义属性
            new_isl = Island(id=island.id, polygon=best)
            new_isl.has_exterior_wall = island.has_exterior_wall
            new_isl.exterior_walls = list(island.exterior_walls)
            new_isl.suggested_zone = island.suggested_zone
            accepted.append(new_isl)

        return accepted

    def _compute_island_semantics(
        self,
        islands: List[Island],
        core: CoreTube,
        entrance: Tuple[float, float],
        corridors: List[Corridor],
    ):
        """计算岛屿语义属性"""
        entrance_point = Point(entrance)
        core_center = Point(core.center)

        for island in islands:
            x_min, y_min, x_max, y_max = island.bounds
            island_center = Point(island.centroid)

            # 外墙方向
            tol = 0.5
            island.exterior_walls = []
            if abs(x_min - self.x_min) < tol:
                island.exterior_walls.append("west")
            if abs(x_max - self.x_max) < tol:
                island.exterior_walls.append("east")
            if abs(y_min - self.y_min) < tol:
                island.exterior_walls.append("south")
            if abs(y_max - self.y_max) < tol:
                island.exterior_walls.append("north")

            island.has_exterior_wall = len(island.exterior_walls) > 0

            # 走廊接触边
            island.corridor_edges = self._detect_corridor_edges(island, corridors)

            # 距离
            island.distance_to_entrance = island_center.distance(entrance_point)
            island.distance_to_core = island_center.distance(core_center)

            # 推荐分区
            island.suggested_zone = self._suggest_zone(island)

    def _detect_corridor_edges(
        self,
        island: Island,
        corridors: List[Corridor],
    ) -> List[str]:
        """检测岛屿的哪些边接触走廊"""
        edges = []
        minx, miny, maxx, maxy = island.polygon.bounds

        # 定义 4 条边界线（buffer 0.05m 容忍数值误差）
        edge_lines = {
            "south": LineString([(minx, miny), (maxx, miny)]).buffer(0.05),
            "north": LineString([(minx, maxy), (maxx, maxy)]).buffer(0.05),
            "west": LineString([(minx, miny), (minx, maxy)]).buffer(0.05),
            "east": LineString([(maxx, miny), (maxx, maxy)]).buffer(0.05),
        }

        corridor_union = unary_union([c.polygon for c in corridors])

        for direction, edge_geom in edge_lines.items():
            if edge_geom.intersects(corridor_union):
                intersection = edge_geom.intersection(corridor_union)
                if hasattr(intersection, "length") and intersection.length > 0.5:
                    edges.append(direction)

        return edges

    def _suggest_zone(self, island: Island) -> ZoneType:
        """推荐功能分区"""
        # 无外墙 → 服务区
        if not island.has_exterior_wall:
            return ZoneType.SERVICE

        # 靠近入口 → 公共区
        avg_distance = (self.floor_width + self.floor_depth) / 4
        if island.distance_to_entrance < avg_distance:
            return ZoneType.PUBLIC

        # 远离入口 + 有外墙 → 私密区
        return ZoneType.PRIVATE

    def _validate(self, islands: List[Island]):
        """验证生成结果"""
        non_rect = [i.id for i in islands if not i.is_rectangular]
        if non_rect:
            logger.warning(f"Non-rectangular islands: {non_rect}")

        total_island_area = sum(i.area for i in islands)
        coverage = total_island_area / self.floor.area if self.floor.area > 0 else 0
        logger.info(f"Generated {len(islands)} islands, coverage: {coverage:.1%}")

        # 硬约束：岛屿不得与核心筒/走廊重叠
        for island in islands:
            overlap = island.polygon.intersection(self._subtract_union)
            if overlap.area > 0.01:
                logger.error(
                    f"Island {island.id} overlaps excluded area by {overlap.area:.2f}m²"
                )
            # 硬约束：岛屿不得超出楼层边界
            outside = island.polygon.difference(self.floor)
            if outside.area > 0.01:
                logger.error(
                    f"Island {island.id} exceeds floor boundary by {outside.area:.2f}m²"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════


def generate_rectangular_topology(
    floor_boundary: Polygon,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "door_side",
    entrance_position: Optional[Tuple[float, float]] = None,
    core_tube_override: Optional[CoreTube] = None,
    core_position: str = "north",
    group_seed: Optional[int] = None,
    force_corridor_boundary_contact: bool = False,
) -> Tuple[CoreTube, List[Corridor], List[Island]]:
    """
    便捷函数：生成矩形拓扑

    Args:
        floor_boundary: 楼层边界多边形
        corridor_width: 走廊宽度（米）
        core_area_ratio: 核心筒占楼层面积比例
        corridor_layout: 走廊布局类型 ('door_side' | 'cross' | 'H' | 'grid')
        entrance_position: 入口位置
        core_tube_override: 复用已有核心筒（跨层共享时使用）
        core_position: 核心筒位置 ('north' | 'south' | 'center' | 'east' | 'west')
        group_seed: organic 模式用的分组种子（用于端头退让/骨架分支选择）

    返回:
        (核心筒, 走廊列表, 岛屿列表)
    """
    generator = RectangularTopologyGenerator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width,
    )

    minx, miny, maxx, maxy = (float(v) for v in floor_boundary.bounds)
    w = maxx - minx
    h = maxy - miny
    floor_area = float(floor_boundary.area)

    if core_tube_override is not None:
        core = core_tube_override
        try:
            cminx, cminy, cmaxx, cmaxy = (float(v) for v in core.polygon.bounds)
            tol = 0.6
            if abs(cmaxy - float(maxy)) <= tol:
                pos = "north"
            elif abs(cminy - float(miny)) <= tol:
                pos = "south"
            elif abs(cmaxx - float(maxx)) <= tol:
                pos = "east"
            elif abs(cminx - float(minx)) <= tol:
                pos = "west"
            else:
                pos = "center"
        except Exception:
            pos = str(core_position or "north").lower()
    else:
        core = CoreTube.create_for_floor(
            floor_boundary.bounds,
            area_ratio=core_area_ratio,
            position=core_position,
        )
        pos = str(core_position or "north").lower()
    if entrance_position is None:
        entrance = ((minx + maxx) / 2.0, float(miny))
    else:
        entrance = entrance_position

    opening_sides: List[str] = []
    if pos in ("north", "south", "east", "west"):
        opening_sides = [{"north": "south", "south": "north", "east": "west", "west": "east"}[pos]]
    else:
        main_axis = "horizontal" if float(w) >= float(h) else "vertical"
        cx, cy = (float(v) for v in core.center)
        ex, ey = (float(v) for v in entrance)
        if main_axis == "horizontal":
            primary = "east" if ex > cx else "west"
            opening_sides = [primary]
            if str(corridor_layout or "").lower() == "organic" and floor_area >= 500.0:
                opening_sides = ["west", "east"]
        else:
            primary = "north" if ey > cy else "south"
            opening_sides = [primary]
            if str(corridor_layout or "").lower() == "organic" and floor_area >= 500.0:
                opening_sides = ["south", "north"]

    try:
        core.set_opening_sides(opening_sides)
    except Exception:
        pass

    return generator.generate(
        core_tube=core,
        corridor_layout=corridor_layout,
        entrance_position=entrance_position,
        group_seed=group_seed,
        force_corridor_boundary_contact=force_corridor_boundary_contact,
    )
