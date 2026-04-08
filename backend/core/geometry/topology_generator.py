from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
from shapely.affinity import scale as scale_geom
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union
import shapely.ops as ops

from ...models import FloorAllocation
from .building_types import FloorSkeleton
from .room_spec import ZoneType

logger = logging.getLogger(__name__)

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


# ═══════════════════════════════════════════════════════════════════════════
# 新版矩形拓扑生成器（Phase 2）
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CoreTube:
    """
    核心筒定义

    设计原则：
    - 紧凑矩形，包含电梯、楼梯、设备间
    - 占楼层面积 5-10%
    - 位置靠近中心或入口
    """
    polygon: Polygon
    center: Tuple[float, float]
    width: float
    depth: float

    @classmethod
    def create(
        cls,
        center: Tuple[float, float],
        width: float,
        depth: float,
    ) -> CoreTube:
        """创建矩形核心筒"""
        cx, cy = center
        polygon = box(
            cx - width / 2, cy - depth / 2,
            cx + width / 2, cy + depth / 2,
        )
        return cls(polygon=polygon, center=center, width=width, depth=depth)

    @classmethod
    def create_for_floor(
        cls,
        floor_bounds: Tuple[float, float, float, float],
        area_ratio: float = 0.08,
        aspect_ratio: float = 1.0,
        position: str = "center",
    ) -> CoreTube:
        """根据楼层自动创建核心筒"""
        x_min, y_min, x_max, y_max = floor_bounds
        floor_area = (x_max - x_min) * (y_max - y_min)

        core_area = floor_area * area_ratio
        width = np.sqrt(core_area * aspect_ratio)
        depth = core_area / width

        if position == "center":
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
        elif position == "entrance":
            cx = (x_min + x_max) / 2
            cy = y_min + depth / 2 + 3  # 距离入口 3m
        else:
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

        return cls.create((cx, cy), width, depth)


@dataclass
class Corridor:
    """走廊定义"""
    id: str
    centerline: LineString
    width: float
    orientation: str  # 'horizontal' | 'vertical'
    polygon: Polygon = field(init=False)

    def __post_init__(self):
        self.polygon = self.centerline.buffer(
            self.width / 2,
            cap_style=3,  # flat cap
            join_style=2,  # mitre join
        )


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
        min_island_area: float = 20.0,
        grid_alignment: float = 0.5,
    ):
        self.floor = floor_boundary
        self.corridor_width = corridor_width
        self.min_island_area = min_island_area
        self.grid_alignment = grid_alignment

        self.bounds = floor_boundary.bounds
        self.x_min, self.y_min, self.x_max, self.y_max = self.bounds
        self.floor_width = self.x_max - self.x_min
        self.floor_depth = self.y_max - self.y_min

    def generate(
        self,
        core_tube: Optional[CoreTube] = None,
        corridor_layout: str = "cross",
        entrance_position: Optional[Tuple[float, float]] = None,
    ) -> Tuple[CoreTube, List[Corridor], List[Island]]:
        """
        生成矩形拓扑

        参数:
            core_tube: 核心筒（如果为 None 则自动创建）
            corridor_layout: 走廊布局类型 ('cross' | 'H' | 'grid')
            entrance_position: 入口位置

        返回:
            (核心筒, 走廊列表, 岛屿列表)
        """
        # Step 1: 创建核心筒
        if core_tube is None:
            core_tube = CoreTube.create_for_floor(self.bounds)

        # Step 2: 生成走廊
        if corridor_layout == "cross":
            corridors = self._generate_cross_corridors(core_tube)
        elif corridor_layout == "H":
            corridors = self._generate_h_corridors(core_tube)
        elif corridor_layout == "grid":
            corridors = self._generate_grid_corridors(core_tube)
        else:
            corridors = self._generate_cross_corridors(core_tube)

        # Step 3: 生成岛屿
        islands = self._generate_islands(core_tube, corridors)

        # Step 4: 解决矩形化后的重叠
        islands = self._resolve_overlaps(islands)

        # Step 5: 计算语义属性
        if entrance_position is None:
            entrance_position = (
                (self.x_min + self.x_max) / 2,
                self.y_min,
            )
        self._compute_island_semantics(islands, core_tube, entrance_position)

        # Step 6: 验证
        self._validate(islands)

        return core_tube, corridors, islands

    def _generate_cross_corridors(self, core: CoreTube) -> List[Corridor]:
        """十字走廊布局"""
        cx, cy = core.center

        h_corridor = Corridor(
            id="corridor_h",
            centerline=LineString([(self.x_min, cy), (self.x_max, cy)]),
            width=self.corridor_width,
            orientation="horizontal",
        )

        v_corridor = Corridor(
            id="corridor_v",
            centerline=LineString([(cx, self.y_min), (cx, self.y_max)]),
            width=self.corridor_width,
            orientation="vertical",
        )

        return [h_corridor, v_corridor]

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

    def _align(self, value: float) -> float:
        """对齐到网格"""
        return round(value / self.grid_alignment) * self.grid_alignment

    def _generate_islands(
        self,
        core: CoreTube,
        corridors: List[Corridor],
    ) -> List[Island]:
        """生成矩形岛屿"""
        # 合并所有要减去的区域
        subtract_regions = [core.polygon]
        for corridor in corridors:
            subtract_regions.append(corridor.polygon)

        subtract_union = unary_union(subtract_regions)

        # 从楼层中减去
        remaining = self.floor.difference(subtract_union)

        # 提取多边形
        if remaining.is_empty:
            return []

        polygons = _as_polygons(remaining)

        # 创建岛屿
        islands = []
        for i, poly in enumerate(polygons):
            if poly.area < self.min_island_area:
                continue

            # 矩形化：取包围盒
            rect_poly = box(*poly.bounds)

            # 裁剪到楼层边界内
            rect_poly = rect_poly.intersection(self.floor)
            if not isinstance(rect_poly, Polygon) or rect_poly.is_empty:
                continue

            # 检查矩形化损失
            if poly.area / rect_poly.area < 0.95:
                logger.warning(f"Island {i} has significant non-rectangular area loss")

            islands.append(Island(
                id=f"island_{i}",
                polygon=rect_poly,
            ))

        return islands

    def _resolve_overlaps(self, islands: List[Island]) -> List[Island]:
        """解决矩形化后的岛屿重叠"""
        for i, island_a in enumerate(islands):
            for j in range(i + 1, len(islands)):
                island_b = islands[j]
                if island_a.polygon.intersects(island_b.polygon):
                    overlap = island_a.polygon.intersection(island_b.polygon)
                    if hasattr(overlap, "area") and overlap.area > 0.1:
                        # 较小的岛屿缩减
                        if island_a.area < island_b.area:
                            new_poly = island_a.polygon.difference(overlap)
                            if isinstance(new_poly, Polygon) and not new_poly.is_empty:
                                island_a = Island(id=island_a.id, polygon=box(*new_poly.bounds))
                                islands[i] = island_a
                        else:
                            new_poly = island_b.polygon.difference(overlap)
                            if isinstance(new_poly, Polygon) and not new_poly.is_empty:
                                island_b = Island(id=island_b.id, polygon=box(*new_poly.bounds))
                                islands[j] = island_b

        # 过滤掉过小的岛屿
        return [isl for isl in islands if isl.area >= self.min_island_area]

    def _compute_island_semantics(
        self,
        islands: List[Island],
        core: CoreTube,
        entrance: Tuple[float, float],
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

            # 距离
            island.distance_to_entrance = island_center.distance(entrance_point)
            island.distance_to_core = island_center.distance(core_center)

            # 推荐分区
            island.suggested_zone = self._suggest_zone(island)

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


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════


def generate_rectangular_topology(
    floor_boundary: Polygon,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "cross",
    entrance_position: Optional[Tuple[float, float]] = None,
) -> Tuple[CoreTube, List[Corridor], List[Island]]:
    """
    便捷函数：生成矩形拓扑

    返回:
        (核心筒, 走廊列表, 岛屿列表)
    """
    generator = RectangularTopologyGenerator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width,
    )

    core = CoreTube.create_for_floor(
        floor_boundary.bounds,
        area_ratio=core_area_ratio,
    )

    return generator.generate(
        core_tube=core,
        corridor_layout=corridor_layout,
        entrance_position=entrance_position,
    )
