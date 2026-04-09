from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union

from .room_spec import ZoneType

logger = logging.getLogger(__name__)

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
    elevator: Optional[Polygon] = None
    staircase: Optional[Polygon] = None
    elevator_area: float = 0.0
    staircase_area: float = 0.0

    @classmethod
    def create(
        cls,
        center: Tuple[float, float],
        width: float,
        depth: float,
        elevator_ratio: float = 0.62,
    ) -> CoreTube:
        """创建矩形核心筒，自动拆分 elevator + staircase"""
        cx, cy = center
        polygon = box(
            cx - width / 2, cy - depth / 2,
            cx + width / 2, cy + depth / 2,
        )

        # 沿短边方向切一刀
        if width <= depth:
            # 短边是 width → 沿 y 方向切
            split_y = cy - depth / 2 + depth * elevator_ratio
            elevator = box(cx - width / 2, cy - depth / 2, cx + width / 2, split_y)
            staircase = box(cx - width / 2, split_y, cx + width / 2, cy + depth / 2)
        else:
            # 短边是 depth → 沿 x 方向切
            split_x = cx - width / 2 + width * elevator_ratio
            elevator = box(cx - width / 2, cy - depth / 2, split_x, cy + depth / 2)
            staircase = box(split_x, cy - depth / 2, cx + width / 2, cy + depth / 2)

        return cls(
            polygon=polygon, center=center, width=width, depth=depth,
            elevator=elevator, staircase=staircase,
            elevator_area=float(elevator.area),
            staircase_area=float(staircase.area),
        )

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
            cy = y_min + depth / 2 + 3
        else:
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

        core = cls.create((cx, cy), width, depth)

        # 边界安全：确保子区域不超出楼层
        floor_poly = box(x_min, y_min, x_max, y_max)
        if core.elevator is not None and not floor_poly.contains(core.elevator):
            logger.warning("Elevator extends beyond floor boundary, shrinking")
            clipped = core.elevator.intersection(floor_poly)
            if not clipped.is_empty and isinstance(clipped, Polygon):
                core.elevator = clipped
                core.elevator_area = float(clipped.area)
            else:
                core.elevator = None
                core.elevator_area = 0.0
        if core.staircase is not None and not floor_poly.contains(core.staircase):
            logger.warning("Staircase extends beyond floor boundary, shrinking")
            clipped = core.staircase.intersection(floor_poly)
            if not clipped.is_empty and isinstance(clipped, Polygon):
                core.staircase = clipped
                core.staircase_area = float(clipped.area)
            else:
                core.staircase = None
                core.staircase_area = 0.0

        return core


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
            cap_style="flat",
            join_style="mitre",
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
        self._compute_island_semantics(islands, core_tube, entrance_position, corridors)

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
                    # 确保结果是矩形
                    if not self._is_rectangular(rect):
                        rect = box(*rect.bounds)
                    # 裁剪到楼层边界内
                    rect = rect.intersection(self.floor)
                    if isinstance(rect, Polygon) and not rect.is_empty and rect.area >= self.min_island_area:
                        islands.append(Island(id=f"island_{idx}", polygon=rect))
                        idx += 1

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


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════


def generate_rectangular_topology(
    floor_boundary: Polygon,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "cross",
    entrance_position: Optional[Tuple[float, float]] = None,
    core_tube_override: Optional[CoreTube] = None,
) -> Tuple[CoreTube, List[Corridor], List[Island]]:
    """
    便捷函数：生成矩形拓扑

    Args:
        floor_boundary: 楼层边界多边形
        corridor_width: 走廊宽度（米）
        core_area_ratio: 核心筒占楼层面积比例
        corridor_layout: 走廊布局类型 ('cross' | 'H' | 'grid')
        entrance_position: 入口位置
        core_tube_override: 复用已有核心筒（跨层共享时使用）

    返回:
        (核心筒, 走廊列表, 岛屿列表)
    """
    generator = RectangularTopologyGenerator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width,
    )

    if core_tube_override is not None:
        core = core_tube_override
    else:
        core = CoreTube.create_for_floor(
            floor_boundary.bounds,
            area_ratio=core_area_ratio,
        )

    return generator.generate(
        core_tube=core,
        corridor_layout=corridor_layout,
        entrance_position=entrance_position,
    )
