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

        cx = (x_min + x_max) / 2
        if position == "north":
            cy = y_max - depth / 2  # 紧贴北墙
        elif position == "south":
            cy = y_min + depth / 2  # 紧贴南墙
        elif position == "center":
            cy = (y_min + y_max) / 2
        elif position == "entrance":
            cy = y_min + depth / 2 + 3
        else:
            cy = y_max - depth / 2  # 默认北墙

        cx = round(cx / grid_alignment) * grid_alignment
        cx = min(max(cx, x_min + width / 2), x_max - width / 2)

        if position == "north":
            cy = np.floor(cy / grid_alignment) * grid_alignment
        elif position == "south":
            cy = np.ceil(cy / grid_alignment) * grid_alignment
        else:
            cy = round(cy / grid_alignment) * grid_alignment
        cy = min(max(cy, y_min + depth / 2), y_max - depth / 2)

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
        corridor_layout: str = "door_side",
        entrance_position: Optional[Tuple[float, float]] = None,
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

        # Step 2: 生成走廊
        if corridor_layout == "door_side":
            corridors = self._generate_cross_corridors(core_tube)
        elif corridor_layout == "cross":
            corridors = self._generate_cross_corridors(core_tube)
        elif corridor_layout == "H":
            corridors = self._generate_h_corridors(core_tube)
        elif corridor_layout == "grid":
            corridors = self._generate_grid_corridors(core_tube)
        else:
            corridors = self._generate_cross_corridors(core_tube)

        # Step 2.5: 核心筒对齐走廊交叉区，确保减去后产生纯矩形岛屿
        core_tube = self._align_core_to_corridors(core_tube, corridors)

        # Step 2.6: 走廊裁剪核心筒（避免几何重叠；安全提取多边形碎片）
        core_poly_for_cut = core_tube.polygon.buffer(1e-4, join_style="mitre")
        cut_corridors: List[Corridor] = []
        for corridor in corridors:
            try:
                diff = corridor.polygon.difference(core_poly_for_cut).simplify(0.01)
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
                c.polygon = poly
                cut_corridors.append(c)
        corridors = cut_corridors

        # Step 2.7: 计算排除区域（核心筒+走廊），存为实例变量供后续方法使用
        self._subtract_union = unary_union(
            [core_tube.polygon] + [c.polygon for c in corridors]
        )

        # Step 3: 生成岛屿（网格切片：保证轴对齐矩形）
        island_polys, cell_map, edge_set_islands = self._generate_perfect_rectangular_islands(
            self._subtract_union
        )
        self._edge_set_islands = edge_set_islands
        islands = [Island(id=f"island_{i}", polygon=p) for i, p in enumerate(island_polys)]

        # Step 4: 解决矩形化后的重叠（理论上不应有重叠；保留防御）
        islands = self._resolve_overlaps(islands)

        # Step 5: 计算语义属性（入口位置投影+幻觉检测）
        if entrance_position is None:
            entrance_position = (
                (self.x_min + self.x_max) / 2,
                self.y_min,
            )
        else:
            entrance_position = self._project_entrance_to_boundary(entrance_position)
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
        core_position: 核心筒位置 ('north' | 'south' | 'center')

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
            position=core_position,
        )

    return generator.generate(
        core_tube=core,
        corridor_layout=corridor_layout,
        entrance_position=entrance_position,
    )
