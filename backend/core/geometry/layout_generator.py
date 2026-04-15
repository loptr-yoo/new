"""
layout_generator.py

房间布局生成器主入口

整合流程（MIQP 版）:
1. 适配 RoomSpec → MIQP RoomSpec
2. IslandPartitionSolver 求解（Treemap warm start + CP-SAT + boundary clip）
3. 网格对齐（可选）
4. 约束验证
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon

from .axis_align import snap_to_grid
from .constraint_validator import (
    ConstraintValidator,
    SemanticConstraintValidator,
    SemanticValidationReport,
    ValidationReport,
)
from .island_partition_solver import IslandPartitionSolver
from .island_partition_solver import RoomResult as MIQPRoomResult
from .island_partition_solver import RoomSpec as MIQPRoomSpec
from .island_partition_solver import partition_island, partition_island_semantic
from .room_spec import (
    IslandContext,
    RoomSpec as SemanticRoomSpec,
    SolverConfig,
    ZoneType,
    apply_room_type_defaults,
)

logger = logging.getLogger(__name__)


# ============================================================
# 旧版 RoomSpec（保持外部接口兼容）
# ============================================================


@dataclass
class RoomSpec:
    """房间需求规格（外部接口，兼容旧调用方）"""

    id: str
    room_type: str
    target_area: float
    min_area: float = 0
    max_area: float = float("inf")
    requires_window: bool = False
    min_width: float = 2.0
    adjacent_to: set = field(default_factory=set)
    not_adjacent_to: set = field(default_factory=set)


# ============================================================
# 结果数据类
# ============================================================


@dataclass
class RoomResult:
    """单个房间结果"""

    id: str
    room_type: str
    polygon: Polygon
    area: float
    target_area: float
    area_error: float
    centroid: Tuple[float, float]
    has_window: bool
    facade_length: float
    aspect_ratio: float


@dataclass
class LayoutResult:
    """布局生成结果"""

    rooms: List[RoomResult]
    boundary: Polygon

    total_area: float
    coverage: float

    validation: Union[ValidationReport, SemanticValidationReport]

    generation_time_ms: float

    method: str = "treemap_miqp_v1"


# ============================================================
# 配置
# ============================================================


@dataclass
class GeneratorConfig:
    """生成器配置"""

    use_warm_start: bool = True
    solver: str = "cpsat"  # 'cpsat' | 'gurobi'
    time_limit: float = 10.0

    snap_grid: float = 0.1  # 网格对齐步长，0 = 关闭

    semantic: bool = False  # 是否使用语义感知求解器

    verbose: bool = False


# ============================================================
# 适配函数
# ============================================================


def _convert_room_spec(old: RoomSpec) -> MIQPRoomSpec:
    """旧 RoomSpec → MIQP RoomSpec 字段适配"""
    if old.target_area > 0 and old.min_area > 0:
        tolerance = 1 - old.min_area / old.target_area
    else:
        tolerance = 0.1
    tolerance = max(tolerance, 0.05)

    return MIQPRoomSpec(
        room_id=old.id,
        room_type=old.room_type,
        target_area=old.target_area,
        area_tolerance=tolerance,
        min_width=old.min_width,
        min_depth=getattr(old, 'min_depth', old.min_width),
        adjacency_required=list(old.adjacent_to),
        adjacency_forbidden=list(old.not_adjacent_to),
        window_access=old.requires_window,
    )


# ============================================================
# 主生成器
# ============================================================


class LayoutGenerator:
    """房间布局生成器。使用 Treemap + MIQP。"""

    def __init__(self, config: GeneratorConfig = None):  # type: ignore[assignment]
        self.config = config or GeneratorConfig()

    def generate(
        self,
        boundary: Polygon,
        room_specs: List[RoomSpec],
    ) -> LayoutResult:
        start_time = time.perf_counter()

        if self.config.verbose:
            logger.info("Generating layout for %d rooms (MIQP)...", len(room_specs))

        # ============ Step 1: 适配 + 求解 ============
        miqp_specs = [_convert_room_spec(s) for s in room_specs]

        miqp_results = partition_island(
            island_polygon=boundary,
            room_specs=miqp_specs,
            solver=self.config.solver,
            time_limit=self.config.time_limit,
            use_warm_start=self.config.use_warm_start,
        )

        # ============ Step 2: 网格对齐 ============
        cells = [r.polygon for r in miqp_results]
        if self.config.snap_grid > 0:
            cells = snap_to_grid(cells, self.config.snap_grid)

        # ============ Step 3: 约束验证 ============
        # 用 snap 后的 cells 重建 MIQPRoomResult 以做验证
        snapped_results = []
        for mr, cell in zip(miqp_results, cells):
            minx, miny, maxx, maxy = cell.bounds
            snapped_results.append(
                MIQPRoomResult(
                    room_id=mr.room_id,
                    x=minx,
                    y=miny,
                    width=maxx - minx,
                    depth=maxy - miny,
                )
            )

        validator = ConstraintValidator(boundary, snapped_results, miqp_specs)
        report = validator.validate()

        if self.config.verbose:
            logger.info("Validation:\n%s", report)

        # ============ Step 4: 构建结果 ============
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        exterior = boundary.exterior

        rooms: List[RoomResult] = []
        for spec, sr, cell in zip(room_specs, snapped_results, cells):
            if cell.is_empty:
                rooms.append(
                    RoomResult(
                        id=spec.id,
                        room_type=spec.room_type,
                        polygon=Polygon(),
                        area=0,
                        target_area=spec.target_area,
                        area_error=1.0,
                        centroid=(0, 0),
                        has_window=False,
                        facade_length=0,
                        aspect_ratio=float("inf"),
                    )
                )
                continue

            try:
                intersection = cell.boundary.intersection(exterior)
                facade = float(intersection.length) if not intersection.is_empty else 0.0
            except Exception:
                facade = 0.0

            aspect = self._compute_aspect_ratio(cell)

            rooms.append(
                RoomResult(
                    id=spec.id,
                    room_type=spec.room_type,
                    polygon=cell,
                    area=float(cell.area),
                    target_area=spec.target_area,
                    area_error=(
                        abs(cell.area - spec.target_area) / spec.target_area
                        if spec.target_area > 0
                        else 0.0
                    ),
                    centroid=(float(cell.centroid.x), float(cell.centroid.y)),
                    has_window=facade > 1.0,
                    facade_length=facade,
                    aspect_ratio=aspect,
                )
            )

        total_area = sum(r.area for r in rooms)

        return LayoutResult(
            rooms=rooms,
            boundary=boundary,
            total_area=total_area,
            coverage=total_area / boundary.area if boundary.area > 0 else 0.0,
            validation=report,
            generation_time_ms=elapsed_ms,
        )

    @staticmethod
    def _compute_aspect_ratio(polygon: Polygon) -> float:
        if polygon.is_empty:
            return float("inf")
        mrr = polygon.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)  # type: ignore[union-attr]
        e1 = float(np.linalg.norm(np.array(coords[1]) - np.array(coords[0])))
        e2 = float(np.linalg.norm(np.array(coords[2]) - np.array(coords[1])))
        if min(e1, e2) < 1e-6:
            return float("inf")
        return max(e1, e2) / min(e1, e2)


# ============================================================
# 便捷函数
# ============================================================


def generate_layout(
    boundary: Polygon,
    room_specs: List[RoomSpec],
    verbose: bool = False,
) -> LayoutResult:
    """便捷函数：使用默认配置生成布局"""
    config = GeneratorConfig(verbose=verbose)
    generator = LayoutGenerator(config)
    return generator.generate(boundary, room_specs)


def generate_layout_simple(
    boundary: Polygon,
    room_areas: List[float],
    room_types: List[str] = None,  # type: ignore[assignment]
) -> LayoutResult:
    """简化版：只指定面积"""
    n = len(room_areas)
    if room_types is None:
        room_types = ["room"] * n

    room_specs = [
        RoomSpec(
            id=f"room_{i}",
            room_type=room_types[i] if i < len(room_types) else "room",
            target_area=room_areas[i],
            min_area=room_areas[i] * 0.8,
            max_area=room_areas[i] * 1.2,
        )
        for i in range(n)
    ]

    return generate_layout(boundary, room_specs)


def generate_layout_semantic(
    boundary: Polygon,
    room_specs: List[SemanticRoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
    exterior_walls: Optional[List[str]] = None,
    config: Optional[SolverConfig] = None,
    snap_grid: float = 0.1,
    verbose: bool = False,
) -> LayoutResult:
    """
    语义感知布局生成便捷函数。

    Args:
        boundary: 岛屿边界
        room_specs: 语义增强版房间规格
        adjacency_graph: 邻接图 {room_id: [neighbor_ids]}，
            None 时从 room_specs.adjacency_required 自动构建
        exterior_walls: 外墙方向，None 时默认四面
        config: 求解器配置
        snap_grid: 网格对齐步长，0 = 关闭
        verbose: 调试输出

    Returns:
        LayoutResult
    """
    start_time = time.perf_counter()

    # 应用 room_type 默认值
    for spec in room_specs:
        apply_room_type_defaults(spec)

    # 自动构建邻接图
    if adjacency_graph is None:
        adjacency_graph = {}
        for spec in room_specs:
            neighbors = list(spec.adjacency_required)
            if spec.adjacency_preferred:
                neighbors.extend(spec.adjacency_preferred)
            if neighbors:
                adjacency_graph[spec.room_id] = neighbors

    if exterior_walls is None:
        exterior_walls = ["north", "south", "east", "west"]

    config = config or SolverConfig()
    context = IslandContext(exterior_walls=exterior_walls)

    # 求解
    miqp_results = partition_island_semantic(
        island_polygon=boundary,
        rooms=room_specs,
        adjacency_graph=adjacency_graph,
        exterior_walls=exterior_walls,
        config=config,
    )

    # 网格对齐
    from .axis_align import snap_to_grid as _snap
    cells = [r.polygon for r in miqp_results]
    if snap_grid > 0:
        cells = _snap(cells, snap_grid)

    # 重建 snapped results
    snapped_results = []
    for mr, cell in zip(miqp_results, cells):
        minx, miny, maxx, maxy = cell.bounds
        snapped_results.append(MIQPRoomResult(
            room_id=mr.room_id, x=minx, y=miny,
            width=maxx - minx, depth=maxy - miny,
        ))

    # 语义验证
    validator = SemanticConstraintValidator(
        island=boundary,
        results=snapped_results,
        specs=room_specs,
        island_context=context,
    )
    report = validator.validate()

    if verbose:
        logger.info("Semantic validation:\n%s", report)

    # 构建结果
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    exterior = boundary.exterior

    rooms: List[RoomResult] = []
    for spec, sr, cell in zip(room_specs, snapped_results, cells):
        if cell.is_empty:
            rooms.append(RoomResult(
                id=spec.room_id, room_type=spec.room_type,
                polygon=Polygon(), area=0, target_area=spec.target_area,
                area_error=1.0, centroid=(0, 0),
                has_window=False, facade_length=0,
                aspect_ratio=float("inf"),
            ))
            continue

        try:
            intersection = cell.boundary.intersection(exterior)
            facade = float(intersection.length) if not intersection.is_empty else 0.0
        except Exception:
            facade = 0.0

        aspect = LayoutGenerator._compute_aspect_ratio(cell)

        rooms.append(RoomResult(
            id=spec.room_id,
            room_type=spec.room_type,
            polygon=cell,
            area=float(cell.area),
            target_area=spec.target_area,
            area_error=(
                abs(cell.area - spec.target_area) / spec.target_area
                if spec.target_area > 0 else 0.0
            ),
            centroid=(float(cell.centroid.x), float(cell.centroid.y)),
            has_window=facade > 1.0,
            facade_length=facade,
            aspect_ratio=aspect,
        ))

    total_area = sum(r.area for r in rooms)

    return LayoutResult(
        rooms=rooms,
        boundary=boundary,
        total_area=total_area,
        coverage=total_area / boundary.area if boundary.area > 0 else 0.0,
        validation=report,
        generation_time_ms=elapsed_ms,
        method="semantic_treemap_miqp_v1",
    )


# ============================================================
# V2 多岛屿管线
# ============================================================


class LayoutGenerationError(Exception):
    """布局生成错误基类"""
    pass


class TopologyError(LayoutGenerationError):
    """拓扑生成错误"""
    pass


class PartitionError(LayoutGenerationError):
    """岛屿划分错误"""
    pass


@dataclass
class LayoutResultV2:
    """布局生成结果（V2 多岛屿版）"""

    # 拓扑（使用 Any 避免循环导入，运行时为 CoreTube/Corridor/Island）
    core_tube: "Any"
    corridors: List["Any"]
    islands: List["Any"]

    # 分配
    assignments: Dict[str, "Any"]

    # 布局
    room_layouts: List[RoomResult]

    # 验证
    validation: "Any"  # ValidationReport | SemanticValidationReport

    # 元数据
    generation_time_ms: float
    warnings: List[str] = field(default_factory=list)

    # 拓扑边（最终房间层 edge_set）
    edge_set: Dict[FrozenSet[str], str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid and len(self.warnings) == 0


def _check_cross_island_adjacency(
    rooms: List[SemanticRoomSpec],
    room_to_island: Dict[str, str],
    islands_by_id: dict,
) -> List[str]:
    """检查跨岛屿的必须邻接约束"""
    warnings = []

    for room in rooms:
        for adj_id in room.adjacency_required:
            if adj_id not in room_to_island or room.room_id not in room_to_island:
                continue
            if room_to_island.get(room.room_id) != room_to_island.get(adj_id):
                island_a = islands_by_id.get(room_to_island[room.room_id])
                island_b = islands_by_id.get(room_to_island[adj_id])
                if island_a and island_b:
                    if not island_a.polygon.touches(island_b.polygon):
                        msg = (
                            f"Required adjacency {room.room_id}-{adj_id} "
                            f"spans non-adjacent islands"
                        )
                        if msg not in warnings:
                            warnings.append(msg)

    return warnings


def check_connectivity(
    room_polygons: List[Tuple[str, Polygon]],
    core_tube_polygon: Optional[Polygon] = None,
    buffer_tolerance: float = 0.1,
    min_shared_length: float = 0.5,
) -> List[str]:
    """
    检查每个房间是否通过邻接关系可达核心筒。

    BFS from core_tube，沿 shared-edge 遍历。
    返回不可达房间的 room_id 列表。
    """
    if not room_polygons:
        return []

    # 构建 ID → polygon 映射
    all_items: List[Tuple[str, Polygon]] = list(room_polygons)
    if core_tube_polygon is not None and not core_tube_polygon.is_empty:
        all_items.append(("_core", core_tube_polygon))

    # 构建邻接图
    adj: Dict[str, set] = {rid: set() for rid, _ in all_items}
    for i in range(len(all_items)):
        for j in range(i + 1, len(all_items)):
            rid_a, poly_a = all_items[i]
            rid_b, poly_b = all_items[j]
            try:
                shared_boundary = poly_a.boundary.intersection(
                    poly_b.buffer(buffer_tolerance)
                )
                if hasattr(shared_boundary, 'length') and shared_boundary.length > min_shared_length:
                    adj[rid_a].add(rid_b)
                    adj[rid_b].add(rid_a)
            except Exception:
                continue

    # BFS from core (or from first room if no core)
    start = "_core" if "_core" in adj else (room_polygons[0][0] if room_polygons else None)
    if start is None:
        return []

    visited: set = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj.get(node, set()) - visited)

    unreachable = [rid for rid, _ in room_polygons if rid not in visited]
    return unreachable


def _build_edge_set_from_rects(
    rects: Dict[str, Tuple[float, float, float, float]],
    tol: float = 0.06,
    min_shared_length: float = 0.3,
) -> Dict[FrozenSet[str], str]:
    edge_set: Dict[FrozenSet[str], str] = {}
    ids = list(rects.keys())

    for i in range(len(ids)):
        id_a = ids[i]
        ax, ay, aw, ah = rects[id_a]
        for j in range(i + 1, len(ids)):
            id_b = ids[j]
            bx, by, bw, bh = rects[id_b]

            if abs(ax + aw - bx) < tol or abs(bx + bw - ax) < tol:
                y0 = max(ay, by)
                y1 = min(ay + ah, by + bh)
                if (y1 - y0) > min_shared_length:
                    edge_set[frozenset({id_a, id_b})] = "vertical"
                continue

            if abs(ay + ah - by) < tol or abs(by + bh - ay) < tol:
                x0 = max(ax, bx)
                x1 = min(ax + aw, bx + bw)
                if (x1 - x0) > min_shared_length:
                    edge_set[frozenset({id_a, id_b})] = "horizontal"

    return edge_set


def check_connectivity_topological(
    edge_set: Dict[FrozenSet[str], str],
    all_zone_ids: List[str],
    entrance_zone_id: Optional[str] = None,
) -> List[str]:
    adj: Dict[str, set] = {z: set() for z in all_zone_ids}
    for edge_key in edge_set.keys():
        id_a, id_b = tuple(edge_key)
        if id_a in adj and id_b in adj:
            adj[id_a].add(id_b)
            adj[id_b].add(id_a)

    start = entrance_zone_id
    if start is None:
        start = next((z for z in all_zone_ids if "corridor" in z), all_zone_ids[0] if all_zone_ids else None)
    if start is None or start not in adj:
        return list(all_zone_ids)

    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj[node] - visited)

    return [z for z in all_zone_ids if z not in visited]


def generate_layout_v2(
    floor_boundary: Polygon,
    room_specs: List[SemanticRoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "cross",
    entrance_position: Optional[Tuple[float, float]] = None,
    config: Optional[SolverConfig] = None,
    snap_grid: float = 0.1,
    verbose: bool = False,
    shared_core_tube: Optional[Any] = None,  # CoreTube, 跨层共享
) -> LayoutResultV2:
    """
    生成多岛屿布局（V2 API）

    完整管线：
    1. 矩形拓扑生成（RectangularTopologyGenerator）
    2. 房间-岛屿分配（IslandRoomAssigner）
    3. 跨岛屿邻接验证
    4. 岛屿内房间划分（Treemap + MIQP）
    5. 网格对齐 + 验证

    Args:
        floor_boundary: 楼层边界多边形
        room_specs: 语义房间规格列表
        adjacency_graph: 邻接关系图 {room_id: [adjacent_room_ids]}
        corridor_width: 走廊宽度（米）
        core_area_ratio: 核心筒占楼层面积比例
        corridor_layout: 走廊布局类型 ('cross' | 'H' | 'grid')
        entrance_position: 入口位置
        config: 求解器配置
        snap_grid: 网格对齐步长，0 = 关闭
        verbose: 调试输出

    Returns:
        LayoutResultV2

    Raises:
        TopologyError: 拓扑生成失败
        AssignmentError: 房间分配失败（来自 island_room_assigner）
        PartitionError: 岛屿划分失败
    """
    # 延迟导入，避免循环依赖
    from .island_room_assigner import assign_rooms_to_islands, DegradationSummary
    from .topology_generator import (
        CoreTube,
        Island,
        RectangularTopologyGenerator,
        generate_rectangular_topology,
    )

    start_time = time.perf_counter()

    # 应用 room_type 默认值
    for spec in room_specs:
        apply_room_type_defaults(spec)

    # 自动构建邻接图
    if adjacency_graph is None:
        adjacency_graph = {}
        for spec in room_specs:
            neighbors = list(spec.adjacency_required)
            if spec.adjacency_preferred:
                neighbors.extend(spec.adjacency_preferred)
            if neighbors:
                adjacency_graph[spec.room_id] = neighbors

    config = config or SolverConfig()
    warnings: List[str] = []

    # ========== Phase 1: 拓扑生成 ==========
    try:
        core_tube, corridors, islands = generate_rectangular_topology(
            floor_boundary=floor_boundary,
            corridor_width=corridor_width,
            core_area_ratio=core_area_ratio,
            corridor_layout=corridor_layout,
            entrance_position=entrance_position,
            core_tube_override=shared_core_tube,
        )
    except Exception as e:
        raise TopologyError(f"Failed to generate topology: {e}") from e

    if verbose:
        logger.info(
            "Topology: %d islands, %d corridors",
            len(islands), len(corridors),
        )

    # ========== Phase 2: 房间-岛屿分配 ==========
    assignments, degradation = assign_rooms_to_islands(
        islands=islands,
        rooms=room_specs,
        adjacency_graph=adjacency_graph,
    )

    # 收集降级 warnings
    if degradation.skipped_rooms:
        warnings.append(f"Skipped rooms (no island): {degradation.skipped_rooms}")
    if degradation.force_shrunk:
        warnings.append(f"Force-shrunk rooms: {degradation.force_shrunk}")

    if verbose:
        for island_id, result in assignments.items():
            logger.info(
                "  %s: %d rooms, %.1f%% utilization",
                island_id, len(result.rooms), result.utilization * 100,
            )

    # ========== Phase 3: 跨岛屿邻接验证 ==========
    # 构建 room_to_island 映射
    room_to_island: Dict[str, str] = {}
    for island_id, assignment in assignments.items():
        for room in assignment.rooms:
            room_to_island[room.room_id] = island_id

    islands_by_id = {i.id: i for i in islands}
    adj_warnings = _check_cross_island_adjacency(
        room_specs, room_to_island, islands_by_id,
    )
    warnings.extend(adj_warnings)

    # ========== Phase 4: 岛屿内划分 ==========
    all_miqp_results: List[MIQPRoomResult] = []
    all_specs_ordered: List[SemanticRoomSpec] = []

    for island_id, assignment in assignments.items():
        island = islands_by_id[island_id]

        # 构建岛屿内邻接子图
        island_room_ids = {r.room_id for r in assignment.rooms}
        island_adjacency = {
            rid: [a for a in adj if a in island_room_ids]
            for rid, adj in adjacency_graph.items()
            if rid in island_room_ids
        }

        exterior_walls = island.exterior_walls or []
        corridor_edges = island.corridor_edges if hasattr(island, 'corridor_edges') else []

        try:
            miqp_results = partition_island_semantic(
                island_polygon=island.polygon,
                rooms=assignment.rooms,
                adjacency_graph=island_adjacency,
                exterior_walls=exterior_walls,
                config=config,
                corridor_edges=corridor_edges,
            )
            all_miqp_results.extend(miqp_results)
            all_specs_ordered.extend(assignment.rooms)
        except Exception as e:
            # MIQP 失败 → fallback 到基础求解器
            logger.warning(
                f"Semantic MIQP failed for {island_id}: {e}, "
                f"falling back to basic solver"
            )
            warnings.append(f"MIQP fallback on {island_id}")
            try:
                basic_specs = [
                    MIQPRoomSpec(
                        room_id=r.room_id, room_type=r.room_type,
                        target_area=r.target_area, area_tolerance=0.3,
                        min_width=r.min_width, min_depth=r.min_depth,
                    ) for r in assignment.rooms
                ]
                miqp_results = partition_island(island.polygon, basic_specs)
                all_miqp_results.extend(miqp_results)
                all_specs_ordered.extend(assignment.rooms)
            except Exception as e2:
                logger.error(f"Basic solver also failed for {island_id}: {e2}")
                warnings.append(f"Partition failed completely on {island_id}")

    # ========== Phase 5: 网格对齐 + 验证 ==========
    cells = [r.polygon for r in all_miqp_results]
    if snap_grid > 0:
        cells = snap_to_grid(cells, snap_grid)

    # 重建 snapped results
    snapped_results = []
    for mr, cell in zip(all_miqp_results, cells):
        minx, miny, maxx, maxy = cell.bounds
        snapped_results.append(MIQPRoomResult(
            room_id=mr.room_id, x=minx, y=miny,
            width=maxx - minx, depth=maxy - miny,
        ))

    # 语义验证（用整个楼层作为边界）
    exterior_walls_all = ["north", "south", "east", "west"]
    context = IslandContext(exterior_walls=exterior_walls_all)
    validator = SemanticConstraintValidator(
        island=floor_boundary,
        results=snapped_results,
        specs=all_specs_ordered,
        island_context=context,
    )
    report = validator.validate()

    if verbose:
        logger.info("Validation: %s", report)

    # ========== 构建结果 ==========
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    exterior = floor_boundary.exterior

    rooms: List[RoomResult] = []
    for spec, sr, cell in zip(all_specs_ordered, snapped_results, cells):
        if cell.is_empty:
            rooms.append(RoomResult(
                id=spec.room_id, room_type=spec.room_type,
                polygon=Polygon(), area=0, target_area=spec.target_area,
                area_error=1.0, centroid=(0, 0),
                has_window=False, facade_length=0,
                aspect_ratio=float("inf"),
            ))
            continue

        try:
            intersection = cell.boundary.intersection(exterior)
            facade = float(intersection.length) if not intersection.is_empty else 0.0
        except Exception:
            facade = 0.0

        aspect = LayoutGenerator._compute_aspect_ratio(cell)

        rooms.append(RoomResult(
            id=spec.room_id,
            room_type=spec.room_type,
            polygon=cell,
            area=float(cell.area),
            target_area=spec.target_area,
            area_error=(
                abs(cell.area - spec.target_area) / spec.target_area
                if spec.target_area > 0 else 0.0
            ),
            centroid=(float(cell.centroid.x), float(cell.centroid.y)),
            has_window=facade > 1.0,
            facade_length=facade,
            aspect_ratio=aspect,
        ))

    # ========== Phase 6: 连通性检查（拓扑 BFS，零浮点缓冲） ==========
    rects: Dict[str, Tuple[float, float, float, float]] = {}
    if core_tube is not None and not core_tube.polygon.is_empty:
        minx, miny, maxx, maxy = core_tube.polygon.bounds
        rects["core_tube"] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
    for c in corridors:
        if hasattr(c, "polygon") and c.polygon is not None and not c.polygon.is_empty:
            minx, miny, maxx, maxy = c.polygon.bounds
            rects[c.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
    room_ids = []
    for r in rooms:
        if not r.polygon.is_empty:
            minx, miny, maxx, maxy = r.polygon.bounds
            rects[r.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
            room_ids.append(r.id)

    edge_set = _build_edge_set_from_rects(rects, tol=0.06)
    unreachable_zones = check_connectivity_topological(edge_set, list(rects.keys()))
    unreachable_rooms = [z for z in unreachable_zones if z in room_ids]
    if unreachable_rooms:
        warnings.append(f"Unreachable rooms: {unreachable_rooms}")

    return LayoutResultV2(
        core_tube=core_tube,
        corridors=corridors,
        islands=islands,
        assignments=assignments,
        room_layouts=rooms,
        edge_set=edge_set,
        validation=report,
        generation_time_ms=elapsed_ms,
        warnings=warnings,
    )
