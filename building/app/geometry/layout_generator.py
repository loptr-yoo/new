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
import hashlib
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

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
from .island_partition_solver import (
    partition_island,
    partition_island_semantic,
    partition_island_semantic_with_metadata,
)
from .coverage_debt_planner import CoverageDebtPolicy, build_coverage_debt_plan
from .core_contracts import (
    CORE_OVERLAP_EPSILON_AREA,
    build_core_footprint_contract,
    reconcile_core_area_for_budget,
    validate_core_exclusion,
)
from .exceptions import LayoutGeometryInvariantError
from .postprocessor import LayoutCoverageError, LayoutTopologyError, SemanticInvalidError
from .residual_sponge import (
    DoorPreflightResult,
    ResidualDecision,
    classify_residual_piece,
)
from .room_spec import (
    IslandContext,
    RoomSpec as SemanticRoomSpec,
    SolverConfig,
    ZoneType,
    apply_room_type_defaults,
)
from .topology_generator import _is_axis_aligned_polygon, _safe_snap_polygon_like
from .topology_snapshot import (
    FloorTopologySnapshot,
    TopologySnapshot,
    snapshot_floor_to_runtime,
    validate_snapshot_for_floor,
)

logger = logging.getLogger(__name__)

try:
    from shapely.validation import make_valid as _make_valid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _make_valid = None  # type: ignore[assignment]


def estimate_corridor_area_upper(
    floor_poly: Polygon,
    cw: float,
    corridor_layout: str,
) -> float:
    if cw <= 0:
        return 0.0
    minx, miny, maxx, maxy = floor_poly.bounds
    w = float(maxx - minx)
    h = float(maxy - miny)
    if w <= 0 or h <= 0:
        return 0.0
    layout_l = (corridor_layout or "").lower()
    if layout_l == "cross":
        return max(0.0, cw * w + cw * h - cw * cw)
    if layout_l == "h":
        return max(0.0, cw * w + cw * h * 1.5)
    if layout_l == "grid":
        return max(0.0, cw * (w + h) * 2.0)
    return max(0.0, cw * (w + h))


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
    is_dummy: bool = False
    target_area_raw: Optional[float] = None


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
                is_dummy=bool(getattr(spec, "is_dummy", False)),
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
            is_dummy=bool(getattr(spec, "is_dummy", False)),
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

    corridor_layout: str = ""
    solver_metadata: Dict[str, Any] = field(default_factory=dict)
    synthetic_rooms: List[Dict[str, Any]] = field(default_factory=list)
    required_adjacency: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid and len(self.warnings) == 0


@dataclass
class CoverageGapResult:
    total_gap_area: float
    max_gap_area: float
    gap_pieces: List[Polygon] = field(default_factory=list)
    raw_total_gap_area: float = 0.0
    raw_max_gap_area: float = 0.0
    raw_gap_pieces: List[Polygon] = field(default_factory=list)
    gap_erosion_tolerance: float = 0.0
    ignored_micro_gap_total: float = 0.0
    ignored_micro_gap_count: int = 0


def _polygon_pieces_only(geom: Any, *, min_area: float = 1e-6) -> List[Polygon]:
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return []
    if isinstance(geom, Polygon):
        return [geom] if float(getattr(geom, "area", 0.0)) > float(min_area) else []
    if isinstance(geom, MultiPolygon):
        return [
            p for p in geom.geoms
            if isinstance(p, Polygon) and float(getattr(p, "area", 0.0)) > float(min_area)
        ]
    pieces: List[Polygon] = []
    for child in getattr(geom, "geoms", []) or []:
        pieces.extend(_polygon_pieces_only(child, min_area=min_area))
    return pieces


def _valid_entity_polygons(entities: List[Any]) -> List[Polygon]:
    polys: List[Polygon] = []
    for entity in entities:
        if isinstance(entity, dict):
            poly = entity.get("polygon")
        else:
            poly = entity if isinstance(entity, Polygon) else getattr(entity, "polygon", None)
        if isinstance(poly, Polygon) and (not poly.is_empty) and float(poly.area) > 1e-6:
            polys.append(poly)
    return polys


def compute_layout_coverage_gap(
    *,
    floor_boundary: Polygon,
    rooms: Optional[List[Any]] = None,
    corridors: Optional[List[Any]] = None,
    core_tube: Optional[Any] = None,
    coverage_features: Optional[List[Any]] = None,
    min_piece_area: float = 1e-6,
    gap_erosion_tolerance: float = 0.02,
    micro_gap_area_threshold: float = 0.1,
) -> CoverageGapResult:
    """Lightweight polygon coverage preflight; never builds walls or graph nodes."""
    if floor_boundary is None or floor_boundary.is_empty:
        return CoverageGapResult(total_gap_area=0.0, max_gap_area=0.0, gap_pieces=[])

    entities: List[Any] = []
    for room in rooms or []:
        if str(getattr(room, "room_type", "") or "").lower() == "void":
            continue
        if bool(getattr(room, "skip_solver", False)):
            continue
        entities.append(room)
    entities.extend(corridors or [])
    entities.extend(coverage_features or [])
    if core_tube is not None:
        poly = getattr(core_tube, "polygon", None)
        if isinstance(poly, Polygon):
            entities.append(poly)

    polys = _valid_entity_polygons(entities)
    if not polys:
        return CoverageGapResult(
            total_gap_area=float(floor_boundary.area),
            max_gap_area=float(floor_boundary.area),
            gap_pieces=[floor_boundary],
            raw_total_gap_area=float(floor_boundary.area),
            raw_max_gap_area=float(floor_boundary.area),
            raw_gap_pieces=[floor_boundary],
            gap_erosion_tolerance=float(gap_erosion_tolerance),
        )

    try:
        occupied = unary_union(polys).intersection(floor_boundary)
        gap_geom = floor_boundary.difference(occupied)
    except Exception:
        repaired = [p.buffer(0) for p in polys if isinstance(p, Polygon) and not p.is_empty]
        occupied = unary_union(repaired).intersection(floor_boundary) if repaired else Polygon()
        gap_geom = floor_boundary.difference(occupied)

    raw_pieces = _polygon_pieces_only(gap_geom, min_area=min_piece_area)
    raw_total_gap = float(sum(float(p.area) for p in raw_pieces))
    raw_max_gap = float(max((float(p.area) for p in raw_pieces), default=0.0))
    ignored_micro = [
        p for p in raw_pieces
        if float(p.area) <= float(micro_gap_area_threshold)
    ]

    gap_for_check = gap_geom
    if float(gap_erosion_tolerance) > 0.0:
        try:
            gap_for_check = gap_geom.buffer(-float(gap_erosion_tolerance))
        except Exception:
            gap_for_check = gap_geom

    pieces = _polygon_pieces_only(
        gap_for_check,
        min_area=max(float(min_piece_area), float(micro_gap_area_threshold)),
    )
    total_gap = float(sum(float(p.area) for p in pieces))
    max_gap = float(max((float(p.area) for p in pieces), default=0.0))
    return CoverageGapResult(
        total_gap_area=total_gap,
        max_gap_area=max_gap,
        gap_pieces=pieces,
        raw_total_gap_area=raw_total_gap,
        raw_max_gap_area=raw_max_gap,
        raw_gap_pieces=raw_pieces,
        gap_erosion_tolerance=float(gap_erosion_tolerance),
        ignored_micro_gap_total=float(sum(float(p.area) for p in ignored_micro)),
        ignored_micro_gap_count=len(ignored_micro),
    )


def _repair_polygon(geom: BaseGeometry) -> Optional[Polygon]:
    if geom is None or bool(getattr(geom, "is_empty", True)):
        return None
    fixed: BaseGeometry = geom
    try:
        if (not bool(getattr(fixed, "is_valid", True))) and _make_valid is not None:
            fixed = _make_valid(fixed)  # type: ignore[assignment]
    except Exception:
        pass
    try:
        fixed = fixed.buffer(0)
    except Exception:
        pass
    pieces = _polygon_pieces_only(fixed)
    if not pieces:
        return None
    poly = max(pieces, key=lambda p: float(p.area))
    try:
        simplified = poly.simplify(1e-3, preserve_topology=True)
        if isinstance(simplified, Polygon) and (not simplified.is_empty):
            poly = simplified
    except Exception:
        pass
    try:
        if not bool(getattr(poly, "is_valid", True)):
            poly = poly.buffer(0)
    except Exception:
        pass
    pieces2 = _polygon_pieces_only(poly)
    if not pieces2:
        return None
    return max(pieces2, key=lambda p: float(p.area))


def _polygon_ring(poly: Polygon) -> List[List[float]]:
    try:
        return [[round(float(x), 4), round(float(y), 4)] for x, y in poly.exterior.coords]
    except Exception:
        return []


def _record_synthetic_storage(
    *,
    room_id: str,
    floor_number: int,
    polygon: Polygon,
    island_id: str,
    source: str = "empty_island_sweep",
) -> Dict[str, Any]:
    minx, miny, maxx, maxy = (float(v) for v in polygon.bounds)
    return {
        "room_id": room_id,
        "room_name": "Auto Storage",
        "room_type": "storage",
        "target_area": float(polygon.area),
        "floor_id": f"F{int(floor_number)}",
        "source": str(source),
        "is_dummy": True,
        "island_id": str(island_id),
        "bbox": [round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4)],
        "polygon": _polygon_ring(polygon),
    }


def _storage_room_from_island(island: Any, *, floor_number: int, index: int) -> Tuple[RoomResult, Dict[str, Any]]:
    poly = _repair_polygon(getattr(island, "polygon", None))
    if poly is None or poly.is_empty:
        raise LayoutTopologyError(
            "Empty island storage conversion failed: invalid polygon",
            floor_number=floor_number,
            metadata={"island_id": str(getattr(island, "id", ""))},
        )
    digest = hashlib.md5(f"{getattr(island, 'id', index)}:{float(poly.area):.4f}".encode("utf-8")).hexdigest()[:8]
    rid = f"room_storage_{digest}"
    record = _record_synthetic_storage(
        room_id=rid,
        floor_number=floor_number,
        polygon=poly,
        island_id=str(getattr(island, "id", "")),
        source="empty_island_sweep",
    )
    room = RoomResult(
        id=rid,
        room_type="storage",
        polygon=poly,
        area=float(poly.area),
        target_area=float(poly.area),
        area_error=0.0,
        centroid=(float(poly.centroid.x), float(poly.centroid.y)),
        has_window=False,
        facade_length=0.0,
        aspect_ratio=1.0,
        is_dummy=True,
        target_area_raw=float(poly.area),
    )
    return room, record


def _storage_room_from_polygon(
    poly: Polygon,
    *,
    floor_number: int,
    island_id: str,
    source: str,
    index: int,
) -> Tuple[RoomResult, Dict[str, Any]]:
    fixed = _repair_polygon(poly)
    if fixed is None or fixed.is_empty:
        raise LayoutTopologyError(
            "Synthetic storage conversion failed: invalid residual polygon",
            floor_number=floor_number,
            metadata={"island_id": str(island_id), "source": str(source)},
        )
    digest = hashlib.md5(
        f"{source}:{island_id}:{index}:{float(fixed.area):.4f}:{fixed.bounds}".encode("utf-8")
    ).hexdigest()[:8]
    rid = f"room_storage_{digest}"
    record = _record_synthetic_storage(
        room_id=rid,
        floor_number=floor_number,
        polygon=fixed,
        island_id=str(island_id),
        source=str(source),
    )
    room = RoomResult(
        id=rid,
        room_type="storage",
        polygon=fixed,
        area=float(fixed.area),
        target_area=float(fixed.area),
        area_error=0.0,
        centroid=(float(fixed.centroid.x), float(fixed.centroid.y)),
        has_window=False,
        facade_length=0.0,
        aspect_ratio=1.0,
        is_dummy=True,
        target_area_raw=float(fixed.area),
    )
    return room, record


def _safe_annex_island_to_corridor(
    *,
    island_poly: Polygon,
    corridor: Any,
    floor_boundary: Polygon,
) -> Optional[Polygon]:
    corridor_poly = getattr(corridor, "polygon", None)
    if not isinstance(island_poly, Polygon) or island_poly.is_empty:
        return None
    if corridor_poly is None or bool(getattr(corridor_poly, "is_empty", True)):
        return None
    try:
        raw = corridor_poly.buffer(1e-6).union(island_poly.buffer(1e-6)).buffer(-1e-6)
        raw = raw.intersection(floor_boundary)
    except Exception:
        return None
    fixed = _repair_polygon(raw)
    if fixed is None or fixed.is_empty:
        return None
    if not isinstance(fixed, Polygon):
        return None
    if not bool(getattr(fixed, "is_valid", True)):
        return None
    try:
        if not floor_boundary.buffer(1e-6).covers(fixed):
            return None
    except Exception:
        return None
    try:
        old_area = float(corridor_poly.area)
        expected = old_area + float(island_poly.area)
        if abs(float(fixed.area) - expected) > max(0.25, float(island_poly.area) * 0.2):
            return None
    except Exception:
        pass
    snapped = _safe_snap_polygon_like(fixed, tol=0.01)
    return snapped if isinstance(snapped, Polygon) and not snapped.is_empty else fixed


def _best_corridor_for_island(island_poly: Polygon, corridors: List[Any]) -> Optional[Any]:
    best = None
    best_score: Tuple[float, float] = (-1.0, -1e9)
    for corridor in corridors or []:
        cpoly = getattr(corridor, "polygon", None)
        if cpoly is None or bool(getattr(cpoly, "is_empty", True)):
            continue
        try:
            shared = float(island_poly.boundary.intersection(cpoly.boundary).length)
            if shared <= 1e-6:
                shared = float(island_poly.boundary.intersection(cpoly).length)
            dist = float(island_poly.distance(cpoly))
        except Exception:
            continue
        score = (shared, -dist)
        if score > best_score:
            best_score = score
            best = corridor
    return best


def _polygon_shape_metadata(poly: Polygon) -> Dict[str, Any]:
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    width = max(0.0, maxx - minx)
    height = max(0.0, maxy - miny)
    bbox_area = max(0.0, width * height)
    fill_rate = float(poly.area) / bbox_area if bbox_area > 1e-9 else 0.0
    short = max(1e-6, min(width, height))
    long = max(width, height)
    aspect = float(long / short) if short > 0 else float("inf")
    if fill_rate < 0.2:
        hint = "极度破碎/边缘细缝型"
        advice = "DO NOT assign a distinct room here. Let backend geometry absorb this sliver into corridors when possible."
    elif aspect >= 4.0:
        hint = "狭长型"
        advice = "Prefer corridor expansion, narrow utility, or elongated storage. Do not assign square rooms."
    elif aspect >= 1.5:
        hint = "矩形偏长"
        advice = "Prefer storage/utility or a split/elongated support space."
    else:
        hint = "方正型"
        advice = "Suitable for small storage, utility, or compact bathroom if it fits the budget."
    return {
        "area": float(poly.area),
        "bbox": [round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4)],
        "width": round(width, 4),
        "height": round(height, 4),
        "bbox_area": round(bbox_area, 4),
        "fill_rate": round(fill_rate, 4),
        "aspect_ratio": round(aspect, 3) if aspect != float("inf") else aspect,
        "shape_hint": hint,
        "repair_advice": advice,
    }


def _shared_boundary_len(a: Polygon, b: Polygon, *, tol: float = 1e-4) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        shared = float(a.boundary.intersection(b.boundary).length)
        if shared > 1e-6:
            return shared
    except Exception:
        pass
    try:
        if float(a.boundary.distance(b.boundary)) > float(tol):
            return 0.0
        shared = a.boundary.intersection(b.boundary.buffer(float(tol), cap_style=2, join_style=2))
        return float(getattr(shared, "length", 0.0))
    except Exception:
        return 0.0


def _safe_merge_piece_into_corridor(
    *,
    piece: Polygon,
    corridor: Any,
    floor_boundary: Polygon,
) -> Optional[Polygon]:
    cpoly = getattr(corridor, "polygon", None)
    if cpoly is None or bool(getattr(cpoly, "is_empty", True)):
        return None
    try:
        raw = cpoly.buffer(1e-6).union(piece.buffer(1e-6)).buffer(-1e-6)
        raw = raw.intersection(floor_boundary)
    except Exception:
        return None
    fixed = _repair_polygon(raw)
    if fixed is None or fixed.is_empty or not isinstance(fixed, Polygon):
        return None
    try:
        if not floor_boundary.buffer(1e-6).covers(fixed):
            return None
    except Exception:
        return None
    try:
        expected = float(cpoly.area) + float(piece.area)
        if abs(float(fixed.area) - expected) > max(0.25, float(piece.area) * 0.25):
            return None
    except Exception:
        pass
    return fixed


def _safe_merge_piece_into_room(
    *,
    piece: Polygon,
    room: RoomResult,
    floor_boundary: Polygon,
) -> Optional[Polygon]:
    rpoly = getattr(room, "polygon", None)
    if not isinstance(rpoly, Polygon) or rpoly.is_empty:
        return None
    try:
        raw = rpoly.buffer(1e-6).union(piece.buffer(1e-6)).buffer(-1e-6)
        raw = raw.intersection(floor_boundary)
    except Exception:
        return None
    fixed = _repair_polygon(raw)
    if fixed is None or fixed.is_empty or not isinstance(fixed, Polygon):
        return None
    try:
        if not floor_boundary.buffer(1e-6).covers(fixed):
            return None
    except Exception:
        return None
    try:
        expected = float(rpoly.area) + float(piece.area)
        if abs(float(fixed.area) - expected) > max(0.20, float(piece.area) * 0.25):
            return None
        minx, miny, maxx, maxy = (float(v) for v in fixed.bounds)
        bbox_area = max(1e-6, (maxx - minx) * (maxy - miny))
        if float(fixed.area) / bbox_area < 0.65:
            return None
    except Exception:
        return None
    return fixed


def _best_corridor_for_piece(piece: Polygon, corridors: List[Any]) -> Tuple[Optional[Any], float]:
    best = None
    best_len = 0.0
    for corridor in corridors or []:
        cpoly = getattr(corridor, "polygon", None)
        if not isinstance(cpoly, Polygon) or cpoly.is_empty:
            continue
        shared = _shared_boundary_len(piece, cpoly)
        if shared > best_len:
            best_len = shared
            best = corridor
    return best, best_len


def _best_room_for_piece(piece: Polygon, rooms: List[RoomResult]) -> Tuple[Optional[RoomResult], float]:
    best = None
    best_len = 0.0
    for room in rooms or []:
        rtype = str(getattr(room, "room_type", "") or "").lower()
        if rtype == "void" or bool(getattr(room, "skip_solver", False)):
            continue
        poly = getattr(room, "polygon", None)
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        shared = _shared_boundary_len(piece, poly)
        if shared > best_len:
            best_len = shared
            best = room
    return best, best_len


def _door_preflight_for_residual(
    *,
    piece: Polygon,
    corridors: List[Any],
    rooms: List[RoomResult],
    floor_boundary: Optional[Polygon] = None,
    min_door_width: float = 0.8,
) -> DoorPreflightResult:
    corridor, shared_corridor = _best_corridor_for_piece(piece, corridors)
    shared_rooms: Dict[str, float] = {}
    best_room_shared = 0.0
    for room in rooms or []:
        rtype = str(getattr(room, "room_type", "") or "").lower()
        if rtype == "void" or bool(getattr(room, "skip_solver", False)):
            continue
        poly = getattr(room, "polygon", None)
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        shared = _shared_boundary_len(piece, poly)
        if shared > 1e-6:
            rid = str(getattr(room, "id", "") or getattr(room, "room_id", ""))
            shared_rooms[rid] = float(shared)
            best_room_shared = max(best_room_shared, float(shared))
    touches_boundary = False
    if floor_boundary is not None:
        try:
            touches_boundary = bool(piece.boundary.intersection(floor_boundary.boundary).length > 1e-6)
        except Exception:
            touches_boundary = False
    can_corridor_door = bool(corridor is not None and float(shared_corridor) >= float(min_door_width))
    can_attach = bool(best_room_shared >= 0.5)
    reason = "corridor_door" if can_corridor_door else (
        "room_attachment" if can_attach else (
            "boundary_feature" if touches_boundary else "no_access_anchor"
        )
    )
    return DoorPreflightResult(
        can_place_corridor_door=can_corridor_door,
        can_attach_to_room=can_attach,
        can_be_non_room_feature=bool(touches_boundary or can_attach),
        shared_len_with_corridor=float(shared_corridor),
        shared_len_with_rooms=shared_rooms,
        touches_floor_boundary=touches_boundary,
        reason=reason,
    )


def _log_residual_action(action: str, meta: Dict[str, Any], **details: Any) -> None:
    logger.debug(
        "[RESIDUAL] Action=%s | Area=%.4fm2 | Bbox=%s | FillRate=%.2f%% | Shape=%s | Details=%s",
        action,
        float(meta.get("area", 0.0) or 0.0),
        meta.get("bbox"),
        float(meta.get("fill_rate", 0.0) or 0.0) * 100.0,
        meta.get("shape_hint"),
        details,
    )


def _serializable_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in feature.items() if k != "polygon"}
    poly = feature.get("polygon")
    if isinstance(poly, Polygon) and not poly.is_empty:
        out.setdefault("area", round(float(poly.area), 4))
        out.setdefault("bbox", [round(float(v), 4) for v in poly.bounds])
    return out


def _coverage_feature_qa(
    *,
    feature: Dict[str, Any],
    floor_boundary: Polygon,
    semantic_rooms: Optional[List[RoomResult]] = None,
    core_tube: Optional[Any] = None,
    core_contract: Optional[Any] = None,
    area_drift_tolerance: float = 0.10,
) -> Tuple[bool, str]:
    poly = feature.get("polygon")
    if not isinstance(poly, Polygon) or poly.is_empty:
        return False, "missing_or_empty_polygon"
    if not bool(poly.is_valid):
        return False, "invalid_polygon"
    try:
        if not floor_boundary.buffer(1e-6).covers(poly):
            return False, "outside_floor_boundary"
    except Exception:
        return False, "floor_boundary_check_failed"
    original_area = float(feature.get("original_area", poly.area) or poly.area)
    if abs(float(poly.area) - original_area) > float(area_drift_tolerance):
        return False, "area_drift_exceeded"
    core_poly = getattr(core_contract, "core_union", None) if core_contract is not None else None
    if core_poly is None:
        core_poly = getattr(core_tube, "polygon", None) if core_tube is not None else None
    if isinstance(core_poly, Polygon) and not core_poly.is_empty:
        try:
            if float(poly.intersection(core_poly).area) > 1e-5:
                return False, "core_overlap"
        except Exception:
            return False, "core_overlap_check_failed"
    for room in semantic_rooms or []:
        rpoly = getattr(room, "polygon", None)
        if not isinstance(rpoly, Polygon) or rpoly.is_empty:
            continue
        try:
            if float(poly.intersection(rpoly).area) > 1e-5:
                return False, "semantic_room_overlap"
        except Exception:
            return False, "semantic_room_overlap_check_failed"
    return True, "pass"


def _accepted_coverage_feature(
    *,
    piece: Polygon,
    floor_number: int,
    island_id: str,
    decision: ResidualDecision,
    plan: Optional[Any] = None,
    source: str,
    core_contract: Optional[Any] = None,
) -> Dict[str, Any]:
    meta = dict(decision.metadata or {})
    residual_hash = hashlib.md5(str(piece.wkt).encode("utf-8")).hexdigest()[:8]
    source_residual_id = str(meta.get("residual_id") or f"residual_F{int(floor_number)}_{island_id}_{residual_hash}")
    role = str(decision.classification)
    if role in {"boundary_trim", "service_niche", "corridor_sponge", "edge_sliver_absorb"}:
        coverage_role = role
    elif role in {"attached_service_niche", "neighbor_absorb"}:
        coverage_role = role
    else:
        coverage_role = str(decision.materialize_as or "coverage_feature")
    feature = {
        "feature_id": f"coverage_feature_F{int(floor_number)}_{island_id}_{residual_hash}",
        "floor_id": f"F{int(floor_number)}",
        "island_id": str(island_id),
        "source_residual_id": source_residual_id,
        "source": str(source),
        "core_contract_id": str(getattr(core_contract, "core_contract_id", "") or ""),
        "core_union_hash": str(getattr(core_contract, "core_union_hash", "") or ""),
        "classification": str(decision.classification),
        "coverage_role": coverage_role,
        "semantic_room": False,
        "generated": True,
        "generated_by": "coverage_debt_planner",
        "counts_as_budget": False,
        "participates_in_budget_validation": False,
        "requires_door": False,
        "door_required": bool(decision.door_required),
        "participates_in_door_graph": False,
        "participates_in_wall_graph": True,
        "render_layer": "coverage_features",
        "semantic_repair_allowed": bool(decision.semantic_repair_allowed),
        "coverage_debt_plan_id": str(getattr(plan, "plan_id", "")) if plan is not None else "",
        "residual_decision": decision.to_dict(),
        "original_area": float(piece.area),
        "final_area": float(piece.area),
        "area_drift": 0.0,
        "qa_status": "pending",
        "final_status": "covered_by_coverage_feature",
        "polygon": piece,
    }
    feature.update({k: v for k, v in meta.items() if k not in feature})
    return feature


def _current_corridor_area(corridors: List[Any]) -> float:
    polys = [
        getattr(c, "polygon", None)
        for c in corridors or []
        if isinstance(getattr(c, "polygon", None), Polygon) and not getattr(c, "polygon").is_empty
    ]
    if not polys:
        return 0.0
    try:
        return float(unary_union(polys).area)
    except Exception:
        return float(sum(float(p.area) for p in polys))


def _corridor_sponge_headroom(
    *,
    corridors: List[Any],
    corridor_allowance_area: Optional[float],
    committed_sponge_area: float,
) -> Tuple[float, Dict[str, Any]]:
    current = _current_corridor_area(corridors)
    if corridor_allowance_area is None or float(corridor_allowance_area) <= 0.0:
        return 0.0, {
            "corridor_allowance_known": False,
            "current_corridor_area": current,
            "corridor_headroom": 0.0,
        }
    limit = float(corridor_allowance_area) * 1.15
    headroom = max(0.0, limit - current - float(committed_sponge_area))
    return headroom, {
        "corridor_allowance_known": True,
        "corridor_allowance_area": float(corridor_allowance_area),
        "corridor_global_limit": limit,
        "current_corridor_area": current,
        "committed_sponge_area": float(committed_sponge_area),
        "corridor_headroom": headroom,
    }


def _raise_sponge_coverage_error(
    *,
    message: str,
    floor_number: int,
    area: float,
    meta: Dict[str, Any],
    decision: ResidualDecision,
    stage: str,
    plan: Optional[Any] = None,
) -> None:
    metadata = dict(meta)
    metadata.update(
        {
            "failure_kind": "coverage",
            "stage": stage,
            "topology_mode": "grid_growth",
            "semantic_repair_allowed": False,
            "coverage_debt_plan_id": str(getattr(plan, "plan_id", "")) if plan is not None else "",
            "residual_decision": decision.to_dict(),
            "max_gap_fill_rate": float(meta.get("fill_rate", 0.0) or 0.0),
            "gap_pieces": [meta],
            "total_gap_area": float(area),
        }
    )
    raise LayoutCoverageError(
        message,
        floor_number=int(floor_number),
        max_gap_area=float(area),
        metadata=metadata,
        stage=stage,
        semantic_repair_allowed=False,
    )


def _apply_intra_island_residual_sweep(
    *,
    islands: List[Any],
    assignments: Dict[str, Any],
    rooms: List[RoomResult],
    corridors: List[Any],
    floor_boundary: Polygon,
    floor_number: int,
    synthetic_records: List[Dict[str, Any]],
    topology_mode: str = "continuous_cpsat",
    coverage_debt_plans: Optional[Dict[str, Any]] = None,
    coverage_features: Optional[List[Dict[str, Any]]] = None,
    corridor_allowance_area: Optional[float] = None,
    core_contract: Optional[Any] = None,
) -> Tuple[List[RoomResult], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rooms_by_id = {str(r.id): r for r in rooms}
    out_rooms = list(rooms)
    actions: List[Dict[str, Any]] = []
    coverage_features = coverage_features if coverage_features is not None else []
    storage_index = len(synthetic_records)
    topology_mode_l = str(topology_mode or "").lower()
    logger.info("[SWEEP] Start Intra-Island Residual Sweep | islands=%d | rooms=%d", len(islands or []), len(rooms or []))

    for island in islands or []:
        island_id = str(getattr(island, "id", ""))
        island_poly = _repair_polygon(getattr(island, "polygon", None))
        if island_poly is None or island_poly.is_empty:
            continue
        assignment = assignments.get(island_id)
        assigned_ids = [
            str(getattr(spec, "room_id", ""))
            for spec in (list(getattr(assignment, "rooms", []) or []) if assignment is not None else [])
        ]
        island_rooms = [rooms_by_id[rid] for rid in assigned_ids if rid in rooms_by_id]
        room_polys = [
            getattr(r, "polygon", None)
            for r in island_rooms
            if isinstance(getattr(r, "polygon", None), Polygon) and not getattr(r, "polygon").is_empty
        ]
        if not room_polys:
            continue
        try:
            residual = island_poly.difference(unary_union(room_polys))
        except Exception:
            fixed_rooms = [_repair_polygon(p) for p in room_polys]
            fixed_rooms = [p for p in fixed_rooms if isinstance(p, Polygon) and not p.is_empty]
            if not fixed_rooms:
                continue
            residual = island_poly.difference(unary_union(fixed_rooms))

        for piece in _polygon_pieces_only(residual, min_area=0.02):
            piece = _repair_polygon(piece)
            if piece is None or piece.is_empty:
                continue
            area = float(piece.area)
            meta = _polygon_shape_metadata(piece)
            meta["island_id"] = island_id
            if topology_mode_l == "grid_growth":
                corridor, shared = _best_corridor_for_piece(piece, corridors)
                door_preflight = _door_preflight_for_residual(
                    piece=piece,
                    corridors=corridors,
                    rooms=island_rooms,
                    floor_boundary=floor_boundary,
                )
                decision = classify_residual_piece(
                    piece,
                    floor_id=f"F{int(floor_number)}",
                    island_id=island_id,
                    floor_boundary=floor_boundary,
                    shared_len_with_corridor=float(door_preflight.shared_len_with_corridor),
                    shared_len_with_rooms=door_preflight.shared_len_with_rooms,
                    can_place_door=bool(door_preflight.can_place_corridor_door),
                    can_place_corridor_door=bool(door_preflight.can_place_corridor_door),
                    can_attach_to_room=bool(door_preflight.can_attach_to_room),
                    can_be_non_room_feature=bool(door_preflight.can_be_non_room_feature),
                    door_preflight_reason=str(door_preflight.reason),
                )
                plan = (coverage_debt_plans or {}).get(str(island_id))
                meta.update(
                    {
                        "classification": decision.classification,
                        "semantic_repair_allowed": bool(decision.semantic_repair_allowed),
                        "door_required": bool(decision.door_required),
                        "door_preflight": door_preflight.to_dict(),
                        "coverage_debt_plan_id": str(getattr(plan, "plan_id", "")) if plan is not None else "",
                    }
                )
                logger.info(
                    "[SPONGE] Residual classified | source=intra_island | floor=F%d | island=%s | "
                    "area=%.2fm2 | fill_rate=%.3f | classification=%s | semantic_repair_allowed=%s",
                    int(floor_number),
                    island_id,
                    area,
                    float(meta.get("fill_rate", 0.0) or 0.0),
                    decision.classification,
                    bool(decision.semantic_repair_allowed),
                )
                if decision.is_low_fill_geometry_debt:
                    committed = float(
                        sum(float(getattr(f.get("polygon"), "area", 0.0) or 0.0) for f in coverage_features)
                    )
                    headroom, headroom_meta = _corridor_sponge_headroom(
                        corridors=corridors,
                        corridor_allowance_area=corridor_allowance_area,
                        committed_sponge_area=committed,
                    )
                    planned_cap = float(getattr(plan, "planned_corridor_sponge_area", 0.0) or 0.0) if plan is not None else 0.0
                    allowed = min(
                        area,
                        float(headroom),
                        planned_cap if planned_cap > 0.0 else float(headroom),
                    )
                    if corridor is not None and shared > 0.5 and area <= allowed + 1e-6:
                        merged = _safe_merge_piece_into_corridor(
                            piece=piece,
                            corridor=corridor,
                            floor_boundary=floor_boundary,
                        )
                        if merged is not None:
                            corridor.polygon = merged
                            act = dict(meta)
                            act.update({
                                "action": "corridor_sponge",
                                "corridor_id": str(getattr(corridor, "id", "")),
                                "shared_len": round(float(shared), 4),
                                **headroom_meta,
                            })
                            actions.append(act)
                            _log_residual_action(
                                "corridor_sponge",
                                act,
                                reason="low-fill geometry debt merged within corridor headroom",
                                target=getattr(corridor, "id", ""),
                                delta_area=round(float(piece.area), 4),
                            )
                            logger.info(
                                "[QA] Coverage materialization checked | result=pass | action=corridor_sponge | area=%.2fm2",
                                area,
                            )
                            continue

                    feature = _accepted_coverage_feature(
                        piece=piece,
                        floor_number=int(floor_number),
                        island_id=island_id,
                        decision=decision,
                        plan=plan,
                        source="intra_island_residual_sweep",
                        core_contract=core_contract,
                    )
                    feature.update(headroom_meta)
                    qa_ok, qa_reason = _coverage_feature_qa(
                        feature=feature,
                        floor_boundary=floor_boundary,
                        semantic_rooms=out_rooms,
                        core_tube=None,
                        core_contract=core_contract,
                    )
                    feature["qa_status"] = "pass" if qa_ok else "failed"
                    feature["qa_reason"] = qa_reason
                    if not qa_ok:
                        _raise_sponge_coverage_error(
                            message=(
                                "Coverage feature QA failed: "
                                f"island={island_id}, area={area:.2f}m2, reason={qa_reason}"
                            ),
                            floor_number=int(floor_number),
                            area=area,
                            meta=meta,
                            decision=decision,
                            stage="coverage_feature_qa_failed",
                            plan=plan,
                        )
                    coverage_features.append(feature)
                    act = dict(meta)
                    act.update({
                        "action": str(decision.classification),
                        "coverage_feature_id": str(feature["feature_id"]),
                        "source_residual_id": str(feature.get("source_residual_id", "")),
                        "accepted_as_coverage_feature": True,
                        "materialized_features": [str(feature["feature_id"])],
                        "remaining_uncovered_area": 0.0,
                        "qa_status": str(feature.get("qa_status", "")),
                        "final_status": "covered_by_coverage_feature",
                        **headroom_meta,
                    })
                    actions.append(act)
                    logger.info(
                        "[SPONGE] Residual materialized | action=coverage_feature | feature=%s | area=%.2fm2 | class=%s",
                        feature["feature_id"],
                        area,
                        decision.classification,
                    )
                    logger.info(
                        "[QA] Coverage materialization checked | result=pass | action=coverage_feature | area=%.2fm2",
                        area,
                    )
                    logger.info(
                        "[LEDGER] Coverage debt updated | floor=F%d | island=%s | residual=%s | action=%s | final_status=covered_by_coverage_feature",
                        int(floor_number),
                        island_id,
                        feature.get("source_residual_id", ""),
                        decision.classification,
                    )
                    continue

                if decision.classification in {"compact_filler", "split_compact_filler"} and area > 2.0:
                    _raise_sponge_coverage_error(
                        message=(
                            "Compact residual filler is not enabled in Stage 2A.1: "
                            f"island={island_id}, area={area:.2f}m2"
                        ),
                        floor_number=int(floor_number),
                        area=area,
                        meta=meta,
                        decision=decision,
                        stage="compact_filler_identity_not_ready",
                        plan=plan,
                    )

                if decision.classification in {
                    "compact_filler_no_door",
                    "split_compact_filler_not_ready",
                } and area > 2.0:
                    _raise_sponge_coverage_error(
                        message=(
                            "Door-first residual classification failed: "
                            f"island={island_id}, area={area:.2f}m2, class={decision.classification}"
                        ),
                        floor_number=int(floor_number),
                        area=area,
                        meta=meta,
                        decision=decision,
                        stage="compact_filler_no_door",
                        plan=plan,
                    )

                if area > 8.0:
                    _raise_sponge_coverage_error(
                        message=f"Unexpected large residual remains: island={island_id}, area={area:.2f}m2",
                        floor_number=int(floor_number),
                        area=area,
                        meta=meta,
                        decision=decision,
                        stage="unexpected_residual_exceeded",
                        plan=plan,
                    )

            if area > 8.0:
                meta["neighbor_rooms"] = assigned_ids
                meta["residual_actions"] = actions
                logger.error(
                    "[COVERAGE] Large intra-island residual | Island=%s | Area=%.2fm2 | Metadata=%s",
                    island_id,
                    area,
                    meta,
                )
                raise LayoutCoverageError(
                    f"Large intra-island residual remains: island={island_id}, area={area:.2f}m2",
                    floor_number=int(floor_number),
                    max_gap_area=area,
                    metadata={
                        "total_gap_area": area,
                        "gap_pieces": [meta],
                        "residual_actions": actions,
                        "synthetic_rooms": synthetic_records,
                    },
                )

            if area < 2.0:
                corridor, shared = _best_corridor_for_piece(piece, corridors)
                if corridor is not None and shared > 0.5:
                    merged = _safe_merge_piece_into_corridor(
                        piece=piece,
                        corridor=corridor,
                        floor_boundary=floor_boundary,
                    )
                    if merged is not None:
                        corridor.polygon = merged
                        act = dict(meta)
                        act.update({
                            "action": "merged_to_corridor",
                            "corridor_id": str(getattr(corridor, "id", "")),
                            "shared_len": round(float(shared), 4),
                        })
                        actions.append(act)
                        _log_residual_action(
                            "merged_to_corridor",
                            act,
                            reason="area<2 and corridor shared boundary >0.5m",
                            target=getattr(corridor, "id", ""),
                            delta_area=round(float(piece.area), 4),
                        )
                        logger.info(
                            "[SWEEP] Intra-island residual merged_to_corridor | island=%s | area=%.2fm2 | corridor=%s",
                            island_id,
                            area,
                            getattr(corridor, "id", ""),
                        )
                        continue

                room, room_shared = _best_room_for_piece(piece, island_rooms)
                if room is not None and room_shared > 0.05:
                    merged_room = _safe_merge_piece_into_room(
                        piece=piece,
                        room=room,
                        floor_boundary=floor_boundary,
                    )
                    if merged_room is not None:
                        _update_room_geometry(room, merged_room, floor_boundary)
                        act = dict(meta)
                        act.update({
                            "action": "merged_to_room",
                            "room_id": str(getattr(room, "id", "")),
                            "shared_len": round(float(room_shared), 4),
                        })
                        actions.append(act)
                        _log_residual_action(
                            "merged_to_room",
                            act,
                            reason="area<2 and room shared boundary is best available",
                            target=getattr(room, "id", ""),
                            delta_area=round(float(piece.area), 4),
                        )
                        logger.info(
                            "[SWEEP] Intra-island residual merged_to_room | island=%s | area=%.2fm2 | room=%s",
                            island_id,
                            area,
                            getattr(room, "id", ""),
                        )
                        continue

            room, record = _storage_room_from_polygon(
                piece,
                floor_number=int(floor_number),
                island_id=island_id,
                source="intra_island_residual_sweep",
                index=storage_index,
            )
            storage_index += 1
            out_rooms.append(room)
            rooms_by_id[str(room.id)] = room
            synthetic_records.append(record)
            act = dict(meta)
            act.update({"action": "synthetic_storage", "room_id": room.id})
            actions.append(act)
            _log_residual_action(
                "synthetic_storage",
                act,
                reason="residual not safely mergeable; converted to storage",
                target=room.id,
                delta_area=round(float(piece.area), 4),
            )
            logger.info(
                "[SWEEP] Intra-island residual synthetic_storage | island=%s | area=%.2fm2 | room=%s",
                island_id,
                area,
                room.id,
            )

    logger.info("[SWEEP] End Intra-Island Residual Sweep | actions=%d | synthetic_total=%d", len(actions), len(synthetic_records))
    return out_rooms, synthetic_records, actions


def _floor_level_occupied_polygons(
    *,
    rooms: List[RoomResult],
    corridors: List[Any],
    core_tube: Optional[Any],
    coverage_features: Optional[List[Dict[str, Any]]] = None,
) -> List[Polygon]:
    polys: List[Polygon] = []
    for room in rooms or []:
        if str(getattr(room, "room_type", "") or "").lower() == "void":
            continue
        if bool(getattr(room, "skip_solver", False)):
            continue
        poly = getattr(room, "polygon", None)
        if isinstance(poly, Polygon) and (not poly.is_empty):
            polys.append(poly)
    for corridor in corridors or []:
        poly = getattr(corridor, "polygon", None)
        if isinstance(poly, Polygon) and (not poly.is_empty):
            polys.append(poly)
    if core_tube is not None:
        poly = getattr(core_tube, "polygon", None)
        if isinstance(poly, Polygon) and (not poly.is_empty):
            polys.append(poly)
    for feature in coverage_features or []:
        poly = feature.get("polygon") if isinstance(feature, dict) else getattr(feature, "polygon", None)
        if isinstance(poly, Polygon) and (not poly.is_empty):
            polys.append(poly)
    return polys


def _apply_floor_level_residual_sweep(
    *,
    rooms: List[RoomResult],
    corridors: List[Any],
    core_tube: Optional[Any],
    floor_boundary: Polygon,
    floor_number: int,
    synthetic_records: List[Dict[str, Any]],
    micro_gap_area_threshold: float = 0.1,
    topology_mode: str = "continuous_cpsat",
    coverage_features: Optional[List[Dict[str, Any]]] = None,
    corridor_allowance_area: Optional[float] = None,
    core_contract: Optional[Any] = None,
) -> Tuple[List[RoomResult], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    out_rooms = list(rooms)
    actions: List[Dict[str, Any]] = []
    coverage_features = coverage_features if coverage_features is not None else []
    topology_mode_l = str(topology_mode or "").lower()
    stats = {
        "ignored_micro_gap_total": 0.0,
        "ignored_micro_gap_count": 0,
    }
    logger.info("[SWEEP] Start Floor-Level Residual Sweep | rooms=%d | corridors=%d", len(rooms or []), len(corridors or []))
    occupied_polys = _floor_level_occupied_polygons(
        rooms=out_rooms,
        corridors=corridors,
        core_tube=core_tube,
        coverage_features=coverage_features,
    )
    if not occupied_polys:
        logger.info("[SWEEP] End Floor-Level Residual Sweep | reason=no_occupied_polygons")
        return out_rooms, synthetic_records, actions, stats
    try:
        gap_geom = floor_boundary.difference(unary_union(occupied_polys).intersection(floor_boundary))
    except Exception:
        repaired = [_repair_polygon(p) for p in occupied_polys]
        repaired = [p for p in repaired if isinstance(p, Polygon) and (not p.is_empty)]
        if not repaired:
            return out_rooms, synthetic_records, actions, stats
        gap_geom = floor_boundary.difference(unary_union(repaired).intersection(floor_boundary))

    pieces = _polygon_pieces_only(gap_geom, min_area=1e-6)
    storage_index = len(synthetic_records)
    for piece in pieces:
        piece = _repair_polygon(piece)
        if piece is None or piece.is_empty:
            continue
        area = float(piece.area)
        meta = _polygon_shape_metadata(piece)
        meta["source"] = "floor_level_residual_sweep"
        if area <= float(micro_gap_area_threshold):
            stats["ignored_micro_gap_total"] = float(stats["ignored_micro_gap_total"]) + area
            stats["ignored_micro_gap_count"] = int(stats["ignored_micro_gap_count"]) + 1
            act = dict(meta)
            act["action"] = "ignored_micro_sliver"
            actions.append(act)
            _log_residual_action(
                "ignored_micro_sliver",
                act,
                reason=f"area<={micro_gap_area_threshold:.2f}m2",
                delta_area=0.0,
            )
            continue
        if topology_mode_l == "grid_growth":
            corridor, shared = _best_corridor_for_piece(piece, corridors)
            door_preflight = _door_preflight_for_residual(
                piece=piece,
                corridors=corridors,
                rooms=out_rooms,
                floor_boundary=floor_boundary,
            )
            decision = classify_residual_piece(
                piece,
                floor_id=f"F{int(floor_number)}",
                island_id="__floor_level__",
                floor_boundary=floor_boundary,
                shared_len_with_corridor=float(door_preflight.shared_len_with_corridor),
                shared_len_with_rooms=door_preflight.shared_len_with_rooms,
                can_place_door=bool(door_preflight.can_place_corridor_door),
                can_place_corridor_door=bool(door_preflight.can_place_corridor_door),
                can_attach_to_room=bool(door_preflight.can_attach_to_room),
                can_be_non_room_feature=bool(door_preflight.can_be_non_room_feature),
                door_preflight_reason=str(door_preflight.reason),
            )
            meta.update(
                {
                    "classification": decision.classification,
                    "semantic_repair_allowed": bool(decision.semantic_repair_allowed),
                    "door_required": bool(decision.door_required),
                    "door_preflight": door_preflight.to_dict(),
                }
            )
            logger.info(
                "[SPONGE] Residual classified | source=floor_level | floor=F%d | "
                "area=%.2fm2 | fill_rate=%.3f | classification=%s | semantic_repair_allowed=%s",
                int(floor_number),
                area,
                float(meta.get("fill_rate", 0.0) or 0.0),
                decision.classification,
                bool(decision.semantic_repair_allowed),
            )
            if decision.is_low_fill_geometry_debt:
                committed = float(
                    sum(float(getattr(f.get("polygon"), "area", 0.0) or 0.0) for f in coverage_features)
                )
                headroom, headroom_meta = _corridor_sponge_headroom(
                    corridors=corridors,
                    corridor_allowance_area=corridor_allowance_area,
                    committed_sponge_area=committed,
                )
                if corridor is not None and shared > 0.5 and area <= headroom + 1e-6:
                    merged = _safe_merge_piece_into_corridor(
                        piece=piece,
                        corridor=corridor,
                        floor_boundary=floor_boundary,
                    )
                    if merged is not None:
                        corridor.polygon = merged
                        act = dict(meta)
                        act.update({
                            "action": "corridor_sponge",
                            "corridor_id": str(getattr(corridor, "id", "")),
                            "shared_len": round(float(shared), 4),
                            **headroom_meta,
                        })
                        actions.append(act)
                        _log_residual_action(
                            "corridor_sponge",
                            act,
                            reason="floor-level low-fill geometry debt merged within corridor headroom",
                            target=getattr(corridor, "id", ""),
                            delta_area=round(float(piece.area), 4),
                        )
                        logger.info(
                            "[QA] Coverage materialization checked | result=pass | action=corridor_sponge | area=%.2fm2",
                            area,
                        )
                        continue

                feature = _accepted_coverage_feature(
                    piece=piece,
                    floor_number=int(floor_number),
                    island_id="__floor_level__",
                    decision=decision,
                    plan=None,
                    source="floor_level_residual_sweep",
                    core_contract=core_contract,
                )
                feature.update(headroom_meta)
                qa_ok, qa_reason = _coverage_feature_qa(
                    feature=feature,
                    floor_boundary=floor_boundary,
                    semantic_rooms=out_rooms,
                    core_tube=core_tube,
                    core_contract=core_contract,
                )
                feature["qa_status"] = "pass" if qa_ok else "failed"
                feature["qa_reason"] = qa_reason
                if not qa_ok:
                    _raise_sponge_coverage_error(
                        message=(
                            "Floor-level coverage feature QA failed: "
                            f"area={area:.2f}m2, reason={qa_reason}"
                        ),
                        floor_number=int(floor_number),
                        area=area,
                        meta=meta,
                        decision=decision,
                        stage="coverage_feature_qa_failed",
                        plan=None,
                    )
                coverage_features.append(feature)
                act = dict(meta)
                act.update({
                    "action": str(decision.classification),
                    "coverage_feature_id": str(feature["feature_id"]),
                    "source_residual_id": str(feature.get("source_residual_id", "")),
                    "accepted_as_coverage_feature": True,
                    "materialized_features": [str(feature["feature_id"])],
                    "remaining_uncovered_area": 0.0,
                    "qa_status": str(feature.get("qa_status", "")),
                    "final_status": "covered_by_coverage_feature",
                    **headroom_meta,
                })
                actions.append(act)
                logger.info(
                    "[SPONGE] Residual materialized | action=coverage_feature | feature=%s | area=%.2fm2 | class=%s",
                    feature["feature_id"],
                    area,
                    decision.classification,
                )
                logger.info(
                    "[QA] Coverage materialization checked | result=pass | action=coverage_feature | area=%.2fm2",
                    area,
                )
                logger.info(
                    "[LEDGER] Coverage debt updated | floor=F%d | island=%s | residual=%s | action=%s | final_status=covered_by_coverage_feature",
                    int(floor_number),
                    "__floor_level__",
                    feature.get("source_residual_id", ""),
                    decision.classification,
                )
                continue

            if decision.classification in {"compact_filler", "split_compact_filler"} and area > 2.0:
                _raise_sponge_coverage_error(
                    message=(
                        "Compact floor-level residual filler is not enabled in Stage 2A.1: "
                        f"area={area:.2f}m2"
                    ),
                    floor_number=int(floor_number),
                    area=area,
                    meta=meta,
                    decision=decision,
                    stage="compact_filler_identity_not_ready",
                    plan=None,
                )
            if decision.classification in {
                "compact_filler_no_door",
                "split_compact_filler_not_ready",
            } and area > 2.0:
                _raise_sponge_coverage_error(
                    message=(
                        "Door-first floor-level residual classification failed: "
                        f"area={area:.2f}m2, class={decision.classification}"
                    ),
                    floor_number=int(floor_number),
                    area=area,
                    meta=meta,
                    decision=decision,
                    stage="compact_filler_no_door",
                    plan=None,
                )
            if area > 8.0:
                _raise_sponge_coverage_error(
                    message=f"Unexpected large floor-level residual remains: area={area:.2f}m2",
                    floor_number=int(floor_number),
                    area=area,
                    meta=meta,
                    decision=decision,
                    stage="unexpected_residual_exceeded",
                    plan=None,
                )
        if area > 8.0:
            meta["floor_residual_actions"] = actions
            logger.error(
                "[COVERAGE] Large floor-level residual | Area=%.2fm2 | Metadata=%s",
                area,
                meta,
            )
            raise LayoutCoverageError(
                f"Large floor-level residual remains: area={area:.2f}m2",
                floor_number=int(floor_number),
                max_gap_area=area,
                metadata={
                    "total_gap_area": area,
                    "gap_pieces": [meta],
                    "floor_residual_actions": actions,
                    "synthetic_rooms": synthetic_records,
                    **stats,
                },
            )

        if area < 2.0:
            corridor, shared = _best_corridor_for_piece(piece, corridors)
            if corridor is not None and shared > 0.5:
                merged = _safe_merge_piece_into_corridor(
                    piece=piece,
                    corridor=corridor,
                    floor_boundary=floor_boundary,
                )
                if merged is not None:
                    corridor.polygon = merged
                    act = dict(meta)
                    act.update({
                        "action": "merged_to_corridor",
                        "corridor_id": str(getattr(corridor, "id", "")),
                        "shared_len": round(float(shared), 4),
                    })
                    actions.append(act)
                    _log_residual_action(
                        "merged_to_corridor",
                        act,
                        reason="area<2 and corridor shared boundary >0.5m",
                        target=getattr(corridor, "id", ""),
                        delta_area=round(float(piece.area), 4),
                    )
                    logger.info(
                        "[SWEEP] Floor residual merged_to_corridor | area=%.2fm2 | corridor=%s",
                        area,
                        getattr(corridor, "id", ""),
                    )
                    continue

            room, room_shared = _best_room_for_piece(piece, out_rooms)
            if room is not None and room_shared > 0.05:
                merged_room = _safe_merge_piece_into_room(
                    piece=piece,
                    room=room,
                    floor_boundary=floor_boundary,
                )
                if merged_room is not None:
                    _update_room_geometry(room, merged_room, floor_boundary)
                    act = dict(meta)
                    act.update({
                        "action": "merged_to_room",
                        "room_id": str(getattr(room, "id", "")),
                        "shared_len": round(float(room_shared), 4),
                    })
                    actions.append(act)
                    _log_residual_action(
                        "merged_to_room",
                        act,
                        reason="no valid corridor merge; room shared boundary is best available",
                        target=getattr(room, "id", ""),
                        delta_area=round(float(piece.area), 4),
                    )
                    logger.info(
                        "[SWEEP] Floor residual merged_to_room | area=%.2fm2 | room=%s",
                        area,
                        getattr(room, "id", ""),
                    )
                    continue

        room, record = _storage_room_from_polygon(
            piece,
            floor_number=int(floor_number),
            island_id="__floor_level__",
            source="floor_level_residual_sweep",
            index=storage_index,
        )
        storage_index += 1
        out_rooms.append(room)
        synthetic_records.append(record)
        act = dict(meta)
        act.update({"action": "synthetic_storage", "room_id": room.id})
        actions.append(act)
        _log_residual_action(
            "synthetic_storage",
            act,
            reason="floor-level residual not safely mergeable; converted to storage",
            target=room.id,
            delta_area=round(float(piece.area), 4),
        )
        logger.info("[SWEEP] Floor residual synthetic_storage | area=%.2fm2 | room=%s", area, room.id)

    logger.info(
        "[SWEEP] End Floor-Level Residual Sweep | actions=%d | ignored_micro_count=%d | ignored_micro_area=%.4fm2",
        len(actions),
        int(stats.get("ignored_micro_gap_count", 0) or 0),
        float(stats.get("ignored_micro_gap_total", 0.0) or 0.0),
    )
    return out_rooms, synthetic_records, actions, stats


def _empty_island_sweep(
    *,
    islands: List[Any],
    assignments: Dict[str, Any],
    corridors: List[Any],
    floor_boundary: Polygon,
    floor_number: int,
) -> Tuple[List[Any], List[RoomResult], List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Any] = []
    synthetic_rooms: List[RoomResult] = []
    synthetic_records: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    for idx, island in enumerate(islands or []):
        island_id = str(getattr(island, "id", idx))
        assignment = assignments.get(island_id)
        assigned_rooms = list(getattr(assignment, "rooms", []) or []) if assignment is not None else []
        if assigned_rooms:
            kept.append(island)
            continue

        poly = _repair_polygon(getattr(island, "polygon", None))
        if poly is None or poly.is_empty:
            assignments.pop(island_id, None)
            actions.append({"island_id": island_id, "action": "dropped_invalid"})
            continue

        area = float(poly.area)
        minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
        base_action: Dict[str, Any] = {
            "island_id": island_id,
            "area": area,
            "bbox": [round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4)],
        }

        if area > 8.0:
            meta = dict(base_action)
            meta["empty_island_actions"] = actions
            raise LayoutTopologyError(
                f"Empty island too large for automatic storage: island={island_id}, area={area:.2f}m2",
                floor_number=floor_number,
                metadata=meta,
            )

        if area < 2.0:
            corridor = _best_corridor_for_island(poly, corridors)
            if corridor is not None:
                annexed = _safe_annex_island_to_corridor(
                    island_poly=poly,
                    corridor=corridor,
                    floor_boundary=floor_boundary,
                )
                if annexed is not None:
                    corridor.polygon = annexed
                    assignments.pop(island_id, None)
                    act = dict(base_action)
                    act.update({"action": "annexed_to_corridor", "corridor_id": str(getattr(corridor, "id", ""))})
                    actions.append(act)
                    logger.info(
                        "Empty island sweep: %s annexed_to_corridor %s area=%.2fm2",
                        island_id,
                        getattr(corridor, "id", ""),
                        area,
                    )
                    continue

        room, record = _storage_room_from_island(island, floor_number=floor_number, index=idx)
        synthetic_rooms.append(room)
        synthetic_records.append(record)
        assignments.pop(island_id, None)
        act = dict(base_action)
        act.update({"action": "converted_to_storage", "room_id": room.id})
        actions.append(act)
        logger.info("Empty island sweep: %s converted_to_storage %s area=%.2fm2", island_id, room.id, area)

    return kept, synthetic_rooms, synthetic_records, actions


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


ENTRANCE_SNAP_THRESHOLD = 1.5
ENTRANCE_SNAP_MAX_CONNECTION_WIDTH = 3.0
ENTRANCE_SNAP_OVERSHOOT = 0.2
ENTRANCE_SNAP_GRID = 0.01
ENTRANCE_SNAP_AREA_EPS = 1e-4
ROOM_EXTERIOR_GAP_MIN = 0.02
ROOM_EXTERIOR_GAP_MAX = 0.60
ROOM_EXTERIOR_GAP_MAX_AREA = 8.0
ROOM_EXTERIOR_GAP_MAX_AREA_RATIO = 0.25


def _axis_snap_bounds(poly: Polygon) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = poly.bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def _nearest_floor_side(poly: Polygon, floor_boundary: Polygon) -> Tuple[str, float]:
    fminx, fminy, fmaxx, fmaxy = _axis_snap_bounds(floor_boundary)
    minx, miny, maxx, maxy = _axis_snap_bounds(poly)
    candidates = [
        ("left", max(0.0, minx - fminx)),
        ("right", max(0.0, fmaxx - maxx)),
        ("bottom", max(0.0, miny - fminy)),
        ("top", max(0.0, fmaxy - maxy)),
    ]
    return min(candidates, key=lambda item: float(item[1]))


def _clamp_interval(center: float, width: float, low: float, high: float) -> Tuple[float, float]:
    low = float(low)
    high = float(high)
    width = min(max(0.0, float(width)), max(0.0, high - low))
    half = width / 2.0
    a0 = float(center) - half
    a1 = float(center) + half
    if a0 < low:
        a1 += low - a0
        a0 = low
    if a1 > high:
        a0 -= a1 - high
        a1 = high
    return (max(low, float(a0)), min(high, float(a1)))


def _connection_width(span: float, min_width: float) -> float:
    return min(
        ENTRANCE_SNAP_MAX_CONNECTION_WIDTH,
        max(float(min_width), min(float(span), ENTRANCE_SNAP_MAX_CONNECTION_WIDTH)),
    )


def _exterior_connection_box(
    poly: Polygon,
    floor_boundary: Polygon,
    side: str,
    min_width: float,
) -> Optional[Polygon]:
    fminx, fminy, fmaxx, fmaxy = _axis_snap_bounds(floor_boundary)
    minx, miny, maxx, maxy = _axis_snap_bounds(poly)
    overshoot = ENTRANCE_SNAP_OVERSHOOT

    if side in ("left", "right"):
        span = max(0.0, maxy - miny)
        width = _connection_width(span, min_width)
        y0, y1 = _clamp_interval((miny + maxy) / 2.0, width, fminy, fmaxy)
        if y1 - y0 <= ENTRANCE_SNAP_AREA_EPS:
            return None
        if side == "left":
            return box(fminx - overshoot, y0, minx, y1)
        return box(maxx, y0, fmaxx + overshoot, y1)

    span = max(0.0, maxx - minx)
    width = _connection_width(span, min_width)
    x0, x1 = _clamp_interval((minx + maxx) / 2.0, width, fminx, fmaxx)
    if x1 - x0 <= ENTRANCE_SNAP_AREA_EPS:
        return None
    if side == "bottom":
        return box(x0, fminy - overshoot, x1, miny)
    return box(x0, maxy, x1, fmaxy + overshoot)


def _connection_box_between(
    source: Polygon,
    target: Polygon,
    min_width: float,
) -> Optional[Polygon]:
    aminx, aminy, amaxx, amaxy = _axis_snap_bounds(source)
    bminx, bminy, bmaxx, bmaxy = _axis_snap_bounds(target)

    if amaxx <= bminx:
        overlap0 = max(aminy, bminy)
        overlap1 = min(amaxy, bmaxy)
        if overlap1 - overlap0 <= ENTRANCE_SNAP_AREA_EPS:
            return None
        width = _connection_width(overlap1 - overlap0, min_width)
        y0, y1 = _clamp_interval((overlap0 + overlap1) / 2.0, width, overlap0, overlap1)
        return box(amaxx, y0, bminx, y1)

    if bmaxx <= aminx:
        overlap0 = max(aminy, bminy)
        overlap1 = min(amaxy, bmaxy)
        if overlap1 - overlap0 <= ENTRANCE_SNAP_AREA_EPS:
            return None
        width = _connection_width(overlap1 - overlap0, min_width)
        y0, y1 = _clamp_interval((overlap0 + overlap1) / 2.0, width, overlap0, overlap1)
        return box(bmaxx, y0, aminx, y1)

    if amaxy <= bminy:
        overlap0 = max(aminx, bminx)
        overlap1 = min(amaxx, bmaxx)
        if overlap1 - overlap0 <= ENTRANCE_SNAP_AREA_EPS:
            return None
        width = _connection_width(overlap1 - overlap0, min_width)
        x0, x1 = _clamp_interval((overlap0 + overlap1) / 2.0, width, overlap0, overlap1)
        return box(x0, amaxy, x1, bminy)

    if bmaxy <= aminy:
        overlap0 = max(aminx, bminx)
        overlap1 = min(amaxx, bmaxx)
        if overlap1 - overlap0 <= ENTRANCE_SNAP_AREA_EPS:
            return None
        width = _connection_width(overlap1 - overlap0, min_width)
        x0, x1 = _clamp_interval((overlap0 + overlap1) / 2.0, width, overlap0, overlap1)
        return box(x0, bmaxy, x1, aminy)

    return None


def _layout_snap_blockers(
    rooms: List[RoomResult],
    core_tube: Any,
    exclude_ids: Optional[set] = None,
) -> List[Polygon]:
    exclude_ids = exclude_ids or set()
    blockers: List[Polygon] = []
    for room in rooms:
        rid = str(getattr(room, "id", "") or "")
        if rid in exclude_ids:
            continue
        rtype = str(getattr(room, "room_type", "") or "").lower()
        if rtype in ("void", "utility_dummy") or bool(getattr(room, "is_dummy", False)):
            continue
        poly = getattr(room, "polygon", None)
        if isinstance(poly, Polygon) and not poly.is_empty:
            blockers.append(poly)

    if core_tube is not None:
        attrs = (
            "polygon",
            "staircase",
            "staircase_hall",
            "staircase_hall_b",
            "staircase_shaft",
            "elevator",
            "elevator_hall",
            "elevator_hall_b",
            "elevator_shaft",
        )
        for attr in attrs:
            poly = getattr(core_tube, attr, None)
            if isinstance(poly, Polygon) and not poly.is_empty:
                blockers.append(poly)
    return blockers


def _patches_clear(patches: List[Polygon], blockers: List[Polygon], floor_boundary: Polygon) -> bool:
    for patch in patches:
        try:
            clipped = patch.intersection(floor_boundary)
        except Exception:
            return False
        if clipped.is_empty:
            return False
        for blocker in blockers:
            try:
                if float(clipped.intersection(blocker).area) > ENTRANCE_SNAP_AREA_EPS:
                    return False
            except Exception:
                return False
    return True


def _merge_snap_patches(poly: Polygon, patches: List[Polygon], floor_boundary: Polygon) -> Optional[Polygon]:
    if not patches:
        return None
    try:
        raw = poly.union(unary_union(patches))
        clipped = raw.intersection(floor_boundary)
    except Exception:
        return None
    if not isinstance(clipped, Polygon):
        return None
    if clipped.is_empty or not bool(getattr(clipped, "is_valid", True)):
        return None
    snapped = _safe_snap_polygon_like(clipped, tol=ENTRANCE_SNAP_GRID)
    if snapped is None or not isinstance(snapped, Polygon) or snapped.is_empty:
        return None
    if not bool(getattr(snapped, "is_valid", True)):
        return None
    if not _is_axis_aligned_polygon(snapped, tol=1e-6):
        return None
    try:
        if not floor_boundary.buffer(1e-6).covers(snapped):
            return None
    except Exception:
        return None
    return snapped


def _update_room_geometry(room: RoomResult, poly: Polygon, floor_boundary: Polygon) -> None:
    room.polygon = poly
    room.area = float(poly.area)
    room.centroid = (float(poly.centroid.x), float(poly.centroid.y))
    room.area_error = (
        abs(float(poly.area) - float(room.target_area)) / float(room.target_area)
        if float(room.target_area) > 0 else 0.0
    )
    try:
        facade = float(poly.boundary.intersection(floor_boundary.exterior).length)
    except Exception:
        facade = 0.0
    room.facade_length = facade
    room.has_window = bool(room.has_window or facade > 1.0)
    room.aspect_ratio = LayoutGenerator._compute_aspect_ratio(poly)


def _real_room_target_area(specs: List[SemanticRoomSpec]) -> float:
    return float(
        sum(
            max(0.0, float(getattr(s, "target_area", 0.0) or 0.0))
            for s in specs
            if not bool(getattr(s, "is_dummy", False))
        )
    )


def is_stage2a_area_contract(topology_mode: str) -> bool:
    """Stage 2A grid growth owns positive residual via geometry sweep, not semantic abort."""
    return str(topology_mode or "").strip().lower() == "grid_growth"


def apply_semantic_residual_gate(
    *,
    floor_id: str,
    topology_mode: str,
    floor_area: float,
    budget_island_area: Optional[float],
    grid_island_area: float,
    llm_room_sum: float,
    semantic_residual_abort_ratio: float,
    stage: str,
    negative_residual_abs_tolerance: float = 2.0,
    negative_residual_floor_ratio: float = 0.03,
) -> None:
    """Apply legacy residual checks while honoring Stage 2A grid-growth ownership."""
    total_island_area = float(grid_island_area)
    room_sum = float(llm_room_sum)
    residual = total_island_area - room_sum
    positive_residual = max(0.0, residual)
    legacy_limit = float(floor_area) * float(semantic_residual_abort_ratio)
    is_grid = is_stage2a_area_contract(topology_mode)

    if is_grid:
        budget_area = float(budget_island_area) if budget_island_area is not None else float("nan")
        delta_area = (
            total_island_area - budget_area
            if budget_island_area is not None
            else float("nan")
        )
        residual_ratio_to_floor = residual / max(1e-9, float(floor_area))
        residual_ratio_to_grid = residual / max(1e-9, total_island_area)
        logger.info(
            "[GRID] Handoff Area Stats | floor=%s | topology_mode=%s | stage=%s | "
            "budget_island_area=%.2f | grid_island_area=%.2f | llm_room_sum=%.2f | "
            "expected_residual=%.2f | delta_area=%.2f | residual_ratio_to_floor=%.3f | "
            "residual_ratio_to_grid=%.3f",
            floor_id,
            topology_mode,
            stage,
            budget_area,
            total_island_area,
            room_sum,
            residual,
            delta_area,
            residual_ratio_to_floor,
            residual_ratio_to_grid,
        )
        negative_limit = max(
            float(negative_residual_abs_tolerance),
            float(negative_residual_floor_ratio) * float(floor_area),
        )
        if residual < -negative_limit:
            raise LayoutCoverageError(
                "Grid growth island area is smaller than requested room area: "
                f"residual={residual:.2f}m2, islands={total_island_area:.2f}m2, "
                f"rooms={room_sum:.2f}m2, tolerance={negative_limit:.2f}m2",
                max_gap_area=abs(residual),
                metadata={
                    "failure_kind": "grid_growth",
                    "floor_id": floor_id,
                    "stage": stage,
                    "topology_mode": topology_mode,
                    "grid_island_area": total_island_area,
                    "llm_room_sum": room_sum,
                    "expected_residual": residual,
                    "negative_residual_tolerance": negative_limit,
                    "total_gap_area": abs(residual),
                },
            )
        if positive_residual > legacy_limit:
            logger.info(
                "[GRID] Skip legacy semantic residual gate | floor=%s | stage=%s | "
                "residual=%.2f | legacy_limit=%.2f | grid_island_area=%.2f | "
                "llm_room_sum=%.2f | reason=stage2a_residual_owned_by_sweep",
                floor_id,
                stage,
                positive_residual,
                legacy_limit,
                total_island_area,
                room_sum,
            )
        return

    if positive_residual > legacy_limit:
        raise SemanticInvalidError(
            "Total room area too small for floor capacity: "
            f"residual={positive_residual:.2f}m2, islands={total_island_area:.2f}m2, "
            f"rooms={room_sum:.2f}m2, ratio={semantic_residual_abort_ratio:.2f}"
        )


def _snapshot_budget_island_area(topology_snapshot: Optional[TopologySnapshot], floor_id: str) -> Optional[float]:
    try:
        if topology_snapshot is None:
            return None
        floor_snap = topology_snapshot.floors.get(str(floor_id))
        if floor_snap is None:
            return None
        total = 0.0
        for ring in getattr(floor_snap, "island_rings", []) or []:
            poly = Polygon([(float(p[0]), float(p[1])) for p in ring])
            if not poly.is_empty:
                total += float(poly.area)
        return total
    except Exception:
        return None


def _real_room_min_required_area(specs: List[SemanticRoomSpec], config: SolverConfig) -> float:
    tol = float(getattr(config, "area_tolerance", 0.15) or 0.15)
    tol = max(0.0, min(0.9, tol))
    return float(
        sum(
            max(0.0, float(getattr(s, "target_area", 0.0) or 0.0) * (1.0 - tol))
            for s in specs
            if not bool(getattr(s, "is_dummy", False))
        )
    )


def _runtime_dummy_facade_limit(config: SolverConfig, residual_area: float) -> float:
    base = float(getattr(config, "dummy_max_facade_shared_length", 3.0) or 3.0)
    if residual_area <= 0:
        return base
    return float(max(base, min(6.0, float(residual_area) / 2.0)))


def _apply_ground_floor_entrance_snap(
    *,
    floor_boundary: Polygon,
    rooms: List[RoomResult],
    corridors: List[Any],
    core_tube: Any,
    corridor_width: float,
    floor_number: Optional[int],
) -> List[str]:
    if int(floor_number or 0) != 1:
        return []
    if floor_boundary is None or floor_boundary.is_empty:
        return []

    warnings_out: List[str] = []
    min_width = max(1.2, float(corridor_width))

    corridor_candidates: List[Tuple[float, Any, str]] = []
    for corridor in corridors or []:
        poly = getattr(corridor, "polygon", None)
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        side, dist = _nearest_floor_side(poly, floor_boundary)
        if dist < ENTRANCE_SNAP_THRESHOLD:
            corridor_candidates.append((float(dist), corridor, side))
    corridor_candidates.sort(key=lambda item: float(item[0]))

    blockers = _layout_snap_blockers(rooms, core_tube)
    for dist, corridor, side in corridor_candidates:
        poly = getattr(corridor, "polygon", None)
        patch = _exterior_connection_box(poly, floor_boundary, side, min_width)
        if patch is None:
            continue
        if not _patches_clear([patch], blockers, floor_boundary):
            continue
        snapped = _merge_snap_patches(poly, [patch], floor_boundary)
        if snapped is None:
            continue
        corridor.polygon = snapped
        warnings_out.append(
            f"Ground entrance snapped corridor={getattr(corridor, 'id', '?')} "
            f"side={side} dist={dist:.2f} max_width={ENTRANCE_SNAP_MAX_CONNECTION_WIDTH:.2f}"
        )
        return warnings_out

    entrances = [
        r for r in rooms
        if str(getattr(r, "room_type", "") or "").lower() == "entrance"
        and isinstance(getattr(r, "polygon", None), Polygon)
        and not getattr(r, "polygon").is_empty
    ]
    for entrance in sorted(entrances, key=lambda r: _nearest_floor_side(r.polygon, floor_boundary)[1]):
        exterior_side, exterior_dist = _nearest_floor_side(entrance.polygon, floor_boundary)
        if exterior_dist >= ENTRANCE_SNAP_THRESHOLD:
            continue
        nearest_corridor = None
        nearest_corridor_dist = float("inf")
        for corridor in corridors or []:
            cpoly = getattr(corridor, "polygon", None)
            if not isinstance(cpoly, Polygon) or cpoly.is_empty:
                continue
            try:
                d = float(entrance.polygon.distance(cpoly))
            except Exception:
                continue
            if d < nearest_corridor_dist:
                nearest_corridor = corridor
                nearest_corridor_dist = d
        if nearest_corridor is None or nearest_corridor_dist >= ENTRANCE_SNAP_THRESHOLD:
            continue

        cpoly = getattr(nearest_corridor, "polygon", None)
        exterior_patch = _exterior_connection_box(entrance.polygon, floor_boundary, exterior_side, min_width)
        corridor_patch = _connection_box_between(entrance.polygon, cpoly, min_width)
        if exterior_patch is None or corridor_patch is None:
            continue
        blockers2 = _layout_snap_blockers(rooms, core_tube, exclude_ids={str(entrance.id)})
        if not _patches_clear([exterior_patch, corridor_patch], blockers2, floor_boundary):
            continue
        snapped = _merge_snap_patches(entrance.polygon, [exterior_patch, corridor_patch], floor_boundary)
        if snapped is None:
            continue
        _update_room_geometry(entrance, snapped, floor_boundary)
        warnings_out.append(
            f"Ground entrance two-way snapped room={entrance.id} "
            f"exterior_dist={exterior_dist:.2f} corridor_dist={nearest_corridor_dist:.2f}"
        )
        return warnings_out

    return warnings_out


def _floor_side_distances(poly: Polygon, floor_boundary: Polygon) -> List[Tuple[str, float]]:
    fminx, fminy, fmaxx, fmaxy = _axis_snap_bounds(floor_boundary)
    minx, miny, maxx, maxy = _axis_snap_bounds(poly)
    return [
        ("left", max(0.0, minx - fminx)),
        ("right", max(0.0, fmaxx - maxx)),
        ("bottom", max(0.0, miny - fminy)),
        ("top", max(0.0, fmaxy - maxy)),
    ]


def _room_exterior_gap_box(poly: Polygon, floor_boundary: Polygon, side: str) -> Optional[Polygon]:
    fminx, fminy, fmaxx, fmaxy = _axis_snap_bounds(floor_boundary)
    minx, miny, maxx, maxy = _axis_snap_bounds(poly)
    overshoot = ENTRANCE_SNAP_OVERSHOOT
    if side == "left":
        return box(fminx - overshoot, miny, minx, maxy)
    if side == "right":
        return box(maxx, miny, fmaxx + overshoot, maxy)
    if side == "bottom":
        return box(minx, fminy - overshoot, maxx, miny)
    if side == "top":
        return box(minx, maxy, maxx, fmaxy + overshoot)
    return None


def _layout_gap_blockers(
    rooms: List[RoomResult],
    corridors: List[Any],
    core_tube: Any,
    exclude_id: str,
    coverage_features: Optional[List[Any]] = None,
    generated_rooms: Optional[List[Any]] = None,
) -> List[Polygon]:
    blockers: List[Polygon] = []
    for room in list(rooms or []) + list(generated_rooms or []):
        rid = str(getattr(room, "id", "") or "")
        if rid == str(exclude_id):
            continue
        rtype = str(getattr(room, "room_type", "") or "").lower()
        if rtype == "void" or bool(getattr(room, "skip_solver", False)):
            continue
        poly = getattr(room, "polygon", None)
        if isinstance(poly, Polygon) and not poly.is_empty:
            blockers.append(poly)
    for corridor in corridors or []:
        poly = getattr(corridor, "polygon", None)
        if isinstance(poly, Polygon) and not poly.is_empty:
            blockers.append(poly)
    for feature in coverage_features or []:
        poly = feature.get("polygon") if isinstance(feature, dict) else getattr(feature, "polygon", None)
        if isinstance(poly, Polygon) and not poly.is_empty:
            blockers.append(poly)
    blockers.extend(_layout_snap_blockers([], core_tube))
    return blockers


def _coverage_feature_snap_state(coverage_features: Optional[List[Any]]) -> Dict[str, Tuple[float, Any]]:
    state: Dict[str, Tuple[float, Any]] = {}
    for index, feature in enumerate(coverage_features or []):
        if isinstance(feature, dict):
            feature_id = str(feature.get("feature_id") or feature.get("id") or f"coverage_feature_{index}")
            poly = feature.get("polygon")
        else:
            feature_id = str(
                getattr(feature, "feature_id", None)
                or getattr(feature, "id", None)
                or f"coverage_feature_{index}"
            )
            poly = getattr(feature, "polygon", None)
        if isinstance(poly, Polygon) and not poly.is_empty:
            state[feature_id] = (float(poly.area), poly.wkb)
    return state


def _apply_room_exterior_gap_snap(
    *,
    floor_boundary: Polygon,
    rooms: List[RoomResult],
    corridors: List[Any],
    core_tube: Any,
    coverage_features: Optional[List[Any]] = None,
    generated_rooms: Optional[List[Any]] = None,
) -> List[str]:
    if floor_boundary is None or floor_boundary.is_empty:
        return []

    feature_count = len(coverage_features or [])
    logger.info("[GAP] Exterior gap snap start | coverage_features=%d", int(feature_count))
    feature_state_before = _coverage_feature_snap_state(coverage_features)
    warnings_out: List[str] = []
    for room in rooms:
        rtype = str(getattr(room, "room_type", "") or "").lower()
        if rtype in ("void", "entrance") or bool(getattr(room, "skip_solver", False)):
            continue
        poly = getattr(room, "polygon", None)
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue

        for _ in range(4):
            candidates = [
                (side, dist)
                for side, dist in _floor_side_distances(room.polygon, floor_boundary)
                if ROOM_EXTERIOR_GAP_MIN < float(dist) <= ROOM_EXTERIOR_GAP_MAX
            ]
            if not candidates:
                break
            candidates.sort(key=lambda item: float(item[1]))
            applied = False
            for side, dist in candidates:
                patch = _room_exterior_gap_box(room.polygon, floor_boundary, side)
                if patch is None or patch.is_empty:
                    continue
                try:
                    clipped_patch = patch.intersection(floor_boundary)
                except Exception:
                    continue
                if clipped_patch.is_empty:
                    continue
                patch_area = float(getattr(clipped_patch, "area", 0.0))
                area_limit = min(
                    ROOM_EXTERIOR_GAP_MAX_AREA,
                    float(room.area) * ROOM_EXTERIOR_GAP_MAX_AREA_RATIO,
                )
                if patch_area > area_limit + ENTRANCE_SNAP_AREA_EPS:
                    continue
                blockers = _layout_gap_blockers(
                    rooms,
                    corridors,
                    core_tube,
                    exclude_id=str(room.id),
                    coverage_features=coverage_features,
                    generated_rooms=generated_rooms,
                )
                if not _patches_clear([patch], blockers, floor_boundary):
                    continue
                snapped = _merge_snap_patches(room.polygon, [patch], floor_boundary)
                if snapped is None:
                    continue
                _update_room_geometry(room, snapped, floor_boundary)
                warnings_out.append(
                    f"Exterior gap snapped room={room.id} side={side} dist={float(dist):.2f} area={patch_area:.2f}"
                )
                applied = True
                break
            if not applied:
                break
    feature_state_after = _coverage_feature_snap_state(coverage_features)
    if set(feature_state_before) != set(feature_state_after):
        raise LayoutCoverageError(
            "Coverage feature identity changed during exterior gap snap",
            stage="coverage_feature_gap_snap_drift",
            semantic_repair_allowed=False,
            metadata={
                "before_feature_ids": sorted(feature_state_before),
                "after_feature_ids": sorted(feature_state_after),
            },
        )
    for feature_id, (before_area, before_wkb) in feature_state_before.items():
        after_area, after_wkb = feature_state_after[feature_id]
        drift = abs(float(after_area) - float(before_area))
        if after_wkb != before_wkb:
            if drift <= 0.10:
                logger.warning(
                    "[GAP] Coverage feature adjustment detected | feature=%s | drift=%.4fm2",
                    feature_id,
                    float(drift),
                )
            else:
                raise LayoutCoverageError(
                    "Coverage feature geometry drifted during exterior gap snap",
                    stage="coverage_feature_gap_snap_drift",
                    semantic_repair_allowed=False,
                    max_gap_area=float(drift),
                    metadata={
                        "feature_id": feature_id,
                        "area_before": float(before_area),
                        "area_after": float(after_area),
                        "drift": float(drift),
                    },
                )
    logger.info("[GAP] Exterior gap snap complete | coverage_features=%d", int(feature_count))
    return warnings_out


def generate_layout_v2(
    floor_boundary: Polygon,
    room_specs: List[SemanticRoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "cross",
    entrance_position: Optional[Tuple[float, float]] = None,
    floor_number: Optional[int] = None,
    config: Optional[SolverConfig] = None,
    snap_grid: float = 0.1,
    verbose: bool = False,
    shared_core_tube: Optional[Any] = None,  # CoreTube, 跨层共享
    group_seed: Optional[int] = None,
    topology_snapshot: Optional[Any] = None,
    topology_attempt: int = 0,
    total_floors: Optional[int] = None,
    topology_mode: str = "continuous_cpsat",
    corridor_allowance_area: Optional[float] = None,
    floor_free_space: Optional[Any] = None,
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
    from .grid_growth_planner import GRID_GROWTH, plan_grid_growth_topology
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
    floor_usable_polygon = getattr(floor_free_space, "free_space_geometry", None) if floor_free_space is not None else None
    stage2a_floor_report = floor_free_space.to_report() if floor_free_space is not None else None

    def _resolve_snapshot_floor(snapshot: Optional[Any]) -> Optional[FloorTopologySnapshot]:
        if snapshot is None:
            return None
        if isinstance(snapshot, FloorTopologySnapshot):
            return snapshot
        fid = f"F{int(floor_number or 0)}"
        if isinstance(snapshot, TopologySnapshot):
            return snapshot.floors.get(fid) or snapshot.floors.get(str(int(floor_number or 0)))
        floors_obj = getattr(snapshot, "floors", None)
        if isinstance(floors_obj, dict):
            return floors_obj.get(fid) or floors_obj.get(str(int(floor_number or 0)))
        if isinstance(snapshot, dict):
            floors_dict = snapshot.get("floors")
            if isinstance(floors_dict, dict):
                candidate = floors_dict.get(fid) or floors_dict.get(str(int(floor_number or 0)))
                if isinstance(candidate, FloorTopologySnapshot):
                    return candidate
                if isinstance(candidate, dict):
                    return FloorTopologySnapshot(**candidate)
            if "island_rings" in snapshot:
                return FloorTopologySnapshot(**snapshot)
        return None

    snapshot_floor = _resolve_snapshot_floor(topology_snapshot)
    use_snapshot_topology = (
        floor_free_space is None
        and snapshot_floor is not None
        and int(topology_attempt or 0) <= 0
    )
    snapshot_runtime = None
    if use_snapshot_topology and snapshot_floor is not None:
        validate_snapshot_for_floor(
            snapshot_floor,
            floor_boundary,
            corridor_layout=corridor_layout,
            corridor_width=corridor_width,
            floor_count=total_floors,
        )
        snapshot_runtime = snapshot_floor_to_runtime(snapshot_floor)
        corridor_width = float(snapshot_floor.corridor_width)

    def _total_target_area(specs: List[SemanticRoomSpec]) -> float:
        return float(sum(max(0.0, float(s.target_area)) for s in specs))

    def _apply_global_area_scale(specs: List[SemanticRoomSpec], scale: float) -> None:
        for s in specs:
            current = float(s.target_area)
            if getattr(s, "raw_allocation_target_area", None) is None:
                setattr(s, "raw_allocation_target_area", current)
            if getattr(s, "target_area_raw", None) is None:
                try:
                    s.target_area_raw = current
                except Exception:
                    pass
            s.target_area = current * float(scale)
            setattr(s, "preferred_target_area", float(s.target_area))

    corridor_width_initial = float(corridor_width)
    max_iter = 1 if use_snapshot_topology else 5

    layout_l = str(corridor_layout or "").lower()
    if layout_l == "organic":
        min_cw = float(corridor_width)
        cw_step = 0.0
    else:
        min_cw = 1.2
        cw_step = 0.2
    if use_snapshot_topology:
        min_cw = float(corridor_width)
        cw_step = 0.0

    total_target_room_area = _total_target_area(room_specs)
    real_target_area = _real_room_target_area(room_specs)
    real_min_required_area = _real_room_min_required_area(room_specs, config)
    floor_area = float(floor_boundary.area)
    capacity_ratio = float(getattr(config, "semantic_capacity_ratio", 0.95) or 0.95)
    eps = 1e-6
    if layout_l != "organic":
        while True:
            a_total = float(floor_boundary.area)
            a_core_est = max(0.0, a_total * float(core_area_ratio))
            a_corr_est = estimate_corridor_area_upper(floor_boundary, float(corridor_width), corridor_layout)
            a_island_est = max(eps, a_total - a_core_est - a_corr_est)
            pressure_est = total_target_room_area / a_island_est
            if pressure_est <= 1.0 or corridor_width <= min_cw + 1e-6:
                break
            corridor_width = max(min_cw, float(corridor_width) - cw_step)

    acceptable = False
    scaled = False
    chosen_degradation = None
    core_tube = None
    core_contract = None
    core_budget_reconciliation: Dict[str, Any] = {}
    corridors = None
    islands = None
    assignments = None
    degradation = None
    solver_config = config
    grid_growth_metadata: Dict[str, Any] = {}
    topology_mode_l = str(topology_mode or "continuous_cpsat").lower()

    def _assign_rooms_to_topology():
        nonlocal corridors, islands, grid_growth_metadata
        if topology_mode_l == GRID_GROWTH:
            grid_result = plan_grid_growth_topology(
                floor_boundary=floor_boundary,
                core_tube=core_tube,
                room_specs=room_specs,
                adjacency_graph=adjacency_graph,
                corridor_width=float(corridor_width),
                corridor_layout=corridor_layout,
                floor_number=floor_number,
                core_contract=core_contract,
                enable_topology_assignment_cp_sat=bool(getattr(solver_config, "enable_topology_assignment_cp_sat", True)),
                topology_assignment_dry_run=bool(getattr(solver_config, "topology_assignment_dry_run", True)),
                enable_topology_assignment_adoption=bool(getattr(solver_config, "enable_topology_assignment_adoption", False)),
                allow_topology_assignment_fallback=bool(getattr(solver_config, "allow_topology_assignment_fallback", True)),
                enable_topology_assignment_relaxation_diagnostics=bool(getattr(solver_config, "enable_topology_assignment_relaxation_diagnostics", False)),
                topology_assignment_relaxation_time_limit_seconds=float(getattr(solver_config, "topology_assignment_relaxation_time_limit_seconds", 0.5)),
                topology_assignment_relaxation_total_time_limit_seconds=float(getattr(solver_config, "topology_assignment_relaxation_total_time_limit_seconds", 3.0)),
                topology_assignment_relaxation_max_levels=int(getattr(solver_config, "topology_assignment_relaxation_max_levels", 6)),
                topology_assignment_relaxation_num_workers=int(getattr(solver_config, "topology_assignment_relaxation_num_workers", 1)),
                floor_usable_polygon=floor_usable_polygon,
                enable_capacity_aware_area_allocation=bool(getattr(solver_config, "enable_capacity_aware_area_allocation", False)),
                apply_capacity_aware_area_allocation=bool(getattr(solver_config, "apply_capacity_aware_area_allocation", False)),
                capacity_aware_area_allocation_strict=bool(getattr(solver_config, "capacity_aware_area_allocation_strict", False)),
                capacity_aware_capacity_source=str(getattr(solver_config, "capacity_aware_capacity_source", "max_variant_effective_capacity")),
                capacity_aware_capacity_slack=float(getattr(solver_config, "capacity_aware_capacity_slack", 1.0)),
                capacity_aware_reserve_area=float(getattr(solver_config, "capacity_aware_reserve_area", 0.0)),
                capacity_aware_area_epsilon=float(getattr(solver_config, "capacity_aware_area_epsilon", 1e-6)),
                capacity_aware_preserve_preferred_when_feasible=bool(getattr(solver_config, "capacity_aware_preserve_preferred_when_feasible", True)),
                capacity_aware_require_apply_for_target_overflow=bool(getattr(solver_config, "capacity_aware_require_apply_for_target_overflow", False)),
                enable_semantic_seeded_territory_variants=bool(getattr(solver_config, "enable_semantic_seeded_territory_variants", False)),
                semantic_seeded_territory_variants_dry_run=bool(getattr(solver_config, "semantic_seeded_territory_variants_dry_run", True)),
            )
            corridors = grid_result.corridors
            islands = grid_result.islands
            grid_growth_metadata = dict(grid_result.metadata or {})
            if grid_result.warnings:
                warnings.extend(grid_result.warnings)
            logger.info(
                "[GRID] topology handoff | floor=%s | islands=%d | assignments=%d",
                floor_number,
                len(islands or []),
                len(grid_result.assignments or {}),
            )
            return grid_result.assignments, grid_result.degradation
        return assign_rooms_to_islands(
            islands=islands,
            rooms=room_specs,
            adjacency_graph=adjacency_graph,
            topology_mode=topology_mode_l,
        )

    for _it in range(max_iter):
        # ========== Phase 1: 拓扑生成 ==========
        try:
            fn = int(floor_number or 0)
            if use_snapshot_topology and snapshot_runtime is not None:
                core_tube = snapshot_runtime.core_tube or shared_core_tube
                corridors = list(snapshot_runtime.corridors)
                islands = list(snapshot_runtime.islands)
            else:
                if layout_l == "organic" and fn == 0:
                    logger.warning("floor_number missing, entrance forcing disabled (organic corridor)")
                core_tube, corridors, islands = generate_rectangular_topology(
                    floor_boundary=floor_boundary,
                    corridor_width=corridor_width,
                    core_area_ratio=core_area_ratio,
                    corridor_layout=corridor_layout,
                    entrance_position=entrance_position,
                    core_tube_override=shared_core_tube,
                    group_seed=group_seed,
                    force_corridor_boundary_contact=(fn == 1),
                )
            if floor_free_space is not None:
                core_tube = floor_free_space.stage1_core_tube
                core_contract = floor_free_space.core_contract
            if core_tube is not None:
                if core_contract is None:
                    core_contract = build_core_footprint_contract(
                        core_tube,
                        floor_id=f"F{int(floor_number or 1)}",
                        topology_mode=topology_mode_l,
                        created_from="generate_layout_v2",
                    )
                core_budget_reconciliation = reconcile_core_area_for_budget(
                    floor_id=f"F{int(floor_number or 1)}",
                    topology_mode=topology_mode_l,
                    core_contract=core_contract,
                    core_tube_area=float(floor_area) * float(core_area_ratio),
                    hard_fail=False,
                )
            if islands and topology_mode_l != GRID_GROWTH:
                current_island_area = float(sum(float(i.area) for i in islands))
                if real_min_required_area > current_island_area * capacity_ratio:
                    if int(topology_attempt or 0) > 0:
                        raise LayoutCoverageError(
                            "Retry topology overloaded before solver: "
                            f"target_sum={real_target_area:.2f}m2, "
                            f"min_required={real_min_required_area:.2f}m2, "
                            f"islands={current_island_area:.2f}m2, "
                            f"ratio={capacity_ratio:.2f}"
                        )
                    if use_snapshot_topology:
                        raise SemanticInvalidError(
                            "Total room area exceeds snapshot island capacity: "
                            f"target_sum={real_target_area:.2f}m2, "
                            f"min_required={real_min_required_area:.2f}m2, "
                            f"islands={current_island_area:.2f}m2, "
                            f"ratio={capacity_ratio:.2f}"
                        )
                    raise SemanticInvalidError(
                        "Total room area exceeds topology island capacity: "
                        f"target_sum={real_target_area:.2f}m2, "
                        f"min_required={real_min_required_area:.2f}m2, "
                        f"islands={current_island_area:.2f}m2, "
                        f"ratio={capacity_ratio:.2f}"
                    )
            if fn == 1:
                try:
                    corridor_union = unary_union([c.polygon for c in (corridors or [])]) if corridors else Polygon()
                    shared_len = float(corridor_union.intersection(floor_boundary.exterior).length)
                except Exception:
                    shared_len = -1.0
                logger.info("force_corridor_boundary_contact=%s, corridor_boundary_shared_len=%.4f", True, float(shared_len))
        except (LayoutCoverageError, SemanticInvalidError, LayoutGeometryInvariantError):
            raise
        except Exception as e:
            raise TopologyError(f"Failed to generate topology: {e}") from e

        pre_total_island_area = float(sum(float(i.area) for i in (islands or []))) if islands else 0.0
        residual_abort_ratio = float(getattr(config, "semantic_residual_abort_ratio", 0.15) or 0.15)
        if not is_stage2a_area_contract(topology_mode_l):
            apply_semantic_residual_gate(
                floor_id=f"F{int(floor_number or 1)}",
                topology_mode=topology_mode_l,
                floor_area=float(floor_area),
                budget_island_area=_snapshot_budget_island_area(topology_snapshot, f"F{int(floor_number or 1)}"),
                grid_island_area=pre_total_island_area,
                llm_room_sum=float(real_target_area),
                semantic_residual_abort_ratio=residual_abort_ratio,
                stage="pre_assignment",
            )

        # ========== Phase 2: 房间-岛屿分配 ==========
        assignments, degradation = _assign_rooms_to_topology()
        chosen_degradation = degradation

        total_island_area = float(sum(float(i.area) for i in islands)) if islands else 0.0
        residual_area = max(0.0, float(total_island_area) - float(real_target_area))
        residual_abort_ratio = float(getattr(config, "semantic_residual_abort_ratio", 0.15) or 0.15)
        apply_semantic_residual_gate(
            floor_id=f"F{int(floor_number or 1)}",
            topology_mode=topology_mode_l,
            floor_area=float(floor_area),
            budget_island_area=_snapshot_budget_island_area(topology_snapshot, f"F{int(floor_number or 1)}"),
            grid_island_area=total_island_area,
            llm_room_sum=float(real_target_area),
            semantic_residual_abort_ratio=residual_abort_ratio,
            stage="post_assignment",
        )
        solver_config = replace(
            config,
            dummy_dynamic_facade_shared_length=_runtime_dummy_facade_limit(config, residual_area),
        )
        pressure = total_target_room_area / max(eps, total_island_area)
        force_ratio = (len(degradation.force_shrunk) / max(1, len(room_specs))) if degradation else 0.0
        acceptable = (pressure <= 0.92) and (not degradation.skipped_rooms) and (force_ratio <= 0.05)
        if acceptable:
            break
        if corridor_width > min_cw + 1e-6:
            corridor_width = max(min_cw, float(corridor_width) - cw_step)
            continue
        break

    if not acceptable and float(corridor_width) <= min_cw + 1e-6:
        if islands:
            total_island_area = float(sum(float(i.area) for i in islands))
        else:
            total_island_area = 0.0
        pressure = total_target_room_area / max(eps, total_island_area)
        if pressure > 0.92 + 1e-6:
            scale = min(0.92 / pressure, 0.95)
            _apply_global_area_scale(room_specs, scale)
            total_target_room_area = _total_target_area(room_specs)
            real_min_required_area = _real_room_min_required_area(room_specs, config)
            scaled = True
            try:
                fn = int(floor_number or 0)
                core_tube, corridors, islands = generate_rectangular_topology(
                    floor_boundary=floor_boundary,
                    corridor_width=corridor_width,
                    core_area_ratio=core_area_ratio,
                    corridor_layout=corridor_layout,
                    entrance_position=entrance_position,
                    core_tube_override=shared_core_tube,
                    group_seed=group_seed,
                    force_corridor_boundary_contact=(fn == 1),
                )
                if fn == 1:
                    try:
                        corridor_union = unary_union([c.polygon for c in (corridors or [])]) if corridors else Polygon()
                        shared_len = float(corridor_union.intersection(floor_boundary.exterior).length)
                    except Exception:
                        shared_len = -1.0
                    logger.info("force_corridor_boundary_contact=%s, corridor_boundary_shared_len=%.4f", True, float(shared_len))
            except Exception as e:
                raise TopologyError(f"Failed to generate topology: {e}") from e
            assignments, degradation = _assign_rooms_to_topology()
            chosen_degradation = degradation
            real_target_area = _real_room_target_area(room_specs)
            real_min_required_area = _real_room_min_required_area(room_specs, config)
            total_island_area = float(sum(float(i.area) for i in islands)) if islands else 0.0
            residual_area = max(0.0, float(total_island_area) - float(real_target_area))
            residual_abort_ratio = float(getattr(config, "semantic_residual_abort_ratio", 0.15) or 0.15)
            apply_semantic_residual_gate(
                floor_id=f"F{int(floor_number or 1)}",
                topology_mode=topology_mode_l,
                floor_area=float(floor_area),
                budget_island_area=_snapshot_budget_island_area(topology_snapshot, f"F{int(floor_number or 1)}"),
                grid_island_area=total_island_area,
                llm_room_sum=float(real_target_area),
                semantic_residual_abort_ratio=residual_abort_ratio,
                stage="post_scale",
            )
            solver_config = replace(
                config,
                dummy_dynamic_facade_shared_length=_runtime_dummy_facade_limit(config, residual_area),
            )

    if abs(float(corridor_width) - corridor_width_initial) > 1e-6:
        warnings.append(f"Corridor width auto-tuned: {corridor_width_initial:.2f}->{float(corridor_width):.2f}")
    if scaled and chosen_degradation is not None:
        warnings.append("Room target areas scaled to fit physical limit")

    if verbose:
        logger.info(
            "Topology: %d islands, %d corridors",
            len(islands or []), len(corridors or []),
        )

    if assignments is None or degradation is None:
        assignments, degradation = _assign_rooms_to_topology()

    if core_tube is not None and core_contract is None:
        core_contract = build_core_footprint_contract(
            core_tube,
            floor_id=f"F{int(floor_number or 1)}",
            topology_mode=topology_mode_l,
            created_from="generate_layout_v2",
        )
        core_budget_reconciliation = reconcile_core_area_for_budget(
            floor_id=f"F{int(floor_number or 1)}",
            topology_mode=topology_mode_l,
            core_contract=core_contract,
            core_tube_area=float(floor_area) * float(core_area_ratio),
            hard_fail=False,
        )

    synthetic_room_results: List[RoomResult] = []
    synthetic_room_records: List[Dict[str, Any]] = []
    empty_island_actions: List[Dict[str, Any]] = []
    islands, synthetic_room_results, synthetic_room_records, empty_island_actions = _empty_island_sweep(
        islands=list(islands or []),
        assignments=assignments,
        corridors=list(corridors or []),
        floor_boundary=floor_boundary,
        floor_number=int(floor_number),
    )
    if empty_island_actions:
        warnings.append(f"Empty island sweep actions: {empty_island_actions}")

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

    islands_by_id = {i.id: i for i in (islands or [])}
    adj_warnings = _check_cross_island_adjacency(
        room_specs, room_to_island, islands_by_id,
    )
    warnings.extend(adj_warnings)

    # ========== Phase 4: 岛屿内划分 ==========
    all_miqp_results: List[MIQPRoomResult] = []
    all_specs_ordered: List[SemanticRoomSpec] = []
    solver_metadata: Dict[str, Any] = {}
    if grid_growth_metadata:
        solver_metadata["grid_growth"] = grid_growth_metadata
    if stage2a_floor_report is not None:
        solver_metadata["floor_free_space"] = dict(stage2a_floor_report)
    if core_contract is not None:
        solver_metadata["core_contract"] = {
            "core_contract_id": core_contract.core_contract_id,
            "version": core_contract.version,
            "created_from": core_contract.created_from,
            "floor_id": core_contract.floor_id,
            "topology_mode": core_contract.topology_mode,
            "core_union_hash": core_contract.core_union_hash,
            "core_union_area": float(core_contract.core_union.area),
            "core_union_bounds": tuple(core_contract.core_union_bounds),
            "core_public_union_area": float(core_contract.core_public_union.area),
            "public_hall_ids": list(core_contract.public_hall_ids),
            "shaft_ids": list(core_contract.shaft_ids),
            "budget_reconciliation": dict(core_budget_reconciliation or {}),
        }
    coverage_debt_plans: Dict[str, Any] = {}

    for island_id, assignment in assignments.items():
        island = islands_by_id[island_id]
        coverage_policy = "coverage_debt" if topology_mode_l == GRID_GROWTH else "legacy"
        coverage_debt_plan = None
        if topology_mode_l == GRID_GROWTH:
            coverage_debt_plan = build_coverage_debt_plan(
                floor_id=f"F{int(floor_number or 1)}",
                island_id=str(island_id),
                topology_mode=topology_mode_l,
                island_area=float(getattr(island, "area", 0.0) or getattr(getattr(island, "polygon", None), "area", 0.0) or 0.0),
                assigned_rooms=list(getattr(assignment, "rooms", []) or []),
                policy=CoverageDebtPolicy(
                    area_tolerance_min_default=max(
                        0.0,
                        1.0 - float(getattr(solver_config, "area_tolerance", 0.15) or 0.15),
                    ),
                    area_tolerance_max_default=1.0 + float(getattr(solver_config, "area_tolerance", 0.15) or 0.15),
                ),
            )
            try:
                coverage_debt_plan.diagnostics.update(
                    {
                        "variant_id": str(getattr(island, "topology_variant_id", "") or grid_growth_metadata.get("runtime_topology_variant_id", "")),
                        "core_union_hash": str(
                            getattr(island, "core_union_hash", "")
                            or getattr(core_contract, "core_union_hash", "")
                            or ""
                        ),
                        "island_area_source": "adopted_variant_polygon"
                        if grid_growth_metadata.get("runtime_topology_variant_id")
                        else "primary_variant_polygon",
                        "coverage_debt_input_area_allocation_id": str(grid_growth_metadata.get("active_area_allocation_id", "") or ""),
                        "coverage_debt_input_area_target_hash": str(grid_growth_metadata.get("active_target_hash", "") or ""),
                        "coverage_debt_uses_geometry_target_area": bool(
                            (grid_growth_metadata.get("capacity_aware_area_allocation") or {}).get("area_compression_applied")
                        )
                        if isinstance(grid_growth_metadata.get("capacity_aware_area_allocation"), dict)
                        else False,
                    }
                )
            except Exception:
                pass
            logger.info(
                "[DEBT] Core-aware debt input | floor=F%s | island=%s | old_area=%.2f | core_aware_area=%.2f | target_sum=%.2f | planned_residual=%.2f | core_union_hash=%s",
                int(floor_number or 1),
                str(island_id),
                float(getattr(island, "legacy_area", getattr(island, "area", 0.0)) or 0.0),
                float(getattr(island, "area", 0.0) or 0.0),
                float(sum(float(getattr(r, "target_area", 0.0) or 0.0) for r in list(getattr(assignment, "rooms", []) or []))),
                float(getattr(coverage_debt_plan, "planned_residual_area", 0.0) or 0.0),
                getattr(island, "core_union_hash", None) or getattr(core_contract, "core_union_hash", None),
            )
            assigned_ids_expected = {
                str(getattr(r, "room_id", "") or getattr(r, "id", ""))
                for r in list(getattr(assignment, "rooms", []) or [])
            }
            if set(coverage_debt_plan.assigned_room_ids) != assigned_ids_expected:
                raise LayoutCoverageError(
                    "Coverage debt plan room assignment mismatch",
                    floor_number=int(floor_number or 1),
                    max_gap_area=0.0,
                    stage="coverage_debt_planning_failed",
                    semantic_repair_allowed=False,
                    metadata={
                        "failure_kind": "coverage",
                        "stage": "coverage_debt_planning_failed",
                        "topology_mode": topology_mode_l,
                        "island_id": str(island_id),
                        "plan_room_ids": list(coverage_debt_plan.assigned_room_ids),
                        "assignment_room_ids": sorted(assigned_ids_expected),
                    },
                )
            coverage_debt_plans[str(island_id)] = coverage_debt_plan
            solver_metadata.setdefault("coverage_debt", {})[str(island_id)] = coverage_debt_plan.to_dict()

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
            miqp_results, island_solver_metadata = partition_island_semantic_with_metadata(
                island_polygon=island.polygon,
                rooms=assignment.rooms,
                adjacency_graph=island_adjacency,
                exterior_walls=exterior_walls,
                config=solver_config,
                corridor_edges=corridor_edges,
                coverage_policy=coverage_policy,
                coverage_debt_plan=coverage_debt_plan,
                core_contract=core_contract,
            )
            solver_metadata[str(island_id)] = island_solver_metadata
            all_miqp_results.extend(miqp_results)
            all_specs_ordered.extend(assignment.rooms)
        except LayoutGeometryInvariantError as e:
            if grid_growth_metadata:
                failed_room_id = str((getattr(e, "metadata", {}) or {}).get("room_id", "") or "")
                proposal = grid_growth_metadata.get("topology_assignment_proposal")
                if failed_room_id and isinstance(proposal, dict):
                    cluster_id = None
                    for cluster in list(grid_growth_metadata.get("clusters", []) or []):
                        if failed_room_id in set(cluster.get("rooms", []) or []):
                            cluster_id = str(cluster.get("cluster_id", "") or "")
                            break
                    if cluster_id:
                        feasibility_rows = list(grid_growth_metadata.get("cluster_island_feasibility", []) or [])

                        def _fit_score(target: Any) -> Optional[float]:
                            if not isinstance(target, dict):
                                return None
                            variant_id = str(target.get("variant_id", "") or "")
                            island_id = str(target.get("island_id", "") or "")
                            for row in feasibility_rows:
                                if (
                                    str(row.get("variant_id", "") or "") == variant_id
                                    and str(row.get("cluster_id", "") or "") == cluster_id
                                    and str(row.get("island_id", "") or "") == island_id
                                ):
                                    try:
                                        return float(row.get("feasibility_score"))
                                    except Exception:
                                        return None
                            return None

                        heuristic = dict((proposal.get("heuristic_cluster_to_island") or {}).get(cluster_id, {}) or {})
                        proposed = dict((proposal.get("proposed_cluster_to_island") or {}).get(cluster_id, {}) or {})
                        proposal["failed_cluster_diagnostics"] = {
                            "status": "ok",
                            "failed_room_id": failed_room_id,
                            "failed_cluster_id": cluster_id,
                            "heuristic_variant_id": heuristic.get("variant_id"),
                            "heuristic_island_id": heuristic.get("island_id"),
                            "proposal_variant_id": proposed.get("variant_id"),
                            "proposal_island_id": proposed.get("island_id"),
                            "heuristic_fit_score": _fit_score(heuristic),
                            "proposal_fit_score": _fit_score(proposed),
                        }
                    else:
                        proposal["failed_cluster_diagnostics"] = {
                            "status": "room_cluster_not_found",
                            "failed_room_id": failed_room_id,
                        }
                e.metadata.setdefault("grid_growth", grid_growth_metadata)
                e.metadata.setdefault("topology_metadata", grid_growth_metadata)
            raise
        except Exception as e:
            # MIQP 失败 → fallback 到基础求解器
            if topology_mode_l == GRID_GROWTH:
                logger.warning(
                    "Semantic MIQP failed for %s under coverage_debt policy: %s",
                    island_id,
                    e,
                )
                metadata = {
                    "failure_kind": "coverage",
                    "stage": "coverage_debt_solver_failed",
                    "topology_mode": topology_mode_l,
                    "island_id": str(island_id),
                    "coverage_debt_plan_id": str(getattr(coverage_debt_plan, "plan_id", "")),
                    "coverage_debt_plan": (
                        coverage_debt_plan.to_dict()
                        if hasattr(coverage_debt_plan, "to_dict")
                        else {}
                    ),
                    "solver_error": str(e),
                    "semantic_repair_allowed": False,
                }
                raise LayoutCoverageError(
                    "Grid growth coverage-debt solver failed before residual sponge: "
                    f"island={island_id}, error={e}",
                    floor_number=int(floor_number or 1),
                    max_gap_area=float(getattr(coverage_debt_plan, "planned_residual_area", 0.0) or 0.0),
                    metadata=metadata,
                    stage="coverage_debt_solver_failed",
                    semantic_repair_allowed=False,
                ) from e
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
                solver_metadata[str(island_id)] = {
                    "status_name": "FALLBACK",
                    "attempt_name": "basic_solver",
                    "error": str(e),
                }
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
                is_dummy=bool(getattr(spec, "is_dummy", False)),
                target_area_raw=float(getattr(spec, "target_area_raw", None) or spec.target_area)
                if bool(getattr(spec, "is_dummy", False)) else None,
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
            is_dummy=bool(getattr(spec, "is_dummy", False)),
            target_area_raw=float(getattr(spec, "target_area_raw", None) or spec.target_area)
            if bool(getattr(spec, "is_dummy", False)) else None,
        ))

    if synthetic_room_results:
        rooms.extend(synthetic_room_results)

    coverage_features: List[Dict[str, Any]] = []
    residual_actions: List[Dict[str, Any]] = []
    coverage_boundary = floor_usable_polygon if floor_usable_polygon is not None else floor_boundary
    rooms, synthetic_room_records, residual_actions = _apply_intra_island_residual_sweep(
        islands=list(islands or []),
        assignments=assignments,
        rooms=rooms,
        corridors=list(corridors or []),
        floor_boundary=coverage_boundary,
        floor_number=int(floor_number),
        synthetic_records=synthetic_room_records,
        topology_mode=topology_mode_l,
        coverage_debt_plans=coverage_debt_plans,
        coverage_features=coverage_features,
        corridor_allowance_area=corridor_allowance_area,
        core_contract=core_contract,
    )
    if residual_actions:
        warnings.append(f"Intra-island residual sweep actions: {residual_actions}")

    floor_residual_actions: List[Dict[str, Any]] = []
    floor_residual_stats: Dict[str, Any] = {}
    rooms, synthetic_room_records, floor_residual_actions, floor_residual_stats = _apply_floor_level_residual_sweep(
        rooms=rooms,
        corridors=list(corridors or []),
        core_tube=core_tube,
        floor_boundary=coverage_boundary,
        floor_number=int(floor_number),
        synthetic_records=synthetic_room_records,
        topology_mode=topology_mode_l,
        coverage_features=coverage_features,
        corridor_allowance_area=corridor_allowance_area,
        core_contract=core_contract,
    )
    if floor_residual_actions:
        warnings.append(f"Floor-level residual sweep actions: {floor_residual_actions}")
    if float(floor_residual_stats.get("ignored_micro_gap_total", 0.0) or 0.0) > 1.0:
        warnings.append(f"Large ignored micro gap total: {floor_residual_stats}")

    # ========== Phase 6: 连通性检查（拓扑 BFS，零浮点缓冲） ==========
    entrance_snap_warnings = _apply_ground_floor_entrance_snap(
        floor_boundary=floor_boundary,
        rooms=rooms,
        corridors=list(corridors or []),
        core_tube=core_tube,
        corridor_width=float(corridor_width),
        floor_number=floor_number,
    )
    warnings.extend(entrance_snap_warnings)
    exterior_gap_warnings = _apply_room_exterior_gap_snap(
        floor_boundary=floor_boundary,
        rooms=rooms,
        corridors=list(corridors or []),
        core_tube=core_tube,
        coverage_features=coverage_features,
    )
    warnings.extend(exterior_gap_warnings)

    coverage_gap = compute_layout_coverage_gap(
        floor_boundary=coverage_boundary,
        rooms=rooms,
        corridors=list(corridors or []),
        core_tube=core_tube,
        coverage_features=coverage_features,
    )
    logger.info(
        "[COVERAGE] Preflight | raw_total=%.4fm2 | significant_total=%.4fm2 | max_piece=%.4fm2 | erosion=%.3fm | ignored_micro_count=%d | ignored_micro_area=%.4fm2",
        float(coverage_gap.raw_total_gap_area),
        float(coverage_gap.total_gap_area),
        float(coverage_gap.max_gap_area),
        float(coverage_gap.gap_erosion_tolerance),
        int(coverage_gap.ignored_micro_gap_count) + int(floor_residual_stats.get("ignored_micro_gap_count", 0) or 0),
        float(coverage_gap.ignored_micro_gap_total) + float(floor_residual_stats.get("ignored_micro_gap_total", 0.0) or 0.0),
    )
    macro_void_threshold = float(getattr(solver_config, "macro_void_threshold", 1.5) or 1.5)
    if coverage_gap.max_gap_area > macro_void_threshold:
        gap_pieces = []
        for piece in coverage_gap.gap_pieces:
            gap_pieces.append(_polygon_shape_metadata(piece))
        logger.error(
            "[COVERAGE] Result: FAIL | Floor=%s | MaxGap=%.2fm2 | TotalGap=%.2fm2 | Threshold=%.2fm2 | GapPieces=%s",
            floor_number,
            float(coverage_gap.max_gap_area),
            float(coverage_gap.total_gap_area),
            macro_void_threshold,
            gap_pieces,
        )
        raise LayoutCoverageError(
            "Macro void remains after layout preflight: "
            f"max_gap={coverage_gap.max_gap_area:.2f}m2, "
            f"total_gap={coverage_gap.total_gap_area:.2f}m2, "
            f"threshold={macro_void_threshold:.2f}m2",
            floor_number=int(floor_number),
            max_gap_area=float(coverage_gap.max_gap_area),
            metadata={
                "total_gap_area": float(coverage_gap.total_gap_area),
                "raw_total_gap_area": float(coverage_gap.raw_total_gap_area),
                "significant_total_gap_area": float(coverage_gap.total_gap_area),
                "gap_erosion_tolerance": float(coverage_gap.gap_erosion_tolerance),
                "ignored_micro_gap_total": float(coverage_gap.ignored_micro_gap_total)
                + float(floor_residual_stats.get("ignored_micro_gap_total", 0.0) or 0.0),
                "ignored_micro_gap_count": int(coverage_gap.ignored_micro_gap_count)
                + int(floor_residual_stats.get("ignored_micro_gap_count", 0) or 0),
                "gap_pieces": gap_pieces,
                "empty_island_actions": empty_island_actions,
                "residual_actions": residual_actions,
                "floor_residual_actions": floor_residual_actions,
                "synthetic_rooms": synthetic_room_records,
                "coverage_features": [_serializable_feature(f) for f in coverage_features],
            },
        )
    logger.info(
        "[COVERAGE] Result: PASS | Floor=%s | MaxGap=%.4fm2 | TotalGap=%.4fm2 | Threshold=%.2fm2",
        floor_number,
        float(coverage_gap.max_gap_area),
        float(coverage_gap.total_gap_area),
        macro_void_threshold,
    )
    if core_contract is not None:
        core_exclusion_diagnostics = validate_core_exclusion(
            floor_id=f"F{int(floor_number or 1)}",
            topology_mode=topology_mode_l,
            core_contract=core_contract,
            rooms=rooms,
            generated_rooms=[],
            coverage_features=coverage_features,
            corridors=list(corridors or []),
            epsilon_area=CORE_OVERLAP_EPSILON_AREA,
            hard_fail=(topology_mode_l == GRID_GROWTH),
        )
        solver_metadata.setdefault("core_contract", {})["exclusion_diagnostics"] = core_exclusion_diagnostics
        if "floor_free_space" in solver_metadata:
            floor_space_meta = solver_metadata["floor_free_space"]
            floor_space_meta["core_positive_overlap_area"] = max(
                float(core_exclusion_diagnostics.get("room_core_overlap_total", 0.0) or 0.0),
                float(core_exclusion_diagnostics.get("corridor_core_overlap_total", 0.0) or 0.0),
                float(core_exclusion_diagnostics.get("coverage_feature_core_overlap_total", 0.0) or 0.0),
            )
            floor_space_meta["coverage_fallback_touched_core"] = (
                float(core_exclusion_diagnostics.get("coverage_feature_core_overlap_total", 0.0) or 0.0)
                > CORE_OVERLAP_EPSILON_AREA
            )
    if coverage_features:
        solver_metadata.setdefault("coverage_debt", {})["coverage_features"] = [
            _serializable_feature(f) for f in coverage_features
        ]
    coverage_ledger = []
    for action in list(residual_actions or []) + list(floor_residual_actions or []):
        if action.get("accepted_as_coverage_feature") or action.get("final_status"):
            coverage_ledger.append({
                "floor_id": f"F{int(floor_number)}",
                "island_id": str(action.get("island_id", action.get("source", "__floor_level__"))),
                "source_residual_id": str(action.get("source_residual_id", "")),
                "source_residual_area": float(action.get("area", 0.0) or 0.0),
                "source_residual_fill_rate": float(action.get("fill_rate", 0.0) or 0.0),
                "action_sequence": [str(action.get("action", ""))],
                "materialized_features": list(action.get("materialized_features", []) or []),
                "remaining_uncovered_area": float(action.get("remaining_uncovered_area", 0.0) or 0.0),
                "qa_status": str(action.get("qa_status", "")),
                "final_status": str(action.get("final_status", "")),
            })
    if coverage_ledger:
        solver_metadata.setdefault("coverage_debt", {})["ledger"] = coverage_ledger
        logger.info(
            "[LEDGER] Coverage debt finalized | floor=F%d | entries=%d | final_status=%s",
            int(floor_number),
            len(coverage_ledger),
            ",".join(sorted({str(e.get("final_status", "")) for e in coverage_ledger})),
        )

    rects: Dict[str, Tuple[float, float, float, float]] = {}
    if core_tube is not None and hasattr(core_tube, "polygon") and not core_tube.polygon.is_empty:
        subzones = [
            ("core_staircase", getattr(core_tube, "staircase", None)),
            ("core_elevator_hall", getattr(core_tube, "elevator_hall", None)),
            ("core_elevator_shaft", getattr(core_tube, "elevator_shaft", None)),
        ]
        has_subzones = all(z is not None and hasattr(z, "is_empty") and not z.is_empty for _, z in subzones)
        if has_subzones:
            for zid, zpoly in subzones:
                if zpoly is None or zpoly.is_empty:
                    continue
                minx, miny, maxx, maxy = zpoly.bounds
                rects[zid] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
        else:
            minx, miny, maxx, maxy = core_tube.polygon.bounds
            rects["core_tube"] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
    for c in (corridors or []):
        if hasattr(c, "polygon") and c.polygon is not None and not c.polygon.is_empty:
            minx, miny, maxx, maxy = c.polygon.bounds
            rects[c.id] = (float(minx), float(miny), float(maxx - minx), float(maxy - miny))
    room_ids = []
    for r in rooms:
        if str(getattr(r, "room_type", "") or "").lower() == "void" or bool(getattr(r, "skip_solver", False)):
            continue
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
        corridors=list(corridors or []),
        islands=list(islands or []),
        assignments=assignments,
        room_layouts=rooms,
        edge_set=edge_set,
        validation=report,
        generation_time_ms=elapsed_ms,
        warnings=warnings,
        corridor_layout=str(corridor_layout or ""),
        solver_metadata=solver_metadata,
        synthetic_rooms=synthetic_room_records,
        required_adjacency={
            str(spec.room_id): [str(adj) for adj in list(getattr(spec, "adjacency_required", []) or [])]
            for spec in room_specs
        },
    )
