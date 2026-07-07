"""
building_orchestrator.py

多层 Building 编排器

职责：
1. 接收 BuildingAllocation（多楼层语义配比）
2. 首层生成核心筒 → 锁定位置
3. 逐层调用 generate_layout_v2，注入共享核心筒
4. 验证垂直一致性
"""
from __future__ import annotations

import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from ..models import BuildingAllocation, FloorAllocation
from .layout_generator import (
    LayoutResultV2,
    generate_layout_v2,
)
from .exceptions import LayoutGeometryInvariantError
from .floor_free_space import FloorFreeSpace
from .postprocessor import LayoutCoverageError, LayoutTopologyError, SemanticInvalidError
from .room_spec import (
    RoomSpec as SemanticRoomSpec,
    SolverConfig,
    ZoneType,
    apply_room_type_defaults,
)
from .topology_generator import CoreTube
from .topology_snapshot import TopologySnapshot, core_tube_from_ring

logger = logging.getLogger(__name__)


@dataclass
class BuildingResult:
    """多层建筑生成结果"""
    core_tube: Optional[CoreTube]  # 跨层共享
    floor_layouts: Dict[str, LayoutResultV2]
    warnings: List[str] = field(default_factory=list)


class BuildingOrchestrator:
    """
    多层建筑编排器

    职责：
    1. 接收 BuildingAllocation
    2. 首层生成核心筒 → 锁定位置
    3. 逐层调用 generate_layout_v2，注入共享核心筒
    4. 验证垂直一致性
    """

    def __init__(
        self,
        floor_boundary: Polygon,
        config: Optional[SolverConfig] = None,
        corridor_width: float = 2.0,
        core_area_ratio: float = 0.08,
        corridor_layout: str = "door_side",
        topology_mode: str = "continuous_cpsat",
        base_seed: Optional[int] = None,
        corridor_grouping: str = "standard",
        topology_snapshot: Optional[TopologySnapshot] = None,
        floor_free_spaces: Optional[Dict[str, FloorFreeSpace]] = None,
    ):
        self.floor_boundary = floor_boundary
        self.config = config or SolverConfig()
        self.corridor_width = corridor_width
        self.core_area_ratio = core_area_ratio
        self.corridor_layout = corridor_layout
        self.topology_mode = topology_mode
        self.base_seed = base_seed
        self.corridor_grouping = corridor_grouping
        self.topology_snapshot = topology_snapshot
        self.floor_free_spaces = dict(floor_free_spaces or {})
        self._shared_core_tube: Optional[CoreTube] = None
        if (
            not self.floor_free_spaces
            and topology_snapshot is not None
            and topology_snapshot.fixed_core_ring is not None
        ):
            self._shared_core_tube = core_tube_from_ring(topology_snapshot.fixed_core_ring)

    def generate(
        self,
        allocation: BuildingAllocation,
        topology_snapshot: Optional[TopologySnapshot] = None,
    ) -> BuildingResult:
        """
        生成整栋建筑

        流程：
        1. 首层生成拓扑 → 锁定核心筒
        2. 后续楼层复用核心筒位置
        3. 验证垂直管井对齐
        """
        floor_layouts: Dict[str, LayoutResultV2] = {}
        warnings: List[str] = []
        active_snapshot = None if self.floor_free_spaces else (topology_snapshot or self.topology_snapshot)
        if active_snapshot is not None and self._shared_core_tube is None and active_snapshot.fixed_core_ring is not None:
            self._shared_core_tube = core_tube_from_ring(active_snapshot.fixed_core_ring)

        import hashlib

        total_floors = int(getattr(allocation, "total_floors", 0) or len(allocation.floors))

        group_seed_cache: Dict[str, int] = {}

        def _group_seed_for_floor(f: FloorAllocation, attempt: int = 0) -> Optional[int]:
            base = int(self.base_seed or 0) + int(attempt) * 100003
            fn = int(getattr(f, "floor_number", 0) or 0)
            if str(self.corridor_layout or "").lower() != "organic":
                if self.base_seed is None and int(attempt) == 0:
                    return None
                return base + fn
            if str(self.corridor_grouping or "").lower() == "by_function_tag":
                tag = str(getattr(f, "floor_function_tag", "") or "unknown")
                if fn in (1, total_floors):
                    key = f"special:{fn}:{tag}:{attempt}"
                else:
                    key = f"tag:{tag}:{attempt}"
                if key not in group_seed_cache:
                    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
                    group_seed_cache[key] = base + int(h, 16)
                return group_seed_cache[key]
            if fn in (1, total_floors):
                return base + fn
            return base + 1000

        for i, floor in enumerate(allocation.floors):
            floor_id = f"F{floor.floor_number}"
            floor_free_space = self.floor_free_spaces.get(floor_id)
            if floor_free_space is not None:
                self._shared_core_tube = floor_free_space.stage1_core_tube

            if not getattr(floor, "rooms", None):
                logger.warning(f"Floor {floor_id} has no rooms, skipping")
                continue

            max_retries = max(1, int(getattr(self.config, "max_layout_retries", 5) or 5))
            layout: Optional[LayoutResultV2] = None
            last_retry_error: Optional[Exception] = None

            for attempt in range(max_retries):
                # generate_layout_v2 may scale specs in-place, so rebuild them per retry.
                room_specs = self._convert_floor_rooms(floor)
                if not room_specs:
                    logger.warning(f"Floor {floor_id} has no rooms, skipping")
                    break

                adjacency_graph = self._build_adjacency_graph(room_specs)
                group_seed = _group_seed_for_floor(floor, attempt)
                corridor_width = float(self.corridor_width)
                try:
                    layout = generate_layout_v2(
                        floor_boundary=self.floor_boundary,
                        room_specs=room_specs,
                        adjacency_graph=adjacency_graph,
                        config=self.config,
                        corridor_width=corridor_width,
                        core_area_ratio=self.core_area_ratio,
                        corridor_layout=self.corridor_layout,
                        topology_mode=self.topology_mode,
                        shared_core_tube=self._shared_core_tube,
                        group_seed=group_seed,
                        floor_number=int(getattr(floor, "floor_number", 0) or 0),
                        topology_snapshot=active_snapshot,
                        topology_attempt=attempt,
                        total_floors=total_floors,
                        corridor_allowance_area=float(getattr(floor, "corridor_allowance_area", 0.0) or 0.0),
                        floor_free_space=floor_free_space,
                    )
                    if attempt > 0:
                        warnings.append(f"Floor {floor_id} recovered after retry {attempt + 1}/{max_retries}")
                    break
                except SemanticInvalidError:
                    raise
                except (LayoutTopologyError, LayoutCoverageError, LayoutGeometryInvariantError) as e:
                    if hasattr(e, "with_floor"):
                        e.with_floor(int(getattr(floor, "floor_number", 0) or 0), floor_id)
                    else:
                        try:
                            setattr(e, "floor_number", int(getattr(floor, "floor_number", 0) or 0))
                            setattr(e, "floor_id", floor_id)
                        except Exception:
                            pass
                    last_retry_error = e
                    msg = (
                        f"Floor {floor_id} seed attempt {attempt + 1}/{max_retries} failed: {e}. "
                        "Retrying..."
                    )
                    warnings.append(msg)
                    logger.warning(msg)
                    continue

            if layout is None:
                if last_retry_error is not None:
                    raise last_retry_error
                continue

            # 首层锁定核心筒
            if floor_free_space is not None:
                self._shared_core_tube = floor_free_space.stage1_core_tube
            elif i == 0 and layout.core_tube is not None:
                core = layout.core_tube
                if isinstance(core, CoreTube):
                    self._shared_core_tube = core

            floor_layouts[floor_id] = layout
            warnings.extend(layout.warnings)

            logger.info(
                f"Floor {floor_id}: {len(layout.room_layouts)} rooms, "
                f"time={layout.generation_time_ms:.0f}ms"
            )

        # 验证垂直一致性
        vertical_warnings = self._validate_vertical_alignment(floor_layouts)
        warnings.extend(vertical_warnings)

        return BuildingResult(
            core_tube=self._shared_core_tube,
            floor_layouts=floor_layouts,
            warnings=warnings,
        )

    def generate_floor_with_retry(
        self,
        floor: FloorAllocation,
        *,
        total_floors: int,
        topology_snapshot: Optional[TopologySnapshot] = None,
        floor_index: int = 0,
    ) -> Tuple[str, LayoutResultV2, List[str]]:
        """Generate one floor with the same retry behavior used by generate()."""
        floor_input = copy.deepcopy(floor)
        floor_id = f"F{floor_input.floor_number}"
        floor_free_space = self.floor_free_spaces.get(floor_id)
        active_snapshot = None if floor_free_space is not None else copy.deepcopy(topology_snapshot or self.topology_snapshot)
        if floor_free_space is not None:
            self._shared_core_tube = floor_free_space.stage1_core_tube
        if active_snapshot is not None and self._shared_core_tube is None and active_snapshot.fixed_core_ring is not None:
            self._shared_core_tube = core_tube_from_ring(active_snapshot.fixed_core_ring)

        import hashlib

        def _group_seed_for_floor(f: FloorAllocation, attempt: int = 0) -> Optional[int]:
            base = int(self.base_seed or 0) + int(attempt) * 100003
            fn = int(getattr(f, "floor_number", 0) or 0)
            if str(self.corridor_layout or "").lower() != "organic":
                if self.base_seed is None and int(attempt) == 0:
                    return None
                return base + fn
            tag = str(getattr(f, "floor_function_tag", "") or "unknown")
            if str(self.corridor_grouping or "").lower() == "by_function_tag":
                key = f"{'special' if fn in (1, total_floors) else 'tag'}:{fn if fn in (1, total_floors) else tag}:{attempt}"
                h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
                return base + int(h, 16)
            if fn in (1, total_floors):
                return base + fn
            return base + 1000

        warnings: List[str] = []
        max_retries = max(1, int(getattr(self.config, "max_layout_retries", 5) or 5))
        last_retry_error: Optional[Exception] = None

        for attempt in range(max_retries):
            room_specs = self._convert_floor_rooms(floor_input)
            if not room_specs:
                raise LayoutTopologyError(f"Floor {floor_id} has no rooms").with_floor(
                    int(getattr(floor_input, "floor_number", 0) or 0),
                    floor_id,
                )

            adjacency_graph = self._build_adjacency_graph(room_specs)
            try:
                logger.info(
                    "[STAGE] Floor snapshot isolated | floor=%s | attempt=%d",
                    floor_id,
                    attempt + 1,
                )
                layout = generate_layout_v2(
                    floor_boundary=self.floor_boundary,
                    room_specs=room_specs,
                    adjacency_graph=adjacency_graph,
                    config=self.config,
                    corridor_width=float(self.corridor_width),
                    core_area_ratio=self.core_area_ratio,
                    corridor_layout=self.corridor_layout,
                    topology_mode=self.topology_mode,
                    shared_core_tube=self._shared_core_tube,
                    group_seed=_group_seed_for_floor(floor_input, attempt),
                    floor_number=int(getattr(floor_input, "floor_number", 0) or 0),
                    topology_snapshot=copy.deepcopy(active_snapshot),
                    topology_attempt=attempt,
                    total_floors=total_floors,
                    corridor_allowance_area=float(getattr(floor_input, "corridor_allowance_area", 0.0) or 0.0),
                    floor_free_space=floor_free_space,
                )
                if attempt > 0:
                    warnings.append(f"Floor {floor_id} recovered after retry {attempt + 1}/{max_retries}")
                if floor_free_space is not None:
                    self._shared_core_tube = floor_free_space.stage1_core_tube
                elif floor_index == 0 and layout.core_tube is not None and isinstance(layout.core_tube, CoreTube):
                    self._shared_core_tube = layout.core_tube
                warnings.extend(list(getattr(layout, "warnings", []) or []))
                logger.info(
                    "Floor %s: %d rooms, time=%.0fms",
                    floor_id,
                    len(layout.room_layouts),
                    layout.generation_time_ms,
                )
                return floor_id, layout, warnings
            except SemanticInvalidError:
                raise
            except (LayoutTopologyError, LayoutCoverageError, LayoutGeometryInvariantError) as e:
                if hasattr(e, "with_floor"):
                    e.with_floor(int(getattr(floor_input, "floor_number", 0) or 0), floor_id)
                last_retry_error = e
                msg = (
                    f"Floor {floor_id} seed attempt {attempt + 1}/{max_retries} failed: {e}. "
                    "Retrying..."
                )
                warnings.append(msg)
                logger.warning(msg)
                continue

        if last_retry_error is not None:
            raise last_retry_error
        raise LayoutTopologyError(f"Floor {floor_id} generation failed").with_floor(
            int(getattr(floor_input, "floor_number", 0) or 0),
            floor_id,
        )

    def _convert_floor_rooms(
        self,
        floor: FloorAllocation,
    ) -> List[SemanticRoomSpec]:
        """将 FloorAllocation.rooms 转换为 SemanticRoomSpec"""
        specs = []
        for idx, room in enumerate(floor.rooms):
            # room_id: 优先使用 LLM 生成的 ID，否则自动生成
            room_id = room.room_id if room.room_id else f"room_{idx:03d}"

            # zone: string → enum
            try:
                zone = ZoneType(room.zone)
            except ValueError:
                zone = ZoneType.PUBLIC

            # needs_window: 已在 _normalize_allocation 中推导完成
            needs_window = room.needs_window

            # aspect_ratio_range
            ar = room.aspect_ratio_range
            if len(ar) == 2:
                aspect_ratio_range = (float(ar[0]), float(ar[1]))
            else:
                aspect_ratio_range = (0.5, 2.0)

            spec = SemanticRoomSpec(
                room_id=room_id,
                room_type=room.room_type,
                target_area=room.target_area,
                zone=zone,
                needs_window=needs_window,
                min_width=room.min_width,
                min_depth=getattr(room, 'min_depth', room.min_width),
                aspect_ratio_range=aspect_ratio_range,
                adjacency_required=list(room.adjacency_required),
                adjacency_preferred=list(room.adjacency_preferred),
                adjacency_forbidden=list(room.adjacency_forbidden),
                area_priority=float(room.weight),
            )
            apply_room_type_defaults(spec)
            try:
                raw_target = float(getattr(spec, "target_area", room.target_area) or 0.0)
                spec.target_area_raw = raw_target
                setattr(spec, "raw_allocation_target_area", raw_target)
                setattr(spec, "preferred_target_area", raw_target)
            except Exception:
                pass
            specs.append(spec)

        return specs

    @staticmethod
    def _build_adjacency_graph(
        room_specs: List[SemanticRoomSpec],
    ) -> Dict[str, List[str]]:
        """从 SemanticRoomSpec 构建邻接图"""
        graph: Dict[str, List[str]] = {}
        for spec in room_specs:
            neighbors = list(spec.adjacency_required)
            if spec.adjacency_preferred:
                neighbors.extend(spec.adjacency_preferred)
            if neighbors:
                graph[spec.room_id] = neighbors
        return graph

    @staticmethod
    def _validate_vertical_alignment(
        floor_layouts: Dict[str, LayoutResultV2],
    ) -> List[str]:
        """验证垂直管井对齐"""
        warnings = []

        if len(floor_layouts) < 2:
            return warnings

        layouts = list(floor_layouts.values())
        floor_ids = list(floor_layouts.keys())

        # 获取首层核心筒位置
        ref_core = layouts[0].core_tube
        if ref_core is None:
            return warnings

        ref_cx = ref_core.polygon.centroid.x
        ref_cy = ref_core.polygon.centroid.y

        for floor_id, layout in zip(floor_ids[1:], layouts[1:]):
            core = layout.core_tube
            if core is None:
                continue

            cx = core.polygon.centroid.x
            cy = core.polygon.centroid.y

            dx = abs(cx - ref_cx)
            dy = abs(cy - ref_cy)

            if dx > 0.1 or dy > 0.1:
                warnings.append(
                    f"Floor {floor_id} core tube offset from reference: "
                    f"dx={dx:.2f}m, dy={dy:.2f}m"
                )

        return warnings
