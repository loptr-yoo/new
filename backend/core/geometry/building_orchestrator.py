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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from ...models import BuildingAllocation, FloorAllocation
from .layout_generator import (
    LayoutResultV2,
    generate_layout_v2,
)
from .room_spec import (
    RoomSpec as SemanticRoomSpec,
    SolverConfig,
    ZoneType,
    apply_room_type_defaults,
)
from .topology_generator import CoreTube

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
        base_seed: Optional[int] = None,
        corridor_grouping: str = "standard",
    ):
        self.floor_boundary = floor_boundary
        self.config = config or SolverConfig()
        self.corridor_width = corridor_width
        self.core_area_ratio = core_area_ratio
        self.corridor_layout = corridor_layout
        self.base_seed = base_seed
        self.corridor_grouping = corridor_grouping
        self._shared_core_tube: Optional[CoreTube] = None

    def generate(
        self,
        allocation: BuildingAllocation,
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

        import hashlib

        total_floors = int(getattr(allocation, "total_floors", 0) or len(allocation.floors))

        group_seed_cache: Dict[str, int] = {}

        def _group_seed_for_floor(f: FloorAllocation) -> Optional[int]:
            if str(self.corridor_layout or "").lower() != "organic":
                return None
            base = int(self.base_seed or 0)
            fn = int(getattr(f, "floor_number", 0) or 0)
            if str(self.corridor_grouping or "").lower() == "by_function_tag":
                tag = str(getattr(f, "floor_function_tag", "") or "unknown")
                if fn in (1, total_floors):
                    key = f"special:{fn}:{tag}"
                else:
                    key = f"tag:{tag}"
                if key not in group_seed_cache:
                    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
                    group_seed_cache[key] = base + int(h, 16)
                return group_seed_cache[key]
            if fn in (1, total_floors):
                return base + fn
            return base + 1000

        for i, floor in enumerate(allocation.floors):
            floor_id = f"F{floor.floor_number}"

            # 转换房间规格
            room_specs = self._convert_floor_rooms(floor)

            if not room_specs:
                logger.warning(f"Floor {floor_id} has no rooms, skipping")
                continue

            # 构建邻接图
            adjacency_graph = self._build_adjacency_graph(room_specs)

            # 生成布局
            group_seed = _group_seed_for_floor(floor)
            corridor_width = float(self.corridor_width)
            if str(self.corridor_layout or "").lower() == "organic":
                corridor_width = float(min(max(corridor_width, 1.5), 1.8))
            layout = generate_layout_v2(
                floor_boundary=self.floor_boundary,
                room_specs=room_specs,
                adjacency_graph=adjacency_graph,
                config=self.config,
                corridor_width=corridor_width,
                core_area_ratio=self.core_area_ratio,
                corridor_layout=self.corridor_layout,
                shared_core_tube=self._shared_core_tube,
                group_seed=group_seed,
            )

            # 首层锁定核心筒
            if i == 0 and layout.core_tube is not None:
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
