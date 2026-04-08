"""
房间-岛屿分配器

实现分层决策：
1. 面积预检查
2. 硬约束过滤（采光、面积）
3. 功能分区匹配
4. 邻接约束优化
5. 面积平衡填充
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .room_spec import RoomSpec, ZoneType
from .topology_generator import Island

logger = logging.getLogger(__name__)


class AssignmentError(Exception):
    """分配错误"""
    pass


@dataclass
class AssignerConfig:
    """分配器配置"""
    # 评分权重
    zone_match_score: int = 50
    required_adj_score: int = 30
    preferred_adj_score: int = 10
    utilization_score: int = 20

    # 阈值
    min_utilization: float = 0.7
    max_utilization: float = 0.9
    area_tolerance: float = 0.85  # 面积容量检查的容差
    capacity_ratio: float = 0.95  # 总面积预检查比例


@dataclass
class AssignmentResult:
    """分配结果"""
    island_id: str
    rooms: List[RoomSpec]
    total_area: float
    utilization: float


class IslandRoomAssigner:
    """
    房间-岛屿分配器

    分配策略：
    1. 面积预检查：总面积是否足够
    2. 硬约束过滤：排除不满足条件的岛屿
    3. 功能分区匹配：公共→公共岛，私密→私密岛
    4. 邻接约束优化：相关房间尽量同岛
    5. 面积平衡填充：最大化利用率
    """

    def __init__(
        self,
        islands: List[Island],
        rooms: List[RoomSpec],
        adjacency_graph: Dict[str, List[str]],
        config: Optional[AssignerConfig] = None,
    ):
        self.islands = {i.id: i for i in islands}
        self.rooms = {r.room_id: r for r in rooms}
        self.adjacency = adjacency_graph
        self.config = config or AssignerConfig()

        # 分配结果
        self.assignments: Dict[str, List[str]] = defaultdict(list)
        self.room_to_island: Dict[str, str] = {}

    def assign(self) -> Dict[str, AssignmentResult]:
        """
        执行分配

        返回:
            {island_id: AssignmentResult}

        Raises:
            AssignmentError: 面积不足或房间无法分配
        """
        # 预检查：总面积
        total_room_area = sum(r.target_area for r in self.rooms.values())
        total_island_area = sum(i.area for i in self.islands.values())

        if total_room_area > total_island_area * self.config.capacity_ratio:
            raise AssignmentError(
                f"Insufficient island capacity: rooms need {total_room_area:.1f}m², "
                f"islands have {total_island_area:.1f}m²"
            )

        # Step 1: 按优先级排序房间
        sorted_rooms = self._sort_rooms()

        # Step 2: 逐个分配房间
        unassigned_rooms: List[RoomSpec] = []

        for room in sorted_rooms:
            # 2.1 获取候选岛屿
            candidates = self._get_candidate_islands(room)

            if not candidates:
                # 放宽约束重试
                candidates = self._get_candidate_islands_relaxed(room)

            if not candidates:
                unassigned_rooms.append(room)
                logger.warning(f"No valid island for room: {room.room_id}")
                continue

            # 2.2 评分并选择最佳岛屿
            best_island = self._select_best_island(room, candidates)

            # 2.3 分配
            self._assign_room(room, best_island)

        if unassigned_rooms:
            raise AssignmentError(
                f"Cannot assign {len(unassigned_rooms)} rooms: "
                f"{[r.room_id for r in unassigned_rooms]}"
            )

        # Step 3: 生成结果
        return self._build_results()

    def _sort_rooms(self) -> List[RoomSpec]:
        """
        排序房间（分配优先级）

        规则：
        1. needs_window 的房间优先（外墙岛屿有限）
        2. adjacency_required 多的优先（约束强）
        3. 面积大的优先（选择余地小）
        """
        def priority_key(room: RoomSpec) -> Tuple:
            return (
                -int(room.needs_window),
                -len(room.adjacency_required),
                -room.target_area,
                -room.area_priority,
            )

        return sorted(self.rooms.values(), key=priority_key)

    def _get_candidate_islands(self, room: RoomSpec) -> List[Island]:
        """获取候选岛屿（通过硬约束过滤）"""
        candidates = []

        for island in self.islands.values():
            # 约束 1: 面积足够
            if island.remaining_capacity < room.target_area * self.config.area_tolerance:
                continue

            # 约束 2: 采光需求
            if room.needs_window and not island.has_exterior_wall:
                continue

            # 约束 3: 禁止邻接
            if self._has_forbidden_neighbor(room, island):
                continue

            candidates.append(island)

        return candidates

    def _get_candidate_islands_relaxed(self, room: RoomSpec) -> List[Island]:
        """放宽约束的候选岛屿（仅检查面积）"""
        candidates = []
        for island in self.islands.values():
            if island.remaining_capacity >= room.target_area * 0.7:
                candidates.append(island)
        return candidates

    def _has_forbidden_neighbor(self, room: RoomSpec, island: Island) -> bool:
        """检查岛屿内是否有禁止邻接的房间"""
        assigned_rooms = self.assignments.get(island.id, [])
        for assigned_id in assigned_rooms:
            if assigned_id in room.adjacency_forbidden:
                return True
            assigned_room = self.rooms.get(assigned_id)
            if assigned_room and room.room_id in assigned_room.adjacency_forbidden:
                return True
        return False

    def _select_best_island(
        self,
        room: RoomSpec,
        candidates: List[Island],
    ) -> Island:
        """
        选择最佳岛屿

        评分规则：
        1. 功能分区匹配
        2. 已有必须邻接的房间
        3. 已有偏好邻接的房间
        4. 剩余容量匹配
        """
        cfg = self.config
        scores: List[Tuple[Island, int]] = []

        for island in candidates:
            score = 0

            # 1. 功能分区匹配
            if island.suggested_zone.value == room.zone.value:
                score += cfg.zone_match_score

            # 2. 必须邻接的房间
            assigned = set(self.assignments.get(island.id, []))
            for adj_id in room.adjacency_required:
                if adj_id in assigned:
                    score += cfg.required_adj_score

            # 3. 偏好邻接的房间
            for adj_id in room.adjacency_preferred:
                if adj_id in assigned:
                    score += cfg.preferred_adj_score

            # 4. 容量匹配（利用率在目标区间最佳）
            utilization_after = (
                (island.area - island.remaining_capacity + room.target_area)
                / island.area
            )
            if cfg.min_utilization <= utilization_after <= cfg.max_utilization:
                score += cfg.utilization_score
            elif utilization_after > 0.95:
                score -= 10  # 太满

            # 5. 邻接关系的对称考虑
            for assigned_id in assigned:
                assigned_room = self.rooms.get(assigned_id)
                if assigned_room:
                    if room.room_id in assigned_room.adjacency_required:
                        score += cfg.required_adj_score
                    if room.room_id in assigned_room.adjacency_preferred:
                        score += cfg.preferred_adj_score

            scores.append((island, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _assign_room(self, room: RoomSpec, island: Island):
        """分配房间到岛屿"""
        self.assignments[island.id].append(room.room_id)
        self.room_to_island[room.room_id] = island.id
        island.remaining_capacity -= room.target_area
        island.assigned_rooms.append(room.room_id)

    def _build_results(self) -> Dict[str, AssignmentResult]:
        """构建分配结果"""
        results = {}

        for island_id, room_ids in self.assignments.items():
            island = self.islands[island_id]
            rooms = [self.rooms[rid] for rid in room_ids]
            total_area = sum(r.target_area for r in rooms)

            results[island_id] = AssignmentResult(
                island_id=island_id,
                rooms=rooms,
                total_area=total_area,
                utilization=total_area / island.area if island.area > 0 else 0,
            )

        return results


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════


def assign_rooms_to_islands(
    islands: List[Island],
    rooms: List[RoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
    config: Optional[AssignerConfig] = None,
) -> Dict[str, AssignmentResult]:
    """
    便捷函数：房间-岛屿分配

    Args:
        islands: 岛屿列表
        rooms: 房间规格列表
        adjacency_graph: 邻接关系图
        config: 分配器配置

    Returns:
        {island_id: AssignmentResult}
    """
    if adjacency_graph is None:
        adjacency_graph = {}
        for room in rooms:
            adjacency_graph[room.room_id] = (
                room.adjacency_required + room.adjacency_preferred
            )

    assigner = IslandRoomAssigner(islands, rooms, adjacency_graph, config)
    return assigner.assign()
