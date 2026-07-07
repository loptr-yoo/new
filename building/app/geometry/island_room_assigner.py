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
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .exceptions import LayoutAssignmentError
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

    hole_fill_enabled: bool = True
    hole_ratio_threshold: float = 0.10
    dummy_room_type: str = "utility"
    dummy_id_prefix: str = "room_dummy_"
    dummy_min_area: float = 2.0
    dummy_max_area: float = 15.0
    dummy_fill_utilization: float = 0.98
    topology_mode: str = "continuous_cpsat"


@dataclass
class DegradationSummary:
    """降级摘要：记录分配过程中的所有降级操作"""
    skipped_rooms: List[str] = field(default_factory=list)
    force_shrunk: List[str] = field(default_factory=list)
    miqp_fallback_floors: List[str] = field(default_factory=list)
    adjacency_dropped: int = 0
    parse_warnings: List[str] = field(default_factory=list)


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
        self._overall_utilization: float = 0.0

        # 分配结果
        self.assignments: Dict[str, List[str]] = defaultdict(list)
        self.room_to_island: Dict[str, str] = {}

    def assign(self) -> Tuple[Dict[str, AssignmentResult], DegradationSummary]:
        """
        执行分配。永不抛 AssignmentError。

        4 级降级策略：
        1. 正常候选（硬约束过滤）
        2. 放宽（任何有剩余容量的岛）
        3. 强制分配（缩小面积，但有 min_area 下限）
        4. 跳过 + warning

        返回:
            (assignments, degradation_summary)
        """
        degradation = DegradationSummary()

        total_room_area = sum(r.target_area for r in self.rooms.values())
        total_island_area = sum(i.area for i in self.islands.values())

        # 动态面积缩放
        if total_island_area <= 0:
            # 0 岛屿：跳过所有房间
            for room in self.rooms.values():
                degradation.skipped_rooms.append(room.room_id)
            logger.warning("No islands available, skipping all rooms")
            return self._build_results(), degradation

        if total_room_area > total_island_area * 0.95:
            scale_factor = (total_island_area * 0.92) / total_room_area
            logger.warning(
                "Scaling down room areas by %.1f%% (rooms=%.1fm2, islands=%.1fm2)",
                (1 - scale_factor) * 100, total_room_area, total_island_area,
            )
            for room in self.rooms.values():
                room.target_area *= scale_factor
            total_room_area = sum(r.target_area for r in self.rooms.values())
        elif total_room_area < total_island_area * 0.7:
            max_island = max(i.area for i in self.islands.values())
            max_room = max(r.target_area for r in self.rooms.values()) if self.rooms else 1
            scale_limit = min(3.0, max_island / max_room) if max_room > 0 else 3.0
            scale_factor = min((total_island_area * 0.85) / total_room_area, scale_limit)
            if scale_factor > 1.0:
                logger.info(
                    "Scaling up room areas by %.1f%% (rooms=%.1fm2, islands=%.1fm2)",
                    (scale_factor - 1) * 100, total_room_area, total_island_area,
                )
                for room in self.rooms.values():
                    room.target_area *= scale_factor
                total_room_area = sum(r.target_area for r in self.rooms.values())

        # round-up 保护：缩放后面积不低于 min_width × min_depth
        for room in self.rooms.values():
            min_viable = room.min_width * room.min_depth
            if room.target_area < min_viable:
                room.target_area = min_viable

        self._overall_utilization = (
            float(total_room_area) / float(total_island_area)
            if total_island_area > 1e-9
            else 0.0
        )

        # 按优先级排序
        sorted_rooms = self._sort_rooms()

        for room in sorted_rooms:
            # 级别 1: 正常候选
            candidates = self._get_candidate_islands(room)

            # 级别 2: 放宽
            if not candidates:
                candidates = self._get_candidate_islands_relaxed(room)

            # 级别 3: 强制分配（缩小面积，有下限）
            if not candidates:
                min_area = room.min_width * room.min_depth
                largest = max(
                    self.islands.values(),
                    key=lambda i: i.remaining_capacity,
                )
                if largest.remaining_capacity >= min_area:
                    room.target_area = largest.remaining_capacity * 0.9
                    candidates = [largest]
                    degradation.force_shrunk.append(room.room_id)
                    logger.warning(
                        f"Force-shrinking room {room.room_id} to "
                        f"{room.target_area:.1f}m2 (island {largest.id})"
                    )

            # 级别 4: 跳过
            if not candidates:
                degradation.skipped_rooms.append(room.room_id)
                logger.warning(f"Skipping room {room.room_id}: no viable island")
                continue

            best = self._select_best_island(room, candidates)
            self._assign_room(room, best)

        self._log_assignment_state("initial")
        self._rebalance_large_empty_islands()
        self._validate_assignment_invariants()
        self._log_assignment_state("post_rebalance")

        if str(getattr(self.config, "topology_mode", "") or "").lower() == "grid_growth":
            logger.info("[ASSIGN] Skip legacy dummy injection | reason=grid_growth_coverage_debt")
        else:
            self._inject_dummy_rooms(degradation)
        return self._build_results(), degradation

    def _is_dummy_or_storage(self, room: RoomSpec) -> bool:
        room_type = str(getattr(room, "room_type", "") or "").lower()
        return bool(getattr(room, "is_dummy", False)) or room_type in {"dummy", "storage", "utility", "void"}

    def _real_assigned_ids(self, island_id: str) -> List[str]:
        return [
            rid for rid in self.assignments.get(island_id, [])
            if rid in self.rooms and not self._is_dummy_or_storage(self.rooms[rid])
        ]

    def _room_area_sum(self, room_ids: List[str]) -> float:
        return sum(float(self.rooms[rid].target_area) for rid in room_ids if rid in self.rooms)

    def _build_required_clusters(self) -> Dict[str, Set[str]]:
        room_ids = {rid for rid, room in self.rooms.items() if not self._is_dummy_or_storage(room)}
        parent = {rid: rid for rid in room_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            if a not in parent or b not in parent:
                return
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for rid in list(room_ids):
            room = self.rooms[rid]
            for adj in list(getattr(room, "adjacency_required", []) or []):
                union(rid, str(adj))

        groups: Dict[str, Set[str]] = defaultdict(set)
        for rid in room_ids:
            groups[find(rid)].add(rid)
        cluster_by_room: Dict[str, Set[str]] = {}
        for ids in groups.values():
            frozen = set(ids)
            for rid in ids:
                cluster_by_room[rid] = frozen
        return cluster_by_room

    def _clusters_on_island(self, island_id: str, cluster_by_room: Dict[str, Set[str]]) -> List[Set[str]]:
        seen: Set[Tuple[str, ...]] = set()
        clusters: List[Set[str]] = []
        assigned = set(self._real_assigned_ids(island_id))
        for rid in sorted(assigned):
            cluster = set(cluster_by_room.get(rid, {rid})) & assigned
            key = tuple(sorted(cluster))
            if key and key not in seen:
                seen.add(key)
                clusters.append(cluster)
        return clusters

    def _forbidden_conflicts_between(self, a_ids: Set[str], b_ids: Set[str]) -> List[Tuple[str, str]]:
        conflicts: List[Tuple[str, str]] = []
        for a in sorted(a_ids):
            a_room = self.rooms.get(a)
            if not a_room:
                continue
            a_forbidden = {str(x) for x in list(getattr(a_room, "adjacency_forbidden", []) or [])}
            for b in sorted(b_ids):
                b_room = self.rooms.get(b)
                if not b_room:
                    continue
                b_forbidden = {str(x) for x in list(getattr(b_room, "adjacency_forbidden", []) or [])}
                if b in a_forbidden or a in b_forbidden:
                    conflicts.append((a, b))
        return conflicts

    def _is_public_access_island(self, island: Island) -> bool:
        return bool(list(getattr(island, "corridor_edges", []) or []))

    def _has_any_public_access_island(self) -> bool:
        return any(self._is_public_access_island(i) for i in self.islands.values())

    def _cluster_requires_public_access(self, cluster: Set[str]) -> bool:
        for rid in cluster:
            room = self.rooms.get(rid)
            if not room or self._is_dummy_or_storage(room):
                continue
            if bool(getattr(room, "needs_corridor_access", True)):
                return True
        return False

    def _cluster_move_rejection_reason(
        self,
        cluster: Set[str],
        donor: Island,
        target: Island,
    ) -> Optional[str]:
        cluster_area = self._room_area_sum(list(cluster))
        if cluster_area <= 1e-9:
            return "empty_cluster"
        if cluster_area > float(target.remaining_capacity) + 1e-6:
            return "insufficient_target_capacity"
        donor_remaining_real = set(self._real_assigned_ids(donor.id)) - set(cluster)
        if float(donor.area) > 8.0 and not donor_remaining_real:
            return "would_empty_donor_island"
        if self._has_any_public_access_island() and self._cluster_requires_public_access(set(cluster)) and not self._is_public_access_island(target):
            return "landlocked_cluster"
        target_real = set(self._real_assigned_ids(target.id))
        conflicts = self._forbidden_conflicts_between(set(cluster), target_real)
        if conflicts:
            return "forbidden_collision:" + ",".join(f"{a}->{b}" for a, b in conflicts)
        return None

    def _move_cluster(self, cluster: Set[str], donor: Island, target: Island) -> None:
        area = self._room_area_sum(list(cluster))
        for rid in sorted(cluster):
            if rid in self.assignments.get(donor.id, []):
                self.assignments[donor.id].remove(rid)
            if rid in donor.assigned_rooms:
                donor.assigned_rooms.remove(rid)
            self.assignments[target.id].append(rid)
            target.assigned_rooms.append(rid)
            self.room_to_island[rid] = target.id
        donor.remaining_capacity += area
        target.remaining_capacity -= area

    def _rebalance_large_empty_islands(self) -> None:
        cluster_by_room = self._build_required_clusters()
        large_empty = [
            island for island in self.islands.values()
            if float(island.area) > 8.0 and not self._real_assigned_ids(island.id)
        ]
        if not large_empty:
            return

        for target in sorted(large_empty, key=lambda i: float(i.area), reverse=True):
            target_goal = min(float(target.area) * 0.70, max(0.0, float(target.area) - 1.0))
            rejection_notes: List[str] = []
            logger.debug(
                "[ASSIGN] Rebalance large empty island start | target=%s | area=%.2fm2 | goal=%.2fm2",
                target.id,
                float(target.area),
                target_goal,
            )

            while self._room_area_sum(self._real_assigned_ids(target.id)) < target_goal:
                best: Optional[Tuple[Island, Set[str], Tuple[int, float, int, float]]] = None
                donors = [
                    island for island in self.islands.values()
                    if island.id != target.id and len(self._clusters_on_island(island.id, cluster_by_room)) > 0
                ]
                donors.sort(
                    key=lambda i: self._room_area_sum(self._real_assigned_ids(i.id)) / max(float(i.area), 1e-9),
                    reverse=True,
                )

                target_assigned_area = self._room_area_sum(self._real_assigned_ids(target.id))
                needed_to_goal = max(0.0, target_goal - target_assigned_area)

                for donor in donors:
                    clusters = self._clusters_on_island(donor.id, cluster_by_room)
                    clusters.sort(
                        key=lambda c: (
                            0 if len(c) == 1 else 1,
                            -self._room_area_sum(list(c)),
                        )
                    )
                    for cluster in clusters:
                        reason = self._cluster_move_rejection_reason(cluster, donor, target)
                        if reason:
                            note = f"{donor.id}->{target.id} cluster={sorted(cluster)} reject_reason={reason}"
                            rejection_notes.append(note)
                            logger.debug("[ASSIGN] Reject cluster move | %s", note)
                            continue
                        cluster_area = self._room_area_sum(list(cluster))
                        donor_fill = self._room_area_sum(self._real_assigned_ids(donor.id)) / max(float(donor.area), 1e-9)
                        independent_rank = 0 if len(cluster) == 1 else 1
                        if cluster_area >= needed_to_goal > 1e-9:
                            score = (0, cluster_area, independent_rank, -donor_fill)
                        else:
                            score = (1, -cluster_area, independent_rank, -donor_fill)
                        if best is None or score < best[2]:
                            best = (donor, set(cluster), score)

                if best is None:
                    self._raise_assignment_error(
                        "Unable to rebalance large empty island with semantic-safe clusters",
                        target,
                        rejection_notes,
                    )

                donor, cluster, _score = best
                before_target = self._room_area_sum(self._real_assigned_ids(target.id)) / max(float(target.area), 1e-9)
                before_donor = self._room_area_sum(self._real_assigned_ids(donor.id)) / max(float(donor.area), 1e-9)
                self._move_cluster(cluster, donor, target)
                after_target = self._room_area_sum(self._real_assigned_ids(target.id)) / max(float(target.area), 1e-9)
                after_donor = self._room_area_sum(self._real_assigned_ids(donor.id)) / max(float(donor.area), 1e-9)
                logger.debug(
                    "[ASSIGN] Move cluster | source=%s | target=%s | rooms=%s | area=%.2fm2 | "
                    "source_fill %.2f->%.2f | target_fill %.2f->%.2f",
                    donor.id,
                    target.id,
                    sorted(cluster),
                    self._room_area_sum(list(cluster)),
                    before_donor,
                    after_donor,
                    before_target,
                    after_target,
                )

                if not donors:
                    break

            if not self._real_assigned_ids(target.id):
                self._raise_assignment_error(
                    "Large usable island remained empty after semantic-safe rebalance",
                    target,
                    rejection_notes,
                )

    def _validate_assignment_invariants(self) -> None:
        required_conflicts: List[Tuple[str, str, str, str]] = []
        forbidden_conflicts: List[Tuple[str, str, str]] = []
        landlocked_conflicts: List[Tuple[str, Tuple[str, ...]]] = []
        cluster_by_room = self._build_required_clusters()
        checked_clusters: Set[Tuple[str, Tuple[str, ...]]] = set()

        for rid, room in self.rooms.items():
            if self._is_dummy_or_storage(room) or rid not in self.room_to_island:
                continue
            island_id = self.room_to_island[rid]
            for adj in list(getattr(room, "adjacency_required", []) or []):
                adj = str(adj)
                if adj in self.room_to_island and self.room_to_island[adj] != island_id:
                    required_conflicts.append((rid, adj, island_id, self.room_to_island[adj]))
            for other in self.assignments.get(island_id, []):
                if other == rid or other not in self.rooms:
                    continue
                conflicts = self._forbidden_conflicts_between({rid}, {other})
                for a, b in conflicts:
                    key = (a, b, island_id)
                    rev = (b, a, island_id)
                    if key not in forbidden_conflicts and rev not in forbidden_conflicts:
                        forbidden_conflicts.append(key)
            cluster = tuple(sorted(cluster_by_room.get(rid, {rid})))
            cluster_key = (str(island_id), cluster)
            if cluster_key not in checked_clusters:
                checked_clusters.add(cluster_key)
                island = self.islands.get(island_id)
                if (
                    island is not None
                    and self._has_any_public_access_island()
                    and self._cluster_requires_public_access(set(cluster))
                    and not self._is_public_access_island(island)
                ):
                    landlocked_conflicts.append((str(island_id), cluster))

        large_empty = [
            island for island in self.islands.values()
            if float(island.area) > 8.0 and not self._real_assigned_ids(island.id)
        ]
        if required_conflicts or forbidden_conflicts or landlocked_conflicts or large_empty:
            target = large_empty[0] if large_empty else next(iter(self.islands.values()))
            notes = []
            notes.extend(f"required_cross_island:{a}->{b}:{ia}->{ib}" for a, b, ia, ib in required_conflicts)
            notes.extend(f"forbidden_same_island:{a}->{b}:{island}" for a, b, island in forbidden_conflicts)
            notes.extend(f"landlocked_cluster:{island}:{list(cluster)}" for island, cluster in landlocked_conflicts)
            self._raise_assignment_error("Island assignment invariant failed", target, notes)

    def _assignment_metadata(self, target: Island, rejection_notes: List[str]) -> Dict[str, object]:
        fill_rates = {}
        for island_id, island in sorted(self.islands.items()):
            real_ids = self._real_assigned_ids(island_id)
            fill_rates[island_id] = {
                "area": round(float(island.area), 4),
                "assigned_area": round(self._room_area_sum(real_ids), 4),
                "fill_rate": round(self._room_area_sum(real_ids) / max(float(island.area), 1e-9), 4),
                "rooms": list(real_ids),
            }
        minx, miny, maxx, maxy = getattr(target, "polygon").bounds if hasattr(target, "polygon") else (0, 0, 0, 0)
        width = max(0.0, float(maxx - minx))
        height = max(0.0, float(maxy - miny))
        aspect = max(width, height) / max(min(width, height), 1e-9)
        return {
            "island_id": str(target.id),
            "area": float(target.area),
            "bbox": [float(minx), float(miny), float(maxx), float(maxy)],
            "aspect_ratio": float(aspect),
            "all_island_fill_rates": fill_rates,
            "room_target_summary": {
                rid: round(float(room.target_area), 4)
                for rid, room in sorted(self.rooms.items())
                if not self._is_dummy_or_storage(room)
            },
            "candidate_rejections": list(rejection_notes)[-50:],
        }

    def _raise_assignment_error(self, message: str, target: Island, rejection_notes: List[str]) -> None:
        metadata = self._assignment_metadata(target, rejection_notes)
        logger.error("[ASSIGN] %s | metadata=%s", message, metadata)
        raise LayoutAssignmentError(message, metadata=metadata)

    def _log_assignment_state(self, label: str) -> None:
        for island_id, island in sorted(self.islands.items()):
            real_ids = self._real_assigned_ids(island_id)
            assigned_area = self._room_area_sum(real_ids)
            fill = assigned_area / max(float(island.area), 1e-9)
            logger.debug(
                "[ASSIGN] State=%s | island=%s | area=%.2fm2 | assigned_area=%.2fm2 | "
                "fill_rate=%.2f | remaining=%.2fm2 | rooms=%s",
                label,
                island_id,
                float(island.area),
                assigned_area,
                fill,
                float(island.remaining_capacity),
                real_ids,
            )

    def _inject_dummy_rooms(self, degradation: DegradationSummary) -> None:
        cfg = self.config
        if not cfg.hole_fill_enabled:
            return
        if float(self._overall_utilization) < float(cfg.min_utilization):
            return

        for island_id, island in sorted(self.islands.items(), key=lambda kv: str(kv[0])):
            if island.area <= 1e-9:
                continue
            assigned_ids = self.assignments.get(island_id, [])
            if not assigned_ids:
                continue
            remaining = max(0.0, float(island.remaining_capacity))
            delta = remaining / float(island.area)
            if delta < float(cfg.hole_ratio_threshold):
                continue

            placed = float(island.area) - remaining
            target_total = float(island.area) * float(cfg.dummy_fill_utilization)
            rem = max(0.0, target_total - placed)

            allow_micro = False
            if rem < float(cfg.dummy_min_area):
                if rem <= 1e-6:
                    continue
                allow_micro = True
                areas = [rem]
            else:
                n = max(1, int(math.ceil(rem / float(cfg.dummy_max_area))))
                while n > 1 and (rem / n) < float(cfg.dummy_min_area):
                    n -= 1

                per = rem / n
                if per < float(cfg.dummy_min_area):
                    per = rem
                    n = 1

                areas = [per] * n
                areas[-1] = rem - per * (n - 1)
                if areas[-1] < float(cfg.dummy_min_area) and n > 1:
                    n -= 1
                    per = rem / n
                    areas = [per] * n
                    areas[-1] = rem - per * (n - 1)

            total_injected = 0.0
            injected_index = 0
            for a in areas:
                a = float(a)
                if (not allow_micro) and a < float(cfg.dummy_min_area):
                    continue
                seed = f"{island_id}_dummy_{injected_index}"
                rid = f"{cfg.dummy_id_prefix}{hashlib.md5(seed.encode('utf-8')).hexdigest()[:6]}"
                while rid in self.rooms:
                    injected_index += 1
                    seed = f"{island_id}_dummy_{injected_index}"
                    rid = f"{cfg.dummy_id_prefix}{hashlib.md5(seed.encode('utf-8')).hexdigest()[:6]}"
                dummy = RoomSpec(
                    room_id=rid,
                    room_type=str(cfg.dummy_room_type),
                    target_area=a,
                    min_width=0.1 if allow_micro else 1.0,
                    min_depth=0.1 if allow_micro else 1.0,
                    aspect_ratio_range=(0.2, 5.0),
                    zone=ZoneType.SERVICE,
                    needs_window=False,
                    needs_corridor_access=False,
                    is_dummy=True,
                    target_area_raw=a,
                )
                self.rooms[rid] = dummy
                self.assignments[island_id].append(rid)
                self.room_to_island[rid] = island_id
                island.assigned_rooms.append(rid)
                island.remaining_capacity = max(0.0, float(island.remaining_capacity) - a)
                total_injected += a
                injected_index += 1

            if total_injected > 0:
                msg = (
                    f"Injected dummy rooms: island={island_id}, "
                    f"delta={delta:.2f}, injected={total_injected:.1f}m2, parts={len(areas)}"
                )
                degradation.parse_warnings.append(msg)
                logger.warning(msg)

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
            if bool(getattr(room, "needs_corridor_access", True)) and (not list(getattr(island, "corridor_edges", []) or [])):
                continue

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
        """放宽约束的候选岛屿（不检查面积比，只要有剩余空间）"""
        candidates = []
        for island in self.islands.values():
            if island.remaining_capacity > 0:
                candidates.append(island)
        need_access = bool(getattr(room, "needs_corridor_access", True))
        if need_access:
            public_candidates = [
                i for i in candidates
                if self._is_public_access_island(i)
            ]
            if public_candidates:
                candidates = public_candidates
            candidates.sort(
                key=lambda i: (
                    bool(list(getattr(i, "corridor_edges", []) or [])),
                    float(i.remaining_capacity),
                ),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda i: i.remaining_capacity, reverse=True)
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
    topology_mode: str = "continuous_cpsat",
) -> Tuple[Dict[str, AssignmentResult], DegradationSummary]:
    """
    便捷函数：房间-岛屿分配

    Returns:
        (assignments, degradation_summary)
    """
    if adjacency_graph is None:
        adjacency_graph = {}
        for room in rooms:
            adjacency_graph[room.room_id] = (
                room.adjacency_required + room.adjacency_preferred
            )

    config = config or AssignerConfig()
    config.topology_mode = str(topology_mode or config.topology_mode)
    assigner = IslandRoomAssigner(islands, rooms, adjacency_graph, config)
    return assigner.assign()
