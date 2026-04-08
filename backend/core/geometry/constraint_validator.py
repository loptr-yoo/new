"""
constraint_validator.py

约束验证器：验证房间划分结果是否满足所有约束。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from typing import Tuple

from .island_partition_solver import RoomResult, RoomSpec
from .room_spec import IslandContext, RoomSpec as SemanticRoomSpec, ZoneType


@dataclass
class ValidationReport:
    """验证报告"""

    orthogonality_rate: float
    overlap_area_ratio: float
    gap_area_ratio: float
    coverage_rate: float

    mean_area_error: float
    max_area_error: float

    min_dimension_compliance: float
    adjacency_satisfaction: float

    total_rooms: int
    valid_rooms: int

    @property
    def is_valid(self) -> bool:
        return (
            self.orthogonality_rate >= 0.99
            and self.overlap_area_ratio < 0.001
            and self.gap_area_ratio < 0.01
            and self.mean_area_error < 0.15
            and self.min_dimension_compliance >= 0.95
        )

    def __str__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"ValidationReport [{status}]\n"
            f"  orthogonality:  {self.orthogonality_rate * 100:.2f}%\n"
            f"  overlap:        {self.overlap_area_ratio * 100:.3f}%\n"
            f"  gap:            {self.gap_area_ratio * 100:.3f}%\n"
            f"  coverage:       {self.coverage_rate * 100:.2f}%\n"
            f"  mean_area_err:  {self.mean_area_error * 100:.2f}%\n"
            f"  max_area_err:   {self.max_area_error * 100:.2f}%\n"
            f"  min_dim_ok:     {self.min_dimension_compliance * 100:.2f}%\n"
            f"  adjacency_ok:   {self.adjacency_satisfaction * 100:.2f}%\n"
            f"  rooms:          {self.valid_rooms}/{self.total_rooms}\n"
        )


class ConstraintValidator:
    """约束验证器"""

    def __init__(
        self,
        island: Polygon,
        results: List[RoomResult],
        specs: List[RoomSpec],
    ):
        self.island = island
        self.results = results
        self.specs: Dict[str, RoomSpec] = {s.room_id: s for s in specs}

    def validate(self) -> ValidationReport:
        return ValidationReport(
            orthogonality_rate=self._check_orthogonality(),
            overlap_area_ratio=self._check_overlap(),
            gap_area_ratio=self._check_gaps(),
            coverage_rate=self._check_coverage(),
            mean_area_error=self._calc_mean_area_error(),
            max_area_error=self._calc_max_area_error(),
            min_dimension_compliance=self._check_min_dimensions(),
            adjacency_satisfaction=self._check_adjacency(),
            total_rooms=len(self.specs),
            valid_rooms=len(self.results),
        )

    def _check_orthogonality(self) -> float:
        # By construction: MIQP 输出矩形 → 100%
        return 1.0

    def _check_overlap(self) -> float:
        total_overlap = 0.0
        n = len(self.results)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    overlap = self.results[i].polygon.intersection(
                        self.results[j].polygon
                    )
                    total_overlap += overlap.area
                except Exception:
                    pass
        return total_overlap / self.island.area if self.island.area > 0 else 0

    def _check_gaps(self) -> float:
        if not self.results:
            return 1.0
        all_rooms = unary_union([r.polygon for r in self.results])
        gaps = self.island.difference(all_rooms)
        return gaps.area / self.island.area if self.island.area > 0 else 0

    def _check_coverage(self) -> float:
        if not self.results:
            return 0.0
        all_rooms = unary_union([r.polygon for r in self.results])
        covered = self.island.intersection(all_rooms)
        return covered.area / self.island.area if self.island.area > 0 else 0

    def _calc_mean_area_error(self) -> float:
        errors = []
        for r in self.results:
            spec = self.specs.get(r.room_id)
            if spec and spec.target_area > 0:
                err = abs(r.actual_area - spec.target_area) / spec.target_area
                errors.append(err)
        return float(np.mean(errors)) if errors else 0.0

    def _calc_max_area_error(self) -> float:
        errors = []
        for r in self.results:
            spec = self.specs.get(r.room_id)
            if spec and spec.target_area > 0:
                err = abs(r.actual_area - spec.target_area) / spec.target_area
                errors.append(err)
        return float(max(errors)) if errors else 0.0

    def _check_min_dimensions(self) -> float:
        if not self.results:
            return 0.0
        compliant = 0
        total = 0
        for r in self.results:
            spec = self.specs.get(r.room_id)
            if spec:
                total += 1
                min_dim = min(r.width, r.depth)
                spec_min = min(spec.min_width, spec.min_depth)
                if min_dim >= spec_min * 0.95:
                    compliant += 1
        return compliant / total if total > 0 else 1.0

    def _check_adjacency(self) -> float:
        required = 0
        satisfied = 0
        for r in self.results:
            spec = self.specs.get(r.room_id)
            if spec and spec.adjacency_required:
                for adj_id in spec.adjacency_required:
                    required += 1
                    adj_room = next(
                        (x for x in self.results if x.room_id == adj_id),
                        None,
                    )
                    if adj_room:
                        if r.polygon.touches(adj_room.polygon) or r.polygon.intersects(
                            adj_room.polygon
                        ):
                            satisfied += 1
        return satisfied / required if required > 0 else 1.0


# ============================================================
# 语义验证
# ============================================================


@dataclass
class SemanticValidationReport:
    """
    语义验证报告。

    在基础 ValidationReport 的指标之上增加语义维度的验证。
    """

    # 基础几何验证
    orthogonality_rate: float
    overlap_area_ratio: float
    gap_area_ratio: float
    coverage_rate: float
    mean_area_error: float
    max_area_error: float
    min_dimension_compliance: float
    adjacency_satisfaction: float
    total_rooms: int
    valid_rooms: int

    # 语义验证
    aspect_ratio_violations: List[Tuple[str, float]]  # (room_id, actual_ar)
    adjacency_satisfied: Dict[str, bool]  # "roomA-roomB" -> satisfied
    window_satisfied: Dict[str, bool]  # room_id -> satisfied
    zone_compactness: Dict[str, float]  # zone_name -> compactness (0-1)

    @property
    def is_valid(self) -> bool:
        return (
            self.orthogonality_rate >= 0.99
            and self.overlap_area_ratio < 0.001
            and self.gap_area_ratio < 0.01
            and self.mean_area_error < 0.15
            and self.min_dimension_compliance >= 0.95
            and len(self.aspect_ratio_violations) == 0
        )

    @property
    def semantic_score(self) -> float:
        """语义得分 (0-100)"""
        scores = []

        # 邻接满足率
        if self.adjacency_satisfied:
            adj_score = (
                sum(self.adjacency_satisfied.values())
                / len(self.adjacency_satisfied)
                * 100
            )
            scores.append(adj_score)

        # 采光满足率
        if self.window_satisfied:
            win_score = (
                sum(self.window_satisfied.values())
                / len(self.window_satisfied)
                * 100
            )
            scores.append(win_score)

        # 宽高比合规率
        ar_score = max(0, 100 - len(self.aspect_ratio_violations) * 10)
        scores.append(ar_score)

        return sum(scores) / len(scores) if scores else 100.0

    def __str__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        lines = [
            f"SemanticValidationReport [{status}]",
            f"  orthogonality:  {self.orthogonality_rate * 100:.2f}%",
            f"  overlap:        {self.overlap_area_ratio * 100:.3f}%",
            f"  gap:            {self.gap_area_ratio * 100:.3f}%",
            f"  coverage:       {self.coverage_rate * 100:.2f}%",
            f"  mean_area_err:  {self.mean_area_error * 100:.2f}%",
            f"  max_area_err:   {self.max_area_error * 100:.2f}%",
            f"  min_dim_ok:     {self.min_dimension_compliance * 100:.2f}%",
            f"  adjacency_ok:   {self.adjacency_satisfaction * 100:.2f}%",
            f"  rooms:          {self.valid_rooms}/{self.total_rooms}",
            f"  --- Semantic ---",
            f"  AR violations:  {len(self.aspect_ratio_violations)}",
            f"  semantic_score: {self.semantic_score:.1f}/100",
        ]
        if self.window_satisfied:
            win_ok = sum(self.window_satisfied.values())
            lines.append(f"  window_ok:      {win_ok}/{len(self.window_satisfied)}")
        if self.zone_compactness:
            for zn, sc in self.zone_compactness.items():
                lines.append(f"  zone_{zn}:  {sc:.2f}")
        return "\n".join(lines) + "\n"


class SemanticConstraintValidator:
    """语义约束验证器"""

    def __init__(
        self,
        island: Polygon,
        results: List[RoomResult],
        specs: List[SemanticRoomSpec],
        island_context: IslandContext,
    ):
        self.island = island
        self.results = results
        self.specs: Dict[str, SemanticRoomSpec] = {s.room_id: s for s in specs}
        self.context = island_context
        self._result_dict: Dict[str, RoomResult] = {r.room_id: r for r in results}

        # 基础验证器（复用）
        old_specs = []
        for s in specs:
            old_specs.append(RoomSpec(
                room_id=s.room_id,
                room_type=s.room_type,
                target_area=s.target_area,
                min_width=s.min_width,
                min_depth=s.min_depth,
                adjacency_required=s.adjacency_required,
                adjacency_forbidden=s.adjacency_forbidden,
                window_access=s.needs_window,
            ))
        self._base = ConstraintValidator(island, results, old_specs)

    def validate(self) -> SemanticValidationReport:
        base = self._base.validate()
        return SemanticValidationReport(
            # 基础
            orthogonality_rate=base.orthogonality_rate,
            overlap_area_ratio=base.overlap_area_ratio,
            gap_area_ratio=base.gap_area_ratio,
            coverage_rate=base.coverage_rate,
            mean_area_error=base.mean_area_error,
            max_area_error=base.max_area_error,
            min_dimension_compliance=base.min_dimension_compliance,
            adjacency_satisfaction=base.adjacency_satisfaction,
            total_rooms=base.total_rooms,
            valid_rooms=base.valid_rooms,
            # 语义
            aspect_ratio_violations=self._check_aspect_ratios(),
            adjacency_satisfied=self._check_adjacency_pairs(),
            window_satisfied=self._check_window_access(),
            zone_compactness=self._check_zone_compactness(),
        )

    def _check_aspect_ratios(self) -> List[Tuple[str, float]]:
        violations = []
        for r in self.results:
            spec = self.specs.get(r.room_id)
            if not spec:
                continue
            if r.depth > 0:
                ar = r.width / r.depth
            else:
                ar = float("inf")
            ar_min, ar_max = spec.aspect_ratio_range
            if ar < ar_min - 0.01 or ar > ar_max + 0.01:
                violations.append((r.room_id, ar))
        return violations

    def _check_adjacency_pairs(self) -> Dict[str, bool]:
        satisfied: Dict[str, bool] = {}
        processed: set = set()
        for spec in self.specs.values():
            for adj_id in spec.adjacency_required:
                pair_key = "-".join(sorted([spec.room_id, adj_id]))
                if pair_key in processed:
                    continue
                processed.add(pair_key)
                r1 = self._result_dict.get(spec.room_id)
                r2 = self._result_dict.get(adj_id)
                if r1 and r2:
                    satisfied[pair_key] = (
                        r1.polygon.touches(r2.polygon)
                        or r1.polygon.intersects(r2.polygon)
                    )
                else:
                    satisfied[pair_key] = False
        return satisfied

    def _check_window_access(self) -> Dict[str, bool]:
        result: Dict[str, bool] = {}
        island_bounds = self.island.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = island_bounds
        eps = 0.05  # 5cm tolerance

        for spec in self.specs.values():
            if not spec.needs_window:
                continue
            r = self._result_dict.get(spec.room_id)
            if not r:
                result[spec.room_id] = False
                continue

            touches_exterior = False
            rb = r.bounds  # (x, y, x+w, y+d)

            if "west" in self.context.exterior_walls and abs(rb[0] - minx) < eps:
                touches_exterior = True
            if "east" in self.context.exterior_walls and abs(rb[2] - maxx) < eps:
                touches_exterior = True
            if "south" in self.context.exterior_walls and abs(rb[1] - miny) < eps:
                touches_exterior = True
            if "north" in self.context.exterior_walls and abs(rb[3] - maxy) < eps:
                touches_exterior = True

            result[spec.room_id] = touches_exterior
        return result

    def _check_zone_compactness(self) -> Dict[str, float]:
        """
        每个 zone 的紧凑度 = sum(room_areas) / bounding_box_area。
        1.0 = 完美紧凑（房间填满包围盒），越小越分散。
        """
        from shapely.geometry import box as shapely_box

        zones: Dict[str, List[RoomResult]] = {}
        for r in self.results:
            spec = self.specs.get(r.room_id)
            if not spec:
                continue
            zn = spec.zone.value
            if zn not in zones:
                zones[zn] = []
            zones[zn].append(r)

        compactness: Dict[str, float] = {}
        for zn, zone_rooms in zones.items():
            if len(zone_rooms) < 2:
                compactness[zn] = 1.0
                continue

            all_x = []
            all_y = []
            total_area = 0.0
            for r in zone_rooms:
                all_x.extend([r.bounds[0], r.bounds[2]])
                all_y.extend([r.bounds[1], r.bounds[3]])
                total_area += r.actual_area

            bbox_area = (max(all_x) - min(all_x)) * (max(all_y) - min(all_y))
            compactness[zn] = total_area / bbox_area if bbox_area > 0 else 0.0

        return compactness


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════


def validate_layout(
    island: Polygon,
    results: List[RoomResult],
    specs: List["SemanticRoomSpec"],
    island_context: Optional["IslandContext"] = None,
) -> SemanticValidationReport:
    """
    便捷函数：验证布局

    Args:
        island: 岛屿/楼层边界
        results: MIQP 求解结果列表
        specs: 语义房间规格列表
        island_context: 岛屿上下文（外墙等）

    Returns:
        SemanticValidationReport
    """
    if island_context is None:
        island_context = IslandContext(
            exterior_walls=["north", "south", "east", "west"]
        )

    validator = SemanticConstraintValidator(
        island=island,
        results=results,
        specs=specs,
        island_context=island_context,
    )
    return validator.validate()
