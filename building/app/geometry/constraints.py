"""
constraints.py

布局约束检查器
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


@dataclass
class ConstraintCheckResult:
    """约束检查结果"""

    total_score: float
    area_score: float
    adjacency_score: float
    window_score: float
    width_score: float

    area_errors: Dict[str, float]
    adjacency_violations: List[Tuple[str, str, str]]
    window_violations: List[str]
    width_violations: List[str]

    is_acceptable: bool


def check_constraints(
    boundary: Polygon,
    room_specs: List,
    room_polygons: List[Polygon],
    area_tolerance: float = 0.1,
    min_facade: float = 1.5,
    min_shared_edge: float = 0.5,
) -> ConstraintCheckResult:
    """检查布局是否满足约束。

    Args:
        boundary: 边界多边形
        room_specs: 房间需求列表（需要有 id, target_area 等属性）
        room_polygons: 房间多边形列表
        area_tolerance: 面积误差容忍度
        min_facade: 最小外墙接触长度 (m)
        min_shared_edge: 最小邻接边长度 (m)

    Returns:
        ConstraintCheckResult
    """
    n = len(room_specs)
    exterior = boundary.exterior

    # 1. 面积检查
    area_errors: Dict[str, float] = {}
    area_violations = 0
    for i, (spec, poly) in enumerate(zip(room_specs, room_polygons)):
        actual = float(poly.area) if not poly.is_empty else 0.0
        target = float(spec.target_area)
        error = abs(actual - target) / target if target > 0 else 0.0
        area_errors[spec.id] = error
        if error > area_tolerance:
            area_violations += 1

    area_score = 1.0 - (area_violations / n) if n > 0 else 1.0

    # 2. 邻接检查
    adjacency_violations: List[Tuple[str, str, str]] = []
    id_to_idx = {s.id: i for i, s in enumerate(room_specs)}

    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if room_polygons[i].is_empty or room_polygons[j].is_empty:
                continue
            try:
                shared = room_polygons[i].boundary.intersection(room_polygons[j].boundary)
                length = float(shared.length) if not shared.is_empty else 0.0
                adj_matrix[i, j] = length
                adj_matrix[j, i] = length
            except Exception:
                pass

    for i, spec in enumerate(room_specs):
        for adj_id in getattr(spec, "adjacent_to", set()):
            j = id_to_idx.get(adj_id)
            if j is not None and adj_matrix[i, j] < min_shared_edge:
                adjacency_violations.append((spec.id, adj_id, "missing"))

        for not_adj_id in getattr(spec, "not_adjacent_to", set()):
            j = id_to_idx.get(not_adj_id)
            if j is not None and adj_matrix[i, j] > 0.1:
                adjacency_violations.append((spec.id, not_adj_id, "forbidden"))

    adjacency_violations = list(set(adjacency_violations))
    total_adj_constraints = sum(
        len(getattr(s, "adjacent_to", set())) + len(getattr(s, "not_adjacent_to", set()))
        for s in room_specs
    )
    adjacency_score = 1.0 - (len(adjacency_violations) / total_adj_constraints) if total_adj_constraints > 0 else 1.0

    # 3. 采光检查
    window_violations: List[str] = []
    window_required_count = 0
    for i, spec in enumerate(room_specs):
        if not getattr(spec, "requires_window", False):
            continue
        window_required_count += 1

        if room_polygons[i].is_empty:
            window_violations.append(spec.id)
            continue

        try:
            intersection = room_polygons[i].boundary.intersection(exterior)
            facade = float(intersection.length) if not intersection.is_empty else 0.0
        except Exception:
            facade = 0.0

        if facade < min_facade:
            window_violations.append(spec.id)

    window_score = 1.0 - (len(window_violations) / window_required_count) if window_required_count > 0 else 1.0

    # 4. 面宽检查
    width_violations: List[str] = []
    for i, spec in enumerate(room_specs):
        min_width = getattr(spec, "min_width", 2.0)
        if room_polygons[i].is_empty:
            width_violations.append(spec.id)
            continue
        try:
            shrunk = room_polygons[i].buffer(-min_width / 2.0)
            if shrunk.is_empty:
                width_violations.append(spec.id)
        except Exception:
            pass

    width_score = 1.0 - (len(width_violations) / n) if n > 0 else 1.0

    # 5. 总分 (面积 40%, 邻接 30%, 采光 20%, 面宽 10%)
    total_score = 0.4 * area_score + 0.3 * adjacency_score + 0.2 * window_score + 0.1 * width_score

    return ConstraintCheckResult(
        total_score=total_score,
        area_score=area_score,
        adjacency_score=adjacency_score,
        window_score=window_score,
        width_score=width_score,
        area_errors=area_errors,
        adjacency_violations=adjacency_violations,
        window_violations=window_violations,
        width_violations=width_violations,
        is_acceptable=total_score >= 0.8,
    )
