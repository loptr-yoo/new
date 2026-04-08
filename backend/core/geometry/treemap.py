"""
treemap.py

Squarified Treemap 算法 + 分层 Treemap（语义感知 warm start）。

包含：
- 底层 squarify 算法（从 island_partition_solver.py 提取，纯 Python 无外部依赖）
- hierarchical_treemap: 按邻接聚类 → 按功能分区排序 → 两级 squarify
- flat_squarify: 与旧 solver 兼容的扁平 squarify 入口
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from .room_spec import RoomSpec, WarmStartRect, ZoneType


# ============================================================
# Squarify 核心算法（纯 Python，无外部依赖）
# ============================================================


def _worst_ratio(row: List[float], side: float) -> float:
    """计算一行中最差的宽高比"""
    if not row or side <= 0:
        return float("inf")
    s = sum(row)
    if s <= 0:
        return float("inf")
    max_r = max(row)
    min_r = min(row)
    return max(side * side * max_r / (s * s), s * s / (side * side * min_r))


def _layout_row(
    sizes: List[float], length: float, breadth: float, total_area: float
) -> Tuple[List[float], List[float]]:
    """贪心选择一行：加入元素直到宽高比变差"""
    if not sizes:
        return [], []
    row = [sizes[0]]
    rest = sizes[1:]
    side = breadth

    current_worst = _worst_ratio(row, side)
    for s in rest:
        candidate = row + [s]
        new_worst = _worst_ratio(candidate, side)
        if new_worst <= current_worst:
            row = candidate
            rest = rest[1:]
            current_worst = new_worst
        else:
            break

    return row, rest


def _do_layout_row(
    row: List[float],
    cx: float,
    cy: float,
    cw: float,
    ch: float,
) -> List[Tuple[float, float, float, float]]:
    """水平方向切割：沿 x 方向占据一个 strip，内部沿 y 排列"""
    s = sum(row)
    strip_w = s / ch if ch > 0 else cw
    rects = []
    ry = cy
    for area in row:
        rh = area / strip_w if strip_w > 0 else ch
        rects.append((cx, ry, strip_w, rh))
        ry += rh
    return rects


def _do_layout_row_v(
    row: List[float],
    cx: float,
    cy: float,
    cw: float,
    ch: float,
) -> List[Tuple[float, float, float, float]]:
    """垂直方向切割：沿 y 方向占据一个 strip，内部沿 x 排列"""
    s = sum(row)
    strip_h = s / cw if cw > 0 else ch
    rects = []
    rx = cx
    for area in row:
        rw = area / strip_h if strip_h > 0 else cw
        rects.append((rx, cy, rw, strip_h))
        rx += rw
    return rects


def squarify(
    sizes: List[float], x: float, y: float, w: float, h: float
) -> List[Tuple[float, float, float, float]]:
    """
    Squarified treemap 纯 Python 实现。

    Args:
        sizes: 面积列表（已归一化到总面积 = w*h）
        x, y: 左下角坐标
        w, h: 宽度和高度

    Returns:
        [(x, y, width, height), ...] 列表，顺序与输入 sizes 对应
    """
    if not sizes:
        return []
    if len(sizes) == 1:
        return [(x, y, w, h)]

    rects: List[Tuple[float, float, float, float]] = []
    remaining = list(sizes)
    cx, cy, cw, ch = x, y, w, h

    while remaining:
        if cw < 1e-9 or ch < 1e-9:
            break
        if cw >= ch:
            row, remaining = _layout_row(remaining, cw, ch, sum(remaining))
            rects.extend(_do_layout_row(row, cx, cy, cw, ch))
            strip_w = sum(row) / ch if ch > 0 else 0
            cx += strip_w
            cw -= strip_w
        else:
            row, remaining = _layout_row(remaining, ch, cw, sum(remaining))
            rects.extend(_do_layout_row_v(row, cx, cy, cw, ch))
            strip_h = sum(row) / cw if cw > 0 else 0
            cy += strip_h
            ch -= strip_h

    return rects


# ============================================================
# 邻接聚类
# ============================================================


def cluster_rooms_by_adjacency(
    rooms: List[RoomSpec],
    adjacency_graph: Dict[str, List[str]],
) -> List[List[RoomSpec]]:
    """
    基于邻接关系的房间聚类（Union-Find）。

    规则：
    1. adjacency_required 中的房间必须在同一组
    2. adjacency_graph 中的邻接对合并到同一组

    Args:
        rooms: 房间列表
        adjacency_graph: 邻接关系图 {room_id: [neighbor_ids]}

    Returns:
        分组后的房间列表，每组为一个 List[RoomSpec]
    """
    room_dict = {r.room_id: r for r in rooms}
    room_ids = set(room_dict.keys())

    # Union-Find
    parent: Dict[str, str] = {rid: rid for rid in room_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # 根据 adjacency_required 合并
    for room in rooms:
        for adj_id in room.adjacency_required:
            if adj_id in room_ids:
                union(room.room_id, adj_id)

    # 根据外部 adjacency_graph 合并
    for room_id, neighbors in adjacency_graph.items():
        if room_id not in room_ids:
            continue
        for neighbor_id in neighbors:
            if neighbor_id in room_ids:
                union(room_id, neighbor_id)

    # 收集组
    groups_dict: Dict[str, List[RoomSpec]] = defaultdict(list)
    for room in rooms:
        root = find(room.room_id)
        groups_dict[root].append(room)

    return list(groups_dict.values())


# ============================================================
# 功能分区排序
# ============================================================

_ZONE_PRIORITY = {
    ZoneType.PUBLIC: 0,
    ZoneType.PRIVATE: 1,
    ZoneType.SERVICE: 2,
    ZoneType.CIRCULATION: 3,
}


def sort_groups_by_zone(groups: List[List[RoomSpec]]) -> List[List[RoomSpec]]:
    """
    按功能分区排序组。

    顺序：PUBLIC > PRIVATE > SERVICE > CIRCULATION
    组的优先级取组内最高优先级的 zone。
    """

    def group_priority(group: List[RoomSpec]) -> int:
        return min(_ZONE_PRIORITY.get(r.zone, 99) for r in group)

    return sorted(groups, key=group_priority)


# ============================================================
# 分层 Treemap
# ============================================================


def hierarchical_treemap(
    island_bounds: Tuple[float, float, float, float],
    rooms: List[RoomSpec],
    adjacency_graph: Dict[str, List[str]],
) -> List[WarmStartRect]:
    """
    分层 Treemap：先按邻接聚类，再按功能分区排序，再两级 squarify。

    比扁平 squarify 产生更好的 warm start：
    - 同组房间在空间上相邻
    - 公共区靠前（通常靠入口侧）

    Args:
        island_bounds: (x_min, y_min, x_max, y_max)
        rooms: 房间规格列表
        adjacency_graph: 邻接关系图

    Returns:
        WarmStartRect 列表，每个房间一个
    """
    if not rooms:
        return []

    x_min, y_min, x_max, y_max = island_bounds
    W = x_max - x_min
    H = y_max - y_min
    total_area = W * H

    if total_area <= 0:
        return []

    # 单房间直接返回
    if len(rooms) == 1:
        return [WarmStartRect(
            room_id=rooms[0].room_id,
            x=x_min, y=y_min, width=W, depth=H,
        )]

    # Step 1: 按邻接关系聚类
    groups = cluster_rooms_by_adjacency(rooms, adjacency_graph)

    # Step 2: 按功能分区排序
    groups = sort_groups_by_zone(groups)

    # Step 3: 组级 Treemap
    group_areas = [sum(r.target_area for r in g) for g in groups]
    total_room_area = sum(group_areas)
    if total_room_area <= 0:
        total_room_area = 1.0  # 防除零

    # 归一化到岛屿面积
    normalized_group_areas = [a / total_room_area * total_area for a in group_areas]

    group_rects = squarify(normalized_group_areas, 0, 0, W, H)

    # Step 4: 组内 Treemap
    results: List[WarmStartRect] = []

    for group, g_rect in zip(groups, group_rects):
        gx, gy, gw, gh = g_rect

        if len(group) == 1:
            # 单房间组直接用组的矩形
            results.append(WarmStartRect(
                room_id=group[0].room_id,
                x=gx + x_min,
                y=gy + y_min,
                width=gw,
                depth=gh,
            ))
            continue

        g_area = gw * gh
        if g_area <= 0:
            # 退化情况：面积为 0，给每个房间一个空矩形
            for room in group:
                results.append(WarmStartRect(
                    room_id=room.room_id,
                    x=gx + x_min, y=gy + y_min,
                    width=max(gw, 0.1), depth=max(gh, 0.1),
                ))
            continue

        # 组内按面积排序（大房间优先）
        sorted_rooms = sorted(group, key=lambda r: r.target_area, reverse=True)
        room_areas = [r.target_area for r in sorted_rooms]
        total_group_room_area = sum(room_areas)
        if total_group_room_area <= 0:
            total_group_room_area = 1.0

        normalized_room_areas = [a / total_group_room_area * g_area for a in room_areas]

        room_rects = squarify(normalized_room_areas, gx, gy, gw, gh)

        for room, r_rect in zip(sorted_rooms, room_rects):
            rx, ry, rw, rh = r_rect
            results.append(WarmStartRect(
                room_id=room.room_id,
                x=rx + x_min,
                y=ry + y_min,
                width=rw,
                depth=rh,
            ))

    return results


# ============================================================
# 兼容旧 solver 的扁平 squarify 入口
# ============================================================


def flat_squarify(
    sizes: List[float], x: float, y: float, w: float, h: float
) -> List[Tuple[float, float, float, float]]:
    """与旧 solver ``_squarify`` 完全兼容的入口函数。"""
    return squarify(sizes, x, y, w, h)
