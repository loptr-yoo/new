"""
room_defaults.py

房间类型默认值 + size_hint → target_area 转换 + adjacency_forbidden 规则表。

LLM 不再输出 target_area / needs_window / adjacency_forbidden，
全部由本模块根据 room_type 和 size_hint 推导。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .room_spec import ROOM_TYPE_DEFAULTS

logger = logging.getLogger(__name__)


# ============================================================
# size_hint → target_area
# ============================================================

SIZE_HINT_WEIGHTS: Dict[str, float] = {
    "large": 3.0,
    "medium": 2.0,
    "small": 1.0,
}

# room_type 最小面积下限（m²）
_MIN_AREA_BY_TYPE: Dict[str, float] = {
    "living_room": 12.0,
    "living": 12.0,
    "bedroom": 9.0,
    "master_bedroom": 12.0,
    "kitchen": 6.0,
    "bathroom": 4.0,
    "toilet": 3.0,
    "study": 6.0,
    "dining_room": 8.0,
    "dining": 8.0,
    "storage": 3.0,
    "laundry": 3.0,
    "utility": 4.0,
    "corridor": 3.0,
    "hallway": 3.0,
    "entrance": 3.0,
    "reception": 10.0,
}

_DEFAULT_MIN_AREA = 6.0


def apply_size_hints(
    rooms: list,
    floor_area: float,
    core_tube_area: float,
) -> None:
    """
    将 size_hint 转为精确 target_area（原地修改）。

    分配逻辑：
    1. usable = floor_area - core_tube_area
    2. corridor 预留 12%
    3. 按 SIZE_HINT_WEIGHTS 比例分配剩余面积
    4. 叠加 room_type 最小面积约束
    """
    usable = floor_area - core_tube_area
    corridor_allowance = usable * 0.12
    distributable = usable - corridor_allowance

    if distributable <= 0:
        logger.warning("No distributable area (floor=%.1f, core=%.1f)", floor_area, core_tube_area)
        for room in rooms:
            room.target_area = _MIN_AREA_BY_TYPE.get(room.room_type, _DEFAULT_MIN_AREA)
        return

    total_weight = sum(SIZE_HINT_WEIGHTS.get(getattr(room, "size_hint", None) or "medium", 2.0) for room in rooms)
    if total_weight <= 0:
        total_weight = len(rooms) * 2.0

    for room in rooms:
        hint = getattr(room, "size_hint", None) or "medium"
        w = SIZE_HINT_WEIGHTS.get(hint, 2.0)
        room.target_area = distributable * (w / total_weight)
        min_area = _MIN_AREA_BY_TYPE.get(room.room_type, _DEFAULT_MIN_AREA)
        room.target_area = max(room.target_area, min_area)


# ============================================================
# needs_window 推导（从 room_type 查表）
# ============================================================

def infer_needs_window(room_type: str) -> bool:
    """根据 room_type 从 ROOM_TYPE_DEFAULTS 推导 needs_window。"""
    defaults = ROOM_TYPE_DEFAULTS.get(room_type)
    if defaults is not None:
        return defaults.get("needs_window", False)
    return False


# ============================================================
# adjacency_forbidden 规则表
# ============================================================

ADJACENCY_FORBIDDEN_RULES: Dict[str, List[str]] = {
    "kitchen": ["bedroom", "study"],  # 油烟噪音
    "bathroom": ["kitchen", "dining_room", "dining"],  # 卫生隔离
    "toilet": ["kitchen", "dining_room", "dining"],  # 卫生隔离
    "utility": ["living_room", "living", "bedroom"],  # 设备噪音
    "garage": ["bedroom", "study"],  # 尾气噪音
}


def compute_adjacency_forbidden(rooms: list) -> Dict[str, List[str]]:
    """
    根据 room_type 规则表计算每个房间的禁邻列表。

    Args:
        rooms: 有 room_id 和 room_type 属性的对象列表

    Returns:
        {room_id: [forbidden_room_ids]}
    """
    room_type_map = {r.room_id: r.room_type for r in rooms}
    forbidden: Dict[str, List[str]] = {}

    for room in rooms:
        rules = ADJACENCY_FORBIDDEN_RULES.get(room.room_type, [])
        forbidden[room.room_id] = [
            rid for rid, rtype in room_type_map.items()
            if rtype in rules and rid != room.room_id
        ]

    return forbidden
