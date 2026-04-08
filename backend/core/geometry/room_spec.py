"""
room_spec.py

语义增强的房间规格与求解器配置。

包含：
- ZoneType: 功能分区枚举
- RoomSpec: 语义增强版房间规格（旧版超集）
- IslandContext: 岛屿上下文（外墙、入口等）
- SolverConfig: 求解器配置（权重、容差、参数）
- WarmStartRect: Treemap warm start 输出的统一数据类
- ROOM_TYPE_DEFAULTS: 房间类型默认映射表（LLM 未输出时 fallback）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ZoneType(Enum):
    """功能分区类型"""

    PUBLIC = "public"  # 公共区：客厅、餐厅、厨房
    PRIVATE = "private"  # 私密区：卧室、书房
    SERVICE = "service"  # 服务区：储藏、设备间
    CIRCULATION = "circulation"  # 交通区：走廊、玄关


@dataclass
class RoomSpec:
    """
    房间规格（语义增强版）

    向后兼容：所有新字段有默认值，旧代码
    ``RoomSpec(room_id=..., room_type=..., target_area=...)`` 仍可使用。

    基础属性（与旧版一致）:
        room_id: 房间唯一标识
        room_type: 房间类型字符串
        target_area: 目标面积 (m²)

    几何约束:
        min_width: 最小宽度 (m)
        min_depth: 最小进深 (m)
        aspect_ratio_range: 宽高比范围 (min, max)

    语义约束:
        zone: 功能分区
        needs_window: 是否需要采光（外墙）
        adjacency_required: 必须相邻的房间列表（硬约束）
        adjacency_preferred: 最好相邻的房间列表（软约束）
        adjacency_forbidden: 禁止相邻的房间列表

    权重:
        area_priority: 面积满足优先级（用于目标函数加权）
    """

    # 基础属性
    room_id: str
    room_type: str
    target_area: float  # m²

    # 几何约束
    min_width: float = 2.5  # m
    min_depth: float = 2.5  # m
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)

    # 语义约束
    zone: ZoneType = ZoneType.PUBLIC
    needs_window: bool = False
    adjacency_required: List[str] = field(default_factory=list)
    adjacency_preferred: List[str] = field(default_factory=list)
    adjacency_forbidden: List[str] = field(default_factory=list)

    # 权重
    area_priority: float = 1.0


@dataclass
class IslandContext:
    """
    岛屿上下文信息

    用于确定采光约束（哪些方向有外墙）和公共区偏好。

    Attributes:
        exterior_walls: 外墙方向列表，如 ['north', 'south', 'east', 'west']
        entrance_direction: 入口方向（可选）
        preferred_public_side: 公共区偏好侧（可选）
    """

    exterior_walls: List[str] = field(default_factory=lambda: ["north", "south", "east", "west"])
    entrance_direction: Optional[str] = None
    preferred_public_side: Optional[str] = None


@dataclass
class SolverConfig:
    """
    求解器配置

    权重调优指南：
    - weight_area (100): 面积是最重要的，通常不需要调
    - weight_adjacency (50): 如果邻接效果差，增大到 70-100
    - weight_compactness (30): 如果分区太分散，增大到 50
    - weight_aspect_ratio (20): 如果出现细长房间，增大到 40
    - weight_alignment (10): 边界对齐奖励，通常不调

    调优方法：
    1. 先跑默认权重，检查 SemanticValidationReport
    2. 找到最不满意的指标对应权重
    3. 增大该权重 20-50%，重新求解
    4. 重复直到满意
    """

    # 求解参数
    time_limit: float = 30.0  # 秒
    num_workers: int = 8

    # 容差
    area_tolerance: float = 0.15  # 面积容差 ±15%

    # 目标函数权重
    weight_area: int = 100  # 面积偏差权重
    weight_adjacency: int = 50  # 邻接距离权重
    weight_compactness: int = 30  # 分区紧凑度权重
    weight_aspect_ratio: int = 20  # 宽高比惩罚权重
    weight_alignment: int = 10  # 边界对齐奖励权重

    # 约束参数
    min_door_width: float = 0.9  # 最小门宽 (m)，用于邻接共享边判定
    adjacency_max_gap: float = 0.0  # 邻接最大间隙（0 = 必须接触）
    non_adjacency_min_gap: float = 0.1  # 禁止邻接最小间隙 (m)


@dataclass
class WarmStartRect:
    """
    Treemap warm start 输出的统一数据类

    字段名与 solver 变量一致（width/depth 而非 w/h），
    避免 treemap 输出与 solver hint 输入的字段不匹配。
    """

    room_id: str
    x: float
    y: float
    width: float
    depth: float


# ============================================================
# 房间类型默认映射表
# ============================================================

ROOM_TYPE_DEFAULTS: Dict[str, Dict] = {
    # 公共区
    "living_room": {"zone": ZoneType.PUBLIC, "needs_window": True, "ar": (0.6, 1.8)},
    "living": {"zone": ZoneType.PUBLIC, "needs_window": True, "ar": (0.6, 1.8)},
    "dining_room": {"zone": ZoneType.PUBLIC, "needs_window": False, "ar": (0.6, 1.8)},
    "dining": {"zone": ZoneType.PUBLIC, "needs_window": False, "ar": (0.6, 1.8)},
    "kitchen": {"zone": ZoneType.PUBLIC, "needs_window": True, "ar": (0.5, 2.0)},
    "reception": {"zone": ZoneType.PUBLIC, "needs_window": False, "ar": (0.6, 1.8)},
    # 私密区
    "bedroom": {"zone": ZoneType.PRIVATE, "needs_window": True, "ar": (0.6, 1.6)},
    "master_bedroom": {"zone": ZoneType.PRIVATE, "needs_window": True, "ar": (0.6, 1.6)},
    "study": {"zone": ZoneType.PRIVATE, "needs_window": True, "ar": (0.6, 1.8)},
    "bathroom": {"zone": ZoneType.PRIVATE, "needs_window": False, "ar": (0.5, 2.0)},
    "toilet": {"zone": ZoneType.PRIVATE, "needs_window": False, "ar": (0.5, 2.0)},
    # 服务区
    "storage": {"zone": ZoneType.SERVICE, "needs_window": False, "ar": (0.5, 2.0)},
    "laundry": {"zone": ZoneType.SERVICE, "needs_window": False, "ar": (0.5, 2.0)},
    "utility": {"zone": ZoneType.SERVICE, "needs_window": False, "ar": (0.5, 2.0)},
    # 交通区
    "corridor": {"zone": ZoneType.CIRCULATION, "needs_window": False, "ar": (0.2, 5.0)},
    "hallway": {"zone": ZoneType.CIRCULATION, "needs_window": False, "ar": (0.2, 5.0)},
    "passage": {"zone": ZoneType.CIRCULATION, "needs_window": False, "ar": (0.2, 5.0)},
    "entrance": {"zone": ZoneType.CIRCULATION, "needs_window": False, "ar": (0.5, 2.0)},
}


def apply_room_type_defaults(spec: RoomSpec) -> RoomSpec:
    """
    根据 room_type 填充未显式指定的语义字段。

    仅当字段处于默认值时才覆盖，已由 LLM 或用户显式设置的字段不受影响。
    返回同一个对象（原地修改）。
    """
    defaults = ROOM_TYPE_DEFAULTS.get(spec.room_type)
    if defaults is None:
        return spec

    # zone: 仅当仍为默认 PUBLIC 且 room_type 有不同默认时覆盖
    if spec.zone == ZoneType.PUBLIC and defaults["zone"] != ZoneType.PUBLIC:
        spec.zone = defaults["zone"]

    # needs_window: 仅当为 False（默认）且 room_type 默认为 True 时覆盖
    if not spec.needs_window and defaults["needs_window"]:
        spec.needs_window = True

    # aspect_ratio_range: 仅当为默认 (0.5, 2.0) 时覆盖
    if spec.aspect_ratio_range == (0.5, 2.0) and defaults["ar"] != (0.5, 2.0):
        spec.aspect_ratio_range = defaults["ar"]

    return spec
