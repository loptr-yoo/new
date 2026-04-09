"""
postprocessor.py

后处理：根据房间 polygon 自动生成墙体、门、窗户。

借鉴 Co-Layout（AAAI 2026）思路：
求解器只负责房间分区，墙/门/窗由后处理启发式规则自动放置。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class WallSegment:
    """墙体段"""
    type: str  # "exterior_wall" | "partition_wall"
    geometry: BaseGeometry  # LineString / MultiLineString
    thickness: float  # 米
    room_ids: List[str]

    @property
    def length(self) -> float:
        return self.geometry.length if self.geometry and not self.geometry.is_empty else 0.0


@dataclass
class DoorPlacement:
    """门的放置"""
    position: Tuple[float, float]
    width: float  # 米
    connects: List[str]  # 连接的两个 room_id
    wall_type: str  # 所在墙的类型


@dataclass
class WindowPlacement:
    """窗户的放置"""
    position: Tuple[float, float]
    width: float  # 米
    room_id: str
    wall_length: float  # 所在墙段长度


@dataclass
class PostprocessResult:
    """后处理结果"""
    walls: List[WallSegment] = field(default_factory=list)
    doors: List[DoorPlacement] = field(default_factory=list)
    windows: List[WindowPlacement] = field(default_factory=list)


# ============================================================
# 墙体生成
# ============================================================

def generate_walls(
    rooms: list,
    floor_boundary: Polygon,
    exterior_thickness: float = 0.24,
    partition_thickness: float = 0.12,
    min_wall_length: float = 0.3,
) -> List[WallSegment]:
    """
    根据房间 polygon 共享边自动生成墙体。

    Args:
        rooms: 有 id/room_id, polygon 属性的 RoomResult 列表
        floor_boundary: 楼层外轮廓
        exterior_thickness: 外墙厚度 (m)
        partition_thickness: 隔墙厚度 (m)
        min_wall_length: 最小墙段长度 (m)

    Returns:
        WallSegment 列表
    """
    walls: List[WallSegment] = []

    # 外墙：房间 polygon 与楼层边界的共享边
    for room in rooms:
        room_id = getattr(room, "id", getattr(room, "room_id", "?"))
        poly = room.polygon
        if poly.is_empty:
            continue

        try:
            shared = poly.boundary.intersection(floor_boundary.boundary)
            if not shared.is_empty and shared.length > min_wall_length:
                walls.append(WallSegment(
                    type="exterior_wall",
                    geometry=shared,
                    thickness=exterior_thickness,
                    room_ids=[room_id],
                ))
        except Exception as e:
            logger.debug(f"Exterior wall calc failed for {room_id}: {e}")

    # 内墙：相邻房间 polygon 的共享边
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]
            if a.polygon.is_empty or b.polygon.is_empty:
                continue

            aid = getattr(a, "id", getattr(a, "room_id", "?"))
            bid = getattr(b, "id", getattr(b, "room_id", "?"))

            try:
                shared = a.polygon.boundary.intersection(b.polygon.boundary)
                if not shared.is_empty and shared.length > min_wall_length:
                    walls.append(WallSegment(
                        type="partition_wall",
                        geometry=shared,
                        thickness=partition_thickness,
                        room_ids=[aid, bid],
                    ))
            except Exception as e:
                logger.debug(f"Partition wall calc failed for {aid}-{bid}: {e}")

    return walls


# ============================================================
# 门的放置
# ============================================================

def generate_doors(
    walls: List[WallSegment],
    door_width: float = 0.9,
) -> List[DoorPlacement]:
    """
    在内墙上放置门。

    规则：每对相邻房间的共享内墙中点放一扇门。

    Args:
        walls: WallSegment 列表
        door_width: 门宽 (m)

    Returns:
        DoorPlacement 列表
    """
    doors: List[DoorPlacement] = []
    rooms_with_doors: set = set()

    for wall in walls:
        if wall.type != "partition_wall":
            continue
        if len(wall.room_ids) != 2:
            continue
        if wall.length < door_width:
            continue

        try:
            midpoint = wall.geometry.interpolate(0.5, normalized=True)
            doors.append(DoorPlacement(
                position=(round(midpoint.x, 2), round(midpoint.y, 2)),
                width=door_width,
                connects=list(wall.room_ids),
                wall_type=wall.type,
            ))
            rooms_with_doors.update(wall.room_ids)
        except Exception as e:
            logger.debug(f"Door placement failed: {e}")

    return doors


# ============================================================
# 窗户放置
# ============================================================

def generate_windows(
    walls: List[WallSegment],
    rooms: list,
    window_width: float = 1.2,
    window_spacing: float = 2.0,
) -> List[WindowPlacement]:
    """
    在外墙上为 needs_window=True 的房间放置窗户。

    规则：沿外墙每 window_spacing 米放一个窗，至少一个。

    Args:
        walls: WallSegment 列表
        rooms: 有 id/room_id, has_window/needs_window 属性的 RoomResult 列表
        window_width: 窗宽 (m)
        window_spacing: 窗间距 (m)

    Returns:
        WindowPlacement 列表
    """
    # 构建需要窗户的房间集合
    window_rooms: set = set()
    for room in rooms:
        room_id = getattr(room, "id", getattr(room, "room_id", "?"))
        has_window = getattr(room, "has_window", False) or getattr(room, "needs_window", False)
        if has_window:
            window_rooms.add(room_id)

    windows: List[WindowPlacement] = []

    for wall in walls:
        if wall.type != "exterior_wall":
            continue
        if len(wall.room_ids) != 1:
            continue

        room_id = wall.room_ids[0]
        if room_id not in window_rooms:
            continue

        wall_length = wall.length
        if wall_length < window_width:
            continue

        num_windows = max(1, int(wall_length / window_spacing))

        for k in range(num_windows):
            try:
                pos = wall.geometry.interpolate((k + 0.5) / num_windows, normalized=True)
                windows.append(WindowPlacement(
                    position=(round(pos.x, 2), round(pos.y, 2)),
                    width=window_width,
                    room_id=room_id,
                    wall_length=round(wall_length, 2),
                ))
            except Exception as e:
                logger.debug(f"Window placement failed: {e}")

    return windows


# ============================================================
# 一站式后处理
# ============================================================

def postprocess_floor(
    rooms: list,
    floor_boundary: Polygon,
) -> PostprocessResult:
    """
    对单层布局执行完整后处理。

    Args:
        rooms: RoomResult 列表
        floor_boundary: 楼层外轮廓

    Returns:
        PostprocessResult
    """
    walls = generate_walls(rooms, floor_boundary)
    doors = generate_doors(walls)
    windows = generate_windows(walls, rooms)

    return PostprocessResult(walls=walls, doors=doors, windows=windows)


# ============================================================
# 序列化辅助
# ============================================================

def wall_to_dict(wall: WallSegment) -> dict:
    """WallSegment → 可序列化 dict"""
    coords = []
    if isinstance(wall.geometry, LineString):
        coords = [[round(x, 2), round(y, 2)] for x, y in wall.geometry.coords]
    elif isinstance(wall.geometry, MultiLineString):
        for line in wall.geometry.geoms:
            coords.extend([[round(x, 2), round(y, 2)] for x, y in line.coords])

    return {
        "type": wall.type,
        "coords": coords,
        "thickness": wall.thickness,
        "length": round(wall.length, 2),
        "room_ids": wall.room_ids,
    }


def door_to_dict(door: DoorPlacement) -> dict:
    """DoorPlacement → 可序列化 dict"""
    return {
        "position": list(door.position),
        "width": door.width,
        "connects": door.connects,
    }


def window_to_dict(window: WindowPlacement) -> dict:
    """WindowPlacement → 可序列化 dict"""
    return {
        "position": list(window.position),
        "width": window.width,
        "room_id": window.room_id,
    }
