from __future__ import annotations

import math
from typing import List, Tuple

from ..models import LayoutElement, ParkingLayout
from ..types import ElementType


def _get_corners(el: LayoutElement) -> List[Tuple[float, float]]:
    rad = ((el.rotation or 0) * math.pi) / 180
    cx = el.x + el.width / 2
    cy = el.y + el.height / 2
    pts = [
        (el.x, el.y),
        (el.x + el.width, el.y),
        (el.x + el.width, el.y + el.height),
        (el.x, el.y + el.height),
    ]
    corners: List[Tuple[float, float]] = []
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    for (px, py) in pts:
        dx = px - cx
        dy = py - cy
        corners.append((cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r))
    return corners


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _project_point_on_segment(p: Tuple[float, float], l1: Tuple[float, float], l2: Tuple[float, float]) -> Tuple[float, float]:
    ax = p[0] - l1[0]
    ay = p[1] - l1[1]
    cx = l2[0] - l1[0]
    cy = l2[1] - l1[1]
    dot = ax * cx + ay * cy
    len_sq = cx * cx + cy * cy
    param = -1.0
    if len_sq != 0:
        param = dot / len_sq
    if param < 0:
        return l1
    if param > 1:
        return l2
    return (l1[0] + param * cx, l1[1] + param * cy)


def enforce_orthogonal_walls(layout: ParkingLayout) -> ParkingLayout:
    elements: List[LayoutElement] = []
    for el in layout.elements:
        if el.type not in (ElementType.WALL_EXTERNAL, ElementType.WALL_INTERNAL):
            elements.append(el)
            continue
        rotation = el.rotation or 0
        rotation = (rotation % 360 + 360) % 360
        if rotation < 45 or rotation >= 315:
            rotation = 0
        elif rotation >= 45 and rotation < 135:
            rotation = 90
        elif rotation >= 135 and rotation < 225:
            rotation = 180
        else:
            rotation = 270
        x = int(round(el.x))
        y = int(round(el.y))
        w = int(round(el.width))
        h = int(round(el.height))
        elements.append(el.model_copy(update={"rotation": rotation, "x": x, "y": y, "width": w, "height": h}))
    return ParkingLayout(width=layout.width, height=layout.height, elements=elements, sceneId=layout.sceneId)


def close_exterior_boundary(layout: ParkingLayout) -> ParkingLayout:
    walls = [e for e in layout.elements if e.type == ElementType.WALL_EXTERNAL]
    if len(walls) < 2:
        return layout
    return layout


def snap_doors_to_walls(layout: ParkingLayout) -> ParkingLayout:
    walls = [
        e
        for e in layout.elements
        if e.type in (ElementType.WALL_EXTERNAL, ElementType.WALL_INTERNAL, ElementType.SHEAR_WALL)
    ]
    if not walls:
        return layout
    elements: List[LayoutElement] = []
    for el in layout.elements:
        if el.type not in (ElementType.DOOR, ElementType.WINDOW, ElementType.FIRE_DOOR):
            elements.append(el)
            continue
        cx = el.x + el.width / 2
        cy = el.y + el.height / 2
        center = (cx, cy)
        best_wall = None
        min_dist = None
        best_proj = center
        for wall in walls:
            corners = _get_corners(wall)
            m1 = ((corners[0][0] + corners[3][0]) / 2, (corners[0][1] + corners[3][1]) / 2)
            m2 = ((corners[1][0] + corners[2][0]) / 2, (corners[1][1] + corners[2][1]) / 2)
            proj = _project_point_on_segment(center, m1, m2)
            d = _distance(center, proj)
            if min_dist is None or d < min_dist:
                min_dist = d
                best_wall = wall
                best_proj = proj
        if best_wall is not None and min_dist is not None and min_dist < 50:
            new_x = best_proj[0] - el.width / 2
            new_y = best_proj[1] - el.height / 2
            new_rot = best_wall.rotation or 0
            elements.append(el.model_copy(update={"x": new_x, "y": new_y, "rotation": new_rot}))
        else:
            elements.append(el)
    return ParkingLayout(width=layout.width, height=layout.height, elements=elements, sceneId=layout.sceneId)

