from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from ..models import ConstraintViolation, LayoutElement, ParkingLayout
from ..types import ElementType


class SpatialGrid:
    def __init__(self, cell_size: int = 100):
        self._grid: Dict[str, List[LayoutElement]] = {}
        self._cell_size = cell_size

    def _keys_for_element(self, el: LayoutElement) -> List[str]:
        start_x = math.floor(el.x / self._cell_size)
        end_x = math.floor((el.x + el.width) / self._cell_size)
        start_y = math.floor(el.y / self._cell_size)
        end_y = math.floor((el.y + el.height) / self._cell_size)
        keys: List[str] = []
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                keys.append(f"{x},{y}")
        return keys

    def add(self, el: LayoutElement) -> None:
        for key in self._keys_for_element(el):
            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(el)

    def get_potential_collisions(self, el: LayoutElement) -> List[LayoutElement]:
        candidates: Dict[str, LayoutElement] = {}
        for key in self._keys_for_element(el):
            cell = self._grid.get(key)
            if not cell:
                continue
            for cand in cell:
                if cand.id != el.id:
                    candidates[cand.id] = cand
        return list(candidates.values())


EPSILON = 1.5
_corner_cache: Dict[str, List[Tuple[float, float]]] = {}


def _get_corners(el: LayoutElement) -> List[Tuple[float, float]]:
    key = f"{el.id}_{el.x}_{el.y}_{el.width}_{el.height}_{el.rotation or 0}"
    cached = _corner_cache.get(key)
    if cached is not None:
        return cached
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
    if len(_corner_cache) > 2000:
        _corner_cache.clear()
    _corner_cache[key] = corners
    return corners


def _polygons_intersect(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False
    polygons = [a, b]
    for polygon in polygons:
        for j in range(len(polygon)):
            k = (j + 1) % len(polygon)
            nx = polygon[k][1] - polygon[j][1]
            ny = polygon[j][0] - polygon[k][0]
            if nx == 0 and ny == 0:
                continue
            min_a = float("inf")
            max_a = float("-inf")
            for (ax, ay) in a:
                proj = nx * ax + ny * ay
                min_a = min(min_a, proj)
                max_a = max(max_a, proj)
            min_b = float("inf")
            max_b = float("-inf")
            for (bx, by) in b:
                proj = nx * bx + ny * by
                min_b = min(min_b, proj)
                max_b = max(max_b, proj)
            if max_a <= min_b + 0.1 or max_b <= min_a + 0.1:
                return False
    return True


def get_intersection_box(r1: LayoutElement, r2: LayoutElement) -> Optional[Dict[str, float]]:
    x1 = max(r1.x, r2.x)
    y1 = max(r1.y, r2.y)
    x2 = min(r1.x + r1.width, r2.x + r2.width)
    y2 = min(r1.y + r1.height, r2.y + r2.height)
    w = x2 - x1
    h = y2 - y1
    if w > 0 and h > 0:
        return {"x": x1, "y": y1, "width": w, "height": h}
    return None


def _is_overlapping(road: LayoutElement, item: LayoutElement, pad: float = 0) -> bool:
    r_rect = {"l": road.x - pad, "r": road.x + road.width + pad, "t": road.y - pad, "b": road.y + road.height + pad}
    i_rect = {"l": item.x, "r": item.x + item.width, "t": item.y, "b": item.y + item.height}
    overlap_x = min(r_rect["r"], i_rect["r"]) - max(r_rect["l"], i_rect["l"])
    overlap_y = min(r_rect["b"], i_rect["b"]) - max(r_rect["t"], i_rect["t"])
    if overlap_x < EPSILON or overlap_y < EPSILON:
        return False
    return _polygons_intersect(_get_corners(road), _get_corners(item))


def _is_touching(a: LayoutElement, b: LayoutElement) -> bool:
    if _polygons_intersect(_get_corners(a), _get_corners(b)):
        return True
    margin = 2.0
    a_rect = {"l": a.x - margin, "r": a.x + a.width + margin, "t": a.y - margin, "b": a.y + a.height + margin}
    b_rect = {"l": b.x - margin, "r": b.x + b.width + margin, "t": b.y - margin, "b": b.y + b.height + margin}
    overlap_x = min(a_rect["r"], b_rect["r"]) - max(a_rect["l"], b_rect["l"])
    overlap_y = min(a_rect["b"], b_rect["b"]) - max(a_rect["t"], b_rect["t"])
    return overlap_x >= 0 and overlap_y >= 0


def _is_point_covered(x: float, y: float, elements: List[LayoutElement], padding: float = 10) -> bool:
    for el in elements:
        if x >= el.x - padding and x <= el.x + el.width + padding and y >= el.y - padding and y <= el.y + el.height + padding:
            return True
    return False


def validate_layout(layout: ParkingLayout) -> List[ConstraintViolation]:
    violations: List[ConstraintViolation] = []
    if not layout or not layout.elements:
        return violations

    grid = SpatialGrid(100)
    for el in layout.elements:
        grid.add(el)

    for el in layout.elements:
        if (
            el.x < -EPSILON
            or el.x + el.width > layout.width + EPSILON
            or el.y < -EPSILON
            or el.y + el.height > layout.height + EPSILON
        ):
            violations.append(
                ConstraintViolation(
                    elementId=el.id,
                    type="out_of_bounds",
                    message="Element is outside boundary.",
                )
            )

    solid_types: Set[str] = {
        ElementType.PARKING_SPACE,
        ElementType.PILLAR,
        ElementType.WALL,
        ElementType.STAIRCASE,
        ElementType.ELEVATOR,
        ElementType.ROAD,
        ElementType.RAMP,
        ElementType.CHARGING_STATION,
        ElementType.ENTRANCE,
        ElementType.EXIT,
        ElementType.FIRE_EXTINGUISHER,
        ElementType.GROUND,
    }
    solids = [e for e in layout.elements if e.type in solid_types]

    allowed_overlaps: Set[str] = {
        f"{ElementType.CHARGING_STATION}:{ElementType.PARKING_SPACE}",
        f"{ElementType.FIRE_EXTINGUISHER}:{ElementType.PARKING_SPACE}",
        f"{ElementType.GROUND}:{ElementType.PARKING_SPACE}",
        f"{ElementType.CHARGING_STATION}:{ElementType.GROUND}",
        f"{ElementType.FIRE_EXTINGUISHER}:{ElementType.GROUND}",
        f"{ElementType.FIRE_EXTINGUISHER}:{ElementType.WALL}",
        f"{ElementType.STAIRCASE}:{ElementType.GROUND}",
        f"{ElementType.ELEVATOR}:{ElementType.GROUND}",
        f"{ElementType.SAFE_EXIT}:{ElementType.GROUND}",
        f"{ElementType.PILLAR}:{ElementType.GROUND}",
        f"{ElementType.GROUND}:{ElementType.ROAD}",
        f"{ElementType.RAMP}:{ElementType.ROAD}",
        f"{ElementType.ENTRANCE}:{ElementType.RAMP}",
        f"{ElementType.EXIT}:{ElementType.RAMP}",
        f"{ElementType.ENTRANCE}:{ElementType.WALL}",
        f"{ElementType.EXIT}:{ElementType.WALL}",
    }

    items_on_ground: Set[str] = {
        ElementType.STAIRCASE,
        ElementType.ELEVATOR,
        ElementType.SAFE_EXIT,
        ElementType.PILLAR,
        ElementType.CHARGING_STATION,
        ElementType.FIRE_EXTINGUISHER,
    }

    for el1 in solids:
        candidates = grid.get_potential_collisions(el1)
        for el2 in candidates:
            if el1.id >= el2.id:
                continue
            types = sorted([el1.type, el2.type])
            pair_key = f"{types[0]}:{types[1]}"
            if pair_key in allowed_overlaps:
                continue
            if el1.type == el2.type and el1.type == ElementType.WALL:
                continue
            if el1.type == el2.type and el1.type == ElementType.ROAD:
                continue

            if el1.type in items_on_ground and el2.type in items_on_ground:
                box = get_intersection_box(el1, el2)
                if box:
                    violations.append(
                        ConstraintViolation(
                            elementId=el1.id,
                            targetId=el2.id,
                            type="overlap",
                            message=f"Facilities cannot overlap: {el1.type} vs {el2.type}",
                        )
                    )

            box = get_intersection_box(el1, el2)
            if box:
                if box["width"] < 1 or box["height"] < 1:
                    continue
                violations.append(
                    ConstraintViolation(
                        elementId=el1.id,
                        targetId=el2.id,
                        type="overlap",
                        message=f"Overlap violation: {el1.type} cannot overlap with {el2.type}",
                    )
                )

    items_on_road: Set[str] = {ElementType.GUIDANCE_SIGN, ElementType.SIDEWALK, ElementType.SPEED_BUMP, ElementType.LANE_LINE}

    for item in layout.elements:
        if item.type in items_on_road:
            nearby_roads = [c for c in grid.get_potential_collisions(item) if c.type == ElementType.ROAD]
            is_on_road = any(_is_overlapping(road, item, 5) for road in nearby_roads)
            if not is_on_road:
                violations.append(
                    ConstraintViolation(
                        elementId=item.id,
                        type="placement_error",
                        message=f"{item.type} must be on Driving Lane.",
                    )
                )

        if item.type in items_on_ground:
            nearby_grounds = [c for c in grid.get_potential_collisions(item) if c.type == ElementType.GROUND]
            supported = False
            for ground in nearby_grounds:
                intersection = get_intersection_box(item, ground)
                if not intersection:
                    continue
                item_area = item.width * item.height
                intersect_area = intersection["width"] * intersection["height"]
                if intersect_area > item_area * 0.8:
                    supported = True
                    break
            if not supported:
                violations.append(
                    ConstraintViolation(
                        elementId=item.id,
                        type="placement_error",
                        message=f"{item.type} must be placed on GROUND element.",
                    )
                )

    roads = [e for e in layout.elements if e.type == ElementType.ROAD]
    for i in range(len(roads)):
        for j in range(i + 1, len(roads)):
            r1 = roads[i]
            r2 = roads[j]
            intersection = get_intersection_box(r1, r2)
            if intersection and intersection["width"] > 20 and intersection["height"] > 20:
                dirty_types: Set[str] = {
                    ElementType.PARKING_SPACE,
                    ElementType.LANE_LINE,
                    ElementType.GUIDANCE_SIGN,
                    ElementType.SPEED_BUMP,
                }
                temp = LayoutElement(
                    id="temp",
                    type="temp",
                    x=intersection["x"],
                    y=intersection["y"],
                    width=intersection["width"],
                    height=intersection["height"],
                    rotation=0,
                )
                for el in layout.elements:
                    if el.type not in dirty_types:
                        continue
                    if _is_overlapping(temp, el):
                        violations.append(
                            ConstraintViolation(
                                elementId=el.id,
                                type="placement_error",
                                message=f"Intersection must be clear. Remove {el.type} from junction.",
                            )
                        )

    gates = [e for e in layout.elements if e.type == ElementType.ENTRANCE or e.type == ElementType.EXIT]
    ramps = [e for e in layout.elements if e.type == ElementType.RAMP]
    for gate in gates:
        nearby_ramps = [c for c in grid.get_potential_collisions(gate) if c.type == ElementType.RAMP]
        if not any(_is_touching(r, gate) for r in nearby_ramps):
            violations.append(
                ConstraintViolation(
                    elementId=gate.id,
                    type="connectivity_error",
                    message=f"{gate.type} needs Ramp.",
                )
            )

    for ramp in ramps:
        nearby_roads = [c for c in grid.get_potential_collisions(ramp) if c.type == ElementType.ROAD]
        if not any(_is_touching(road, ramp) for road in nearby_roads):
            violations.append(
                ConstraintViolation(
                    elementId=ramp.id,
                    type="connectivity_error",
                    message="Ramp disconnected.",
                )
            )

    walls = [e for e in layout.elements if e.type == ElementType.WALL]
    corners = [
        {"x": 0, "y": 0, "label": "Top-Left"},
        {"x": layout.width, "y": 0, "label": "Top-Right"},
        {"x": layout.width, "y": layout.height, "label": "Bottom-Right"},
        {"x": 0, "y": layout.height, "label": "Bottom-Left"},
    ]
    for pt in corners:
        if not _is_point_covered(pt["x"], pt["y"], walls, 12):
            violations.append(
                ConstraintViolation(
                    elementId=f"wall_gap_{pt['label']}",
                    type="connectivity_error",
                    message=f"CRITICAL: The perimeter is broken at the {pt['label']} corner ({pt['x']}, {pt['y']}). Walls must overlap or touch to form a closed loop.",
                )
            )

    grounds = [e for e in layout.elements if e.type == ElementType.GROUND]
    if len(grounds) == 0:
        violations.append(
            ConstraintViolation(
                elementId="global_ground_check",
                type="placement_error",
                message="CRITICAL: No 'ground' elements found. Background islands must be preserved.",
            )
        )

    return violations


def calculate_void_ratio(layout: ParkingLayout, step: int = 10) -> float:
    cover_types: Set[str] = {ElementType.GROUND, ElementType.ROAD, ElementType.WALL}
    elements = [e for e in layout.elements if e.type in cover_types]
    width = max(1, round(layout.width))
    height = max(1, round(layout.height))
    s = max(1, int(step))
    total = 0
    uncovered = 0
    for y in range(0, height, s):
        for x in range(0, width, s):
            total += 1
            covered = False
            for el in elements:
                if x >= el.x and x <= el.x + el.width and y >= el.y and y <= el.y + el.height:
                    covered = True
                    break
            if not covered:
                uncovered += 1
    return 0 if total == 0 else uncovered / total

