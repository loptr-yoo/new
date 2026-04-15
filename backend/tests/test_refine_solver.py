import pytest

from backend.core.interior.models import (
    FurnitureCategory,
    FurnitureSpec,
    LLMCoarseLayout,
    LLMCoarseLayoutItem,
    Obstacle,
    RoomBoundary,
)
from backend.core.interior.refine_solver import solve_refined_layout


def _rect_from_center(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 < bx1) and (bx0 < ax1) and (ay0 < by1) and (by0 < ay1)


def test_refine_solver_separates_overlapping_furniture():
    room = RoomBoundary(x_min=0.0, y_min=0.0, x_max=4.0, y_max=3.0)
    furnitures = [
        FurnitureSpec(id="bed_1", name="床", category=FurnitureCategory.BEDDING, width=1.8, height=2.0),
        FurnitureSpec(id="cab_1", name="柜", category=FurnitureCategory.CABINET, width=1.2, height=0.6),
    ]
    coarse = LLMCoarseLayout(
        reasoning="- coarse",
        items=[
            LLMCoarseLayoutItem(furniture_id="bed_1", cx=2.0, cy=1.5, rotation=0),
            LLMCoarseLayoutItem(furniture_id="cab_1", cx=2.0, cy=1.5, rotation=0),
        ],
    )
    refined = solve_refined_layout(coarse, room, furnitures, obstacles=[], time_limit=1.0)
    assert refined.status in ("optimal", "feasible")
    by_id = {it.furniture_id: it for it in refined.items}

    bed = by_id["bed_1"]
    cab = by_id["cab_1"]
    bed_rect = _rect_from_center(bed.cx, bed.cy, 1.8, 2.0)
    cab_rect = _rect_from_center(cab.cx, cab.cy, 1.2, 0.6)
    assert not _overlap(bed_rect, cab_rect)

    for it in refined.items:
        assert room.x_min <= it.cx <= room.x_max
        assert room.y_min <= it.cy <= room.y_max


def test_refine_solver_avoids_obstacle():
    room = RoomBoundary(x_min=0.0, y_min=0.0, x_max=4.0, y_max=3.0)
    furnitures = [
        FurnitureSpec(id="bed_1", name="床", category=FurnitureCategory.BEDDING, width=1.8, height=2.0),
    ]
    obstacles = [
        Obstacle(name="door", x_min=0.0, y_min=0.0, x_max=1.2, y_max=1.0),
    ]
    coarse = LLMCoarseLayout(
        reasoning="- coarse",
        items=[
            LLMCoarseLayoutItem(furniture_id="bed_1", cx=0.6, cy=0.5, rotation=0),
        ],
    )
    refined = solve_refined_layout(coarse, room, furnitures, obstacles=obstacles, time_limit=1.0)
    assert refined.status in ("optimal", "feasible")
    bed = refined.items[0]
    assert not (obstacles[0].x_min <= bed.cx <= obstacles[0].x_max and obstacles[0].y_min <= bed.cy <= obstacles[0].y_max)


def test_crowded_room_is_infeasible():
    room = RoomBoundary(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    furnitures = [
        FurnitureSpec(id="bed_1", name="床", category=FurnitureCategory.BEDDING, width=2.0, height=2.0),
    ]
    coarse = LLMCoarseLayout(
        reasoning="- coarse",
        items=[
            LLMCoarseLayoutItem(furniture_id="bed_1", cx=0.5, cy=0.5, rotation=0),
        ],
    )
    with pytest.raises(Exception):
        solve_refined_layout(coarse, room, furnitures, obstacles=[], time_limit=0.5)

