import pytest

from backend.core.interior.coarse_layout_agent import _validate_layout_coarsely
from backend.core.interior.models import (
    FurnitureCategory,
    FurnitureSpec,
    LLMCoarseLayout,
    LLMCoarseLayoutItem,
    Obstacle,
    RoomBoundary,
)
from backend.core.llm.provider import LLMOutputFormatError


def test_furniture_category_is_limited_to_8_values():
    assert {c.value for c in FurnitureCategory} == {
        "床具",
        "坐具",
        "电器",
        "柜子",
        "桌子",
        "椅子",
        "挂件",
        "摆件",
    }


def test_rotation_is_limited_to_orthogonal_values():
    ok = LLMCoarseLayoutItem(furniture_id="f1", cx=1.0, cy=2.0, rotation=90)
    assert ok.rotation == 90
    with pytest.raises(Exception):
        LLMCoarseLayoutItem(furniture_id="f1", cx=1.0, cy=2.0, rotation=45)  # type: ignore[arg-type]


def test_room_and_obstacle_minmax_validation():
    RoomBoundary(x_min=0.0, y_min=0.0, x_max=4.0, y_max=3.0)
    Obstacle(name="door", x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

    with pytest.raises(ValueError):
        RoomBoundary(x_min=1.0, y_min=0.0, x_max=1.0, y_max=3.0)
    with pytest.raises(ValueError):
        Obstacle(name="bad", x_min=1.0, y_min=0.0, x_max=0.0, y_max=1.0)


def test_validate_layout_strict_id_set_and_count():
    room = RoomBoundary(x_min=0.0, y_min=0.0, x_max=4.0, y_max=3.0)
    furnitures = [
        FurnitureSpec(id="bed_1", name="床", category=FurnitureCategory.BEDDING, width=2.0, height=1.8),
        FurnitureSpec(id="cab_1", name="柜子", category=FurnitureCategory.CABINET, width=1.0, height=0.5),
    ]
    obstacles = []

    layout = LLMCoarseLayout(
        reasoning="- ok",
        items=[
            LLMCoarseLayoutItem(furniture_id="bed_1", cx=2.0, cy=2.0, rotation=0),
            LLMCoarseLayoutItem(furniture_id="cab_1", cx=3.0, cy=1.0, rotation=90),
        ],
    )
    warnings = _validate_layout_coarsely(layout, room, furnitures, obstacles, soft_margin=0.5)
    assert isinstance(warnings, list)

    bad_count = LLMCoarseLayout(reasoning="- bad", items=[LLMCoarseLayoutItem(furniture_id="bed_1", cx=2.0, cy=2.0, rotation=0)])
    with pytest.raises(LLMOutputFormatError):
        _validate_layout_coarsely(bad_count, room, furnitures, obstacles, soft_margin=0.5)

    bad_ids = LLMCoarseLayout(
        reasoning="- bad",
        items=[
            LLMCoarseLayoutItem(furniture_id="bed_1", cx=2.0, cy=2.0, rotation=0),
            LLMCoarseLayoutItem(furniture_id="bed_1", cx=3.0, cy=1.0, rotation=90),
        ],
    )
    with pytest.raises(LLMOutputFormatError):
        _validate_layout_coarsely(bad_ids, room, furnitures, obstacles, soft_margin=0.5)


def test_validate_layout_centerpoint_out_of_room_or_in_obstacle_triggers_retry():
    room = RoomBoundary(x_min=0.0, y_min=0.0, x_max=4.0, y_max=3.0)
    furnitures = [
        FurnitureSpec(id="bed_1", name="床", category=FurnitureCategory.BEDDING, width=2.0, height=1.8),
    ]
    obstacles = [Obstacle(name="door", x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)]

    out_room = LLMCoarseLayout(
        reasoning="- bad",
        items=[LLMCoarseLayoutItem(furniture_id="bed_1", cx=5.0, cy=2.0, rotation=0)],
    )
    with pytest.raises(LLMOutputFormatError):
        _validate_layout_coarsely(out_room, room, furnitures, obstacles, soft_margin=0.5)

    in_ob = LLMCoarseLayout(
        reasoning="- bad",
        items=[LLMCoarseLayoutItem(furniture_id="bed_1", cx=0.5, cy=0.5, rotation=0)],
    )
    with pytest.raises(LLMOutputFormatError):
        _validate_layout_coarsely(in_ob, room, furnitures, obstacles, soft_margin=0.5)

