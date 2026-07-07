from __future__ import annotations

import pytest

from building.app.models import BuildingAllocation, FloorAllocation, RoomAllocation


def _room(room_id: str = "room_1") -> RoomAllocation:
    return RoomAllocation(room_id=room_id, room_name="Bedroom", room_type="bedroom", target_area=10.0)


def _floor(n: int) -> FloorAllocation:
    return FloorAllocation(
        floor_number=n,
        floor_function_tag="residential",
        floor_total_area=60.0,
        core_tube_area=6.0,
        corridor_allowance_area=6.0,
        rooms=[_room(f"room_{n}")],
    )


def test_building_allocation_accepts_contiguous_multifloor() -> None:
    alloc = BuildingAllocation(
        building_name="Two Floor House",
        total_floors=2,
        overall_total_area=120.0,
        floors=[_floor(1), _floor(2)],
    )
    assert alloc.total_floors == 2


def test_building_allocation_rejects_single_floor() -> None:
    with pytest.raises(Exception):
        BuildingAllocation(
            building_name="One Floor House",
            total_floors=1,
            overall_total_area=60.0,
            floors=[_floor(1)],
        )


def test_building_allocation_rejects_floor_count_mismatch() -> None:
    with pytest.raises(Exception):
        BuildingAllocation(
            building_name="Mismatch",
            total_floors=2,
            overall_total_area=120.0,
            floors=[_floor(1)],
        )


def test_building_allocation_rejects_non_contiguous_floor_numbers() -> None:
    with pytest.raises(Exception):
        BuildingAllocation(
            building_name="Gap",
            total_floors=2,
            overall_total_area=120.0,
            floors=[_floor(1), _floor(3)],
        )
