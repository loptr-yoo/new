import pytest

from backend.models import FloorAllocation, RoomAllocation


def test_floor_allocation_scales_rooms_when_mismatch_exceeds_5_percent():
    rooms = [
        RoomAllocation(
            room_name="A",
            room_type="office",
            target_area=10.0,
            requires_window=True,
            weight=5,
            adjacency_tags=["work"],
        ),
        RoomAllocation(
            room_name="B",
            room_type="office",
            target_area=10.0,
            requires_window=True,
            weight=5,
            adjacency_tags=["work"],
        ),
        RoomAllocation(
            room_name="C",
            room_type="meeting",
            target_area=10.0,
            requires_window=False,
            weight=5,
            adjacency_tags=["meeting"],
        ),
    ]

    floor = FloorAllocation(
        floor_number=1,
        floor_function_tag="office",
        floor_total_area=100.0,
        core_tube_area=20.0,
        corridor_allowance_area=20.0,
        rooms=rooms,
    )

    room_sum = sum(r.target_area for r in floor.rooms)
    assert pytest.approx(room_sum, rel=1e-9, abs=1e-9) == 60.0
    assert pytest.approx(room_sum + floor.core_tube_area + floor.corridor_allowance_area, rel=1e-9, abs=1e-9) == 100.0


def test_floor_allocation_does_not_scale_when_within_5_percent():
    floor = FloorAllocation(
        floor_number=1,
        floor_function_tag="office",
        floor_total_area=100.0,
        core_tube_area=20.0,
        corridor_allowance_area=20.0,
        rooms=[
            RoomAllocation(
                room_name="A",
                room_type="office",
                target_area=59.0,
                requires_window=True,
                weight=5,
                adjacency_tags=[],
            ),
        ],
    )

    assert pytest.approx(sum(r.target_area for r in floor.rooms), rel=1e-9, abs=1e-9) == 59.0
