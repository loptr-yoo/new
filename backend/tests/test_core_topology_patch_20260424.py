from shapely.geometry import box

from backend.core.geometry.postprocessor import generate_doors, generate_walls_from_topology


def test_new_adjacency_wall_and_no_door_on_it() -> None:
    floor = (0.0, 0.0, 10.0, 10.0)
    room_rects = {
        "core_staircase_hall": (4.0, 7.0, 1.0, 0.75),
        "core_staircase_shaft": (4.0, 7.75, 1.0, 1.75),
        "core_elevator_hall": (5.0, 7.0, 1.5, 1.25),
        "core_elevator_shaft": (5.0, 8.25, 1.5, 1.25),
    }
    zone_types = {
        "core_staircase_hall": "staircase_hall",
        "core_staircase_shaft": "staircase_shaft",
        "core_elevator_hall": "elevator_hall",
        "core_elevator_shaft": "elevator_shaft",
    }

    walls = generate_walls_from_topology(
        room_rects=room_rects,
        edge_set={},
        floor_bounds=floor,
        zone_types=zone_types,
    )

    def _has_wall(a: str, b: str) -> bool:
        for w in walls:
            if w.type != "partition_wall":
                continue
            if set(w.room_ids or []) == {a, b}:
                return True
        return False

    assert _has_wall("core_staircase_shaft", "core_elevator_hall")

    doors = generate_doors(
        walls=walls,
        zone_types=zone_types,
        zone_rects=room_rects,
        door_width=0.9,
    )
    assert not any(set(d.connects or []) == {"core_staircase_shaft", "core_elevator_hall"} for d in doors)


def test_hall_hall_door_hard_takeover_has_required_fields() -> None:
    floor = (0.0, 0.0, 10.0, 10.0)
    room_rects = {
        "core_staircase_hall": (4.0, 7.0, 1.0, 0.75),
        "core_staircase_shaft": (4.0, 7.75, 1.0, 1.75),
        "core_elevator_hall": (5.0, 7.0, 1.5, 1.25),
        "core_elevator_shaft": (5.0, 8.25, 1.5, 1.25),
    }
    zone_types = {
        "core_staircase_hall": "staircase_hall",
        "core_staircase_shaft": "staircase_shaft",
        "core_elevator_hall": "elevator_hall",
        "core_elevator_shaft": "elevator_shaft",
    }

    walls = generate_walls_from_topology(
        room_rects=room_rects,
        edge_set={},
        floor_bounds=floor,
        zone_types=zone_types,
    )
    doors = generate_doors(
        walls=walls,
        zone_types=zone_types,
        zone_rects=room_rects,
        door_width=0.9,
    )
    hall_hall = [d for d in doors if set(d.connects or []) == {"core_staircase_hall", "core_elevator_hall"}]
    assert len(hall_hall) == 1
    d = hall_hall[0]
    assert 0.6 <= float(d.width) <= 0.8
    assert d.thickness > 0
    assert d.forward is not None
    assert len(d.forward) == 3
    assert abs(float(d.forward[2])) <= 1e-6
    assert abs(float(d.rotation)) in (0.0, 90.0)


def test_no_single_sided_cap_wall_when_shaft_gap_small() -> None:
    floor_bounds = (0.0, 0.0, 10.0, 10.0)
    floor_poly = box(*floor_bounds)
    room_rects = {
        "core_elevator_shaft": (4.0, 9.4, 1.5, 0.2),
    }
    zone_types = {
        "core_elevator_shaft": "elevator_shaft",
    }
    walls = generate_walls_from_topology(
        room_rects=room_rects,
        edge_set={},
        floor_bounds=floor_poly.bounds,
        zone_types=zone_types,
        exterior_thickness=0.24,
        wall_thickness=0.12,
    )
    assert not any(w.type == "partition_wall" and (w.room_ids or []) == ["core_elevator_shaft"] for w in walls)


def test_single_sided_cap_wall_kept_when_shaft_gap_large() -> None:
    floor_bounds = (0.0, 0.0, 10.0, 10.0)
    floor_poly = box(*floor_bounds)
    room_rects = {
        "core_elevator_shaft": (4.0, 8.0, 1.5, 0.5),
    }
    zone_types = {
        "core_elevator_shaft": "elevator_shaft",
    }
    walls = generate_walls_from_topology(
        room_rects=room_rects,
        edge_set={},
        floor_bounds=floor_poly.bounds,
        zone_types=zone_types,
        exterior_thickness=0.24,
        wall_thickness=0.12,
    )
    assert any(w.type == "partition_wall" and (w.room_ids or []) == ["core_elevator_shaft"] for w in walls)
