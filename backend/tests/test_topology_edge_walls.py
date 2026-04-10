from shapely.geometry import LineString, box


def test_core_tube_north_stays_inside_floor():
    from backend.core.geometry.topology_generator import CoreTube

    boundary = box(0, 0, 11.62, 7.75)
    minx, miny, maxx, maxy = boundary.bounds
    core = CoreTube.create_for_floor((minx, miny, maxx, maxy), position="north", grid_alignment=0.5)

    assert boundary.contains(core.polygon)
    if core.elevator is not None:
        assert boundary.contains(core.elevator)
    if core.staircase is not None:
        assert boundary.contains(core.staircase)


def test_rectangular_topology_islands_are_rectangles():
    from backend.core.geometry.topology_generator import generate_rectangular_topology

    boundary = box(0, 0, 20, 15)
    core, corridors, islands = generate_rectangular_topology(
        floor_boundary=boundary,
        corridor_width=1.5,
        corridor_layout="cross",
    )

    assert core.polygon.is_valid
    assert len(corridors) > 0
    assert len(islands) > 0
    assert all(i.is_rectangular for i in islands)


def test_generate_walls_from_topology_outputs_unique_two_point_lines():
    from backend.core.geometry.postprocessor import generate_walls_from_topology

    rects = {
        "room_a": (0.0, 0.0, 5.0, 5.0),
        "corridor_0": (5.0, 0.0, 2.0, 5.0),
        "room_b": (7.0, 0.0, 5.0, 5.0),
    }
    edge_set = {
        frozenset({"room_a", "corridor_0"}): "vertical",
        frozenset({"corridor_0", "room_b"}): "vertical",
    }
    walls = generate_walls_from_topology(rects, edge_set, floor_bounds=(0.0, 0.0, 12.0, 5.0))

    assert all(isinstance(w.geometry, LineString) for w in walls)
    assert all(len(list(w.geometry.coords)) == 2 for w in walls)

    keys = set()
    for w in walls:
        (x0, y0), (x1, y1) = list(w.geometry.coords)
        a = (round(x0, 2), round(y0, 2))
        b = (round(x1, 2), round(y1, 2))
        p1, p2 = (a, b) if a <= b else (b, a)
        key = (w.type, round(float(w.thickness), 3), p1, p2)
        assert key not in keys
        keys.add(key)


def test_generate_doors_whitelist_blocks_core_non_door_face():
    from backend.core.geometry.postprocessor import DoorPlacement, WallSegment, generate_doors

    zone_types = {"corridor_0": "corridor", "core_tube": "core"}
    zone_rects = {"core_tube": (4.0, 4.0, 2.0, 2.0)}

    south_wall = WallSegment(
        type="partition_wall",
        geometry=LineString([(4.0, 4.0), (6.0, 4.0)]),
        thickness=0.12,
        room_ids=["corridor_0", "core_tube"],
    )
    east_wall = WallSegment(
        type="partition_wall",
        geometry=LineString([(6.0, 4.0), (6.0, 6.0)]),
        thickness=0.12,
        room_ids=["corridor_0", "core_tube"],
    )

    doors = generate_doors([south_wall, east_wall], zone_types=zone_types, zone_rects=zone_rects)
    assert all(isinstance(d, DoorPlacement) for d in doors)
    assert len(doors) == 1
    assert doors[0].rotation == 0.0


def test_check_connectivity_topological_reaches_rooms_via_corridor():
    from backend.core.geometry.layout_generator import check_connectivity_topological

    edge_set = {
        frozenset({"corridor_0", "room_a"}): "vertical",
        frozenset({"room_a", "room_b"}): "vertical",
    }
    unreachable = check_connectivity_topological(edge_set, ["corridor_0", "room_a", "room_b"])
    assert unreachable == []
