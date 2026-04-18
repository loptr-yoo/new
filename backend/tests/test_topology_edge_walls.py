from shapely.geometry import JOIN_STYLE, LineString, Polygon, box
from shapely.ops import unary_union


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
    zone_types = {k: "room" for k in rects.keys()}
    walls = generate_walls_from_topology(rects, edge_set, floor_bounds=(0.0, 0.0, 12.0, 5.0), zone_types=zone_types)

    partition_walls = [w for w in walls if w.type == "partition_wall"]
    assert partition_walls
    assert all(isinstance(w.geometry, LineString) for w in partition_walls)
    assert all(len(list(w.geometry.coords)) == 2 for w in partition_walls)

    keys = set()
    for w in partition_walls:
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


def test_generate_doors_allows_core_staircase_and_elevator_hall_and_blocks_shaft():
    from backend.core.geometry.postprocessor import DoorPlacement, WallSegment, generate_doors

    zone_types = {
        "corridor_0": "corridor",
        "core_staircase": "staircase",
        "core_elevator_hall": "elevator_hall",
        "core_elevator_shaft": "elevator_shaft",
    }

    walls = [
        WallSegment(
            type="partition_wall",
            geometry=LineString([(0.0, 0.0), (2.0, 0.0)]),
            thickness=0.12,
            room_ids=["corridor_0", "core_staircase"],
        ),
        WallSegment(
            type="partition_wall",
            geometry=LineString([(2.0, 0.0), (4.0, 0.0)]),
            thickness=0.12,
            room_ids=["corridor_0", "core_elevator_hall"],
        ),
        WallSegment(
            type="partition_wall",
            geometry=LineString([(2.0, 0.0), (2.0, 2.0)]),
            thickness=0.12,
            room_ids=["core_staircase", "core_elevator_hall"],
        ),
        WallSegment(
            type="partition_wall",
            geometry=LineString([(2.0, 2.0), (4.0, 2.0)]),
            thickness=0.12,
            room_ids=["core_elevator_hall", "core_elevator_shaft"],
        ),
    ]

    doors = generate_doors(walls, zone_types=zone_types, zone_rects={})
    assert all(isinstance(d, DoorPlacement) for d in doors)

    connects = [frozenset(d.connects) for d in doors]
    assert frozenset({"corridor_0", "core_elevator_hall"}) in connects
    assert frozenset({"core_staircase", "core_elevator_hall"}) in connects
    assert all("core_elevator_shaft" not in d.connects for d in doors)
    assert frozenset({"corridor_0", "core_staircase"}) not in connects


def test_generate_walls_from_topology_blocks_elevator_hall_to_shaft_wall():
    from backend.core.geometry.postprocessor import generate_walls_from_topology

    rects = {
        "core_elevator_hall": (5.0, 4.0, 2.0, 1.0),
        "core_elevator_shaft": (5.0, 5.0, 2.0, 1.0),
    }
    zone_types = {"core_elevator_hall": "elevator_hall", "core_elevator_shaft": "elevator_shaft"}
    edge_set = {frozenset({"core_elevator_hall", "core_elevator_shaft"}): "horizontal"}
    walls = generate_walls_from_topology(rects, edge_set=edge_set, floor_bounds=(0.0, 0.0, 10.0, 8.0), zone_types=zone_types)
    blocked = [
        w for w in walls
        if w.type == "partition_wall" and set(w.room_ids) == {"core_elevator_hall", "core_elevator_shaft"}
    ]
    assert blocked == []


def test_generate_walls_from_topology_adds_core_north_shell_wall_when_void_gap_exists():
    from backend.core.geometry.postprocessor import generate_walls_from_topology

    floor_bounds = (0.0, 0.0, 10.0, 10.0)
    rects = {
        "core_staircase": (2.0, 8.0, 2.0, 1.0),
    }
    zone_types = {"core_staircase": "staircase"}
    walls = generate_walls_from_topology(rects, edge_set={}, floor_bounds=floor_bounds, zone_types=zone_types)
    top_y = 9.0
    shell = []
    for w in walls:
        if w.type != "partition_wall":
            continue
        if w.room_ids != ["core_staircase"]:
            continue
        (x0, y0), (x1, y1) = list(w.geometry.coords)
        if abs(y0 - top_y) < 0.02 and abs(y1 - top_y) < 0.02 and (max(x0, x1) - min(x0, x1)) > 1.5:
            shell.append(w)
    assert shell


def test_check_connectivity_topological_reaches_rooms_via_corridor():
    from backend.core.geometry.layout_generator import check_connectivity_topological

    edge_set = {
        frozenset({"corridor_0", "room_a"}): "vertical",
        frozenset({"room_a", "room_b"}): "vertical",
    }
    unreachable = check_connectivity_topological(edge_set, ["corridor_0", "room_a", "room_b"])
    assert unreachable == []


def test_exterior_walls_do_not_depend_on_rooms_touching_bounds():
    from backend.core.geometry.postprocessor import generate_walls_from_topology

    rects = {
        "room_a": (2.0, 2.0, 3.0, 3.0),
    }
    walls = generate_walls_from_topology(rects, edge_set={}, floor_bounds=(0.0, 0.0, 10.0, 8.0), zone_types={"room_a": "room"})
    exterior = [w for w in walls if w.type == "exterior_wall"]
    assert len(exterior) == 4
    assert all(isinstance(w.geometry, Polygon) for w in exterior)


def test_exterior_wall_pieces_are_mutually_exclusive_and_cover_ring():
    from backend.core.geometry.postprocessor import generate_walls_from_topology

    floor_bounds = (0.0, 0.0, 10.0, 8.0)
    rects = {"room_a": (2.0, 2.0, 3.0, 3.0)}
    zone_types = {k: "room" for k in rects.keys()}
    walls = generate_walls_from_topology(rects, edge_set={}, floor_bounds=floor_bounds, zone_types=zone_types)
    exterior_polys = [w.geometry for w in walls if w.type == "exterior_wall"]
    assert len(exterior_polys) == 4

    for i in range(len(exterior_polys)):
        for j in range(i + 1, len(exterior_polys)):
            assert exterior_polys[i].intersection(exterior_polys[j]).area <= 1e-6

    outer = box(*floor_bounds)
    t = 0.24
    inner = outer.buffer(-t, join_style=JOIN_STYLE.mitre)
    ring = (outer if inner.is_empty else outer.difference(inner)).buffer(0)
    union = unary_union(exterior_polys).buffer(0)
    assert abs(union.area - ring.area) <= 1e-3


def test_corner_gap_fixed_by_partition_extension_overlaps_exterior_wall():
    from backend.core.geometry.postprocessor import generate_walls_from_topology, wall_to_dict

    floor_bounds = (0.0, 0.0, 10.0, 10.0)
    rects = {
        "room_a": (0.0, 0.24, 5.0, 5.0),
        "corridor_0": (5.0, 0.24, 5.0, 5.0),
    }
    edge_set = {frozenset({"room_a", "corridor_0"}): "vertical"}
    zone_types = {k: "room" for k in rects.keys()}
    walls = generate_walls_from_topology(rects, zone_types={}, edge_set=edge_set, floor_bounds=floor_bounds)

    exterior_union = unary_union([w.geometry for w in walls if w.type == "exterior_wall"]).buffer(0)
    part = next(w for w in walls if w.type == "partition_wall")
    part_poly_coords = wall_to_dict(part)["polygon"]
    part_poly = Polygon(part_poly_coords)
    assert part_poly.intersection(exterior_union).area > 1e-6


def test_partition_walls_do_not_extend_outside_floor_bounds():
    from backend.core.geometry.postprocessor import generate_walls_from_topology

    floor_bounds = (0.0, 0.0, 11.29, 7.53)
    rects = {
        "room_001": (0.0, 0.0, 8.4, 2.5),
        "corridor_h": (0.0, 2.5, 11.29, 2.0),
    }
    edge_set = {frozenset({"room_001", "corridor_h"}): "horizontal"}
    walls = generate_walls_from_topology(rects, edge_set=edge_set, floor_bounds=floor_bounds,zone_types={})
    floor = box(*floor_bounds)
    for w in walls:
        if w.type != "partition_wall":
            continue
        (x0, y0), (x1, y1) = list(w.geometry.coords)
        assert x0 >= 0.0 and x1 >= 0.0 and y0 >= 0.0 and y1 >= 0.0
        assert x0 <= floor_bounds[2] and x1 <= floor_bounds[2]
        assert y0 <= floor_bounds[3] and y1 <= floor_bounds[3]


def test_edge_set_allows_small_gaps_between_spaces():
    from backend.core.geometry.layout_generator import _build_edge_set_from_rects

    rects = {
        "room_001": (0.0, 0.0, 6.4, 2.7),
        "corridor_h": (0.0, 2.75, 10.61, 1.5),
    }
    edge_set = _build_edge_set_from_rects(rects, tol=0.06)
    key = frozenset({"room_001", "corridor_h"})
    assert edge_set.get(key) == "horizontal"


def test_extend_line_slanted_and_zero_length():
    from backend.core.geometry.postprocessor import _extend_line

    a = LineString([(0.0, 0.0), (1.0, 1.0)])
    b = _extend_line(a, 0.5)
    assert abs(b.length - (a.length + 1.0)) <= 1e-6

    z = LineString([(1.0, 1.0), (1.0, 1.0)])
    z2 = _extend_line(z, 0.5)
    assert z2.length == 0.0


def test_windows_generated_from_floor_bounds_have_correct_rotation():
    from backend.core.geometry.postprocessor import generate_windows_from_floor_boundary

    floor_bounds = (0.0, 0.0, 10.0, 8.0)
    room_rects = {"room_a": (0.01, 2.0, 3.0, 3.0)}
    zone_types = {"room_a": "room"}
    rooms_needing_window = {"room_a"}

    windows = generate_windows_from_floor_boundary(
        room_rects=room_rects,
        zone_types=zone_types,
        rooms_needing_window=rooms_needing_window,
        floor_bounds=floor_bounds,
        exterior_thickness=0.24,
    )
    assert windows
    assert all(w.rotation == 90.0 for w in windows)
    assert all(abs(w.position[0] - 0.12) <= 1e-6 for w in windows)
