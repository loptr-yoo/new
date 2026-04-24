from shapely.geometry import LineString, box

from backend.core.geometry.postprocessor import WallSegment, postprocess_floor


def test_ground_floor_corridor_exterior_door_right_wall() -> None:
    floor = box(0.0, 0.0, 10.0, 8.0)
    walls = [
        WallSegment(
            type="exterior_wall",
            geometry=LineString([(10.0, 0.24), (10.0, 7.76)]),
            thickness=0.24,
            room_ids=["__exterior__"],
        ),
    ]
    zone_rects = {
        "corridor_h": (8.0, 2.0, 2.0, 2.0),
    }
    zone_types = {
        "corridor_h": "corridor",
    }
    pp = postprocess_floor(
        rooms=[],
        floor_boundary=floor,
        corridors=[],
        is_ground_floor=True,
        walls=walls,
        zone_types=zone_types,
        zone_rects=zone_rects,
        rooms_needing_window=set(),
        floor_bounds=floor.bounds,
    )
    exterior_doors = [d for d in pp.doors if "__exterior__" in (d.connects or [])]
    assert len(exterior_doors) == 1
    d = exterior_doors[0]
    assert d.connects == ["corridor_h", "__exterior__"]
    assert abs(float(d.width) - 0.24) < 1e-6
    assert abs(float(d.thickness) - 0.24) < 1e-6
    assert abs(float(d.position[0]) - 9.88) < 1e-2
    assert abs(float(d.rotation) - 90.0) < 1e-6


def test_non_ground_floor_no_exterior_door() -> None:
    floor = box(0.0, 0.0, 10.0, 8.0)
    walls = [
        WallSegment(
            type="exterior_wall",
            geometry=LineString([(10.0, 0.24), (10.0, 7.76)]),
            thickness=0.24,
            room_ids=["__exterior__"],
        ),
    ]
    zone_rects = {
        "corridor_h": (8.0, 2.0, 2.0, 2.0),
    }
    zone_types = {
        "corridor_h": "corridor",
    }
    pp = postprocess_floor(
        rooms=[],
        floor_boundary=floor,
        corridors=[],
        is_ground_floor=False,
        walls=walls,
        zone_types=zone_types,
        zone_rects=zone_rects,
        rooms_needing_window=set(),
        floor_bounds=floor.bounds,
    )
    exterior_doors = [d for d in pp.doors if "__exterior__" in (d.connects or [])]
    assert len(exterior_doors) == 0

