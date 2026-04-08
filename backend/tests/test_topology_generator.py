import pytest

pytestmark = pytest.mark.skip(reason="deprecated: generate_floor_skeleton removed, use RectangularTopologyGenerator")

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon


def _floor(area: float, core: float, corridor: float) -> FloorAllocation:
    return FloorAllocation(
        floor_number=1,
        floor_function_tag="office",
        floor_total_area=area,
        core_tube_area=core,
        corridor_allowance_area=corridor,
        rooms=[
            RoomAllocation(
                room_name="A",
                room_type="office",
                target_area=max(1.0, area - core - corridor),
                requires_window=True,
                weight=5,
                adjacency_tags=[],
            )
        ],
    )


def test_default_boundary_area_matches_floor_total_area():
    floor = _floor(area=1200.0, core=180.0, corridor=180.0)
    sk = generate_floor_skeleton(floor, outline_points=None)
    assert sk.boundary_polygon.area == pytest.approx(1200.0, rel=1e-6, abs=1e-6)
    assert sk.core_tube_polygon.within(sk.boundary_polygon.buffer(-1e-6))
    assert sk.corridor_polygon.intersects(sk.boundary_polygon)


def test_outline_points_are_scaled_to_match_floor_total_area():
    rect = Polygon([(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)])
    assert rect.area == 5000.0

    floor = _floor(area=1000.0, core=150.0, corridor=150.0)
    outline = [(float(x), float(y)) for x, y in list(rect.exterior.coords)[:-1]]
    sk = generate_floor_skeleton(floor, outline_points=outline)
    assert sk.boundary_polygon.area == pytest.approx(1000.0, rel=1e-6, abs=1e-6)


def test_concave_outline_core_uses_interior_anchor():
    outline = [
        (0.0, 0.0),
        (6.0, 0.0),
        (6.0, 2.0),
        (2.0, 2.0),
        (2.0, 6.0),
        (0.0, 6.0),
    ]
    poly = Polygon(outline)
    floor = _floor(area=float(poly.area), core=2.0, corridor=2.0)
    sk = generate_floor_skeleton(floor, outline_points=outline)
    assert sk.core_tube_polygon.within(sk.boundary_polygon.buffer(-1e-6))
    assert sk.core_tube_polygon.intersects(sk.corridor_polygon) or sk.core_tube_polygon.touches(sk.corridor_polygon)


def test_corridor_components_connected_to_core_are_kept():
    outline = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (7.0, 10.0),
        (7.0, 3.0),
        (3.0, 3.0),
        (3.0, 10.0),
        (0.0, 10.0),
    ]
    poly = Polygon(outline)
    floor = _floor(area=float(poly.area), core=4.0, corridor=6.0)
    sk = generate_floor_skeleton(floor, outline_points=outline)

    if isinstance(sk.corridor_polygon, MultiPolygon):
        for g in sk.corridor_polygon.geoms:
            assert g.intersects(sk.core_tube_polygon.buffer(1e-6))


def test_usable_islands_are_reachable_from_corridor():
    floor = _floor(area=600.0, core=80.0, corridor=80.0)
    sk = generate_floor_skeleton(floor, outline_points=None)
    assert all(p.area >= 2.0 for p in sk.usable_islands)
    assert all(p.intersects(sk.corridor_polygon) for p in sk.usable_islands)
