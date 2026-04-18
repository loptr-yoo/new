from shapely.geometry import box

from backend.core.geometry.layout_generator import estimate_corridor_area_upper


def test_estimate_corridor_area_upper_cross():
    floor = box(0.0, 0.0, 10.0, 8.0)
    cw = 2.0
    est = estimate_corridor_area_upper(floor, cw=cw, corridor_layout="cross")
    assert abs(est - (cw * 10.0 + cw * 8.0 - cw * cw)) < 1e-6


def test_estimate_corridor_area_upper_non_negative():
    floor = box(0.0, 0.0, 10.0, 8.0)
    assert estimate_corridor_area_upper(floor, cw=-1.0, corridor_layout="cross") == 0.0
