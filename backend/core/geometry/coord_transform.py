from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


Bounds = Tuple[float, float, float, float]
Point2 = Tuple[float, float]


def to_screen_point(x: float, y: float, *, bounds: Bounds) -> Point2:
    minx, _miny, _maxx, maxy = bounds
    return (float(x) - float(minx), float(maxy) - float(y))


def to_screen_polygon(points: Sequence[Sequence[float]], *, bounds: Bounds) -> List[List[float]]:
    out: List[List[float]] = []
    for p in points:
        if len(p) < 2:
            continue
        sx, sy = to_screen_point(float(p[0]), float(p[1]), bounds=bounds)
        out.append([sx, sy])
    return out


def to_screen_rect_min(x: float, y: float, w: float, h: float, *, bounds: Bounds) -> Tuple[float, float, float, float]:
    minx, _miny, _maxx, maxy = bounds
    return (
        float(x) - float(minx),
        float(maxy) - (float(y) + float(h)),
        float(w),
        float(h),
    )


def to_screen_rect_center(
    cx: float, cy: float, w: float, h: float, *, bounds: Bounds
) -> Tuple[float, float, float, float]:
    minx, _miny, _maxx, maxy = bounds
    return (
        (float(cx) - float(w) / 2.0) - float(minx),
        float(maxy) - (float(cy) + float(h) / 2.0),
        float(w),
        float(h),
    )


def to_screen_rotation_cw(rotation_ccw: float) -> float:
    return float((-float(rotation_ccw)) % 360.0)


def to_screen_forward(fx: float, fy: float) -> Tuple[float, float, float]:
    return (float(fx), -float(fy), 0.0)


def _norm2(x: float, y: float) -> float:
    return float(math.hypot(float(x), float(y)))


def is_unit_2d(x: float, y: float, *, eps: float = 1e-6) -> bool:
    return abs(_norm2(x, y) - 1.0) <= float(eps)

