from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Bounds = Tuple[float, float, float, float]
Point2 = Tuple[float, float]


def to_screen_point(x: float, y: float, *, bounds: Bounds) -> Point2:
    minx, _miny, _maxx, maxy = bounds
    return (float(x) - float(minx), float(maxy) - float(y))


def to_screen_polygon(points: Sequence[Sequence[float]], *, bounds: Bounds, preserve_winding: bool = True) -> List[List[float]]:
    out: List[List[float]] = []
    for p in points:
        if len(p) < 2:
            continue
        sx, sy = to_screen_point(float(p[0]), float(p[1]), bounds=bounds)
        out.append([sx, sy])
    if preserve_winding and len(out) >= 3:
        out = list(reversed(out))
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


def mirror_swing_dir(swing_dir: Any) -> Any:
    val = str(swing_dir or "").lower()
    if val == "left":
        return "right"
    if val == "right":
        return "left"
    return swing_dir


def math_to_screen_element(element: Dict[str, Any], *, bounds: Bounds) -> Dict[str, Any]:
    """Convert an exported element dict from math-space into screen-space.

    Point-like fields and bounding-box fields use different Y formulas. The
    function mutates and returns a shallow copy so callers can keep a single
    conversion choke point without accidentally double-flipping elements.
    """
    out = dict(element)

    if {"x", "y", "width", "height"}.issubset(out):
        try:
            sx, sy, sw, sh = to_screen_rect_min(
                float(out["x"]),
                float(out["y"]),
                float(out["width"]),
                float(out["height"]),
                bounds=bounds,
            )
            out["x"], out["y"], out["width"], out["height"] = sx, sy, sw, sh
        except Exception:
            pass

    if "position" in out and isinstance(out.get("position"), (list, tuple)) and len(out["position"]) >= 2:
        try:
            sx, sy = to_screen_point(float(out["position"][0]), float(out["position"][1]), bounds=bounds)
            out["position"] = [sx, sy]
        except Exception:
            pass

    for key in ("center",):
        val = out.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                sx, sy = to_screen_point(float(val[0]), float(val[1]), bounds=bounds)
                out[key] = [sx, sy]
            except Exception:
                pass

    if "cx" in out and "cy" in out:
        try:
            sx, sy = to_screen_point(float(out["cx"]), float(out["cy"]), bounds=bounds)
            out["cx"], out["cy"] = sx, sy
        except Exception:
            pass

    poly = out.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        try:
            out["polygon"] = to_screen_polygon(poly, bounds=bounds)
        except Exception:
            pass

    coords = out.get("coords")
    if isinstance(coords, list) and coords:
        try:
            out["coords"] = to_screen_polygon(coords, bounds=bounds, preserve_winding=False)
        except Exception:
            pass

    fwd = out.get("forward")
    if isinstance(fwd, (list, tuple)) and len(fwd) >= 2:
        try:
            fx, fy, _ = to_screen_forward(float(fwd[0]), float(fwd[1]))
            out["forward"] = [fx, fy, 0.0]
            if out.get("type") == "furniture":
                angle = math.degrees(math.atan2(fy, fx))
                out["rotation"] = float(angle % 360.0)
        except Exception:
            pass
    elif out.get("type") == "furniture" and "rotation" in out:
        try:
            out["rotation"] = to_screen_rotation_cw(float(out["rotation"]))
        except Exception:
            pass

    if out.get("type") == "door" and "swing_dir" in out:
        out["swing_dir"] = mirror_swing_dir(out.get("swing_dir"))

    out["coord_space"] = "screen"
    return out


def _norm2(x: float, y: float) -> float:
    return float(math.hypot(float(x), float(y)))


def is_unit_2d(x: float, y: float, *, eps: float = 1e-6) -> bool:
    return abs(_norm2(x, y) - 1.0) <= float(eps)
