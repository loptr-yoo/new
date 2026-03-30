from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from .building_types import FloorSkeleton
from .style_constants import ELEMENT_COLORS


def _as_polygons(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_as_polygons(g))
        return out
    return []


def _collect_bounds(polys: Iterable[Polygon]) -> Tuple[float, float, float, float]:
    minx = float("inf")
    miny = float("inf")
    maxx = float("-inf")
    maxy = float("-inf")
    any_found = False
    for p in polys:
        if p.is_empty:
            continue
        bx0, by0, bx1, by1 = p.bounds
        minx = min(minx, bx0)
        miny = min(miny, by0)
        maxx = max(maxx, bx1)
        maxy = max(maxy, by1)
        any_found = True
    if not any_found:
        return 0.0, 0.0, 1.0, 1.0
    return minx, miny, maxx, maxy


def _ring_to_path(coords) -> str:
    pts = list(coords)
    if len(pts) < 3:
        return ""
    parts = [f"M {pts[0][0]} {pts[0][1]}"]
    for x, y in pts[1:]:
        parts.append(f"L {x} {y}")
    parts.append("Z")
    return " ".join(parts)


def _polygon_to_path(p: Polygon) -> str:
    d = _ring_to_path(p.exterior.coords)
    for hole in p.interiors:
        d2 = _ring_to_path(hole.coords)
        if d2:
            d = f"{d} {d2}"
    return d


def _path_element(d: str, fill: str, stroke: str, stroke_width: float, fill_rule: str = "evenodd") -> str:
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'fill-rule="{fill_rule}" />'
    )


def export_skeleton_to_svg(skeleton: FloorSkeleton, filepath: str) -> None:
    all_polys: List[Polygon] = []
    all_polys.extend(_as_polygons(skeleton.boundary_polygon))
    all_polys.extend(_as_polygons(skeleton.core_tube_polygon))
    all_polys.extend(_as_polygons(skeleton.corridor_polygon))
    for p in skeleton.usable_islands:
        all_polys.extend(_as_polygons(p))

    minx, miny, maxx, maxy = _collect_bounds(all_polys)
    pad = max(1.0, 0.05 * max(maxx - minx, maxy - miny))
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    width = maxx - minx
    height = maxy - miny
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    translate_y = miny + maxy

    elems: List[str] = []

    for island in skeleton.usable_islands:
        for p in _as_polygons(island):
            d = _polygon_to_path(p)
            if not d:
                continue
            elems.append(
                _path_element(
                    d=d,
                    fill=ELEMENT_COLORS["usable_island"],
                    stroke=ELEMENT_COLORS["boundary"],
                    stroke_width=0.2,
                    fill_rule="evenodd",
                )
            )

    for p in _as_polygons(skeleton.corridor_polygon):
        d = _polygon_to_path(p)
        if not d:
            continue
        elems.append(
            _path_element(
                d=d,
                fill=ELEMENT_COLORS["corridor"],
                stroke=ELEMENT_COLORS["boundary"],
                stroke_width=0.2,
                fill_rule="evenodd",
            )
        )

    for p in _as_polygons(skeleton.core_tube_polygon):
        d = _polygon_to_path(p)
        if not d:
            continue
        elems.append(
            _path_element(
                d=d,
                fill=ELEMENT_COLORS["core_tube"],
                stroke=ELEMENT_COLORS["boundary"],
                stroke_width=0.2,
                fill_rule="evenodd",
            )
        )

    boundary_paths = [_polygon_to_path(p) for p in _as_polygons(skeleton.boundary_polygon)]
    for d in boundary_paths:
        if not d:
            continue
        elems.append(
            _path_element(
                d=d,
                fill="none",
                stroke=ELEMENT_COLORS["boundary"],
                stroke_width=0.6,
                fill_rule="evenodd",
            )
        )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{minx} {miny} {width} {height}">\n'
        f'  <g transform="translate(0,{translate_y}) scale(1,-1)">\n'
        f"    {' '.join(elems)}\n"
        "  </g>\n"
        "</svg>\n"
    )

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
