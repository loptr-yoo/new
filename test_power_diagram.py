import os
from pathlib import Path
from typing import Iterable, List, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, Point

from backend.core.geometry.power_diagram import generate_power_diagram


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


def _fmt(v: float) -> str:
    s = f"{float(v):.3f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        return "0"
    return s


def _ring_to_path(coords) -> str:
    pts = list(coords)
    if len(pts) < 3:
        return ""
    parts = [f"M {_fmt(pts[0][0])} {_fmt(pts[0][1])}"]
    for x, y in pts[1:]:
        parts.append(f"L {_fmt(x)} {_fmt(y)}")
    parts.append("Z")
    return " ".join(parts)


def _polygon_to_path(p: Polygon) -> str:
    d = _ring_to_path(p.exterior.coords)
    for hole in p.interiors:
        d2 = _ring_to_path(hole.coords)
        if d2:
            d = f"{d} {d2}"
    return d


def _path_element(d: str, fill: str, stroke: str, stroke_width: float, fill_opacity: float, fill_rule: str = "evenodd") -> str:
    return (
        f'<path d="{d}" fill="{fill}" fill-opacity="{fill_opacity}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" fill-rule="{fill_rule}" />'
    )


def _circle(cx: float, cy: float, r: float, fill: str) -> str:
    return f'<circle cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}" fill="{fill}" />'


def export_power_diagram_svg(
    boundary: Polygon,
    cells,
    seeds: List[Tuple[float, float]],
    filepath: str,
) -> None:
    polys: List[Polygon] = []
    polys.extend(_as_polygons(boundary))
    for g in cells:
        polys.extend(_as_polygons(g))

    minx, miny, maxx, maxy = _collect_bounds(polys)
    width0 = maxx - minx
    height0 = maxy - miny
    if width0 <= 0:
        width0 = 1.0
    if height0 <= 0:
        height0 = 1.0

    pad_x = max(1.0, 0.02 * width0)
    pad_y = max(1.0, 0.02 * height0)
    pad_x = min(pad_x, 0.10 * height0)
    pad_y = min(pad_y, 0.10 * width0)

    minx -= pad_x
    miny -= pad_y
    maxx += pad_x
    maxy += pad_y

    width = maxx - minx
    height = maxy - miny
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    translate_y = miny + maxy

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    elems: List[str] = []
    for i, geom in enumerate(cells):
        color = palette[i % len(palette)]
        for p in _as_polygons(geom):
            d = _polygon_to_path(p)
            if not d:
                continue
            elems.append(_path_element(d=d, fill=color, stroke="none", stroke_width=0.0, fill_opacity=1.0))

    for p in _as_polygons(boundary):
        d = _polygon_to_path(p)
        if d:
            elems.append(_path_element(d=d, fill="none", stroke="#000000", stroke_width=0.6, fill_opacity=0.0))

    for x, y in seeds:
        elems.append(_circle(float(x), float(y), r=0.5, fill="#ff0000"))

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{_fmt(minx)} {_fmt(miny)} {_fmt(width)} {_fmt(height)}">\n'
        f'  <rect x="{_fmt(minx)}" y="{_fmt(miny)}" width="{_fmt(width)}" height="{_fmt(height)}" fill="#ffffff" />\n'
        f'  <g transform="translate(0,{_fmt(translate_y)}) scale(1,-1)" shape-rendering="crispEdges">\n'
        f"    {' '.join(elems)}\n"
        "  </g>\n"
        "</svg>\n"
    )

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    boundary = Polygon(
        [
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, 40.0),
            (0.0, 40.0),
            (0.0, 25.0),
            (15.0, 25.0),
            (15.0, 15.0),
            (0.0, 15.0),
        ]
    )

    seeds = [(8.0, 8.0), (32.0, 8.0), (8.0, 32.0), (32.0, 32.0)]
    weights = [100.0, 200.0, 400.0, 800.0]
    resolution = 0.5

    for i, (x, y) in enumerate(seeds):
        if not Point(float(x), float(y)).within(boundary):
            raise ValueError(f"seed {i} is not within boundary")

    cells = generate_power_diagram(boundary=boundary, seeds=seeds, weights=weights, resolution=resolution)

    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "debug_power_diagram.svg")
    export_power_diagram_svg(boundary=boundary, cells=cells, seeds=seeds, filepath=output_filepath)
    print(f"已生成 SVG: {os.path.abspath(output_filepath)}")


if __name__ == "__main__":
    main()
