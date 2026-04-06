from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


class PrecomputedGrid(NamedTuple):
    """Cached static grid data for a given boundary + resolution."""

    x0: np.ndarray
    y0: np.ndarray
    mask: np.ndarray
    px: np.ndarray
    py: np.ndarray


def _mask_points_in_boundary(boundary: Polygon, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    orig_shape = x.shape
    flat_x = x.ravel()
    flat_y = y.ravel()
    try:
        import shapely as _shp
        _covers_xy = getattr(_shp, "covers_xy")
        mask_flat = _covers_xy(boundary, flat_x, flat_y)
    except Exception:
        from shapely.prepared import prep
        prepared_b = prep(boundary)
        mask_flat = np.array([prepared_b.covers(Point(px, py)) for px, py in zip(flat_x, flat_y)])
    return np.asarray(mask_flat, dtype=bool).reshape(orig_shape)


def precompute_grid(boundary: Polygon, resolution: float) -> PrecomputedGrid:
    """Precompute the static grid and boundary mask for a given resolution."""
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    minx, miny, maxx, maxy = boundary.bounds
    x0 = np.arange(float(minx), float(maxx), float(resolution), dtype=float)
    y0 = np.arange(float(miny), float(maxy), float(resolution), dtype=float)
    if x0.size == 0 or y0.size == 0:
        return PrecomputedGrid(
            x0=x0, y0=y0,
            mask=np.empty((0, 0), dtype=bool),
            px=np.empty((0,), dtype=float),
            py=np.empty((0,), dtype=float),
        )
    X0, Y0 = np.meshgrid(x0, y0)
    Xc = X0 + float(resolution) / 2.0
    Yc = Y0 + float(resolution) / 2.0
    mask = _mask_points_in_boundary(boundary, Xc, Yc)
    px = Xc[mask].astype(float)
    py = Yc[mask].astype(float)
    return PrecomputedGrid(x0=x0, y0=y0, mask=mask, px=px, py=py)


def _assign_power_labels(
    px: np.ndarray,
    py: np.ndarray,
    seeds: np.ndarray,
    weights: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    n_points = int(px.shape[0])
    n_seeds = int(seeds.shape[0])
    if n_points == 0 or n_seeds == 0:
        return np.zeros((n_points,), dtype=np.int32)

    labels = np.empty((n_points,), dtype=np.int32)
    for start in range(0, n_points, chunk_size):
        end = min(n_points, start + chunk_size)
        dx = px[start:end, None] - seeds[None, :, 0]
        dy = py[start:end, None] - seeds[None, :, 1]
        d = dx * dx + dy * dy - weights[None, :]
        labels[start:end] = np.argmin(d, axis=1).astype(np.int32)
    return labels


def _rectangles_by_id_from_grid(
    id_grid: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    resolution: float,
    n_seeds: int,
) -> Dict[int, List[Polygon]]:
    ny, nx = id_grid.shape
    out: Dict[int, List[Polygon]] = {i: [] for i in range(n_seeds)}
    if ny == 0 or nx == 0:
        return out

    for r in range(ny):
        row = id_grid[r]
        yb = float(y0[r])
        yt = yb + float(resolution)
        c = 0
        while c < nx:
            cell_id = int(row[c])
            if cell_id < 0:
                c += 1
                continue
            start = c
            c += 1
            while c < nx and int(row[c]) == cell_id:
                c += 1
            end = c - 1
            if 0 <= cell_id < n_seeds:
                xl = float(x0[start])
                xr = float(x0[end]) + float(resolution)
                out[cell_id].append(box(xl, yb, xr, yt).buffer(1e-4, cap_style="square", join_style="mitre"))
    return out


def generate_power_diagram(
    boundary: Polygon,
    seeds: List[Tuple[float, float]],
    weights: List[float],
    resolution: float = 0.2,
    *,
    precomputed: Optional[PrecomputedGrid] = None,
    return_id_grid: bool = False,
) -> Union[List[BaseGeometry], Tuple[List[BaseGeometry], np.ndarray]]:
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if len(seeds) != len(weights):
        raise ValueError("seeds and weights must have the same length")

    def _empty(
        grid_shape: Tuple[int, ...] = (0, 0),
    ) -> Union[List[BaseGeometry], Tuple[List[BaseGeometry], np.ndarray]]:
        cells: List[BaseGeometry] = [Polygon() for _ in seeds] if seeds else []
        if return_id_grid:
            return cells, np.full(grid_shape, -1, dtype=np.int32)
        return cells

    if len(seeds) == 0:
        return _empty()

    if boundary.is_empty:
        return _empty()
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    if boundary.is_empty:
        return _empty()

    # --- Grid setup: use precomputed cache or generate from scratch ---
    if precomputed is not None:
        x0, y0, mask, px, py = precomputed
        if x0.size == 0 or y0.size == 0:
            return _empty()
        if not bool(np.any(mask)):
            return _empty(mask.shape)
    else:
        minx, miny, maxx, maxy = boundary.bounds
        x0 = np.arange(float(minx), float(maxx), float(resolution), dtype=float)
        y0 = np.arange(float(miny), float(maxy), float(resolution), dtype=float)
        if x0.size == 0 or y0.size == 0:
            return _empty()

        X0, Y0 = np.meshgrid(x0, y0)
        Xc = X0 + float(resolution) / 2.0
        Yc = Y0 + float(resolution) / 2.0

        mask = _mask_points_in_boundary(boundary, Xc, Yc)
        if not bool(np.any(mask)):
            return _empty(mask.shape)

        px = Xc[mask].astype(float)
        py = Yc[mask].astype(float)

    seeds_arr = np.asarray(seeds, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    labels = _assign_power_labels(px, py, seeds_arr, weights_arr)

    id_grid = np.full(mask.shape, -1, dtype=np.int32)
    id_grid[mask] = labels

    rects_by_id = _rectangles_by_id_from_grid(id_grid, x0, y0, float(resolution), int(seeds_arr.shape[0]))

    out: List[BaseGeometry] = []
    for i in range(int(seeds_arr.shape[0])):
        rects = rects_by_id.get(i, [])
        if not rects:
            out.append(Polygon())
            continue
        merged = unary_union(rects)
        clipped = merged.intersection(boundary)
        if clipped.is_empty:
            out.append(Polygon())
        else:
            out.append(clipped)

    if return_id_grid:
        return out, id_grid
    return out
