"""
axis_align.py

杞村榻愬伐鍏凤紙绠€鍖栫増锛?
MIQP 绠楁硶澶╃劧浜х敓姝ｄ氦甯冨眬锛屼笉鍐嶉渶瑕佸鏉傜殑姝ｄ氦鍖栧悗澶勭悊銆?浠呬繚鐣欑綉鏍煎榻愬姛鑳姐€?"""
from __future__ import annotations

import logging
from typing import List

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

logger = logging.getLogger(__name__)

try:
    from shapely.validation import make_valid  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore[assignment]


def snap_to_grid(
    cells: List[Polygon],
    grid_size: float = 0.1,
) -> List[Polygon]:
    """灏嗗杈瑰舰椤剁偣瀵归綈鍒扮綉鏍笺€?
    Args:
        cells: 澶氳竟褰㈠垪琛?        grid_size: 缃戞牸澶у皬锛堢背锛夛紝榛樿 10cm

    Returns:
        缃戞牸瀵归綈鍚庣殑澶氳竟褰㈠垪琛?    """
    if grid_size <= 0:
        return list(cells)

    results: List[Polygon] = []
    inv = 1.0 / grid_size

    for cell in cells:
        if cell.is_empty:
            results.append(cell)
            continue

        # 瀵归綈澶栫幆椤剁偣
        snapped_coords = [
            (round(x * inv) / inv, round(y * inv) / inv)
            for x, y in cell.exterior.coords
        ]

        try:
            poly = Polygon(snapped_coords)
            if not poly.is_valid:
                if make_valid is not None:
                    fixed = make_valid(poly)
                    if isinstance(fixed, Polygon):
                        poly = fixed
                    elif isinstance(fixed, (MultiPolygon, GeometryCollection)):
                        polys = [g for g in fixed.geoms if isinstance(g, Polygon) and (not g.is_empty)]
                        if polys:
                            poly = max(polys, key=lambda g: float(g.area))
            if poly.is_empty:
                results.append(cell)  # fallback to original
            else:
                results.append(poly)
        except Exception:
            results.append(cell)

    return results

