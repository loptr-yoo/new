"""
axis_align.py

轴对齐工具（简化版）

MIQP 算法天然产生正交布局，不再需要复杂的正交化后处理。
仅保留网格对齐功能。
"""
from __future__ import annotations

import logging
from typing import List

from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def snap_to_grid(
    cells: List[Polygon],
    grid_size: float = 0.1,
) -> List[Polygon]:
    """将多边形顶点对齐到网格。

    Args:
        cells: 多边形列表
        grid_size: 网格大小（米），默认 10cm

    Returns:
        网格对齐后的多边形列表
    """
    if grid_size <= 0:
        return list(cells)

    results: List[Polygon] = []
    inv = 1.0 / grid_size

    for cell in cells:
        if cell.is_empty:
            results.append(cell)
            continue

        # 对齐外环顶点
        snapped_coords = [
            (round(x * inv) / inv, round(y * inv) / inv)
            for x, y in cell.exterior.coords
        ]

        try:
            poly = Polygon(snapped_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                results.append(cell)  # fallback to original
            else:
                results.append(poly)
        except Exception:
            results.append(cell)

    return results
