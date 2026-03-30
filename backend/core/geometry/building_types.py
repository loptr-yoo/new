from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

from shapely.geometry import MultiPolygon, Polygon


@dataclass(frozen=True)
class FloorSkeleton:
    boundary_polygon: Polygon
    core_tube_polygon: Polygon
    corridor_polygon: Union[Polygon, MultiPolygon]
    usable_islands: List[Polygon]
