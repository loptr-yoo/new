from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union


def _poly(coords: List[List[float]]) -> Polygon:
    if not coords:
        return Polygon()
    pts = [(float(x), float(y)) for x, y in coords]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    try:
        return Polygon(pts)
    except Exception:
        return Polygon()


def _load_floor(path: Path, floor_id: str) -> Tuple[Polygon, List[Polygon], List[Polygon], List[Polygon]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "building" in data:
        floors: Dict[str, Any] = data["building"]["floors"]
        floor: Dict[str, Any] = floors[floor_id]
        slab = _poly(floor["floor_slab"]["polygon"])
        rooms = [_poly(r.get("polygon") or []) for r in floor.get("rooms") or []]
        corridors = [_poly(c.get("polygon") or []) for c in floor.get("corridors") or []]
    else:
        elems = data.get("elements") or []
        slab_poly = next((e.get("polygon") for e in elems if e.get("type") == "floor_slab"), [])
        slab = _poly(slab_poly or [])
        rooms = [_poly(e.get("polygon") or []) for e in elems if str(e.get("type") or "") not in ("floor_slab", "corridor", "partition_wall", "exterior_wall", "wall", "door", "window")]
        corridors = [_poly(e.get("polygon") or []) for e in elems if e.get("type") == "corridor"]
    core_polys: List[Polygon] = []
    core = data.get("core_tube") or {}
    if isinstance(core, dict):
        for k in ("elevator", "staircase", "elevator_hall", "elevator_shaft", "staircase_hall", "staircase_shaft"):
            info = core.get(k)
            if isinstance(info, dict):
                core_polys.append(_poly(info.get("polygon") or []))
    return slab, rooms, corridors, core_polys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--floor", default="F1")
    args = ap.parse_args()

    slab, rooms, corridors, core = _load_floor(args.json_path, args.floor)
    union = unary_union([p for p in (rooms + corridors + core) if (not p.is_empty)])
    hole = slab.difference(union)
    outside = union.difference(slab)

    slab_area = float(slab.area)
    hole_area = float(hole.area) if not hole.is_empty else 0.0
    outside_area = float(outside.area) if not outside.is_empty else 0.0
    ratio = hole_area / slab_area if slab_area > 1e-9 else 0.0

    print(
        f"{args.json_path.name} {args.floor}: "
        f"slab={slab_area:.3f} "
        f"hole={hole_area:.3f} ({ratio:.3%}) "
        f"outside={outside_area:.6f}"
    )


if __name__ == "__main__":
    main()
