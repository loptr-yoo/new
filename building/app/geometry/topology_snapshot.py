"""
JSON-safe topology snapshots and area budgets for building generation.

The snapshot deliberately stores only primitive data, so it can pass through
logs, Pydantic models, or json.dumps without carrying Shapely objects.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

try:  # Shapely 2.x
    from shapely.validation import make_valid
except Exception:  # pragma: no cover - Shapely 1.x fallback
    make_valid = None  # type: ignore

from .room_spec import ZoneType
from .topology_generator import CoreTube, Corridor, Island, generate_rectangular_topology


class TopologySnapshotInvalidError(RuntimeError):
    """Raised when a JSON-safe topology snapshot cannot be restored."""


class TopologySnapshotMismatchError(RuntimeError):
    """Raised when a snapshot clearly belongs to a different floor setup."""


@dataclass
class FloorTopologySnapshot:
    floor_number: int
    floor_boundary_hash: str
    floor_boundary_ring: List[List[float]]
    core_ring: Optional[List[List[float]]] = None
    corridor_rings: List[List[List[float]]] = field(default_factory=list)
    island_rings: List[List[List[float]]] = field(default_factory=list)
    island_metadata: List[Dict[str, Any]] = field(default_factory=list)
    corridor_layout: str = "door_side"
    corridor_width: float = 2.0
    group_seed: Optional[int] = None
    floor_count: Optional[int] = None
    fixed_core_ring: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologySnapshot:
    floor_boundary_hash: str
    corridor_layout: str
    corridor_width: float
    floor_count: int
    floors: Dict[str, FloorTopologySnapshot] = field(default_factory=dict)
    fixed_core_ring: Optional[List[List[float]]] = None


@dataclass
class FloorAreaBudget:
    floor_number: int
    floor_total_area: float
    core_tube_area: float
    corridor_allowance_area: float
    total_island_area: float
    room_sum_min: float
    room_sum_recommended: float
    room_sum_max: float
    floor_boundary_hash: str


@dataclass
class BuildingAreaBudget:
    floors: Dict[str, FloorAreaBudget] = field(default_factory=dict)
    topology_snapshot: Optional[TopologySnapshot] = None


@dataclass
class LayoutFailureReport:
    floor_id: str
    failure_type: str
    message: str
    failure_kind: str = "unknown"
    room_ids: List[str] = field(default_factory=list)
    room_target_sum: float = 0.0
    island_area: float = 0.0
    max_gap_area: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeFloorTopology:
    core_tube: Optional[CoreTube]
    corridors: List[Corridor]
    islands: List[Island]


def _polygon_pieces(geom: Any) -> List[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if isinstance(p, Polygon) and not p.is_empty]
    geoms = getattr(geom, "geoms", None)
    if geoms is None:
        return []
    pieces: List[Polygon] = []
    for g in geoms:
        pieces.extend(_polygon_pieces(g))
    return pieces


def _largest_polygon(geom: Any) -> Optional[Polygon]:
    pieces = _polygon_pieces(geom)
    if not pieces:
        return None
    return max(pieces, key=lambda p: float(p.area))


def _clean_polygon(poly: Polygon) -> Polygon:
    geom: Any = poly
    try:
        if geom.is_empty:
            raise TopologySnapshotInvalidError("Snapshot polygon is empty")
        if not geom.is_valid:
            if make_valid is not None:
                geom = make_valid(geom)
            else:
                geom = geom.buffer(0)
        if not isinstance(geom, Polygon):
            geom = _largest_polygon(geom)
        if geom is None or geom.is_empty or not isinstance(geom, Polygon):
            raise TopologySnapshotInvalidError("Snapshot polygon is not recoverable")
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not isinstance(geom, Polygon):
            geom = _largest_polygon(geom)
        if geom is None or geom.is_empty or not isinstance(geom, Polygon):
            raise TopologySnapshotInvalidError("Snapshot polygon cleanup produced no Polygon")
        return geom
    except TopologySnapshotInvalidError:
        raise
    except Exception as exc:
        raise TopologySnapshotInvalidError(f"Invalid snapshot polygon: {exc}") from exc


def polygon_to_ring(poly: BaseGeometry) -> List[List[float]]:
    """Serialize a polygon exterior ring to JSON-safe math-coordinate points."""
    if poly is None or poly.is_empty:
        raise TopologySnapshotInvalidError("Cannot snapshot an empty polygon")
    if not isinstance(poly, Polygon):
        largest = _largest_polygon(poly)
        if largest is None:
            raise TopologySnapshotInvalidError("Cannot snapshot a non-polygon geometry")
        poly = largest
    coords = [[float(x), float(y)] for x, y in poly.exterior.coords]
    if coords and coords[0] != coords[-1]:
        coords.append(list(coords[0]))
    if len(coords) < 4:
        raise TopologySnapshotInvalidError("Polygon ring needs at least 4 coordinates")
    return coords


def ring_to_polygon(ring: Sequence[Sequence[float]]) -> Polygon:
    """Restore a Polygon from JSON coordinates and defensively legalize it."""
    try:
        coords = [(float(p[0]), float(p[1])) for p in ring]
    except Exception as exc:
        raise TopologySnapshotInvalidError(f"Bad ring coordinates: {exc}") from exc
    if len(coords) < 4:
        raise TopologySnapshotInvalidError("Polygon ring needs at least 4 coordinates")
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return _clean_polygon(Polygon(coords))


def floor_boundary_hash(poly: Polygon) -> str:
    """Stable hash for floor boundaries, rounded to 1mm to absorb float drift."""
    clean = _clean_polygon(poly)
    rounded = [(round(float(x), 3), round(float(y), 3)) for x, y in clean.exterior.coords]
    payload = json.dumps(rounded, separators=(",", ":"), ensure_ascii=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _corridor_from_polygon(idx: int, poly: Polygon, width: float) -> Corridor:
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    if (maxx - minx) >= (maxy - miny):
        cy = (miny + maxy) / 2.0
        centerline = LineString([(minx, cy), (maxx, cy)])
        orientation = "horizontal"
    else:
        cx = (minx + maxx) / 2.0
        centerline = LineString([(cx, miny), (cx, maxy)])
        orientation = "vertical"
    c = Corridor(id=f"corridor_snapshot_{idx}", centerline=centerline, width=float(width), orientation=orientation)
    c.polygon = poly
    return c


def core_tube_from_polygon(poly: Optional[Polygon]) -> Optional[CoreTube]:
    if poly is None:
        return None
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    core = CoreTube(
        polygon=poly,
        center=((minx + maxx) / 2.0, (miny + maxy) / 2.0),
        width=max(0.0, maxx - minx),
        depth=max(0.0, maxy - miny),
    )
    try:
        core.build_subzones_from_bounds()
    except Exception:
        pass
    return core


def core_tube_from_ring(ring: Optional[Sequence[Sequence[float]]]) -> Optional[CoreTube]:
    if ring is None:
        return None
    return core_tube_from_polygon(ring_to_polygon(ring))


def validate_snapshot_for_floor(
    snapshot: FloorTopologySnapshot,
    floor_boundary: Polygon,
    *,
    corridor_layout: str,
    corridor_width: float,
    floor_count: Optional[int] = None,
) -> None:
    expected_hash = floor_boundary_hash(floor_boundary)
    if str(snapshot.floor_boundary_hash) != str(expected_hash):
        raise TopologySnapshotMismatchError("Topology snapshot floor boundary hash mismatch")
    if str(snapshot.corridor_layout or "").lower() != str(corridor_layout or "").lower():
        raise TopologySnapshotMismatchError(
            "Topology snapshot corridor layout mismatch: "
            f"snapshot={snapshot.corridor_layout!r}, runtime={corridor_layout!r}. "
            "Ensure upstream sanitization is consistent."
        )
    if abs(float(snapshot.corridor_width) - float(corridor_width)) > 1e-3:
        raise TopologySnapshotMismatchError(
            "Topology snapshot corridor width mismatch: "
            f"snapshot={float(snapshot.corridor_width):.2f}m, "
            f"runtime={float(corridor_width):.2f}m. "
            "Ensure upstream sanitization is consistent."
        )
    if floor_count is not None and snapshot.floor_count is not None and int(snapshot.floor_count) != int(floor_count):
        raise TopologySnapshotMismatchError("Topology snapshot floor count mismatch")


def snapshot_floor_to_runtime(snapshot: FloorTopologySnapshot) -> RuntimeFloorTopology:
    core_ring = snapshot.core_ring or snapshot.fixed_core_ring
    core = core_tube_from_ring(core_ring) if core_ring else None
    corridors = [
        _corridor_from_polygon(i, ring_to_polygon(r), snapshot.corridor_width)
        for i, r in enumerate(snapshot.corridor_rings)
    ]
    islands = []
    for i, ring in enumerate(snapshot.island_rings):
        island = Island(id=f"island_{i}", polygon=ring_to_polygon(ring))
        meta = snapshot.island_metadata[i] if i < len(snapshot.island_metadata) else {}
        island.has_exterior_wall = bool(meta.get("has_exterior_wall", island.has_exterior_wall))
        island.exterior_walls = [str(v) for v in meta.get("exterior_walls", island.exterior_walls) or []]
        island.corridor_edges = [str(v) for v in meta.get("corridor_edges", island.corridor_edges) or []]
        island.distance_to_entrance = float(meta.get("distance_to_entrance", island.distance_to_entrance) or 0.0)
        island.distance_to_core = float(meta.get("distance_to_core", island.distance_to_core) or 0.0)
        zone_value = str(meta.get("suggested_zone", getattr(island.suggested_zone, "value", island.suggested_zone)) or "public")
        try:
            island.suggested_zone = ZoneType(zone_value)
        except Exception:
            pass
        islands.append(island)
    return RuntimeFloorTopology(core_tube=core, corridors=corridors, islands=islands)


def snapshot_to_runtime_topology(snapshot: TopologySnapshot) -> Dict[str, RuntimeFloorTopology]:
    return {fid: snapshot_floor_to_runtime(floor) for fid, floor in snapshot.floors.items()}


def floor_snapshot_from_runtime(
    *,
    floor_number: int,
    floor_boundary: Polygon,
    core_tube: Optional[CoreTube],
    corridors: Iterable[Corridor],
    islands: Iterable[Island],
    corridor_layout: str,
    corridor_width: float,
    group_seed: Optional[int] = None,
    floor_count: Optional[int] = None,
    fixed_core_ring: Optional[List[List[float]]] = None,
) -> FloorTopologySnapshot:
    return FloorTopologySnapshot(
        floor_number=int(floor_number),
        floor_boundary_hash=floor_boundary_hash(floor_boundary),
        floor_boundary_ring=polygon_to_ring(floor_boundary),
        core_ring=polygon_to_ring(core_tube.polygon) if core_tube is not None else None,
        corridor_rings=[polygon_to_ring(c.polygon) for c in corridors if c.polygon is not None and not c.polygon.is_empty],
        island_rings=[polygon_to_ring(i.polygon) for i in islands if i.polygon is not None and not i.polygon.is_empty],
        island_metadata=[
            {
                "has_exterior_wall": bool(getattr(i, "has_exterior_wall", False)),
                "exterior_walls": list(getattr(i, "exterior_walls", []) or []),
                "corridor_edges": list(getattr(i, "corridor_edges", []) or []),
                "distance_to_entrance": float(getattr(i, "distance_to_entrance", 0.0) or 0.0),
                "distance_to_core": float(getattr(i, "distance_to_core", 0.0) or 0.0),
                "suggested_zone": str(getattr(getattr(i, "suggested_zone", ""), "value", getattr(i, "suggested_zone", "")) or ""),
            }
            for i in islands
            if i.polygon is not None and not i.polygon.is_empty
        ],
        corridor_layout=str(corridor_layout),
        corridor_width=float(corridor_width),
        group_seed=group_seed,
        floor_count=floor_count,
        fixed_core_ring=fixed_core_ring,
    )


def compute_building_area_budget(
    *,
    floor_boundary: Polygon,
    floors: Sequence[Any],
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "door_side",
    base_seed: Optional[int] = None,
    fixed_core_tube: Optional[CoreTube] = None,
) -> BuildingAreaBudget:
    """Compute physical island budgets from a deterministic topology pass."""
    floor_count = len(floors)
    fixed_core: Optional[CoreTube] = fixed_core_tube
    fixed_core_ring: Optional[List[List[float]]] = None
    floor_snaps: Dict[str, FloorTopologySnapshot] = {}
    budgets: Dict[str, FloorAreaBudget] = {}

    for idx, floor in enumerate(floors):
        floor_number = int(getattr(floor, "floor_number", idx + 1) or (idx + 1))
        group_seed = None if base_seed is None else int(base_seed) + floor_number
        core, corridors, islands = generate_rectangular_topology(
            floor_boundary=floor_boundary,
            corridor_width=float(corridor_width),
            core_area_ratio=float(core_area_ratio),
            corridor_layout=corridor_layout,
            core_tube_override=fixed_core,
            group_seed=group_seed,
            force_corridor_boundary_contact=(floor_number == 1),
        )
        if fixed_core is None:
            fixed_core = core
            fixed_core_ring = polygon_to_ring(core.polygon)
        elif fixed_core_ring is None and fixed_core is not None:
            fixed_core_ring = polygon_to_ring(fixed_core.polygon)

        fid = f"F{floor_number}"
        snap = floor_snapshot_from_runtime(
            floor_number=floor_number,
            floor_boundary=floor_boundary,
            core_tube=core,
            corridors=corridors,
            islands=islands,
            corridor_layout=corridor_layout,
            corridor_width=float(corridor_width),
            group_seed=group_seed,
            floor_count=floor_count,
            fixed_core_ring=fixed_core_ring,
        )
        floor_snaps[fid] = snap
        island_area = float(sum(float(i.area) for i in islands))
        core_area = float(core.polygon.area) if core is not None else 0.0
        corridor_area = float(unary_union([c.polygon for c in corridors]).area) if corridors else 0.0
        budgets[fid] = FloorAreaBudget(
            floor_number=floor_number,
            floor_total_area=float(floor_boundary.area),
            core_tube_area=core_area,
            corridor_allowance_area=corridor_area,
            total_island_area=island_area,
            room_sum_min=0.82 * island_area,
            room_sum_recommended=0.88 * island_area,
            room_sum_max=0.92 * island_area,
            floor_boundary_hash=floor_boundary_hash(floor_boundary),
        )

    snapshot = TopologySnapshot(
        floor_boundary_hash=floor_boundary_hash(floor_boundary),
        corridor_layout=str(corridor_layout),
        corridor_width=float(corridor_width),
        floor_count=floor_count,
        floors=floor_snaps,
        fixed_core_ring=fixed_core_ring,
    )
    return BuildingAreaBudget(floors=budgets, topology_snapshot=snapshot)


def repair_building_allocation_with_budget(
    *,
    failure: LayoutFailureReport,
    budget: Optional[BuildingAreaBudget] = None,
    current_rooms: Optional[Sequence[Any]] = None,
) -> str:
    """Build a compact semantic-repair instruction for the LLM planner."""
    lines = [
        f"Floor {failure.floor_id} failed during layout generation.",
        f"Failure type: {failure.failure_type}.",
        f"Reason: {failure.message}",
    ]
    if failure.room_target_sum or failure.island_area:
        lines.append(
            f"Current room target sum={failure.room_target_sum:.2f}m2; "
            f"physical island capacity={failure.island_area:.2f}m2."
        )
    if budget and failure.floor_id in budget.floors:
        b = budget.floors[failure.floor_id]
        lines.append(
            "Use this hard budget: "
            f"room target sum should be {b.room_sum_min:.2f}-{b.room_sum_max:.2f}m2 "
            f"(recommended {b.room_sum_recommended:.2f}m2)."
        )
    if current_rooms:
        room_bits = []
        for r in current_rooms:
            rid = str(getattr(r, "room_id", "") or getattr(r, "id", "") or "?")
            rtype = str(getattr(r, "room_type", "") or "?")
            area = float(getattr(r, "target_area", 0.0) or 0.0)
            room_bits.append(f"{rid}:{rtype}:{area:.1f}m2")
        lines.append("Current rooms: " + ", ".join(room_bits))

    gap_pieces = (failure.metadata or {}).get("gap_pieces") or []
    if gap_pieces:
        bits = []
        for gp in list(gap_pieces)[:3]:
            if isinstance(gp, dict):
                area = float(gp.get("area", 0.0) or 0.0)
                bbox = gp.get("bbox", [])
                width = gp.get("width")
                height = gp.get("height")
                aspect = gp.get("aspect_ratio")
                fill_rate = gp.get("fill_rate")
                hint = str(gp.get("shape_hint", "") or "")
                repair_advice = str(gp.get("repair_advice", "") or "")
                shape_bits = []
                if hint:
                    shape_bits.append(hint)
                if width is not None and height is not None:
                    shape_bits.append(f"width={float(width):.2f}m height={float(height):.2f}m")
                if aspect is not None:
                    shape_bits.append(f"aspect={float(aspect):.2f}:1")
                if fill_rate is not None:
                    shape_bits.append(f"fill_rate={float(fill_rate):.1%}")
                if not repair_advice:
                    if hint == "狭长型":
                        repair_advice = "Prefer corridor expansion, narrow utility, or elongated storage. Do not assign square rooms."
                    elif hint == "矩形偏长":
                        repair_advice = "Prefer storage/utility or a split/elongated support space."
                    elif hint == "方正型":
                        repair_advice = "Suitable for small storage, utility, or compact bathroom if it fits the budget."
                    elif hint:
                        repair_advice = "Do not force a square room unless the geometry is truly regular."
                suggestion = f" {repair_advice}" if repair_advice else ""
                shape = f" ({', '.join(shape_bits)})" if shape_bits else ""
                bits.append(f"area={area:.2f}m2 bbox={bbox}{shape}.{suggestion}")
        if bits:
            lines.append("Detected macro gaps: " + "; ".join(bits))

    synthetic_rooms = (failure.metadata or {}).get("synthetic_rooms") or []
    if synthetic_rooms:
        bits = []
        for sr in list(synthetic_rooms)[:8]:
            if isinstance(sr, dict):
                bits.append(
                    f"{sr.get('room_id', '?')}:storage:{float(sr.get('target_area', 0.0) or 0.0):.1f}m2"
                )
        if bits:
            lines.append("Backend synthetic rooms already present: " + ", ".join(bits))

    kind = str(failure.failure_kind or "unknown").lower()
    if kind == "unknown" and float(failure.max_gap_area or 0.0) > 0.0:
        kind = "coverage"
    if kind == "capacity":
        lines.append("Repair by shrinking or removing rooms; do not increase total target area.")
    elif kind == "coverage":
        lines.append(
            "Macro void repair: do not shrink, delete, or reduce explicit user-requested rooms. "
            "do not blindly enlarge existing rooms. "
            "Prefer adding small storage/utility rooms or splitting oversized rooms so the leftover space is occupied. "
            "Keep the final room target sum inside the frozen budget interval."
        )
    elif kind == "infeasible":
        lines.append("Repair by relaxing extreme room areas/aspect ratios and avoiding over-constrained adjacency.")
    elif kind == "assignment":
        lines.append(
            "Assignment repair: the backend produced a large usable island but could not move a legal room cluster into it. "
            "Do not change the frozen topology snapshot, core, corridor budget, or successful floors. "
            "Repair only this failed floor by splitting oversized rooms, reducing rigid adjacency_required links, "
            "or adding small independent functional rooms so room clusters can be distributed across islands. "
            "Avoid forcing mutually forbidden rooms into the same zone."
        )
    elif kind == "reachability":
        unreachable = (failure.metadata or {}).get("unreachable_rooms") or (failure.metadata or {}).get("door_fallback", {}).get("unreachable_after") or []
        if unreachable:
            lines.append("Dead rooms after door fallback: " + ", ".join(str(x) for x in unreachable))
        lines.append(
            "Reachability repair: ordinary rooms must physically touch corridor, hall, living, dining, or another public circulation space. "
            "Ensuite bathrooms, walk-in closets, and private storage may connect only to their required parent bedroom/room, "
            "but that parent must itself be reachable from the public circulation network. "
            "Do not change the frozen topology snapshot, core, corridor budget, or successful floors."
        )
    elif kind == "topology":
        lines.append(
            "Topology repair: keep explicit rooms and budget stable, but adjust room mix/adjacency so all spaces can be placed."
        )
    else:
        lines.append("Repair only the failed floor rooms; keep topology snapshot, core, and corridor budget unchanged.")
    return "\n".join(lines)
