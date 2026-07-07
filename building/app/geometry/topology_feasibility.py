"""Topology variant metrics and cluster-island feasibility diagnostics.

This module is intentionally diagnostic-only. It produces JSON-serializable
resource profiles for future topology assignment, but it does not choose a
topology or mutate solver inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon, box
from shapely.ops import unary_union

from .core_contracts import CORE_OVERLAP_EPSILON_AREA
from .room_spec import RoomSpec, ZoneType


EPSILON = 1e-6
LARGE_SLOT_AREA = 18.0
LARGE_SLOT_MIN_SIDE = 2.7
MEDIUM_SLOT_AREA = 8.0
MEDIUM_SLOT_MIN_SIDE = 1.8
SMALL_SLOT_AREA = 3.0
SMALL_SLOT_MIN_SIDE = 1.2


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _bounds4(bounds: Sequence[float]) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def _as_lines(geom: Any) -> List[LineString]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, LineString):
        return [geom] if float(geom.length) > EPSILON else []
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if float(g.length) > EPSILON]
    if isinstance(geom, GeometryCollection) or hasattr(geom, "geoms"):
        out: List[LineString] = []
        for g in geom.geoms:
            out.extend(_as_lines(g))
        return out
    return []


def _line_lengths(geom: Any) -> List[float]:
    return [float(line.length) for line in _as_lines(geom) if float(line.length) > EPSILON]


def _slot_count(lengths: Iterable[float], *, threshold: float, spacing: float) -> Tuple[int, float, float, int]:
    usable = [float(v) for v in lengths if float(v) + EPSILON >= float(threshold)]
    count = sum(max(0, int(math.floor(v / max(float(spacing), EPSILON)))) for v in usable)
    return count, float(sum(usable)), max(usable, default=0.0), len(usable)


def _polygon_vertex_count(poly: Polygon) -> int:
    try:
        return max(0, len(list(poly.exterior.coords)) - 1)
    except Exception:
        return 0


def _reflex_vertex_proxy(poly: Polygon) -> int:
    # Conservative proxy used elsewhere in grid growth: any vertices beyond a
    # rectangle are treated as notch/reflex risk.
    return max(0, _polygon_vertex_count(poly) - 4)


def _compactness(poly: Polygon) -> float:
    perimeter = float(getattr(poly, "length", 0.0) or 0.0)
    if perimeter <= EPSILON:
        return 0.0
    return _clamp((4.0 * math.pi * float(poly.area)) / (perimeter * perimeter))


def _bbox_waste_ratio(poly: Polygon) -> float:
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    bbox_area = max(0.0, (maxx - minx) * (maxy - miny))
    if bbox_area <= EPSILON:
        return 1.0
    return _clamp(1.0 - float(poly.area) / bbox_area)


def _coordinate_grid(poly: Polygon, *, max_coords: int = 42) -> Tuple[List[float], List[float]]:
    xs = {float(poly.bounds[0]), float(poly.bounds[2])}
    ys = {float(poly.bounds[1]), float(poly.bounds[3])}
    rings = [poly.exterior] + list(poly.interiors)
    for ring in rings:
        for x, y in ring.coords:
            xs.add(round(float(x), 6))
            ys.add(round(float(y), 6))
    x_list = sorted(xs)
    y_list = sorted(ys)
    if len(x_list) > max_coords:
        minx, maxx = x_list[0], x_list[-1]
        step = (maxx - minx) / float(max_coords - 1)
        x_list = [minx + step * i for i in range(max_coords)]
    if len(y_list) > max_coords:
        miny, maxy = y_list[0], y_list[-1]
        step = (maxy - miny) / float(max_coords - 1)
        y_list = [miny + step * i for i in range(max_coords)]
    return x_list, y_list


def largest_axis_aligned_rect_estimate(poly: Polygon) -> Tuple[float, float, float]:
    """Return a conservative contained axis-aligned rectangle estimate.

    For rectilinear island polygons this grid is usually exact because it is
    derived from polygon vertices. For more complex polygons it remains a
    conservative heuristic and is only hard-rejecting with tolerance upstream.
    """

    if poly is None or poly.is_empty:
        return 0.0, 0.0, 0.0
    xs, ys = _coordinate_grid(poly)
    best_area = 0.0
    best_w = 0.0
    best_h = 0.0
    for xi, x1 in enumerate(xs[:-1]):
        for x2 in xs[xi + 1:]:
            width = float(x2 - x1)
            if width <= EPSILON:
                continue
            for yi, y1 in enumerate(ys[:-1]):
                for y2 in ys[yi + 1:]:
                    height = float(y2 - y1)
                    area = width * height
                    if area <= best_area + EPSILON:
                        continue
                    rect = box(x1, y1, x2, y2)
                    try:
                        if poly.covers(rect):
                            best_area = area
                            best_w = width
                            best_h = height
                    except Exception:
                        continue
    return best_area, best_w, best_h


def _room_type(room: RoomSpec) -> str:
    return str(getattr(room, "room_type", "") or "").lower()


def _is_service_room(room: RoomSpec) -> bool:
    kind = _room_type(room)
    zone = getattr(room, "zone", None)
    zone_value = getattr(zone, "value", str(zone)).lower() if zone is not None else ""
    return zone == ZoneType.SERVICE or zone_value == "service" or kind in {
        "bathroom",
        "toilet",
        "storage",
        "utility",
        "closet",
        "wardrobe",
        "void",
    }


def _is_public_room(room: RoomSpec) -> bool:
    kind = _room_type(room)
    zone = getattr(room, "zone", None)
    zone_value = getattr(zone, "value", str(zone)).lower() if zone is not None else ""
    return zone == ZoneType.PUBLIC or zone_value == "public" or kind in {
        "living_room",
        "living",
        "dining_room",
        "dining",
        "kitchen",
        "reception",
    }


def _is_private_room(room: RoomSpec) -> bool:
    return (not _is_public_room(room)) and (not _is_service_room(room))


def _needs_hard_window(room: RoomSpec) -> bool:
    kind = _room_type(room)
    return bool(getattr(room, "needs_window", False)) and kind in {
        "bedroom",
        "master_bedroom",
        "living_room",
        "living",
        "study",
        "office",
        "kitchen",
    }


def _room_type_min_side(room: RoomSpec) -> float:
    kind = _room_type(room)
    if "bath" in kind or kind in {"toilet"}:
        return 1.5
    if "kitchen" in kind:
        return 2.2
    if "bed" in kind:
        return 2.7
    if kind in {"living_room", "living", "dining_room", "dining", "reception"}:
        return 3.2
    return 1.2 if _is_service_room(room) else 2.5


def estimate_room_min_dimensions(room: RoomSpec) -> Tuple[float, float]:
    area = max(0.01, float(getattr(room, "target_area", 0.0) or 0.0))
    ar_min, ar_max = getattr(room, "aspect_ratio_range", (0.5, 2.0)) or (0.5, 2.0)
    aspect_max = max(float(ar_max or 2.0), 1.0)
    short_side = math.sqrt(area / aspect_max)
    floor_side = _room_type_min_side(room)
    min_width = max(short_side, floor_side, float(getattr(room, "min_width", 0.0) or 0.0))
    min_depth = max(short_side, floor_side, float(getattr(room, "min_depth", 0.0) or 0.0))
    return float(min_width), float(min_depth)


@dataclass
class IslandMetrics:
    variant_id: str
    island_id: str
    area: float
    effective_capacity_area: float
    bounds: Tuple[float, float, float, float]
    compactness: float
    reflex_vertex_count: int
    reflex_complexity: float
    bbox_waste_ratio: float
    notch_complexity: float
    facade_len: float
    facade_total_usable_len: float
    window_slot_count: int
    max_single_facade_edge_len: float
    corridor_frontage_len: float
    access_edge_total_len: float
    access_total_usable_len: float
    corridor_door_slot_count: int
    max_single_corridor_edge_len: float
    largest_empty_rect_estimate: float
    largest_empty_rect_width: float
    largest_empty_rect_height: float
    slot_count_large: int
    slot_count_medium: int
    slot_count_small: int
    core_overlap_area: float
    forbidden_overlap_area: float
    core_contract_id: str = ""
    core_union_hash: str = ""
    valid: bool = True
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "island_id": self.island_id,
            "area": _r(self.area),
            "effective_capacity_area": _r(self.effective_capacity_area),
            "bounds": [_r(v) for v in self.bounds],
            "compactness": _r(self.compactness),
            "reflex_vertex_count": int(self.reflex_vertex_count),
            "reflex_complexity": _r(self.reflex_complexity),
            "bbox_waste_ratio": _r(self.bbox_waste_ratio),
            "notch_complexity": _r(self.notch_complexity),
            "facade_len": _r(self.facade_len),
            "facade_total_usable_len": _r(self.facade_total_usable_len),
            "window_slot_count": int(self.window_slot_count),
            "max_single_facade_edge_len": _r(self.max_single_facade_edge_len),
            "corridor_frontage_len": _r(self.corridor_frontage_len),
            "access_edge_total_len": _r(self.access_edge_total_len),
            "access_total_usable_len": _r(self.access_total_usable_len),
            "corridor_door_slot_count": int(self.corridor_door_slot_count),
            "max_single_corridor_edge_len": _r(self.max_single_corridor_edge_len),
            "largest_empty_rect_estimate": _r(self.largest_empty_rect_estimate),
            "largest_empty_rect_width": _r(self.largest_empty_rect_width),
            "largest_empty_rect_height": _r(self.largest_empty_rect_height),
            "slot_count_large": int(self.slot_count_large),
            "slot_count_medium": int(self.slot_count_medium),
            "slot_count_small": int(self.slot_count_small),
            "core_overlap_area": _r(self.core_overlap_area),
            "forbidden_overlap_area": _r(self.forbidden_overlap_area),
            "core_contract_id": self.core_contract_id,
            "core_union_hash": self.core_union_hash,
            "valid": bool(self.valid),
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass
class ClusterMetrics:
    cluster_id: str
    room_ids: List[str]
    target_area_sum: float
    min_area_sum: float
    max_area_sum: float
    largest_room_area: float
    largest_room_type: str
    largest_room_min_width_estimate: float
    largest_room_min_depth_estimate: float
    room_count: int
    large_room_count: int
    medium_room_count: int
    small_room_count: int
    needs_window_count: int
    needs_corridor_access_count: int
    public_room_count: int
    private_room_count: int
    service_room_count: int
    adjacency_degree: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "room_ids": list(self.room_ids),
            "target_area_sum": _r(self.target_area_sum),
            "min_area_sum": _r(self.min_area_sum),
            "max_area_sum": _r(self.max_area_sum),
            "largest_room_area": _r(self.largest_room_area),
            "largest_room_type": self.largest_room_type,
            "largest_room_min_width_estimate": _r(self.largest_room_min_width_estimate),
            "largest_room_min_depth_estimate": _r(self.largest_room_min_depth_estimate),
            "room_count": int(self.room_count),
            "large_room_count": int(self.large_room_count),
            "medium_room_count": int(self.medium_room_count),
            "small_room_count": int(self.small_room_count),
            "needs_window_count": int(self.needs_window_count),
            "needs_corridor_access_count": int(self.needs_corridor_access_count),
            "public_room_count": int(self.public_room_count),
            "private_room_count": int(self.private_room_count),
            "service_room_count": int(self.service_room_count),
            "adjacency_degree": int(self.adjacency_degree),
        }


@dataclass
class ClusterIslandFeasibility:
    variant_id: str
    cluster_id: str
    island_id: str
    hard_feasible: bool
    feasibility_score: float
    rejection_reasons: List[str]
    area_fit_ratio: float
    largest_room_fit_ratio: float
    corridor_fit_ratio: float
    facade_fit_ratio: float
    slot_fit_ratio: float
    capacity_margin: float
    capacity_margin_ratio: float
    shape_penalty: float
    access_penalty: float
    window_penalty: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "cluster_id": self.cluster_id,
            "island_id": self.island_id,
            "hard_feasible": bool(self.hard_feasible),
            "feasibility_score": _r(self.feasibility_score),
            "rejection_reasons": list(self.rejection_reasons),
            "area_fit_ratio": _r(self.area_fit_ratio),
            "largest_room_fit_ratio": _r(self.largest_room_fit_ratio),
            "corridor_fit_ratio": _r(self.corridor_fit_ratio),
            "facade_fit_ratio": _r(self.facade_fit_ratio),
            "slot_fit_ratio": _r(self.slot_fit_ratio),
            "capacity_margin": _r(self.capacity_margin),
            "capacity_margin_ratio": _r(self.capacity_margin_ratio),
            "shape_penalty": _r(self.shape_penalty),
            "access_penalty": _r(self.access_penalty),
            "window_penalty": _r(self.window_penalty),
        }


@dataclass
class TopologyVariant:
    variant_id: str
    seed: int
    is_primary: bool
    primary_compatible: bool
    variant_profile: Dict[str, Any]
    corridor_skeleton: Sequence[Any] = field(default_factory=list)
    candidate_islands: Sequence[Any] = field(default_factory=list)
    island_metrics: List[IslandMetrics] = field(default_factory=list)
    feasibility_matrix: List[ClusterIslandFeasibility] = field(default_factory=list)
    corridor_access_edges: List[Dict[str, Any]] = field(default_factory=list)
    facade_edges: List[Dict[str, Any]] = field(default_factory=list)
    core_docking_candidates: List[Dict[str, Any]] = field(default_factory=list)
    core_contract_id: str = ""
    core_union_hash: str = ""
    corridor_area: float = 0.0
    valid: bool = True
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "seed": int(self.seed),
            "is_primary": bool(self.is_primary),
            "primary_compatible": bool(self.primary_compatible),
            "variant_profile": dict(self.variant_profile),
            "islands": [m.to_dict() for m in self.island_metrics],
            "feasibility": [f.to_dict() for f in self.feasibility_matrix],
            "corridor_access_edges": list(self.corridor_access_edges),
            "facade_edges": list(self.facade_edges),
            "core_docking_candidates": list(self.core_docking_candidates),
            "core_contract_id": self.core_contract_id,
            "core_union_hash": self.core_union_hash,
            "corridor_area": _r(self.corridor_area),
            "valid": bool(self.valid),
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass
class TopologyFeasibilityReport:
    topology_seed_list: List[int]
    primary_variant_id: str
    variants: List[TopologyVariant]
    cluster_metrics: List[ClusterMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_seed_list": list(self.topology_seed_list),
            "primary_variant_id": self.primary_variant_id,
            "topology_variants": [v.to_dict() for v in self.variants],
            "cluster_metrics": [c.to_dict() for c in self.cluster_metrics],
            "cluster_island_feasibility": [
                f.to_dict()
                for v in self.variants
                for f in v.feasibility_matrix
            ],
        }


def build_island_metrics(
    *,
    variant_id: str,
    islands: Sequence[Any],
    corridors: Sequence[Any],
    floor_boundary: Polygon,
    core_union: Optional[Any] = None,
    forbidden_union: Optional[Any] = None,
    min_door_width: float = 0.8,
    min_anchor_frontage: float = 1.0,
    core_contract_id: str = "",
    core_union_hash: str = "",
) -> List[IslandMetrics]:
    corridor_polys = [getattr(c, "polygon", None) for c in corridors or [] if getattr(c, "polygon", None) is not None]
    corridor_union = unary_union(corridor_polys) if corridor_polys else GeometryCollection()
    metrics: List[IslandMetrics] = []
    for island in islands or []:
        poly = getattr(island, "polygon", None)
        if poly is None or getattr(poly, "is_empty", True):
            continue
        area = float(poly.area)
        compact = _compactness(poly)
        reflex_count = _reflex_vertex_proxy(poly)
        vertex_count = max(1, _polygon_vertex_count(poly))
        reflex_complexity = _clamp(reflex_count / float(vertex_count))
        bbox_waste = _bbox_waste_ratio(poly)
        notch_complexity = _clamp(max(reflex_complexity, bbox_waste))

        facade_lengths = _line_lengths(poly.boundary.intersection(floor_boundary.exterior))
        window_slots, facade_usable, max_facade, facade_edge_count = _slot_count(
            facade_lengths,
            threshold=min_anchor_frontage,
            spacing=min_anchor_frontage,
        )

        access_geom = poly.boundary.intersection(corridor_union.boundary)
        access_lengths = _line_lengths(access_geom)
        if not access_lengths:
            access_lengths = _line_lengths(poly.boundary.intersection(corridor_union.buffer(0.02).boundary))
        door_slots, access_usable, max_access, access_edge_count = _slot_count(
            access_lengths,
            threshold=min_door_width,
            spacing=min_door_width,
        )

        ler_area, ler_w, ler_h = largest_axis_aligned_rect_estimate(poly)
        ler_min_side = min(ler_w, ler_h)
        access_fragmentation = 0.0
        if access_usable > EPSILON:
            access_fragmentation = _clamp(1.0 - max_access / access_usable)
        packing_factor = 0.92
        packing_factor -= 0.10 * notch_complexity
        packing_factor -= 0.08 * max(0.0, 0.65 - compact)
        packing_factor -= 0.05 * access_fragmentation
        packing_factor = _clamp(packing_factor, 0.65, 0.92)
        effective_capacity = area * packing_factor

        def _slot_capacity(threshold_area: float, min_side: float) -> int:
            if ler_min_side + EPSILON < min_side:
                return 0
            return max(0, int(math.floor(effective_capacity / threshold_area)))

        core_overlap = 0.0
        if core_union is not None and not getattr(core_union, "is_empty", True):
            try:
                core_overlap = float(poly.intersection(core_union).area)
            except Exception:
                core_overlap = float("inf")
        forbidden_overlap = 0.0
        if forbidden_union is not None and not getattr(forbidden_union, "is_empty", True):
            try:
                forbidden_overlap = float(poly.intersection(forbidden_union).area)
            except Exception:
                forbidden_overlap = float("inf")
        rejection_reasons: List[str] = []
        if core_overlap > CORE_OVERLAP_EPSILON_AREA:
            rejection_reasons.append("core_overlap")

        metrics.append(
            IslandMetrics(
                variant_id=variant_id,
                island_id=str(getattr(island, "id", "")),
                area=area,
                effective_capacity_area=effective_capacity,
                bounds=_bounds4(poly.bounds),
                compactness=compact,
                reflex_vertex_count=reflex_count,
                reflex_complexity=reflex_complexity,
                bbox_waste_ratio=bbox_waste,
                notch_complexity=notch_complexity,
                facade_len=float(sum(facade_lengths)),
                facade_total_usable_len=facade_usable,
                window_slot_count=window_slots,
                max_single_facade_edge_len=max_facade,
                corridor_frontage_len=float(sum(access_lengths)),
                access_edge_total_len=float(sum(access_lengths)),
                access_total_usable_len=access_usable,
                corridor_door_slot_count=door_slots,
                max_single_corridor_edge_len=max_access,
                largest_empty_rect_estimate=ler_area,
                largest_empty_rect_width=ler_w,
                largest_empty_rect_height=ler_h,
                slot_count_large=_slot_capacity(LARGE_SLOT_AREA, LARGE_SLOT_MIN_SIDE),
                slot_count_medium=_slot_capacity(MEDIUM_SLOT_AREA, MEDIUM_SLOT_MIN_SIDE),
                slot_count_small=_slot_capacity(SMALL_SLOT_AREA, SMALL_SLOT_MIN_SIDE),
                core_overlap_area=core_overlap,
                forbidden_overlap_area=forbidden_overlap,
                core_contract_id=core_contract_id,
                core_union_hash=core_union_hash,
                valid=(not rejection_reasons),
                rejection_reasons=rejection_reasons,
            )
        )
    return metrics


def _cluster_groups(room_specs: Sequence[RoomSpec], adjacency_graph: Optional[Dict[str, List[str]]] = None) -> List[List[RoomSpec]]:
    rooms_by_id = {r.room_id: r for r in room_specs or []}
    parent: Dict[str, str] = {r.room_id: r.room_id for r in room_specs or []}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        if a in rooms_by_id and b in rooms_by_id:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    graph = dict(adjacency_graph or {})
    for room in room_specs or []:
        for other in list(getattr(room, "adjacency_required", []) or []) + list(graph.get(room.room_id, []) or []):
            union(room.room_id, other)

    groups: Dict[str, List[RoomSpec]] = {}
    for room in room_specs or []:
        groups.setdefault(find(room.room_id), []).append(room)
    return list(groups.values())


def build_cluster_metrics(
    room_specs: Sequence[RoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
) -> List[ClusterMetrics]:
    metrics: List[ClusterMetrics] = []
    for idx, group in enumerate(_cluster_groups(room_specs, adjacency_graph)):
        ordered = sorted(group, key=lambda r: (-float(getattr(r, "target_area", 0.0) or 0.0), r.room_id))
        largest = ordered[0] if ordered else None
        largest_w = 0.0
        largest_d = 0.0
        if largest is not None:
            largest_w, largest_d = estimate_room_min_dimensions(largest)
        room_ids = [r.room_id for r in group]
        room_id_set = set(room_ids)
        adjacency_edges = set()
        graph = dict(adjacency_graph or {})
        for room in group:
            neighbors = (
                list(getattr(room, "adjacency_required", []) or [])
                + list(getattr(room, "adjacency_preferred", []) or [])
                + list(graph.get(room.room_id, []) or [])
            )
            for other in neighbors:
                if other and (other in room_id_set or room.room_id in room_id_set):
                    adjacency_edges.add(tuple(sorted((room.room_id, str(other)))))

        def _bucket(room: RoomSpec) -> str:
            area = float(getattr(room, "target_area", 0.0) or 0.0)
            w, d = estimate_room_min_dimensions(room)
            min_side = min(w, d)
            if area >= LARGE_SLOT_AREA and min_side >= LARGE_SLOT_MIN_SIDE:
                return "large"
            if area >= MEDIUM_SLOT_AREA and min_side >= MEDIUM_SLOT_MIN_SIDE:
                return "medium"
            return "small"

        buckets = [_bucket(r) for r in group]
        metrics.append(
            ClusterMetrics(
                cluster_id=f"cluster_{idx}",
                room_ids=room_ids,
                target_area_sum=float(sum(max(0.0, float(r.target_area)) for r in group)),
                min_area_sum=float(sum(max(0.0, float(r.target_area)) * 0.85 for r in group)),
                max_area_sum=float(sum(max(0.0, float(r.target_area)) * 1.15 for r in group)),
                largest_room_area=float(getattr(largest, "target_area", 0.0) or 0.0) if largest is not None else 0.0,
                largest_room_type=_room_type(largest) if largest is not None else "",
                largest_room_min_width_estimate=largest_w,
                largest_room_min_depth_estimate=largest_d,
                room_count=len(group),
                large_room_count=sum(1 for b in buckets if b == "large"),
                medium_room_count=sum(1 for b in buckets if b == "medium"),
                small_room_count=sum(1 for b in buckets if b == "small"),
                needs_window_count=sum(1 for r in group if _needs_hard_window(r)),
                needs_corridor_access_count=sum(1 for r in group if bool(getattr(r, "needs_corridor_access", True))),
                public_room_count=sum(1 for r in group if _is_public_room(r)),
                private_room_count=sum(1 for r in group if _is_private_room(r)),
                service_room_count=sum(1 for r in group if _is_service_room(r)),
                adjacency_degree=len(adjacency_edges),
            )
        )
    metrics.sort(key=lambda m: (m.needs_window_count <= 0, -float(m.target_area_sum), m.cluster_id))
    return metrics


def evaluate_cluster_island_feasibility(
    *,
    variant_id: str,
    cluster_metrics: Sequence[ClusterMetrics],
    island_metrics: Sequence[IslandMetrics],
) -> List[ClusterIslandFeasibility]:
    rows: List[ClusterIslandFeasibility] = []
    for cluster in cluster_metrics or []:
        for island in island_metrics or []:
            reasons: List[str] = []
            area_fit_ratio = island.effective_capacity_area / max(cluster.target_area_sum, EPSILON)
            capacity_margin = island.effective_capacity_area - cluster.target_area_sum
            capacity_margin_ratio = capacity_margin / max(island.effective_capacity_area, EPSILON)

            ler_area = max(island.largest_empty_rect_estimate, EPSILON)
            largest_area_ratio = ler_area / max(cluster.largest_room_area, EPSILON)
            width_ratio = island.largest_empty_rect_width / max(cluster.largest_room_min_width_estimate, EPSILON)
            height_ratio = island.largest_empty_rect_height / max(cluster.largest_room_min_depth_estimate, EPSILON)
            largest_room_fit_ratio = min(largest_area_ratio, width_ratio, height_ratio)

            corridor_fit_ratio = (
                island.corridor_door_slot_count / max(cluster.needs_corridor_access_count, 1)
                if cluster.needs_corridor_access_count > 0 else 1.0
            )
            facade_fit_ratio = (
                island.window_slot_count / max(cluster.needs_window_count, 1)
                if cluster.needs_window_count > 0 else 1.0
            )
            slot_demand = max(cluster.large_room_count + cluster.medium_room_count + cluster.small_room_count, 1)
            slot_supply = island.slot_count_large + island.slot_count_medium + island.slot_count_small
            slot_fit_ratio = slot_supply / slot_demand

            hard_feasible = bool(island.valid)
            if not island.valid:
                reasons.extend(island.rejection_reasons or ["invalid_island"])
            if cluster.min_area_sum > island.effective_capacity_area * 1.10 + EPSILON:
                hard_feasible = False
                reasons.append("cluster_min_area_over_effective_capacity")
            if cluster.largest_room_area > island.largest_empty_rect_estimate * 1.25 + EPSILON:
                hard_feasible = False
                reasons.append("largest_room_area_exceeds_largest_rect")
            if width_ratio < (1.0 / 1.25) or height_ratio < (1.0 / 1.25):
                hard_feasible = False
                reasons.append("largest_room_dimensions_exceed_largest_rect")
            if cluster.needs_window_count > 0 and island.window_slot_count <= 0:
                hard_feasible = False
                reasons.append("missing_window_slot_capacity")
            if cluster.needs_corridor_access_count > 0 and island.corridor_door_slot_count <= 0:
                hard_feasible = False
                reasons.append("missing_corridor_access_slot_capacity")

            area_penalty = _clamp(max(0.0, 1.0 - area_fit_ratio))
            shape_penalty = _clamp(max(island.notch_complexity, max(0.0, 1.0 - largest_room_fit_ratio)))
            access_penalty = _clamp(max(0.0, 1.0 - corridor_fit_ratio)) if cluster.needs_corridor_access_count > 0 else 0.0
            window_penalty = _clamp(max(0.0, 1.0 - facade_fit_ratio)) if cluster.needs_window_count > 0 else 0.0
            slot_penalty = _clamp(max(0.0, 1.0 - slot_fit_ratio))
            total_penalty = (
                0.32 * area_penalty
                + 0.24 * shape_penalty
                + 0.18 * access_penalty
                + 0.18 * window_penalty
                + 0.08 * slot_penalty
            )
            score = _clamp(1.0 - total_penalty)
            rows.append(
                ClusterIslandFeasibility(
                    variant_id=variant_id,
                    cluster_id=cluster.cluster_id,
                    island_id=island.island_id,
                    hard_feasible=hard_feasible,
                    feasibility_score=score,
                    rejection_reasons=sorted(set(reasons)),
                    area_fit_ratio=area_fit_ratio,
                    largest_room_fit_ratio=largest_room_fit_ratio,
                    corridor_fit_ratio=corridor_fit_ratio,
                    facade_fit_ratio=facade_fit_ratio,
                    slot_fit_ratio=slot_fit_ratio,
                    capacity_margin=capacity_margin,
                    capacity_margin_ratio=capacity_margin_ratio,
                    shape_penalty=shape_penalty,
                    access_penalty=access_penalty,
                    window_penalty=window_penalty,
                )
            )
    return rows
