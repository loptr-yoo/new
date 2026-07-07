"""Experimental semantic-first grid growth topology planner.

This module intentionally only replaces the brittle island-assignment front-end.
It returns the same Island + AssignmentResult shape consumed by the existing
CP-SAT, residual sweep, door, reachability and wall-graph pipeline.
"""
from __future__ import annotations

import logging
import math
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from shapely.geometry import GeometryCollection, LineString, Polygon, box
from shapely.ops import unary_union

from .core_contracts import CORE_OVERLAP_EPSILON_AREA, CoreFootprintContract
from .capacity_aware_area_allocator import (
    CapacityAwareAreaAllocationConfig,
    apply_capacity_aware_targets_to_room_specs,
    build_capacity_aware_targets,
)
from .exceptions import LayoutGeometryInvariantError, LayoutTopologyError
from .island_room_assigner import AssignmentResult, DegradationSummary
from .room_spec import RoomSpec, ZoneType
from .topology_feasibility import (
    TopologyFeasibilityReport,
    TopologyVariant,
    build_cluster_metrics,
    build_island_metrics,
    evaluate_cluster_island_feasibility,
)
from .topology_assignment_solver import TopologyAssignmentConfig, TopologyAssignmentSolver
from .topology_generator import CoreTube, Corridor, Island

logger = logging.getLogger(__name__)


GRID_GROWTH = "grid_growth"
DEFAULT_GROWTH_RESOLUTION = 0.5
DEFAULT_REFINEMENT_RESOLUTION = 0.25
DEFAULT_MIN_DOOR_WIDTH = 0.8
DEFAULT_ANCHOR_FRONTAGE = 1.0
DEFAULT_TOPOLOGY_SEEDS = [0, 1, 2, 3, 5, 8, 13]


@dataclass
class GridGrowthWeights:
    access: float = 4.0
    required_cohesion: float = 3.5
    area_pressure: float = 2.5
    window_access: float = 2.0
    compactness: float = 1.8
    staircase_penalty: float = 3.0
    privacy: float = 1.0
    core_noise: float = 0.8
    forbidden_neighbor: float = 6.0


@dataclass
class GridGrowthConfig:
    growth_resolution: float = DEFAULT_GROWTH_RESOLUTION
    refinement_resolution: float = DEFAULT_REFINEMENT_RESOLUTION
    min_door_width: float = DEFAULT_MIN_DOOR_WIDTH
    min_anchor_frontage: float = DEFAULT_ANCHOR_FRONTAGE
    corridor_budget_tolerance: float = 0.35
    max_unclaimed_component_area: float = 8.0
    storage_fill_allowance_ratio: float = 0.12
    corridor_fill_allowance_ratio: float = 0.20
    semantic_growth_max_iterations: int = 20000
    semantic_growth_max_cells: int = 100000
    semantic_growth_max_residual_passes: int = 3
    weights: GridGrowthWeights = field(default_factory=GridGrowthWeights)


@dataclass
class GridCluster:
    cluster_id: str
    rooms: List[RoomSpec]
    target_sum: float
    min_sum: float
    largest_room_area: float
    needs_window: bool
    requires_public_access: bool
    required_room_ids: Set[str] = field(default_factory=set)


@dataclass
class GridGrowthResult:
    corridors: List[Corridor]
    islands: List[Island]
    assignments: Dict[str, AssignmentResult]
    degradation: DegradationSummary
    metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def _room_meta(room: RoomSpec) -> str:
    return str(getattr(room, "room_type", "") or "").lower()


def _is_leaf_room(room: RoomSpec) -> bool:
    kind = _room_meta(room)
    return kind in {"bathroom", "toilet", "storage", "utility", "closet", "wardrobe", "void"}


def _is_habitable(room: RoomSpec) -> bool:
    return not _is_leaf_room(room) and not bool(getattr(room, "is_dummy", False))


def _needs_hard_window(room: RoomSpec) -> bool:
    kind = _room_meta(room)
    return bool(getattr(room, "needs_window", False)) and kind in {
        "bedroom",
        "master_bedroom",
        "living_room",
        "living",
        "study",
        "office",
    }


def _polygon_pieces(geom: Any, *, min_area: float = 0.01) -> List[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom] if float(geom.area) > min_area else []
    pieces: List[Polygon] = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            if isinstance(g, Polygon) and float(g.area) > min_area:
                pieces.append(g)
            elif hasattr(g, "geoms"):
                pieces.extend(_polygon_pieces(g, min_area=min_area))
    return pieces


def _exterior_sides(poly: Polygon, floor: Polygon, tol: float = 0.08) -> List[str]:
    if poly.is_empty:
        return []
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    fminx, fminy, fmaxx, fmaxy = (float(v) for v in floor.bounds)
    sides: List[str] = []
    if abs(minx - fminx) <= tol:
        sides.append("west")
    if abs(maxx - fmaxx) <= tol:
        sides.append("east")
    if abs(miny - fminy) <= tol:
        sides.append("south")
    if abs(maxy - fmaxy) <= tol:
        sides.append("north")
    return sides


def _corridor_frontage(poly: Polygon, corridor_union: Any) -> float:
    if corridor_union is None or getattr(corridor_union, "is_empty", True) or poly.is_empty:
        return 0.0
    try:
        shared = float(poly.boundary.intersection(corridor_union.boundary).length)
        if shared <= 1e-6:
            shared = float(poly.boundary.intersection(corridor_union.buffer(0.02).boundary).length)
        return max(0.0, shared)
    except Exception:
        return 0.0


def _facade_frontage(poly: Polygon, floor: Polygon) -> float:
    if poly.is_empty or floor.is_empty:
        return 0.0
    try:
        return max(0.0, float(poly.boundary.intersection(floor.exterior).length))
    except Exception:
        return 0.0


def _reflex_vertex_count(poly: Polygon) -> int:
    # For the PoC planner we use a conservative proxy: non-rectangle vertices
    # beyond four are treated as reflex risk. Full orientation analysis can
    # replace this without changing the public contract.
    try:
        coords = list(poly.exterior.coords)
    except Exception:
        return 999
    unique_count = max(0, len(coords) - 1)
    return max(0, unique_count - 4)


def _bbox_short_side(poly: Polygon) -> float:
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    return min(maxx - minx, maxy - miny)


def _fill_rate(poly: Polygon) -> float:
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    bbox_area = max(0.0, (maxx - minx) * (maxy - miny))
    return float(poly.area) / bbox_area if bbox_area > 1e-9 else 0.0


def _variant_profile_for_seed(seed: int) -> Dict[str, Any]:
    """Deterministic topology perturbation profile.

    Seed 0 is intentionally the legacy/current behavior. Other seeds are a
    fixed Fibonacci-ish set that nudges the corridor spine without uncontrolled
    randomness.
    """

    seed = int(seed or 0)
    profiles: Dict[int, Tuple[float, str, str, str]] = {
        0: (0.0, "center", "direct", "balanced"),
        1: (-0.24, "west", "low", "boundary"),
        2: (0.24, "east", "high", "boundary"),
        3: (-0.16, "east", "low", "balanced"),
        5: (0.16, "west", "high", "balanced"),
        8: (-0.30, "west", "outer", "boundary"),
        13: (0.30, "east", "outer", "boundary"),
    }
    offset, detour, link_priority, boundary_pref = profiles.get(seed, profiles[0])
    return {
        "seed": seed,
        "spine_offset_ratio": float(offset),
        "core_detour_side": detour,
        "link_priority": link_priority,
        "boundary_contact_preference": boundary_pref,
    }


class GridGrowthPlanner:
    """Semantic-first topology planner with V4.5-compatible handoff."""

    def __init__(
        self,
        *,
        floor_boundary: Polygon,
        core_tube: CoreTube,
        room_specs: Sequence[RoomSpec],
        adjacency_graph: Optional[Dict[str, List[str]]] = None,
        corridor_width: float = 1.8,
        corridor_layout: str = "organic",
        floor_number: Optional[int] = None,
        config: Optional[GridGrowthConfig] = None,
        core_contract: Optional[CoreFootprintContract] = None,
        floor_usable_polygon: Optional[Any] = None,
        topology_seed: int = 0,
        variant_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.floor = floor_boundary
        self.core = core_tube
        self.core_contract = core_contract
        self.floor_usable_polygon = floor_usable_polygon
        self.rooms = list(room_specs or [])
        self.adjacency = dict(adjacency_graph or {})
        self.corridor_width = float(corridor_width)
        self.corridor_layout = str(corridor_layout or "")
        self.floor_number = floor_number
        self.config = config or GridGrowthConfig()
        self.topology_seed = int(topology_seed or 0)
        self.variant_profile = dict(variant_profile or _variant_profile_for_seed(self.topology_seed))
        self.metadata: Dict[str, Any] = {
            "topology_mode": GRID_GROWTH,
            "topology_seed": self.topology_seed,
            "variant_profile": dict(self.variant_profile),
            "resolution": {
                "growth": float(self.config.growth_resolution),
                "refinement": float(self.config.refinement_resolution),
            },
            "weights": self.config.weights.__dict__.copy(),
            "clusters": [],
            "corridor": {},
            "core_contract": self._core_metadata(),
            "core_docking_candidates": [],
            "handoff": [],
            "frontier_trace": [],
        }

    def _core_union(self) -> Any:
        if self.core_contract is not None and getattr(self.core_contract, "core_union", None) is not None:
            return self.core_contract.core_union
        return getattr(self.core, "polygon", None)

    def _core_metadata(self) -> Dict[str, Any]:
        cc = self.core_contract
        if cc is None:
            return {}
        return {
            "core_contract_id": cc.core_contract_id,
            "version": cc.version,
            "core_union_hash": cc.core_union_hash,
            "core_union_area": float(cc.core_union_area),
            "core_union_bounds": tuple(cc.core_union_bounds),
        }

    def _usable_polygon(self) -> Any:
        if self.floor_usable_polygon is not None and not getattr(self.floor_usable_polygon, "is_empty", True):
            return self.floor_usable_polygon
        core_union = self._core_union()
        if core_union is None or getattr(core_union, "is_empty", True):
            return self.floor
        try:
            usable = self.floor.difference(core_union)
        except Exception:
            usable = self.floor
        self.floor_usable_polygon = usable
        logger.info(
            "[CORE] Usable polygon built | floor=%s | core_contract_id=%s | floor_area=%.2f | core_union_area=%.2f | usable_area=%.2f",
            self.floor_number,
            (self._core_metadata() or {}).get("core_contract_id"),
            float(self.floor.area),
            float(getattr(core_union, "area", 0.0) or 0.0),
            float(getattr(usable, "area", 0.0) or 0.0),
        )
        return usable

    def _preserve_core_docking_candidates(self, usable: Any) -> None:
        contract = self.core_contract
        if contract is None or usable is None or getattr(usable, "is_empty", True):
            return
        candidates: List[Dict[str, Any]] = []
        usable_boundary = getattr(usable, "boundary", None)
        for component in getattr(contract, "core_public_halls", []) or []:
            hall_poly = getattr(component, "polygon", None)
            if hall_poly is None or getattr(hall_poly, "is_empty", True):
                continue
            try:
                edge = hall_poly.boundary.intersection(usable_boundary)
                length = float(getattr(edge, "length", 0.0) or 0.0)
            except Exception:
                continue
            if length <= max(0.1, self.corridor_width * 0.25):
                continue
            try:
                bbox = tuple(round(float(v), 4) for v in edge.bounds)
            except Exception:
                bbox = None
            candidates.append({
                "core_contract_id": contract.core_contract_id,
                "core_union_hash": contract.core_union_hash,
                "core_public_hall_id": component.component_id,
                "component_type": component.component_type,
                "candidate_edge_length": round(length, 4),
                "candidate_edge_bbox": bbox,
            })
        self.metadata["core_docking_candidates"] = candidates
        logger.info(
            "[CORE] Docking candidates preserved | floor=%s | contract=%s | candidates=%d",
            self.floor_number,
            getattr(contract, "core_contract_id", None),
            len(candidates),
        )

    def _raise_core_generation_error(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        meta = {
            "failure_kind": "geometry_invariant",
            "stage": "core_generation_invariant_failed",
            "topology_mode": GRID_GROWTH,
            **self._core_metadata(),
        }
        meta.update(metadata or {})
        raise LayoutGeometryInvariantError(
            message,
            floor_id=f"F{int(self.floor_number or 1)}",
            stage="core_generation_invariant_failed",
            metadata={**meta, "semantic_repair_allowed": False},
        )

    def plan(self) -> GridGrowthResult:
        corridors = self._build_corridor_skeleton()
        self._validate_corridor_skeleton(corridors)
        clusters = self._build_clusters()
        islands = self._build_candidate_islands(corridors)
        assignments = self._assign_clusters_to_islands(clusters, islands, corridors)

        for island in islands:
            assigned = assignments.get(island.id)
            if assigned:
                island.assigned_rooms = [r.room_id for r in assigned.rooms]
                island.remaining_capacity = max(0.0, float(island.area) - float(assigned.total_area))

        self.metadata["island_count"] = len(islands)
        self.metadata["assigned_room_count"] = sum(len(a.rooms) for a in assignments.values())
        logger.info(
            "[GRID] Handoff complete | floor=%s | islands=%d | clusters=%d | corridors=%d",
            self.floor_number,
            len(islands),
            len(clusters),
            len(corridors),
        )
        return GridGrowthResult(
            corridors=corridors,
            islands=islands,
            assignments=assignments,
            degradation=DegradationSummary(),
            metadata=self.metadata,
        )

    def _build_corridor_skeleton(self) -> List[Corridor]:
        minx, miny, maxx, maxy = (float(v) for v in self.floor.bounds)
        cx, cy = self.core.center
        # A central horizontal spine creates doorable frontage on both sides and
        # avoids the 3m-deep strip pathology visible in the latest assignment logs.
        height = max(0.0, maxy - miny)
        offset_ratio = float(self.variant_profile.get("spine_offset_ratio", 0.0) or 0.0)
        spine_y = float(cy) + offset_ratio * height
        spine_y = min(max(spine_y, miny + self.corridor_width), maxy - self.corridor_width)
        if abs(spine_y - miny) < self.corridor_width or abs(maxy - spine_y) < self.corridor_width:
            spine_y = (miny + maxy) / 2.0
        corridors = [
            Corridor(
                id="corridor_grid_main",
                centerline=LineString([(minx, spine_y), (maxx, spine_y)]),
                width=self.corridor_width,
                orientation="horizontal",
            )
        ]
        core_poly = self._core_union()
        if core_poly is not None and not core_poly.is_empty:
            cminx, cminy, cmaxx, cmaxy = (float(v) for v in core_poly.bounds)
            if not (cminy <= spine_y <= cmaxy):
                link_x = min(max(float(cx), minx + self.corridor_width), maxx - self.corridor_width)
                detour = str(self.variant_profile.get("core_detour_side", "") or "").lower()
                if detour == "west":
                    link_x = min(max(cminx - self.corridor_width * 0.55, minx + self.corridor_width), maxx - self.corridor_width)
                elif detour == "east":
                    link_x = min(max(cmaxx + self.corridor_width * 0.55, minx + self.corridor_width), maxx - self.corridor_width)
                target_y = cminy if spine_y < cminy else cmaxy
                corridors.append(
                    Corridor(
                        id="corridor_grid_core_link",
                        centerline=LineString([(link_x, spine_y), (link_x, target_y)]),
                        width=self.corridor_width,
                        orientation="vertical",
                    )
                )
        usable = self._usable_polygon()
        self._preserve_core_docking_candidates(usable)
        core_overlap_before = 0.0
        core_overlap_after = 0.0
        if core_poly is not None and not getattr(core_poly, "is_empty", True):
            for corridor in corridors:
                try:
                    before = float(corridor.polygon.intersection(core_poly).area)
                except Exception:
                    before = 0.0
                core_overlap_before += before
                if before > CORE_OVERLAP_EPSILON_AREA:
                    clipped = corridor.polygon.difference(core_poly).intersection(usable)
                    if getattr(clipped, "is_empty", True):
                        self._raise_core_generation_error(
                            "Core-aware corridor clipping removed corridor skeleton",
                            {
                                "corridor_id": corridor.id,
                                "overlap_before": before,
                            },
                        )
                    corridor.polygon = clipped.buffer(0)
                try:
                    core_overlap_after += float(corridor.polygon.intersection(core_poly).area)
                except Exception:
                    pass
            logger.info(
                "[CORE] Corridor clipped by core_union | floor=%s | core_contract_id=%s | overlap_before=%.4f | overlap_after=%.4f",
                self.floor_number,
                (self._core_metadata() or {}).get("core_contract_id"),
                core_overlap_before,
                core_overlap_after,
            )
            if core_overlap_after > CORE_OVERLAP_EPSILON_AREA:
                self._raise_core_generation_error(
                    "Core-aware corridor still overlaps core after clipping",
                    {
                        "overlap_before": core_overlap_before,
                        "overlap_after": core_overlap_after,
                    },
                )
        corridor_union = unary_union([c.polygon for c in corridors])
        self.metadata["corridor"] = {
            "mode": self.corridor_layout,
            "width": float(self.corridor_width),
            "area": float(corridor_union.intersection(self.floor).area),
            "ids": [c.id for c in corridors],
            "core_overlap_before": float(core_overlap_before),
            "core_overlap_after": float(core_overlap_after),
        }
        logger.info(
            "[GRID] CorridorSkeletonGate start | width=%.2f | corridors=%s | area=%.2fm2",
            self.corridor_width,
            [c.id for c in corridors],
            float(corridor_union.intersection(self.floor).area),
        )
        return corridors

    def _validate_corridor_skeleton(self, corridors: Sequence[Corridor]) -> None:
        if not corridors:
            self._raise_grid_error("GRID_CORRIDOR_DISCONNECTED", "No corridor skeleton generated")
        corridor_union = unary_union([c.polygon for c in corridors]).intersection(self.floor)
        if corridor_union.is_empty:
            self._raise_grid_error("GRID_CORRIDOR_DISCONNECTED", "Empty corridor skeleton")
        erosion = max(0.0, self.corridor_width / 2.0 - self.config.growth_resolution / 2.0)
        probe = corridor_union.buffer(-erosion, join_style=2) if erosion > 1e-6 else corridor_union
        probe_pieces = _polygon_pieces(probe, min_area=0.01)
        if not probe_pieces:
            self._raise_grid_error(
                "GRID_CORRIDOR_BOTTLENECK",
                "Corridor probe cannot traverse skeleton; bottleneck below corridor width",
            )
        if len(probe_pieces) > 1:
            self._raise_core_generation_error(
                "Core-aware corridor clipping disconnected corridor skeleton",
                {
                    "probe_pieces": len(probe_pieces),
                    "probe_area": float(sum(p.area for p in probe_pieces)),
                },
            )
        self.metadata["corridor"]["probe_area"] = float(sum(p.area for p in probe_pieces))
        self.metadata["corridor"]["probe_pieces"] = len(probe_pieces)
        logger.info(
            "[GRID] CorridorSkeletonGate pass | probe_pieces=%d | probe_area=%.2fm2",
            len(probe_pieces),
            float(sum(p.area for p in probe_pieces)),
        )

    def _build_clusters(self) -> List[GridCluster]:
        rooms_by_id = {r.room_id: r for r in self.rooms}
        parent: Dict[str, str] = {r.room_id: r.room_id for r in self.rooms}

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

        for room in self.rooms:
            for other in list(getattr(room, "adjacency_required", []) or []) + list(self.adjacency.get(room.room_id, []) or []):
                union(room.room_id, other)

        groups: Dict[str, List[RoomSpec]] = {}
        for room in self.rooms:
            groups.setdefault(find(room.room_id), []).append(room)

        clusters: List[GridCluster] = []
        for idx, group in enumerate(groups.values()):
            target = float(sum(max(0.0, float(r.target_area)) for r in group))
            min_sum = float(sum(max(0.0, float(r.target_area)) * 0.85 for r in group))
            largest = max((float(r.target_area) for r in group), default=0.0)
            needs_window = any(_needs_hard_window(r) for r in group)
            requires_public_access = any(_is_habitable(r) and bool(getattr(r, "needs_corridor_access", True)) for r in group)
            cluster = GridCluster(
                cluster_id=f"cluster_{idx}",
                rooms=group,
                target_sum=target,
                min_sum=min_sum,
                largest_room_area=largest,
                needs_window=needs_window,
                requires_public_access=requires_public_access,
                required_room_ids={r.room_id for r in group},
            )
            clusters.append(cluster)
            logger.debug(
                "[GRID] Cluster built | id=%s | rooms=%s | target=%.2f | min=%.2f | largest=%.2f | window=%s | access=%s",
                cluster.cluster_id,
                [r.room_id for r in group],
                target,
                min_sum,
                largest,
                needs_window,
                requires_public_access,
            )
        clusters.sort(key=lambda c: (not c.needs_window, -c.target_sum, c.cluster_id))
        self.metadata["clusters"] = [
            {
                "cluster_id": c.cluster_id,
                "rooms": [r.room_id for r in c.rooms],
                "target_sum": round(c.target_sum, 3),
                "min_sum": round(c.min_sum, 3),
                "largest_room_area": round(c.largest_room_area, 3),
                "needs_window": c.needs_window,
                "requires_public_access": c.requires_public_access,
            }
            for c in clusters
        ]
        return clusters

    def _build_candidate_islands(self, corridors: Sequence[Corridor]) -> List[Island]:
        usable = self._usable_polygon()
        blocked = [c.polygon for c in corridors]
        blocked = [g for g in blocked if g is not None and not getattr(g, "is_empty", True)]
        available = usable.difference(unary_union(blocked)) if blocked else usable
        corridor_union = unary_union([c.polygon for c in corridors]) if corridors else Polygon()
        pieces = sorted(_polygon_pieces(available, min_area=4.0), key=lambda p: p.area, reverse=True)
        islands: List[Island] = []
        old_available_area = float(self.floor.difference(unary_union([getattr(self.core, "polygon", GeometryCollection())] + blocked)).area) if blocked else float(self.floor.area)
        for idx, poly in enumerate(pieces):
            fixed = poly.buffer(0)
            if fixed.is_empty:
                continue
            core_union = self._core_union()
            if core_union is not None and not getattr(core_union, "is_empty", True):
                try:
                    core_overlap = float(fixed.intersection(core_union).area)
                except Exception:
                    core_overlap = float("inf")
                if core_overlap > CORE_OVERLAP_EPSILON_AREA:
                    self._raise_core_generation_error(
                        "Candidate island overlaps core footprint",
                        {
                            "island_index": idx,
                            "overlap_area": core_overlap,
                            "threshold": CORE_OVERLAP_EPSILON_AREA,
                        },
                    )
            # Keep rectilinear pieces stable for CP-SAT; complex pieces are still
            # accepted but marked by metadata/gates.
            island = Island(id=f"grid_island_{idx}", polygon=fixed)
            frontage = _corridor_frontage(fixed, corridor_union)
            facade = _facade_frontage(fixed, self.floor)
            island.corridor_edges = ["grid_access"] if frontage >= self.config.min_anchor_frontage else []
            island.exterior_walls = _exterior_sides(fixed, self.floor)
            island.has_exterior_wall = bool(island.exterior_walls)
            island.suggested_zone = ZoneType.PRIVATE if island.has_exterior_wall else ZoneType.PUBLIC
            try:
                island.core_contract_id = (self._core_metadata() or {}).get("core_contract_id")
                island.core_union_hash = (self._core_metadata() or {}).get("core_union_hash")
                island.core_aware_area = float(fixed.area)
            except Exception:
                pass
            islands.append(island)
            logger.debug(
                "[GRID] Candidate island | id=%s | area=%.2f | frontage=%.2f | facade=%.2f | fill_rate=%.2f | reflex=%d",
                island.id,
                float(island.area),
                frontage,
                facade,
                _fill_rate(fixed),
                _reflex_vertex_count(fixed),
            )
        if not islands:
            self._raise_grid_error("GRID_NO_USABLE_ISLAND", "No usable islands after corridor/core subtraction")
        new_area = float(sum(float(i.area) for i in islands))
        core_removed = max(0.0, float(old_available_area) - new_area)
        self.metadata["core_aware_islands"] = {
            "old_area": float(old_available_area),
            "core_aware_area": new_area,
            "core_removed_area": core_removed,
            **self._core_metadata(),
        }
        logger.info(
            "[GRID] Core-aware island stats | floor=%s | old_area=%.2f | core_aware_area=%.2f | core_removed_area=%.2f | core_union_hash=%s",
            self.floor_number,
            float(old_available_area),
            new_area,
            core_removed,
            (self._core_metadata() or {}).get("core_union_hash"),
        )
        return islands

    def _assign_clusters_to_islands(
        self,
        clusters: Sequence[GridCluster],
        islands: Sequence[Island],
        corridors: Sequence[Corridor],
    ) -> Dict[str, AssignmentResult]:
        corridor_union = unary_union([c.polygon for c in corridors]) if corridors else Polygon()
        state: Dict[str, Dict[str, Any]] = {
            i.id: {"rooms": [], "target": 0.0, "min": 0.0, "clusters": []}
            for i in islands
        }

        def _can_host(island: Island, cluster: GridCluster) -> Tuple[bool, str, Dict[str, Any]]:
            poly = island.polygon
            frontage = _corridor_frontage(poly, corridor_union)
            facade = _facade_frontage(poly, self.floor)
            fill = _fill_rate(poly)
            reflex = _reflex_vertex_count(poly)
            short = _bbox_short_side(poly)
            current_min = float(state[island.id]["min"])
            hard_largest_req = cluster.largest_room_area * (1.15 if cluster.needs_window else 0.85)
            if cluster.requires_public_access and frontage + 1e-6 < self.config.min_anchor_frontage:
                return False, "missing_doorable_access_anchor", {"frontage": frontage}
            if cluster.needs_window and facade + 1e-6 < self.config.min_anchor_frontage:
                return False, "missing_facade_frontage", {"facade_frontage": facade}
            if float(island.area) + 1e-6 < hard_largest_req:
                return False, "largest_room_does_not_fit", {"required": hard_largest_req, "island_area": float(island.area)}
            if current_min + cluster.min_sum > float(island.area) * 1.02:
                return False, "cluster_min_area_over_capacity", {"current_min": current_min, "cluster_min": cluster.min_sum}
            if fill < 0.65:
                return False, "low_fill_rate", {"fill_rate": fill}
            if reflex > 12:
                return False, "too_many_reflex_vertices", {"reflex_vertices": reflex}
            min_width = max((float(getattr(r, "min_width", 2.5) or 2.5) for r in cluster.rooms), default=2.5)
            if short + 1e-6 < min_width:
                return False, "bbox_short_side_too_small", {"bbox_short_side": short, "min_width": min_width}
            proxy = self._pack_proxy(poly, cluster)
            if not proxy.get("ok"):
                return False, str(proxy.get("failed_reason", "pack_proxy_failed")), proxy
            return True, "ok", {
                "frontage": frontage,
                "facade_frontage": facade,
                "fill_rate": fill,
                "reflex_vertices": reflex,
                "pack_proxy_result": proxy,
            }

        for cluster in clusters:
            candidates: List[Tuple[float, Island, Dict[str, Any]]] = []
            rejects: List[Dict[str, Any]] = []
            for island in islands:
                ok, reason, detail = _can_host(island, cluster)
                if not ok:
                    rejects.append({"island_id": island.id, "reason": reason, **detail})
                    continue
                frontage = float(detail.get("frontage", 0.0))
                facade = float(detail.get("facade_frontage", 0.0))
                remaining = float(island.area) - float(state[island.id]["min"])
                compact = _fill_rate(island.polygon)
                score = (
                    remaining
                    + self.config.weights.access * min(1.0, frontage / max(self.config.min_anchor_frontage, 1e-6))
                    + self.config.weights.window_access * (min(1.0, facade / max(self.config.min_anchor_frontage, 1e-6)) if cluster.needs_window else 0.0)
                    + self.config.weights.compactness * compact
                )
                candidates.append((score, island, detail))
            if not candidates:
                self._raise_grid_error(
                    "GRID_CLUSTER_INFEASIBLE",
                    f"Cluster {cluster.cluster_id} cannot be hosted by any grid island",
                    {
                        "cluster_id": cluster.cluster_id,
                        "rooms": [r.room_id for r in cluster.rooms],
                        "target_sum": cluster.target_sum,
                        "largest_room_area": cluster.largest_room_area,
                        "candidate_rejections": rejects,
                    },
                )
            candidates.sort(key=lambda t: (t[0], float(t[1].area)), reverse=True)
            _, chosen, detail = candidates[0]
            state[chosen.id]["rooms"].extend(cluster.rooms)
            state[chosen.id]["target"] += cluster.target_sum
            state[chosen.id]["min"] += cluster.min_sum
            state[chosen.id]["clusters"].append(cluster.cluster_id)
            self.metadata["frontier_trace"].append(
                {
                    "cluster_id": cluster.cluster_id,
                    "chosen_island": chosen.id,
                    "chosen_cell": list(chosen.centroid),
                    "cost_components": {
                        "access": round(float(detail.get("frontage", 0.0)), 3),
                        "window": round(float(detail.get("facade_frontage", 0.0)), 3),
                        "compactness": round(float(detail.get("fill_rate", 0.0)), 3),
                    },
                    "reason": "best_normalized_score",
                }
            )
            logger.info(
                "[GRID] Cluster assigned | cluster=%s | island=%s | rooms=%s | target=%.2f | frontage=%.2f | facade=%.2f",
                cluster.cluster_id,
                chosen.id,
                [r.room_id for r in cluster.rooms],
                cluster.target_sum,
                float(detail.get("frontage", 0.0)),
                float(detail.get("facade_frontage", 0.0)),
            )

        assignments: Dict[str, AssignmentResult] = {}
        for island in islands:
            rooms = list(state[island.id]["rooms"])
            if not rooms:
                continue
            total_area = float(sum(float(r.target_area) for r in rooms))
            assignments[island.id] = AssignmentResult(
                island_id=island.id,
                rooms=rooms,
                total_area=total_area,
                utilization=total_area / float(island.area) if float(island.area) > 1e-9 else 0.0,
            )
            self.metadata["handoff"].append(
                {
                    "island_id": island.id,
                    "area": round(float(island.area), 3),
                    "core_aware_area": round(float(island.area), 3),
                    "core_union_hash": (self._core_metadata() or {}).get("core_union_hash"),
                    "target_area": round(total_area, 3),
                    "min_area": round(float(state[island.id]["min"]), 3),
                    "rooms": [r.room_id for r in rooms],
                    "clusters": list(state[island.id]["clusters"]),
                    "frontage": round(_corridor_frontage(island.polygon, corridor_union), 3),
                    "facade": round(_facade_frontage(island.polygon, self.floor), 3),
                    "fill_rate": round(_fill_rate(island.polygon), 3),
                    "reflex_vertices": _reflex_vertex_count(island.polygon),
                    "handoff_polygon_type": "single_polygon",
                }
            )
        return assignments

    def _pack_proxy(self, poly: Polygon, cluster: GridCluster) -> Dict[str, Any]:
        minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
        width = maxx - minx
        depth = maxy - miny
        facade_len = _facade_frontage(poly, self.floor)

        ordered = sorted(
            cluster.rooms,
            key=lambda r: (not _needs_hard_window(r), -float(getattr(r, "target_area", 0.0) or 0.0), r.room_id),
        )[:3]
        shelf_x = 0.0
        shelf_y = 0.0
        shelf_height = 0.0
        placements: List[Dict[str, Any]] = []
        for room in ordered:
            area = max(0.1, float(room.target_area) * 0.85)
            min_w = max(1.0, float(getattr(room, "min_width", 2.5) or 2.5))
            min_d = max(1.0, float(getattr(room, "min_depth", 2.5) or 2.5))
            aspect_min, aspect_max = getattr(room, "aspect_ratio_range", (0.5, 2.0)) or (0.5, 2.0)
            rw = max(min_w, math.sqrt(area))
            rd = max(min_d, area / max(rw, 1e-6))
            if rd > depth + 1e-6 and depth >= min_d:
                rd = depth
                rw = max(min_w, area / max(rd, 1e-6))
            if rw > width + 1e-6 and width >= min_w:
                rw = width
                rd = max(min_d, area / max(rw, 1e-6))
            if rw / max(rd, 1e-6) > float(aspect_max):
                rw = min(width, math.sqrt(area * float(aspect_max)))
                rd = max(min_d, area / max(rw, 1e-6))
            if rd / max(rw, 1e-6) > 1.0 / max(float(aspect_min), 1e-6):
                rd = min(depth, math.sqrt(area / max(float(aspect_min), 1e-6)))
                rw = max(min_w, area / max(rd, 1e-6))
            if rw > width + 1e-6 or rd > depth + 1e-6:
                return {
                    "ok": False,
                    "failed_room_id": room.room_id,
                    "failed_reason": "room_proxy_rectangle_exceeds_bbox",
                    "required_width": round(rw, 3),
                    "required_depth": round(rd, 3),
                    "bbox_width": round(width, 3),
                    "bbox_depth": round(depth, 3),
                }
            if _needs_hard_window(room) and facade_len + 1e-6 < self.config.min_anchor_frontage:
                return {
                    "ok": False,
                    "failed_room_id": room.room_id,
                    "failed_reason": "window_room_has_no_facade_proxy",
                    "facade_frontage": round(facade_len, 3),
                }
            if shelf_x + rw > width + 1e-6:
                shelf_x = 0.0
                shelf_y += shelf_height
                shelf_height = 0.0
            if shelf_y + rd > depth + 1e-6:
                return {
                    "ok": False,
                    "failed_room_id": room.room_id,
                    "failed_reason": "greedy_shelf_fit_failed",
                    "used_height": round(shelf_y, 3),
                    "room_depth": round(rd, 3),
                    "bbox_depth": round(depth, 3),
                }
            placements.append({"room_id": room.room_id, "w": round(rw, 3), "d": round(rd, 3)})
            shelf_x += rw
            shelf_height = max(shelf_height, rd)
        return {
            "ok": True,
            "order": [r.room_id for r in ordered],
            "placements": placements,
            "facade_frontage": round(facade_len, 3),
            "bbox": [round(width, 3), round(depth, 3)],
        }

    def _raise_grid_error(self, reason: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        metadata = dict(self.metadata)
        metadata.update(extra or {})
        metadata.update({"failure_kind": "grid_growth", "reject_reason": reason})
        logger.error("[GRID] %s | %s | metadata=%s", reason, message, metadata)
        raise LayoutTopologyError(message, metadata=metadata)


def plan_grid_growth_topology(
    *,
    floor_boundary: Polygon,
    core_tube: CoreTube,
    room_specs: Sequence[RoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
    corridor_width: float = 1.8,
    corridor_layout: str = "organic",
    floor_number: Optional[int] = None,
    config: Optional[GridGrowthConfig] = None,
    core_contract: Optional[CoreFootprintContract] = None,
    floor_usable_polygon: Optional[Any] = None,
    topology_seed_list: Optional[Sequence[int]] = None,
    include_topology_variants: bool = True,
    include_topology_assignment_proposal: bool = True,
    enable_topology_assignment_cp_sat: bool = True,
    topology_assignment_dry_run: bool = True,
    enable_topology_assignment_adoption: bool = False,
    allow_topology_assignment_fallback: bool = True,
    enable_topology_assignment_relaxation_diagnostics: bool = False,
    topology_assignment_relaxation_time_limit_seconds: float = 0.5,
    topology_assignment_relaxation_total_time_limit_seconds: float = 3.0,
    topology_assignment_relaxation_max_levels: int = 6,
    topology_assignment_relaxation_num_workers: int = 1,
    enable_capacity_aware_area_allocation: bool = False,
    apply_capacity_aware_area_allocation: bool = False,
    capacity_aware_area_allocation_strict: bool = False,
    capacity_aware_capacity_source: str = "max_variant_effective_capacity",
    capacity_aware_capacity_slack: float = 1.0,
    capacity_aware_reserve_area: float = 0.0,
    capacity_aware_area_epsilon: float = 1e-6,
    capacity_aware_preserve_preferred_when_feasible: bool = True,
    capacity_aware_require_apply_for_target_overflow: bool = False,
    enable_semantic_seeded_territory_variants: bool = False,
    semantic_seeded_territory_variants_dry_run: bool = True,
) -> GridGrowthResult:
    primary = GridGrowthPlanner(
        floor_boundary=floor_boundary,
        core_tube=core_tube,
        room_specs=room_specs,
        adjacency_graph=adjacency_graph,
        corridor_width=corridor_width,
        corridor_layout=corridor_layout,
        floor_number=floor_number,
        config=config,
        core_contract=core_contract,
        floor_usable_polygon=floor_usable_polygon,
        topology_seed=0,
        variant_profile=_variant_profile_for_seed(0),
    ).plan()
    if include_topology_variants:
        try:
            report = plan_grid_growth_topology_variants(
                floor_boundary=floor_boundary,
                core_tube=core_tube,
                room_specs=room_specs,
                adjacency_graph=adjacency_graph,
                corridor_width=corridor_width,
                corridor_layout=corridor_layout,
                floor_number=floor_number,
                config=config,
                core_contract=core_contract,
                floor_usable_polygon=floor_usable_polygon,
                seed_list=topology_seed_list or DEFAULT_TOPOLOGY_SEEDS,
                primary_result=primary,
            )
            primary.metadata.update(report.to_dict())
            _refresh_grid_growth_metadata_gap_closure(
                primary,
                report,
                floor_number=int(floor_number or 1),
                floor_boundary=floor_boundary,
                load_source="heuristic_primary",
            )
            assignment_proposal: Dict[str, Any] = {}
            active_report = report
            active_proposal_source = "before_area_allocation"
            adoption_requested = bool(enable_topology_assignment_adoption)
            adoption_enabled = bool(enable_topology_assignment_cp_sat)
            solver_config: Optional[TopologyAssignmentConfig] = None
            if include_topology_assignment_proposal and bool(enable_topology_assignment_cp_sat):
                solver_config = TopologyAssignmentConfig(
                    enable_topology_assignment_cp_sat=bool(enable_topology_assignment_cp_sat),
                    topology_assignment_dry_run=bool(topology_assignment_dry_run),
                    enable_topology_assignment_adoption=adoption_requested,
                    allow_topology_assignment_fallback=bool(allow_topology_assignment_fallback),
                    enable_topology_assignment_relaxation_diagnostics=bool(enable_topology_assignment_relaxation_diagnostics),
                    topology_assignment_relaxation_time_limit_seconds=float(topology_assignment_relaxation_time_limit_seconds),
                    topology_assignment_relaxation_total_time_limit_seconds=float(topology_assignment_relaxation_total_time_limit_seconds),
                    topology_assignment_relaxation_max_levels=int(topology_assignment_relaxation_max_levels),
                    topology_assignment_relaxation_num_workers=int(topology_assignment_relaxation_num_workers),
                )

            def _solve_assignment_proposal(report_to_solve: TopologyFeasibilityReport) -> Dict[str, Any]:
                if solver_config is None:
                    return {}
                return TopologyAssignmentSolver(solver_config).solve(
                    report_to_solve,
                    heuristic_assignments=primary.assignments,
                    floor_id=f"F{int(floor_number or 1)}",
                )

            if include_topology_assignment_proposal and bool(enable_topology_assignment_cp_sat):
                assignment_proposal = _solve_assignment_proposal(report)
                primary.metadata["topology_assignment_proposal"] = assignment_proposal
            elif include_topology_assignment_proposal:
                assignment_proposal = {
                    "status": "skipped",
                    "reason": "topology_assignment_cp_sat_disabled",
                    "dry_run": bool(topology_assignment_dry_run),
                    "used_for_main_path": False,
                    "cp_sat_enabled": False,
                    "adoption_requested": adoption_requested,
                    "adoption_implemented": True,
                    "allow_topology_assignment_fallback": bool(allow_topology_assignment_fallback),
                }
                primary.metadata["topology_assignment_proposal"] = assignment_proposal

            if bool(enable_capacity_aware_area_allocation):
                before_proposal = dict(assignment_proposal or {})
                allocation_config = CapacityAwareAreaAllocationConfig(
                    enabled=True,
                    apply=bool(apply_capacity_aware_area_allocation),
                    strict=bool(capacity_aware_area_allocation_strict),
                    capacity_source=str(capacity_aware_capacity_source or "max_variant_effective_capacity"),
                    capacity_slack=float(capacity_aware_capacity_slack),
                    reserve_area=float(capacity_aware_reserve_area),
                    area_epsilon=float(capacity_aware_area_epsilon),
                    preserve_preferred_when_feasible=bool(capacity_aware_preserve_preferred_when_feasible),
                    require_apply_for_target_overflow=bool(capacity_aware_require_apply_for_target_overflow),
                )
                allocation_result = build_capacity_aware_targets(
                    floor_id=f"F{int(floor_number or 1)}",
                    room_specs=room_specs,
                    report=report,
                    config=allocation_config,
                )
                allocation_dict = allocation_result.to_dict()
                if include_topology_assignment_proposal and bool(before_proposal):
                    before_proposal.setdefault("area_allocation_applied", False)
                    before_proposal.setdefault("area_allocation_id", allocation_dict.get("area_allocation_id", ""))
                    before_proposal.setdefault("target_hash", allocation_dict.get("area_target_hash_before", ""))
                    primary.metadata["topology_assignment_proposal_before_area_allocation"] = before_proposal
                primary.metadata["capacity_aware_area_allocation"] = allocation_dict
                primary.metadata["active_area_allocation_id"] = allocation_dict.get("area_allocation_id", "")
                primary.metadata["active_target_hash"] = allocation_dict.get("area_target_hash_before", "")
                primary.metadata["active_proposal_source"] = active_proposal_source
                if (
                    allocation_result.status == "min_capacity_infeasible"
                    and bool(capacity_aware_area_allocation_strict)
                ):
                    raise _adoption_topology_error(
                        message="Program minimum area exceeds topology effective capacity",
                        stage="program_min_capacity_infeasible",
                        semantic_repair_allowed=True,
                        metadata={
                            "floor_id": f"F{int(floor_number or 1)}",
                            "capacity_aware_area_allocation": allocation_dict,
                            "suggested_actions": [
                                "reduce room count",
                                "reduce target areas",
                                "increase floor size",
                                "reduce core area ratio",
                                "change corridor/core strategy",
                            ],
                        },
                    )
                should_apply_area = (
                    bool(apply_capacity_aware_area_allocation)
                    and allocation_result.status == "target_overflow_min_feasible"
                    and bool(allocation_result.compression_plan_feasible)
                )
                if should_apply_area:
                    apply_capacity_aware_targets_to_room_specs(room_specs, allocation_result)
                    _refresh_grid_growth_area_handoff(primary)
                    active_report = _rebuild_report_for_area_targets(report, room_specs, adjacency_graph)
                    active_proposal_source = "after_area_allocation"
                    report = active_report
                    primary.metadata["pre_allocation_report_stale"] = True
                    primary.metadata["post_allocation_report_active"] = True
                    primary.metadata.update(active_report.to_dict())
                    allocation_dict = allocation_result.to_dict()
                    primary.metadata["capacity_aware_area_allocation"] = allocation_dict
                    primary.metadata["active_target_hash"] = allocation_dict.get("area_target_hash_after", "")
                    primary.metadata["active_proposal_source"] = active_proposal_source
                    _refresh_grid_growth_metadata_gap_closure(
                        primary,
                        active_report,
                        floor_number=int(floor_number or 1),
                        floor_boundary=floor_boundary,
                        load_source="heuristic_primary",
                    )
                    if include_topology_assignment_proposal and bool(enable_topology_assignment_cp_sat):
                        assignment_proposal = _solve_assignment_proposal(active_report)
                        assignment_proposal = dict(assignment_proposal or {})
                        assignment_proposal["area_allocation_applied"] = True
                        assignment_proposal["area_allocation_id"] = allocation_dict.get("area_allocation_id", "")
                        assignment_proposal["target_hash"] = allocation_dict.get("area_target_hash_after", "")
                        primary.metadata["topology_assignment_proposal_after_area_allocation"] = assignment_proposal
                        primary.metadata["topology_assignment_proposal"] = assignment_proposal
                    if (
                        allocation_result.status == "target_overflow_min_feasible"
                        and allocation_dict.get("area_target_hash_before") == allocation_dict.get("area_target_hash_after")
                    ):
                        primary.metadata.setdefault("capacity_aware_area_allocation_violations", []).append(
                            "capacity_aware_apply_no_target_change"
                        )
                else:
                    if include_topology_assignment_proposal and bool(before_proposal):
                        before_proposal.setdefault("area_allocation_applied", False)
                        before_proposal.setdefault("area_allocation_id", allocation_dict.get("area_allocation_id", ""))
                        before_proposal.setdefault("target_hash", allocation_dict.get("area_target_hash_before", ""))
                        primary.metadata["topology_assignment_proposal_before_area_allocation"] = before_proposal
                        primary.metadata["topology_assignment_proposal"] = before_proposal
                        assignment_proposal = before_proposal
                    if (
                        allocation_result.status == "target_overflow_min_feasible"
                        and bool(capacity_aware_require_apply_for_target_overflow)
                        and not bool(apply_capacity_aware_area_allocation)
                    ):
                        primary.metadata.setdefault("capacity_aware_area_allocation_warnings", []).append(
                            "target_overflow_min_feasible_but_apply_disabled"
                        )
            else:
                primary.metadata["active_proposal_source"] = active_proposal_source

            selected_variant_id_for_gate = str(assignment_proposal.get("selected_variant_id", "") or "")
            selected_variant_for_gate = next(
                (variant for variant in list(active_report.variants or []) if str(variant.variant_id) == selected_variant_id_for_gate),
                None,
            ) if selected_variant_id_for_gate else None
            adoption_gate = _evaluate_topology_assignment_adoption_gate(
                cp_sat_enabled=bool(enable_topology_assignment_cp_sat),
                dry_run=bool(topology_assignment_dry_run),
                adoption_enabled=adoption_requested,
                fallback_allowed=bool(allow_topology_assignment_fallback),
                proposal=assignment_proposal,
                selected_variant=selected_variant_for_gate,
            )
            adoption_record = _topology_assignment_adoption_record(
                requested=adoption_requested,
                enabled=adoption_enabled,
                applied=False,
                used_for_main_path=False,
                proposal=assignment_proposal,
                adoption_failed_reason=str(adoption_gate.get("gate_block_reason") or ""),
                adoption_gate=adoption_gate,
            )
            primary.metadata["topology_assignment_adoption"] = adoption_record

            if bool(enable_semantic_seeded_territory_variants):
                try:
                    semantic_report, semantic_diagnostics, semantic_v0_report = build_semantic_seeded_territory_variants(
                        floor_boundary=floor_boundary,
                        core_tube=core_tube,
                        room_specs=room_specs,
                        adjacency_graph=adjacency_graph,
                        corridor_width=corridor_width,
                        corridor_layout=corridor_layout,
                        floor_number=floor_number,
                        config=config,
                        core_contract=core_contract,
                        floor_usable_polygon=floor_usable_polygon,
                        active_report=active_report,
                    )
                    semantic_config = copy.copy(solver_config) if solver_config is not None else TopologyAssignmentConfig()
                    semantic_config.enable_topology_assignment_cp_sat = True
                    semantic_config.topology_assignment_dry_run = True
                    semantic_config.enable_topology_assignment_adoption = False
                    semantic_v0_proposal = TopologyAssignmentSolver(semantic_config).solve(
                        semantic_v0_report,
                        heuristic_assignments=primary.assignments,
                        floor_id=f"F{int(floor_number or 1)}",
                    )
                    semantic_v0_proposal = dict(semantic_v0_proposal or {})
                    semantic_v0_proposal["variant_family"] = "semantic_seeded_territory"
                    semantic_v0_proposal["semantic_growth_algorithm_version"] = "v0"
                    semantic_v0_proposal["used_for_main_path"] = False
                    semantic_v0_proposal["used_for_adoption"] = False
                    semantic_v0_proposal["dry_run"] = True
                    semantic_proposal = TopologyAssignmentSolver(semantic_config).solve(
                        semantic_report,
                        heuristic_assignments=primary.assignments,
                        floor_id=f"F{int(floor_number or 1)}",
                    )
                    semantic_proposal = dict(semantic_proposal or {})
                    semantic_proposal["variant_family"] = "semantic_seeded_territory"
                    semantic_proposal["semantic_growth_algorithm_version"] = "v1"
                    semantic_proposal["used_for_main_path"] = False
                    semantic_proposal["used_for_adoption"] = False
                    semantic_proposal["dry_run"] = True
                    semantic_candidate_metadata = _candidate_island_metadata(
                        semantic_report,
                        floor_id=f"F{int(floor_number or 1)}",
                    )
                    semantic_diagnostics["semantic_source_island_count"] = sum(
                        1 for row in semantic_candidate_metadata if row.get("source_cluster_ids")
                    )
                    primary.metadata["semantic_seeded_territory_diagnostics"] = semantic_diagnostics
                    primary.metadata["semantic_seeded_v0_assignment_proposal"] = semantic_v0_proposal
                    primary.metadata["semantic_seeded_topology_variants"] = [v.to_dict() for v in list(semantic_report.variants or [])]
                    primary.metadata["semantic_seeded_candidate_island_metadata"] = semantic_candidate_metadata
                    primary.metadata["semantic_seeded_cluster_feasibility_summary"] = _cluster_feasibility_summary(
                        semantic_report,
                        floor_id=f"F{int(floor_number or 1)}",
                    )
                    primary.metadata["semantic_seeded_assignment_proposal"] = semantic_proposal
                    primary.metadata["semantic_seeded_comparison"] = _semantic_seeded_comparison(
                        base_report=active_report,
                        semantic_report=semantic_report,
                        base_proposal=assignment_proposal,
                        semantic_proposal=semantic_proposal,
                        v0_report=semantic_v0_report,
                        v0_proposal=semantic_v0_proposal,
                        diagnostics=semantic_diagnostics,
                    )
                except Exception as exc:
                    primary.metadata["semantic_seeded_territory_diagnostics"] = {
                        "report_version": "semantic_seeded_territory_variants_v1",
                        "analysis_only": True,
                        "used_for_solver_decision": False,
                        "used_for_adoption": False,
                        "used_for_main_path": False,
                        "enabled": True,
                        "dry_run": True,
                        "error": str(exc),
                    }

            should_adopt = (
                bool(enable_topology_assignment_cp_sat)
                and adoption_requested
                and not bool(topology_assignment_dry_run)
            )
            if should_adopt:
                if str(assignment_proposal.get("status", "")) != "success":
                    reason = str(assignment_proposal.get("reason", "proposal_failure") or "proposal_failure")
                    if bool(allow_topology_assignment_fallback):
                        primary.metadata["topology_assignment_adoption"] = _topology_assignment_adoption_record(
                            requested=True,
                            enabled=True,
                            applied=False,
                            used_for_main_path=False,
                            proposal=assignment_proposal,
                            fallback_to_heuristic=True,
                            adoption_failed_reason=str(adoption_gate.get("gate_block_reason") or f"proposal_{reason}"),
                            adoption_gate=adoption_gate,
                            fallback_kind="expected_proposal_failure",
                            fallback_reason=str(adoption_gate.get("gate_block_reason") or f"proposal_{reason}"),
                        )
                        logger.info("[TOPO-CP] Adoption fallback | reason=proposal_%s", reason)
                    else:
                        failure_adoption_record = _topology_assignment_adoption_record(
                            requested=True,
                            enabled=True,
                            applied=False,
                            used_for_main_path=False,
                            proposal=assignment_proposal,
                            adoption_failed_reason=str(adoption_gate.get("gate_block_reason") or f"proposal_{reason}"),
                            adoption_gate=adoption_gate,
                        )
                        raise _adoption_topology_error(
                            message=f"Topology assignment proposal failed: {reason}",
                            stage="topology_assignment_infeasible",
                            semantic_repair_allowed=True,
                            metadata={
                                "proposal_status": assignment_proposal.get("status"),
                                "proposal_reason": reason,
                                "topology_assignment_proposal": assignment_proposal,
                                "topology_assignment_adoption": failure_adoption_record,
                            },
                        )
                else:
                    selected_variant_id = str(assignment_proposal.get("selected_variant_id", "") or "")
                    selected_variant = next(
                        (variant for variant in list(report.variants or []) if str(variant.variant_id) == selected_variant_id),
                        None,
                    )
                    adoption_gate = _evaluate_topology_assignment_adoption_gate(
                        cp_sat_enabled=bool(enable_topology_assignment_cp_sat),
                        dry_run=bool(topology_assignment_dry_run),
                        adoption_enabled=adoption_requested,
                        fallback_allowed=bool(allow_topology_assignment_fallback),
                        proposal=assignment_proposal,
                        selected_variant=selected_variant,
                    )
                    if selected_variant is None:
                        error = _adoption_topology_error(
                            message=f"Topology assignment selected unknown variant: {selected_variant_id}",
                            stage="topology_assignment_adoption_inconsistent",
                            semantic_repair_allowed=False,
                            metadata={
                                "selected_variant_id": selected_variant_id,
                                "reason": "selected_variant_missing",
                                "topology_assignment_proposal": assignment_proposal,
                                "topology_assignment_adoption": _topology_assignment_adoption_record(
                                    requested=True,
                                    enabled=True,
                                    applied=False,
                                    used_for_main_path=False,
                                    proposal=assignment_proposal,
                                    adoption_failed_reason=str(adoption_gate.get("gate_block_reason") or "selected_variant_missing"),
                                    adoption_gate=adoption_gate,
                                ),
                            },
                        )
                        if bool(allow_topology_assignment_fallback):
                            primary.metadata["topology_assignment_adoption"] = _topology_assignment_adoption_record(
                                requested=True,
                                enabled=True,
                                applied=False,
                                used_for_main_path=False,
                                proposal=assignment_proposal,
                                fallback_to_heuristic=True,
                                adoption_failed_reason=str(adoption_gate.get("gate_block_reason") or error.metadata.get("reason", "selected_variant_missing")),
                                adoption_gate=adoption_gate,
                                fallback_kind="adoption_gate_blocked",
                                fallback_reason=str(adoption_gate.get("gate_block_reason") or error.metadata.get("reason", "selected_variant_missing")),
                            )
                            logger.warning("[TOPO-CP] Adoption fallback | reason=%s", error.metadata)
                        else:
                            raise error
                    else:
                        logger.info("[TOPO-CP] Adoption start | selected_variant=%s", selected_variant_id)
                        try:
                            assignment_proposal = dict(assignment_proposal)
                            assignment_proposal["adoption_gate"] = adoption_gate
                            primary.metadata["topology_assignment_proposal"] = assignment_proposal
                            return build_grid_growth_result_from_variant(
                                selected_variant=selected_variant,
                                proposed_cluster_to_island=dict(assignment_proposal.get("proposed_cluster_to_island") or {}),
                                room_specs=room_specs,
                                report=report,
                                primary_metadata=primary.metadata,
                                proposal=assignment_proposal,
                                floor_boundary=floor_boundary,
                                floor_id=f"F{int(floor_number or 1)}",
                            )
                        except LayoutTopologyError as exc:
                            meta = dict(getattr(exc, "metadata", {}) or {})
                            reason = str(meta.get("reason") or meta.get("stage") or exc)
                            failed_gate = _evaluate_topology_assignment_adoption_gate(
                                cp_sat_enabled=bool(enable_topology_assignment_cp_sat),
                                dry_run=bool(topology_assignment_dry_run),
                                adoption_enabled=adoption_requested,
                                fallback_allowed=bool(allow_topology_assignment_fallback),
                                proposal=assignment_proposal,
                                selected_variant=selected_variant,
                                consistency_check_pass=False,
                                consistency_detail={"reason": reason},
                            )
                            if bool(allow_topology_assignment_fallback):
                                primary.metadata["topology_assignment_adoption"] = _topology_assignment_adoption_record(
                                    requested=True,
                                    enabled=True,
                                    applied=False,
                                    used_for_main_path=False,
                                    selected_variant=selected_variant,
                                    proposal=assignment_proposal,
                                    fallback_to_heuristic=True,
                                    adoption_failed_reason=reason,
                                    adoption_gate=failed_gate,
                                    fallback_kind="unexpected_adoption_failure" if adoption_gate.get("gate_opened") else "adoption_gate_blocked",
                                    fallback_reason=str(failed_gate.get("gate_block_reason") or reason),
                                )
                                logger.warning("[TOPO-CP] Adoption fallback | reason=%s", reason)
                            else:
                                adoption_metadata = _topology_assignment_adoption_record(
                                    requested=True,
                                    enabled=True,
                                    applied=False,
                                    used_for_main_path=False,
                                    selected_variant=selected_variant,
                                    proposal=assignment_proposal,
                                    adoption_failed_reason=reason,
                                    adoption_gate=failed_gate,
                                )
                                try:
                                    exc.metadata.setdefault("topology_assignment_proposal", assignment_proposal)
                                    exc.metadata.setdefault("topology_assignment_adoption", adoption_metadata)
                                except Exception:
                                    pass
                                logger.error("[TOPO-CP] Adoption inconsistent | error=%s", exc)
                                raise
        except Exception as exc:
            if isinstance(exc, LayoutTopologyError):
                raise
            logger.warning("[TOPO] Variant diagnostics failed | error=%s", exc)
            primary.metadata["topology_variants_error"] = str(exc)
    return primary


def _rebuild_report_for_area_targets(
    report: TopologyFeasibilityReport,
    room_specs: Sequence[RoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]],
) -> TopologyFeasibilityReport:
    cluster_metrics = build_cluster_metrics(room_specs, adjacency_graph)
    rebuilt_variants: List[TopologyVariant] = []
    for variant in list(report.variants or []):
        feasibility = evaluate_cluster_island_feasibility(
            variant_id=str(variant.variant_id),
            cluster_metrics=cluster_metrics,
            island_metrics=list(variant.island_metrics or []),
        )
        rebuilt_variants.append(
            TopologyVariant(
                variant_id=variant.variant_id,
                seed=variant.seed,
                is_primary=variant.is_primary,
                primary_compatible=variant.primary_compatible,
                variant_profile=dict(variant.variant_profile or {}),
                corridor_skeleton=variant.corridor_skeleton,
                candidate_islands=variant.candidate_islands,
                island_metrics=list(variant.island_metrics or []),
                feasibility_matrix=feasibility,
                corridor_access_edges=list(variant.corridor_access_edges or []),
                facade_edges=list(variant.facade_edges or []),
                core_docking_candidates=list(variant.core_docking_candidates or []),
                core_contract_id=str(variant.core_contract_id or ""),
                core_union_hash=str(variant.core_union_hash or ""),
                corridor_area=float(variant.corridor_area or 0.0),
                valid=bool(variant.valid),
                rejection_reasons=list(variant.rejection_reasons or []),
            )
        )
    return TopologyFeasibilityReport(
        topology_seed_list=list(report.topology_seed_list or []),
        primary_variant_id=str(report.primary_variant_id or "topo_seed_0"),
        variants=rebuilt_variants,
        cluster_metrics=list(cluster_metrics),
    )


def _refresh_grid_growth_area_handoff(result: GridGrowthResult) -> None:
    room_by_id: Dict[str, RoomSpec] = {}
    for assignment in list(result.assignments.values() or []):
        for room in list(getattr(assignment, "rooms", []) or []):
            room_by_id[str(getattr(room, "room_id", "") or "")] = room
    island_by_id = {str(getattr(island, "id", "")): island for island in list(result.islands or [])}
    for island_id, assignment in list(result.assignments.items()):
        rooms = list(getattr(assignment, "rooms", []) or [])
        total_area = float(sum(float(getattr(room, "target_area", 0.0) or 0.0) for room in rooms))
        assignment.total_area = total_area
        island = island_by_id.get(str(island_id))
        island_area = float(getattr(island, "area", 0.0) or 0.0) if island is not None else 0.0
        assignment.utilization = total_area / island_area if island_area > 1e-9 else 0.0
        if island is not None:
            try:
                island.assigned_rooms = [str(getattr(room, "room_id", "")) for room in rooms]
                island.remaining_capacity = max(0.0, island_area - total_area)
                setattr(island, "remaining_capacity_area", island_area - total_area)
            except Exception:
                pass
    handoff = result.metadata.get("handoff") if isinstance(result.metadata, dict) else None
    if isinstance(handoff, list):
        for row in handoff:
            if not isinstance(row, dict):
                continue
            room_ids = [str(x) for x in list(row.get("rooms", []) or [])]
            target_area = float(sum(float(getattr(room_by_id.get(rid), "target_area", 0.0) or 0.0) for rid in room_ids))
            row["target_area"] = round(target_area, 3)
            area = float(row.get("area", 0.0) or 0.0)
            row["remaining_capacity_area"] = round(area - target_area, 3)
            if "effective_capacity_area" in row:
                row["remaining_effective_capacity_area"] = round(float(row.get("effective_capacity_area", 0.0) or 0.0) - target_area, 3)
    if isinstance(result.metadata, dict):
        result.metadata["assigned_room_count"] = sum(len(getattr(a, "rooms", []) or []) for a in result.assignments.values())


def _round_meta(value: Any, ndigits: int = 4) -> float:
    try:
        return round(float(value or 0.0), int(ndigits))
    except Exception:
        return 0.0


def _mapping_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _floor_id_for_metadata(floor_number: int) -> str:
    try:
        return f"F{int(floor_number or 1)}"
    except Exception:
        return "F1"


def _candidate_island_metadata(report: TopologyFeasibilityReport, *, floor_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in list(report.variants or []):
        variant_id = str(variant.variant_id)
        island_by_id = {
            str(getattr(island, "id", "")): island
            for island in list(getattr(variant, "candidate_islands", []) or [])
        }
        for metric in list(variant.island_metrics or []):
            island = island_by_id.get(str(metric.island_id))
            generation_source = str(
                getattr(island, "territory_generation_source", "")
                or getattr(island, "island_generation_source", "")
                or "usable_polygon_minus_corridor_core"
            )
            source_cluster_ids = [
                str(cid)
                for cid in list(
                    getattr(island, "source_cluster_ids", [])
                    or getattr(island, "generated_from_cluster_ids", [])
                    or []
                )
                if str(cid)
            ]
            seed_cluster_id = getattr(island, "seed_cluster_id", None)
            if seed_cluster_id and str(seed_cluster_id) not in source_cluster_ids:
                source_cluster_ids.append(str(seed_cluster_id))
            provenance_type = (
                "semantic_cluster_grown"
                if source_cluster_ids or generation_source == "semantic_cluster_seeded"
                else "corridor_partitioned"
            )
            rows.append(
                {
                    "floor_id": str(floor_id),
                    "variant_id": variant_id,
                    "island_id": str(metric.island_id),
                    "canonical_island_id": f"{floor_id}:{variant_id}:{metric.island_id}",
                    "variant_family": str((variant.variant_profile or {}).get("variant_family") or "corridor_partitioned"),
                    "semantic_growth_algorithm_version": str((variant.variant_profile or {}).get("semantic_growth_algorithm_version") or ""),
                    "parent_variant_id": str((variant.variant_profile or {}).get("parent_variant_id") or ""),
                    "corridor_source_variant_id": str((variant.variant_profile or {}).get("corridor_source_variant_id") or ""),
                    "island_generation_source": generation_source,
                    "provenance_type": provenance_type,
                    "source_cluster_ids": sorted(set(source_cluster_ids)),
                    "seed_cluster_id": str(seed_cluster_id or "") or None,
                    "target_area": _round_meta(getattr(island, "target_area", 0.0)) if island is not None else 0.0,
                    "min_area": _round_meta(getattr(island, "min_area", 0.0)) if island is not None else 0.0,
                    "raw_area_margin": _round_meta(getattr(island, "raw_area_margin", 0.0)) if island is not None else 0.0,
                    "target_margin": _round_meta(getattr(island, "target_margin", 0.0)) if island is not None else 0.0,
                    "min_margin": _round_meta(getattr(island, "min_margin", 0.0)) if island is not None else 0.0,
                    "territory_shortfall_area": _round_meta(getattr(island, "territory_shortfall_area", 0.0)) if island is not None else 0.0,
                    "seed_status": str(getattr(island, "seed_status", "") or ""),
                    "seed_failure_reason": str(getattr(island, "seed_failure_reason", "") or ""),
                    "growth_failure_reason": str(getattr(island, "growth_failure_reason", "") or ""),
                    "frontier_exhausted": bool(getattr(island, "frontier_exhausted", False)) if island is not None else False,
                    "geometry_status": str(getattr(island, "geometry_status", "") or ""),
                    "territory_connected": bool(getattr(island, "territory_connected", True)) if island is not None else True,
                    "component_count": int(getattr(island, "component_count", 1) or 1) if island is not None else 1,
                    "largest_component_area": _round_meta(getattr(island, "largest_component_area", metric.area)),
                    "discarded_component_area": _round_meta(getattr(island, "discarded_component_area", 0.0)),
                    "discarded_component_reason": str(getattr(island, "discarded_component_reason", "") or ""),
                    "core_overlap_area": _round_meta(getattr(island, "core_overlap_area", 0.0)),
                    "corridor_overlap_area": _round_meta(getattr(island, "corridor_overlap_area", 0.0)),
                    "actual_area": _round_meta(metric.area),
                    "effective_capacity": _round_meta(metric.effective_capacity_area),
                    "effective_capacity_area": _round_meta(metric.effective_capacity_area),
                    "core_contract_id": str(metric.core_contract_id or variant.core_contract_id or ""),
                    "core_union_hash": str(metric.core_union_hash or variant.core_union_hash or ""),
                }
            )
    return sorted(rows, key=lambda row: (row["variant_id"], row["island_id"]))


def _cluster_metrics_by_id(report: TopologyFeasibilityReport) -> Dict[str, Any]:
    return {str(cluster.cluster_id): cluster for cluster in list(report.cluster_metrics or [])}


def _island_metrics_by_key(report: TopologyFeasibilityReport) -> Dict[Tuple[str, str], Any]:
    out: Dict[Tuple[str, str], Any] = {}
    for variant in list(report.variants or []):
        for metric in list(variant.island_metrics or []):
            out[(str(variant.variant_id), str(metric.island_id))] = metric
    return out


def _assigned_load_by_island(
    result: GridGrowthResult,
    report: TopologyFeasibilityReport,
    *,
    floor_id: str,
    load_source: str = "heuristic_primary",
) -> Dict[str, Any]:
    primary_variant_id = str(report.primary_variant_id or "topo_seed_0")
    primary = next((variant for variant in list(report.variants or []) if str(variant.variant_id) == primary_variant_id), None)
    island_by_id = {
        str(metric.island_id): metric
        for metric in list(getattr(primary, "island_metrics", []) or [])
    }
    cluster_by_room: Dict[str, Any] = {}
    for cluster in list(report.cluster_metrics or []):
        for room_id in list(cluster.room_ids or []):
            cluster_by_room[str(room_id)] = cluster
    items: List[Dict[str, Any]] = []
    for island_id, assignment in sorted(dict(result.assignments or {}).items(), key=lambda item: str(item[0])):
        metric = island_by_id.get(str(island_id))
        if metric is None:
            continue
        clusters: Dict[str, Any] = {}
        assigned_room_ids: List[str] = []
        for room in list(getattr(assignment, "rooms", []) or []):
            room_id = str(getattr(room, "room_id", room))
            assigned_room_ids.append(room_id)
            cluster = cluster_by_room.get(room_id)
            if cluster is not None:
                clusters[str(cluster.cluster_id)] = cluster
        target = sum(float(getattr(cluster, "target_area_sum", 0.0) or 0.0) for cluster in clusters.values())
        min_sum = sum(float(getattr(cluster, "min_area_sum", 0.0) or 0.0) for cluster in clusters.values())
        effective = float(getattr(metric, "effective_capacity_area", 0.0) or 0.0)
        margin = effective - target
        items.append(
            {
                "floor_id": str(floor_id),
                "variant_id": primary_variant_id,
                "island_id": str(island_id),
                "canonical_island_id": f"{floor_id}:{primary_variant_id}:{island_id}",
                "assigned_cluster_ids": sorted(clusters),
                "assigned_room_ids": sorted(set(assigned_room_ids)),
                "assigned_target_sum": _round_meta(target),
                "assigned_min_sum": _round_meta(min_sum),
                "effective_capacity": _round_meta(effective),
                "effective_capacity_area": _round_meta(effective),
                "capacity_margin": _round_meta(margin),
                "overload_area": _round_meta(max(0.0, -margin)),
                "load_source": str(load_source),
            }
        )
    if not items:
        return {
            "available": False,
            "missing_reason": "assignment_handoff_absent",
            "load_source": str(load_source),
            "items": [],
        }
    return {
        "available": True,
        "load_source": str(load_source),
        "items": items,
    }


def _cluster_feasibility_summary(report: TopologyFeasibilityReport, *, floor_id: str) -> Dict[str, Any]:
    island_by_key = _island_metrics_by_key(report)
    rows_by_cluster: Dict[str, List[Any]] = defaultdict(list)
    rejection_counts_by_cluster: Dict[str, Counter[str]] = defaultdict(Counter)
    for variant in list(report.variants or []):
        for row in list(variant.feasibility_matrix or []):
            rows_by_cluster[str(row.cluster_id)].append(row)
            for reason in list(row.rejection_reasons or []):
                rejection_counts_by_cluster[str(row.cluster_id)][str(reason)] += 1
    items: List[Dict[str, Any]] = []
    for cluster_id, cluster in sorted(_cluster_metrics_by_id(report).items()):
        rows = rows_by_cluster.get(cluster_id, [])
        feasible_rows = [row for row in rows if bool(row.hard_feasible)]
        best_row = None
        if rows:
            best_row = sorted(
                rows,
                key=lambda row: (
                    -float(row.feasibility_score),
                    -float(row.capacity_margin),
                    str(row.variant_id),
                    str(row.island_id),
                ),
            )[0]
        max_effective = max(
            (
                float(getattr(metric, "effective_capacity_area", 0.0) or 0.0)
                for metric in island_by_key.values()
            ),
            default=0.0,
        )
        best_metric = island_by_key.get((str(getattr(best_row, "variant_id", "")), str(getattr(best_row, "island_id", "")))) if best_row is not None else None
        items.append(
            {
                "floor_id": str(floor_id),
                "variant_id": "all_variants",
                "cluster_id": cluster_id,
                "room_ids": list(cluster.room_ids or []),
                "target_sum": _round_meta(cluster.target_area_sum),
                "min_sum": _round_meta(cluster.min_area_sum),
                "feasible_island_count": len(feasible_rows),
                "best_candidate_variant_id": str(getattr(best_row, "variant_id", "") or "") if best_row is not None else "",
                "best_candidate_island_id": str(getattr(best_row, "island_id", "") or "") if best_row is not None else "",
                "best_capacity_margin": _round_meta(getattr(best_row, "capacity_margin", 0.0)) if best_row is not None else 0.0,
                "best_candidate_effective_capacity": _round_meta(getattr(best_metric, "effective_capacity_area", 0.0)) if best_metric is not None else 0.0,
                "max_island_effective_capacity": _round_meta(max_effective),
                "target_too_large_for_any_island": bool(float(cluster.target_area_sum) > max_effective + 1e-6) if max_effective > 0 else False,
                "min_too_large_for_any_island": bool(float(cluster.min_area_sum) > max_effective + 1e-6) if max_effective > 0 else False,
                "failed_constraint_counts": dict(sorted(rejection_counts_by_cluster.get(cluster_id, Counter()).items())),
            }
        )
    return {
        "available": bool(items),
        "floor_id": str(floor_id),
        "scope": "all_variants",
        "items": items,
    }


def _corridor_evidence(
    result: GridGrowthResult,
    *,
    floor_id: str,
    floor_boundary: Optional[Polygon] = None,
) -> Dict[str, Any]:
    corridor_meta = result.metadata.get("corridor") if isinstance(result.metadata.get("corridor"), Mapping) else {}
    corridor_area = float(corridor_meta.get("area", 0.0) or 0.0)
    core_overlap = float(corridor_meta.get("core_overlap_after", corridor_meta.get("corridor_core_overlap_area", 0.0)) or 0.0)
    corridor_ids = [str(c.id) for c in list(result.corridors or [])]
    connected_island_ids = sorted(str(island_id) for island_id in dict(result.assignments or {}).keys())
    cluster_ids: Set[str] = set()
    for row in list(result.metadata.get("handoff", []) or []):
        if isinstance(row, Mapping):
            cluster_ids.update(str(x) for x in list(row.get("clusters", []) or []) if str(x))
    touches_core = bool(result.metadata.get("core_docking_candidates")) or any("core" in cid.lower() for cid in corridor_ids)
    touches_exterior = False
    if floor_boundary is not None:
        try:
            boundary = floor_boundary.boundary
            touches_exterior = any(float(corridor.polygon.boundary.intersection(boundary).length) > 1e-6 for corridor in list(result.corridors or []))
        except Exception:
            touches_exterior = False
    return {
        "available": True,
        "floor_id": str(floor_id),
        "corridor_area": _round_meta(corridor_area),
        "corridor_core_overlap_area": _round_meta(core_overlap),
        "touches_core": bool(touches_core),
        "touches_exterior": bool(touches_exterior),
        "connected_island_ids": connected_island_ids,
        "connected_cluster_ids": sorted(cluster_ids),
        "evidence_source": "polygon_touch" if touches_exterior else "metadata",
    }


def _refresh_grid_growth_metadata_gap_closure(
    result: GridGrowthResult,
    report: TopologyFeasibilityReport,
    *,
    floor_number: int,
    floor_boundary: Optional[Polygon] = None,
    load_source: str = "heuristic_primary",
) -> None:
    if not isinstance(result.metadata, dict):
        return
    floor_id = _floor_id_for_metadata(floor_number)
    result.metadata["candidate_island_metadata"] = _candidate_island_metadata(report, floor_id=floor_id)
    result.metadata["assigned_load_by_island"] = _assigned_load_by_island(
        result,
        report,
        floor_id=floor_id,
        load_source=load_source,
    )
    result.metadata["cluster_feasibility_summary"] = _cluster_feasibility_summary(report, floor_id=floor_id)
    result.metadata["corridor_evidence"] = _corridor_evidence(
        result,
        floor_id=floor_id,
        floor_boundary=floor_boundary,
    )


def _area_signature(islands: Sequence[Island]) -> List[float]:
    return sorted(round(float(getattr(i, "area", 0.0) or 0.0), 3) for i in islands or [])


def _safe_corridor_area(corridors: Sequence[Corridor], floor: Polygon) -> float:
    polys = [getattr(c, "polygon", None) for c in corridors or [] if getattr(c, "polygon", None) is not None]
    if not polys:
        return 0.0
    try:
        return float(unary_union(polys).intersection(floor).area)
    except Exception:
        return float(sum(float(getattr(p, "area", 0.0) or 0.0) for p in polys))


def _semantic_core_union(core_tube: CoreTube, core_contract: Optional[CoreFootprintContract]) -> Any:
    core_union = getattr(core_contract, "core_union", None) if core_contract is not None else None
    if core_union is None or getattr(core_union, "is_empty", True):
        core_union = getattr(core_tube, "polygon", None)
    return core_union if core_union is not None else GeometryCollection()


def _semantic_base_usable_polygon(
    floor_boundary: Polygon,
    core_tube: CoreTube,
    core_contract: Optional[CoreFootprintContract],
    floor_usable_polygon: Optional[Any],
) -> Any:
    if floor_usable_polygon is not None and not getattr(floor_usable_polygon, "is_empty", True):
        return floor_usable_polygon
    core_union = _semantic_core_union(core_tube, core_contract)
    if core_union is None or getattr(core_union, "is_empty", True):
        return floor_boundary
    try:
        return floor_boundary.difference(core_union)
    except Exception:
        return floor_boundary


def _semantic_free_polygon(
    *,
    floor_boundary: Polygon,
    core_tube: CoreTube,
    core_contract: Optional[CoreFootprintContract],
    floor_usable_polygon: Optional[Any],
    corridors: Sequence[Corridor],
) -> Any:
    usable = _semantic_base_usable_polygon(floor_boundary, core_tube, core_contract, floor_usable_polygon)
    blocked = [getattr(c, "polygon", None) for c in corridors or [] if getattr(c, "polygon", None) is not None]
    blocked.append(_semantic_core_union(core_tube, core_contract))
    blocked = [g for g in blocked if g is not None and not getattr(g, "is_empty", True)]
    if not blocked:
        return usable
    try:
        return usable.difference(unary_union(blocked)).buffer(0)
    except Exception:
        return usable


def _semantic_grid_cells(free_polygon: Any, resolution: float) -> Dict[Tuple[int, int], Polygon]:
    cells: Dict[Tuple[int, int], Polygon] = {}
    pieces = _polygon_pieces(free_polygon, min_area=0.01)
    if not pieces:
        return cells
    free = unary_union(pieces).buffer(0)
    minx, miny, maxx, maxy = (float(v) for v in free.bounds)
    step = max(0.25, float(resolution or DEFAULT_GROWTH_RESOLUTION))
    min_area = max(0.01, step * step * 0.15)
    ix0 = int(math.floor(minx / step)) - 1
    iy0 = int(math.floor(miny / step)) - 1
    ix1 = int(math.ceil(maxx / step)) + 1
    iy1 = int(math.ceil(maxy / step)) + 1
    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            cell = box(ix * step, iy * step, (ix + 1) * step, (iy + 1) * step)
            try:
                clipped = cell.intersection(free)
            except Exception:
                continue
            if getattr(clipped, "is_empty", True) or float(getattr(clipped, "area", 0.0) or 0.0) < min_area:
                continue
            polys = _polygon_pieces(clipped, min_area=min_area)
            if not polys:
                continue
            cells[(ix, iy)] = max(polys, key=lambda p: float(p.area)).buffer(0)
    return cells


def _semantic_cell_score(
    cluster: ClusterMetrics,
    cell: Polygon,
    *,
    floor_boundary: Polygon,
    corridor_union: Any,
    core_union: Any,
) -> Tuple[float, float, float, float]:
    centroid = cell.centroid
    facade_distance = float(centroid.distance(floor_boundary.exterior)) if int(cluster.needs_window_count or 0) > 0 else 0.0
    access_distance = (
        float(centroid.distance(corridor_union))
        if int(cluster.needs_corridor_access_count or 0) > 0 and corridor_union is not None and not getattr(corridor_union, "is_empty", True)
        else 0.0
    )
    core_distance = (
        float(centroid.distance(core_union))
        if core_union is not None and not getattr(core_union, "is_empty", True)
        else 0.0
    )
    return (
        facade_distance,
        access_distance,
        core_distance,
        -float(cell.area),
    )


def _semantic_neighbor_keys(key: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = key
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def _largest_component_info(geom: Any) -> Tuple[Optional[Polygon], int, float, float, bool]:
    pieces = sorted(_polygon_pieces(geom, min_area=0.01), key=lambda p: float(p.area), reverse=True)
    if not pieces:
        return None, 0, 0.0, 0.0, False
    largest = pieces[0].buffer(0)
    total = sum(float(p.area) for p in pieces)
    largest_area = float(largest.area)
    discarded = max(0.0, total - largest_area)
    return largest, len(pieces), largest_area, discarded, len(pieces) == 1


def _semantic_cluster_category(cluster: ClusterMetrics) -> str:
    if int(cluster.public_room_count or 0) > 0:
        return "public"
    if int(cluster.service_room_count or 0) > 0 and int(cluster.private_room_count or 0) <= 0:
        return "service"
    if int(cluster.private_room_count or 0) > 0:
        return "private"
    return "unknown"


def _grow_semantic_territories_for_parent_v0(
    *,
    parent_variant: TopologyVariant,
    variant_id: str,
    floor_boundary: Polygon,
    core_tube: CoreTube,
    core_contract: Optional[CoreFootprintContract],
    floor_usable_polygon: Optional[Any],
    cluster_metrics: Sequence[ClusterMetrics],
    resolution: float,
    floor_id: str,
) -> Tuple[List[Island], List[Dict[str, Any]], Dict[str, Any]]:
    corridors = list(parent_variant.corridor_skeleton or [])
    corridor_polys = [getattr(c, "polygon", None) for c in corridors if getattr(c, "polygon", None) is not None]
    corridor_union = unary_union(corridor_polys) if corridor_polys else GeometryCollection()
    core_union = _semantic_core_union(core_tube, core_contract)
    free_polygon = _semantic_free_polygon(
        floor_boundary=floor_boundary,
        core_tube=core_tube,
        core_contract=core_contract,
        floor_usable_polygon=floor_usable_polygon,
        corridors=corridors,
    )
    cells = _semantic_grid_cells(free_polygon, resolution)
    available = set(cells)
    islands: List[Island] = []
    cluster_reports: List[Dict[str, Any]] = []
    ordered_clusters = sorted(
        list(cluster_metrics or []),
        key=lambda c: (
            -int(c.needs_window_count or 0),
            -int(c.needs_corridor_access_count or 0),
            -float(c.target_area_sum or 0.0),
            str(c.cluster_id),
        ),
    )
    for cluster in ordered_clusters:
        report: Dict[str, Any] = {
            "floor_id": floor_id,
            "variant_id": variant_id,
            "semantic_growth_algorithm_version": "v0",
            "parent_variant_id": parent_variant.variant_id,
            "cluster_id": str(cluster.cluster_id),
            "cluster_room_ids": list(cluster.room_ids or []),
            "target_area": _round_meta(cluster.target_area_sum),
            "min_area": _round_meta(cluster.min_area_sum),
            "needs_window": bool(int(cluster.needs_window_count or 0) > 0),
            "needs_corridor_access": bool(int(cluster.needs_corridor_access_count or 0) > 0),
            "category": _semantic_cluster_category(cluster),
            "seed_status": "failed",
            "seed_failure_reason": "",
        }
        if not available:
            report["seed_failure_reason"] = "no_free_cells"
            cluster_reports.append(report)
            continue
        seed_candidates = sorted(
            available,
            key=lambda key: (
                _semantic_cell_score(
                    cluster,
                    cells[key],
                    floor_boundary=floor_boundary,
                    corridor_union=corridor_union,
                    core_union=core_union,
                ),
                key,
            ),
        )
        seed_key = seed_candidates[0] if seed_candidates else None
        if seed_key is None:
            report["seed_failure_reason"] = "unknown"
            cluster_reports.append(report)
            continue
        selected = {seed_key}
        available.remove(seed_key)
        target_area = float(cluster.target_area_sum or 0.0)
        selected_area = float(cells[seed_key].area)
        frontier = set(k for k in _semantic_neighbor_keys(seed_key) if k in available)
        seed_centroid = cells[seed_key].centroid
        while selected_area + 1e-6 < target_area and frontier:
            next_key = sorted(
                frontier,
                key=lambda key: (
                    float(cells[key].centroid.distance(seed_centroid)),
                    _semantic_cell_score(
                        cluster,
                        cells[key],
                        floor_boundary=floor_boundary,
                        corridor_union=corridor_union,
                        core_union=core_union,
                    ),
                    key,
                ),
            )[0]
            frontier.remove(next_key)
            if next_key not in available:
                continue
            selected.add(next_key)
            available.remove(next_key)
            selected_area += float(cells[next_key].area)
            for neighbor in _semantic_neighbor_keys(next_key):
                if neighbor in available:
                    frontier.add(neighbor)
        union = unary_union([cells[key] for key in selected]).buffer(0)
        polygon, component_count, largest_area, discarded_area, connected = _largest_component_info(union)
        if polygon is None or polygon.is_empty:
            report["seed_failure_reason"] = "unknown"
            cluster_reports.append(report)
            continue
        island = Island(id=f"semantic_island_{cluster.cluster_id}", polygon=polygon)
        target_margin = float(polygon.area) - target_area
        min_margin = float(polygon.area) - float(cluster.min_area_sum or 0.0)
        territory_shortfall = max(0.0, target_area - float(polygon.area))
        try:
            island.territory_generation_source = "semantic_cluster_seeded"
            island.island_generation_source = "semantic_cluster_seeded"
            island.variant_family = "semantic_seeded_territory"
            island.semantic_growth_algorithm_version = "v0"
            island.parent_variant_id = str(parent_variant.variant_id)
            island.corridor_source_variant_id = str(parent_variant.variant_id)
            island.source_cluster_ids = [str(cluster.cluster_id)]
            island.generated_from_cluster_ids = [str(cluster.cluster_id)]
            island.seed_cluster_id = str(cluster.cluster_id)
            island.target_area = target_area
            island.min_area = float(cluster.min_area_sum or 0.0)
            island.raw_area_margin = target_margin
            island.target_margin = target_margin
            island.min_margin = min_margin
            island.territory_shortfall_area = territory_shortfall
            island.seed_status = "placed"
            island.seed_failure_reason = "" if territory_shortfall <= 1e-6 else "insufficient_area"
            island.growth_failure_reason = "" if territory_shortfall <= 1e-6 else "frontier_exhausted_before_target"
            island.frontier_exhausted = bool(territory_shortfall > 1e-6)
            island.territory_connected = bool(connected)
            island.component_count = int(component_count)
            island.largest_component_area = float(largest_area)
            island.discarded_component_area = float(discarded_area)
            island.discarded_component_reason = "kept_largest_component" if discarded_area > 1e-6 else ""
            island.geometry_status = "valid" if bool(getattr(polygon, "is_valid", True)) else "invalid_territory_geometry"
            island.territory_type = "room_territory"
            island.effective_capacity = float(polygon.area)
        except Exception:
            pass
        islands.append(island)
        report.update(
            {
                "seed_status": "placed",
                "seed_failure_reason": "" if territory_shortfall <= 1e-6 else "insufficient_area",
                "island_id": island.id,
                "actual_area": _round_meta(polygon.area),
                "effective_capacity": _round_meta(polygon.area),
                "target_margin": _round_meta(target_margin),
                "min_margin": _round_meta(min_margin),
                "raw_area_margin": _round_meta(target_margin),
                "territory_shortfall_area": _round_meta(territory_shortfall),
                "growth_failure_reason": "" if territory_shortfall <= 1e-6 else "frontier_exhausted_before_target",
                "frontier_exhausted": bool(territory_shortfall > 1e-6),
                "territory_connected": bool(connected),
                "component_count": int(component_count),
                "largest_component_area": _round_meta(largest_area),
                "discarded_component_area": _round_meta(discarded_area),
                "discarded_component_reason": "kept_largest_component" if discarded_area > 1e-6 else "",
                "geometry_status": "valid" if bool(getattr(polygon, "is_valid", True)) else "invalid_territory_geometry",
            }
        )
        cluster_reports.append(report)
    diagnostics = {
        "floor_id": floor_id,
        "variant_id": variant_id,
        "variant_family": "semantic_seeded_territory",
        "semantic_growth_algorithm_version": "v0",
        "parent_variant_id": str(parent_variant.variant_id),
        "corridor_source_variant_id": str(parent_variant.variant_id),
        "free_cell_count": len(cells),
        "generated_island_count": len(islands),
        "cluster_generation": sorted(cluster_reports, key=lambda row: str(row.get("cluster_id") or "")),
    }
    return islands, cluster_reports, diagnostics


def _semantic_component_labels(cells: Mapping[Tuple[int, int], Polygon]) -> Tuple[Dict[Tuple[int, int], int], Dict[int, float]]:
    labels: Dict[Tuple[int, int], int] = {}
    component_area: Dict[int, float] = {}
    component_id = 0
    for start in sorted(cells):
        if start in labels:
            continue
        component_id += 1
        stack = [start]
        labels[start] = component_id
        area = 0.0
        while stack:
            key = stack.pop()
            area += float(cells[key].area)
            for neighbor in _semantic_neighbor_keys(key):
                if neighbor in cells and neighbor not in labels:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
        component_area[component_id] = area
    return labels, component_area


def _semantic_growth_deficit(state: Mapping[str, Any], phase: str) -> float:
    area = float(state.get("area", 0.0) or 0.0)
    min_area = float(state.get("min_area", 0.0) or 0.0)
    target_area = float(state.get("target_area", 0.0) or 0.0)
    if phase == "min":
        return max(0.0, min_area - area) / max(min_area, 1e-6)
    if area + 1e-6 < min_area:
        return 0.0
    return max(0.0, target_area - area) / max(target_area, 1e-6)


def _semantic_growth_failure_reason(
    *,
    seed_status: str,
    area: float,
    min_area: float,
    target_area: float,
    frontier: Sequence[Any],
    stopped_by: str,
    residual_reasons: Sequence[str],
    valid_polygon: bool,
) -> str:
    if seed_status != "placed":
        return "seed_placement_failed"
    if not bool(valid_polygon):
        return "invalid_territory_geometry"
    if area + 1e-6 < min_area:
        if stopped_by in {"max_iterations", "max_cells", "no_progress"}:
            return "competition_starvation"
        if not frontier:
            return "frontier_exhausted_before_min"
        return "cluster_min_too_large_for_component"
    if area + 1e-6 < target_area:
        if "not_adjacent_to_any_territory" in set(residual_reasons):
            return "residual_unusable"
        if stopped_by in {"max_iterations", "max_cells", "no_progress"}:
            return "competition_starvation"
        if not frontier:
            return "frontier_exhausted_before_target"
        return "cluster_target_too_large_for_component"
    return ""


def _grow_semantic_territories_for_parent_v1(
    *,
    parent_variant: TopologyVariant,
    variant_id: str,
    floor_boundary: Polygon,
    core_tube: CoreTube,
    core_contract: Optional[CoreFootprintContract],
    floor_usable_polygon: Optional[Any],
    cluster_metrics: Sequence[ClusterMetrics],
    resolution: float,
    floor_id: str,
    max_iterations: int,
    max_cells: int,
    max_residual_passes: int,
) -> Tuple[List[Island], List[Dict[str, Any]], Dict[str, Any]]:
    corridors = list(parent_variant.corridor_skeleton or [])
    corridor_polys = [getattr(c, "polygon", None) for c in corridors if getattr(c, "polygon", None) is not None]
    corridor_union = unary_union(corridor_polys) if corridor_polys else GeometryCollection()
    core_union = _semantic_core_union(core_tube, core_contract)
    free_polygon = _semantic_free_polygon(
        floor_boundary=floor_boundary,
        core_tube=core_tube,
        core_contract=core_contract,
        floor_usable_polygon=floor_usable_polygon,
        corridors=corridors,
    )
    cells = _semantic_grid_cells(free_polygon, resolution)
    component_labels, component_area = _semantic_component_labels(cells)
    available: Set[Tuple[int, int]] = set(cells)
    states: Dict[str, Dict[str, Any]] = {}
    reports: Dict[str, Dict[str, Any]] = {}
    ordered_clusters = sorted(
        list(cluster_metrics or []),
        key=lambda c: (
            -int(c.needs_window_count or 0),
            -int(c.needs_corridor_access_count or 0),
            -float(c.target_area_sum or 0.0),
            str(c.cluster_id),
        ),
    )

    for cluster in ordered_clusters:
        cluster_id = str(cluster.cluster_id)
        report: Dict[str, Any] = {
            "floor_id": floor_id,
            "variant_id": variant_id,
            "semantic_growth_algorithm_version": "v1",
            "parent_variant_id": parent_variant.variant_id,
            "cluster_id": cluster_id,
            "cluster_room_ids": list(cluster.room_ids or []),
            "target_area": _round_meta(cluster.target_area_sum),
            "min_area": _round_meta(cluster.min_area_sum),
            "needs_window": bool(int(cluster.needs_window_count or 0) > 0),
            "needs_corridor_access": bool(int(cluster.needs_corridor_access_count or 0) > 0),
            "category": _semantic_cluster_category(cluster),
            "seed_status": "failed",
            "seed_failure_reason": "",
            "growth_failure_reason": "",
            "frontier_exhausted": False,
            "free_component_id": None,
            "component_area_available": 0.0,
            "component_area_used": 0.0,
            "component_area_remaining": 0.0,
            "geometry_status": "missing",
        }
        reports[cluster_id] = report
        if not available:
            report["seed_failure_reason"] = "no_free_cells"
            report["growth_failure_reason"] = "seed_placement_failed"
            continue
        seed_candidates = sorted(
            available,
            key=lambda key: (
                _semantic_cell_score(
                    cluster,
                    cells[key],
                    floor_boundary=floor_boundary,
                    corridor_union=corridor_union,
                    core_union=core_union,
                ),
                key,
            ),
        )
        seed_key = seed_candidates[0] if seed_candidates else None
        if seed_key is None:
            report["seed_failure_reason"] = "unknown"
            report["growth_failure_reason"] = "seed_placement_failed"
            continue
        available.remove(seed_key)
        component_id = int(component_labels.get(seed_key, 0) or 0)
        seed_area = float(cells[seed_key].area)
        states[cluster_id] = {
            "cluster": cluster,
            "cluster_id": cluster_id,
            "selected": {seed_key},
            "frontier": set(k for k in _semantic_neighbor_keys(seed_key) if k in available),
            "seed_key": seed_key,
            "seed_centroid": cells[seed_key].centroid,
            "area": seed_area,
            "target_area": float(cluster.target_area_sum or 0.0),
            "min_area": float(cluster.min_area_sum or 0.0),
            "component_id": component_id,
            "residual_unusable_reasons": [],
        }
        report.update(
            {
                "seed_status": "placed",
                "seed_failure_reason": "",
                "free_component_id": component_id,
                "component_area_available": _round_meta(component_area.get(component_id, 0.0)),
            }
        )

    def _remove_from_other_frontiers(chosen: Tuple[int, int]) -> None:
        for other in states.values():
            frontier = other.get("frontier")
            if isinstance(frontier, set):
                frontier.discard(chosen)

    def _add_cell(state: Dict[str, Any], key: Tuple[int, int]) -> None:
        if key not in available:
            return
        selected = state["selected"]
        frontier = state["frontier"]
        selected.add(key)
        available.remove(key)
        frontier.discard(key)
        _remove_from_other_frontiers(key)
        state["area"] = float(state.get("area", 0.0) or 0.0) + float(cells[key].area)
        for neighbor in _semantic_neighbor_keys(key):
            if neighbor in available:
                frontier.add(neighbor)

    def _best_frontier_cell(state: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        frontier = [key for key in list(state.get("frontier", set()) or set()) if key in available]
        if not frontier:
            return None
        cluster = state["cluster"]
        seed_centroid = state["seed_centroid"]
        selected = set(state.get("selected", set()) or set())
        other_frontier_counts: Dict[Tuple[int, int], int] = Counter()
        for other_id, other in states.items():
            if other_id == state["cluster_id"]:
                continue
            for key in list(other.get("frontier", set()) or set()):
                if key in available:
                    other_frontier_counts[key] += 1
        return sorted(
            frontier,
            key=lambda key: (
                -sum(1 for neighbor in _semantic_neighbor_keys(key) if neighbor in selected),
                other_frontier_counts.get(key, 0),
                float(cells[key].centroid.distance(seed_centroid)),
                _semantic_cell_score(
                    cluster,
                    cells[key],
                    floor_boundary=floor_boundary,
                    corridor_union=corridor_union,
                    core_union=core_union,
                ),
                key,
            ),
        )[0]

    stopped_by = "complete"
    iterations = 0
    max_iterations = max(1, int(max_iterations or 20000))
    max_cells = max(1, int(max_cells or 100000))
    assigned_cell_count = sum(len(state.get("selected", set()) or set()) for state in states.values())

    for phase in ("min", "target"):
        no_progress_steps = 0
        while True:
            candidates = [
                state
                for state in states.values()
                if _semantic_growth_deficit(state, phase) > 1e-6
                and any(key in available for key in list(state.get("frontier", set()) or set()))
            ]
            if not candidates:
                break
            state = sorted(
                candidates,
                key=lambda item: (
                    -_semantic_growth_deficit(item, phase),
                    float(item.get("area", 0.0) or 0.0),
                    str(item.get("cluster_id")),
                ),
            )[0]
            next_key = _best_frontier_cell(state)
            if next_key is None:
                no_progress_steps += 1
                if no_progress_steps >= max(1, len(states)):
                    stopped_by = "no_progress"
                    break
                continue
            before_area = float(state.get("area", 0.0) or 0.0)
            _add_cell(state, next_key)
            assigned_cell_count += 1
            iterations += 1
            if float(state.get("area", 0.0) or 0.0) <= before_area + 1e-9:
                no_progress_steps += 1
            else:
                no_progress_steps = 0
            if iterations >= max_iterations:
                stopped_by = "max_iterations"
                break
            if assigned_cell_count >= max_cells:
                stopped_by = "max_cells"
                break
        if stopped_by != "complete":
            break

    residual_reasons = Counter()
    residual_assignments = 0
    for _pass in range(max(0, int(max_residual_passes or 0))):
        made_progress = False
        for key in sorted(list(available)):
            if not any(_semantic_growth_deficit(state, "target") > 1e-6 for state in states.values()):
                break
            adjacent_states = [
                state
                for state in states.values()
                if key in available
                and _semantic_growth_deficit(state, "target") > 1e-6
                and any(neighbor in state.get("selected", set()) for neighbor in _semantic_neighbor_keys(key))
            ]
            if not adjacent_states:
                residual_reasons["not_adjacent_to_any_territory"] += 1
                continue
            state = sorted(
                adjacent_states,
                key=lambda item: (
                    -_semantic_growth_deficit(item, "target"),
                    float(item.get("area", 0.0) or 0.0),
                    str(item.get("cluster_id")),
                ),
            )[0]
            _add_cell(state, key)
            residual_assignments += 1
            made_progress = True
        if not made_progress:
            break

    islands: List[Island] = []
    cluster_reports: List[Dict[str, Any]] = []
    for cluster in ordered_clusters:
        cluster_id = str(cluster.cluster_id)
        report = reports.get(cluster_id, {})
        state = states.get(cluster_id)
        if state is None:
            cluster_reports.append(report)
            continue
        selected = sorted(list(state.get("selected", set()) or set()))
        union = unary_union([cells[key] for key in selected]).buffer(0) if selected else GeometryCollection()
        polygon, component_count, largest_area, discarded_area, connected = _largest_component_info(union)
        if polygon is None or getattr(polygon, "is_empty", True):
            report["growth_failure_reason"] = "invalid_territory_geometry"
            report["geometry_status"] = "invalid_territory_geometry"
            cluster_reports.append(report)
            continue
        actual_area = float(polygon.area)
        target_area = float(cluster.target_area_sum or 0.0)
        min_area = float(cluster.min_area_sum or 0.0)
        raw_area_margin = actual_area - target_area
        target_margin = raw_area_margin
        min_margin = actual_area - min_area
        territory_shortfall = max(0.0, target_area - actual_area)
        core_overlap = float(polygon.intersection(core_union).area) if core_union is not None and not getattr(core_union, "is_empty", True) else 0.0
        corridor_overlap = float(polygon.intersection(corridor_union).area) if corridor_union is not None and not getattr(corridor_union, "is_empty", True) else 0.0
        valid_polygon = bool(getattr(polygon, "is_valid", True)) and core_overlap <= CORE_OVERLAP_EPSILON_AREA and corridor_overlap <= CORE_OVERLAP_EPSILON_AREA
        residual_list = list(state.get("residual_unusable_reasons", []) or []) + list(residual_reasons.keys())
        growth_failure = _semantic_growth_failure_reason(
            seed_status=str(report.get("seed_status") or ""),
            area=actual_area,
            min_area=min_area,
            target_area=target_area,
            frontier=list(state.get("frontier", set()) or set()),
            stopped_by=stopped_by,
            residual_reasons=residual_list,
            valid_polygon=valid_polygon,
        )
        island = Island(id=f"semantic_island_{cluster.cluster_id}", polygon=polygon)
        try:
            island.territory_generation_source = "semantic_cluster_seeded"
            island.island_generation_source = "semantic_cluster_seeded"
            island.variant_family = "semantic_seeded_territory"
            island.semantic_growth_algorithm_version = "v1"
            island.parent_variant_id = str(parent_variant.variant_id)
            island.corridor_source_variant_id = str(parent_variant.variant_id)
            island.source_cluster_ids = [cluster_id]
            island.generated_from_cluster_ids = [cluster_id]
            island.seed_cluster_id = cluster_id
            island.target_area = target_area
            island.min_area = min_area
            island.raw_area_margin = raw_area_margin
            island.target_margin = target_margin
            island.min_margin = min_margin
            island.territory_shortfall_area = territory_shortfall
            island.seed_status = "placed"
            island.seed_failure_reason = ""
            island.growth_failure_reason = growth_failure
            island.frontier_exhausted = bool(not state.get("frontier") and territory_shortfall > 1e-6)
            island.territory_connected = bool(connected)
            island.component_count = int(component_count)
            island.largest_component_area = float(largest_area)
            island.discarded_component_area = float(discarded_area)
            island.discarded_component_reason = "kept_largest_component" if discarded_area > 1e-6 else ""
            island.core_overlap_area = core_overlap
            island.corridor_overlap_area = corridor_overlap
            island.geometry_status = "valid" if valid_polygon else "invalid_territory_geometry"
            island.territory_type = "room_territory"
            island.effective_capacity = actual_area
        except Exception:
            pass
        islands.append(island)
        component_id = int(state.get("component_id", 0) or 0)
        component_available = float(component_area.get(component_id, 0.0) or 0.0)
        report.update(
            {
                "seed_status": "placed",
                "seed_failure_reason": "",
                "growth_failure_reason": growth_failure,
                "frontier_exhausted": bool(not state.get("frontier") and territory_shortfall > 1e-6),
                "island_id": island.id,
                "actual_area": _round_meta(actual_area),
                "effective_capacity": _round_meta(actual_area),
                "raw_area_margin": _round_meta(raw_area_margin),
                "target_margin": _round_meta(target_margin),
                "min_margin": _round_meta(min_margin),
                "territory_shortfall_area": _round_meta(territory_shortfall),
                "component_area_used": _round_meta(actual_area),
                "component_area_remaining": _round_meta(max(0.0, component_available - actual_area)),
                "territory_connected": bool(connected),
                "component_count": int(component_count),
                "largest_component_area": _round_meta(largest_area),
                "discarded_component_area": _round_meta(discarded_area),
                "discarded_component_reason": "kept_largest_component" if discarded_area > 1e-6 else "",
                "core_overlap_area": _round_meta(core_overlap),
                "corridor_overlap_area": _round_meta(corridor_overlap),
                "valid_polygon": bool(valid_polygon),
                "geometry_status": "valid" if valid_polygon else "invalid_territory_geometry",
            }
        )
        cluster_reports.append(report)

    if stopped_by == "complete":
        any_min_shortfall = any(float(row.get("actual_area", 0.0) or 0.0) + 1e-6 < float(row.get("min_area", 0.0) or 0.0) for row in cluster_reports)
        any_target_shortfall = any(float(row.get("actual_area", 0.0) or 0.0) + 1e-6 < float(row.get("target_area", 0.0) or 0.0) for row in cluster_reports)
        if any_min_shortfall or any_target_shortfall:
            stopped_by = "frontier_exhausted"

    diagnostics = {
        "report_version": "semantic_seeded_growth_v1",
        "analysis_only": True,
        "used_for_solver_decision": False,
        "used_for_adoption": False,
        "floor_id": floor_id,
        "variant_id": variant_id,
        "variant_family": "semantic_seeded_territory",
        "semantic_growth_algorithm_version": "v1",
        "parent_variant_id": str(parent_variant.variant_id),
        "corridor_source_variant_id": str(parent_variant.variant_id),
        "free_cell_count": len(cells),
        "generated_island_count": len(islands),
        "assigned_cell_count": int(assigned_cell_count),
        "residual_assignment_count": int(residual_assignments),
        "residual_unusable_reasons": dict(sorted(residual_reasons.items())),
        "stopped_by": stopped_by,
        "iterations": int(iterations),
        "cluster_generation": sorted(cluster_reports, key=lambda row: str(row.get("cluster_id") or "")),
        "per_cluster_growth_audit": sorted(cluster_reports, key=lambda row: str(row.get("cluster_id") or "")),
    }
    return islands, cluster_reports, diagnostics


def build_semantic_seeded_territory_variants(
    *,
    floor_boundary: Polygon,
    core_tube: CoreTube,
    room_specs: Sequence[RoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]],
    corridor_width: float,
    corridor_layout: str,
    floor_number: Optional[int],
    config: Optional[GridGrowthConfig],
    core_contract: Optional[CoreFootprintContract],
    floor_usable_polygon: Optional[Any],
    active_report: TopologyFeasibilityReport,
) -> Tuple[TopologyFeasibilityReport, Dict[str, Any], TopologyFeasibilityReport]:
    floor_id = _floor_id_for_metadata(int(floor_number or 1))
    cfg = config or GridGrowthConfig()
    cluster_metrics = list(active_report.cluster_metrics or build_cluster_metrics(room_specs, adjacency_graph))
    core_union = _semantic_core_union(core_tube, core_contract)
    core_contract_id = str(getattr(core_contract, "core_contract_id", "") or "")
    core_union_hash = str(getattr(core_contract, "core_union_hash", "") or "")

    def _build_algorithm_report(
        *,
        algorithm_version: str,
        variant_suffix: str,
        seed_offset: int,
    ) -> Tuple[TopologyFeasibilityReport, List[Dict[str, Any]]]:
        semantic_variants: List[TopologyVariant] = []
        variant_diagnostics: List[Dict[str, Any]] = []
        for parent in list(active_report.variants or []):
            if not bool(parent.valid):
                continue
            variant_id = f"{parent.variant_id}_semantic_seeded{variant_suffix}"
            if algorithm_version == "v0":
                islands, _cluster_reports, diagnostics = _grow_semantic_territories_for_parent_v0(
                    parent_variant=parent,
                    variant_id=variant_id,
                    floor_boundary=floor_boundary,
                    core_tube=core_tube,
                    core_contract=core_contract,
                    floor_usable_polygon=floor_usable_polygon,
                    cluster_metrics=cluster_metrics,
                    resolution=float(cfg.growth_resolution),
                    floor_id=floor_id,
                )
            else:
                islands, _cluster_reports, diagnostics = _grow_semantic_territories_for_parent_v1(
                    parent_variant=parent,
                    variant_id=variant_id,
                    floor_boundary=floor_boundary,
                    core_tube=core_tube,
                    core_contract=core_contract,
                    floor_usable_polygon=floor_usable_polygon,
                    cluster_metrics=cluster_metrics,
                    resolution=float(cfg.growth_resolution),
                    floor_id=floor_id,
                    max_iterations=int(getattr(cfg, "semantic_growth_max_iterations", 20000) or 20000),
                    max_cells=int(getattr(cfg, "semantic_growth_max_cells", 100000) or 100000),
                    max_residual_passes=int(getattr(cfg, "semantic_growth_max_residual_passes", 3) or 3),
                )
            island_metrics = build_island_metrics(
                variant_id=variant_id,
                islands=islands,
                corridors=list(parent.corridor_skeleton or []),
                floor_boundary=floor_boundary,
                core_union=core_union,
                forbidden_union=core_union,
                min_door_width=float(cfg.min_door_width),
                min_anchor_frontage=float(cfg.min_anchor_frontage),
                core_contract_id=core_contract_id,
                core_union_hash=core_union_hash,
            )
            metrics_by_id = {str(metric.island_id): metric for metric in island_metrics}
            rows_by_island = {
                str(row.get("island_id") or ""): row
                for row in list(diagnostics.get("cluster_generation", []) or [])
                if isinstance(row, dict)
            }
            for island in islands:
                metric = metrics_by_id.get(str(island.id))
                if metric is None:
                    continue
                try:
                    target_area = float(getattr(island, "target_area", 0.0) or 0.0)
                    min_area = float(getattr(island, "min_area", 0.0) or 0.0)
                    actual_area = float(metric.area)
                    effective_capacity = float(metric.effective_capacity_area)
                    island.effective_capacity = effective_capacity
                    island.raw_area_margin = actual_area - target_area
                    island.target_margin = effective_capacity - target_area
                    island.min_margin = effective_capacity - min_area
                    island.territory_shortfall_area = max(0.0, target_area - effective_capacity)
                    if effective_capacity + 1e-6 < min_area:
                        island.growth_failure_reason = "cluster_min_too_large_for_component"
                    elif effective_capacity + 1e-6 < target_area and not str(getattr(island, "growth_failure_reason", "") or ""):
                        island.growth_failure_reason = "frontier_exhausted_before_target"
                    row = rows_by_island.get(str(island.id))
                    if row is not None:
                        row["actual_area"] = _round_meta(actual_area)
                        row["effective_capacity"] = _round_meta(effective_capacity)
                        row["raw_area_margin"] = _round_meta(actual_area - target_area)
                        row["target_margin"] = _round_meta(effective_capacity - target_area)
                        row["min_margin"] = _round_meta(effective_capacity - min_area)
                        row["territory_shortfall_area"] = _round_meta(max(0.0, target_area - effective_capacity))
                        if effective_capacity + 1e-6 < min_area:
                            row["growth_failure_reason"] = "cluster_min_too_large_for_component"
                        elif effective_capacity + 1e-6 < target_area and not str(row.get("growth_failure_reason") or ""):
                            row["growth_failure_reason"] = "frontier_exhausted_before_target"
                        elif max(0.0, target_area - effective_capacity) <= 1e-6 and str(row.get("growth_failure_reason") or "") in {
                            "frontier_exhausted_before_target",
                            "cluster_target_too_large_for_component",
                        }:
                            row["growth_failure_reason"] = ""
                except Exception:
                    pass
            feasibility = evaluate_cluster_island_feasibility(
                variant_id=variant_id,
                cluster_metrics=cluster_metrics,
                island_metrics=island_metrics,
            )
            generation_failures = [
                row
                for row in list(diagnostics.get("cluster_generation", []) or [])
                if str(row.get("seed_status") or "") != "placed"
                or str(row.get("growth_failure_reason") or "")
                or float(row.get("territory_shortfall_area", 0.0) or 0.0) > 1e-6
            ]
            valid = bool(islands) and all(bool(metric.valid) for metric in island_metrics)
            semantic_variants.append(
                TopologyVariant(
                    variant_id=variant_id,
                    seed=int(parent.seed) + int(seed_offset),
                    is_primary=False,
                    primary_compatible=False,
                    variant_profile={
                        **dict(parent.variant_profile or {}),
                        "variant_family": "semantic_seeded_territory",
                        "semantic_growth_algorithm_version": algorithm_version,
                        "parent_variant_id": str(parent.variant_id),
                        "corridor_source_variant_id": str(parent.variant_id),
                    },
                    corridor_skeleton=list(parent.corridor_skeleton or []),
                    candidate_islands=islands,
                    island_metrics=island_metrics,
                    feasibility_matrix=feasibility,
                    corridor_access_edges=_summarize_access_edges(island_metrics),
                    facade_edges=_summarize_facade_edges(island_metrics),
                    core_docking_candidates=list(parent.core_docking_candidates or []),
                    core_contract_id=core_contract_id or str(parent.core_contract_id or ""),
                    core_union_hash=core_union_hash or str(parent.core_union_hash or ""),
                    corridor_area=float(parent.corridor_area or 0.0),
                    valid=valid,
                    rejection_reasons=sorted(
                        set(reason for metric in island_metrics for reason in list(metric.rejection_reasons or []))
                        | (
                            {
                                str(row.get("growth_failure_reason") or row.get("seed_failure_reason") or "semantic_seeded_generation_shortfall")
                                for row in generation_failures
                            }
                        )
                    ),
                )
            )
            diagnostics["valid"] = bool(valid)
            diagnostics["generation_failure_count"] = len(generation_failures)
            diagnostics["per_cluster_growth_audit"] = sorted(
                list(diagnostics.get("cluster_generation", []) or []),
                key=lambda row: str(row.get("cluster_id") or ""),
            )
            variant_diagnostics.append(diagnostics)
        report = TopologyFeasibilityReport(
            topology_seed_list=[int(v.seed) for v in semantic_variants],
            primary_variant_id=str(semantic_variants[0].variant_id) if semantic_variants else "",
            variants=semantic_variants,
            cluster_metrics=list(cluster_metrics),
        )
        return report, variant_diagnostics

    v0_report, v0_diagnostics = _build_algorithm_report(
        algorithm_version="v0",
        variant_suffix="_v0",
        seed_offset=9000,
    )
    report, variant_diagnostics = _build_algorithm_report(
        algorithm_version="v1",
        variant_suffix="",
        seed_offset=10000,
    )
    circulation_territories = [
        {
            "floor_id": floor_id,
            "variant_id": str(parent.variant_id),
            "variant_family": "semantic_seeded_territory",
            "territory_type": "circulation",
            "source_type": "parent_corridor_skeleton",
            "effective_room_capacity": 0,
            "corridor_area": _round_meta(float(parent.corridor_area or 0.0)),
            "touches_core": bool(parent.core_docking_candidates),
            "touches_exterior": bool(str(floor_id).upper() == "F1"),
            "confidence": "medium" if parent.core_docking_candidates else "low",
        }
        for parent in list(active_report.variants or [])
        if bool(parent.valid)
    ]
    diagnostics = {
        "report_version": "semantic_seeded_territory_variants_v1",
        "analysis_only": True,
        "used_for_solver_decision": False,
        "used_for_adoption": False,
        "used_for_main_path": False,
        "enabled": True,
        "dry_run": True,
        "floor_id": floor_id,
        "variant_family": "semantic_seeded_territory",
        "variant_count": len(report.variants or []),
        "semantic_source_island_count": sum(len(v.candidate_islands or []) for v in list(report.variants or [])),
        "semantic_growth_algorithm_version": "v1",
        "variants": variant_diagnostics,
        "semantic_growth_v0_baseline": {
            "report_version": "semantic_seeded_growth_v0_baseline",
            "analysis_only": True,
            "used_for_solver_decision": False,
            "used_for_adoption": False,
            "variant_count": len(v0_report.variants or []),
            "variants": v0_diagnostics,
        },
        "semantic_growth_v1": {
            "report_version": "semantic_seeded_growth_v1",
            "analysis_only": True,
            "used_for_solver_decision": False,
            "used_for_adoption": False,
            "variant_count": len(report.variants or []),
            "variants": variant_diagnostics,
        },
        "semantic_growth_v1_delta": {
            "available": True,
            "computed_in": "semantic_seeded_comparison",
        },
        "circulation_territories": circulation_territories,
    }
    return report, diagnostics, v0_report


def _proposal_blocking_clusters(proposal: Mapping[str, Any]) -> List[str]:
    clusters: Set[str] = set()
    failed = proposal.get("failed_cluster_diagnostics")
    if isinstance(failed, Mapping):
        for key in ("failed_cluster_id", "cluster_id"):
            value = str(failed.get(key) or "")
            if value:
                clusters.add(value)
    for key in ("clusters_without_feasible_island", "blocking_clusters"):
        for item in list(proposal.get(key, []) or []):
            if isinstance(item, Mapping):
                cid = str(item.get("cluster_id") or "")
            else:
                cid = str(item or "")
            if cid:
                clusters.add(cid)
    summary = proposal.get("capacity_conflict_summary")
    if isinstance(summary, Mapping):
        diagnostic_loads = summary.get("diagnostic_island_loads")
        if isinstance(diagnostic_loads, Mapping):
            overloaded = diagnostic_loads.get("overloaded_islands")
            if isinstance(overloaded, Mapping):
                for item in list(overloaded.get("items", []) or []):
                    if not isinstance(item, Mapping):
                        continue
                    for cid in list(
                        item.get(
                            "assigned_or_candidate_cluster_ids",
                            item.get("assigned_cluster_ids", []),
                        )
                        or []
                    ):
                        if str(cid):
                            clusters.add(str(cid))
        without = summary.get("clusters_without_feasible_island")
        if isinstance(without, Mapping):
            for item in list(without.get("items", []) or []):
                if isinstance(item, Mapping) and str(item.get("cluster_id") or ""):
                    clusters.add(str(item.get("cluster_id")))
    for item in list(proposal.get("missing_feasibility_pairs", []) or []):
        if isinstance(item, Mapping) and str(item.get("cluster_id") or ""):
            clusters.add(str(item.get("cluster_id")))
    return sorted(clusters)


def _proposal_overloaded_islands(proposal: Mapping[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    island_loads = proposal.get("island_loads")
    if isinstance(island_loads, Mapping):
        for island_key, row in sorted(island_loads.items(), key=lambda item: str(item[0])):
            if not isinstance(row, Mapping):
                continue
            overload = max(0.0, -float(row.get("capacity_margin", 0.0) or 0.0))
            if overload <= 1e-6:
                continue
            items.append(
                {
                    "island_key": str(island_key),
                    "variant_id": str(row.get("variant_id") or ""),
                    "island_id": str(row.get("island_id") or island_key),
                    "overload_area": _round_meta(overload),
                    "assigned_cluster_ids": sorted(str(x) for x in list(row.get("clusters", row.get("cluster_ids", [])) or []) if str(x)),
                }
            )
    summary = proposal.get("capacity_conflict_summary")
    if isinstance(summary, Mapping):
        diagnostic_loads = summary.get("diagnostic_island_loads")
        if isinstance(diagnostic_loads, Mapping):
            overloaded = diagnostic_loads.get("overloaded_islands")
            if isinstance(overloaded, Mapping):
                for item in list(overloaded.get("items", []) or []):
                    if not isinstance(item, Mapping):
                        continue
                    overload = float(item.get("area_shortfall", item.get("overload_area", 0.0)) or 0.0)
                    if overload <= 1e-6:
                        continue
                    items.append(
                        {
                            "variant_id": str(item.get("variant_id") or ""),
                            "island_id": str(item.get("island_id") or ""),
                            "overload_area": _round_meta(overload),
                            "assigned_cluster_ids": sorted(
                                str(x)
                                for x in list(
                                    item.get(
                                        "assigned_or_candidate_cluster_ids",
                                        item.get("assigned_cluster_ids", []),
                                    )
                                    or []
                                )
                                if str(x)
                            ),
                            "load_source": str(diagnostic_loads.get("load_source") or "diagnostic_best_effort"),
                        }
                    )
    return sorted(items, key=lambda row: (-float(row.get("overload_area", 0.0)), str(row.get("variant_id")), str(row.get("island_id"))))[:20]


def _proposal_capacity_margin(proposal: Mapping[str, Any]) -> float:
    margins: List[float] = []
    island_loads = proposal.get("island_loads")
    if isinstance(island_loads, Mapping):
        for row in island_loads.values():
            if isinstance(row, Mapping) and "capacity_margin" in row:
                try:
                    margins.append(float(row.get("capacity_margin", 0.0) or 0.0))
                except Exception:
                    pass
    if margins:
        return _round_meta(min(margins))
    overloaded = _proposal_overloaded_islands(proposal)
    if overloaded:
        return _round_meta(-sum(float(row.get("overload_area", 0.0) or 0.0) for row in overloaded))
    return 0.0


def _best_variant_summary(report: TopologyFeasibilityReport, proposal: Mapping[str, Any]) -> Dict[str, Any]:
    selected = str(proposal.get("selected_variant_id") or "")
    variants = list(report.variants or [])
    if selected:
        variant = next((v for v in variants if str(v.variant_id) == selected), None)
        if variant is not None:
            return {
                "variant_id": str(variant.variant_id),
                "status": str(proposal.get("status") or ""),
                "selected_by_proposal": True,
                "feasible_pair_count": sum(1 for row in list(variant.feasibility_matrix or []) if bool(row.hard_feasible)),
                "island_count": len(variant.island_metrics or []),
            }
    candidates: List[Tuple[int, float, str, TopologyVariant]] = []
    for variant in variants:
        feasible_count = sum(1 for row in list(variant.feasibility_matrix or []) if bool(row.hard_feasible))
        score_sum = sum(float(row.feasibility_score) for row in list(variant.feasibility_matrix or []) if bool(row.hard_feasible))
        candidates.append((feasible_count, score_sum, str(variant.variant_id), variant))
    if not candidates:
        return {}
    feasible_count, score_sum, _variant_id, variant = sorted(candidates, key=lambda row: (-row[0], -row[1], row[2]))[0]
    return {
        "variant_id": str(variant.variant_id),
        "status": str(proposal.get("status") or ""),
        "selected_by_proposal": False,
        "feasible_pair_count": int(feasible_count),
        "feasibility_score_sum": _round_meta(score_sum),
        "island_count": len(variant.island_metrics or []),
    }


def _semantic_generation_failure_reasons(diagnostics: Mapping[str, Any]) -> List[str]:
    reasons: Set[str] = set()
    for variant in list(diagnostics.get("variants", []) or []):
        if not isinstance(variant, Mapping):
            continue
        for row in list(variant.get("cluster_generation", []) or []):
            if not isinstance(row, Mapping):
                continue
            reason = str(row.get("growth_failure_reason") or row.get("seed_failure_reason") or "")
            status = str(row.get("seed_status") or "")
            if reason and (status != "placed" or float(row.get("territory_shortfall_area", 0.0) or 0.0) > 1e-6):
                reasons.add(reason)
            stopped_by = str(row.get("stopped_by") or "")
            if stopped_by and stopped_by not in {"complete", "frontier_exhausted"}:
                reasons.add(stopped_by)
    return sorted(reasons)


def _semantic_growth_margin(diagnostics: Mapping[str, Any], field: str) -> float:
    values: List[float] = []
    for variant in list(diagnostics.get("variants", []) or []):
        if not isinstance(variant, Mapping):
            continue
        rows = variant.get("per_cluster_growth_audit", variant.get("cluster_generation", []))
        for row in list(rows or []):
            if not isinstance(row, Mapping):
                continue
            try:
                values.append(float(row.get(field, 0.0) or 0.0))
            except Exception:
                pass
    return _round_meta(min(values)) if values else 0.0


def _semantic_per_cluster_growth_audit(diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    variants = [variant for variant in list(diagnostics.get("variants", []) or []) if isinstance(variant, Mapping)]
    if not variants:
        return {"items": [], "shown_count": 0, "total_available_count": 0}
    best_variant = sorted(
        variants,
        key=lambda variant: (
            -sum(1 for row in list(variant.get("per_cluster_growth_audit", variant.get("cluster_generation", [])) or []) if isinstance(row, Mapping)),
            str(variant.get("variant_id") or ""),
        ),
    )[0]
    rows = [
        dict(row)
        for row in list(best_variant.get("per_cluster_growth_audit", best_variant.get("cluster_generation", [])) or [])
        if isinstance(row, Mapping)
    ]
    rows = sorted(rows, key=lambda row: str(row.get("cluster_id") or ""))[:20]
    return {
        "variant_id": str(best_variant.get("variant_id") or ""),
        "items": rows,
        "shown_count": len(rows),
        "total_available_count": len(list(best_variant.get("per_cluster_growth_audit", best_variant.get("cluster_generation", [])) or [])),
    }


def _semantic_primary_failure_reason(reasons: Sequence[str]) -> str:
    priority = [
        "cluster_min_too_large_for_component",
        "corridor_blocks_growth",
        "seed_placement_failed",
        "frontier_exhausted_before_min",
        "frontier_exhausted_before_target",
        "cluster_target_too_large_for_component",
        "residual_unusable",
        "competition_starvation",
        "invalid_territory_geometry",
        "max_iterations",
        "max_cells",
        "no_progress",
    ]
    reason_set = set(str(r) for r in reasons if str(r))
    for reason in priority:
        if reason in reason_set:
            return reason
    return sorted(reason_set)[0] if reason_set else ""


def _semantic_recommendation_from_reason(reason: str, *, resolved: Sequence[str], after_clusters: Sequence[str], semantic_status: str) -> Tuple[str, str, str]:
    if semantic_status == "success" and resolved:
        return (
            "semantic_seeded_adoption_gate",
            "medium",
            "semantic seeded dry-run resolved at least one previous blocking cluster",
        )
    mapping = {
        "cluster_min_too_large_for_component": (
            "cluster_split_diagnostic",
            "high",
            "at least one semantic cluster cannot reach minimum area within its free component",
        ),
        "corridor_blocks_growth": (
            "circulation_routing_repair",
            "medium",
            "parent corridor leaves insufficient free growth territory",
        ),
        "competition_starvation": (
            "growth_priority_refinement",
            "medium",
            "territory competition starved one or more clusters",
        ),
        "seed_placement_failed": (
            "seed_placement_refinement",
            "medium",
            "semantic seed placement failed for one or more clusters",
        ),
        "residual_unusable": (
            "residual_redistribution_refinement",
            "medium",
            "residual free cells could not be assigned without breaking growth constraints",
        ),
        "invalid_territory_geometry": (
            "geometry_cleanup_or_resolution_adjustment",
            "high",
            "semantic territory geometry became invalid or overlapped excluded regions",
        ),
        "frontier_exhausted_before_min": (
            "semantic_seeded_growth_refinement",
            "medium",
            "semantic frontier exhausted before minimum area for one or more clusters",
        ),
        "frontier_exhausted_before_target": (
            "semantic_seeded_growth_refinement",
            "medium",
            "semantic frontier exhausted before target area for one or more clusters",
        ),
        "cluster_target_too_large_for_component": (
            "semantic_seeded_growth_refinement",
            "medium",
            "semantic cluster target is larger than the available grown component",
        ),
    }
    if reason in mapping:
        return mapping[reason]
    if after_clusters:
        return (
            "cluster_split_diagnostic",
            "medium",
            "semantic seeded dry-run still has blocking clusters",
        )
    return (
        "semantic_seeded_growth_refinement",
        "low",
        "semantic seeded diagnostics available but did not provide a clear improvement",
    )


def _semantic_seeded_comparison(
    *,
    base_report: TopologyFeasibilityReport,
    semantic_report: TopologyFeasibilityReport,
    base_proposal: Mapping[str, Any],
    semantic_proposal: Mapping[str, Any],
    v0_report: Optional[TopologyFeasibilityReport] = None,
    v0_proposal: Optional[Mapping[str, Any]] = None,
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    before_clusters = set(_proposal_blocking_clusters(base_proposal))
    after_clusters = set(_proposal_blocking_clusters(semantic_proposal))
    v0_proposal_map = v0_proposal or {}
    v0_clusters = set(_proposal_blocking_clusters(v0_proposal_map))
    before_overloaded = _proposal_overloaded_islands(base_proposal)
    after_overloaded = _proposal_overloaded_islands(semantic_proposal)
    v1_diag = _mapping_or_empty(diagnostics.get("semantic_growth_v1")) if "semantic_growth_v1" in diagnostics else diagnostics
    v0_diag = _mapping_or_empty(diagnostics.get("semantic_growth_v0_baseline"))
    reasons = _semantic_generation_failure_reasons(v1_diag)
    source_count = int(diagnostics.get("semantic_source_island_count") or 0)
    resolved = sorted(before_clusters - after_clusters)
    remaining = sorted(before_clusters & after_clusters)
    new = sorted(after_clusters - before_clusters)
    v1_resolved_vs_v0 = sorted(v0_clusters - after_clusters)
    v1_new_vs_v0 = sorted(after_clusters - v0_clusters)
    semantic_status = str(semantic_proposal.get("status") or "")
    primary_growth_failure = _semantic_primary_failure_reason(reasons)
    recommendation, confidence, reason = _semantic_recommendation_from_reason(
        primary_growth_failure,
        resolved=resolved,
        after_clusters=sorted(after_clusters),
        semantic_status=semantic_status,
    )
    if source_count <= 0:
        recommendation = "add_missing_metadata"
        confidence = "low"
        reason = "semantic seeded diagnostics produced no source islands"
    if recommendation == "semantic_seeded_adoption_gate" and (after_clusters or not resolved):
        recommendation = "semantic_seeded_growth_refinement"
        confidence = "medium"
        reason = "semantic provenance improved but blockers remain"
    v0_capacity_margin = _proposal_capacity_margin(v0_proposal_map)
    v1_capacity_margin = _proposal_capacity_margin(semantic_proposal)
    v0_target_margin = _semantic_growth_margin(v0_diag, "target_margin")
    v1_target_margin = _semantic_growth_margin(v1_diag, "target_margin")
    v0_min_margin = _semantic_growth_margin(v0_diag, "min_margin")
    v1_min_margin = _semantic_growth_margin(v1_diag, "min_margin")
    semantic_growth_v1 = {
        "enabled": True,
        "variant_count": len(semantic_report.variants or []),
        "feasible_count": sum(1 for variant in list(semantic_report.variants or []) if bool(variant.valid)),
        "source_island_count": source_count,
        "blocking_clusters_before": sorted(before_clusters),
        "blocking_clusters_after": sorted(after_clusters),
        "blocking_clusters_resolved": resolved,
        "blocking_clusters_remaining": remaining,
        "new_blocking_clusters": new,
        "overloaded_islands_before": before_overloaded,
        "overloaded_islands_after": after_overloaded,
        "capacity_margin_before": _proposal_capacity_margin(base_proposal),
        "capacity_margin_after": v1_capacity_margin,
        "capacity_margin_delta": _round_meta(v1_capacity_margin - _proposal_capacity_margin(base_proposal)),
        "min_margin_before": v0_min_margin,
        "min_margin_after": v1_min_margin,
        "target_margin_before": v0_target_margin,
        "target_margin_after": v1_target_margin,
        "per_cluster_growth_audit": _semantic_per_cluster_growth_audit(v1_diag),
        "primary_growth_failure_reason": primary_growth_failure,
        "recommended_next_phase": recommendation,
    }
    return {
        "report_version": "semantic_seeded_comparison_v0",
        "analysis_only": True,
        "used_for_solver_decision": False,
        "used_for_adoption": False,
        "used_for_main_path": False,
        "best_corridor_partitioned_variant": _best_variant_summary(base_report, base_proposal),
        "best_semantic_seeded_variant": _best_variant_summary(semantic_report, semantic_proposal),
        "best_semantic_seeded_v0_variant": _best_variant_summary(v0_report, v0_proposal_map) if v0_report is not None else {},
        "semantic_seeded_variant_count": len(semantic_report.variants or []),
        "semantic_seeded_feasible_count": sum(1 for variant in list(semantic_report.variants or []) if bool(variant.valid)),
        "semantic_source_island_count": source_count,
        "blocking_clusters_before": sorted(before_clusters),
        "blocking_clusters_after": sorted(after_clusters),
        "blocking_clusters_resolved": resolved,
        "blocking_clusters_remaining": remaining,
        "new_blocking_clusters": new,
        "overloaded_islands_before": before_overloaded,
        "overloaded_islands_after": after_overloaded,
        "capacity_margin_before": _proposal_capacity_margin(base_proposal),
        "capacity_margin_after": _proposal_capacity_margin(semantic_proposal),
        "capacity_margin_delta": _round_meta(_proposal_capacity_margin(semantic_proposal) - _proposal_capacity_margin(base_proposal)),
        "v0_blocking_clusters": sorted(v0_clusters),
        "v1_blocking_clusters": sorted(after_clusters),
        "v1_resolved_vs_v0": v1_resolved_vs_v0,
        "v1_new_vs_v0": v1_new_vs_v0,
        "v0_capacity_margin": v0_capacity_margin,
        "v1_capacity_margin": v1_capacity_margin,
        "min_margin_before": v0_min_margin,
        "min_margin_after": v1_min_margin,
        "target_margin_before": v0_target_margin,
        "target_margin_after": v1_target_margin,
        "generation_failure_reasons": reasons,
        "primary_growth_failure_reason": primary_growth_failure,
        "semantic_growth_v1": semantic_growth_v1,
        "semantic_growth_v1_delta": {
            "v0_blocking_clusters": sorted(v0_clusters),
            "v1_blocking_clusters": sorted(after_clusters),
            "v1_resolved_vs_v0": v1_resolved_vs_v0,
            "v1_new_vs_v0": v1_new_vs_v0,
            "v0_capacity_margin": v0_capacity_margin,
            "v1_capacity_margin": v1_capacity_margin,
            "capacity_margin_delta": _round_meta(v1_capacity_margin - v0_capacity_margin),
            "min_margin_delta": _round_meta(v1_min_margin - v0_min_margin),
            "target_margin_delta": _round_meta(v1_target_margin - v0_target_margin),
        },
        "recommended_next_phase": recommendation,
        "recommendation_confidence": confidence,
        "recommendation_reason": reason,
    }


def _summarize_access_edges(island_metrics: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "island_id": m.island_id,
            "access_edge_count": int(m.corridor_door_slot_count),
            "access_edge_total_len": round(float(m.access_edge_total_len), 4),
            "access_total_usable_len": round(float(m.access_total_usable_len), 4),
            "max_single_corridor_edge_len": round(float(m.max_single_corridor_edge_len), 4),
        }
        for m in island_metrics
    ]


def _summarize_facade_edges(island_metrics: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "island_id": m.island_id,
            "window_slot_count": int(m.window_slot_count),
            "facade_len": round(float(m.facade_len), 4),
            "facade_total_usable_len": round(float(m.facade_total_usable_len), 4),
            "max_single_facade_edge_len": round(float(m.max_single_facade_edge_len), 4),
        }
        for m in island_metrics
    ]


def _gate_condition(
    check: str,
    *,
    expected: Any,
    actual: Any,
    passed: Optional[bool] = None,
    severity: str = "blocking",
    semantic_repair_allowed: bool = False,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if passed is None:
        passed = actual == expected
    record: Dict[str, Any] = {
        "check": str(check),
        "condition": str(check),
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
        "severity": str(severity),
        "semantic_repair_allowed": bool(semantic_repair_allowed),
    }
    if detail:
        record["detail"] = dict(detail)
    return record


def _proposal_gate_state(proposal: Optional[Mapping[str, Any]]) -> Tuple[str, str, str]:
    if not isinstance(proposal, Mapping) or not proposal:
        return "not_run", "not_run", "proposal_missing"
    status = str(proposal.get("status") or "").strip().lower()
    reason = str(proposal.get("reason") or "").strip()
    if status == "success":
        return "success", status, reason
    if status == "unavailable" or reason == "ortools_unavailable":
        return "unavailable", status or "unavailable", reason or "ortools_unavailable"
    if status in {"skipped", "not_run", ""}:
        return "not_run", status or "not_run", reason or "proposal_not_run"
    return "failure", status or "failure", reason or "failure"


def _proposal_unavailable_block_reason(reason: str) -> Tuple[str, str]:
    raw = str(reason or "")
    normalized = raw.lower()
    if "ortools" in normalized:
        return "proposal_unavailable_ortools", raw
    if "feasibility" in normalized:
        return "proposal_unavailable_missing_feasibility_report", raw
    if "variant" in normalized:
        return "proposal_unavailable_missing_variants", raw
    return "proposal_unavailable_unknown", raw


def _next_action_hint(gate_block_reason: str) -> str:
    reason = str(gate_block_reason or "")
    if reason == "proposal_capacity_conflict":
        return "inspect_topology_assignment_capacity_conflict"
    if reason == "dry_run_enabled":
        return "enable_adoption_mode_to_test_handoff"
    if reason == "adoption_disabled":
        return "enable_topology_assignment_adoption"
    if reason == "cp_sat_disabled":
        return "enable_topology_assignment_cp_sat"
    if reason in {"selected_variant_missing", "selected_variant_invalid", "assignment_variant_mismatch", "consistency_failure"}:
        return "inspect_topology_assignment_adoption_consistency"
    if reason.startswith("proposal_unavailable"):
        return "inspect_topology_assignment_solver_availability"
    if reason.startswith("proposal_"):
        return "inspect_topology_assignment_failure"
    return ""


def _capacity_conflict_summary(proposal: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    proposal = proposal if isinstance(proposal, Mapping) else {}
    loads = proposal.get("island_loads")
    overloaded: List[Dict[str, Any]] = []
    conflict_clusters: List[str] = []
    if isinstance(loads, Mapping):
        load_iter = loads.values()
    elif isinstance(loads, Sequence) and not isinstance(loads, (str, bytes)):
        load_iter = loads
    else:
        load_iter = []
    for raw_load in load_iter:
        if not isinstance(raw_load, Mapping):
            continue
        target = float(raw_load.get("target_area_sum", raw_load.get("area_load_target", 0.0)) or 0.0)
        effective = float(raw_load.get("effective_capacity_area", 0.0) or 0.0)
        window_load = int(raw_load.get("needs_window_count", raw_load.get("window_slot_load", 0)) or 0)
        window_cap = int(raw_load.get("window_slot_count", raw_load.get("window_slot_capacity", 0)) or 0)
        access_load = int(raw_load.get("needs_corridor_access_count", raw_load.get("access_slot_load", 0)) or 0)
        access_cap = int(raw_load.get("corridor_door_slot_count", raw_load.get("access_slot_capacity", 0)) or 0)
        large_load = int(raw_load.get("large_room_count", raw_load.get("large_slot_load", 0)) or 0)
        large_cap = int(raw_load.get("slot_count_large", raw_load.get("large_slot_capacity", 0)) or 0)
        area_over = target > effective + 1e-6 if effective > 0 else target > 0
        window_over = window_load > window_cap if window_cap or window_load else False
        access_over = access_load > access_cap if access_cap or access_load else False
        large_over = large_load > large_cap if large_cap or large_load else False
        clusters = [str(x) for x in list(raw_load.get("cluster_ids", []) or [])[:10]]
        if area_over or window_over or access_over or large_over:
            overloaded.append(
                {
                    "variant_id": str(raw_load.get("variant_id", "") or ""),
                    "island_id": str(raw_load.get("island_id", "") or ""),
                    "target_area_sum": round(target, 4),
                    "effective_capacity_area": round(effective, 4),
                    "area_slack": round(effective - target, 4),
                    "window_slot_load": window_load,
                    "window_slot_capacity": window_cap,
                    "access_slot_load": access_load,
                    "access_slot_capacity": access_cap,
                    "large_slot_load": large_load,
                    "large_slot_capacity": large_cap,
                    "cluster_ids": clusters,
                }
            )
            conflict_clusters.extend(clusters)
    if overloaded:
        return {
            "available": True,
            "overloaded_islands": overloaded[:5],
            "cluster_ids_in_conflict": sorted(set(conflict_clusters))[:10],
        }
    return {
        "available": False,
        "reason": "capacity_load_details_unavailable",
        "solver_status": str(proposal.get("solver_status", "") or ""),
    }


def _evaluate_topology_assignment_adoption_gate(
    *,
    cp_sat_enabled: bool,
    dry_run: bool,
    adoption_enabled: bool,
    fallback_allowed: bool,
    proposal: Optional[Mapping[str, Any]],
    selected_variant: Optional[TopologyVariant] = None,
    consistency_check_pass: bool = True,
    consistency_detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    proposal = dict(proposal or {})
    proposal_kind, proposal_status, proposal_reason = _proposal_gate_state(proposal)
    proposal_success = proposal_kind == "success"
    selected_variant_id = str(proposal.get("selected_variant_id") or "")
    selected_variant_exists = selected_variant is not None
    selected_variant_valid = bool(getattr(selected_variant, "valid", False)) if selected_variant_exists else False
    cluster_to_island = proposal.get("proposed_cluster_to_island")
    assignment_variant_ids: List[str] = []
    if isinstance(cluster_to_island, Mapping):
        for assignment in cluster_to_island.values():
            if isinstance(assignment, Mapping):
                assignment_variant_ids.append(str(assignment.get("variant_id") or ""))
    all_assignment_variant_ids_match = bool(selected_variant_id) and all(
        variant_id == selected_variant_id for variant_id in assignment_variant_ids
    )
    if bool(selected_variant_id) and not assignment_variant_ids:
        all_assignment_variant_ids_match = True

    failed_checks: List[Dict[str, Any]] = []
    block_reason = ""
    proposal_unavailable_detail = ""
    if not bool(cp_sat_enabled):
        block_reason = "cp_sat_disabled"
        failed_checks.append(_gate_condition("cp_sat_enabled", expected=True, actual=False))
    elif bool(dry_run):
        block_reason = "dry_run_enabled"
        failed_checks.append(_gate_condition("dry_run", expected=False, actual=True))
    elif not bool(adoption_enabled):
        block_reason = "adoption_disabled"
        failed_checks.append(_gate_condition("adoption_enabled", expected=True, actual=False))
    elif proposal_kind == "not_run":
        block_reason = "proposal_not_run"
        failed_checks.append(
            _gate_condition(
                "proposal_status",
                expected="success",
                actual=proposal_status,
                detail={"proposal_reason": proposal_reason},
                semantic_repair_allowed=True,
            )
        )
    elif proposal_kind == "unavailable":
        block_reason, proposal_unavailable_detail = _proposal_unavailable_block_reason(proposal_reason)
        failed_checks.append(
            _gate_condition(
                "proposal_available",
                expected=True,
                actual=False,
                detail={"proposal_status": proposal_status, "proposal_reason": proposal_reason},
                semantic_repair_allowed=True,
            )
        )
    elif not proposal_success:
        block_reason = f"proposal_{proposal_reason or 'failure'}"
        failed_checks.append(
            _gate_condition(
                "proposal_success",
                expected=True,
                actual=False,
                detail={"proposal_status": proposal_status, "proposal_reason": proposal_reason},
                semantic_repair_allowed=True,
            )
        )
    elif not selected_variant_exists:
        block_reason = "selected_variant_missing"
        failed_checks.append(_gate_condition("selected_variant_exists", expected=True, actual=False))
    elif not selected_variant_valid:
        block_reason = "selected_variant_invalid"
        failed_checks.append(_gate_condition("selected_variant_valid", expected=True, actual=False))
    elif not all_assignment_variant_ids_match:
        block_reason = "assignment_variant_mismatch"
        failed_checks.append(
            _gate_condition(
                "all_assignment_variant_ids_match",
                expected=True,
                actual=False,
                detail={
                    "selected_variant_id": selected_variant_id,
                    "assignment_variant_ids": sorted(set(assignment_variant_ids)),
                },
            )
        )
    elif not bool(consistency_check_pass):
        block_reason = "consistency_failure"
        failed_checks.append(
            _gate_condition(
                "consistency_check_pass",
                expected=True,
                actual=False,
                detail=dict(consistency_detail or {}),
            )
        )

    gate_opened = not bool(block_reason)
    gate: Dict[str, Any] = {
        "cp_sat_enabled": bool(cp_sat_enabled),
        "dry_run": bool(dry_run),
        "adoption_enabled": bool(adoption_enabled),
        "fallback_allowed": bool(fallback_allowed),
        "proposal_status": proposal_status,
        "proposal_reason": proposal_reason,
        "proposal_success": bool(proposal_success),
        "selected_variant_id": selected_variant_id,
        "selected_variant_exists": bool(selected_variant_exists),
        "selected_variant_valid": bool(selected_variant_valid),
        "all_assignment_variant_ids_match": bool(all_assignment_variant_ids_match),
        "consistency_check_pass": bool(consistency_check_pass),
        "gate_opened": bool(gate_opened),
        "gate_block_reason": block_reason,
        "next_action_hint": _next_action_hint(block_reason),
        "blocking_conditions": [dict(item) for item in failed_checks if item.get("severity") == "blocking"],
        "nonblocking_conditions": [
            _gate_condition("fallback_allowed", expected=True, actual=bool(fallback_allowed), severity="nonblocking"),
        ],
        "failed_checks": [dict(item) for item in failed_checks],
    }
    if proposal_unavailable_detail:
        gate["proposal_unavailable_detail"] = proposal_unavailable_detail
    if block_reason == "proposal_capacity_conflict":
        existing_summary = proposal.get("capacity_conflict_summary") if isinstance(proposal.get("capacity_conflict_summary"), Mapping) else None
        summary = dict(existing_summary) if existing_summary is not None else _capacity_conflict_summary(proposal)
        gate["capacity_conflict_summary"] = summary
        diagnosis = summary.get("diagnosis") if isinstance(summary.get("diagnosis"), Mapping) else {}
        if diagnosis.get("next_action_hint"):
            gate["next_action_hint"] = str(diagnosis.get("next_action_hint") or "")
    return gate


def _topology_assignment_adoption_record(
    *,
    requested: bool,
    enabled: bool,
    applied: bool,
    used_for_main_path: bool,
    selected_variant: Optional[TopologyVariant] = None,
    proposal: Optional[Dict[str, Any]] = None,
    fallback_to_heuristic: bool = False,
    adoption_failed_reason: str = "",
    adoption_gate: Optional[Dict[str, Any]] = None,
    fallback_kind: str = "",
    fallback_reason: str = "",
    recomputed_handoff_area_stats: bool = False,
    recomputed_coverage_debt_inputs: bool = False,
    object_identity_verified: bool = False,
) -> Dict[str, Any]:
    proposal = dict(proposal or {})
    cluster_to_island = dict(proposal.get("proposed_cluster_to_island") or {})
    island_metrics = list(getattr(selected_variant, "island_metrics", []) or []) if selected_variant is not None else []
    feasibility = list(getattr(selected_variant, "feasibility_matrix", []) or []) if selected_variant is not None else []
    return {
        "requested": bool(requested),
        "enabled": bool(enabled),
        "applied": bool(applied),
        "used_for_main_path": bool(used_for_main_path),
        "runtime_topology_variant_id": getattr(selected_variant, "variant_id", None) if applied else None,
        "selected_variant_id": getattr(selected_variant, "variant_id", None) if selected_variant is not None else proposal.get("selected_variant_id"),
        "selected_seed": getattr(selected_variant, "seed", None) if selected_variant is not None else proposal.get("selected_seed"),
        "cluster_to_island": cluster_to_island,
        "adopted_corridor_area": round(float(getattr(selected_variant, "corridor_area", 0.0) or 0.0), 4) if applied else 0.0,
        "adopted_island_count": len(island_metrics) if applied else 0,
        "adopted_island_area_distribution": [
            round(float(getattr(metric, "area", 0.0) or 0.0), 4)
            for metric in island_metrics
        ] if applied else [],
        "adopted_island_metrics": [m.to_dict() for m in island_metrics] if applied else [],
        "adopted_cluster_island_feasibility": [f.to_dict() for f in feasibility] if applied else [],
        "recomputed_handoff_area_stats": bool(recomputed_handoff_area_stats),
        "recomputed_coverage_debt_inputs": bool(recomputed_coverage_debt_inputs),
        "fallback_to_heuristic": bool(fallback_to_heuristic),
        "adoption_failed_reason": str(adoption_failed_reason or ""),
        "fallback_kind": str(fallback_kind or ""),
        "fallback_reason": str(fallback_reason or adoption_failed_reason or ""),
        "adoption_gate": dict(adoption_gate or {}),
        "object_identity_verified": bool(object_identity_verified),
    }


def _adoption_topology_error(
    *,
    message: str,
    stage: str,
    semantic_repair_allowed: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> LayoutTopologyError:
    meta = {
        "failure_kind": "topology_assignment",
        "stage": str(stage),
        "topology_mode": GRID_GROWTH,
        "semantic_repair_allowed": bool(semantic_repair_allowed),
    }
    meta.update(metadata or {})
    return LayoutTopologyError(message, metadata=meta)


def _variant_core_matches_primary(selected_variant: TopologyVariant, primary_metadata: Dict[str, Any]) -> bool:
    core_meta = dict(primary_metadata.get("core_contract") or {})
    expected_id = str(core_meta.get("core_contract_id", "") or "")
    expected_hash = str(core_meta.get("core_union_hash", "") or "")
    if expected_id and str(selected_variant.core_contract_id or "") != expected_id:
        return False
    if expected_hash and str(selected_variant.core_union_hash or "") != expected_hash:
        return False
    return True


def build_grid_growth_result_from_variant(
    *,
    selected_variant: TopologyVariant,
    proposed_cluster_to_island: Dict[str, Dict[str, str]],
    room_specs: Sequence[RoomSpec],
    report: TopologyFeasibilityReport,
    primary_metadata: Dict[str, Any],
    proposal: Dict[str, Any],
    floor_boundary: Polygon,
    floor_id: str = "",
) -> GridGrowthResult:
    """Build a fresh runtime GridGrowthResult from selected variant geometry."""

    selected_variant_id = str(selected_variant.variant_id)
    if not bool(selected_variant.valid):
        raise _adoption_topology_error(
            message=f"Selected topology variant is invalid: {selected_variant_id}",
            stage="topology_assignment_adoption_inconsistent",
            semantic_repair_allowed=False,
            metadata={"selected_variant_id": selected_variant_id, "reason": "selected_variant_invalid"},
        )
    if not _variant_core_matches_primary(selected_variant, primary_metadata):
        raise _adoption_topology_error(
            message=f"Selected topology variant core contract mismatch: {selected_variant_id}",
            stage="topology_assignment_adoption_inconsistent",
            semantic_repair_allowed=False,
            metadata={
                "selected_variant_id": selected_variant_id,
                "reason": "core_contract_mismatch",
                "variant_core_contract_id": selected_variant.core_contract_id,
                "variant_core_union_hash": selected_variant.core_union_hash,
                "primary_core_contract": dict(primary_metadata.get("core_contract") or {}),
            },
        )

    rooms_by_id = {str(r.room_id): r for r in list(room_specs or [])}
    cluster_by_id = {str(c.cluster_id): c for c in list(report.cluster_metrics or [])}
    source_islands_by_id = {
        str(getattr(island, "id", "")): island
        for island in list(selected_variant.candidate_islands or [])
    }
    island_metrics_by_id = {
        str(metric.island_id): metric
        for metric in list(selected_variant.island_metrics or [])
    }
    unknown_clusters = sorted(set(proposed_cluster_to_island) - set(cluster_by_id))
    if unknown_clusters:
        raise _adoption_topology_error(
            message="Topology assignment references unknown cluster ids",
            stage="topology_assignment_adoption_inconsistent",
            semantic_repair_allowed=False,
            metadata={"unknown_clusters": unknown_clusters, "selected_variant_id": selected_variant_id},
        )

    adopted_corridors = copy.deepcopy(list(selected_variant.corridor_skeleton or []))
    for corridor in adopted_corridors:
        try:
            setattr(corridor, "topology_variant_id", selected_variant_id)
        except Exception:
            pass
    adopted_islands = copy.deepcopy(list(selected_variant.candidate_islands or []))
    adopted_islands_by_id = {str(getattr(island, "id", "")): island for island in adopted_islands}
    for island in adopted_islands:
        try:
            setattr(island, "topology_variant_id", selected_variant_id)
            setattr(island, "core_contract_id", selected_variant.core_contract_id)
            setattr(island, "core_union_hash", selected_variant.core_union_hash)
        except Exception:
            pass

    state: Dict[str, Dict[str, Any]] = {
        str(getattr(island, "id", "")): {"rooms": [], "clusters": [], "target": 0.0}
        for island in adopted_islands
    }
    seen_room_ids: List[str] = []
    for cluster_id, assignment in dict(proposed_cluster_to_island or {}).items():
        target = dict(assignment or {})
        variant_id = str(target.get("variant_id", "") or "")
        island_id = str(target.get("island_id", "") or "")
        if variant_id != selected_variant_id:
            raise _adoption_topology_error(
                message="Topology assignment variant id does not match selected variant",
                stage="topology_assignment_adoption_inconsistent",
                semantic_repair_allowed=False,
                metadata={
                    "cluster_id": str(cluster_id),
                    "assignment_variant_id": variant_id,
                    "selected_variant_id": selected_variant_id,
                },
            )
        if island_id not in source_islands_by_id or island_id not in adopted_islands_by_id:
            raise _adoption_topology_error(
                message="Topology assignment references island outside selected variant",
                stage="topology_assignment_adoption_inconsistent",
                semantic_repair_allowed=False,
                metadata={
                    "cluster_id": str(cluster_id),
                    "island_id": island_id,
                    "selected_variant_id": selected_variant_id,
                },
            )
        cluster = cluster_by_id[str(cluster_id)]
        rooms = []
        for room_id in list(cluster.room_ids or []):
            if room_id not in rooms_by_id:
                raise _adoption_topology_error(
                    message="Topology assignment references unknown room id",
                    stage="topology_assignment_adoption_inconsistent",
                    semantic_repair_allowed=False,
                    metadata={"cluster_id": str(cluster_id), "room_id": str(room_id)},
                )
            rooms.append(rooms_by_id[room_id])
            seen_room_ids.append(str(room_id))
        state[island_id]["rooms"].extend(rooms)
        state[island_id]["clusters"].append(str(cluster_id))
        state[island_id]["target"] += float(sum(float(r.target_area) for r in rooms))

    explicit_room_ids = sorted(
        str(r.room_id)
        for r in list(room_specs or [])
        if not bool(getattr(r, "is_dummy", False))
    )
    if sorted(seen_room_ids) != explicit_room_ids:
        missing = sorted(set(explicit_room_ids) - set(seen_room_ids))
        duplicate = sorted({rid for rid in seen_room_ids if seen_room_ids.count(rid) > 1})
        raise _adoption_topology_error(
            message="Topology assignment did not assign each explicit room exactly once",
            stage="topology_assignment_adoption_inconsistent",
            semantic_repair_allowed=False,
            metadata={
                "selected_variant_id": selected_variant_id,
                "missing_room_ids": missing,
                "duplicate_room_ids": duplicate,
            },
        )

    assignments: Dict[str, AssignmentResult] = {}
    handoff: List[Dict[str, Any]] = []
    object_identity_verified = True
    for island_id, island in adopted_islands_by_id.items():
        rooms = list(state[island_id]["rooms"])
        try:
            source_island = source_islands_by_id[island_id]
            if source_island is island:
                object_identity_verified = False
        except Exception:
            object_identity_verified = False
        target_area = float(sum(float(r.target_area) for r in rooms))
        island_area = float(getattr(island, "area", 0.0) or getattr(getattr(island, "polygon", None), "area", 0.0) or 0.0)
        metric = island_metrics_by_id.get(island_id)
        effective_capacity = float(getattr(metric, "effective_capacity_area", island_area) or island_area)
        try:
            island.assigned_rooms = [str(r.room_id) for r in rooms]
            island.remaining_capacity = island_area - target_area
            setattr(island, "remaining_capacity_area", island_area - target_area)
            setattr(island, "remaining_effective_capacity_area", effective_capacity - target_area)
        except Exception:
            pass
        if rooms:
            assignments[island_id] = AssignmentResult(
                island_id=island_id,
                rooms=rooms,
                total_area=target_area,
                utilization=target_area / island_area if island_area > 1e-9 else 0.0,
            )
        handoff.append(
            {
                "variant_id": selected_variant_id,
                "island_id": island_id,
                "area": round(island_area, 3),
                "core_aware_area": round(island_area, 3),
                "effective_capacity_area": round(effective_capacity, 3),
                "remaining_capacity_area": round(island_area - target_area, 3),
                "remaining_effective_capacity_area": round(effective_capacity - target_area, 3),
                "core_union_hash": selected_variant.core_union_hash,
                "target_area": round(target_area, 3),
                "rooms": [str(r.room_id) for r in rooms],
                "clusters": list(state[island_id]["clusters"]),
                "handoff_polygon_type": "single_polygon",
            }
        )

    metadata = copy.deepcopy(dict(primary_metadata or {}))
    metadata.update(report.to_dict())
    proposal_copy = copy.deepcopy(dict(proposal or {}))
    adoption_gate = dict(proposal_copy.get("adoption_gate") or {})
    if not adoption_gate:
        adoption_gate = _evaluate_topology_assignment_adoption_gate(
            cp_sat_enabled=True,
            dry_run=False,
            adoption_enabled=True,
            fallback_allowed=True,
            proposal=proposal_copy,
            selected_variant=selected_variant,
        )
    proposal_copy["used_for_main_path"] = True
    proposal_copy["adoption_implemented"] = True
    metadata["topology_assignment_proposal"] = proposal_copy
    metadata["runtime_topology_variant_id"] = selected_variant_id
    metadata["handoff_variant_id"] = selected_variant_id
    metadata["topology_seed"] = int(selected_variant.seed)
    metadata["variant_profile"] = dict(selected_variant.variant_profile)
    metadata["handoff"] = handoff
    metadata["core_docking_candidates"] = list(selected_variant.core_docking_candidates or [])
    metadata["adopted_island_metrics"] = [m.to_dict() for m in list(selected_variant.island_metrics or [])]
    metadata["adopted_cluster_island_feasibility"] = [
        f.to_dict() for f in list(selected_variant.feasibility_matrix or [])
    ]
    metadata["island_count"] = len(adopted_islands)
    metadata["assigned_room_count"] = len(explicit_room_ids)
    metadata["corridor"] = {
        **dict(metadata.get("corridor") or {}),
        "variant_id": selected_variant_id,
        "area": float(_safe_corridor_area(adopted_corridors, floor_boundary)),
        "ids": [str(getattr(c, "id", "")) for c in adopted_corridors],
    }
    adoption_record = _topology_assignment_adoption_record(
        requested=True,
        enabled=True,
        applied=True,
        used_for_main_path=True,
        selected_variant=selected_variant,
        proposal=proposal_copy,
        adoption_gate=adoption_gate,
        recomputed_handoff_area_stats=True,
        recomputed_coverage_debt_inputs=True,
        object_identity_verified=object_identity_verified,
    )
    metadata["topology_assignment_adoption"] = adoption_record
    logger.info(
        "[TOPO-CP] Adoption applied | selected_variant=%s | clusters=%d | islands=%d",
        selected_variant_id,
        len(proposed_cluster_to_island or {}),
        len(adopted_islands),
    )
    result = GridGrowthResult(
        corridors=adopted_corridors,
        islands=adopted_islands,
        assignments=assignments,
        degradation=DegradationSummary(),
        metadata=metadata,
    )
    refresh_floor_id = str(floor_id or primary_metadata.get("floor_id") or "F1")
    try:
        refresh_floor_number = int(refresh_floor_id.upper().lstrip("F"))
    except Exception:
        refresh_floor_number = 1
    _refresh_grid_growth_metadata_gap_closure(
        result,
        report,
        floor_number=refresh_floor_number,
        floor_boundary=floor_boundary,
        load_source="adopted_runtime",
    )
    return result


def _build_topology_variant(
    *,
    variant_id: str,
    seed: int,
    result: GridGrowthResult,
    floor_boundary: Polygon,
    core_tube: CoreTube,
    core_contract: Optional[CoreFootprintContract],
    cluster_metrics: Sequence[Any],
    min_door_width: float,
    min_anchor_frontage: float,
    primary_signature: Sequence[float],
    variant_profile: Dict[str, Any],
) -> TopologyVariant:
    core_union = getattr(core_contract, "core_union", None) if core_contract is not None else getattr(core_tube, "polygon", None)
    core_contract_id = str(getattr(core_contract, "core_contract_id", "") or "")
    core_union_hash = str(getattr(core_contract, "core_union_hash", "") or "")
    island_metrics = build_island_metrics(
        variant_id=variant_id,
        islands=result.islands,
        corridors=result.corridors,
        floor_boundary=floor_boundary,
        core_union=core_union,
        forbidden_union=core_union,
        min_door_width=min_door_width,
        min_anchor_frontage=min_anchor_frontage,
        core_contract_id=core_contract_id,
        core_union_hash=core_union_hash,
    )
    feasibility = evaluate_cluster_island_feasibility(
        variant_id=variant_id,
        cluster_metrics=cluster_metrics,
        island_metrics=island_metrics,
    )
    valid = all(m.valid for m in island_metrics)
    rejection_reasons = sorted({reason for m in island_metrics for reason in m.rejection_reasons})
    corridor_area = _safe_corridor_area(result.corridors, floor_boundary)
    primary_compatible = bool(seed == 0 and _area_signature(result.islands) == list(primary_signature))
    variant = TopologyVariant(
        variant_id=variant_id,
        seed=int(seed),
        is_primary=(int(seed) == 0),
        primary_compatible=primary_compatible,
        variant_profile=dict(variant_profile),
        corridor_skeleton=result.corridors,
        candidate_islands=result.islands,
        island_metrics=island_metrics,
        feasibility_matrix=feasibility,
        corridor_access_edges=_summarize_access_edges(island_metrics),
        facade_edges=_summarize_facade_edges(island_metrics),
        core_docking_candidates=list((result.metadata or {}).get("core_docking_candidates", []) or []),
        core_contract_id=core_contract_id,
        core_union_hash=core_union_hash,
        corridor_area=corridor_area,
        valid=valid,
        rejection_reasons=rejection_reasons,
    )
    logger.info(
        "[TOPO] Variant built | variant=%s | seed=%s | islands=%d | corridor_area=%.2f",
        variant_id,
        seed,
        len(result.islands or []),
        corridor_area,
    )
    for metric in island_metrics:
        logger.info(
            "[TOPO] Island metrics | variant=%s | island=%s | area=%.2f | effective_capacity=%.2f | window_slots=%d | door_slots=%d",
            variant_id,
            metric.island_id,
            float(metric.area),
            float(metric.effective_capacity_area),
            int(metric.window_slot_count),
            int(metric.corridor_door_slot_count),
        )
    for row in feasibility:
        logger.info(
            "[TOPO] Cluster-island feasibility | variant=%s | cluster=%s | island=%s | hard_feasible=%s | score=%.3f",
            variant_id,
            row.cluster_id,
            row.island_id,
            bool(row.hard_feasible),
            float(row.feasibility_score),
        )
    return variant


def plan_grid_growth_topology_variants(
    *,
    floor_boundary: Polygon,
    core_tube: CoreTube,
    room_specs: Sequence[RoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
    corridor_width: float = 1.8,
    corridor_layout: str = "organic",
    floor_number: Optional[int] = None,
    config: Optional[GridGrowthConfig] = None,
    core_contract: Optional[CoreFootprintContract] = None,
    floor_usable_polygon: Optional[Any] = None,
    seed_list: Optional[Sequence[int]] = None,
    primary_result: Optional[GridGrowthResult] = None,
) -> TopologyFeasibilityReport:
    seeds = [int(s) for s in (seed_list or DEFAULT_TOPOLOGY_SEEDS)]
    cluster_metrics = build_cluster_metrics(room_specs, adjacency_graph)
    primary_signature = _area_signature(primary_result.islands if primary_result is not None else [])
    variants: List[TopologyVariant] = []
    for seed in seeds:
        variant_id = f"topo_seed_{seed}"
        profile = _variant_profile_for_seed(seed)
        try:
            if seed == 0 and primary_result is not None:
                result = primary_result
            else:
                result = GridGrowthPlanner(
                    floor_boundary=floor_boundary,
                    core_tube=core_tube,
                    room_specs=room_specs,
                    adjacency_graph=adjacency_graph,
                    corridor_width=corridor_width,
                    corridor_layout=corridor_layout,
                    floor_number=floor_number,
                    config=config,
                    core_contract=core_contract,
                    floor_usable_polygon=floor_usable_polygon,
                    topology_seed=seed,
                    variant_profile=profile,
                ).plan()
            variants.append(
                _build_topology_variant(
                    variant_id=variant_id,
                    seed=seed,
                    result=result,
                    floor_boundary=floor_boundary,
                    core_tube=core_tube,
                    core_contract=core_contract,
                    cluster_metrics=cluster_metrics,
                    min_door_width=float((config or GridGrowthConfig()).min_door_width),
                    min_anchor_frontage=float((config or GridGrowthConfig()).min_anchor_frontage),
                    primary_signature=primary_signature or _area_signature(result.islands),
                    variant_profile=profile,
                )
            )
        except Exception as exc:
            logger.info("[TOPO] Variant built | variant=%s | seed=%s | islands=0 | corridor_area=0.00", variant_id, seed)
            variants.append(
                TopologyVariant(
                    variant_id=variant_id,
                    seed=seed,
                    is_primary=(seed == 0),
                    primary_compatible=False,
                    variant_profile=profile,
                    core_contract_id=str(getattr(core_contract, "core_contract_id", "") or ""),
                    core_union_hash=str(getattr(core_contract, "core_union_hash", "") or ""),
                    valid=False,
                    rejection_reasons=[str(exc)],
                )
            )
    return TopologyFeasibilityReport(
        topology_seed_list=seeds,
        primary_variant_id="topo_seed_0",
        variants=variants,
        cluster_metrics=list(cluster_metrics),
    )
