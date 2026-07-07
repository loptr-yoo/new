from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from shapely.geometry import Polygon


logger = logging.getLogger(__name__)


LOW_FILL_CLASSIFICATIONS = {
    "edge_sliver_absorb",
    "corridor_sponge",
    "boundary_trim",
    "service_niche",
}

STAGE2A2_DOOR_FIRST_RESIDUAL_ENABLED = True
STAGE2A2_GENERATED_FILLER_ENABLED = False
STAGE2A2_COVERAGE_FEATURE_UNION_ENABLED = True


@dataclass
class ResidualSpongePolicy:
    low_fill_threshold: float = 0.25
    medium_fill_threshold: float = 0.45
    compact_fill_threshold: float = 0.45
    small_area_threshold: float = 2.0
    compact_max_area: float = 8.0
    compact_area_epsilon: float = 0.5
    max_boundary_trim_area_per_piece: float = 8.5
    split_max_area: float = 16.0
    compactness_threshold: float = 0.35
    min_door_width: float = 0.8


@dataclass
class DoorPreflightResult:
    can_place_corridor_door: bool = False
    can_attach_to_room: bool = False
    can_be_non_room_feature: bool = False
    shared_len_with_corridor: float = 0.0
    shared_len_with_rooms: Dict[str, float] = field(default_factory=dict)
    touches_floor_boundary: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResidualShapeMetadata:
    area: float
    width: float
    height: float
    bbox_area: float
    fill_rate: float
    aspect_ratio: float
    compactness: float = 0.0
    residual_id: str = ""
    floor_id: str = ""
    island_id: str = ""
    bbox: List[float] = field(default_factory=list)
    touches_floor_boundary: bool = False
    shared_len_with_corridor: float = 0.0
    shared_len_with_rooms: Dict[str, float] = field(default_factory=dict)
    distance_to_corridor: Optional[float] = None
    can_place_door: bool = False
    can_place_corridor_door: bool = False
    can_attach_to_room: bool = False
    can_be_non_room_feature: bool = False
    door_preflight_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResidualDecision:
    classification: str
    door_required: bool
    semantic_repair_allowed: bool
    is_low_fill_geometry_debt: bool
    counts_as_room: bool
    counts_as_budget: bool
    requires_materialization: bool
    allowed_actions: List[str]
    forbidden_actions: List[str]
    materialize_as: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def metadata_from_polygon(
    poly: Polygon,
    *,
    floor_boundary: Optional[Polygon] = None,
    shared_len_with_corridor: float = 0.0,
    shared_len_with_rooms: Optional[Dict[str, float]] = None,
    distance_to_corridor: Optional[float] = None,
    can_place_door: bool = False,
    can_place_corridor_door: Optional[bool] = None,
    can_attach_to_room: bool = False,
    can_be_non_room_feature: bool = False,
    door_preflight_reason: str = "",
) -> ResidualShapeMetadata:
    minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
    width = max(0.0, maxx - minx)
    height = max(0.0, maxy - miny)
    bbox_area = max(0.0, width * height)
    fill_rate = float(poly.area) / bbox_area if bbox_area > 1e-9 else 0.0
    short = max(1e-6, min(width, height))
    aspect = max(width, height) / short
    try:
        perimeter = float(poly.length)
        compactness = (4.0 * math.pi * float(poly.area) / (perimeter * perimeter)) if perimeter > 1e-9 else 0.0
    except Exception:
        compactness = 0.0
    touches_boundary = False
    if floor_boundary is not None:
        try:
            touches_boundary = bool(poly.boundary.intersection(floor_boundary.boundary).length > 1e-6)
        except Exception:
            touches_boundary = False
    corridor_door = bool(can_place_door if can_place_corridor_door is None else can_place_corridor_door)
    return ResidualShapeMetadata(
        area=float(poly.area),
        width=width,
        height=height,
        bbox_area=bbox_area,
        fill_rate=fill_rate,
        aspect_ratio=aspect,
        compactness=compactness,
        bbox=[round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4)],
        touches_floor_boundary=touches_boundary,
        shared_len_with_corridor=float(shared_len_with_corridor),
        shared_len_with_rooms=dict(shared_len_with_rooms or {}),
        distance_to_corridor=distance_to_corridor,
        can_place_door=bool(corridor_door),
        can_place_corridor_door=bool(corridor_door),
        can_attach_to_room=bool(can_attach_to_room),
        can_be_non_room_feature=bool(can_be_non_room_feature or touches_boundary or can_attach_to_room),
        door_preflight_reason=str(door_preflight_reason or ""),
    )


def classify_residual_metadata(
    meta: ResidualShapeMetadata,
    policy: ResidualSpongePolicy | None = None,
) -> ResidualDecision:
    policy = policy or ResidualSpongePolicy()
    area = float(meta.area)
    fill = float(meta.fill_rate)
    compact = float(meta.compactness)
    can_corridor_door = bool(meta.can_place_corridor_door or meta.can_place_door)
    can_attach = bool(meta.can_attach_to_room)
    touches_boundary = bool(meta.touches_floor_boundary)
    compact_max = float(policy.compact_max_area) + float(policy.compact_area_epsilon)
    boundary_trim_max = float(policy.max_boundary_trim_area_per_piece)

    if area <= float(policy.small_area_threshold):
        classification = "corridor_sponge"
        reason = "small residual can be absorbed by corridor or neighbor"
    elif fill <= float(policy.low_fill_threshold):
        classification = "boundary_trim" if bool(meta.touches_floor_boundary) else "edge_sliver_absorb"
        reason = "low fill-rate residual is geometry debt, not a room"
    elif fill <= float(policy.medium_fill_threshold):
        if can_corridor_door and compact >= float(policy.compactness_threshold) and area <= compact_max:
            classification = "compact_filler"
            reason = "medium fill residual is compact and doorable"
        elif touches_boundary:
            classification = "boundary_trim"
            reason = "medium fill residual has no corridor door and is boundary-owned geometry"
        elif can_attach:
            classification = "attached_service_niche"
            reason = "medium fill residual has no corridor door but can attach to a room"
        else:
            classification = "corridor_sponge" if float(meta.shared_len_with_corridor) > 0.05 else "service_niche"
            reason = "medium fill residual lacks compact doorable room geometry"
    elif area <= compact_max:
        if can_corridor_door:
            classification = "compact_filler"
            reason = "compact residual can become generated filler if identity and door checks pass"
        elif touches_boundary and area <= boundary_trim_max:
            classification = "boundary_trim"
            reason = "no-door compact residual is owned as boundary/service coverage feature"
        elif can_attach:
            classification = "neighbor_absorb"
            reason = "no-door compact residual can attach to neighboring room/service niche"
        else:
            classification = "compact_filler_no_door"
            reason = "compact residual is not doorable, boundary-owned, or attachable"
    elif area <= float(policy.split_max_area):
        if can_corridor_door:
            classification = "split_compact_filler"
            reason = "large compact residual requires split filler support"
        elif touches_boundary and area <= boundary_trim_max:
            classification = "boundary_trim"
            reason = "large no-door compact residual is still within boundary trim cap"
        elif can_attach:
            classification = "attached_service_niche"
            reason = "large no-door compact residual can attach to room as service niche"
        else:
            classification = "split_compact_filler_not_ready"
            reason = "large compact residual cannot satisfy door-first split branches"
    else:
        classification = "unexpected_large_residual"
        reason = "large compact residual exceeds Stage 2A.1 sponge scope"

    is_low_fill = classification in LOW_FILL_CLASSIFICATIONS or fill <= float(policy.low_fill_threshold)
    door_required = classification in {"compact_filler", "split_compact_filler"}
    counts_as_room = classification in {"compact_filler", "split_compact_filler"}
    coverage_feature_classes = LOW_FILL_CLASSIFICATIONS | {
        "attached_service_niche",
        "neighbor_absorb",
    }
    forbidden = []
    if is_low_fill or not door_required:
        forbidden = ["synthetic_storage", "compact_filler", "split_compact_filler"]
    decision = ResidualDecision(
        classification=classification,
        door_required=door_required,
        semantic_repair_allowed=False,
        is_low_fill_geometry_debt=is_low_fill or classification in coverage_feature_classes,
        counts_as_room=counts_as_room,
        counts_as_budget=False,
        requires_materialization=True,
        allowed_actions=[
            classification,
            "coverage_feature",
        ],
        forbidden_actions=forbidden,
        materialize_as=(
            "coverage_feature" if (is_low_fill or classification in coverage_feature_classes) else classification
        ),
        reason=reason,
        metadata=meta.to_dict(),
    )
    logger.debug(
        "[SPONGE] Residual classified | area=%.2f | fill_rate=%.3f | class=%s | door_required=%s | semantic_repair_allowed=%s",
        area,
        fill,
        classification,
        bool(decision.door_required),
        bool(decision.semantic_repair_allowed),
    )
    return decision


def classify_residual_piece(
    poly: Polygon,
    *,
    floor_id: str,
    island_id: str,
    floor_boundary: Optional[Polygon] = None,
    shared_len_with_corridor: float = 0.0,
    shared_len_with_rooms: Optional[Dict[str, float]] = None,
    distance_to_corridor: Optional[float] = None,
    can_place_door: bool = False,
    can_place_corridor_door: Optional[bool] = None,
    can_attach_to_room: bool = False,
    can_be_non_room_feature: bool = False,
    door_preflight_reason: str = "",
    policy: ResidualSpongePolicy | None = None,
) -> ResidualDecision:
    meta = metadata_from_polygon(
        poly,
        floor_boundary=floor_boundary,
        shared_len_with_corridor=shared_len_with_corridor,
        shared_len_with_rooms=shared_len_with_rooms,
        distance_to_corridor=distance_to_corridor,
        can_place_door=can_place_door,
        can_place_corridor_door=can_place_corridor_door,
        can_attach_to_room=can_attach_to_room,
        can_be_non_room_feature=can_be_non_room_feature,
        door_preflight_reason=door_preflight_reason,
    )
    decision = classify_residual_metadata(meta, policy=policy)
    decision.metadata.update({"floor_id": str(floor_id), "island_id": str(island_id)})
    return decision
