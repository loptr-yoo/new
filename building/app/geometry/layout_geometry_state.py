from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shapely.geometry import Polygon


class LayoutObjectContractError(ValueError):
    """Raised when a layout object cannot satisfy the Phase 1 geometry contract."""

    def __init__(
        self,
        message: str,
        *,
        object_id: Optional[str] = None,
        kind: Optional[str] = None,
        coverage_role: Optional[str] = None,
        missing_fields: Optional[Iterable[str]] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.object_id = object_id
        self.kind = kind
        self.coverage_role = coverage_role
        self.missing_fields = list(missing_fields or [])
        self.reason = reason or message
        self.metadata = {
            "object_id": object_id,
            "kind": kind,
            "coverage_role": coverage_role,
            "missing_fields": self.missing_fields,
            "reason": self.reason,
        }


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_attr_or_key(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = _get_attr_or_key(obj, name, None)
        if value is not None:
            return value
    return default


def get_polygon_area(poly: Any, *, object_id: Optional[str] = None, kind: Optional[str] = None) -> float:
    try:
        area = getattr(poly, "area")
    except Exception as exc:
        raise LayoutObjectContractError(
            "Layout object polygon has no readable area",
            object_id=object_id,
            kind=kind,
            missing_fields=["polygon.area"],
            reason=str(exc),
        ) from exc
    try:
        return float(area)
    except Exception as exc:
        raise LayoutObjectContractError(
            "Layout object polygon area is not numeric",
            object_id=object_id,
            kind=kind,
            missing_fields=["polygon.area"],
            reason=str(exc),
        ) from exc


def get_polygon_bounds(poly: Any, *, object_id: Optional[str] = None, kind: Optional[str] = None) -> Tuple[float, float, float, float]:
    try:
        bounds = getattr(poly, "bounds")
    except Exception as exc:
        raise LayoutObjectContractError(
            "Layout object polygon has no readable bounds",
            object_id=object_id,
            kind=kind,
            missing_fields=["polygon.bounds"],
            reason=str(exc),
        ) from exc
    try:
        minx, miny, maxx, maxy = bounds
        return (float(minx), float(miny), float(maxx), float(maxy))
    except Exception as exc:
        raise LayoutObjectContractError(
            "Layout object polygon bounds are invalid",
            object_id=object_id,
            kind=kind,
            missing_fields=["polygon.bounds"],
            reason=str(exc),
        ) from exc


def is_polygon_empty(poly: Any) -> bool:
    return bool(getattr(poly, "is_empty", False))


def is_polygon_valid(poly: Any) -> bool:
    return bool(getattr(poly, "is_valid", True))


def _validate_polygon(poly: Any, *, object_id: str, kind: str, coverage_role: Optional[str] = None) -> None:
    if poly is None:
        raise LayoutObjectContractError(
            "Layout object is missing polygon",
            object_id=object_id,
            kind=kind,
            coverage_role=coverage_role,
            missing_fields=["polygon"],
            reason="missing_polygon",
        )
    get_polygon_area(poly, object_id=object_id, kind=kind)
    get_polygon_bounds(poly, object_id=object_id, kind=kind)
    if is_polygon_empty(poly):
        raise LayoutObjectContractError(
            "Layout object polygon is empty",
            object_id=object_id,
            kind=kind,
            coverage_role=coverage_role,
            reason="empty_polygon",
        )
    if not is_polygon_valid(poly):
        raise LayoutObjectContractError(
            "Layout object polygon is invalid",
            object_id=object_id,
            kind=kind,
            coverage_role=coverage_role,
            reason="invalid_polygon",
        )


@dataclass
class LayoutObjectContract:
    id: str
    kind: str
    polygon: Any
    legacy_id: Optional[str] = None
    floor_id: Optional[str] = None
    island_id: Optional[str] = None
    coverage_role: Optional[str] = None
    semantic_room: bool = False
    generated: bool = False
    generated_by: Optional[str] = None
    source: Optional[str] = None
    source_stage: Optional[str] = None
    source_residual_id: Optional[str] = None
    counts_as_budget: bool = False
    participates_in_budget_validation: bool = False
    participates_in_door_graph: bool = False
    participates_in_wall_graph: bool = False
    participates_in_final_coverage: bool = True
    requires_door: bool = False
    render_layer: Optional[str] = None
    mutable_by_gap_snap: bool = False
    protected_geometry: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "LayoutObjectContract":
        missing: List[str] = []
        if not self.id:
            missing.append("id")
        if not self.kind:
            missing.append("kind")
        if self.polygon is None:
            missing.append("polygon")
        if missing:
            raise LayoutObjectContractError(
                "Layout object contract missing required fields",
                object_id=self.id,
                kind=self.kind,
                coverage_role=self.coverage_role,
                missing_fields=missing,
                reason="missing_required_fields",
            )
        _validate_polygon(
            self.polygon,
            object_id=str(self.id),
            kind=str(self.kind),
            coverage_role=self.coverage_role,
        )
        if self.kind == "coverage_feature":
            owner_missing = ["floor_id", "source_residual_id"]
            owner_missing = [field_name for field_name in owner_missing if not getattr(self, field_name)]
            if owner_missing:
                raise LayoutObjectContractError(
                    "Coverage feature is missing ownership fields",
                    object_id=self.id,
                    kind=self.kind,
                    coverage_role=self.coverage_role,
                    missing_fields=owner_missing,
                    reason="ownerless_coverage_feature",
                )
            if self.counts_as_budget or self.participates_in_budget_validation:
                raise LayoutObjectContractError(
                    "Coverage feature cannot participate in budget validation",
                    object_id=self.id,
                    kind=self.kind,
                    coverage_role=self.coverage_role,
                    reason="coverage_feature_budget_participation",
                )
            if self.participates_in_door_graph:
                raise LayoutObjectContractError(
                    "Coverage feature cannot participate in door graph",
                    object_id=self.id,
                    kind=self.kind,
                    coverage_role=self.coverage_role,
                    reason="coverage_feature_door_participation",
                )
        return self


def make_semantic_room_object(room: Any, *, floor_id: Optional[str] = None) -> LayoutObjectContract:
    object_id = str(_first_attr_or_key(room, ("id", "room_id"), "room"))
    poly = _get_attr_or_key(room, "polygon")
    return LayoutObjectContract(
        id=object_id,
        legacy_id=object_id,
        floor_id=floor_id or _get_attr_or_key(room, "floor_id"),
        kind="semantic_room",
        coverage_role=None,
        polygon=poly,
        semantic_room=True,
        generated=bool(_get_attr_or_key(room, "generated", False)),
        counts_as_budget=True,
        participates_in_budget_validation=True,
        participates_in_door_graph=True,
        participates_in_wall_graph=True,
        participates_in_final_coverage=True,
        requires_door=True,
        render_layer="rooms",
        mutable_by_gap_snap=True,
        protected_geometry=False,
        metadata={"room_type": _get_attr_or_key(room, "room_type")},
    ).validate()


def make_generated_compact_filler_object(obj: Any, *, floor_id: Optional[str] = None) -> LayoutObjectContract:
    object_id = str(_first_attr_or_key(obj, ("id", "room_id", "feature_id"), "generated_compact_filler"))
    poly = _get_attr_or_key(obj, "polygon")
    return LayoutObjectContract(
        id=object_id,
        legacy_id=object_id,
        floor_id=floor_id or _get_attr_or_key(obj, "floor_id"),
        island_id=_get_attr_or_key(obj, "island_id"),
        kind="generated_room",
        coverage_role="compact_filler",
        polygon=poly,
        semantic_room=False,
        generated=True,
        generated_by=str(_get_attr_or_key(obj, "generated_by", "coverage_debt_planner")),
        source=str(_get_attr_or_key(obj, "source", "coverage_debt")),
        source_stage=str(_get_attr_or_key(obj, "source_stage", "residual_sweep")),
        source_residual_id=_get_attr_or_key(obj, "source_residual_id"),
        counts_as_budget=False,
        participates_in_budget_validation=False,
        participates_in_door_graph=True,
        participates_in_wall_graph=True,
        participates_in_final_coverage=True,
        requires_door=True,
        render_layer="generated_rooms",
        mutable_by_gap_snap=False,
        protected_geometry=True,
        metadata=dict(_get_attr_or_key(obj, "metadata", {}) or {}),
    ).validate()


def make_coverage_feature_object(obj: Any, *, floor_id: Optional[str] = None) -> LayoutObjectContract:
    metadata_pair: Dict[str, Any] = {}
    raw_obj = obj
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], Polygon):
        raw_obj = obj[0]
        metadata_pair = dict(obj[1] or {}) if isinstance(obj[1], dict) else {}
    object_id = str(_first_attr_or_key(metadata_pair or raw_obj, ("feature_id", "id", "legacy_id"), "coverage_feature"))
    poly = raw_obj if isinstance(raw_obj, Polygon) else _get_attr_or_key(raw_obj, "polygon")
    effective_floor_id = floor_id or _get_attr_or_key(metadata_pair or raw_obj, "floor_id")
    if not effective_floor_id and not isinstance(raw_obj, Polygon):
        effective_floor_id = _get_attr_or_key(_get_attr_or_key(raw_obj, "metadata", {}) or {}, "floor_id")
    source_residual_id = (
        _get_attr_or_key(metadata_pair, "source_residual_id")
        if metadata_pair
        else (None if isinstance(raw_obj, Polygon) else _get_attr_or_key(raw_obj, "source_residual_id"))
    )
    if not source_residual_id and isinstance(obj, Polygon):
        source_residual_id = f"raw_residual_{abs(hash(obj.wkb)) & 0xFFFFFFFF:x}"
    if not source_residual_id and isinstance(raw_obj, Polygon):
        source_residual_id = f"raw_residual_{abs(hash(raw_obj.wkb)) & 0xFFFFFFFF:x}"
    return LayoutObjectContract(
        id=object_id,
        legacy_id=str(_first_attr_or_key(metadata_pair or raw_obj, ("feature_id", "id"), object_id)) if not isinstance(raw_obj, Polygon) else object_id,
        floor_id=effective_floor_id,
        island_id=_get_attr_or_key(metadata_pair, "island_id") if metadata_pair else (None if isinstance(raw_obj, Polygon) else _get_attr_or_key(raw_obj, "island_id")),
        kind="coverage_feature",
        coverage_role=_get_attr_or_key(metadata_pair, "coverage_role") if metadata_pair else (None if isinstance(raw_obj, Polygon) else _get_attr_or_key(raw_obj, "coverage_role", _get_attr_or_key(raw_obj, "classification"))),
        polygon=poly,
        semantic_room=False,
        generated=True,
        generated_by=_get_attr_or_key(metadata_pair, "generated_by", "coverage_debt_planner") if metadata_pair else (None if isinstance(raw_obj, Polygon) else _get_attr_or_key(raw_obj, "generated_by", "coverage_debt_planner")),
        source=_get_attr_or_key(metadata_pair, "source", "coverage_debt") if metadata_pair else (None if isinstance(raw_obj, Polygon) else _get_attr_or_key(raw_obj, "source", "coverage_debt")),
        source_stage=_get_attr_or_key(metadata_pair, "source_stage", "residual_sweep") if metadata_pair else (None if isinstance(raw_obj, Polygon) else _get_attr_or_key(raw_obj, "source_stage", "residual_sweep")),
        source_residual_id=source_residual_id,
        counts_as_budget=False,
        participates_in_budget_validation=False,
        participates_in_door_graph=False,
        participates_in_wall_graph=True,
        participates_in_final_coverage=True,
        requires_door=False,
        render_layer="coverage_features",
        mutable_by_gap_snap=False,
        protected_geometry=True,
        metadata=metadata_pair or (dict(_get_attr_or_key(raw_obj, "metadata", {}) or {}) if not isinstance(raw_obj, Polygon) else {}),
    ).validate()


def make_corridor_object(obj: Any, *, floor_id: Optional[str] = None) -> LayoutObjectContract:
    object_id = str(_first_attr_or_key(obj, ("id", "corridor_id"), "corridor"))
    poly = obj if isinstance(obj, Polygon) else _get_attr_or_key(obj, "polygon")
    return LayoutObjectContract(
        id=object_id,
        legacy_id=object_id,
        floor_id=floor_id or (None if isinstance(obj, Polygon) else _get_attr_or_key(obj, "floor_id")),
        kind="corridor",
        polygon=poly,
        semantic_room=False,
        generated=False,
        counts_as_budget=False,
        participates_in_budget_validation=False,
        participates_in_door_graph=True,
        participates_in_wall_graph=True,
        participates_in_final_coverage=True,
        requires_door=False,
        render_layer="corridors",
        mutable_by_gap_snap=False,
        protected_geometry=True,
    ).validate()


def make_core_object(obj: Any, *, floor_id: Optional[str] = None, object_id: Optional[str] = None) -> LayoutObjectContract:
    poly = obj if isinstance(obj, Polygon) else _get_attr_or_key(obj, "polygon")
    cid = str(object_id or _first_attr_or_key(obj, ("id", "core_id"), "core"))
    return LayoutObjectContract(
        id=cid,
        legacy_id=cid,
        floor_id=floor_id or (None if isinstance(obj, Polygon) else _get_attr_or_key(obj, "floor_id")),
        kind="core",
        polygon=poly,
        semantic_room=False,
        generated=False,
        counts_as_budget=False,
        participates_in_budget_validation=False,
        participates_in_door_graph=False,
        participates_in_wall_graph=True,
        participates_in_final_coverage=True,
        requires_door=False,
        render_layer="core",
        mutable_by_gap_snap=False,
        protected_geometry=True,
    ).validate()


@dataclass
class LayoutGeometryState:
    floor_id: Optional[str] = None
    topology_mode: Optional[str] = None
    semantic_rooms: List[LayoutObjectContract] = field(default_factory=list)
    generated_rooms: List[LayoutObjectContract] = field(default_factory=list)
    coverage_features: List[LayoutObjectContract] = field(default_factory=list)
    corridors: List[LayoutObjectContract] = field(default_factory=list)
    core_objects: List[LayoutObjectContract] = field(default_factory=list)
    forbidden_zones: List[LayoutObjectContract] = field(default_factory=list)
    residual_ledger: Dict[str, Any] = field(default_factory=dict)
    solver_metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(
        cls,
        *,
        floor_id: Optional[str] = None,
        topology_mode: Optional[str] = None,
        rooms: Optional[List[Any]] = None,
        generated_rooms: Optional[List[Any]] = None,
        coverage_features: Optional[List[Any]] = None,
        corridors: Optional[List[Any]] = None,
        core_tube: Optional[Any] = None,
        core_polygons: Optional[List[Any]] = None,
        residual_ledger: Optional[Dict[str, Any]] = None,
        solver_metadata: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "LayoutGeometryState":
        state = cls(
            floor_id=floor_id,
            topology_mode=topology_mode,
            residual_ledger=dict(residual_ledger or {}),
            solver_metadata=dict(solver_metadata or {}),
            diagnostics=dict(diagnostics or {}),
        )
        state.semantic_rooms = [
            make_semantic_room_object(room, floor_id=floor_id)
            for room in rooms or []
        ]
        state.generated_rooms = [
            make_generated_compact_filler_object(room, floor_id=floor_id)
            for room in generated_rooms or []
        ]
        state.coverage_features = [
            make_coverage_feature_object(feature, floor_id=floor_id)
            for feature in coverage_features or []
        ]
        state.corridors = [
            make_corridor_object(corridor, floor_id=floor_id)
            for corridor in corridors or []
        ]
        for poly in core_polygons or []:
            state.core_objects.append(make_core_object(poly, floor_id=floor_id))
        if core_tube is not None:
            for attr in (
                "polygon",
                "staircase",
                "staircase_hall",
                "staircase_shaft",
                "elevator",
                "elevator_hall",
                "elevator_shaft",
            ):
                poly = getattr(core_tube, attr, None)
                if isinstance(poly, Polygon) and not poly.is_empty:
                    state.core_objects.append(
                        make_core_object(poly, floor_id=floor_id, object_id=f"core_{attr}")
                    )
        return state.validate_contracts()

    def all_objects(self) -> List[LayoutObjectContract]:
        return (
            list(self.semantic_rooms)
            + list(self.generated_rooms)
            + list(self.coverage_features)
            + list(self.corridors)
            + list(self.core_objects)
            + list(self.forbidden_zones)
        )

    def budget_objects(self) -> List[LayoutObjectContract]:
        return [
            obj for obj in self.all_objects()
            if obj.counts_as_budget or obj.participates_in_budget_validation
        ]

    def door_graph_objects(self) -> List[LayoutObjectContract]:
        return [obj for obj in self.all_objects() if obj.participates_in_door_graph]

    def wall_graph_objects(self) -> List[LayoutObjectContract]:
        return [obj for obj in self.all_objects() if obj.participates_in_wall_graph]

    def final_coverage_objects(self) -> List[LayoutObjectContract]:
        return [obj for obj in self.all_objects() if obj.participates_in_final_coverage]

    def filter_coverage_features_for_scope(
        self,
        *,
        scope: str,
        floor_id: Optional[str],
        island_id: Optional[str] = None,
        source_residual_id: Optional[str] = None,
    ) -> List[LayoutObjectContract]:
        scope_l = str(scope or "floor").lower()
        accepted: List[LayoutObjectContract] = []
        for feature in self.coverage_features:
            if floor_id is not None and str(feature.floor_id) != str(floor_id):
                continue
            if source_residual_id is not None and str(feature.source_residual_id) != str(source_residual_id):
                continue
            if scope_l == "island":
                if not feature.island_id or island_id is None:
                    continue
                if str(feature.island_id) != str(island_id):
                    continue
            elif scope_l != "floor":
                raise LayoutObjectContractError(
                    "Unknown coverage feature scope",
                    kind="coverage_feature",
                    reason=f"unknown_scope:{scope}",
                )
            accepted.append(feature)
        return accepted

    def validate_contracts(self) -> "LayoutGeometryState":
        for obj in self.all_objects():
            obj.validate()
        return self
