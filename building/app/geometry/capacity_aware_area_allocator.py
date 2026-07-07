"""Capacity-aware room target allocation diagnostics.

This module derives geometry targets after topology effective capacity is known.
It is intentionally geometry-free and diagnostic-friendly.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


VERSION = "capacity_aware_area_allocation_v1"
DEFAULT_AREA_EPSILON = 1e-6


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _room_id(room: Any) -> str:
    return str(getattr(room, "room_id", "") or getattr(room, "id", "") or "")


def _room_type(room: Any) -> str:
    return str(getattr(room, "room_type", "") or getattr(room, "type", "") or "").lower()


def _room_min_area(room: Any, preferred: float) -> float:
    min_geom = float(getattr(room, "min_area", 0.0) or 0.0)
    if min_geom <= 0.0:
        min_w = float(getattr(room, "min_width", 0.0) or 0.0)
        min_d = float(getattr(room, "min_depth", min_w) or min_w or 0.0)
        min_geom = max(0.0, min_w * min_d)
    return max(min_geom, float(preferred) * 0.85)


def _room_max_area(room: Any, preferred: float) -> float:
    max_geom = float(getattr(room, "max_area", 0.0) or 0.0)
    fallback = float(preferred) * 1.15
    if max_geom <= 0.0:
        return max(fallback, _room_min_area(room, preferred))
    return max(float(max_geom), _room_min_area(room, preferred))


def _default_area_weight(room_type: str) -> float:
    t = str(room_type or "").lower()
    if "living" in t or "public" in t or t in {"dining", "dining_room", "reception"}:
        return 1.4
    if "bed" in t or t in {"study", "office"}:
        return 1.0
    if "kitchen" in t:
        return 0.9
    if "bath" in t or "toilet" in t or "wc" in t:
        return 0.5
    if "service" in t or "storage" in t or "utility" in t or "laundry" in t:
        return 0.4
    return 1.0


def _area_weight(room: Any) -> float:
    for key in ("area_weight", "area_preference_weight", "priority"):
        try:
            value = float(getattr(room, key))
            if value > 0:
                return value
        except Exception:
            pass
    return _default_area_weight(_room_type(room))


@dataclass(frozen=True)
class CapacityAwareAreaAllocationConfig:
    enabled: bool = False
    apply: bool = False
    strict: bool = False
    capacity_source: str = "max_variant_effective_capacity"
    capacity_slack: float = 1.0
    reserve_area: float = 0.0
    area_epsilon: float = DEFAULT_AREA_EPSILON
    preserve_preferred_when_feasible: bool = True
    require_apply_for_target_overflow: bool = False


@dataclass
class PerRoomAreaAllocation:
    floor_id: str
    room_id: str
    room_type: str
    raw_allocation_target_area: float
    preferred_target_area: float
    effective_preferred_area: float
    geometry_target_area: float
    min_area: float
    max_area: float
    area_weight: float
    area_reduction: float
    target_changed: bool
    preferred_target_clamped_to_min: bool = False
    min_area_source: str = "max(preferred*0.85,min_width*min_depth)"
    max_area_source: str = "preferred*1.15"
    min_area_frozen_before_capacity_allocation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "floor_id": self.floor_id,
            "room_id": self.room_id,
            "room_type": self.room_type,
            "raw_allocation_target_area": _r(self.raw_allocation_target_area),
            "preferred_target_area": _r(self.preferred_target_area),
            "effective_preferred_area": _r(self.effective_preferred_area),
            "geometry_target_area": _r(self.geometry_target_area),
            "min_area": _r(self.min_area),
            "max_area": _r(self.max_area),
            "area_weight": _r(self.area_weight),
            "area_reduction": _r(self.area_reduction),
            "target_changed": bool(self.target_changed),
            "preferred_target_clamped_to_min": bool(self.preferred_target_clamped_to_min),
            "min_area_source": self.min_area_source,
            "max_area_source": self.max_area_source,
            "min_area_frozen_before_capacity_allocation": bool(self.min_area_frozen_before_capacity_allocation),
        }


@dataclass
class CapacityAwareAreaAllocationResult:
    enabled: bool
    applied: bool
    status: str
    floor_id: str
    capacity_source: str
    raw_capacity_budget: float
    capacity_slack: float
    reserve_area: float
    epsilon: float
    allocation_capacity_budget_effective: float
    best_capacity_variant_id: str = ""
    actual_island_area_reference: float = 0.0
    raw_allocation_target_area_sum: float = 0.0
    preferred_target_area_sum: float = 0.0
    geometry_target_area_sum: float = 0.0
    min_area_sum: float = 0.0
    max_area_sum: float = 0.0
    required_reduction: float = 0.0
    total_compressible_area: float = 0.0
    compression_plan_feasible: bool = False
    area_compression_applied: bool = False
    per_room_allocation: List[PerRoomAreaAllocation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    next_action_hint: str = ""
    area_allocation_version: str = VERSION
    area_allocation_id: str = ""
    area_target_hash_before: str = ""
    area_target_hash_after: str = ""
    preferred_target_source: str = "post_global_area_scale"
    global_area_scale_applied: bool = True
    allocation_scope: str = "global_floor_capacity"
    does_not_guarantee_per_island_feasibility: bool = True
    per_floor_allocation: bool = True
    cross_floor_budget_transfer: bool = False
    target_semantics: str = "preferred_soft_target"
    downstream_current_usage: str = "capacity_load_estimate"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "applied": bool(self.applied),
            "status": self.status,
            "floor_id": self.floor_id,
            "area_allocation_version": self.area_allocation_version,
            "area_allocation_id": self.area_allocation_id,
            "capacity_source": self.capacity_source,
            "raw_capacity_budget": _r(self.raw_capacity_budget),
            "capacity_slack": _r(self.capacity_slack),
            "reserve_area": _r(self.reserve_area),
            "epsilon": float(self.epsilon),
            "allocation_capacity_budget_effective": _r(self.allocation_capacity_budget_effective),
            "best_capacity_variant_id": self.best_capacity_variant_id,
            "actual_island_area_reference": _r(self.actual_island_area_reference),
            "raw_allocation_target_area_sum": _r(self.raw_allocation_target_area_sum),
            "preferred_target_area_sum": _r(self.preferred_target_area_sum),
            "geometry_target_area_sum": _r(self.geometry_target_area_sum),
            "min_area_sum": _r(self.min_area_sum),
            "max_area_sum": _r(self.max_area_sum),
            "required_reduction": _r(self.required_reduction),
            "total_compressible_area": _r(self.total_compressible_area),
            "compression_plan_feasible": bool(self.compression_plan_feasible),
            "area_compression_applied": bool(self.area_compression_applied),
            "area_target_hash_before": self.area_target_hash_before,
            "area_target_hash_after": self.area_target_hash_after,
            "target_hash_changed": self.area_target_hash_before != self.area_target_hash_after,
            "preferred_target_source": self.preferred_target_source,
            "global_area_scale_applied": bool(self.global_area_scale_applied),
            "allocation_scope": self.allocation_scope,
            "does_not_guarantee_per_island_feasibility": bool(self.does_not_guarantee_per_island_feasibility),
            "per_floor_allocation": bool(self.per_floor_allocation),
            "cross_floor_budget_transfer": bool(self.cross_floor_budget_transfer),
            "target_semantics": self.target_semantics,
            "downstream_current_usage": self.downstream_current_usage,
            "per_room_allocation": [p.to_dict() for p in sorted(self.per_room_allocation, key=lambda p: (p.floor_id, p.room_id))],
            "warnings": list(self.warnings),
            "next_action_hint": self.next_action_hint,
        }


def build_area_target_hash(floor_id: str, room_allocations: Sequence[PerRoomAreaAllocation], *, use_geometry: bool) -> str:
    rows = []
    for item in sorted(room_allocations, key=lambda p: (p.floor_id, p.room_id)):
        target = item.geometry_target_area if use_geometry else item.preferred_target_area
        rows.append(
            {
                "floor_id": str(floor_id),
                "room_id": item.room_id,
                "room_type": item.room_type,
                "target_area": _r(target, 6),
                "min_area": _r(item.min_area, 6),
                "max_area": _r(item.max_area, 6),
                "area_weight": _r(item.area_weight, 6),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _allocation_id(floor_id: str, before_hash: str, after_hash: str) -> str:
    payload = f"{VERSION}:{floor_id}:{before_hash}:{after_hash}"
    return "caa_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _variant_capacity(report: Any, capacity_source: str) -> Dict[str, Any]:
    if str(capacity_source or "") != "max_variant_effective_capacity":
        return {"raw_capacity_budget": 0.0, "best_capacity_variant_id": "", "actual_island_area_reference": 0.0}
    best_variant_id = ""
    best_effective = 0.0
    best_actual = 0.0
    for variant in list(getattr(report, "variants", []) or []):
        if not bool(getattr(variant, "valid", False)):
            continue
        effective = 0.0
        actual = 0.0
        for island in list(getattr(variant, "island_metrics", []) or []):
            effective += float(getattr(island, "effective_capacity_area", 0.0) or 0.0)
            actual += float(getattr(island, "area", 0.0) or 0.0)
        if effective > best_effective + DEFAULT_AREA_EPSILON or (
            abs(effective - best_effective) <= DEFAULT_AREA_EPSILON
            and str(getattr(variant, "variant_id", "")) < best_variant_id
        ):
            best_effective = effective
            best_actual = actual
            best_variant_id = str(getattr(variant, "variant_id", "") or "")
    return {
        "raw_capacity_budget": best_effective,
        "best_capacity_variant_id": best_variant_id,
        "actual_island_area_reference": best_actual,
    }


def _bounded_water_fill(
    *,
    remaining_budget: float,
    mins: Sequence[float],
    uppers: Sequence[float],
    weights: Sequence[float],
) -> List[float]:
    extras = [0.0 for _ in mins]
    remaining = max(0.0, float(remaining_budget))
    active = {i for i, (mn, up) in enumerate(zip(mins, uppers)) if up > mn + DEFAULT_AREA_EPSILON}
    while active and remaining > DEFAULT_AREA_EPSILON:
        total_weight = sum(max(DEFAULT_AREA_EPSILON, float(weights[i])) for i in active)
        if total_weight <= DEFAULT_AREA_EPSILON:
            break
        consumed = 0.0
        capped: List[int] = []
        for i in sorted(active):
            share = remaining * max(DEFAULT_AREA_EPSILON, float(weights[i])) / total_weight
            cap = max(0.0, float(uppers[i]) - float(mins[i]) - extras[i])
            add = min(cap, share)
            extras[i] += add
            consumed += add
            if cap - add <= DEFAULT_AREA_EPSILON:
                capped.append(i)
        if consumed <= DEFAULT_AREA_EPSILON:
            break
        remaining = max(0.0, remaining - consumed)
        for i in capped:
            active.discard(i)
    return [float(mins[i]) + extras[i] for i in range(len(mins))]


def build_capacity_aware_targets(
    *,
    floor_id: str,
    room_specs: Sequence[Any],
    report: Any,
    config: Optional[CapacityAwareAreaAllocationConfig] = None,
) -> CapacityAwareAreaAllocationResult:
    cfg = config or CapacityAwareAreaAllocationConfig()
    capacity = _variant_capacity(report, cfg.capacity_source)
    raw_capacity_budget = float(capacity.get("raw_capacity_budget", 0.0) or 0.0)
    effective_budget = max(
        0.0,
        raw_capacity_budget * float(cfg.capacity_slack) - float(cfg.reserve_area) - float(cfg.area_epsilon),
    )

    prepared: List[PerRoomAreaAllocation] = []
    for room in list(room_specs or []):
        if bool(getattr(room, "is_dummy", False)):
            continue
        rid = _room_id(room)
        rtype = _room_type(room)
        preferred = float(getattr(room, "target_area", 0.0) or 0.0)
        raw = float(getattr(room, "raw_allocation_target_area", preferred) or preferred)
        min_area = _room_min_area(room, preferred)
        max_area = _room_max_area(room, preferred)
        clamped = preferred < min_area - float(cfg.area_epsilon)
        effective_preferred = max(preferred, min_area)
        effective_preferred = min(effective_preferred, max_area)
        weight = _area_weight(room)
        prepared.append(
            PerRoomAreaAllocation(
                floor_id=str(floor_id),
                room_id=rid,
                room_type=rtype,
                raw_allocation_target_area=raw,
                preferred_target_area=preferred,
                effective_preferred_area=effective_preferred,
                geometry_target_area=effective_preferred,
                min_area=min_area,
                max_area=max_area,
                area_weight=weight,
                area_reduction=max(0.0, preferred - effective_preferred),
                target_changed=abs(effective_preferred - preferred) > float(cfg.area_epsilon),
                preferred_target_clamped_to_min=clamped,
            )
        )

    preferred_sum = sum(p.effective_preferred_area for p in prepared)
    raw_sum = sum(p.raw_allocation_target_area for p in prepared)
    preferred_original_sum = sum(p.preferred_target_area for p in prepared)
    min_sum = sum(p.min_area for p in prepared)
    max_sum = sum(p.max_area for p in prepared)
    warnings = []
    if any(p.preferred_target_clamped_to_min for p in prepared):
        warnings.append("preferred_target_below_min_area_clamped")

    status = "insufficient_capacity_metadata"
    compression_feasible = False
    compression_applied = False
    next_hint = ""
    if raw_capacity_budget <= float(cfg.area_epsilon):
        status = "insufficient_capacity_metadata"
        next_hint = "inspect_topology_effective_capacity_metadata"
    elif preferred_sum <= effective_budget + float(cfg.area_epsilon) and cfg.preserve_preferred_when_feasible:
        status = "preferred_within_capacity"
        compression_feasible = True
    elif min_sum <= effective_budget + float(cfg.area_epsilon) < preferred_sum:
        status = "target_overflow_min_feasible"
        compression_feasible = True
        targets = _bounded_water_fill(
            remaining_budget=effective_budget - min_sum,
            mins=[p.min_area for p in prepared],
            uppers=[min(p.effective_preferred_area, p.max_area) for p in prepared],
            weights=[p.area_weight for p in prepared],
        )
        for p, target in zip(prepared, targets):
            p.geometry_target_area = max(p.min_area, min(p.max_area, target))
            p.area_reduction = p.preferred_target_area - p.geometry_target_area
            p.target_changed = abs(p.geometry_target_area - p.preferred_target_area) > float(cfg.area_epsilon)
        compression_applied = True
        next_hint = "apply_geometry_targets_and_rebuild_topology_assignment"
    elif effective_budget < min_sum - float(cfg.area_epsilon):
        status = "min_capacity_infeasible"
        next_hint = "program_min_capacity_infeasible"
    else:
        status = "insufficient_capacity_metadata"
        next_hint = "inspect_capacity_aware_area_allocation_inputs"

    geometry_sum = sum(p.geometry_target_area for p in prepared)
    total_compressible = sum(max(0.0, p.effective_preferred_area - p.min_area) for p in prepared)
    required_reduction = max(0.0, preferred_sum - effective_budget)
    before_hash = build_area_target_hash(str(floor_id), prepared, use_geometry=False)
    after_hash = build_area_target_hash(str(floor_id), prepared, use_geometry=True)
    return CapacityAwareAreaAllocationResult(
        enabled=bool(cfg.enabled),
        applied=bool(cfg.apply and compression_applied),
        status=status,
        floor_id=str(floor_id),
        capacity_source=str(cfg.capacity_source),
        raw_capacity_budget=raw_capacity_budget,
        capacity_slack=float(cfg.capacity_slack),
        reserve_area=float(cfg.reserve_area),
        epsilon=float(cfg.area_epsilon),
        allocation_capacity_budget_effective=effective_budget,
        best_capacity_variant_id=str(capacity.get("best_capacity_variant_id", "") or ""),
        actual_island_area_reference=float(capacity.get("actual_island_area_reference", 0.0) or 0.0),
        raw_allocation_target_area_sum=raw_sum,
        preferred_target_area_sum=preferred_original_sum,
        geometry_target_area_sum=geometry_sum,
        min_area_sum=min_sum,
        max_area_sum=max_sum,
        required_reduction=required_reduction,
        total_compressible_area=total_compressible,
        compression_plan_feasible=bool(compression_feasible),
        area_compression_applied=bool(compression_applied),
        per_room_allocation=prepared,
        warnings=warnings,
        next_action_hint=next_hint,
        area_allocation_id=_allocation_id(str(floor_id), before_hash, after_hash),
        area_target_hash_before=before_hash,
        area_target_hash_after=after_hash,
    )


def apply_capacity_aware_targets_to_room_specs(
    room_specs: Sequence[Any],
    allocation: CapacityAwareAreaAllocationResult,
) -> None:
    by_id = {p.room_id: p for p in allocation.per_room_allocation}
    for room in list(room_specs or []):
        rid = _room_id(room)
        item = by_id.get(rid)
        if item is None:
            continue
        setattr(room, "raw_allocation_target_area", item.raw_allocation_target_area)
        setattr(room, "preferred_target_area", item.preferred_target_area)
        setattr(room, "geometry_target_area", item.geometry_target_area)
        setattr(room, "capacity_aware_min_area", item.min_area)
        setattr(room, "capacity_aware_max_area", item.max_area)
        setattr(room, "area_weight", item.area_weight)
        setattr(room, "area_allocation_id", allocation.area_allocation_id)
        setattr(room, "area_target_hash", allocation.area_target_hash_after)
        room.target_area = float(item.geometry_target_area)

