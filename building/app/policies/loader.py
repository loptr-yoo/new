from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


POLICY_SCHEMA_VERSION = "policy.v1"
_POLICY_ROOT = Path(__file__).resolve().parent


@dataclass
class PolicyValidationReport:
    schema_version: str = POLICY_SCHEMA_VERSION
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_usages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fallback_count(self) -> int:
        return len(self.fallback_usages)

    def error(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def record_fallback(self, *, key: str, fallback_source: str, value: Any = None) -> None:
        self.fallback_usages.append(
            {"key": str(key), "fallback_source": str(fallback_source), "value": value}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "fallback_usages": list(self.fallback_usages),
            "fallback_count": self.fallback_count,
        }


def _read_policy_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is intentionally parsed as JSON-compatible YAML; invalid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if key == "__append__" and isinstance(value, Mapping):
            for list_key, list_value in value.items():
                if not isinstance(list_value, list):
                    base[list_key] = copy.deepcopy(list_value)
                    continue
                existing = base.get(list_key)
                if isinstance(existing, list):
                    existing.extend(copy.deepcopy(list_value))
                else:
                    base[list_key] = copy.deepcopy(list_value)
            continue
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base


def _validate_room_rules(policy: Mapping[str, Any], report: PolicyValidationReport) -> None:
    rooms = policy.get("room_rules", {})
    if not isinstance(rooms, Mapping) or not rooms:
        report.error("room_rules must be a non-empty object")
        return
    for room_type, rule in rooms.items():
        if not isinstance(rule, Mapping):
            report.error(f"room_rules.{room_type} must be an object")
            continue
        for key in ("min_area", "target_area", "max_area"):
            if key not in rule:
                report.error(f"room_rules.{room_type}.{key} is required")
        try:
            mn = float(rule.get("min_area"))
            target = float(rule.get("target_area"))
            mx = float(rule.get("max_area"))
        except Exception:
            report.error(f"room_rules.{room_type} min/target/max must be numeric")
            continue
        if not (mn <= target <= mx):
            report.error(f"room_rules.{room_type} must satisfy min_area <= target_area <= max_area")


def _validate_ratios(policy: Mapping[str, Any], report: PolicyValidationReport) -> None:
    corridor = policy.get("corridor_rules", {})
    core = policy.get("vertical_core_rules", {})
    if isinstance(corridor, Mapping):
        try:
            reserve = float(corridor.get("reserve_ratio", 0.0))
            if reserve < 0.0 or reserve >= 0.6:
                report.error("corridor_rules.reserve_ratio must be in [0, 0.6)")
        except Exception:
            report.error("corridor_rules.reserve_ratio must be numeric")
    if isinstance(core, Mapping):
        try:
            ratio = float(core.get("area_ratio", 0.0))
            if ratio <= 0.0 or ratio >= 0.5:
                report.error("vertical_core_rules.area_ratio must be in (0, 0.5)")
        except Exception:
            report.error("vertical_core_rules.area_ratio must be numeric")


def _validate_rule_references(policy: Mapping[str, Any], report: PolicyValidationReport) -> None:
    room_types = set((policy.get("room_rules") or {}).keys())
    for section in ("adjacency_rules", "door_rules", "window_rules"):
        rules = policy.get(section, {})
        if not isinstance(rules, Mapping):
            continue
        for key, value in rules.items():
            refs: list[Any] = []
            if key in room_types:
                pass
            elif key not in {"forbidden", "preferred", "required", "default_width", "required_room_types"}:
                report.warn(f"{section}.{key} does not match a known room type")
            if isinstance(value, list):
                refs.extend(value)
            elif isinstance(value, Mapping):
                for inner in value.values():
                    if isinstance(inner, list):
                        refs.extend(inner)
            for ref in refs:
                if isinstance(ref, str) and ref not in room_types:
                    report.warn(f"{section}.{key} references unknown room type {ref!r}")


def validate_policy(policy: Mapping[str, Any]) -> PolicyValidationReport:
    report = PolicyValidationReport()
    version = str(policy.get("policy_schema_version") or "")
    if version != POLICY_SCHEMA_VERSION:
        report.error(f"policy_schema_version must be {POLICY_SCHEMA_VERSION!r}, got {version!r}")
    _validate_room_rules(policy, report)
    _validate_ratios(policy, report)
    _validate_rule_references(policy, report)
    return report


def load_policy(
    archetype: str = "residential",
    *,
    request_overrides: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, Any], PolicyValidationReport]:
    policy: dict[str, Any] = {"policy_schema_version": POLICY_SCHEMA_VERSION}
    for name in (
        "common/room_rules.yaml",
        "common/adjacency_rules.yaml",
        "common/vertical_core_rules.yaml",
        "common/corridor_rules.yaml",
        "common/door_rules.yaml",
        "common/window_rules.yaml",
        f"archetypes/{str(archetype or 'residential').lower()}.yaml",
    ):
        _deep_merge(policy, _read_policy_file(_POLICY_ROOT / name))
    if request_overrides:
        _deep_merge(policy, dict(request_overrides))
    report = validate_policy(policy)
    return policy, report

