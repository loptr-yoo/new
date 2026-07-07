from __future__ import annotations

from typing import Any, Dict, Optional


class LayoutGenerationError(RuntimeError):
    """Base class for geometry/layout failures with optional floor context."""

    def __init__(
        self,
        message: str,
        *,
        floor_number: Optional[int] = None,
        floor_id: Optional[str] = None,
        max_gap_area: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.floor_number = int(floor_number) if floor_number is not None else None
        self.floor_id = str(floor_id) if floor_id is not None else (
            f"F{self.floor_number}" if self.floor_number is not None else None
        )
        self.max_gap_area = float(max_gap_area) if max_gap_area is not None else None
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def with_floor(self, floor_number: Optional[int] = None, floor_id: Optional[str] = None) -> "LayoutGenerationError":
        if self.floor_number is None and floor_number is not None:
            self.floor_number = int(floor_number)
        if self.floor_id is None:
            if floor_id is not None:
                self.floor_id = str(floor_id)
            elif self.floor_number is not None:
                self.floor_id = f"F{self.floor_number}"
        return self


class LayoutCoverageError(LayoutGenerationError):
    """Raised when macro-scale unassigned interior space remains after layout."""

    def __init__(
        self,
        message: str,
        *,
        floor_number: Optional[int] = None,
        floor_id: Optional[str] = None,
        max_gap_area: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
        semantic_repair_allowed: bool = True,
    ) -> None:
        meta = dict(metadata or {})
        if stage is not None:
            meta.setdefault("stage", str(stage))
        meta.setdefault("failure_kind", "coverage")
        meta.setdefault("semantic_repair_allowed", bool(semantic_repair_allowed))
        super().__init__(
            message,
            floor_number=floor_number,
            floor_id=floor_id,
            max_gap_area=max_gap_area,
            metadata=meta,
        )
        self.stage = str(stage) if stage is not None else meta.get("stage")
        self.semantic_repair_allowed = bool(semantic_repair_allowed)


class LayoutTopologyError(LayoutGenerationError):
    """Raised when generated doors do not form a valid reachable floor graph."""


class LayoutGeometryInvariantError(LayoutGenerationError):
    """Raised when immutable geometry contracts are violated."""

    def __init__(
        self,
        message: str,
        *,
        floor_number: Optional[int] = None,
        floor_id: Optional[str] = None,
        max_gap_area: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stage: Optional[str] = None,
    ) -> None:
        meta = dict(metadata or {})
        if stage is not None:
            meta.setdefault("stage", str(stage))
        meta.setdefault("failure_kind", "geometry_invariant")
        meta.setdefault("semantic_repair_allowed", False)
        super().__init__(
            message,
            floor_number=floor_number,
            floor_id=floor_id,
            max_gap_area=max_gap_area,
            metadata=meta,
        )
        self.stage = str(stage) if stage is not None else meta.get("stage")
        self.semantic_repair_allowed = False


class LayoutAssignmentError(LayoutTopologyError):
    """Raised when semantic rooms cannot be assigned to usable topology islands."""


class SemanticInvalidError(LayoutGenerationError):
    """Raised when semantic area requirements cannot produce a valid layout."""
