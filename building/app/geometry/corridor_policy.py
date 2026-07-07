"""Shared corridor sizing policy for geometry and pipeline handoff."""

from __future__ import annotations


def normalize_corridor_width(requested_width: float, corridor_mode: str) -> float:
    """Return the corridor width that all pipeline stages must use."""

    width = float(requested_width)
    mode = str(corridor_mode or "").lower()
    if mode == "organic":
        return float(min(max(width, 1.5), 1.8))
    return width
