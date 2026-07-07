from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .coarse_layout_agent import AsyncOpenAI, generate_coarse_layout
from .models import FurnitureSpec, Obstacle, RefinedLayout, RefinedLayoutItem, RoomBoundary
from .refine_solver import solve_nonoverlap_layout_greedy, solve_refined_layout

logger = logging.getLogger(__name__)


def _rect_bbox(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (min(ax1, bx1) > max(ax0, bx0)) and (min(ay1, by1) > max(ay0, by0))


def _validate_nonoverlap(
    room: RoomBoundary,
    furnitures: List[FurnitureSpec],
    obstacles: List[Obstacle],
    layout: RefinedLayout,
    margin: float = 0.0,
) -> bool:
    spec_by_id = {f.id: f for f in furnitures}
    placed: List[Tuple[float, float, float, float]] = []
    room_rect = (float(room.x_min) + margin, float(room.y_min) + margin, float(room.x_max) - margin, float(room.y_max) - margin)
    if room_rect[0] >= room_rect[2] or room_rect[1] >= room_rect[3]:
        return False
    obs_rects = [(float(o.x_min), float(o.y_min), float(o.x_max), float(o.y_max)) for o in obstacles]

    for it in layout.items:
        spec = spec_by_id.get(it.furniture_id)
        if spec is None:
            return False
        w = float(spec.width)
        h = float(spec.height)
        if int(it.rotation) in (90, 270):
            w, h = h, w
        r = _rect_bbox(float(it.cx), float(it.cy), w, h)
        if r[0] < room_rect[0] or r[1] < room_rect[1] or r[2] > room_rect[2] or r[3] > room_rect[3]:
            return False
        for o in obs_rects:
            if _overlap(r, o):
                return False
        for p in placed:
            if _overlap(r, p):
                return False
        placed.append(r)
    return True


def _apply_density_cap(
    room: RoomBoundary,
    furnitures: List[FurnitureSpec],
    cap_ratio: float = 0.4,
) -> Tuple[List[FurnitureSpec], List[str]]:
    room_area = float(room.x_max - room.x_min) * float(room.y_max - room.y_min)
    if room_area <= 0:
        return furnitures, []
    limit = float(cap_ratio) * room_area
    kept = list(furnitures)
    dropped: List[str] = []

    def _area(f: FurnitureSpec) -> float:
        return float(f.width) * float(f.height)

    def _sum_area(fs: List[FurnitureSpec]) -> float:
        return float(sum(_area(f) for f in fs))

    kept.sort(key=lambda f: (int(getattr(f, "priority", 1)), -_area(f)))
    while kept and _sum_area(kept) > limit:
        droppable = [f for f in kept if int(getattr(f, "priority", 1)) > 0]
        if not droppable:
            break
        victim = max(droppable, key=lambda f: (int(getattr(f, "priority", 1)), _area(f)))
        kept = [f for f in kept if f.id != victim.id]
        dropped.append(victim.id)
    return kept, dropped


async def layout_room_pipeline(
    room: RoomBoundary,
    furnitures: List[FurnitureSpec],
    obstacles: List[Obstacle],
    client: AsyncOpenAI,
    model: Optional[str] = None,
    time_limit: float = 5.0,
) -> RefinedLayout:
    room_area = float(room.x_max - room.x_min) * float(room.y_max - room.y_min)
    furn_area_sum = sum(float(f.width) * float(f.height) for f in furnitures)
    if room_area <= 0:
        raise ValueError("RoomBoundary invalid")
    furnitures, pre_dropped = _apply_density_cap(room, furnitures, cap_ratio=0.4)
    furn_area_sum = sum(float(f.width) * float(f.height) for f in furnitures)
    if not furnitures:
        return RefinedLayout(
            status="fallback",
            solver="dropped_all_by_density_cap",
            objective_l1=None,
            reasoning="- Dropped all furnitures due to density cap",
            items=[],
            warnings=[f"dropped_furnitures={pre_dropped}"],
        )

    coarse = await generate_coarse_layout(
        room=room,
        furnitures=furnitures,
        obstacles=obstacles,
        client=client,
        model=model,
    )

    try:
        refined = solve_refined_layout(
            coarse_layout=coarse,
            room=room,
            furnitures=furnitures,
            obstacles=obstacles,
            time_limit=time_limit,
        )
        if _validate_nonoverlap(room, furnitures, obstacles, refined):
            if pre_dropped:
                refined.warnings.append(f"dropped_furnitures={pre_dropped}")
            return refined
    except Exception as e:
        logger.warning(f"Refine solver failed, fallback to coarse layout: {type(e).__name__}: {e}")

    try:
        refined = solve_nonoverlap_layout_greedy(
            room=room,
            furnitures=furnitures,
            obstacles=obstacles,
            coarse_layout=coarse,
        )
        if pre_dropped:
            refined.warnings.append(f"dropped_furnitures={pre_dropped}")
        if _validate_nonoverlap(room, furnitures, obstacles, refined):
            return refined
        raise ValueError("nonoverlap_greedy returned invalid layout")
    except Exception as e2:
        logger.warning(f"Greedy pack failed: {type(e2).__name__}: {e2}")

    return RefinedLayout(
        status="error",
        solver="interior_failed",
        objective_l1=None,
        reasoning="- All interior solvers failed; refuse to output overlapping layout",
        items=[],
        warnings=["refine_solver_failed", "greedy_pack_failed"] + ([f"dropped_furnitures={pre_dropped}"] if pre_dropped else []),
    )
