from __future__ import annotations

import logging
from typing import List, Optional

from .coarse_layout_agent import AsyncOpenAI, generate_coarse_layout
from .models import FurnitureSpec, Obstacle, RefinedLayout, RefinedLayoutItem, RoomBoundary
from .refine_solver import solve_nonoverlap_layout_greedy, solve_refined_layout

logger = logging.getLogger(__name__)


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
    if furn_area_sum > 0.8 * room_area:
        raise ValueError("Room is too crowded")

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
        return refined
    except Exception as e:
        logger.warning(f"Refine solver failed, fallback to coarse layout: {type(e).__name__}: {e}")

    try:
        return solve_nonoverlap_layout_greedy(
            room=room,
            furnitures=furnitures,
            obstacles=obstacles,
            coarse_layout=coarse,
        )
    except Exception as e2:
        logger.warning(f"Greedy pack failed, fallback to coarse layout: {type(e2).__name__}: {e2}")

    items = [RefinedLayoutItem(
        furniture_id=it.furniture_id,
        cx=it.cx,
        cy=it.cy,
        rotation=it.rotation,
    ) for it in coarse.items]
    return RefinedLayout(
        status="fallback",
        solver="fallback_coarse",
        objective_l1=None,
        reasoning="\n".join([
            "- Fallback: solver+greedy both failed",
            "- Return coarse layout to keep pipeline running",
        ]),
        items=items,
        warnings=["refine_solver_failed", "greedy_pack_failed"],
    )
