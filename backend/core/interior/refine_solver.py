from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from .models import FurnitureSpec, LLMCoarseLayout, Obstacle, RefinedLayout, RefinedLayoutItem, RoomBoundary

logger = logging.getLogger(__name__)


SCALE = 100


@dataclass(frozen=True)
class _ResolvedItem:
    furniture_id: str
    rotation: int
    w_s: int
    h_s: int
    hw_s: int
    hh_s: int
    x_llm_s: int
    y_llm_s: int


def _to_int(v: float) -> int:
    return int(round(float(v) * SCALE))


def _is_finite(v: float) -> bool:
    return math.isfinite(float(v))


def _normalize_rotation(r: int) -> int:
    if r in (0, 90, 180, 270):
        return r
    raise ValueError(f"rotation 非法：{r}")


def _resolve_inputs(
    coarse_layout: LLMCoarseLayout,
    furnitures: List[FurnitureSpec],
) -> List[_ResolvedItem]:
    furn_by_id: Dict[str, FurnitureSpec] = {f.id: f for f in furnitures}
    if len(furn_by_id) != len(furnitures):
        raise ValueError("furnitures.id 存在重复")

    items: List[_ResolvedItem] = []
    for it in coarse_layout.items:
        spec = furn_by_id.get(it.furniture_id)
        if spec is None:
            raise ValueError(f"LLMCoarseLayout 包含未知 furniture_id={it.furniture_id}")
        if not (_is_finite(it.cx) and _is_finite(it.cy)):
            raise ValueError(f"LLMCoarseLayout cx/cy 非法：furniture_id={it.furniture_id}")
        rot = _normalize_rotation(int(it.rotation))

        w = float(spec.width)
        h = float(spec.height)
        if rot in (90, 270):
            w, h = h, w

        w_s = _to_int(w)
        h_s = _to_int(h)
        if w_s <= 0 or h_s <= 0:
            raise ValueError(f"家具尺寸非法：furniture_id={it.furniture_id} w={w} h={h}")
        hw_s = (w_s + 1) // 2
        hh_s = (h_s + 1) // 2

        items.append(_ResolvedItem(
            furniture_id=it.furniture_id,
            rotation=rot,
            w_s=w_s,
            h_s=h_s,
            hw_s=hw_s,
            hh_s=hh_s,
            x_llm_s=_to_int(float(it.cx)),
            y_llm_s=_to_int(float(it.cy)),
        ))

    expected = {f.id for f in furnitures}
    got = {i.furniture_id for i in items}
    if expected != got:
        raise ValueError("LLMCoarseLayout items 与 furnitures 不一致（缺失/多余）")

    return items


def solve_refined_layout(
    coarse_layout: LLMCoarseLayout,
    room: RoomBoundary,
    furnitures: List[FurnitureSpec],
    obstacles: List[Obstacle],
    time_limit: float = 5.0,
) -> RefinedLayout:
    room_xmin_s = _to_int(room.x_min)
    room_ymin_s = _to_int(room.y_min)
    room_xmax_s = _to_int(room.x_max)
    room_ymax_s = _to_int(room.y_max)
    if room_xmin_s >= room_xmax_s or room_ymin_s >= room_ymax_s:
        raise ValueError("RoomBoundary 非法")

    items = _resolve_inputs(coarse_layout, furnitures)

    model = cp_model.CpModel()

    x_vars: Dict[str, cp_model.IntVar] = {}
    y_vars: Dict[str, cp_model.IntVar] = {}
    dx_vars: Dict[str, cp_model.IntVar] = {}
    dy_vars: Dict[str, cp_model.IntVar] = {}
    x_intervals: List[cp_model.IntervalVar] = []
    y_intervals: List[cp_model.IntervalVar] = []

    max_span = (room_xmax_s - room_xmin_s) + (room_ymax_s - room_ymin_s)
    diff_bound = max(10_000, int(max_span * 2))

    for i in items:
        sx_lb = room_xmin_s
        sx_ub = room_xmax_s - i.w_s
        sy_lb = room_ymin_s
        sy_ub = room_ymax_s - i.h_s
        if sx_ub < sx_lb or sy_ub < sy_lb:
            raise ValueError(f"房间太小，无法放入家具：furniture_id={i.furniture_id}")

        start_x = model.NewIntVar(sx_lb, sx_ub, f"start_x_{i.furniture_id}")
        end_x = model.NewIntVar(room_xmin_s, room_xmax_s, f"end_x_{i.furniture_id}")
        model.Add(end_x == start_x + i.w_s)
        x_interval = model.NewIntervalVar(start_x, i.w_s, end_x, f"x_interval_{i.furniture_id}")

        start_y = model.NewIntVar(sy_lb, sy_ub, f"start_y_{i.furniture_id}")
        end_y = model.NewIntVar(room_ymin_s, room_ymax_s, f"end_y_{i.furniture_id}")
        model.Add(end_y == start_y + i.h_s)
        y_interval = model.NewIntervalVar(start_y, i.h_s, end_y, f"y_interval_{i.furniture_id}")

        x_i = model.NewIntVar(room_xmin_s, room_xmax_s, f"cx_{i.furniture_id}")
        y_i = model.NewIntVar(room_ymin_s, room_ymax_s, f"cy_{i.furniture_id}")
        model.Add(x_i == start_x + i.hw_s)
        model.Add(y_i == start_y + i.hh_s)

        diff_x = model.NewIntVar(-diff_bound, diff_bound, f"diff_x_{i.furniture_id}")
        diff_y = model.NewIntVar(-diff_bound, diff_bound, f"diff_y_{i.furniture_id}")
        model.Add(diff_x == x_i - i.x_llm_s)
        model.Add(diff_y == y_i - i.y_llm_s)

        dx_i = model.NewIntVar(0, diff_bound, f"dx_{i.furniture_id}")
        dy_i = model.NewIntVar(0, diff_bound, f"dy_{i.furniture_id}")
        model.AddAbsEquality(dx_i, diff_x)
        model.AddAbsEquality(dy_i, diff_y)

        x_vars[i.furniture_id] = x_i
        y_vars[i.furniture_id] = y_i
        dx_vars[i.furniture_id] = dx_i
        dy_vars[i.furniture_id] = dy_i
        x_intervals.append(x_interval)
        y_intervals.append(y_interval)

    for k, o in enumerate(obstacles):
        ox0 = _to_int(o.x_min)
        oy0 = _to_int(o.y_min)
        ox1 = _to_int(o.x_max)
        oy1 = _to_int(o.y_max)
        w = ox1 - ox0
        h = oy1 - oy0
        if w <= 0 or h <= 0:
            continue
        sx = model.NewConstant(ox0)
        ex = model.NewConstant(ox0 + w)
        sy = model.NewConstant(oy0)
        ey = model.NewConstant(oy0 + h)
        x_intervals.append(model.NewIntervalVar(sx, w, ex, f"obs_x_{k}"))
        y_intervals.append(model.NewIntervalVar(sy, h, ey, f"obs_y_{k}"))

    model.AddNoOverlap2D(x_intervals, y_intervals)
    model.Minimize(sum(dx_vars.values()) + sum(dy_vars.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Refine solver failed: {status_name}")

    refined_items: List[RefinedLayoutItem] = []
    l1_sum_s = 0
    for i in items:
        x_val = int(solver.Value(x_vars[i.furniture_id]))
        y_val = int(solver.Value(y_vars[i.furniture_id]))
        refined_items.append(RefinedLayoutItem(
            furniture_id=i.furniture_id,
            cx=round(x_val / SCALE, 4),
            cy=round(y_val / SCALE, 4),
            rotation=i.rotation,
        ))
        l1_sum_s += int(solver.Value(dx_vars[i.furniture_id])) + int(solver.Value(dy_vars[i.furniture_id]))

    status_out = "optimal" if status == cp_model.OPTIMAL else "feasible"
    return RefinedLayout(
        status=status_out,
        solver="ortools_cpsat",
        objective_l1=round(l1_sum_s / SCALE, 4),
        reasoning="\n".join([
            "- CP-SAT refine: minimize L1 displacement to LLM coarse centers",
            f"- status: {status_out} ({status_name})",
        ]),
        items=refined_items,
        warnings=[],
    )
