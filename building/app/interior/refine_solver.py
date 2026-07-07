from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

cp_model = importlib.import_module("ortools.sat.python.cp_model")

from .models import FurnitureSpec, LLMCoarseLayout, Obstacle, RefinedLayout, RefinedLayoutItem, RoomBoundary, Rotation

logger = logging.getLogger(__name__)


SCALE = 100


@dataclass(frozen=True)
class _ResolvedItem:
    furniture_id: str
    rotation: int
    priority: int
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
            priority=int(getattr(spec, "priority", 1)),
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

    x_vars: Dict[str, Any] = {}
    y_vars: Dict[str, Any] = {}
    dx_vars: Dict[str, Any] = {}
    dy_vars: Dict[str, Any] = {}
    x_intervals: List[Any] = []
    y_intervals: List[Any] = []

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
            rotation=cast(Rotation, i.rotation),
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


def solve_nonoverlap_layout_greedy(
    room: RoomBoundary,
    furnitures: List[FurnitureSpec],
    obstacles: List[Obstacle],
    coarse_layout: Optional[LLMCoarseLayout] = None,
    step: float = 0.1,
    margin: float = 0.05,
    center_validator: Optional[Callable[[float, float], bool]] = None,
) -> RefinedLayout:
    room_xmin_s = _to_int(room.x_min)
    room_ymin_s = _to_int(room.y_min)
    room_xmax_s = _to_int(room.x_max)
    room_ymax_s = _to_int(room.y_max)
    if room_xmin_s >= room_xmax_s or room_ymin_s >= room_ymax_s:
        raise ValueError("RoomBoundary 非法")

    if coarse_layout is None:
        center_x = (room_xmin_s + room_xmax_s) // 2
        center_y = (room_ymin_s + room_ymax_s) // 2
        items = []
        for f in furnitures:
            w = float(f.width)
            h = float(f.height)
            w_s = _to_int(w)
            h_s = _to_int(h)
            hw_s = (w_s + 1) // 2
            hh_s = (h_s + 1) // 2
            items.append(_ResolvedItem(
                furniture_id=f.id,
                rotation=0,
                priority=int(getattr(f, "priority", 1)),
                w_s=w_s,
                h_s=h_s,
                hw_s=hw_s,
                hh_s=hh_s,
                x_llm_s=center_x,
                y_llm_s=center_y,
            ))
    else:
        items = _resolve_inputs(coarse_layout, furnitures)

    obs_rects: List[Tuple[int, int, int, int]] = []
    for o in obstacles:
        ox0 = _to_int(o.x_min)
        oy0 = _to_int(o.y_min)
        ox1 = _to_int(o.x_max)
        oy1 = _to_int(o.y_max)
        if ox1 > ox0 and oy1 > oy0:
            obs_rects.append((ox0, oy0, ox1, oy1))

    step_s = max(1, _to_int(step))
    margin_s = max(0, _to_int(margin))

    def _inside_room(x0: int, y0: int, x1: int, y1: int) -> bool:
        return (
            x0 >= room_xmin_s + margin_s and
            y0 >= room_ymin_s + margin_s and
            x1 <= room_xmax_s - margin_s and
            y1 <= room_ymax_s - margin_s
        )

    def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (min(ax1, bx1) > max(ax0, bx0)) and (min(ay1, by1) > max(ay0, by0))

    def _valid_rect(
        rect: Tuple[int, int, int, int],
        placed: List[Tuple[int, int, int, int]],
        center_s: Optional[Tuple[int, int]] = None,
    ) -> bool:
        if not _inside_room(*rect):
            return False
        if center_validator is not None and center_s is not None:
            cx, cy = center_s
            if not center_validator(cx / SCALE, cy / SCALE):
                return False
        for o in obs_rects:
            if _overlaps(rect, o):
                return False
        for p in placed:
            if _overlaps(rect, p):
                return False
        return True

    def _rect_for_center(i: _ResolvedItem, cx_s: int, cy_s: int) -> Tuple[int, int, int, int]:
        x0 = cx_s - i.hw_s
        y0 = cy_s - i.hh_s
        return (x0, y0, x0 + i.w_s, y0 + i.h_s)

    def _clamp_center(i: _ResolvedItem, cx_s: int, cy_s: int) -> Tuple[int, int]:
        min_cx = room_xmin_s + margin_s + i.hw_s
        max_cx = room_xmax_s - margin_s - i.hw_s
        min_cy = room_ymin_s + margin_s + i.hh_s
        max_cy = room_ymax_s - margin_s - i.hh_s
        if min_cx > max_cx or min_cy > max_cy:
            return cx_s, cy_s
        cx_s = min(max(cx_s, min_cx), max_cx)
        cy_s = min(max(cy_s, min_cy), max_cy)
        return cx_s, cy_s

    def _candidate_centers(i: _ResolvedItem) -> List[Tuple[int, int]]:
        base_x, base_y = _clamp_center(i, i.x_llm_s, i.y_llm_s)
        out: List[Tuple[int, int]] = [(base_x, base_y)]
        max_r = 120
        for r in range(1, max_r + 1):
            d = r * step_s
            out.extend([
                (base_x + d, base_y),
                (base_x - d, base_y),
                (base_x, base_y + d),
                (base_x, base_y - d),
                (base_x + d, base_y + d),
                (base_x + d, base_y - d),
                (base_x - d, base_y + d),
                (base_x - d, base_y - d),
            ])
            if len(out) > 600:
                break
        return [(_clamp_center(i, cx, cy)) for cx, cy in out]

    def _scan_centers(i: _ResolvedItem) -> List[Tuple[int, int]]:
        min_cx = room_xmin_s + margin_s + i.hw_s
        max_cx = room_xmax_s - margin_s - i.hw_s
        min_cy = room_ymin_s + margin_s + i.hh_s
        max_cy = room_ymax_s - margin_s - i.hh_s
        out: List[Tuple[int, int]] = []
        if min_cx > max_cx or min_cy > max_cy:
            return out
        y = min_cy
        while y <= max_cy and len(out) < 5000:
            x = min_cx
            while x <= max_cx and len(out) < 5000:
                out.append((x, y))
                x += step_s
            y += step_s
        return out

    items_sorted = sorted(items, key=lambda it: (it.priority, -(it.w_s * it.h_s)))
    placed_rects: List[Tuple[int, int, int, int]] = []
    placed_centers: Dict[str, Tuple[int, int]] = {}
    dropped: List[str] = []

    for it in items_sorted:
        placed = False
        for cx_s, cy_s in _candidate_centers(it):
            rect = _rect_for_center(it, cx_s, cy_s)
            if _valid_rect(rect, placed_rects, center_s=(cx_s, cy_s)):
                placed_rects.append(rect)
                placed_centers[it.furniture_id] = (cx_s, cy_s)
                placed = True
                break
        if placed:
            continue
        for cx_s, cy_s in _scan_centers(it):
            rect = _rect_for_center(it, cx_s, cy_s)
            if _valid_rect(rect, placed_rects, center_s=(cx_s, cy_s)):
                placed_rects.append(rect)
                placed_centers[it.furniture_id] = (cx_s, cy_s)
                placed = True
                break
        if not placed:
            dropped.append(it.furniture_id)

    out_items: List[RefinedLayoutItem] = []
    for it in items:
        if it.furniture_id not in placed_centers:
            continue
        cx_s, cy_s = placed_centers[it.furniture_id]
        out_items.append(RefinedLayoutItem(
            furniture_id=it.furniture_id,
            cx=round(cx_s / SCALE, 4),
            cy=round(cy_s / SCALE, 4),
            rotation=cast(Rotation, it.rotation),
        ))

    return RefinedLayout(
        status="fallback",
        solver="greedy_packer",
        objective_l1=None,
        reasoning="\n".join([
            "- Greedy pack: enforce non-overlap among furnitures and obstacles",
            "- Prefer staying near coarse centers when provided",
        ]),
        items=out_items,
        warnings=(["nonoverlap_greedy_fallback"] + ([f"dropped_furnitures={dropped}"] if dropped else [])),
    )
