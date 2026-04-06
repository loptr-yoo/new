from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple, cast

import cma  # type: ignore
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from ...models import RoomAllocation
from .debug_exporter import export_layout_svg
from .orthogonalization import orthogonalize_layout
from .power_diagram import generate_power_diagram, precompute_grid

logger = logging.getLogger(__name__)



def _build_bounds(boundary: Polygon, n: int) -> Tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = boundary.bounds
    lower: List[float] = []
    upper: List[float] = []
    for _ in range(n):
        lower.extend([float(minx), float(miny), 10.0])
        upper.extend([float(maxx), float(maxy), 5000.0])
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _unpack_params(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[List[Tuple[float, float]], List[float]]:
    x = np.asarray(x, dtype=float)
    x = np.minimum(np.maximum(x, lower), upper)
    n = x.size // 3
    seeds: List[Tuple[float, float]] = []
    weights: List[float] = []
    for i in range(n):
        xi = float(x[3 * i])
        yi = float(x[3 * i + 1])
        wi = float(x[3 * i + 2])
        seeds.append((xi, yi))
        weights.append(wi)
    return seeds, weights


def _sample_points_within(boundary: Polygon, n: int, max_rounds: int = 100) -> List[Tuple[float, float]]:
    minx, miny, maxx, maxy = boundary.bounds
    rng = np.random.default_rng()
    batch = max(n * 4, 256)
    out: List[Tuple[float, float]] = []
    try:
        from shapely import contains_xy  # type: ignore[attr-defined]
        _fast_check = lambda xs, ys: contains_xy(boundary, xs, ys)
    except Exception:
        _fast_check = None

    for _ in range(max_rounds):
        xs = rng.uniform(minx, maxx, size=batch)
        ys = rng.uniform(miny, maxy, size=batch)
        if _fast_check is not None:
            mask = np.asarray(_fast_check(xs, ys), dtype=bool)
        else:
            mask = np.array([Point(float(x), float(y)).within(boundary) for x, y in zip(xs, ys)])
        for x, y in zip(xs[mask], ys[mask]):
            out.append((float(x), float(y)))
            if len(out) >= n:
                return out[:n]
    # Fallback for very narrow boundaries
    while len(out) < n:
        p = boundary.representative_point()
        out.append((float(p.x), float(p.y)))
    return out[:n]


def optimize_island(
    boundary: Polygon,
    rooms: List[RoomAllocation],
    resolution: float = 1.0,
) -> Tuple[List[Tuple[float, float]], List[float], List[BaseGeometry]]:
    if not rooms:
        return [], [], []
    if boundary.is_empty:
        return [], [], []
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    if boundary.is_empty:
        return [], [], []
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    n = len(rooms)
    lower, upper = _build_bounds(boundary, n)

    # --- Phase 4: Smart Initialization ---
    seeds0 = _sample_points_within(boundary, n=n)

    # 采光感知：requires_window 的房间种子靠外墙
    exterior = boundary.exterior
    for i, r in enumerate(rooms):
        if r.requires_window:
            point_on_ext = exterior.interpolate(np.random.uniform(0, exterior.length))
            centroid = boundary.centroid
            dx = centroid.x - point_on_ext.x
            dy = centroid.y - point_on_ext.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-6:
                offset = 2.0
                sx = point_on_ext.x + offset * dx / dist
                sy = point_on_ext.y + offset * dy / dist
                if boundary.covers(Point(sx, sy)):
                    seeds0[i] = (sx, sy)
                else:
                    # 安全回退：偏移量太大导致出界，尝试缩小偏移
                    sx = point_on_ext.x + 0.5 * dx / dist
                    sy = point_on_ext.y + 0.5 * dy / dist
                    if boundary.covers(Point(sx, sy)):
                        seeds0[i] = (sx, sy)
                    # 否则保留原始随机种子

    # 权重正比于目标面积
    avg_area = sum(r.target_area for r in rooms) / n
    base_weight_factor = 1000.0 / max(avg_area, 1e-6)
    weights0 = [r.target_area * base_weight_factor for r in rooms]

    x0_list: List[float] = []
    for (x, y), w in zip(seeds0, weights0):
        x0_list.extend([float(x), float(y), float(w)])
    x0 = np.asarray(x0_list, dtype=float)

    minx, miny, maxx, maxy = boundary.bounds
    span = max(float(maxx - minx), float(maxy - miny))
    sigma0 = 0.2 * span
    if not math.isfinite(sigma0) or sigma0 <= 1e-6:
        sigma0 = 1.0

    target_w_std = 500.0
    cma_stds: List[float] = []
    for _ in range(n):
        cma_stds.extend([1.0, 1.0, float(target_w_std / sigma0)])

    cluster_tags_by_room: List[List[str]] = []
    room_index_by_name = {r.room_name: i for i, r in enumerate(rooms)}
    explicit_edges: List[Tuple[int, int]] = []
    for i, r in enumerate(rooms):
        tags = list(r.adjacency_tags or [])
        cluster = []
        for t in tags:
            if isinstance(t, str) and t.startswith("adj:"):
                target_name = t[4:]
                j = room_index_by_name.get(target_name)
                if j is not None and j != i:
                    a, b = (i, j) if i < j else (j, i)
                    explicit_edges.append((a, b))
            else:
                if isinstance(t, str) and t:
                    cluster.append(t)
        cluster_tags_by_room.append(cluster)

    cluster_map = {}
    for i, tags in enumerate(cluster_tags_by_room):
        for t in tags:
            cluster_map.setdefault(t, []).append(i)

    # --- Phase 3: Precompute static grids for each annealing resolution ---
    grid_cache = {res: precompute_grid(boundary, res) for res in [2.0, 1.0, 0.5]}

    best_seen = {"f": float("inf")}

    def objective(vec: np.ndarray, res: float) -> float:
        seeds, weights = _unpack_params(vec, lower, upper)

        # 1. 越界惩罚 (Out Penalty)
        out_penalty = 0.0
        for x, y in seeds:
            p = Point(float(x), float(y))
            if not boundary.covers(p):
                out_penalty += 10000.0 + 1000.0 * float(p.distance(boundary))

        topo_penalty = 0.0
        area_penalty = 0.0
        shape_penalty = 0.0
        adjacency_penalty = 0.0
        wind_penalty = 0.0
        width_penalty = 0.0

        eps_area = 1e-8
        pi4 = 4.0 * math.pi
        shape_weight = 100.0

        # 2. 调用底层切肉机生成多边形（使用缓存网格 + 返回 id_grid）
        cells: List[BaseGeometry] = []
        id_grid: Optional[np.ndarray] = None
        try:
            cached = grid_cache.get(res)
            raw = generate_power_diagram(
                boundary=boundary, seeds=seeds, weights=weights,
                resolution=float(res),
                precomputed=cached,
                return_id_grid=True,
            )
            cells, id_grid = cast(Tuple[List[BaseGeometry], np.ndarray], raw)
        except Exception:
            topo_penalty += 30000.0
            cells = [Polygon() for _ in range(n)]

        # 3. 房间级多目标评估
        eps = max(1e-6, float(res) * 0.51)
        for i in range(n):
            cell = cells[i] if i < len(cells) else Polygon()
            p = Point(float(seeds[i][0]), float(seeds[i][1]))
            room_req = rooms[i]

            # 3.1 空洞检查 — 不 continue，打通梯度流
            if cell.is_empty:
                topo_penalty += 50000.0
                actual_area = 0.0
            else:
                # 3.2 种子归属检查 (Seed Ownership)
                try:
                    covers_seed = bool(cell.buffer(eps).covers(p))
                except Exception:
                    covers_seed = False
                if not covers_seed:
                    topo_penalty += 1000.0 + 2000.0 * float(p.distance(cell))

                # 3.3 采光与靠窗检查 (Window/Facade Access)
                if room_req.requires_window:
                    try:
                        shared_facade = float(cell.intersection(boundary.exterior).length)
                    except Exception:
                        shared_facade = 0.0
                    required_facade = 1.2
                    if shared_facade < required_facade:
                        deficit = required_facade - shared_facade
                        wind_penalty += 1000.0 * deficit

                # 3.4 拓扑与飞地检查 (Topology)
                if isinstance(cell, MultiPolygon):
                    k = float(max(0, len(cell.geoms) - 1))
                    topo_penalty += 2000.0 + 1000.0 * k
                    actual_area = float(cell.area)
                elif isinstance(cell, Polygon):
                    actual_area = float(cell.area)
                    if len(cell.interiors) > 0:
                        hole_area = 0.0
                        for ring in cell.interiors:
                            try:
                                hole_area += float(abs(Polygon(ring).area))
                            except Exception:
                                hole_area += 0.0
                        topo_penalty += min(5000.0, 50.0 * hole_area)
                else:
                    topo_penalty += 10000.0
                    actual_area = float(getattr(cell, "area", 0.0) or 0.0)

            # 3.5 核心面积拟合 — Huber Loss（所有房间必须经历，包括空房间）
            target_area = float(room_req.target_area)
            denom = max(abs(target_area), 1e-6)
            rel_err = abs(actual_area - target_area) / denom
            huber_threshold = 0.3
            if rel_err <= huber_threshold:
                area_loss = rel_err ** 2
            else:
                area_loss = huber_threshold * (2.0 * rel_err - huber_threshold)
            area_penalty += area_loss

            # 3.6 形状紧凑度与最小面宽 (Shape & Anti-Narrow)
            if not cell.is_empty and actual_area > eps_area:
                try:
                    perimeter = float(cell.length)
                    compactness = (perimeter * perimeter) / (pi4 * actual_area)
                    if compactness > 1.5:
                        shape_penalty += (compactness - 1.5) ** 2
                except Exception:
                    shape_penalty += 10.0
                try:
                    min_radius = 0.6
                    core_area = cell.buffer(-float(min_radius))
                    if core_area.is_empty:
                        width_penalty += 1000.0
                except Exception:
                    width_penalty += 1000.0

        # 4. 动线与相邻性检查 — Phase 3: Numpy 矩阵邻接
        all_edges = set(explicit_edges)
        for idxs in cluster_map.values():
            if len(idxs) < 2:
                continue
            for ii in range(len(idxs)):
                for jj in range(ii + 1, len(idxs)):
                    a, b = idxs[ii], idxs[jj]
                    if a > b:
                        a, b = b, a
                    all_edges.add((a, b))

        # Precompute grid slices once (avoid re-slicing per edge pair)
        if id_grid is not None:
            h_left = id_grid[:, :-1]
            h_right = id_grid[:, 1:]
            valid_h = (h_left != -1) & (h_right != -1)
            v_top = id_grid[:-1, :]
            v_bot = id_grid[1:, :]
            valid_v = (v_top != -1) & (v_bot != -1)

        for a, b in all_edges:
            ca = cells[a] if a < len(cells) else Polygon()
            cb = cells[b] if b < len(cells) else Polygon()
            if ca.is_empty or cb.is_empty:
                adjacency_penalty += 3000.0
                continue

            if id_grid is not None:
                h_count = int(np.sum(
                    valid_h & (
                        ((h_left == a) & (h_right == b)) |
                        ((h_left == b) & (h_right == a))
                    )
                ))
                v_count = int(np.sum(
                    valid_v & (
                        ((v_top == a) & (v_bot == b)) |
                        ((v_top == b) & (v_bot == a))
                    )
                ))
                shared_edge = (h_count + v_count) * float(res)
            else:
                # Shapely fallback（仅在异常时触发）
                tol = max(1e-6, float(res) * 0.25)
                try:
                    inter_area = float(ca.buffer(tol).intersection(cb.buffer(tol)).area)
                except Exception:
                    inter_area = 0.0
                shared_edge = inter_area / (2.0 * tol) if tol > 0 else 0.0

            required_edge = 0.6
            if shared_edge < required_edge:
                deficit = required_edge - shared_edge
                adjacency_penalty += 1000.0 * deficit

        # 5. 总分核算与探针打印
        total = float(out_penalty + topo_penalty + 20000.0 * area_penalty + shape_weight * shape_penalty + wind_penalty + width_penalty + adjacency_penalty)

        should_print = bool(total < best_seen["f"] - 1e-12) or bool(np.random.rand() < 0.005)
        if total < best_seen["f"]:
            best_seen["f"] = total
        if should_print:
            print(
                f"[Debug] Out:{out_penalty:.1f}, Topo:{topo_penalty:.1f}, Area:{(20000.0*area_penalty):.1f}, Shape:{(shape_weight*shape_penalty):.1f}, Window:{wind_penalty:.1f}, Width:{width_penalty:.1f}, Adj:{adjacency_penalty:.1f}"
            )

        return total

    options = {
        "bounds": [lower.tolist(), upper.tolist()],
        "popsize": 20,
        "maxiter": 500,
        "CMA_stds": cma_stds,
    }

    # --- Phase 1: Only CMA-ES path (no fallback) ---
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    maxiter = int(options.get("maxiter", 100))
    no_improve_count = 0
    best_f_history = float('inf')


    for gen in range(maxiter):
        # --- Phase 2: Adaptive annealing based on ratio ---
        ratio = gen / maxiter
        if ratio < 0.3:
            cur_res = 2.0
        elif ratio < 0.7:
            cur_res = 1.0
        else:
            cur_res = 0.5
        xs = es.ask()
        fs = [objective(np.asarray(x, dtype=float), cur_res) for x in xs]
        es.tell(xs, fs)
        es.disp()

        current_best = np.min(fs)
        # 如果改进小于 1e-4，计数器 +1
        if best_f_history - current_best < 1e-4:
            no_improve_count += 1
        else:
            no_improve_count = 0
            best_f_history = current_best

        # 如果连续 30 代分数不降，提前结束！
        if no_improve_count >= 30 and (gen / maxiter) > 0.7: # 确保至少进入了最细网格阶段
            print(f"Early stopping at gen {gen}")
            break
    xbest = np.asarray(es.result.xbest, dtype=float)

    best_seeds, best_weights = _unpack_params(xbest, lower, upper)
    logger.info("CMA-ES finished. Generating final layout at resolution=0.2 ...")

    # ---- 生成最终光栅化布局 ----
    final_cells = cast(
        List[BaseGeometry],
        generate_power_diagram(boundary=boundary, seeds=best_seeds, weights=best_weights, resolution=0.2),
    )

    # ---- 清洗 MultiPolygon → 取最大连通块 ----
    cleaned_cells: List[Polygon] = []
    for i, c in enumerate(final_cells):
        if isinstance(c, Polygon):
            cleaned_cells.append(c)
        elif isinstance(c, MultiPolygon):
            if len(c.geoms) > 0:
                cleaned_cells.append(max(c.geoms, key=lambda p: p.area))
            else:
                cleaned_cells.append(Polygon())
        else:
            cleaned_cells.append(Polygon())

    # ---- 日志：原始 cells 面积统计 ----
    for i, c in enumerate(cleaned_cells):
        area = float(c.area) if not c.is_empty else 0.0
        geom_type = type(final_cells[i]).__name__
        logger.info("  Room %d: area=%.2f m², geom_type=%s", i, area, geom_type)

    # ---- 阶段 A: 保存正交化前的光栅化布局 SVG ----
    debug_dir = os.path.join("artifacts")
    try:
        export_layout_svg(
            boundary, cleaned_cells, best_seeds,
            os.path.join(debug_dir, "layout_before_ortho.svg"),
            title="Before Orthogonalization (raw rasterized)",
        )
        logger.info("Saved pre-ortho SVG → artifacts/layout_before_ortho.svg")
    except Exception as exc:
        logger.warning("Failed to export pre-ortho SVG: %s", exc)

    # ---- 正交化 ----
    ortho_cells = orthogonalize_layout(
        boundary=boundary,
        cells=cleaned_cells,
        seeds=best_seeds,
    )

    # ---- 日志：正交化前后面积对比 ----
    for i in range(len(cleaned_cells)):
        area_before = float(cleaned_cells[i].area) if not cleaned_cells[i].is_empty else 0.0
        area_after = float(ortho_cells[i].area) if i < len(ortho_cells) and not ortho_cells[i].is_empty else 0.0
        delta = area_after - area_before
        logger.info("  Room %d: before=%.2f → after=%.2f (Δ=%.2f m²)", i, area_before, area_after, delta)

    # ---- 阶段 B: 保存正交化后的布局 SVG ----
    try:
        export_layout_svg(
            boundary, ortho_cells, best_seeds,
            os.path.join(debug_dir, "layout_after_ortho.svg"),
            title="After Orthogonalization",
        )
        logger.info("Saved post-ortho SVG → artifacts/layout_after_ortho.svg")
    except Exception as exc:
        logger.warning("Failed to export post-ortho SVG: %s", exc)

    return best_seeds, best_weights, [cast(BaseGeometry, c) for c in ortho_cells]
