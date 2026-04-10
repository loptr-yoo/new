"""
island_partition_solver.py

岛屿房间划分求解器：Treemap Warm Start + 连续坐标 MIQP

算法流程：
1. Squarified Treemap 生成初始布局（< 1ms）
2. CP-SAT 整数规划优化（1-5s）
3. 边界裁剪（适配非矩形岛屿）

正交性保证：房间定义为 (x, y, w, d) 矩形，天然正交。
回退机制：MIQP 失败时退化为 Treemap-only。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from typing import Dict

from shapely.geometry import Polygon, box

from .treemap import flat_squarify as _squarify_impl
from .treemap import (
    _worst_ratio,
    _layout_row,
    _do_layout_row,
    _do_layout_row_v,
    hierarchical_treemap,
)
from .room_spec import (
    IslandContext,
    RoomSpec as SemanticRoomSpec,
    SolverConfig,
    WarmStartRect,
    ZoneType,
)

logger = logging.getLogger(__name__)


# ============================================================
# 数据类型
# ============================================================


@dataclass
class RoomSpec:
    """房间规格输入"""

    room_id: str
    room_type: str
    target_area: float  # m²
    area_tolerance: float = 0.1  # ±10%
    min_width: float = 2.0  # m
    min_depth: float = 2.0  # m
    adjacency_required: List[str] = field(default_factory=list)
    adjacency_forbidden: List[str] = field(default_factory=list)
    window_access: bool = False


@dataclass
class RoomResult:
    """房间划分结果"""

    room_id: str
    x: float  # 左下角 x
    y: float  # 左下角 y
    width: float
    depth: float

    @property
    def actual_area(self) -> float:
        return self.width * self.depth

    @property
    def polygon(self) -> Polygon:
        return box(self.x, self.y, self.x + self.width, self.y + self.depth)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.depth)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.depth / 2)


# ============================================================
# 主求解器
# ============================================================


class IslandPartitionSolver:
    """
    岛屿房间划分求解器

    用法：
        solver = IslandPartitionSolver(island_polygon, room_specs)
        results = solver.solve()
    """

    def __init__(
        self,
        island_polygon: Polygon,
        rooms: List[RoomSpec],
        solver: str = "cpsat",
        time_limit: float = 10.0,
        use_warm_start: bool = True,
    ):
        self.island = island_polygon
        self.rooms = rooms
        self.solver_type = solver
        self.time_limit = time_limit
        self.use_warm_start = use_warm_start

        # 岛屿 AABB
        self.x_min, self.y_min, self.x_max, self.y_max = island_polygon.bounds
        self.W = self.x_max - self.x_min
        self.D = self.y_max - self.y_min

        self.n = len(rooms)
        self.room_index = {r.room_id: i for i, r in enumerate(rooms)}

        # 预检查
        total_required = sum(r.target_area * (1 - r.area_tolerance) for r in rooms)
        if total_required > island_polygon.area * 1.5:
            raise ValueError(
                f"Island too small: area={island_polygon.area:.1f}m², "
                f"rooms require at least {total_required:.1f}m²"
            )

        # 修正不合理的 min_width/min_depth
        for r in self.rooms:
            if r.min_width > self.W * 0.9:
                logger.warning(
                    "Room %s min_width=%.1f exceeds island width=%.1f, clamping",
                    r.room_id, r.min_width, self.W,
                )
                r.min_width = self.W * 0.9
            if r.min_depth > self.D * 0.9:
                logger.warning(
                    "Room %s min_depth=%.1f exceeds island depth=%.1f, clamping",
                    r.room_id, r.min_depth, self.D,
                )
                r.min_depth = self.D * 0.9

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def solve(self) -> List[RoomResult]:
        """执行三阶段求解，MIQP 失败时回退 Treemap-only"""
        # Stage 1: Treemap warm start
        t0 = time.perf_counter()
        warm_start = self._generate_treemap_warmstart()
        logger.info(
            "Treemap warm start: %.1fms, %d rooms",
            (time.perf_counter() - t0) * 1000,
            len(warm_start),
        )

        # 快捷路径：≤2 个房间直接用 treemap，跳过 MIQP
        if len(self.rooms) <= 2:
            logger.info("≤2 rooms, using treemap-only (skip MIQP)")
            results = warm_start
            results = self._clip_to_boundary(results)
            return results

        # Stage 2: MIQP 优化（带回退）
        try:
            t1 = time.perf_counter()
            if self.solver_type == "gurobi":
                results = self._solve_gurobi(warm_start)
            else:
                results = self._solve_cpsat(warm_start)
            logger.info(
                "MIQP solve: %.1fms", (time.perf_counter() - t1) * 1000
            )
        except Exception as e:
            logger.warning("MIQP failed (%s), falling back to treemap-only", e)
            results = warm_start

        # Stage 3: 边界裁剪
        results = self._clip_to_boundary(results)
        return results

    # ----------------------------------------------------------
    # Stage 1: Treemap Warm Start
    # ----------------------------------------------------------

    def _generate_treemap_warmstart(self) -> List[RoomResult]:
        """Squarified Treemap 生成初始布局"""
        areas = [r.target_area for r in self.rooms]
        total_target = sum(areas)
        aabb_area = self.W * self.D

        # 按面积比例分配到 AABB
        normalized = [a / total_target * aabb_area for a in areas]

        rects = self._squarify(normalized, 0, 0, self.W, self.D)

        warm_start = []
        for room, rect in zip(self.rooms, rects):
            warm_start.append(
                RoomResult(
                    room_id=room.room_id,
                    x=rect[0] + self.x_min,
                    y=rect[1] + self.y_min,
                    width=max(rect[2], room.min_width),
                    depth=max(rect[3], room.min_depth),
                )
            )
        return warm_start

    @staticmethod
    def _squarify(
        sizes: List[float], x: float, y: float, w: float, h: float
    ) -> List[Tuple[float, float, float, float]]:
        """Squarified treemap（委托到 treemap.py 模块）。"""
        return _squarify_impl(sizes, x, y, w, h)

    # ----------------------------------------------------------
    # 拓扑冻结：Treemap → 正交位置关系 → 线性约束
    # ----------------------------------------------------------

    def _freeze_topology_from_treemap(
        self, warm_starts: List[RoomResult],
    ) -> List[Tuple[str, str, str]]:
        """从 Treemap Guillotine Cut 结果提取正交相对位置关系。

        Treemap 的 Guillotine Cut 保证 abs(dx) > abs(dy) 方向判断安全。
        返回 [(room_a_id, room_b_id, relation), ...] 其中 relation ∈ {left, right, below, above}
        """
        relations: List[Tuple[str, str, str]] = []
        for i, a in enumerate(warm_starts):
            for j, b in enumerate(warm_starts):
                if i >= j:
                    continue
                a_cx = a.x + a.width / 2
                a_cy = a.y + a.depth / 2
                b_cx = b.x + b.width / 2
                b_cy = b.y + b.depth / 2
                dx = b_cx - a_cx
                dy = b_cy - a_cy

                if abs(dx) > abs(dy):
                    relations.append((a.room_id, b.room_id, "left" if dx > 0 else "right"))
                else:
                    relations.append((a.room_id, b.room_id, "below" if dy > 0 else "above"))
        return relations

    def _add_frozen_topology_constraints(
        self, model, x, y, w, d, relations, SCALE: int,
    ):
        """将拓扑关系冻结为线性约束，消除 Big-M 二元变量 → MIQP 退化为 LP"""
        for room_a_id, room_b_id, rel in relations:
            ai = self.room_index.get(room_a_id)
            bi = self.room_index.get(room_b_id)
            if ai is None or bi is None:
                continue
            if rel == "left":
                model.Add(x[ai] + w[ai] <= x[bi])
            elif rel == "right":
                model.Add(x[bi] + w[bi] <= x[ai])
            elif rel == "below":
                model.Add(y[ai] + d[ai] <= y[bi])
            elif rel == "above":
                model.Add(y[bi] + d[bi] <= y[ai])

    # ----------------------------------------------------------
    # 走廊挂载约束 + INFEASIBLE 防护
    # ----------------------------------------------------------

    def _add_corridor_attachment_constraints(
        self, model, x, y, w, d, corridor_edges: List[str], SCALE: int,
    ):
        """强制需要走廊入口的房间贴边，含 INFEASIBLE 防护。

        corridor_edges: 岛屿接触走廊的方向列表 ['south', 'north', 'west', 'east']
        """
        if not corridor_edges:
            return

        island_bounds = self.island.bounds  # (minx, miny, maxx, maxy)

        # 选择主走廊边（取最长的走廊接触边）
        edge_lengths: Dict[str, float] = {}
        if "south" in corridor_edges:
            edge_lengths["south"] = island_bounds[2] - island_bounds[0]
        if "north" in corridor_edges:
            edge_lengths["north"] = island_bounds[2] - island_bounds[0]
        if "west" in corridor_edges:
            edge_lengths["west"] = island_bounds[3] - island_bounds[1]
        if "east" in corridor_edges:
            edge_lengths["east"] = island_bounds[3] - island_bounds[1]

        if not edge_lengths:
            return

        primary_edge = max(edge_lengths, key=lambda k: edge_lengths[k])
        edge_length = edge_lengths[primary_edge]

        # 收集需要贴边的房间，按面积降序（大房间保留走廊路权）
        access_rooms = [
            (i, r) for i, r in enumerate(self.rooms)
            if getattr(r, "needs_corridor_access", True)
        ]
        access_rooms.sort(key=lambda ir: ir[1].target_area, reverse=True)

        # INFEASIBLE 防护：边长不够时降级面积最小的房间
        total_min_width = sum(r.min_width for _, r in access_rooms)
        while access_rooms and total_min_width > edge_length:
            idx, demoted = access_rooms.pop()
            total_min_width -= demoted.min_width
            logger.warning(
                "Demoting %s from corridor attachment "
                "(edge=%.1fm, needed=%.1fm)",
                demoted.room_id, edge_length, total_min_width + demoted.min_width,
            )

        # 添加贴边硬约束
        for i, room in access_rooms:
            if primary_edge == "south":
                model.Add(y[i] == 0)
            elif primary_edge == "north":
                model.Add(y[i] + d[i] == int(self.D * SCALE))
            elif primary_edge == "west":
                model.Add(x[i] == 0)
            elif primary_edge == "east":
                model.Add(x[i] + w[i] == int(self.W * SCALE))

    # ----------------------------------------------------------
    # Stage 2a: CP-SAT 求解器
    # ----------------------------------------------------------

    def _solve_cpsat(
        self, warm_start: Optional[List[RoomResult]]
    ) -> List[RoomResult]:
        """Google OR-Tools CP-SAT 求解"""
        try:
            from ortools.sat.python import cp_model  # type: ignore[import-not-found]
        except ImportError:
            raise RuntimeError(
                "ortools not installed. Run: pip install ortools"
            )

        SCALE = 100  # 转为 cm 精度
        W_s = int(self.W * SCALE)
        D_s = int(self.D * SCALE)

        model = cp_model.CpModel()

        # === 决策变量 ===
        x = [model.NewIntVar(0, W_s, f"x_{i}") for i in range(self.n)]
        y = [model.NewIntVar(0, D_s, f"y_{i}") for i in range(self.n)]
        w = [
            model.NewIntVar(
                int(self.rooms[i].min_width * SCALE), W_s, f"w_{i}"
            )
            for i in range(self.n)
        ]
        d = [
            model.NewIntVar(
                int(self.rooms[i].min_depth * SCALE), D_s, f"d_{i}"
            )
            for i in range(self.n)
        ]

        # === end 变量 ===
        x_end = [model.NewIntVar(0, W_s, f"xe_{i}") for i in range(self.n)]
        y_end = [model.NewIntVar(0, D_s, f"ye_{i}") for i in range(self.n)]
        for i in range(self.n):
            model.Add(x_end[i] == x[i] + w[i])
            model.Add(y_end[i] == y[i] + d[i])

        # === 边界约束 ===
        for i in range(self.n):
            model.Add(x_end[i] <= W_s)
            model.Add(y_end[i] <= D_s)

        # === 不重叠约束 ===
        x_intervals = [
            model.NewIntervalVar(x[i], w[i], x_end[i], f"xi_{i}")
            for i in range(self.n)
        ]
        y_intervals = [
            model.NewIntervalVar(y[i], d[i], y_end[i], f"yi_{i}")
            for i in range(self.n)
        ]

        # 禁区：bounding box 内但岛屿外（L 型缺口等）
        from shapely.geometry import box as shapely_box
        bbox = shapely_box(self.x_min, self.y_min, self.x_max, self.y_max)
        forbidden = bbox.difference(self.island)
        if not forbidden.is_empty:
            parts = list(getattr(forbidden, 'geoms', [forbidden]))
            for fi, part in enumerate(parts):
                if not isinstance(part, Polygon) or part.is_empty or part.area < 0.01:
                    continue
                fminx, fminy, fmaxx, fmaxy = part.bounds
                fz_x_s = int((fminx - self.x_min) * SCALE)
                fz_y_s = int((fminy - self.y_min) * SCALE)
                fz_w_s = max(1, int((fmaxx - fminx) * SCALE))
                fz_d_s = max(1, int((fmaxy - fminy) * SCALE))
                fx = model.NewIntVar(fz_x_s, fz_x_s, f"bfz_x_{fi}")
                fy = model.NewIntVar(fz_y_s, fz_y_s, f"bfz_y_{fi}")
                fw = model.NewIntVar(fz_w_s, fz_w_s, f"bfz_w_{fi}")
                fd = model.NewIntVar(fz_d_s, fz_d_s, f"bfz_d_{fi}")
                fxe = model.NewIntVar(fz_x_s + fz_w_s, fz_x_s + fz_w_s, f"bfz_xe_{fi}")
                fye = model.NewIntVar(fz_y_s + fz_d_s, fz_y_s + fz_d_s, f"bfz_ye_{fi}")
                x_intervals.append(model.NewIntervalVar(fx, fw, fxe, f"bfzxi_{fi}"))
                y_intervals.append(model.NewIntervalVar(fy, fd, fye, f"bfzyi_{fi}"))

        model.AddNoOverlap2D(x_intervals, y_intervals)

        # === 面积约束 ===
        area_vars = []
        for i, room in enumerate(self.rooms):
            area_target_s = int(room.target_area * SCALE * SCALE)
            area_lb = int(area_target_s * (1 - room.area_tolerance))
            area_ub = int(area_target_s * (1 + room.area_tolerance))
            # 确保上界不超过总面积
            area_ub = min(area_ub, W_s * D_s)

            area_var = model.NewIntVar(area_lb, area_ub, f"area_{i}")
            model.AddMultiplicationEquality(area_var, w[i], d[i])
            area_vars.append(area_var)

        # === 目标函数：最小化面积偏差 ===
        area_deviations = []
        for i, room in enumerate(self.rooms):
            area_target_s = int(room.target_area * SCALE * SCALE)
            dev = model.NewIntVar(0, W_s * D_s, f"dev_{i}")
            diff = model.NewIntVar(-W_s * D_s, W_s * D_s, f"diff_{i}")
            model.Add(diff == area_vars[i] - area_target_s)
            model.AddAbsEquality(dev, diff)
            area_deviations.append(dev)

        model.Minimize(sum(area_deviations))

        # === 拓扑冻结：从 Treemap 提取正交相对位置 → 线性约束替代 Big-M ===
        if warm_start and len(warm_start) >= 2:
            relations = self._freeze_topology_from_treemap(warm_start)
            self._add_frozen_topology_constraints(model, x, y, w, d, relations, SCALE)

        # === Warm Start Hints ===
        if warm_start:
            for i, ws in enumerate(warm_start):
                if i >= self.n:
                    break
                model.AddHint(x[i], int((ws.x - self.x_min) * SCALE))
                model.AddHint(y[i], int((ws.y - self.y_min) * SCALE))
                model.AddHint(w[i], int(ws.width * SCALE))
                model.AddHint(d[i], int(ws.depth * SCALE))

        # === 求解 ===
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = 4

        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        logger.info("CP-SAT status: %s", status_name)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"CP-SAT solve failed, status: {status_name}")

        # === 提取结果 ===
        results = []
        for i, room in enumerate(self.rooms):
            results.append(
                RoomResult(
                    room_id=room.room_id,
                    x=solver.Value(x[i]) / SCALE + self.x_min,
                    y=solver.Value(y[i]) / SCALE + self.y_min,
                    width=solver.Value(w[i]) / SCALE,
                    depth=solver.Value(d[i]) / SCALE,
                )
            )
        return results

    # ----------------------------------------------------------
    # Stage 2b: Gurobi 求解器（可选）
    # ----------------------------------------------------------

    def _solve_gurobi(
        self, warm_start: Optional[List[RoomResult]]
    ) -> List[RoomResult]:
        """Gurobi MIQP 求解，未安装时回退 CP-SAT"""
        try:
            import gurobipy as gp  # type: ignore[import-not-found]
            from gurobipy import GRB  # type: ignore[import-not-found]
        except ImportError:
            logger.info("Gurobi not installed, falling back to CP-SAT")
            return self._solve_cpsat(warm_start)

        M = self.W + self.D  # Big-M

        mdl = gp.Model("island_partition")
        mdl.Params.TimeLimit = self.time_limit
        mdl.Params.OutputFlag = 0

        x = mdl.addVars(self.n, lb=0, ub=self.W, name="x")
        y = mdl.addVars(self.n, lb=0, ub=self.D, name="y")
        w = mdl.addVars(self.n, name="w")
        d = mdl.addVars(self.n, name="d")

        for i, room in enumerate(self.rooms):
            w[i].lb = room.min_width
            w[i].ub = self.W
            d[i].lb = room.min_depth
            d[i].ub = self.D

        # 边界
        for i in range(self.n):
            mdl.addConstr(x[i] + w[i] <= self.W)
            mdl.addConstr(y[i] + d[i] <= self.D)

        # 不重叠 (Big-M)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                s_L = mdl.addVar(vtype=GRB.BINARY)
                s_R = mdl.addVar(vtype=GRB.BINARY)
                s_F = mdl.addVar(vtype=GRB.BINARY)
                s_B = mdl.addVar(vtype=GRB.BINARY)
                mdl.addConstr(x[i] + w[i] <= x[j] + M * (1 - s_L))
                mdl.addConstr(x[j] + w[j] <= x[i] + M * (1 - s_R))
                mdl.addConstr(y[i] + d[i] <= y[j] + M * (1 - s_F))
                mdl.addConstr(y[j] + d[j] <= y[i] + M * (1 - s_B))
                mdl.addConstr(s_L + s_R + s_F + s_B >= 1)

        # 面积
        area = mdl.addVars(self.n, name="area")
        for i, room in enumerate(self.rooms):
            mdl.addConstr(area[i] == w[i] * d[i])
            mdl.addConstr(area[i] >= room.target_area * (1 - room.area_tolerance))
            mdl.addConstr(area[i] <= room.target_area * (1 + room.area_tolerance))

        obj = gp.quicksum(
            (area[i] - self.rooms[i].target_area) ** 2 for i in range(self.n)
        )
        mdl.setObjective(obj, GRB.MINIMIZE)

        if warm_start:
            for i, ws in enumerate(warm_start):
                if i >= self.n:
                    break
                x[i].Start = ws.x - self.x_min
                y[i].Start = ws.y - self.y_min
                w[i].Start = ws.width
                d[i].Start = ws.depth

        mdl.optimize()

        if mdl.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi solve failed, status: {mdl.Status}")

        results = []
        for i, room in enumerate(self.rooms):
            results.append(
                RoomResult(
                    room_id=room.room_id,
                    x=x[i].X + self.x_min,
                    y=y[i].X + self.y_min,
                    width=w[i].X,
                    depth=d[i].X,
                )
            )
        return results

    # ----------------------------------------------------------
    # Stage 3: 边界裁剪
    # ----------------------------------------------------------

    def _clip_to_boundary(self, results: List[RoomResult]) -> List[RoomResult]:
        """将房间裁剪到实际岛屿边界"""
        clipped = []
        for room in results:
            if self.island.contains(room.polygon):
                clipped.append(room)
                continue

            clipped_geom = room.polygon.intersection(self.island)

            if clipped_geom.is_empty or clipped_geom.area < 0.5:
                logger.warning(
                    "Room %s clipped to empty, removing", room.room_id
                )
                continue

            minx, miny, maxx, maxy = clipped_geom.bounds
            bounds_area = (maxx - minx) * (maxy - miny)
            if bounds_area > 0 and abs(clipped_geom.area - bounds_area) / bounds_area < 0.01:
                # 裁剪后仍是矩形
                clipped.append(RoomResult(
                    room_id=room.room_id, x=minx, y=miny,
                    width=maxx - minx, depth=maxy - miny,
                ))
            else:
                # 非矩形裁剪 — 缩小到实际面积
                w_clip = maxx - minx
                h_clip = maxy - miny
                if w_clip > 0 and h_clip > 0:
                    ratio = clipped_geom.area / bounds_area
                    if w_clip >= h_clip:
                        w_clip *= ratio
                    else:
                        h_clip *= ratio
                    cx = clipped_geom.centroid.x
                    cy = clipped_geom.centroid.y
                    clipped.append(RoomResult(
                        room_id=room.room_id,
                        x=max(minx, cx - w_clip / 2),
                        y=max(miny, cy - h_clip / 2),
                        width=w_clip, depth=h_clip,
                    ))
        return clipped


# ============================================================
# 便捷函数
# ============================================================


def partition_island(
    island_polygon: Polygon,
    room_specs: List[RoomSpec],
    solver: str = "cpsat",
    time_limit: float = 10.0,
    use_warm_start: bool = True,
) -> List[RoomResult]:
    """
    岛屿房间划分便捷函数

    Args:
        island_polygon: 岛屿边界
        room_specs: 房间规格列表
        solver: 'cpsat' | 'gurobi'
        time_limit: 求解时间上限（秒）
        use_warm_start: 是否用 Treemap 热启动

    Returns:
        房间划分结果列表
    """
    solver_inst = IslandPartitionSolver(
        island_polygon=island_polygon,
        rooms=room_specs,
        solver=solver,
        time_limit=time_limit,
        use_warm_start=use_warm_start,
    )
    return solver_inst.solve()


# ============================================================
# 语义感知求解器
# ============================================================


class SemanticIslandPartitionSolver:
    """
    语义感知的岛屿房间划分求解器。

    相比 IslandPartitionSolver，新增：
    1. 分层 Treemap warm start（按邻接聚类 + 功能分区排序）
    2. 宽高比硬约束
    3. 采光约束（需要窗的房间靠外墙）
    4. 邻接必须/禁止硬约束
    5. 多目标优化（面积 + 邻接距离 + 分区紧凑度 + 宽高比惩罚）

    回退机制：语义求解失败时降级到旧 IslandPartitionSolver。
    """

    def __init__(
        self,
        island_polygon: Polygon,
        rooms: List[SemanticRoomSpec],
        adjacency_graph: Dict[str, List[str]],
        island_context: IslandContext,
        config: Optional[SolverConfig] = None,
    ):
        self.island = island_polygon
        self.rooms = rooms
        self.adjacency = adjacency_graph
        self.context = island_context
        self.config = config or SolverConfig()

        # 走廊边信息
        self.corridor_edges = island_context.corridor_edges if hasattr(island_context, 'corridor_edges') else []

        # 边界
        x_min, y_min, x_max, y_max = island_polygon.bounds
        self.x_min, self.y_min = x_min, y_min
        self.W = x_max - x_min
        self.D = y_max - y_min

        # 索引
        self.room_index: Dict[str, int] = {r.room_id: i for i, r in enumerate(rooms)}
        self.n = len(rooms)

        # 按 zone 分组
        self.zones = self._group_by_zone()

        # 修正不合理的 min_width/min_depth
        for r in self.rooms:
            if r.min_width > self.W * 0.9:
                r.min_width = self.W * 0.9
            if r.min_depth > self.D * 0.9:
                r.min_depth = self.D * 0.9

    def _group_by_zone(self) -> Dict[ZoneType, List[str]]:
        zones: Dict[ZoneType, List[str]] = {}
        for room in self.rooms:
            if room.zone not in zones:
                zones[room.zone] = []
            zones[room.zone].append(room.room_id)
        return zones

    def _get_forbidden_zones(self) -> List[Tuple[float, float, float, float]]:
        """
        计算禁区：bounding box 内但岛屿 polygon 外的区域。
        将每个连通分量取 bounding box 作为禁止矩形。
        返回 [(x, y, w, d), ...] 世界坐标。
        """
        from shapely.geometry import box as shapely_box
        bbox = shapely_box(self.x_min, self.y_min,
                           self.x_min + self.W, self.y_min + self.D)
        forbidden = bbox.difference(self.island)

        if forbidden.is_empty:
            return []

        parts = []
        if hasattr(forbidden, 'geoms'):
            for g in getattr(forbidden, 'geoms', []):
                if isinstance(g, Polygon) and not g.is_empty and g.area > 0.01:
                    parts.append(g)
        elif isinstance(forbidden, Polygon) and forbidden.area > 0.01:
            parts.append(forbidden)

        zones = []
        for part in parts:
            minx, miny, maxx, maxy = part.bounds
            zones.append((minx, miny, maxx - minx, maxy - miny))
        return zones

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def solve(self) -> List[RoomResult]:
        """
        求解入口：分层 Treemap → CP-SAT MIQP → 边界裁剪。
        语义求解失败时降级到旧 solver。
        """
        # Stage 1: 分层 Treemap warm start
        t0 = time.perf_counter()
        warm_start = self._generate_warm_start()
        logger.info(
            "Hierarchical treemap warm start: %.1fms, %d rooms",
            (time.perf_counter() - t0) * 1000, len(warm_start),
        )

        # 快捷路径：≤2 个房间直接用 treemap，跳过 MIQP
        if self.n <= 2:
            logger.info("≤2 rooms, using treemap-only (skip semantic MIQP)")
            # 将 WarmStartRect 转为 RoomResult
            results = []
            for ws in warm_start:
                results.append(RoomResult(
                    room_id=ws.room_id,
                    x=ws.x, y=ws.y,
                    width=ws.width, depth=ws.depth,
                ))
            results = self._clip_to_boundary(results)
            return results

        # Stage 2: 语义 CP-SAT
        try:
            t1 = time.perf_counter()
            results = self._solve_cpsat(warm_start)
            logger.info(
                "Semantic MIQP solve: %.1fms",
                (time.perf_counter() - t1) * 1000,
            )
        except Exception as e:
            logger.warning(
                "Semantic MIQP failed (%s), falling back to basic solver", e
            )
            results = self._fallback_solve()

        # Stage 3: 边界裁剪
        results = self._clip_to_boundary(results)
        return results

    def _generate_warm_start(self) -> List[WarmStartRect]:
        bounds = (self.x_min, self.y_min,
                  self.x_min + self.W, self.y_min + self.D)
        return hierarchical_treemap(bounds, self.rooms, self.adjacency)

    def _fallback_solve(self) -> List[RoomResult]:
        """降级到旧 solver（转换 RoomSpec）"""
        old_specs = []
        for r in self.rooms:
            old_specs.append(RoomSpec(
                room_id=r.room_id,
                room_type=r.room_type,
                target_area=r.target_area,
                area_tolerance=self.config.area_tolerance,
                min_width=r.min_width,
                min_depth=r.min_depth,
                adjacency_required=r.adjacency_required,
                adjacency_forbidden=r.adjacency_forbidden,
                window_access=r.needs_window,
            ))
        fallback = IslandPartitionSolver(
            island_polygon=self.island,
            rooms=old_specs,
            time_limit=self.config.time_limit,
        )
        return fallback.solve()

    # ----------------------------------------------------------
    # Stage 2: 语义 CP-SAT 求解
    # ----------------------------------------------------------

    def _solve_cpsat(self, warm_start: List[WarmStartRect]) -> List[RoomResult]:
        try:
            from ortools.sat.python import cp_model  # type: ignore[import-not-found]
        except ImportError:
            raise RuntimeError("ortools not installed. Run: pip install ortools")

        SCALE = 100  # cm 精度
        W_s = int(self.W * SCALE)
        D_s = int(self.D * SCALE)

        model = cp_model.CpModel()

        # ═══════════════════════════════════════════════════════════
        # 变量定义
        # ═══════════════════════════════════════════════════════════

        x = [model.NewIntVar(0, W_s, f"x_{i}") for i in range(self.n)]
        y = [model.NewIntVar(0, D_s, f"y_{i}") for i in range(self.n)]
        w = [
            model.NewIntVar(
                max(1, int(self.rooms[i].min_width * SCALE)), W_s, f"w_{i}"
            )
            for i in range(self.n)
        ]
        d = [
            model.NewIntVar(
                max(1, int(self.rooms[i].min_depth * SCALE)), D_s, f"d_{i}"
            )
            for i in range(self.n)
        ]

        # 辅助变量：右边界 / 上边界
        x_end = [model.NewIntVar(0, W_s, f"xe_{i}") for i in range(self.n)]
        y_end = [model.NewIntVar(0, D_s, f"ye_{i}") for i in range(self.n)]
        for i in range(self.n):
            model.Add(x_end[i] == x[i] + w[i])
            model.Add(y_end[i] == y[i] + d[i])

        # 面积变量
        area = [model.NewIntVar(0, W_s * D_s, f"area_{i}") for i in range(self.n)]
        for i in range(self.n):
            model.AddMultiplicationEquality(area[i], w[i], d[i])

        # ═══════════════════════════════════════════════════════════
        # Warm Start Hints
        # ═══════════════════════════════════════════════════════════

        ws_dict = {ws.room_id: ws for ws in warm_start}
        for i, room in enumerate(self.rooms):
            if room.room_id in ws_dict:
                ws = ws_dict[room.room_id]
                model.AddHint(x[i], int((ws.x - self.x_min) * SCALE))
                model.AddHint(y[i], int((ws.y - self.y_min) * SCALE))
                model.AddHint(w[i], int(ws.width * SCALE))
                model.AddHint(d[i], int(ws.depth * SCALE))

        # ═══════════════════════════════════════════════════════════
        # 硬约束
        # ═══════════════════════════════════════════════════════════

        # 约束 1: 边界
        for i in range(self.n):
            model.Add(x_end[i] <= W_s)
            model.Add(y_end[i] <= D_s)

        # 约束 2: 不重叠 (NoOverlap2D) + 禁区
        x_intervals = [
            model.NewIntervalVar(x[i], w[i], x_end[i], f"xi_{i}")
            for i in range(self.n)
        ]
        y_intervals = [
            model.NewIntervalVar(y[i], d[i], y_end[i], f"yi_{i}")
            for i in range(self.n)
        ]

        # 禁区：bounding box 内但岛屿外的区域，作为不可移动的"房间"
        forbidden_zones = self._get_forbidden_zones()
        for fi, (fz_x, fz_y, fz_w, fz_d) in enumerate(forbidden_zones):
            fz_x_s = int((fz_x - self.x_min) * SCALE)
            fz_y_s = int((fz_y - self.y_min) * SCALE)
            fz_w_s = max(1, int(fz_w * SCALE))
            fz_d_s = max(1, int(fz_d * SCALE))
            # 用固定域 IntVar 模拟常量
            fx = model.NewIntVar(fz_x_s, fz_x_s, f"fz_x_{fi}")
            fy = model.NewIntVar(fz_y_s, fz_y_s, f"fz_y_{fi}")
            fw = model.NewIntVar(fz_w_s, fz_w_s, f"fz_w_{fi}")
            fd = model.NewIntVar(fz_d_s, fz_d_s, f"fz_d_{fi}")
            fxe = model.NewIntVar(fz_x_s + fz_w_s, fz_x_s + fz_w_s, f"fz_xe_{fi}")
            fye = model.NewIntVar(fz_y_s + fz_d_s, fz_y_s + fz_d_s, f"fz_ye_{fi}")
            x_intervals.append(model.NewIntervalVar(fx, fw, fxe, f"fzxi_{fi}"))
            y_intervals.append(model.NewIntervalVar(fy, fd, fye, f"fzyi_{fi}"))
            logger.info("Forbidden zone %d: (%.1f,%.1f) %.1fx%.1f",
                        fi, fz_x, fz_y, fz_w, fz_d)

        model.AddNoOverlap2D(x_intervals, y_intervals)

        # 约束 3: 面积容差
        tol = self.config.area_tolerance
        for i, room in enumerate(self.rooms):
            target_s = int(room.target_area * SCALE * SCALE)
            model.Add(area[i] >= int(target_s * (1 - tol)))
            model.Add(area[i] <= int(target_s * (1 + tol)))

        # 约束 3b: 总面积覆盖率
        total_island_area_s = int(self.island.area * SCALE * SCALE)
        total_target_s = sum(int(r.target_area * SCALE * SCALE) for r in self.rooms)
        min_required = min(total_island_area_s, total_target_s)
        is_rectangular = len(forbidden_zones) == 0
        if is_rectangular:
            # 矩形岛屿：硬约束 95%，杜绝条带空隙
            model.Add(sum(area) >= int(min_required * 0.95))
        # 非矩形岛屿：覆盖率作为软目标，在下方目标函数中加入

        # 约束 4: 宽高比
        for i, room in enumerate(self.rooms):
            ar_min, ar_max = room.aspect_ratio_range
            # ar_min <= w/d <= ar_max  =>  ar_min * d <= w <= ar_max * d
            ar_min_100 = int(ar_min * 100)
            ar_max_100 = int(ar_max * 100)
            model.Add(w[i] * 100 >= ar_min_100 * d[i])
            model.Add(w[i] * 100 <= ar_max_100 * d[i])

        # 约束 5: 采光
        for i, room in enumerate(self.rooms):
            if room.needs_window:
                self._add_window_constraint(model, i, x, y, x_end, y_end, W_s, D_s)

        # 约束 6: 邻接必须（共享边）
        # 非矩形岛屿时降级为软约束（在目标函数中由邻接距离项处理）
        adj_req_processed: set = set()
        if is_rectangular:
            for room in self.rooms:
                i = self.room_index[room.room_id]
                for adj_id in room.adjacency_required:
                    if adj_id not in self.room_index:
                        continue
                    j = self.room_index[adj_id]
                    pair = tuple(sorted([i, j]))
                    if pair in adj_req_processed:
                        continue
                    adj_req_processed.add(pair)
                    self._add_adjacency_required_constraint(
                        model, i, j, x, y, w, d, x_end, y_end, W_s, D_s, SCALE,
                    )
        else:
            logger.info(
                "Non-rectangular island: adjacency_required relaxed to soft objective"
            )

        # 约束 7: 禁止邻接
        adj_forb_processed: set = set()
        for room in self.rooms:
            i = self.room_index[room.room_id]
            for forbidden_id in room.adjacency_forbidden:
                if forbidden_id not in self.room_index:
                    continue
                j = self.room_index[forbidden_id]
                pair = tuple(sorted([i, j]))
                if pair in adj_forb_processed:
                    continue
                adj_forb_processed.add(pair)
                self._add_non_adjacency_constraint(
                    model, i, j, x_end, y_end, x, y, SCALE,
                )

        # ═══════════════════════════════════════════════════════════
        # 走廊可达性约束
        # ═══════════════════════════════════════════════════════════
        if self.corridor_edges:
            self._add_corridor_accessibility_constraints(
                model, x, y, x_end, y_end, W_s, D_s,
            )

        # ═══════════════════════════════════════════════════════════
        # 目标函数
        # ═══════════════════════════════════════════════════════════

        objectives = []

        # 目标 1: 面积偏差
        for i, room in enumerate(self.rooms):
            target_s = int(room.target_area * SCALE * SCALE)
            dev = model.NewIntVar(0, W_s * D_s, f"area_dev_{i}")
            diff = model.NewIntVar(-W_s * D_s, W_s * D_s, f"area_diff_{i}")
            model.Add(diff == area[i] - target_s)
            model.AddAbsEquality(dev, diff)
            weight = int(self.config.weight_area * room.area_priority)
            objectives.append(dev * weight)

        # 目标 2: 邻接距离
        adjacency_terms = self._build_adjacency_objective(
            model, x, y, w, d, W_s, D_s,
        )
        for term in adjacency_terms:
            objectives.append(term * self.config.weight_adjacency)

        # 目标 3: 分区紧凑度
        compactness_terms = self._build_compactness_objective(
            model, x, y, x_end, y_end, W_s, D_s,
        )
        for term in compactness_terms:
            objectives.append(term * self.config.weight_compactness)

        # 目标 4: 宽高比惩罚
        for i, room in enumerate(self.rooms):
            if room.room_type in ("corridor", "hallway", "passage"):
                continue
            ar_pen = model.NewIntVar(0, max(W_s, D_s), f"ar_pen_{i}")
            ar_diff = model.NewIntVar(-max(W_s, D_s), max(W_s, D_s), f"ar_diff_{i}")
            model.Add(ar_diff == w[i] - d[i])
            model.AddAbsEquality(ar_pen, ar_diff)
            objectives.append(ar_pen * self.config.weight_aspect_ratio)

        # 目标 5: 覆盖率（非矩形岛屿时作为软目标，最大化总面积）
        if not is_rectangular:
            coverage_gap = model.NewIntVar(0, W_s * D_s, "coverage_gap")
            coverage_diff = model.NewIntVar(-W_s * D_s, W_s * D_s, "cov_diff")
            model.Add(coverage_diff == total_island_area_s - sum(area))
            model.AddMaxEquality(coverage_gap, [coverage_diff, model.NewConstant(0)])
            objectives.append(coverage_gap * self.config.weight_area)

        # 总目标
        model.Minimize(sum(objectives))

        # ═══════════════════════════════════════════════════════════
        # 求解
        # ═══════════════════════════════════════════════════════════

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.time_limit
        solver.parameters.num_search_workers = self.config.num_workers

        status = solver.Solve(model)

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise RuntimeError(
                f"Semantic MIQP solve failed, status: {status}. "
                f"Try relaxing constraints or increasing time_limit."
            )

        # 提取结果
        results = []
        for i, room in enumerate(self.rooms):
            results.append(RoomResult(
                room_id=room.room_id,
                x=solver.Value(x[i]) / SCALE + self.x_min,
                y=solver.Value(y[i]) / SCALE + self.y_min,
                width=solver.Value(w[i]) / SCALE,
                depth=solver.Value(d[i]) / SCALE,
            ))

        return results

    # ----------------------------------------------------------
    # 约束辅助方法
    # ----------------------------------------------------------

    def _add_window_constraint(
        self, model, i: int,
        x, y, x_end, y_end,
        W_s: int, D_s: int,
    ):
        """采光约束：needs_window 的房间至少一边靠外墙"""
        touches = []

        if "west" in self.context.exterior_walls:
            t = model.NewBoolVar(f"touch_west_{i}")
            model.Add(x[i] == 0).OnlyEnforceIf(t)
            model.Add(x[i] > 0).OnlyEnforceIf(t.Not())
            touches.append(t)

        if "east" in self.context.exterior_walls:
            t = model.NewBoolVar(f"touch_east_{i}")
            model.Add(x_end[i] == W_s).OnlyEnforceIf(t)
            model.Add(x_end[i] < W_s).OnlyEnforceIf(t.Not())
            touches.append(t)

        if "south" in self.context.exterior_walls:
            t = model.NewBoolVar(f"touch_south_{i}")
            model.Add(y[i] == 0).OnlyEnforceIf(t)
            model.Add(y[i] > 0).OnlyEnforceIf(t.Not())
            touches.append(t)

        if "north" in self.context.exterior_walls:
            t = model.NewBoolVar(f"touch_north_{i}")
            model.Add(y_end[i] == D_s).OnlyEnforceIf(t)
            model.Add(y_end[i] < D_s).OnlyEnforceIf(t.Not())
            touches.append(t)

        if touches:
            model.Add(sum(touches) >= 1)

    def _add_adjacency_required_constraint(
        self, model, i: int, j: int,
        x, y, w, d, x_end, y_end,
        W_s: int, D_s: int, SCALE: int,
    ):
        """
        邻接必须约束：两个房间必须共享一段边（>= min_door_width）。

        4 个方向选 1：i 的右边 = j 的左边（或反之），
        且在正交方向有 >= MIN_DOOR_S 的重叠段。
        """
        MIN_DOOR_S = max(1, int(self.config.min_door_width * SCALE))

        touch_L = model.NewBoolVar(f"touch_{i}_{j}_L")
        touch_R = model.NewBoolVar(f"touch_{i}_{j}_R")
        touch_F = model.NewBoolVar(f"touch_{i}_{j}_F")
        touch_B = model.NewBoolVar(f"touch_{i}_{j}_B")

        # --- 方向 L: i 的右边 = j 的左边, y 方向重叠 ---
        model.Add(x_end[i] == x[j]).OnlyEnforceIf(touch_L)
        # y overlap: min(y_end[i], y_end[j]) - max(y[i], y[j]) >= MIN_DOOR_S
        ov_y_min_L = model.NewIntVar(0, D_s, f"ov_ymin_L_{i}_{j}")
        ov_y_max_L = model.NewIntVar(0, D_s, f"ov_ymax_L_{i}_{j}")
        model.AddMinEquality(ov_y_min_L, [y_end[i], y_end[j]])
        model.AddMaxEquality(ov_y_max_L, [y[i], y[j]])
        model.Add(ov_y_min_L - ov_y_max_L >= MIN_DOOR_S).OnlyEnforceIf(touch_L)

        # --- 方向 R: j 的右边 = i 的左边 ---
        model.Add(x_end[j] == x[i]).OnlyEnforceIf(touch_R)
        ov_y_min_R = model.NewIntVar(0, D_s, f"ov_ymin_R_{i}_{j}")
        ov_y_max_R = model.NewIntVar(0, D_s, f"ov_ymax_R_{i}_{j}")
        model.AddMinEquality(ov_y_min_R, [y_end[i], y_end[j]])
        model.AddMaxEquality(ov_y_max_R, [y[i], y[j]])
        model.Add(ov_y_min_R - ov_y_max_R >= MIN_DOOR_S).OnlyEnforceIf(touch_R)

        # --- 方向 F: i 的上边 = j 的下边 ---
        model.Add(y_end[i] == y[j]).OnlyEnforceIf(touch_F)
        ov_x_min_F = model.NewIntVar(0, W_s, f"ov_xmin_F_{i}_{j}")
        ov_x_max_F = model.NewIntVar(0, W_s, f"ov_xmax_F_{i}_{j}")
        model.AddMinEquality(ov_x_min_F, [x_end[i], x_end[j]])
        model.AddMaxEquality(ov_x_max_F, [x[i], x[j]])
        model.Add(ov_x_min_F - ov_x_max_F >= MIN_DOOR_S).OnlyEnforceIf(touch_F)

        # --- 方向 B: j 的上边 = i 的下边 ---
        model.Add(y_end[j] == y[i]).OnlyEnforceIf(touch_B)
        ov_x_min_B = model.NewIntVar(0, W_s, f"ov_xmin_B_{i}_{j}")
        ov_x_max_B = model.NewIntVar(0, W_s, f"ov_xmax_B_{i}_{j}")
        model.AddMinEquality(ov_x_min_B, [x_end[i], x_end[j]])
        model.AddMaxEquality(ov_x_max_B, [x[i], x[j]])
        model.Add(ov_x_min_B - ov_x_max_B >= MIN_DOOR_S).OnlyEnforceIf(touch_B)

        # 至少一个方向成立
        model.Add(touch_L + touch_R + touch_F + touch_B >= 1)

    def _add_non_adjacency_constraint(
        self, model, i: int, j: int,
        x_end, y_end, x, y, SCALE: int,
    ):
        """禁止邻接：两个房间必须有间隙（不能共享边）"""
        MIN_GAP_S = max(1, int(self.config.non_adjacency_min_gap * SCALE))

        sep_L = model.NewBoolVar(f"sep_{i}_{j}_L")
        sep_R = model.NewBoolVar(f"sep_{i}_{j}_R")
        sep_F = model.NewBoolVar(f"sep_{i}_{j}_F")
        sep_B = model.NewBoolVar(f"sep_{i}_{j}_B")

        model.Add(x_end[i] + MIN_GAP_S <= x[j]).OnlyEnforceIf(sep_L)
        model.Add(x_end[j] + MIN_GAP_S <= x[i]).OnlyEnforceIf(sep_R)
        model.Add(y_end[i] + MIN_GAP_S <= y[j]).OnlyEnforceIf(sep_F)
        model.Add(y_end[j] + MIN_GAP_S <= y[i]).OnlyEnforceIf(sep_B)

        model.Add(sep_L + sep_R + sep_F + sep_B >= 1)

    def _add_corridor_accessibility_constraints(
        self, model, x, y, x_end, y_end, W_s: int, D_s: int,
    ):
        """
        走廊可达性约束：每个房间至少有一条边接触走廊侧

        如果岛屿无走廊边（内岛），跳过约束。
        """
        if not self.corridor_edges:
            logger.warning(
                "Island has no corridor edges, skipping accessibility constraint. "
                "Rooms may not be directly accessible."
            )
            return

        TOUCH_THRESHOLD = 10  # 10cm = 可开门宽度（SCALE=100 时 10 个单位）

        for i, room in enumerate(self.rooms):
            touches_any_corridor = []

            if "south" in self.corridor_edges:
                touch_south = model.NewBoolVar(f"touch_south_{i}")
                model.Add(y[i] <= TOUCH_THRESHOLD).OnlyEnforceIf(touch_south)
                model.Add(y[i] > TOUCH_THRESHOLD).OnlyEnforceIf(touch_south.Not())
                touches_any_corridor.append(touch_south)

            if "north" in self.corridor_edges:
                touch_north = model.NewBoolVar(f"touch_north_{i}")
                model.Add(y_end[i] >= D_s - TOUCH_THRESHOLD).OnlyEnforceIf(touch_north)
                model.Add(y_end[i] < D_s - TOUCH_THRESHOLD).OnlyEnforceIf(touch_north.Not())
                touches_any_corridor.append(touch_north)

            if "west" in self.corridor_edges:
                touch_west = model.NewBoolVar(f"touch_west_{i}")
                model.Add(x[i] <= TOUCH_THRESHOLD).OnlyEnforceIf(touch_west)
                model.Add(x[i] > TOUCH_THRESHOLD).OnlyEnforceIf(touch_west.Not())
                touches_any_corridor.append(touch_west)

            if "east" in self.corridor_edges:
                touch_east = model.NewBoolVar(f"touch_east_{i}")
                model.Add(x_end[i] >= W_s - TOUCH_THRESHOLD).OnlyEnforceIf(touch_east)
                model.Add(x_end[i] < W_s - TOUCH_THRESHOLD).OnlyEnforceIf(touch_east.Not())
                touches_any_corridor.append(touch_east)

            # 至少接触一个走廊边
            if touches_any_corridor:
                model.Add(sum(touches_any_corridor) >= 1)

    # ----------------------------------------------------------
    # 目标函数辅助方法
    # ----------------------------------------------------------

    def _build_adjacency_objective(
        self, model, x, y, w, d, W_s: int, D_s: int,
    ) -> list:
        """邻接距离目标：最小化必须/偏好相邻房间的质心距离"""
        terms = []
        processed: set = set()

        for room in self.rooms:
            i = self.room_index[room.room_id]

            # adjacency_required（全权重）
            for adj_id in room.adjacency_required:
                if adj_id not in self.room_index:
                    continue
                j = self.room_index[adj_id]
                pair = tuple(sorted([i, j]))
                if pair in processed:
                    continue
                processed.add(pair)
                dist = self._centroid_manhattan_distance(
                    model, i, j, x, y, w, d, W_s, D_s, f"adj_req_{i}_{j}",
                )
                terms.append(dist)

            # adjacency_preferred（半权重）
            for adj_id in room.adjacency_preferred:
                if adj_id not in self.room_index:
                    continue
                j = self.room_index[adj_id]
                pair = tuple(sorted([i, j]))
                if pair in processed:
                    continue
                processed.add(pair)
                dist = self._centroid_manhattan_distance(
                    model, i, j, x, y, w, d, W_s, D_s, f"adj_pref_{i}_{j}",
                )
                # 半权重：整除 2
                half = model.NewIntVar(0, 2 * (W_s + D_s), f"adj_pref_half_{i}_{j}")
                model.AddDivisionEquality(half, dist, 2)
                terms.append(half)

        return terms

    def _centroid_manhattan_distance(
        self, model, i: int, j: int,
        x, y, w, d,
        W_s: int, D_s: int,
        prefix: str,
    ):
        """两个房间质心的曼哈顿距离（使用 2 倍值避免除法）"""
        # centroid * 2
        cx_i_2 = model.NewIntVar(0, 2 * W_s, f"{prefix}_cx2_i")
        cy_i_2 = model.NewIntVar(0, 2 * D_s, f"{prefix}_cy2_i")
        cx_j_2 = model.NewIntVar(0, 2 * W_s, f"{prefix}_cx2_j")
        cy_j_2 = model.NewIntVar(0, 2 * D_s, f"{prefix}_cy2_j")

        model.Add(cx_i_2 == 2 * x[i] + w[i])
        model.Add(cy_i_2 == 2 * y[i] + d[i])
        model.Add(cx_j_2 == 2 * x[j] + w[j])
        model.Add(cy_j_2 == 2 * y[j] + d[j])

        # |dx| + |dy|
        dx_raw = model.NewIntVar(-2 * W_s, 2 * W_s, f"{prefix}_dx_raw")
        dy_raw = model.NewIntVar(-2 * D_s, 2 * D_s, f"{prefix}_dy_raw")
        dx = model.NewIntVar(0, 2 * W_s, f"{prefix}_dx")
        dy = model.NewIntVar(0, 2 * D_s, f"{prefix}_dy")

        model.Add(dx_raw == cx_i_2 - cx_j_2)
        model.Add(dy_raw == cy_i_2 - cy_j_2)
        model.AddAbsEquality(dx, dx_raw)
        model.AddAbsEquality(dy, dy_raw)

        dist = model.NewIntVar(0, 2 * (W_s + D_s), f"{prefix}_dist")
        model.Add(dist == dx + dy)

        return dist

    def _build_compactness_objective(
        self, model, x, y, x_end, y_end, W_s: int, D_s: int,
    ) -> list:
        """分区紧凑度：最小化每个功能分区的包围盒周长"""
        terms = []

        for zone, room_ids in self.zones.items():
            if len(room_ids) < 2:
                continue

            indices = [self.room_index[rid] for rid in room_ids
                       if rid in self.room_index]
            if len(indices) < 2:
                continue

            zone_name = zone.value

            # 包围盒边界（用 AddMinEquality/AddMaxEquality 正确计算）
            z_x_min = model.NewIntVar(0, W_s, f"zone_{zone_name}_xmin")
            z_x_max = model.NewIntVar(0, W_s, f"zone_{zone_name}_xmax")
            z_y_min = model.NewIntVar(0, D_s, f"zone_{zone_name}_ymin")
            z_y_max = model.NewIntVar(0, D_s, f"zone_{zone_name}_ymax")

            model.AddMinEquality(z_x_min, [x[idx] for idx in indices])
            model.AddMaxEquality(z_x_max, [x_end[idx] for idx in indices])
            model.AddMinEquality(z_y_min, [y[idx] for idx in indices])
            model.AddMaxEquality(z_y_max, [y_end[idx] for idx in indices])

            # 包围盒宽高
            z_w = model.NewIntVar(0, W_s, f"zone_{zone_name}_w")
            z_h = model.NewIntVar(0, D_s, f"zone_{zone_name}_h")
            model.Add(z_w == z_x_max - z_x_min)
            model.Add(z_h == z_y_max - z_y_min)

            # 周长
            z_perim = model.NewIntVar(0, 2 * (W_s + D_s), f"zone_{zone_name}_perim")
            model.Add(z_perim == 2 * (z_w + z_h))

            terms.append(z_perim)

        return terms

    # ----------------------------------------------------------
    # Stage 3: 边界裁剪（复用旧 solver 逻辑）
    # ----------------------------------------------------------

    def _clip_to_boundary(self, results: List[RoomResult]) -> List[RoomResult]:
        clipped = []
        for room in results:
            # 如果房间完全在岛屿内部，保留原样
            if self.island.contains(room.polygon):
                clipped.append(room)
                continue

            clipped_geom = room.polygon.intersection(self.island)
            if clipped_geom.is_empty or clipped_geom.area < 0.5:
                logger.warning("Room %s clipped to empty, removing", room.room_id)
                continue

            # 裁剪后如果仍是矩形（面积误差 < 1%），用精确 bounds
            minx, miny, maxx, maxy = clipped_geom.bounds
            bounds_area = (maxx - minx) * (maxy - miny)
            if bounds_area > 0 and abs(clipped_geom.area - bounds_area) / bounds_area < 0.01:
                clipped.append(RoomResult(
                    room_id=room.room_id,
                    x=minx, y=miny,
                    width=maxx - minx, depth=maxy - miny,
                ))
            else:
                # 非矩形裁剪结果 — 取岛屿内的最大内接矩形近似
                # 保守做法：收缩到裁剪区域的实际面积对应的矩形
                w = maxx - minx
                h = maxy - miny
                if w > 0 and h > 0:
                    ratio = clipped_geom.area / bounds_area
                    # 按比例缩小较长边
                    if w >= h:
                        w *= ratio
                    else:
                        h *= ratio
                    cx, cy = clipped_geom.centroid.x, clipped_geom.centroid.y
                    new_x = max(minx, cx - w / 2)
                    new_y = max(miny, cy - h / 2)
                    clipped.append(RoomResult(
                        room_id=room.room_id,
                        x=new_x, y=new_y,
                        width=w, depth=h,
                    ))
                else:
                    clipped.append(RoomResult(
                        room_id=room.room_id,
                        x=minx, y=miny,
                        width=max(w, 0.1), depth=max(h, 0.1),
                    ))
        return clipped


# ============================================================
# 语义求解便捷函数
# ============================================================


def partition_island_semantic(
    island_polygon: Polygon,
    rooms: List[SemanticRoomSpec],
    adjacency_graph: Dict[str, List[str]],
    exterior_walls: List[str],
    config: Optional[SolverConfig] = None,
    corridor_edges: Optional[List[str]] = None,
) -> List[RoomResult]:
    """
    语义感知的岛屿房间划分便捷函数。

    Args:
        island_polygon: 岛屿边界
        rooms: 语义增强版房间规格列表
        adjacency_graph: 邻接关系图 {room_id: [neighbor_ids]}
        exterior_walls: 外墙方向列表 ['north', 'south', 'east', 'west']
        config: 求解器配置
        corridor_edges: 走廊接触边列表（用于可达性约束）

    Returns:
        房间划分结果列表
    """
    context = IslandContext(
        exterior_walls=exterior_walls,
        corridor_edges=corridor_edges or [],
    )
    solver = SemanticIslandPartitionSolver(
        island_polygon=island_polygon,
        rooms=rooms,
        adjacency_graph=adjacency_graph,
        island_context=context,
        config=config,
    )
    return solver.solve()
