# 数据准备（Building 语义）
## 输入

- 来源代码：`scripts/full_pipeline.py` → `_main_async`
- 输入变量（示例来自命令行参数）：
  - `args.prompt: str`（示例：`"生成一个二层住宅区..."`）
  - `args.model: str`（示例：`"gemini-2.5-pro"`）
  - `args.core: str`（示例：`"north"`）
- 语义请求对象：
  - `req: GenerateSemanticsRequest`
  - 关键字段：
    - `scene_type=SceneType.BUILDING`
    - `user_prompt=args.prompt`
    - `provider=args.provider`
    - `model=args.model`

关键代码（原样拷贝）：

```python
req = GenerateSemanticsRequest(
    scene_type=SceneType.BUILDING,
    user_prompt=args.prompt,
    provider=args.provider,
    model=args.model,
)
allocation, parse_warnings = await building_semantic_flow.generate_building_semantics(req)
```

## 算法

- 语义规划分两部分：
  1) 构建面向 LLM 的用户 prompt（把用户需求与可选的总面积/楼层数写入提示词）。
  2) 对 LLM 返回的文本做“多层防御解析”（JSON 提取 → 预处理 → Pydantic 校验 → 后处理归一）。

关键代码（原样拷贝，`backend/core/flows/building_semantic_flow.py`）：

```python
obj = _robust_json_loads(text, provider, model)
if obj is None:
    raise RuntimeError("无法从 LLM 响应中提取有效 JSON")
allocation = BuildingAllocation.model_validate(obj)
```

## 输出

- 输出对象：
  - `allocation: BuildingAllocation`
  - `parse_warnings: List[str]`
- 在 `scripts/full_pipeline.py` 中，`allocation` 会被用于后续拓扑/几何生成，不会直接落盘；落盘从“粗布局导出”阶段开始（见后文）。


# 几何划分（以核心筒为“原点”的空间划分）
## 输入

- 核心筒生成输入（来自 `RectangularTopologyGenerator.generate(...)`）：
  - `floor_bounds: Tuple[float, float, float, float]`（示例：`(0.0, 0.0, 11.29, 7.53)`）
  - `area_ratio: float`（示例：`0.08`）
  - `position: str`（示例：`"north"|"south"|"center"|"entrance"`）
  - `grid_alignment: float`（示例：`0.5`）
- 空间切割输入：
  - `core_tube: CoreTube`
  - `corridor_layout: str`（示例：`"cross"|"H"|"grid"|"door_side"`）
  - `corridor_width: float`（示例：`2.0`）

关键代码（原样拷贝，`backend/core/geometry/topology_generator.py`）：

```python
core_tube = CoreTube.create_for_floor(self.bounds, grid_alignment=self.grid_alignment)
if corridor_layout == "door_side":
    corridors = self._generate_cross_corridors(core_tube)
elif corridor_layout == "cross":
    corridors = self._generate_cross_corridors(core_tube)
elif corridor_layout == "H":
    corridors = self._generate_h_corridors(core_tube)
elif corridor_layout == "grid":
    corridors = self._generate_grid_corridors(core_tube)
else:
    corridors = self._generate_cross_corridors(core_tube)
```

## 算法

### 1.1 核心筒坐标提取与对齐（当前实现为轴对齐矩形）

当前代码并没有“辐射状（radial）划分”或“以方向向量做扇区划分”的实现；核心筒与走廊均为**正交轴对齐**（horizontal/vertical）结构：

- 核心筒中心点 `center=(cx,cy)` 由楼板 bounds 计算并按 `grid_alignment` 对齐；
- 核心筒整体 polygon 为 `box(cx±width/2, cy±depth/2)`；
- 核心筒内部拆分为 `staircase + elevator_hall + elevator_shaft` 三个子矩形。

关键代码（原样拷贝，`CoreTube.create_for_floor`）：

```python
core_area = floor_area * area_ratio
width = np.sqrt(core_area * aspect_ratio)
depth = core_area / width

width = max(grid_alignment, round(width / grid_alignment) * grid_alignment)
depth = max(grid_alignment, round(depth / grid_alignment) * grid_alignment)

cx = (x_min + x_max) / 2
if position == "north":
    cy = y_max - depth / 2  # 紧贴北墙
elif position == "south":
    cy = y_min + depth / 2  # 紧贴南墙
elif position == "center":
    cy = (y_min + y_max) / 2
elif position == "entrance":
    cy = y_min + depth / 2 + 3
else:
    cy = y_max - depth / 2  # 默认北墙
```

### 1.1网格状空间划分（grid slicing）

“空间划分”的核心是 `_generate_perfect_rectangular_islands(...)`：收集外轮廓/走廊/核心筒的关键坐标 → 划分网格 cell → 过滤被 corridor/core 覆盖的 cell → 合并连续 cell 成矩形岛屿。

关键数据结构：
- `xs_list/ys_list: List[float]`：网格切片坐标轴
- `free[i][j]: bool`：cell 是否可用
- `cell_map: Dict[(i,j), island_id]`：cell → island 映射
- `edge_set_islands: Dict[frozenset({id_a,id_b}), "vertical"|"horizontal"]`：岛屿邻接

关键代码（原样拷贝，`_generate_perfect_rectangular_islands`）：

```python
free = [[False for _ in range(len(ys_list) - 1)] for _ in range(len(xs_list) - 1)]

for i in range(len(xs_list) - 1):
    x0, x1 = xs_list[i], xs_list[i + 1]
    if (x1 - x0) < 0.05:
        continue
    for j in range(len(ys_list) - 1):
        y0, y1 = ys_list[j], ys_list[j + 1]
        if (y1 - y0) < 0.05:
            continue

        cell = box(x0, y0, x1, y1)
        cell2 = cell.intersection(self.floor)
        if cell2.area / cell.area < 0.99:
            continue
        overlap = cell2.intersection(subtract_union)
        overlap_ratio = overlap.area / cell2.area if cell2.area > 0 else 1.0
        if overlap_ratio > 0.05:
            continue

        free[i][j] = True
```

### 1.3 最终几何拓扑（节点=空间，边=邻接/连通）

当前项目的“拓扑图”并不会被单独持久化为 JSON/protobuf；而是以 `edge_set: Dict[FrozenSet[str], str]` 的形式挂在 `LayoutResultV2.edge_set` 上，并用于后处理生成墙体。

拓扑边判定逻辑（共享边长阈值 + 位置容差）在 `layout_generator._build_edge_set_from_rects(...)`：

```python
if abs(ax + aw - bx) < tol or abs(bx + bw - ax) < tol:
    y0 = max(ay, by)
    y1 = min(ay + ah, by + bh)
    if (y1 - y0) > min_shared_length:
        edge_set[frozenset({id_a, id_b})] = "vertical"
    continue

if abs(ay + ah - by) < tol or abs(by + bh - ay) < tol:
    x0 = max(ax, bx)
    x1 = min(ax + aw, bx + bw)
    if (x1 - x0) > min_shared_length:
        edge_set[frozenset({id_a, id_b})] = "horizontal"
```

连通性检查（BFS）在 `check_connectivity_topological(...)`：

```python
for edge_key in edge_set.keys():
    id_a, id_b = tuple(edge_key)
    if id_a in adj and id_b in adj:
        adj[id_a].add(id_b)
        adj[id_b].add(id_a)
```

单元测试：
- [test_topology_edge_walls.py](file:///f:/Ledzepplin/city_ge/new-main/backend/tests/test_topology_edge_walls.py#L261-L271) 校验 `_build_edge_set_from_rects` 对小缝隙容忍度
- [test_topology_edge_walls.py](file:///f:/Led%20zepplin/city_ge/new-main/backend/tests/test_topology_edge_walls.py#L177-L185) 校验 `check_connectivity_topological` 可达性

**当前代码未实现“针对 L 形平面稳定输出拓扑节点/边”的能力**；要支持 L 形，需要放宽这条面积比过滤规则并引入更通用的 polygon 分割（目前仓库中不存在对应实现）。

## 输出

- `core_tube: CoreTube`（存入 `LayoutResultV2.core_tube`，并在 [serializers.py](file:///f:/Led%20zepplin/city_ge/new-main/backend/core/geometry/serializers.py#L156-L294) 中序列化到顶层 `core_tube` 与各层 `room_rects`）
- `corridors: List[Corridor]`（各层 `floors[floor_id]["corridors"]`）
- `islands: List[Island]`（仅在内存用于分配与求解；不直接落盘）
- `edge_set: Dict[FrozenSet[str], str]`（仅用于墙体生成；不直接落盘）


# 几何拆分（走廊膨胀 corridor dilation）
## 输入

- `Corridor.centerline: LineString`
- `Corridor.width: float`

关键代码（原样拷贝，`Corridor.__post_init__`）：

```python
self.polygon = self.centerline.buffer(
    self.width / 2,
    cap_style="flat",
    join_style="mitre",
)
```

## 算法

- 当前“膨胀核大小”就是 `width/2`，并且是一次性 `buffer`（无迭代次数概念）。
- 当前实现没有“防火规范距离/与外墙最小距离”的独立常量；走廊只会被 clamp 到楼板 bounds，并在切割时减去 core_tube 防止重叠。

走廊裁剪核心筒（避免重叠）关键代码（原样拷贝，`RectangularTopologyGenerator.generate`）：

```python
core_poly_for_cut = core_tube.polygon.buffer(1e-4, join_style="mitre")
for corridor in corridors:
    diff = corridor.polygon.difference(core_poly_for_cut).simplify(0.01)
```

## 输出

- `corridor.polygon: Polygon`（用于后续 subtract_union 切割楼板并生成 islands）


# 房间生成（Treemap 布局与语义切割）
## 输入

- `rooms: List[RoomSpec]`（语义 RoomSpec，字段：`target_area/zone/adjacency_required/...`）
- `adjacency_graph: Dict[str, List[str]]`
- `island_bounds: Tuple[float,float,float,float]`

关键代码（原样拷贝，`hierarchical_treemap` 签名）：

```python
def hierarchical_treemap(
    island_bounds: Tuple[float, float, float, float],
    rooms: List[RoomSpec],
    adjacency_graph: Dict[str, List[str]],
) -> List[WarmStartRect]:
```

## 算法

### 2.1 权重指标在代码中的映射

- 面积权重：来自 `RoomSpec.target_area`（Treemap 的 `sizes`）
- 功能优先级：来自 `RoomSpec.zone`，用 `_ZONE_PRIORITY` 排序
- 采光系数：当前代码没有单独的“采光权重表”；`needs_window` 会进入后续约束/门窗生成阶段，而非 Treemap 权重

关键代码（原样拷贝，`treemap.py`）：

```python
_ZONE_PRIORITY = {
    ZoneType.PUBLIC: 0,
    ZoneType.PRIVATE: 1,
    ZoneType.SERVICE: 2,
    ZoneType.CIRCULATION: 3,
}
```

### 2.1递归切割（squarify 贪心条带切割）

本项目没有“严格递归二分切割”的实现；实际使用的是 `squarify()` 的贪心条带切割：在 `while remaining:` 中按当前长边方向切 strip，并在 strip 内按面积分配。

关键代码（原样拷贝，`squarify`）：

```python
while remaining:
    if cw >= ch:
        row, remaining = _layout_row(remaining, cw, ch, sum(remaining))
        rects.extend(_do_layout_row(row, cx, cy, cw, ch))
        strip_w = sum(row) / ch if ch > 0 else 0
        cx += strip_w
        cw -= strip_w
    else:
        row, remaining = _layout_row(remaining, ch, cw, sum(remaining))
        rects.extend(_do_layout_row_v(row, cx, cy, cw, ch))
        strip_h = sum(row) / cw if cw > 0 else 0
        cy += strip_h
        ch -= strip_h
```

复杂度（以代码行为为准的文字说明）：
- 每次迭代至少消耗 1 个 `remaining` 元素，整体 O(n²) 的上界来自 `_layout_row` 的逐个尝试；实际 n 通常较小。

## 输出

- `warm_start: List[WarmStartRect]`（后续作为 CP-SAT 的初值/参考）
- 使用位置：`SemanticIslandPartitionSolver.solve()` 的 Stage 1


# 房间生成（语义 CP-SAT MIQP）
## 输入

- 输入房间规格：`self.rooms: List[RoomSpec]`（语义 solver 内部 RoomSpec）
- 输入岛屿边界：`self.island: Polygon`
- warm start：`warm_start: List[WarmStartRect]`
- solver 配置：`self.config`（包含 `time_limit/area_tolerance/weight_*` 等）

关键代码（原样拷贝，`_solve_cpsat`）：

```python
SCALE = 100  # cm 精度
W_s = int(self.W * SCALE)
D_s = int(self.D * SCALE)
model = cp_model.CpModel()
```

## 算法

当前实现使用 OR-Tools CP-SAT（整数规划），但历史命名沿用 “MIQP”：

- 决策变量（矩形房间）：
  - `x[i], y[i], w[i], d[i]`（均为整数，单位 cm）
  - `x_end[i], y_end[i], area[i]` 等派生变量
- 约束：
  - `AddNoOverlap2D(...)`：矩形不重叠
  - 面积上下界：`area[i] >= target*(1-tol)` / `<= target*(1+tol)`
  - 宽高比：`w[i] * 100 >= ar_min_100 * d[i]` / `<= ar_max_100 * d[i]`
  - 走廊可达性：`sum(touches_any_corridor) >= 1`（根据 corridor_mode 受控放宽）
- 目标函数（线性组合）：
  - 面积偏差、邻接软目标、紧凑度、宽高比惩罚、coverage gap 等

关键代码（原样拷贝，`_solve_cpsat`）：

```python
model.AddNoOverlap2D(x_intervals, y_intervals)

model.Add(area[i] >= int(target_s * (1 - tol)))
model.Add(area[i] <= int(target_s * (1 + tol)))

model.Add(w[i] * 100 >= ar_min_100 * d[i])
model.Add(w[i] * 100 <= ar_max_100 * d[i])

model.Minimize(sum(objectives))
```

求解器封装与异常重试/回退：

```python
for attempt in attempts:
    try:
        results, dropped_rooms = self._solve_cpsat(
            warm_start,
            corridor_mode=str(attempt["corridor_mode"]),
            area_tolerance=float(attempt["area_tolerance"]),
            aspect_relax_factor=float(attempt["aspect_relax_factor"]),
        )
        solved = True
        break
    except SemanticSolveError as e:
        if not e.is_infeasible:
            break
        continue
```

## 输出

- `results: List[RoomResult]`（每个房间矩形：`x/y/width/depth`）
- 若语义 CP-SAT 不可行，则回退到旧 solver：`self._fallback_solve()`（见 [island_partition_solver.py]


# 房间生成（代码-算法一致性验证）
## 输入

- 当前仓库存在的“可审计输入”：
  - `LayoutResultV2` 内部 room specs、warm start、solver config
  - `Semantic CP-SAT model summary` 日志（结构化字段）

关键代码（原样拷贝，`island_partition_solver.py`）：

```python
logger.info(
    "Semantic CP-SAT model summary: rooms=%d, needs_window=%d, needs_corridor_access=%d, corridor_mode=%s, dropped_corridor_access=%d, forbidden_zones=%d, area_tolerance=%.2f, aspect_relax=%.2f, corridor_edges=%s",
    self.n,
    needs_window,
    needs_corridor_access,
    corridor_mode,
    len(dropped_corridor_access_rooms),
    len(forbidden_zones),
    float(tol),
    float(aspect_relax_factor),
    corridor_edges,
)
```

## 算法

- 当前能够做的一致性核查主要依赖：
  - 上述 model summary 日志字段
  - `backend/core/geometry/debug_exporter.py` 的 SVG 导出能力（用于几何结果可视化对比，而非 CP-SAT 输入 JSON）

## 输出

- 当前阶段无独立落盘脚本输出（除 debug SVG 导出外）。


# 门窗处理（扩充：定位、尺寸、验证）
## 输入

- 门洞生成输入：`walls: List[WallSegment]`、`zone_types/zone_rects`、`door_width`
- 窗户生成输入：
  - `generate_windows(...)`：在外墙段上按 spacing 摆放
  - `generate_windows_from_floor_boundary(...)`：按房间 bbox 与 floor bounds proximity 摆放

## 算法

### 3.1 门窗定位与“朝向 forward”（当前实现）

当前代码未实现“射线法/辐射度”的采光向量场；窗户的 `forward` 由几何规则确定：
- 若由 wall 段生成：按墙旋转（水平墙 forward=(0,0,1)，垂直墙 forward=(1,0,0)）
- 若由 floor bounds 推导：`forward` 指向房间中心（`_normalize_2d(cx-x, cy-wy)`）

关键代码（原样拷贝，`generate_windows_from_floor_boundary`）：

```python
cx = float(rx + rw / 2)
cy = float(ry + rh / 2)
fx, fy = _normalize_2d(float(cx) - float(x), float(cy) - float(wy))
windows.append(WindowPlacement(
    position=(round(float(x), 2), round(float(wy), 2)),
    width=window_width,
    room_id=rid,
    wall_length=round(float(wall_len), 2),
    rotation=90.0,
    thickness=float(exterior_thickness),
    forward=(float(fx), 0.0, float(fy)),
))
```

### 3.2 门窗尺寸优化

门窗尺寸由函数参数常量控制：
- `door_width: float = 0.9`
- `window_width: float = 1.2`
- `window_spacing: float = 2.0`

### 3.3 输出验证

现有验证主要来自单测，例如：
- [test_windows_generated_from_floor_bounds_have_correct_rotation]

## 输出

- 生成物落在 `building_dict["building"]["floors"][floor_id]["doors"/"windows"]`，并最终落盘进入 `layout_F{n}.json/refined_layout_F{n}.json`


# 家具摆放（LLM coarse + CP-SAT refine）
## 输入

- 房间边界：`room: RoomBoundary`
- 家具规格：`furnitures: List[FurnitureSpec]`
- 障碍物：`obstacles: List[Obstacle]`
- LLM client：`client: AsyncOpenAI`

## 算法

### 4.1 LLM 语义预测（coarse layout）

当前仓库的 prompt 模板是代码内字符串拼装（没有独立 few-shot 文件路径）。

关键代码（原样拷贝，`coarse_layout_agent.py`）：

```python
system_prompt = _build_system_prompt()
user_prompt = _build_user_prompt(room, furnitures, obstacles)
messages = [
    ChatMessage(role="system", content=system_prompt),
    ChatMessage(role="user", content=user_prompt),
]
raw = await call_llm_with_retry(client, messages, llm_config)
```

重试与 backoff（真实存在）：

```python
for i in range(MAX_RETRIES):
    try:
        if i > 0:
            await sleep(1000 * i)
        return await client.chat(compressed, config)
```

### 4.2 “MIQP 精调”（当前实现为 CP-SAT）

当前仓库未实现“人体工学评分/插座可达性”等约束；精调 solver 目标为“最小化 LLM 中心点位移（L1）”，并强制 `NoOverlap2D` 避让障碍物：

关键代码（原样拷贝，`refine_solver.py`）：

```python
model.AddNoOverlap2D(x_intervals, y_intervals)
model.Minimize(sum(dx_vars.values()) + sum(dy_vars.values()))
```

### 4.3 约束可解释性注释（REQ-ID）

现有约束以代码逻辑表达为主。

## 输出

- 输出类型：`RefinedLayout`
- 在 `scripts/full_pipeline.py` 中转换为 `type="furniture"` 的 elements，并落盘到 `refined_layout_F{n}.json`


# 渲染模块策略（本地渲染）
## 输入

- 本地渲染输入：`layout.json`（必须含 `width/height/elements[]`）
- 渲染模式：`mode in {"seg","cad"}`

关键代码（原样拷贝，`scripts/local_renderer.py`）：

```python
if mode == "seg":
    if out_path.suffix.lower() != ".png":
        raise ValueError("seg 模式仅允许输出 PNG（无损）")
    _validate_segmentation_palette(SEGMENTATION_COLORS if isinstance(SEGMENTATION_COLORS, dict) else {})
```

## 算法

### 5.1 本地渲染

当前仓库本地渲染采用 Matplotlib（Agg）进行 2D 栅格化/矢量化绘制，不包含 Embree/OptiX 路径追踪、PBR 材质、光照烘焙、多线程调度等模块。

关键绘制循环（原样拷贝）：

```python
for elem in sorted((e for e in elements if isinstance(e, dict)), key=sort_key):
    poly = elem.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        if mode == "seg":
            _draw_polygon_seg(ax, elem, elements)
        else:
            _draw_polygon(ax, elem)
        continue
```



## 输出

- `seg`：`*.png`（如 `refined_mask_F1.png`）
- `cad`：`*.png` 或 `*.svg`（仅在开启导出时生成）
