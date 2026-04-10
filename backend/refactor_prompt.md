🏗️ Building 管线重构：终极避坑与优化总建议
这份总建议分为前端渲染、后端几何、求解器防御和测试规范四个维度，确保重构后的管线不仅跑得通，而且在各种极端边界下都具备极高的健壮性。

一、 前端渲染层：视觉健壮性与尺度自适应
1. 开启“极细线兜底” (Hairline Fallback) 代替彻底删除描边
[采纳你的建议 1.1] 永远不要完全信任计算几何的墙体生成。当 generate_wall_mesh 因为微小的浮点容差或超短边（<0.3m）过滤掉了某堵墙时，同色房间直接融合是灾难性的。

最佳实践： 在 MapRenderer.tsx 中保留 d.polygon 的 stroke，但将其降级为 CAD 风格的极细基准线。

TypeScript
// frontend/src/components/MapRenderer.tsx
.attr("stroke", "#334155")         // 深石板灰
.attr("stroke-width", 0.3)         // 极细线，墙体正常时会被完全遮挡
.attr("stroke-opacity", 0.4)       // 半透明，降低视觉干扰
.style("shape-rendering", "geometricPrecision");
注意层级 (Z-Order)： 确保 SVG 渲染顺序为：Polygon 房间 (带极细线) -> 实心墙体 Wall -> 门窗。

2. 门窗视觉厚度响应式动态计算
[采纳你的建议 1.5] 建筑平面图的跨度极大（从 10x10 的别墅到 100x100 的展馆），硬编码门窗厚度（如 0.5m）在宏观尺度下会直接隐形。

最佳实践： 在 aiService.ts 转换层，利用当前楼层的最小物理维度动态计算视觉厚度，并设置物理底线（0.3m）。

TypeScript
// frontend/src/services/aiService.ts
const floorMinDim = Math.min(v2.building.width, v2.building.depth);
// 动态系数 2.5%，且最小不低于 0.3 米
const VISUAL_THICKNESS = Math.max(0.3, floorMinDim * 0.025); 

// 在转换门窗时统一使用 VISUAL_THICKNESS
// ...
二、 后端几何层：摒弃启发式猜测，拥抱纯粹拓扑
3. 彻底禁用 MRR，改用“边界容差求交法”提取墙体中心线
[结合我之前的致命 Bug 预警] 绝对不能使用 minimum_rotated_rectangle (MRR) 来提取墙体！当相邻房间的共享边界是 L 形或阶梯形时，MRR 的对角线会直接生成一条横穿房间的斜墙。

最佳实践： 在 generate_wall_mesh 中，利用 Shapely 的 buffer 容差直接求拓扑交集，提取出绝对正交的线段。

Python
# backend/core/geometry/postprocessor.py 的严谨墙体求交逻辑
bound_a = poly_a.boundary
bound_b = poly_b.boundary.buffer(0.002) # 2毫米容差防浮点丢失

shared_edges = bound_a.intersection(bound_b)

if not shared_edges.is_empty:
    # 提取 shared_edges 中的所有 LineString
    lines = extract_linestrings(shared_edges)
    for line in lines:
        if line.length > 0.3: # 过滤极短碎片
            walls.append(WallSegment(geometry=line, ...))
三、 求解器防御层：防止硬约束导致系统崩溃
4. 走廊挂载约束的“防死锁 (INFEASIBLE)”降级
[结合我之前的求解器死锁预警] 强制所有 needs_corridor_access=True 的房间贴合走廊是一把双刃剑。如果走廊很短，但房间很多，CP-SAT 求解器会因为空间不足直接返回无解（INFEASIBLE），导致整个楼层生成失败。

最佳实践： 在 island_partition_solver.py 中，写入硬约束前进行“边长容量预检”。

Python
# 写入 model.Add(room.y == edge) 之前
total_min_width_needed = sum([2.5 for r in rooms if r.needs_corridor_access])
available_edge_length = calculate_island_corridor_edge_length(island)

if total_min_width_needed > available_edge_length:
    # 容量爆满！必须执行降级策略
    # 策略：按面积从小到大排序，将部分次要房间的 needs_corridor_access 临时设为 False，改为软约束或不贴边。
    relax_corridor_constraints(rooms)
四、 测试规范：拒绝“自欺欺人”的伪验证
5. 端到端 (E2E) 的集成测试替代 Mock 模块测试
[采纳你的验证脚本漏洞提示] 如果测试脚本直接把生成的拓扑“空岛屿”当成“房间”塞给墙体生成器，它将永远遇不到 MIQP 内部切分后产生的微小浮点缝隙、L 型拐角等真实痛点。通过了这种测试毫无意义。

最佳实践： 更新 Phase 1 的验证脚本，必须拉起真实的 generate_layout_v2，让 Treemap 和 MIQP 跑完真实切分后，再验证全局墙网是否能正确捕捉到子房间之间的墙体。

Python
# 真实的验证脚本逻辑必须是这样：
boundary = box(0, 0, 20, 15)
rooms_specs = [...] # 定义真实房间需求

# 1. 跑完真实的全局 MIQP 分配和求解
result = generate_layout_v2(boundary, rooms_specs) 

# 2. 从真实的切分结果中生成墙网
walls = generate_wall_mesh(result.room_layouts, result.corridors, result.core_tube, boundary)

# 3. 断言验证
assert any(w.type == 'partition_wall' for w in walls), "内部隔墙生成失败！"