# Building 模块深度优化 Prompt

> 给 Claude Code 的综合优化任务指令

---

## 项目背景

这是一个 3D 建筑场景生成管线，核心流程为：

```
自然语言 → LLM语义规划 → 拓扑骨架生成 → 房间-岛屿分配 → Treemap+MIQP划分 → 验证输出
```

当前代码库位于 `backend/core/geometry/`，已实现 Treemap warm start + CP-SAT MIQP 求解器，但存在以下需要修复的问题。

---

## 🔴 P0 阻塞问题

### 问题 1.1：BuildingOrchestrator 编排器缺失

**现状**：
- `building_semantic_flow.py` 输出 `BuildingAllocation`（多楼层语义配比）
- `generate_layout_v2()` 只处理单层
- 缺少将多楼层串联起来的编排代码

**修复任务**：

创建 `building_orchestrator.py`：

```python
"""
building_orchestrator.py

多层 Building 编排器
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from .layout_generator import (
    LayoutResultV2,
    SemanticRoomSpec,
    SolverConfig,
    generate_layout_v2,
)
from .topology_generator import CoreTube


@dataclass
class BuildingResult:
    """多层建筑生成结果"""
    core_tube: CoreTube  # 跨层共享
    floor_layouts: Dict[str, LayoutResultV2]  # floor_id -> layout
    warnings: List[str]


class BuildingOrchestrator:
    """
    多层建筑编排器
    
    职责：
    1. 接收 BuildingAllocation
    2. 首层生成核心筒 → 锁定位置
    3. 逐层调用 generate_layout_v2，注入共享核心筒
    4. 验证垂直一致性
    """
    
    def __init__(
        self,
        floor_boundary: Polygon,
        config: Optional[SolverConfig] = None,
    ):
        self.floor_boundary = floor_boundary
        self.config = config or SolverConfig()
        self._shared_core_tube: Optional[CoreTube] = None
    
    def generate(
        self,
        allocation: "BuildingAllocation",  # 来自 building_semantic_flow
    ) -> BuildingResult:
        """
        生成整栋建筑
        
        流程：
        1. 首层生成拓扑 → 锁定核心筒
        2. 后续楼层复用核心筒位置
        3. 验证垂直管井对齐
        """
        floor_layouts: Dict[str, LayoutResultV2] = {}
        warnings: List[str] = []
        
        for i, floor in enumerate(allocation.floors):
            # 转换房间规格
            room_specs = self._convert_floor_rooms(floor)
            
            # 构建邻接图
            adjacency_graph = self._build_adjacency_graph(floor)
            
            # 生成布局
            layout = generate_layout_v2(
                floor_boundary=self.floor_boundary,
                room_specs=room_specs,
                adjacency_graph=adjacency_graph,
                config=self.config,
                # 注入共享核心筒（首层为 None，后续层复用）
                shared_core_tube=self._shared_core_tube,
            )
            
            # 首层锁定核心筒
            if i == 0:
                self._shared_core_tube = layout.core_tube
            
            floor_layouts[floor.floor_id] = layout
            warnings.extend(layout.warnings)
        
        # 验证垂直一致性
        vertical_warnings = self._validate_vertical_alignment(floor_layouts)
        warnings.extend(vertical_warnings)
        
        return BuildingResult(
            core_tube=self._shared_core_tube,
            floor_layouts=floor_layouts,
            warnings=warnings,
        )
    
    def _convert_floor_rooms(self, floor) -> List[SemanticRoomSpec]:
        """将 FloorAllocation.rooms 转换为 SemanticRoomSpec"""
        # TODO: 实现转换逻辑，填充 zone, needs_window, adjacency_required 等
        pass
    
    def _build_adjacency_graph(self, floor) -> Dict[str, List[str]]:
        """从 adjacency_tags 构建邻接图"""
        # TODO: 实现
        pass
    
    def _validate_vertical_alignment(
        self, 
        floor_layouts: Dict[str, LayoutResultV2],
    ) -> List[str]:
        """验证垂直管井对齐"""
        # TODO: 检查电梯井、楼梯间、管道井的 x/y 坐标是否跨层一致
        pass
```

**修改 `generate_layout_v2()` 签名**：

```python
def generate_layout_v2(
    floor_boundary: Polygon,
    room_specs: List[SemanticRoomSpec],
    adjacency_graph: Optional[Dict[str, List[str]]] = None,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "cross",
    entrance_position: Optional[Tuple[float, float]] = None,
    config: Optional[SolverConfig] = None,
    snap_grid: float = 0.1,
    verbose: bool = False,
    shared_core_tube: Optional[CoreTube] = None,  # 新增：跨层共享核心筒
) -> LayoutResultV2:
    # ...
    
    # Phase 1: 拓扑生成
    if shared_core_tube is not None:
        # 复用已有核心筒位置
        core_tube, corridors, islands = generate_rectangular_topology(
            floor_boundary=floor_boundary,
            corridor_width=corridor_width,
            core_tube_override=shared_core_tube,  # 强制使用已有位置
            corridor_layout=corridor_layout,
            entrance_position=entrance_position,
        )
    else:
        # 首层：正常生成
        core_tube, corridors, islands = generate_rectangular_topology(...)
```

---

### 问题 1.2：LLM Prompt 与后端字段未对齐

**现状**：

`building_prompts.py` 中 `BUILDING_PLANNER_SYSTEM_PROMPT` 的 JSON Schema：

```json
{
  "room_name": "string",
  "room_type": "string", 
  "target_area": 1.0,
  "requires_window": false,
  "weight": 1,
  "adjacency_tags": ["string"]  // ⚠️ 格式不匹配后端
}
```

**后端 `SemanticRoomSpec` 需要**：

```python
@dataclass
class RoomSpec:
    room_id: str
    room_type: str
    target_area: float
    zone: ZoneType  # 🔴 缺失
    needs_window: bool
    adjacency_required: List[str]  # 🔴 格式不匹配
    adjacency_preferred: List[str]  # 🔴 缺失
    adjacency_forbidden: List[str]  # 🔴 缺失
    min_width: float  # 🔴 缺失
    aspect_ratio_range: Tuple[float, float]  # 🔴 缺失
```

**修复任务**：

更新 `BUILDING_PLANNER_SYSTEM_PROMPT`：

```python
BUILDING_PLANNER_SYSTEM_PROMPT = """
...

JSON Schema（字段名必须完全一致）：
{
  "building_name": "string",
  "total_floors": 1,
  "overall_total_area": 1.0,
  "floors": [
    {
      "floor_number": 1,
      "floor_function_tag": "string",
      "floor_total_area": 1.0,
      "core_tube_area": 1.0,
      "corridor_allowance_area": 1.0,
      "rooms": [
        {
          "room_id": "room_001",           // 唯一标识（必填）
          "room_name": "客厅",              // 显示名称
          "room_type": "living_room",       // 类型标识
          "target_area": 25.0,              // 目标面积 m²
          "zone": "public",                 // 功能分区: public|private|service|circulation
          "needs_window": true,             // 是否需要采光
          "min_width": 3.5,                 // 最小开间 m
          "aspect_ratio_range": [0.6, 1.8], // 宽高比范围
          "adjacency_required": ["room_002"], // 必须相邻的房间ID
          "adjacency_preferred": ["room_003"], // 偏好相邻的房间ID
          "adjacency_forbidden": ["room_010"], // 禁止相邻的房间ID
          "weight": 1                       // 面积优先级 1-10
        }
      ]
    }
  ]
}

【邻接约束填写规则】
- adjacency_required: 功能上必须相邻的房间，如厨房-餐厅、主卧-主卫
- adjacency_preferred: 最好相邻但非强制，如客厅-阳台
- adjacency_forbidden: 禁止相邻的房间，如厨房-卧室、卫生间-餐厅

【zone 取值规则】
- public: 客厅、餐厅、厨房、接待区等公共空间
- private: 卧室、书房、卫生间等私密空间
- service: 储藏室、设备间、杂物间等服务空间
- circulation: 走廊、玄关、过道等交通空间
""".strip()
```

---

## 🟡 P1 架构问题

### 问题 2.1：代码冗余 — 新旧拓扑生成器共存

**现状**：

`topology_generator.py` 同时存在：
- 旧的 `generate_floor_skeleton()` — 基于 buffer 的不规则逻辑，返回 `FloorSkeleton`
- 新的 `RectangularTopologyGenerator` — 强制矩形化，返回 `(CoreTube, List[Corridor], List[Island])`

**问题**：
- 调用方不确定该用哪个
- `FloorSkeleton.usable_islands` 是 `List[Polygon]`，无语义属性
- `Island` dataclass 有完整语义属性

**修复任务**：

1. **删除** `generate_floor_skeleton()` 及相关的 `FloorSkeleton` 类
2. **保留** `RectangularTopologyGenerator` 作为唯一入口
3. **迁移** 所有调用方到新 API

```python
# topology_generator.py 重构后的结构

@dataclass
class CoreTube:
    """核心筒"""
    polygon: Polygon
    area: float
    position: Tuple[float, float]  # 中心坐标，用于跨层对齐


@dataclass
class Corridor:
    """走廊"""
    id: str
    centerline: LineString
    polygon: Polygon
    width: float


@dataclass
class Island:
    """可用岛屿（统一数据结构）"""
    id: str
    polygon: Polygon
    area: float
    remaining_capacity: float
    has_exterior_wall: bool
    exterior_walls: List[str]  # ['north', 'south', 'east', 'west']
    corridor_edges: List[str]  # ['south', 'west'] — 哪些边接触走廊（用于可达性约束）
    suggested_zone: ZoneType
    assigned_rooms: List[str] = field(default_factory=list)
    
    @property
    def is_rectangular(self) -> bool:
        """检查是否为矩形"""
        bbox_area = (self.polygon.bounds[2] - self.polygon.bounds[0]) * \
                    (self.polygon.bounds[3] - self.polygon.bounds[1])
        return self.polygon.area / bbox_area > 0.99


class RectangularTopologyGenerator:
    """矩形拓扑生成器（唯一入口）"""
    
    def __init__(
        self,
        floor_boundary: Polygon,
        corridor_width: float = 2.0,
        core_area_ratio: float = 0.08,
        corridor_layout: str = "cross",  # 'cross' | 'H' | 'grid'
        entrance_position: Optional[Tuple[float, float]] = None,
        core_tube_override: Optional[CoreTube] = None,  # 跨层复用
    ):
        self.floor_boundary = floor_boundary
        self.corridor_width = corridor_width
        self.core_area_ratio = core_area_ratio
        self.corridor_layout = corridor_layout
        self.entrance_position = entrance_position
        self.core_tube_override = core_tube_override
    
    def generate(self) -> Tuple[CoreTube, List[Corridor], List[Island]]:
        """
        生成矩形拓扑
        
        步骤：
        1. 创建/复用核心筒
        2. 生成走廊网格
        3. 切割出矩形岛屿
        4. 标注岛屿属性
        """
        # 1. 核心筒
        if self.core_tube_override:
            core_tube = self.core_tube_override
        else:
            core_tube = self._create_core_tube()
        
        # 2. 走廊
        corridors = self._create_corridors(core_tube)
        
        # 3. 岛屿
        islands = self._create_islands(core_tube, corridors)
        
        # 4. 标注语义属性
        self._annotate_islands(islands, corridors)
        
        return core_tube, corridors, islands
    
    def _annotate_islands(self, islands: List[Island], corridors: List[Corridor]):
        """标注岛屿的语义属性"""
        for island in islands:
            # 检测外墙方向
            island.exterior_walls = self._detect_exterior_walls(island)
            island.has_exterior_wall = len(island.exterior_walls) > 0
            
            # 检测走廊接触边（关键：用于可达性约束）
            island.corridor_edges = self._detect_corridor_edges(island, corridors)
            
            # 推断功能分区
            island.suggested_zone = self._suggest_zone(island)
    
    def _detect_corridor_edges(
        self, 
        island: Island, 
        corridors: List[Corridor],
    ) -> List[str]:
        """检测岛屿的哪些边接触走廊"""
        edges = []
        minx, miny, maxx, maxy = island.polygon.bounds
        
        for corridor in corridors:
            corridor_poly = corridor.polygon
            if not island.polygon.touches(corridor_poly) and \
               not island.polygon.intersects(corridor_poly):
                continue
            
            # 检测接触的是哪条边
            # TODO: 精确计算相交边
            ...
        
        return edges


# 便捷函数（唯一出口）
def generate_rectangular_topology(
    floor_boundary: Polygon,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = "cross",
    entrance_position: Optional[Tuple[float, float]] = None,
    core_tube_override: Optional[CoreTube] = None,
) -> Tuple[CoreTube, List[Corridor], List[Island]]:
    generator = RectangularTopologyGenerator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width,
        core_area_ratio=core_area_ratio,
        corridor_layout=corridor_layout,
        entrance_position=entrance_position,
        core_tube_override=core_tube_override,
    )
    return generator.generate()


# ❌ 删除以下旧代码
# class FloorSkeleton: ...
# def generate_floor_skeleton(...): ...
```

**迁移清单**：
- [ ] `debug_exporter.py` 的 `export_skeleton_to_svg` 适配新 `Island` 结构
- [ ] 删除 `building_types.py` 中的 `FloorSkeleton`（如果还在使用）
- [ ] 更新所有 import 语句

---

### 问题 2.2：垂直一致性风险

**现状**：

`generate_layout_v2()` 每层独立调用 `generate_rectangular_topology()`，各层核心筒位置可能不一致。

**问题后果**：
- 电梯井跨层错位
- 楼梯间无法垂直连通
- 管道井无法穿楼板

**修复任务**：

已在问题 1.1 中通过 `core_tube_override` 参数解决。需要额外添加验证：

```python
# building_orchestrator.py

def _validate_vertical_alignment(
    self,
    floor_layouts: Dict[str, LayoutResultV2],
) -> List[str]:
    """验证垂直管井对齐"""
    warnings = []
    
    if len(floor_layouts) < 2:
        return warnings
    
    # 获取首层核心筒位置
    first_floor_id = list(floor_layouts.keys())[0]
    ref_core = floor_layouts[first_floor_id].core_tube
    ref_cx, ref_cy = ref_core.polygon.centroid.x, ref_core.polygon.centroid.y
    
    for floor_id, layout in floor_layouts.items():
        if floor_id == first_floor_id:
            continue
        
        core = layout.core_tube
        cx, cy = core.polygon.centroid.x, core.polygon.centroid.y
        
        # 检查偏差
        dx = abs(cx - ref_cx)
        dy = abs(cy - ref_cy)
        
        if dx > 0.1 or dy > 0.1:  # 10cm 容差
            warnings.append(
                f"Floor {floor_id} core tube offset from reference: "
                f"dx={dx:.2f}m, dy={dy:.2f}m"
            )
    
    return warnings
```

---

### 问题 2.3：矩形化损失

**现状**：

`island_partition_solver.py` 中：

```python
self.x_min, self.y_min, self.x_max, self.y_max = island_polygon.bounds
self.W = self.x_max - self.x_min
self.D = self.y_max - self.y_min
```

当岛屿非矩形时（如 L 形、梯形），使用 AABB 会导致：
- 面积损失（AABB 内有空白区域）
- 与其他区域重叠（AABB 超出实际边界）

当前的 `_clip_to_boundary()` 只是事后裁剪，无法优化布局。

**修复任务**：

**方案 A：分割非矩形岛屿（推荐）**

在 `RectangularTopologyGenerator._create_islands()` 阶段，将非矩形岛屿分割为多个矩形子岛：

```python
def _create_islands(
    self, 
    core_tube: CoreTube, 
    corridors: List[Corridor],
) -> List[Island]:
    """创建岛屿，确保每个岛屿都是矩形"""
    
    # 1. 从楼层边界减去核心筒和走廊，得到可用区域
    usable = self.floor_boundary.difference(core_tube.polygon)
    for corridor in corridors:
        usable = usable.difference(corridor.polygon)
    
    # 2. 将可用区域分割为矩形
    islands = []
    for i, poly in enumerate(self._to_polygons(usable)):
        if self._is_rectangular(poly):
            # 已经是矩形，直接使用
            islands.append(self._create_island(f"island_{i}", poly))
        else:
            # 非矩形，分割为多个矩形
            sub_rects = self._split_to_rectangles(poly)
            for j, rect in enumerate(sub_rects):
                islands.append(self._create_island(f"island_{i}_{j}", rect))
    
    return islands

def _split_to_rectangles(self, poly: Polygon) -> List[Polygon]:
    """
    将非矩形多边形分割为矩形集合
    
    算法：递归二分法
    1. 找到多边形的最大内接矩形
    2. 从多边形中减去该矩形
    3. 对剩余区域递归分割
    4. 过滤掉过小的碎片
    """
    MIN_AREA = 4.0  # 最小岛屿面积 4m²
    
    if poly.area < MIN_AREA:
        return []
    
    # 尝试找最大内接矩形
    mir = self._max_inscribed_rectangle(poly)
    
    if mir is None or mir.area < MIN_AREA:
        # 退化情况：直接用 AABB 裁剪
        return [box(*poly.bounds).intersection(poly)]
    
    result = [mir]
    
    # 递归处理剩余区域
    remainder = poly.difference(mir)
    for sub in self._to_polygons(remainder):
        if sub.area >= MIN_AREA:
            result.extend(self._split_to_rectangles(sub))
    
    return result

def _max_inscribed_rectangle(self, poly: Polygon) -> Optional[Polygon]:
    """
    找多边形的最大内接矩形
    
    简化算法：
    1. 在多边形内采样网格点
    2. 对每个点，向四个方向扩展直到碰到边界
    3. 返回面积最大的矩形
    """
    from shapely.geometry import box as shapely_box
    
    minx, miny, maxx, maxy = poly.bounds
    step = max((maxx - minx) / 20, (maxy - miny) / 20, 0.5)
    
    best = None
    best_area = 0
    
    for x in np.arange(minx + step, maxx - step, step):
        for y in np.arange(miny + step, maxy - step, step):
            if not poly.contains(Point(x, y)):
                continue
            
            # 向四个方向扩展
            left = self._expand_direction(poly, x, y, -1, 0, minx)
            right = self._expand_direction(poly, x, y, 1, 0, maxx)
            down = self._expand_direction(poly, x, y, 0, -1, miny)
            up = self._expand_direction(poly, x, y, 0, 1, maxy)
            
            rect = shapely_box(left, down, right, up)
            if rect.area > best_area and poly.contains(rect):
                best = rect
                best_area = rect.area
    
    return best
```

**方案 B：改进 `_resolve_overlaps`（辅助）**

```python
def _resolve_overlaps(self, results: List[RoomResult]) -> List[RoomResult]:
    """
    解决房间重叠，优先保留面积大的房间
    
    改进：
    1. 按面积降序排列
    2. 后处理的房间避让先处理的房间
    3. 如果避让后面积损失 >30%，记录警告
    """
    sorted_results = sorted(results, key=lambda r: r.actual_area, reverse=True)
    final = []
    occupied = Polygon()
    
    for room in sorted_results:
        if occupied.is_empty:
            final.append(room)
            occupied = room.polygon
            continue
        
        # 检查重叠
        if not room.polygon.intersects(occupied):
            final.append(room)
            occupied = occupied.union(room.polygon)
            continue
        
        # 裁剪
        clipped = room.polygon.difference(occupied)
        if clipped.is_empty or clipped.area < room.actual_area * 0.3:
            logger.warning(
                "Room %s lost >70%% area due to overlap, consider re-partitioning",
                room.room_id,
            )
            continue
        
        # 取最大连通分量
        if hasattr(clipped, 'geoms'):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        
        # 重新矩形化
        minx, miny, maxx, maxy = clipped.bounds
        final.append(RoomResult(
            room_id=room.room_id,
            x=minx, y=miny,
            width=maxx - minx, depth=maxy - miny,
        ))
        occupied = occupied.union(box(minx, miny, maxx, maxy))
    
    return final
```

---

### 问题 2.4：走廊可达性缺失

**现状**：

`SemanticIslandPartitionSolver` 的 MIQP 模型没有强制房间接触走廊。

**问题后果**：
- 房间被其他房间包围，成为"死房间"
- 无法开门、无法到达

**修复任务**：

在 `Island` dataclass 中添加 `corridor_edges` 属性（已在问题 2.1 中添加），然后在 MIQP 约束中使用：

```python
# island_partition_solver.py

class SemanticIslandPartitionSolver:
    
    def __init__(
        self,
        island_polygon: Polygon,
        rooms: List[SemanticRoomSpec],
        adjacency_graph: Dict[str, List[str]],
        island_context: IslandContext,
        config: Optional[SolverConfig] = None,
    ):
        # ...
        
        # 新增：走廊边信息
        self.corridor_edges = island_context.corridor_edges or []
    
    def _solve_cpsat(
        self, 
        warm_start: Optional[List[RoomResult]],
    ) -> List[RoomResult]:
        # ...
        
        # ========== 走廊可达性约束（新增） ==========
        if self.corridor_edges:
            self._add_corridor_accessibility_constraints(
                model, x, y, x_end, y_end, W_s, D_s,
            )
    
    def _add_corridor_accessibility_constraints(
        self,
        model,
        x, y, x_end, y_end,
        W_s: int, D_s: int,
    ):
        """
        走廊可达性约束：每个房间至少有一条边接触走廊侧
        
        原理：
        - 如果岛屿的南侧是走廊，则每个房间的 y[i] 必须 == 0（接触南边界）
          或者该房间被另一个接触南边界的房间"传递可达"
        
        简化实现：
        - 至少 1 个房间直接接触走廊边
        - 其他房间通过邻接关系可达该房间
        
        这里采用更强的约束：每个房间必须至少接触一个边界（简化）
        """
        TOUCH_THRESHOLD = 10  # 10cm = 可开门宽度
        
        for i, room in enumerate(self.rooms):
            touches_any_corridor = []
            
            if 'south' in self.corridor_edges:
                # 房间的 y[i] <= TOUCH_THRESHOLD
                touch_south = model.NewBoolVar(f"touch_south_{i}")
                model.Add(y[i] <= TOUCH_THRESHOLD).OnlyEnforceIf(touch_south)
                touches_any_corridor.append(touch_south)
            
            if 'north' in self.corridor_edges:
                touch_north = model.NewBoolVar(f"touch_north_{i}")
                model.Add(y_end[i] >= D_s - TOUCH_THRESHOLD).OnlyEnforceIf(touch_north)
                touches_any_corridor.append(touch_north)
            
            if 'west' in self.corridor_edges:
                touch_west = model.NewBoolVar(f"touch_west_{i}")
                model.Add(x[i] <= TOUCH_THRESHOLD).OnlyEnforceIf(touch_west)
                touches_any_corridor.append(touch_west)
            
            if 'east' in self.corridor_edges:
                touch_east = model.NewBoolVar(f"touch_east_{i}")
                model.Add(x_end[i] >= W_s - TOUCH_THRESHOLD).OnlyEnforceIf(touch_east)
                touches_any_corridor.append(touch_east)
            
            # 至少接触一个走廊边
            if touches_any_corridor:
                model.Add(sum(touches_any_corridor) >= 1)
```

**更新 `IslandContext`**：

```python
# room_spec.py

@dataclass
class IslandContext:
    """岛屿上下文信息"""
    
    exterior_walls: List[str] = field(
        default_factory=lambda: ["north", "south", "east", "west"]
    )
    corridor_edges: List[str] = field(default_factory=list)  # 新增
    entrance_direction: Optional[str] = None
    preferred_public_side: Optional[str] = None
```

---

### 问题 2.5：面积弹性处理不足

**现状**：

`island_room_assigner.py` 的 `area_tolerance = 0.85` 只是过滤阈值，不会动态调整房间面积。

当 `sum(room.target_area) < island.area * 0.7` 时，布局会出现大量空白。
当 `sum(room.target_area) > island.area` 时，分配失败。

**修复任务**：

**方案 A：分配前预缩放（推荐）**

```python
# island_room_assigner.py

class IslandRoomAssigner:
    
    def assign(self) -> Dict[str, AssignmentResult]:
        # ...
        
        # 预检查：总面积
        total_room_area = sum(r.target_area for r in self.rooms.values())
        total_island_area = sum(i.area for i in self.islands.values())
        
        # ========== 动态缩放（新增） ==========
        scale_factor = 1.0
        if total_room_area > total_island_area * 0.95:
            # 房间面积超出岛屿容量，等比缩小
            scale_factor = (total_island_area * 0.92) / total_room_area
            logger.warning(
                "Scaling down room areas by %.1f%% to fit islands",
                (1 - scale_factor) * 100,
            )
            for room in self.rooms.values():
                room.target_area *= scale_factor
        
        elif total_room_area < total_island_area * 0.7:
            # 房间面积不足，等比放大以减少空白
            scale_factor = (total_island_area * 0.85) / total_room_area
            logger.info(
                "Scaling up room areas by %.1f%% to fill islands",
                (scale_factor - 1) * 100,
            )
            for room in self.rooms.values():
                room.target_area *= scale_factor
        
        # 继续正常分配...
```

**方案 B：MIQP 后处理填充（辅助）**

在 `island_partition_solver.py` 的 `_clip_to_boundary()` 之后添加间隙填充：

```python
def _fill_gaps(
    self, 
    results: List[RoomResult],
) -> List[RoomResult]:
    """
    填充房间之间的间隙
    
    策略：
    1. 计算所有房间的并集
    2. 找到岛屿内的空白区域
    3. 将空白区域按比例分配给相邻房间
    """
    if not results:
        return results
    
    all_rooms = unary_union([r.polygon for r in results])
    gaps = self.island.difference(all_rooms)
    
    if gaps.is_empty or gaps.area < 0.5:
        return results
    
    # 对每个间隙，找到最近的房间，扩展该房间
    for gap_poly in self._to_polygons(gaps):
        if gap_poly.area < 0.1:
            continue
        
        # 找最近的房间
        best_room_idx = None
        min_dist = float('inf')
        for i, room in enumerate(results):
            dist = room.polygon.distance(gap_poly)
            if dist < min_dist:
                min_dist = dist
                best_room_idx = i
        
        if best_room_idx is not None and min_dist < 0.5:
            # 扩展该房间
            room = results[best_room_idx]
            expanded = room.polygon.union(gap_poly)
            minx, miny, maxx, maxy = expanded.bounds
            results[best_room_idx] = RoomResult(
                room_id=room.room_id,
                x=minx, y=miny,
                width=maxx - minx, depth=maxy - miny,
            )
    
    return results
```

---

## 📋 验收标准

### 功能验收

| 测试项 | 预期结果 |
|--------|----------|
| 多层 Building 生成 | 3 层楼成功生成，核心筒位置跨层一致 |
| LLM 输出解析 | `zone`, `adjacency_required` 字段正确解析为 `SemanticRoomSpec` |
| 非矩形边界 | 岛屿覆盖率 > 90%，无明显空白 |
| 走廊可达性 | 每个房间至少有一条边接触走廊侧 |
| 面积弹性 | 目标面积 ±30% 时仍能生成有效布局 |

### 性能验收

| 指标 | 阈值 |
|------|------|
| 10 房间 / 层 | < 5s |
| 15 房间 / 层 | < 15s |
| 3 层 Building | < 45s |

### 代码质量

- [ ] 删除所有旧 `FloorSkeleton` 相关代码
- [ ] 所有 `# TODO` 注释已实现或有明确 issue 追踪
- [ ] 添加单元测试覆盖 5 个边界场景：
  - `test_single_room_island`
  - `test_l_shaped_boundary`
  - `test_all_rooms_need_window`
  - `test_conflicting_adjacency`
  - `test_area_overflow`

---

## 📁 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `building_orchestrator.py` | 新建 | 多层编排器 |
| `building_prompts.py` | 修改 | 更新 JSON Schema |
| `topology_generator.py` | 重构 | 删除旧代码，添加 `corridor_edges` |
| `island_partition_solver.py` | 修改 | 添加走廊可达性约束 |
| `island_room_assigner.py` | 修改 | 添加动态面积缩放 |
| `room_spec.py` | 修改 | `IslandContext` 添加 `corridor_edges` |
| `layout_generator.py` | 修改 | `generate_layout_v2` 添加 `shared_core_tube` 参数 |
| `building_types.py` | 删除/迁移 | 删除 `FloorSkeleton` |
| `debug_exporter.py` | 修改 | 适配新 `Island` 结构 |
| `__init__.py` | 修改 | 更新导出 |

---

## 执行顺序建议

1. **Phase 1**：更新 `building_prompts.py` 的 JSON Schema（影响最小，可立即部署）
2. **Phase 2**：重构 `topology_generator.py`，添加 `corridor_edges`
3. **Phase 3**：创建 `building_orchestrator.py`，实现垂直一致性
4. **Phase 4**：修改 `island_partition_solver.py`，添加走廊可达性约束
5. **Phase 5**：修改 `island_room_assigner.py`，添加面积弹性
6. **Phase 6**：添加单元测试，验证边界场景
