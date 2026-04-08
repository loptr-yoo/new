# Claude Code Prompt: 建筑布局生成流水线全面优化

## 🎯 任务概述

对 `backend/core/geometry/` 目录下的建筑布局生成代码进行全面优化：

| 任务 | 目标 |
|------|------|
| **Phase 1** | 清理冗余代码，删除废弃的 Power Diagram、L-BFGS、CMA-ES 等实现 |
| **Phase 2** | 优化拓扑骨架生成，确保岛屿为矩形，适配 Treemap+MIQP |
| **Phase 3** | 新增房间-岛屿分配层，实现分层决策 |
| **Phase 4** | 补充优化，改进整体架构 |

---

## 📁 目标文件结构

```
backend/core/geometry/
│
│ ══════════════════════════════════════════════════════════
│ 核心文件（保留并优化）
│ ══════════════════════════════════════════════════════════
├── layout_generator.py           # 主入口 ← 修改：集成新流程
├── topology_generator.py         # 拓扑生成 ← 重写：矩形岛屿生成
├── island_partition_solver.py    # 岛屿划分 ← 保留：Treemap+MIQP
├── constraints.py                # 约束定义 ← 保留
├── constraint_validator.py       # 约束验证 ← 扩展：语义验证
│
│ ══════════════════════════════════════════════════════════
│ 新建文件
│ ══════════════════════════════════════════════════════════
├── room_spec.py                  # 新建：房间规格 + 语义属性
├── island_room_assigner.py       # 新建：房间-岛屿分配
├── treemap.py                    # 新建：分层 Treemap 算法
│
│ ══════════════════════════════════════════════════════════
│ 简化文件
│ ══════════════════════════════════════════════════════════
├── axis_align.py                 # 简化：仅保留 snap_to_grid()
│
│ ══════════════════════════════════════════════════════════
│ 废弃文件（删除或归档到 _deprecated/）
│ ══════════════════════════════════════════════════════════
├── optimizer_v2.py               # ❌ 删除：L-BFGS 优化器
├── analytic_power_diagram.py     # ❌ 删除：Power Diagram
├── power_diagram.py              # ❌ 删除：Power Diagram
├── layout_optimizer.py           # ❌ 删除：CMA-ES 优化器
├── orthogonalization.py          # ❌ 删除：正交化后处理
│
└── __init__.py                   # 更新导出
```

---

## 📋 Phase 1: 清理冗余代码

### 1.1 执行步骤

```bash
# Step 1: 检查废弃文件的引用
grep -rn "from.*optimizer_v2" backend/
grep -rn "from.*power_diagram" backend/
grep -rn "from.*analytic_power_diagram" backend/
grep -rn "from.*layout_optimizer" backend/
grep -rn "from.*orthogonalization" backend/
grep -rn "PowerDiagram" backend/
grep -rn "LBFGSOptimizer" backend/
grep -rn "CMAESOptimizer" backend/

# Step 2: 创建归档目录
mkdir -p backend/core/geometry/_deprecated

# Step 3: 移动废弃文件
mv backend/core/geometry/optimizer_v2.py backend/core/geometry/_deprecated/
mv backend/core/geometry/analytic_power_diagram.py backend/core/geometry/_deprecated/
mv backend/core/geometry/power_diagram.py backend/core/geometry/_deprecated/
mv backend/core/geometry/layout_optimizer.py backend/core/geometry/_deprecated/
mv backend/core/geometry/orthogonalization.py backend/core/geometry/_deprecated/

# Step 4: 更新引用
# 删除 layout_generator.py 中对废弃模块的导入和调用
```

### 1.2 axis_align.py 简化

```python
# axis_align.py - 简化版

"""
轴对齐工具（简化版）

MIQP 算法天然产生正交布局，不再需要复杂的正交化后处理。
仅保留网格对齐功能。
"""

from shapely.geometry import Polygon
from shapely import affinity
import numpy as np


def snap_to_grid(polygon: Polygon, grid_size: float = 0.1) -> Polygon:
    """
    将多边形顶点对齐到网格
    
    参数:
        polygon: 输入多边形
        grid_size: 网格大小（米）
    
    返回:
        对齐后的多边形
    """
    coords = list(polygon.exterior.coords)
    snapped_coords = [
        (
            round(x / grid_size) * grid_size,
            round(y / grid_size) * grid_size
        )
        for x, y in coords
    ]
    return Polygon(snapped_coords)


# ═══════════════════════════════════════════════════════════════
# 以下函数已废弃，MIQP 天然保证正交
# ═══════════════════════════════════════════════════════════════
# def axis_align_cells() -> 已删除
# def ensure_rectangles() -> 已删除
# def orthogonalize_polygon() -> 已删除
```

---

## 📋 Phase 2: 矩形拓扑骨架生成

### 2.1 设计原则

```
目标：确保核心筒+走廊切割后，所有岛屿都是矩形

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌──────────────┐          │          ┌──────────────┐            │
│   │              │          │          │              │            │
│   │   岛屿 1     │    走    │    走    │   岛屿 2     │            │
│   │   (矩形)     │    廊    │    廊    │   (矩形)     │            │
│   │              │          │          │              │            │
│   └──────────────┘          │          └──────────────┘            │
│ ────────────────────────────┼──────────────────────────────────────│
│            走廊             │              走廊                    │
│ ────────────────────────────┼──────────────────────────────────────│
│   ┌──────────────┐    ┌─────┴─────┐    ┌──────────────┐            │
│   │              │    │           │    │              │            │
│   │   岛屿 3     │    │  核心筒   │    │   岛屿 4     │            │
│   │   (矩形)     │    │  (矩形)   │    │   (矩形)     │            │
│   │              │    │           │    │              │            │
│   └──────────────┘    └───────────┘    └──────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

关键：
1. 核心筒为矩形，位置对齐网格
2. 走廊为正交线段（严格水平或垂直）
3. 走廊与核心筒共同将楼层划分为矩形网格
4. 岛屿 = 网格单元
```

### 2.2 topology_generator.py 重写

```python
# topology_generator.py

"""
矩形拓扑生成器

确保生成的所有岛屿都是矩形，适配 Treemap+MIQP 算法。

设计原则：
1. 核心筒为矩形，紧凑设计（占楼层面积 5-10%）
2. 走廊为正交网格，宽度统一
3. 岛屿 = 走廊切割后的矩形区域
4. 每个岛屿带有语义属性（外墙方向、推荐分区等）
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from shapely.geometry import Polygon, box, LineString, Point
from shapely.ops import unary_union
import numpy as np


class ZoneType(Enum):
    """功能分区类型"""
    PUBLIC = "public"
    PRIVATE = "private"
    SERVICE = "service"
    CIRCULATION = "circulation"


@dataclass
class CoreTube:
    """
    核心筒定义
    
    设计原则：
    - 紧凑矩形，包含电梯、楼梯、设备间
    - 占楼层面积 5-10%
    - 位置靠近中心或入口
    """
    polygon: Polygon
    center: Tuple[float, float]
    width: float
    depth: float
    
    @classmethod
    def create(
        cls,
        center: Tuple[float, float],
        width: float,
        depth: float
    ) -> 'CoreTube':
        """创建矩形核心筒"""
        cx, cy = center
        polygon = box(
            cx - width / 2, cy - depth / 2,
            cx + width / 2, cy + depth / 2
        )
        return cls(polygon=polygon, center=center, width=width, depth=depth)
    
    @classmethod
    def create_for_floor(
        cls,
        floor_bounds: Tuple[float, float, float, float],
        area_ratio: float = 0.08,  # 占楼层面积 8%
        aspect_ratio: float = 1.0,  # 宽高比
        position: str = 'center'   # 'center' | 'entrance'
    ) -> 'CoreTube':
        """根据楼层自动创建核心筒"""
        x_min, y_min, x_max, y_max = floor_bounds
        floor_area = (x_max - x_min) * (y_max - y_min)
        
        # 计算核心筒尺寸
        core_area = floor_area * area_ratio
        # area = width * depth, width/depth = aspect_ratio
        # => width = sqrt(area * aspect_ratio)
        width = np.sqrt(core_area * aspect_ratio)
        depth = core_area / width
        
        # 确定位置
        if position == 'center':
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
        elif position == 'entrance':
            # 假设入口在南侧中央
            cx = (x_min + x_max) / 2
            cy = y_min + depth / 2 + 3  # 距离入口 3m
        else:
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
        
        return cls.create((cx, cy), width, depth)


@dataclass
class Corridor:
    """走廊定义"""
    id: str
    centerline: LineString
    width: float
    orientation: str  # 'horizontal' | 'vertical'
    polygon: Polygon = field(init=False)
    
    def __post_init__(self):
        # 使用方形端点（cap_style=3）确保正交
        self.polygon = self.centerline.buffer(
            self.width / 2, 
            cap_style=3,  # flat cap
            join_style=2  # mitre join
        )


@dataclass
class Island:
    """
    岛屿定义
    
    属性：
    - 几何：矩形多边形
    - 语义：外墙方向、推荐分区、到入口/核心筒距离
    """
    id: str
    polygon: Polygon
    
    # 几何属性（自动计算）
    area: float = field(init=False)
    bounds: Tuple[float, float, float, float] = field(init=False)
    width: float = field(init=False)
    depth: float = field(init=False)
    
    # 语义属性
    has_exterior_wall: bool = False
    exterior_walls: List[str] = field(default_factory=list)
    distance_to_entrance: float = 0.0
    distance_to_core: float = 0.0
    suggested_zone: ZoneType = ZoneType.PUBLIC
    
    # 容量跟踪（用于房间分配）
    assigned_rooms: List[str] = field(default_factory=list)
    remaining_capacity: float = field(init=False)
    
    def __post_init__(self):
        self.area = self.polygon.area
        self.bounds = self.polygon.bounds
        self.width = self.bounds[2] - self.bounds[0]
        self.depth = self.bounds[3] - self.bounds[1]
        self.remaining_capacity = self.area
    
    @property
    def is_rectangular(self) -> bool:
        """检查是否为矩形"""
        bbox_area = self.width * self.depth
        return self.area / bbox_area > 0.99
    
    @property
    def centroid(self) -> Tuple[float, float]:
        c = self.polygon.centroid
        return (c.x, c.y)


class RectangularTopologyGenerator:
    """
    矩形拓扑生成器
    
    生成策略：
    1. 计算核心筒位置和尺寸
    2. 生成正交走廊网格（经过核心筒）
    3. 用核心筒和走廊切割楼层
    4. 提取矩形岛屿
    5. 计算岛屿语义属性
    """
    
    def __init__(
        self,
        floor_boundary: Polygon,
        corridor_width: float = 2.0,
        min_island_area: float = 20.0,
        grid_alignment: float = 0.5
    ):
        self.floor = floor_boundary
        self.corridor_width = corridor_width
        self.min_island_area = min_island_area
        self.grid_alignment = grid_alignment
        
        self.bounds = floor_boundary.bounds
        self.x_min, self.y_min, self.x_max, self.y_max = self.bounds
        self.floor_width = self.x_max - self.x_min
        self.floor_depth = self.y_max - self.y_min
    
    def generate(
        self,
        core_tube: Optional[CoreTube] = None,
        corridor_layout: str = 'cross',  # 'cross' | 'H' | 'grid'
        entrance_position: Optional[Tuple[float, float]] = None
    ) -> Tuple[CoreTube, List[Corridor], List[Island]]:
        """
        生成矩形拓扑
        
        参数:
            core_tube: 核心筒（如果为 None 则自动创建）
            corridor_layout: 走廊布局类型
            entrance_position: 入口位置
        
        返回:
            (核心筒, 走廊列表, 岛屿列表)
        """
        # Step 1: 创建核心筒
        if core_tube is None:
            core_tube = CoreTube.create_for_floor(self.bounds)
        
        # Step 2: 生成走廊
        if corridor_layout == 'cross':
            corridors = self._generate_cross_corridors(core_tube)
        elif corridor_layout == 'H':
            corridors = self._generate_h_corridors(core_tube)
        elif corridor_layout == 'grid':
            corridors = self._generate_grid_corridors(core_tube)
        else:
            corridors = self._generate_cross_corridors(core_tube)
        
        # Step 3: 生成岛屿
        islands = self._generate_islands(core_tube, corridors)
        
        # Step 4: 计算语义属性
        if entrance_position is None:
            entrance_position = (
                (self.x_min + self.x_max) / 2,
                self.y_min
            )
        self._compute_island_semantics(islands, core_tube, entrance_position)
        
        # Step 5: 验证
        self._validate(islands)
        
        return core_tube, corridors, islands
    
    def _generate_cross_corridors(self, core: CoreTube) -> List[Corridor]:
        """
        十字走廊布局
        
        ┌─────────────────────────────┐
        │             │               │
        │   岛屿      │      岛屿     │
        │             │               │
        ├─────────────┼───────────────┤
        │    走廊     │核心│   走廊   │
        ├─────────────┼───┼───────────┤
        │             │               │
        │   岛屿      │      岛屿     │
        │             │               │
        └─────────────────────────────┘
        """
        cx, cy = core.center
        
        # 水平走廊
        h_corridor = Corridor(
            id='corridor_h',
            centerline=LineString([
                (self.x_min, cy),
                (self.x_max, cy)
            ]),
            width=self.corridor_width,
            orientation='horizontal'
        )
        
        # 垂直走廊
        v_corridor = Corridor(
            id='corridor_v',
            centerline=LineString([
                (cx, self.y_min),
                (cx, self.y_max)
            ]),
            width=self.corridor_width,
            orientation='vertical'
        )
        
        return [h_corridor, v_corridor]
    
    def _generate_h_corridors(self, core: CoreTube) -> List[Corridor]:
        """
        H 型走廊布局（适合长条形楼层）
        """
        cx, cy = core.center
        
        # 主水平走廊
        h_corridor = Corridor(
            id='corridor_h',
            centerline=LineString([
                (self.x_min, cy),
                (self.x_max, cy)
            ]),
            width=self.corridor_width,
            orientation='horizontal'
        )
        
        # 两端垂直走廊
        v_left = Corridor(
            id='corridor_v_left',
            centerline=LineString([
                (self.x_min + self.floor_width * 0.15, self.y_min),
                (self.x_min + self.floor_width * 0.15, self.y_max)
            ]),
            width=self.corridor_width,
            orientation='vertical'
        )
        
        v_right = Corridor(
            id='corridor_v_right',
            centerline=LineString([
                (self.x_max - self.floor_width * 0.15, self.y_min),
                (self.x_max - self.floor_width * 0.15, self.y_max)
            ]),
            width=self.corridor_width,
            orientation='vertical'
        )
        
        return [h_corridor, v_left, v_right]
    
    def _generate_grid_corridors(self, core: CoreTube) -> List[Corridor]:
        """
        网格走廊布局（适合大型楼层）
        """
        corridors = []
        cx, cy = core.center
        
        # 主走廊
        corridors.extend(self._generate_cross_corridors(core))
        
        # 计算是否需要额外走廊
        # 如果岛屿尺寸超过 15m，添加额外走廊
        max_island_dim = 15.0
        
        # 检查水平方向
        left_width = cx - self.corridor_width / 2 - self.x_min
        right_width = self.x_max - (cx + self.corridor_width / 2)
        
        if left_width > max_island_dim:
            x_pos = self._align((self.x_min + cx) / 2)
            corridors.append(Corridor(
                id='corridor_v_extra_left',
                centerline=LineString([(x_pos, self.y_min), (x_pos, self.y_max)]),
                width=self.corridor_width,
                orientation='vertical'
            ))
        
        if right_width > max_island_dim:
            x_pos = self._align((self.x_max + cx) / 2)
            corridors.append(Corridor(
                id='corridor_v_extra_right',
                centerline=LineString([(x_pos, self.y_min), (x_pos, self.y_max)]),
                width=self.corridor_width,
                orientation='vertical'
            ))
        
        return corridors
    
    def _align(self, value: float) -> float:
        """对齐到网格"""
        return round(value / self.grid_alignment) * self.grid_alignment
    
    def _generate_islands(
        self,
        core: CoreTube,
        corridors: List[Corridor]
    ) -> List[Island]:
        """生成矩形岛屿"""
        
        # 合并所有要减去的区域
        subtract_regions = [core.polygon]
        for corridor in corridors:
            subtract_regions.append(corridor.polygon)
        
        subtract_union = unary_union(subtract_regions)
        
        # 从楼层中减去
        remaining = self.floor.difference(subtract_union)
        
        # 提取多边形
        if remaining.is_empty:
            return []
        
        if remaining.geom_type == 'MultiPolygon':
            polygons = list(remaining.geoms)
        elif remaining.geom_type == 'Polygon':
            polygons = [remaining]
        else:
            polygons = []
        
        # 创建岛屿
        islands = []
        for i, poly in enumerate(polygons):
            if poly.area < self.min_island_area:
                continue
            
            # 矩形化：取包围盒
            rect_poly = box(*poly.bounds)
            
            # 检查矩形化损失
            if poly.area / rect_poly.area < 0.95:
                print(f"Warning: Island {i} has significant non-rectangular area")
            
            islands.append(Island(
                id=f"island_{i}",
                polygon=rect_poly
            ))
        
        return islands
    
    def _compute_island_semantics(
        self,
        islands: List[Island],
        core: CoreTube,
        entrance: Tuple[float, float]
    ):
        """计算岛屿语义属性"""
        
        entrance_point = Point(entrance)
        core_center = Point(core.center)
        
        for island in islands:
            x_min, y_min, x_max, y_max = island.bounds
            island_center = Point(island.centroid)
            
            # 外墙方向
            tol = 0.5
            island.exterior_walls = []
            if abs(x_min - self.x_min) < tol:
                island.exterior_walls.append('west')
            if abs(x_max - self.x_max) < tol:
                island.exterior_walls.append('east')
            if abs(y_min - self.y_min) < tol:
                island.exterior_walls.append('south')
            if abs(y_max - self.y_max) < tol:
                island.exterior_walls.append('north')
            
            island.has_exterior_wall = len(island.exterior_walls) > 0
            
            # 距离
            island.distance_to_entrance = island_center.distance(entrance_point)
            island.distance_to_core = island_center.distance(core_center)
            
            # 推荐分区
            island.suggested_zone = self._suggest_zone(island)
    
    def _suggest_zone(self, island: Island) -> ZoneType:
        """推荐功能分区"""
        
        # 无外墙 → 服务区（设备、储藏）
        if not island.has_exterior_wall:
            return ZoneType.SERVICE
        
        # 靠近入口 → 公共区
        avg_distance = (self.floor_width + self.floor_depth) / 4
        if island.distance_to_entrance < avg_distance:
            return ZoneType.PUBLIC
        
        # 远离入口 + 有外墙 → 私密区
        return ZoneType.PRIVATE
    
    def _validate(self, islands: List[Island]):
        """验证生成结果"""
        non_rect = [i.id for i in islands if not i.is_rectangular]
        if non_rect:
            print(f"⚠️ Non-rectangular islands: {non_rect}")
        
        total_island_area = sum(i.area for i in islands)
        coverage = total_island_area / self.floor.area
        print(f"✓ Generated {len(islands)} islands, coverage: {coverage:.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def generate_rectangular_topology(
    floor_boundary: Polygon,
    corridor_width: float = 2.0,
    core_area_ratio: float = 0.08,
    corridor_layout: str = 'cross',
    entrance_position: Optional[Tuple[float, float]] = None
) -> Tuple[CoreTube, List[Corridor], List[Island]]:
    """
    便捷函数：生成矩形拓扑
    
    返回:
        (核心筒, 走廊列表, 岛屿列表)
    """
    generator = RectangularTopologyGenerator(
        floor_boundary=floor_boundary,
        corridor_width=corridor_width
    )
    
    core = CoreTube.create_for_floor(
        floor_boundary.bounds,
        area_ratio=core_area_ratio
    )
    
    return generator.generate(
        core_tube=core,
        corridor_layout=corridor_layout,
        entrance_position=entrance_position
    )
```

---

## 📋 Phase 3: 房间-岛屿分配

### 3.1 数据结构

```python
# room_spec.py

"""
房间规格定义（语义增强版）
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ZoneType(Enum):
    """功能分区"""
    PUBLIC = "public"
    PRIVATE = "private"
    SERVICE = "service"
    CIRCULATION = "circulation"


@dataclass
class RoomSpec:
    """
    房间规格
    
    基础属性:
        room_id: 唯一标识
        room_type: 房间类型（living_room, bedroom, ...）
        target_area: 目标面积 (m²)
    
    几何约束:
        min_width: 最小宽度 (m)
        min_depth: 最小进深 (m)
        aspect_ratio_range: 宽高比范围 (min, max)
    
    语义约束:
        zone: 功能分区
        needs_window: 是否需要采光
        adjacency_required: 必须相邻的房间
        adjacency_preferred: 最好相邻的房间
        adjacency_forbidden: 禁止相邻的房间
    """
    # 基础属性
    room_id: str
    room_type: str
    target_area: float
    
    # 几何约束
    min_width: float = 2.5
    min_depth: float = 2.5
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    
    # 语义约束
    zone: ZoneType = ZoneType.PUBLIC
    needs_window: bool = False
    adjacency_required: List[str] = field(default_factory=list)
    adjacency_preferred: List[str] = field(default_factory=list)
    adjacency_forbidden: List[str] = field(default_factory=list)
    
    # 优先级
    priority: float = 1.0  # 面积满足优先级
```

### 3.2 分配算法

```python
# island_room_assigner.py

"""
房间-岛屿分配器

实现分层决策：
1. 硬约束过滤（采光、面积）
2. 功能分区匹配
3. 邻接约束优化
4. 面积平衡填充
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from .room_spec import RoomSpec, ZoneType
from .topology_generator import Island


@dataclass
class AssignmentResult:
    """分配结果"""
    island_id: str
    rooms: List[RoomSpec]
    total_area: float
    utilization: float  # 利用率


class IslandRoomAssigner:
    """
    房间-岛屿分配器
    
    分配策略：
    1. 硬约束过滤：排除不满足条件的岛屿
    2. 功能分区匹配：公共→公共岛，私密→私密岛
    3. 邻接约束优化：相关房间尽量同岛
    4. 面积平衡填充：最大化利用率
    """
    
    def __init__(
        self,
        islands: List[Island],
        rooms: List[RoomSpec],
        adjacency_graph: Dict[str, List[str]]
    ):
        self.islands = {i.id: i for i in islands}
        self.rooms = {r.room_id: r for r in rooms}
        self.adjacency = adjacency_graph
        
        # 分配结果
        self.assignments: Dict[str, List[str]] = defaultdict(list)
        self.room_to_island: Dict[str, str] = {}
    
    def assign(self) -> Dict[str, AssignmentResult]:
        """
        执行分配
        
        返回:
            {island_id: AssignmentResult}
        """
        # Step 1: 按优先级排序房间（大面积/高优先级优先）
        sorted_rooms = self._sort_rooms()
        
        # Step 2: 逐个分配房间
        for room in sorted_rooms:
            # 2.1 获取候选岛屿
            candidates = self._get_candidate_islands(room)
            
            if not candidates:
                print(f"⚠️ No valid island for room: {room.room_id}")
                continue
            
            # 2.2 评分并选择最佳岛屿
            best_island = self._select_best_island(room, candidates)
            
            # 2.3 分配
            self._assign_room(room, best_island)
        
        # Step 3: 生成结果
        return self._build_results()
    
    def _sort_rooms(self) -> List[RoomSpec]:
        """
        排序房间（分配优先级）
        
        规则：
        1. needs_window 的房间优先（外墙岛屿有限）
        2. adjacency_required 多的优先（约束强）
        3. 面积大的优先（选择余地小）
        """
        def priority_key(room: RoomSpec) -> Tuple:
            return (
                -int(room.needs_window),  # 需要窗的优先
                -len(room.adjacency_required),  # 约束多的优先
                -room.target_area,  # 面积大的优先
                -room.priority
            )
        
        return sorted(self.rooms.values(), key=priority_key)
    
    def _get_candidate_islands(self, room: RoomSpec) -> List[Island]:
        """
        获取候选岛屿（通过硬约束过滤）
        """
        candidates = []
        
        for island in self.islands.values():
            # 约束 1: 面积足够
            if island.remaining_capacity < room.target_area * 0.85:
                continue
            
            # 约束 2: 采光需求
            if room.needs_window and not island.has_exterior_wall:
                continue
            
            # 约束 3: 禁止邻接（如果同岛已有禁止邻接的房间）
            if self._has_forbidden_neighbor(room, island):
                continue
            
            candidates.append(island)
        
        return candidates
    
    def _has_forbidden_neighbor(self, room: RoomSpec, island: Island) -> bool:
        """检查岛屿内是否有禁止邻接的房间"""
        assigned_rooms = self.assignments.get(island.id, [])
        for assigned_id in assigned_rooms:
            if assigned_id in room.adjacency_forbidden:
                return True
            assigned_room = self.rooms.get(assigned_id)
            if assigned_room and room.room_id in assigned_room.adjacency_forbidden:
                return True
        return False
    
    def _select_best_island(
        self, 
        room: RoomSpec, 
        candidates: List[Island]
    ) -> Island:
        """
        选择最佳岛屿
        
        评分规则：
        1. 功能分区匹配（+50分）
        2. 已有必须邻接的房间（+30分/个）
        3. 已有偏好邻接的房间（+10分/个）
        4. 剩余容量匹配（避免浪费）
        """
        scores = []
        
        for island in candidates:
            score = 0
            
            # 1. 功能分区匹配
            if island.suggested_zone.value == room.zone.value:
                score += 50
            
            # 2. 必须邻接的房间
            assigned = set(self.assignments.get(island.id, []))
            for adj_id in room.adjacency_required:
                if adj_id in assigned:
                    score += 30
            
            # 3. 偏好邻接的房间
            for adj_id in room.adjacency_preferred:
                if adj_id in assigned:
                    score += 10
            
            # 4. 容量匹配（利用率 70-90% 最佳）
            utilization_after = (
                (island.area - island.remaining_capacity + room.target_area) 
                / island.area
            )
            if 0.7 <= utilization_after <= 0.9:
                score += 20
            elif utilization_after > 0.95:
                score -= 10  # 太满，可能放不下
            
            # 5. 邻接关系的对称考虑
            for assigned_id in assigned:
                assigned_room = self.rooms.get(assigned_id)
                if assigned_room:
                    if room.room_id in assigned_room.adjacency_required:
                        score += 30
                    if room.room_id in assigned_room.adjacency_preferred:
                        score += 10
            
            scores.append((island, score))
        
        # 返回最高分的岛屿
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]
    
    def _assign_room(self, room: RoomSpec, island: Island):
        """分配房间到岛屿"""
        self.assignments[island.id].append(room.room_id)
        self.room_to_island[room.room_id] = island.id
        island.remaining_capacity -= room.target_area
        island.assigned_rooms.append(room.room_id)
    
    def _build_results(self) -> Dict[str, AssignmentResult]:
        """构建分配结果"""
        results = {}
        
        for island_id, room_ids in self.assignments.items():
            island = self.islands[island_id]
            rooms = [self.rooms[rid] for rid in room_ids]
            total_area = sum(r.target_area for r in rooms)
            
            results[island_id] = AssignmentResult(
                island_id=island_id,
                rooms=rooms,
                total_area=total_area,
                utilization=total_area / island.area
            )
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def assign_rooms_to_islands(
    islands: List[Island],
    rooms: List[RoomSpec],
    adjacency_graph: Dict[str, List[str]] = None
) -> Dict[str, AssignmentResult]:
    """
    便捷函数：房间-岛屿分配
    """
    if adjacency_graph is None:
        # 从 RoomSpec 构建邻接图
        adjacency_graph = {}
        for room in rooms:
            adjacency_graph[room.room_id] = (
                room.adjacency_required + room.adjacency_preferred
            )
    
    assigner = IslandRoomAssigner(islands, rooms, adjacency_graph)
    return assigner.assign()
```

---

## 📋 Phase 4: 补充优化

### 4.1 layout_generator.py 集成

```python
# layout_generator.py - 主入口更新

"""
布局生成器（重构版）

完整流程：
1. LLM 生成房间规格
2. 矩形拓扑生成
3. 房间-岛屿分配
4. 岛屿内房间划分（Treemap + MIQP）
5. 结果合并与验证
"""

from typing import List, Dict, Optional, Tuple
from shapely.geometry import Polygon

from .room_spec import RoomSpec, ZoneType
from .topology_generator import (
    RectangularTopologyGenerator, 
    CoreTube, 
    Corridor, 
    Island,
    generate_rectangular_topology
)
from .island_room_assigner import assign_rooms_to_islands, AssignmentResult
from .island_partition_solver import partition_island, RoomResult
from .constraint_validator import validate_layout


class LayoutGenerator:
    """
    布局生成器
    """
    
    def __init__(
        self,
        floor_boundary: Polygon,
        corridor_width: float = 2.0,
        core_area_ratio: float = 0.08
    ):
        self.floor = floor_boundary
        self.corridor_width = corridor_width
        self.core_area_ratio = core_area_ratio
    
    def generate(
        self,
        rooms: List[RoomSpec],
        adjacency_graph: Dict[str, List[str]] = None,
        entrance_position: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        生成布局
        
        参数:
            rooms: 房间规格列表
            adjacency_graph: 邻接关系图
            entrance_position: 入口位置
        
        返回:
            {
                'core_tube': CoreTube,
                'corridors': List[Corridor],
                'islands': List[Island],
                'room_layouts': List[RoomResult],
                'validation': ValidationReport
            }
        """
        # Phase 2: 拓扑生成
        core_tube, corridors, islands = generate_rectangular_topology(
            floor_boundary=self.floor,
            corridor_width=self.corridor_width,
            core_area_ratio=self.core_area_ratio,
            entrance_position=entrance_position
        )
        
        # Phase 3: 房间-岛屿分配
        assignments = assign_rooms_to_islands(
            islands=islands,
            rooms=rooms,
            adjacency_graph=adjacency_graph
        )
        
        # Phase 4: 岛屿内划分
        all_room_layouts = []
        for island_id, assignment in assignments.items():
            island = next(i for i in islands if i.id == island_id)
            
            # 构建邻接子图（仅该岛屿内的房间）
            island_room_ids = set(r.room_id for r in assignment.rooms)
            island_adjacency = {
                rid: [a for a in adj if a in island_room_ids]
                for rid, adj in (adjacency_graph or {}).items()
                if rid in island_room_ids
            }
            
            # 调用 Treemap + MIQP
            room_layouts = partition_island(
                island_polygon=island.polygon,
                rooms=assignment.rooms,
                adjacency_graph=island_adjacency,
                exterior_walls=island.exterior_walls
            )
            
            all_room_layouts.extend(room_layouts)
        
        # 验证
        validation = validate_layout(
            rooms=rooms,
            layouts=all_room_layouts,
            floor_boundary=self.floor
        )
        
        return {
            'core_tube': core_tube,
            'corridors': corridors,
            'islands': islands,
            'room_layouts': all_room_layouts,
            'validation': validation
        }


# ═══════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def generate_layout(
    floor_boundary: Polygon,
    rooms: List[RoomSpec],
    adjacency_graph: Dict[str, List[str]] = None,
    **kwargs
) -> Dict:
    """便捷函数：生成布局"""
    generator = LayoutGenerator(floor_boundary, **kwargs)
    return generator.generate(rooms, adjacency_graph)
```

### 4.2 __init__.py 更新

```python
# __init__.py

"""
Geometry 模块导出
"""

# 数据结构
from .room_spec import RoomSpec, ZoneType

# 拓扑生成
from .topology_generator import (
    CoreTube,
    Corridor,
    Island,
    RectangularTopologyGenerator,
    generate_rectangular_topology
)

# 房间-岛屿分配
from .island_room_assigner import (
    AssignmentResult,
    IslandRoomAssigner,
    assign_rooms_to_islands
)

# 岛屿划分
from .island_partition_solver import (
    RoomResult,
    IslandPartitionSolver,
    partition_island
)

# 布局生成
from .layout_generator import (
    LayoutGenerator,
    generate_layout
)

# 验证
from .constraint_validator import (
    ValidationReport,
    validate_layout
)

# 工具
from .axis_align import snap_to_grid


__all__ = [
    # 数据结构
    'RoomSpec',
    'ZoneType',
    
    # 拓扑
    'CoreTube',
    'Corridor', 
    'Island',
    'RectangularTopologyGenerator',
    'generate_rectangular_topology',
    
    # 分配
    'AssignmentResult',
    'IslandRoomAssigner',
    'assign_rooms_to_islands',
    
    # 划分
    'RoomResult',
    'IslandPartitionSolver',
    'partition_island',
    
    # 生成
    'LayoutGenerator',
    'generate_layout',
    
    # 验证
    'ValidationReport',
    'validate_layout',
    
    # 工具
    'snap_to_grid',
]
```

---

## 📋 Phase 5: 补充优化建议

### 5.1 性能优化

```python
# 建议添加到 island_partition_solver.py

class IslandPartitionSolver:
    """增加的优化"""
    
    def solve(self, ...):
        # 优化 1: 小岛屿直接用 Treemap，跳过 MIQP
        if len(self.rooms) <= 2:
            return self._treemap_only()
        
        # 优化 2: 大岛屿使用 coarse-to-fine
        if len(self.rooms) > 10:
            return self._coarse_to_fine_solve()
        
        # 正常流程
        return self._standard_solve()
    
    def _coarse_to_fine_solve(self):
        """
        粗到细求解（借鉴 Co-Layout）
        
        Step 1: 低精度快速求解（SCALE=10）
        Step 2: 高精度精修（SCALE=100）
        """
        # 粗糙求解
        coarse_result = self._solve_cpsat(scale=10, time_limit=5)
        
        # 精细求解（用粗糙结果作为 warm start）
        fine_result = self._solve_cpsat(
            scale=100, 
            time_limit=30,
            warm_start=coarse_result
        )
        
        return fine_result
```

### 5.2 错误处理

```python
# 建议添加异常处理

class LayoutGenerationError(Exception):
    """布局生成错误基类"""
    pass

class TopologyError(LayoutGenerationError):
    """拓扑生成错误"""
    pass

class AssignmentError(LayoutGenerationError):
    """分配错误"""
    pass

class PartitionError(LayoutGenerationError):
    """划分错误"""
    pass


# 在 LayoutGenerator.generate() 中使用
try:
    core_tube, corridors, islands = generate_rectangular_topology(...)
except Exception as e:
    raise TopologyError(f"Failed to generate topology: {e}")

try:
    assignments = assign_rooms_to_islands(...)
except Exception as e:
    raise AssignmentError(f"Failed to assign rooms: {e}")

try:
    room_layouts = partition_island(...)
except Exception as e:
    # 降级：使用 Treemap-only
    print(f"Warning: MIQP failed, falling back to Treemap: {e}")
    room_layouts = treemap_only(...)
```

### 5.3 日志和调试

```python
# 建议添加日志

import logging

logger = logging.getLogger(__name__)

class LayoutGenerator:
    def generate(self, ...):
        logger.info(f"Starting layout generation: {len(rooms)} rooms")
        
        # Phase 2
        logger.info("Phase 2: Generating topology...")
        core_tube, corridors, islands = generate_rectangular_topology(...)
        logger.info(f"  Generated {len(islands)} rectangular islands")
        
        # Phase 3
        logger.info("Phase 3: Assigning rooms to islands...")
        assignments = assign_rooms_to_islands(...)
        for island_id, result in assignments.items():
            logger.info(f"  {island_id}: {len(result.rooms)} rooms, {result.utilization:.1%} utilization")
        
        # Phase 4
        logger.info("Phase 4: Partitioning islands...")
        # ...
        
        logger.info(f"Layout generation complete: {len(all_room_layouts)} room layouts")
        return result
```

### 5.4 测试文件

```python
# tests/test_layout_pipeline.py

"""
布局生成流水线测试
"""

import pytest
from shapely.geometry import box

from backend.core.geometry import (
    RoomSpec, ZoneType,
    generate_rectangular_topology,
    assign_rooms_to_islands,
    partition_island,
    generate_layout
)


class TestTopologyGeneration:
    """拓扑生成测试"""
    
    def test_rectangular_islands(self):
        """所有岛屿应该是矩形"""
        floor = box(0, 0, 50, 30)
        core, corridors, islands = generate_rectangular_topology(floor)
        
        for island in islands:
            assert island.is_rectangular, f"Island {island.id} is not rectangular"
    
    def test_exterior_walls(self):
        """边缘岛屿应该有外墙"""
        floor = box(0, 0, 50, 30)
        core, corridors, islands = generate_rectangular_topology(floor)
        
        exterior_count = sum(1 for i in islands if i.has_exterior_wall)
        assert exterior_count >= 2, "Should have at least 2 islands with exterior walls"


class TestRoomAssignment:
    """房间分配测试"""
    
    def test_window_constraint(self):
        """需要窗户的房间应该分配到有外墙的岛屿"""
        floor = box(0, 0, 50, 30)
        core, corridors, islands = generate_rectangular_topology(floor)
        
        rooms = [
            RoomSpec('bedroom', 'bedroom', 20, needs_window=True),
            RoomSpec('storage', 'storage', 5, needs_window=False),
        ]
        
        assignments = assign_rooms_to_islands(islands, rooms)
        
        for island_id, result in assignments.items():
            island = next(i for i in islands if i.id == island_id)
            for room in result.rooms:
                if room.needs_window:
                    assert island.has_exterior_wall
    
    def test_adjacency_same_island(self):
        """有邻接需求的房间应该尽量在同一岛屿"""
        floor = box(0, 0, 50, 30)
        core, corridors, islands = generate_rectangular_topology(floor)
        
        rooms = [
            RoomSpec('master_br', 'bedroom', 20, 
                    adjacency_required=['master_bath']),
            RoomSpec('master_bath', 'bathroom', 8,
                    adjacency_required=['master_br']),
        ]
        
        assignments = assign_rooms_to_islands(islands, rooms)
        
        # 找到两个房间的岛屿
        br_island = None
        bath_island = None
        for island_id, result in assignments.items():
            for room in result.rooms:
                if room.room_id == 'master_br':
                    br_island = island_id
                if room.room_id == 'master_bath':
                    bath_island = island_id
        
        assert br_island == bath_island, "Adjacent rooms should be in same island"


class TestIslandPartition:
    """岛屿划分测试"""
    
    def test_no_overlap(self):
        """房间不应该重叠"""
        island = box(0, 0, 20, 15)
        rooms = [
            RoomSpec('r1', 'room', 50),
            RoomSpec('r2', 'room', 50),
            RoomSpec('r3', 'room', 50),
        ]
        
        results = partition_island(island, rooms, {}, ['south', 'east'])
        
        for i, r1 in enumerate(results):
            for r2 in results[i+1:]:
                assert not r1.polygon.intersects(r2.polygon) or \
                       r1.polygon.intersection(r2.polygon).area < 0.01
    
    def test_aspect_ratio(self):
        """房间宽高比应该在合理范围内"""
        island = box(0, 0, 20, 15)
        rooms = [
            RoomSpec('r1', 'room', 100, aspect_ratio_range=(0.5, 2.0)),
        ]
        
        results = partition_island(island, rooms, {}, ['south'])
        
        for r in results:
            ar = r.width / r.depth
            assert 0.4 <= ar <= 2.5, f"Aspect ratio {ar} out of range"


class TestEndToEnd:
    """端到端测试"""
    
    def test_full_pipeline(self):
        """完整流水线测试"""
        floor = box(0, 0, 50, 30)
        
        rooms = [
            RoomSpec('living', 'living_room', 35, 
                    zone=ZoneType.PUBLIC, needs_window=True,
                    adjacency_required=['dining']),
            RoomSpec('dining', 'dining_room', 15,
                    zone=ZoneType.PUBLIC,
                    adjacency_required=['living', 'kitchen']),
            RoomSpec('kitchen', 'kitchen', 12,
                    zone=ZoneType.PUBLIC,
                    adjacency_required=['dining']),
            RoomSpec('master_br', 'bedroom', 20,
                    zone=ZoneType.PRIVATE, needs_window=True,
                    adjacency_required=['master_bath']),
            RoomSpec('master_bath', 'bathroom', 8,
                    zone=ZoneType.PRIVATE,
                    adjacency_required=['master_br']),
        ]
        
        adjacency = {
            'living': ['dining'],
            'dining': ['living', 'kitchen'],
            'kitchen': ['dining'],
            'master_br': ['master_bath'],
            'master_bath': ['master_br'],
        }
        
        result = generate_layout(floor, rooms, adjacency)
        
        # 验证
        assert len(result['room_layouts']) == len(rooms)
        assert result['validation'].is_valid
        
        # 检查面积
        for room_spec in rooms:
            layout = next(
                r for r in result['room_layouts'] 
                if r.room_id == room_spec.room_id
            )
            area_error = abs(layout.area - room_spec.target_area) / room_spec.target_area
            assert area_error < 0.2, f"Area error {area_error:.1%} for {room_spec.room_id}"
```

---

## 📊 验收标准

| 维度 | 标准 |
|------|------|
| **代码清理** | 废弃文件已删除/归档，无引用错误 |
| **矩形岛屿** | 100% 岛屿为矩形（`is_rectangular = True`）|
| **房间分配** | 采光约束 100% 满足，邻接约束 >90% 满足 |
| **布局质量** | 面积偏差 <15%，无重叠，正交性 100% |
| **性能** | 10 房间 <5s，15 房间 <15s |
| **测试** | 所有测试通过 |

---

## 🚀 执行顺序

```
Week 1:
├── Day 1-2: Phase 1 清理冗余代码
│   ├── 检查引用
│   ├── 移动废弃文件
│   └── 更新导入
│
├── Day 3-4: Phase 2 拓扑重写
│   ├── 新建 topology_generator.py
│   └── 测试矩形岛屿生成
│
└── Day 5: Phase 3 房间分配
    ├── 新建 room_spec.py
    └── 新建 island_room_assigner.py

Week 2:
├── Day 1-2: Phase 4 集成
│   ├── 更新 layout_generator.py
│   └── 更新 __init__.py
│
├── Day 3-4: Phase 5 优化
│   ├── 性能优化
│   ├── 错误处理
│   └── 日志
│
└── Day 5: 测试和调优
    ├── 编写测试
    ├── 运行测试
    └── 性能调优
```