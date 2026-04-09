# Building 管线鲁棒性根治方案（终版）

## Context

Building 管线反复崩溃，根因不是单个 bug 而是架构级脆弱性：

| 对比 | Parking（可靠） | Building（脆弱） |
|---|---|---|
| LLM 输出字段数 | 7 个/元素（t,x,y,w,h,id,r） | 12 个/房间 + 跨层引用 |
| JSON 修复 | safe_parse_response() + json_repair + 多策略链 | 弱修复，Pydantic 前预处理不足 |
| 重试策略 | 3 次 + 具体错误反馈 + 迭代修复循环 | 2 次 + 模糊重试提示 |
| 验证后恢复 | run_iterative_fix() 违规反馈 → LLM 修 → 自动补丁 | 无 |
| 几何容错 | N/A（LLM 直出坐标） | 0 岛屿崩溃 / 面积不匹配崩溃 / 分配失败崩溃 |

核心原则：不逐个修 bug，而是在每个环节加防御层，让管线永不崩溃。

借鉴来源：Co-Layout（AAAI 2026）— LLM 多 Agent 分步提取 + IP 求解器 + 后处理生成墙/门/窗。

---

## Phase 1: LLM Prompt 改造（借鉴 Co-Layout 多 Agent 分步 + 只出约束）

文件: `backend/core/prompts/building_prompts.py`, `backend/core/flows/building_semantic_flow.py`

### Step 1.1 — LLM 拆为两步调用（借鉴 Co-Layout 多 Agent 分步提取）

Co-Layout 用 5 个专职 Agent 串行工作（基本信息→环境→空间→房间→家具），每步输出范围窄、出错率低。当前项目一个 prompt 让 LLM 同时输出所有楼层所有房间的完整配置，字段多、结构深，是崩溃高发区。

改为两步 LLM 调用，每步独立校验：

**Step 1（房间清单）— 只提取结构骨架：**

```
用户需求: "三层住宅，一楼客厅餐厅厨房，二楼三间卧室，三楼书房露台"

请输出每层的房间清单，仅包含以下字段：
{
  "floors": [
    {
      "floor_number": 1,
      "floor_name": "一层",
      "rooms": [
        { "room_id": "room_001", "room_name": "客厅", "room_type": "living_room", "zone": "public" }
      ]
    }
  ]
}
```

校验通过后进入 Step 2。

**Step 2（属性补全）— 在已确认的骨架上补充细节：**

```
以下是已确认的房间清单：[Step 1 输出]
请为每个房间补充以下字段：
- size_hint: "large" / "medium" / "small"（不需要精确面积）
- needs_window: true/false
- adjacency_required: 同层 room_id 列表
```

每步失败只重试该步，不需要从头来。

### Step 1.2 — LLM 只出约束，几何全交给求解器（借鉴 Co-Layout）

Co-Layout 中 LLM 完全不碰坐标和精确面积，只输出语义约束（"客厅要大、朝南、挨着餐厅"），面积数值由求解器根据楼层总面积和房间数自动分配。

改动：
- `target_area` 从 LLM 必填字段中移除，替换为 `size_hint: "large" | "medium" | "small"`
- 后端新增 `apply_size_hints()` 函数，根据楼层总面积 + 房间数 + room_type 规则表自动计算精确面积

```python
SIZE_HINT_WEIGHTS = {"large": 3.0, "medium": 2.0, "small": 1.0}

def apply_size_hints(rooms: List[RoomAllocation], floor_area: float, core_tube_area: float) -> None:
    """将 size_hint 转为精确 target_area"""
    usable = floor_area - core_tube_area
    corridor_allowance = usable * 0.12  # 走廊预留
    distributable = usable - corridor_allowance
    total_weight = sum(SIZE_HINT_WEIGHTS.get(r.size_hint, 2.0) for r in rooms)
    for room in rooms:
        w = SIZE_HINT_WEIGHTS.get(room.size_hint, 2.0)
        room.target_area = distributable * (w / total_weight)
        # 叠加 room_type 约束
        defaults = ROOM_TYPE_DEFAULTS.get(room.room_type, {})
        room.target_area = max(room.target_area, defaults.get("min_area", 6.0))
```

### Step 1.3 — 精简最终 Schema 为 5 个字段

从 12 个字段砍到 5 个（比 Parking 的 7 个更少）：

```json
{
  "room_id": "room_001",
  "room_name": "客厅",
  "room_type": "living_room",
  "zone": "public",
  "needs_window": true
}
```

Step 2 额外补充 2 个：`size_hint`, `adjacency_required`。

删除的字段（后端自动推导）：
- `target_area` → 由 `size_hint` + `apply_size_hints()` 计算
- `min_width` → 由 `room_type` 查表（`ROOM_TYPE_DEFAULTS`）
- `aspect_ratio_range` → 同上
- `adjacency_preferred` → LLM 不擅长区分 required vs preferred，全部用 required
- `adjacency_forbidden` → 由 `room_type` 查表推导（见 Step 1.4）
- `weight` → 默认 5，由 `room_type` 微调

### Step 1.4 — 新建 adjacency_forbidden 规则表

上一版计划砍掉了 `adjacency_forbidden` 说"由 room_type 推导"，但没有定义规则表。补上：

```python
# backend/core/geometry/room_defaults.py

ADJACENCY_FORBIDDEN_RULES = {
    "kitchen":   ["bedroom", "study"],       # 油烟噪音
    "bathroom":  ["kitchen", "dining_room"], # 卫生隔离
    "utility":   ["living_room", "bedroom"], # 设备噪音
    "garage":    ["bedroom", "study"],       # 尾气噪音
}

def compute_adjacency_forbidden(rooms: List[RoomAllocation]) -> Dict[str, List[str]]:
    """根据 room_type 规则表计算每个房间的禁邻列表"""
    forbidden = {}
    room_type_map = {r.room_id: r.room_type for r in rooms}
    for room in rooms:
        rules = ADJACENCY_FORBIDDEN_RULES.get(room.room_type, [])
        forbidden[room.room_id] = [
            rid for rid, rtype in room_type_map.items()
            if rtype in rules and rid != room.room_id
        ]
    return forbidden
```

### Step 1.5 — Prompt 中加入强约束提示

在 schema 后添加硬性格式约束（参考 parking prompt 的风格）：

```
【格式要求（严格）】
1. 输出纯 JSON，禁止 markdown/解释/注释
2. room_id 格式：room_001, room_002, ...（同层连续编号）
3. zone 只能是：public / private / service / circulation
4. adjacency_required 只引用同层的 room_id
5. size_hint 只能是：large / medium / small
```

---

## Phase 2: 强化解析层（永不崩溃的 JSON → BuildingAllocation）

文件: `backend/core/flows/building_semantic_flow.py`

### Step 2.1 — 重写 `_parse_building_allocation()` 为多层防御

解析链：`json.loads → json_repair → 正则修复 → 字段补全 → 类型修复 → 枚举修复 → 引用清洗 → Pydantic`

关键新增：Pydantic 前的智能预处理

对每个 room dict：
- 缺失 `room_id` → 自动生成 `F{层号}_R{序号}`
- 缺失 `zone` → 默认 `"public"`
- 缺失 `needs_window` → 默认 `true`（安全侧）
- `target_area` 是字符串 → `float()` 转换
- `weight` 是字符串 → `int()` 转换
- `zone` 不在枚举内 → 映射到最近值（`"common"→"public"`, `"bedroom"→"private"`）
- `aspect_ratio_range` 格式错误 → 用默认值 `[0.5, 2.0]`
- `adjacency_required` 中引用不存在的 `room_id` → 移除 **并记录 warning 到响应**（非静默丢弃）

对每个 floor dict：
- 缺失 `floor_number` → 用遍历序号
- 缺失 `floor_function_tag` → 默认 `"standard"`
- 面积字段缺失 → 从 rooms 反推

**旧字段兼容逻辑（修正）：**

```python
# 仅当新字段未被显式设置时才回退到旧字段
if room.needs_window is None and room.requires_window is not None:
    room.needs_window = room.requires_window

if not room.adjacency_required and room.adjacency_tags:
    room.adjacency_required = room.adjacency_tags
```

注意：不能用 `not room.needs_window`，因为 `needs_window=False` 是有效的显式值，不应被旧字段覆盖。

### Step 2.2 — 非法值处理（补充负面路径）

上一版只测了"缺失字段"和"正确格式"，缺少非法值处理：

```python
# zone 非法值映射
ZONE_FUZZY_MAP = {
    "common": "public", "shared": "public", "living": "public",
    "bedroom": "private", "sleeping": "private", "personal": "private",
    "utility": "service", "storage": "service", "mechanical": "service",
    "hallway": "circulation", "corridor": "circulation",
}

def _fix_zone(zone_str: str) -> str:
    zone_str = zone_str.lower().strip()
    if zone_str in ("public", "private", "service", "circulation"):
        return zone_str
    mapped = ZONE_FUZZY_MAP.get(zone_str)
    if mapped:
        return mapped
    logger.warning(f"Unknown zone '{zone_str}', defaulting to 'public'")
    return "public"

# room_type 非法值
ROOM_TYPE_FUZZY_MAP = {
    "toilet": "bathroom", "washroom": "bathroom", "wc": "bathroom",
    "lounge": "living_room", "parlor": "living_room",
    "pantry": "kitchen", "cooking": "kitchen",
}
```

### Step 2.3 — 升级重试逻辑为 3 次 + 具体错误反馈

```python
for attempt in range(3):  # 从 2 → 3
    if attempt > 0:
        # 将具体 Pydantic 错误反馈给 LLM
        error_msg = _extract_validation_errors(last_error)
        messages.append(ChatMessage(role="user", content=error_msg))
```

`_extract_validation_errors()` 从 Pydantic `ValidationError` 提取字段级错误：

```
字段校验失败：
  • floors -> 0 -> rooms -> 2 -> target_area: value is not a valid float
  • floors -> 1 -> rooms -> 0 -> zone: unexpected value; permitted: 'public', 'private', ...
```

---

## Phase 3: 几何层永不崩溃

文件: `backend/core/geometry/island_room_assigner.py`, `backend/core/geometry/layout_generator.py`

### Step 3.1 — 分配器容错：从"抛异常"变为"尽力分配"

核心改造：`assign()` 方法永不抛 `AssignmentError`，改为 4 级降级：
1. 正常分配 → 评分选最佳岛
2. 候选为空 → 放宽到任何有剩余容量的岛
3. 仍为空 → 强制分配到最大岛，缩小房间面积
4. 完全无岛 → 跳过该房间，记录 warning

**第 3 级降级增加面积下限（修正）：**

```python
def _force_find_any_island(self, room: RoomSpec) -> Optional[IslandCandidate]:
    """强制分配到最大岛，但不允许缩放到无意义面积"""
    largest = max(self.islands, key=lambda i: i.remaining_area, default=None)
    if not largest:
        return None
    min_viable_area = room.min_width * room.min_depth  # 最小可用面积
    if largest.remaining_area < min_viable_area:
        logger.warning(f"Room {room.room_id}: largest island ({largest.remaining_area:.1f}m²) "
                       f"< min viable ({min_viable_area:.1f}m²), skipping")
        return None  # 走第 4 级：跳过
    room.target_area = min(room.target_area, largest.remaining_area * 0.9)
    return largest
```

### Step 3.2 — `generate_layout_v2()` 容错包装 + treemap fallback 实现

```python
try:
    miqp_results = partition_island_semantic(...)
except Exception as e:
    logger.warning(f"MIQP failed for island {island_id}: {e}, using treemap fallback")
    miqp_results = _treemap_fallback(island, assignment.rooms)
```

**补充 `_treemap_fallback()` 的具体实现（上一版缺失）：**

```python
def _treemap_fallback(island: Polygon, rooms: List[RoomSpec]) -> List[RoomResult]:
    """当 MIQP 求解失败时，直接用 treemap 切分结果作为最终输出。
    
    质量预期：不保证邻接关系和最小宽度约束，但保证：
    - 所有房间都有分配
    - 面积比例大致正确
    - 不崩溃
    """
    from backend.core.geometry.treemap import compute_treemap
    
    areas = [r.target_area for r in rooms]
    rects = compute_treemap(island.bounds, areas)
    
    results = []
    for room, rect in zip(rooms, rects):
        x, y, w, h = rect
        poly = box(x, y, x + w, y + h)
        # 裁剪到岛屿边界
        clipped = poly.intersection(island)
        if clipped.is_empty:
            continue
        results.append(RoomResult(
            room_id=room.room_id,
            room_type=room.room_type,
            polygon=clipped,
            target_area=room.target_area,
            actual_area=clipped.area,
        ))
    return results
```

### Step 3.3 — 动态缩放上限调整

当前缩放上限 3x 太高会导致单个房间超过岛屿容量。改为：

```python
max_scale = min(3.0, max_single_island_area / max_room_area)
```

### Step 3.4 — 布局连通性检查（新增，借鉴 Co-Layout flow-based 连通性）

Co-Layout 用 flow-based 模型保证走廊连通。短期不需要完整实现 flow 约束，但在后处理阶段加连通性检查：

```python
def check_connectivity(rooms: List[RoomResult], core_tube: CoreTube) -> List[str]:
    """检查每个房间是否通过邻接关系可达入口/核心筒。
    
    用 BFS 从核心筒出发，沿 shared-edge 关系遍历。
    返回不可达房间的 room_id 列表。
    """
    # 构建邻接图：两个房间的 polygon 共享边（buffer 容差 0.1m）→ 相邻
    adj = defaultdict(set)
    all_rooms = rooms + [RoomResult(room_id="_core", polygon=core_tube.polygon, ...)]
    
    for i, a in enumerate(all_rooms):
        for j, b in enumerate(all_rooms):
            if i >= j:
                continue
            shared = a.polygon.buffer(0.1).intersection(b.polygon.buffer(0.1))
            if shared.length > 0.5:  # 共享边 > 0.5m
                adj[a.room_id].add(b.room_id)
                adj[b.room_id].add(a.room_id)
    
    # BFS from core
    visited = set()
    queue = deque(["_core"])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj[node] - visited)
    
    unreachable = [r.room_id for r in rooms if r.room_id not in visited]
    if unreachable:
        logger.warning(f"Unreachable rooms: {unreachable}")
    return unreachable
```

不可达房间记录到响应的 `warnings` 中，不阻塞生成。

---

## Phase 4: 后处理自动生成墙/门/窗（借鉴 Co-Layout）

文件: `backend/core/geometry/postprocessor.py`（新建）

Co-Layout 的墙、门、窗不由 LLM 生成也不由求解器约束，而是求解完成后用启发式规则自动放置。借鉴此思路，在序列化层前新增后处理步骤。

### Step 4.1 — 自动推导墙体

```python
def generate_walls(rooms: List[RoomResult], floor_boundary: Polygon) -> List[WallSegment]:
    """根据房间 polygon 共享边自动生成墙体。"""
    walls = []
    
    # 外墙：房间 polygon 与楼层边界的共享边
    for room in rooms:
        shared = room.polygon.boundary.intersection(floor_boundary.boundary)
        if not shared.is_empty:
            walls.append(WallSegment(
                type="exterior_wall",
                geometry=shared,
                thickness=0.24,  # 240mm 外墙
                room_ids=[room.room_id],
            ))
    
    # 内墙：相邻房间 polygon 的共享边
    for i, a in enumerate(rooms):
        for j, b in enumerate(rooms):
            if i >= j:
                continue
            shared = a.polygon.boundary.intersection(b.polygon.boundary)
            if not shared.is_empty and shared.length > 0.3:
                walls.append(WallSegment(
                    type="partition_wall",
                    geometry=shared,
                    thickness=0.12,  # 120mm 隔墙
                    room_ids=[a.room_id, b.room_id],
                ))
    
    return walls
```

### Step 4.2 — 自动放置门

```python
def generate_doors(walls: List[WallSegment], rooms: List[RoomResult], 
                   connectivity_graph: Dict) -> List[DoorPlacement]:
    """在内墙上放置门，确保每个房间至少一个门。"""
    doors = []
    rooms_with_doors = set()
    
    for wall in walls:
        if wall.type != "partition_wall":
            continue
        if len(wall.room_ids) != 2:
            continue
        
        r1, r2 = wall.room_ids
        # 在共享边中点放置门
        midpoint = wall.geometry.interpolate(0.5, normalized=True)
        doors.append(DoorPlacement(
            position=midpoint,
            wall=wall,
            width=0.9,  # 900mm 标准门宽
            connects=[r1, r2],
        ))
        rooms_with_doors.update([r1, r2])
    
    return doors
```

### Step 4.3 — 自动放置窗户

```python
def generate_windows(walls: List[WallSegment], rooms: List[RoomResult]) -> List[WindowPlacement]:
    """在外墙上为 needs_window=True 的房间放置窗户。"""
    windows = []
    
    for wall in walls:
        if wall.type != "exterior_wall":
            continue
        room_id = wall.room_ids[0]
        room = next((r for r in rooms if r.room_id == room_id), None)
        if not room or not room.has_window:
            continue
        
        # 沿外墙均匀放置窗户（每 2m 一个）
        wall_length = wall.geometry.length
        num_windows = max(1, int(wall_length / 2.0))
        for k in range(num_windows):
            pos = wall.geometry.interpolate((k + 0.5) / num_windows, normalized=True)
            windows.append(WindowPlacement(
                position=pos,
                wall=wall,
                width=1.2,  # 1200mm 标准窗宽
                room_id=room_id,
            ))
    
    return windows
```

### Step 4.4 — 数据结构定义

```python
@dataclass
class WallSegment:
    type: str               # "exterior_wall" | "partition_wall"
    geometry: BaseGeometry   # LineString / MultiLineString
    thickness: float         # 米
    room_ids: List[str]

@dataclass
class DoorPlacement:
    position: Point
    wall: WallSegment
    width: float
    connects: List[str]     # 连接的两个 room_id

@dataclass
class WindowPlacement:
    position: Point
    wall: WallSegment
    width: float
    room_id: str
```

### Step 4.5 — 集成到序列化层

在 `serializers.py` 的 `building_result_to_dict()` 中调用后处理：

```python
def building_result_to_dict(result, floor_boundary):
    # ... 现有序列化 ...
    
    # 后处理：生成墙/门/窗
    for floor_id, layout in result.floor_layouts.items():
        walls = generate_walls(layout.room_layouts, floor_boundary)
        doors = generate_doors(walls, layout.room_layouts, {})
        windows = generate_windows(walls, layout.room_layouts)
        
        floors[floor_id]["walls"] = [wall_to_dict(w) for w in walls]
        floors[floor_id]["doors"] = [door_to_dict(d) for d in doors]
        floors[floor_id]["windows"] = [window_to_dict(w) for w in windows]
    
    return response
```

---

## Phase 5: 可观测性 — 降级摘要（新增）

上一版到处加了 `logger.warning` 和静默降级，但没有说明 warning 怎么汇总返回前端。

### Step 5.1 — 降级事件收集器

```python
class DegradationCollector:
    """收集管线执行过程中的所有降级事件"""
    def __init__(self):
        self.events: List[Dict] = []
    
    def record(self, phase: str, event_type: str, detail: str):
        self.events.append({"phase": phase, "type": event_type, "detail": detail})
    
    def summary(self) -> Dict:
        return {
            "total_degradations": len(self.events),
            "skipped_rooms": [e["detail"] for e in self.events if e["type"] == "room_skipped"],
            "miqp_fallback_floors": [e["detail"] for e in self.events if e["type"] == "miqp_fallback"],
            "adjacency_dropped": len([e for e in self.events if e["type"] == "adjacency_dropped"]),
            "unreachable_rooms": [e["detail"] for e in self.events if e["type"] == "unreachable"],
            "parse_fixes": len([e for e in self.events if e["type"] == "parse_fix"]),
        }
```

### Step 5.2 — API 响应增加降级摘要

```json
{
  "building": { ... },
  "core_tube": { ... },
  "warnings": ["..."],
  "degradation_summary": {
    "total_degradations": 3,
    "skipped_rooms": [],
    "miqp_fallback_floors": ["F2"],
    "adjacency_dropped": 2,
    "unreachable_rooms": ["F1_storage"],
    "parse_fixes": 1
  }
}
```

前端可根据 `total_degradations > 0` 显示提示条，让用户知道布局质量可能受影响。

---

## Phase 6: 端到端测试

文件: `backend/tests/test_building_robustness.py`（新建）

### 测试矩阵

| 测试 | 模拟场景 |
|---|---|
| test_malformed_json_repair | 尾逗号、注释、截断、markdown 包裹 |
| test_missing_fields_filled | 缺 zone/room_id/weight → 自动补全 |
| test_invalid_zone_mapped | zone="common" → "public"、zone="semi-public" → "public" |
| test_invalid_room_type_mapped | room_type="toilet" → "bathroom" |
| test_invalid_adjacency_cleaned | adjacency 引用不存在 room_id → 移除 + warning |
| test_type_coercion | weight="5" (string) → 5 (int) |
| test_old_field_compat | requires_window=True + needs_window 未设置 → needs_window=True |
| test_old_field_no_override | needs_window=False + requires_window=True → needs_window=False（不覆盖） |
| test_size_hint_to_area | size_hint="large"/"small" → 面积按比例分配 |
| test_small_floor_no_crash | 100m² 楼层 5 个房间 → 成功生成 |
| test_oversized_rooms_no_crash | 房间面积 > 岛屿面积 → 自动缩放（不低于 min_width×min_depth） |
| test_zero_islands_no_crash | 极端小楼层 → 有降级输出 |
| test_miqp_failure_fallback | MIQP 求解超时 → treemap 兜底 |
| test_connectivity_check | 生成后检测不可达房间 → warning |
| test_wall_door_window_generation | 后处理生成墙/门/窗 → 格式正确 |
| test_combined_failures | JSON 截断 + 面积超标 + 非法 adjacency 同时发生 → 不崩溃 |
| test_combined_zone_and_missing | zone 非法 + room_id 缺失 + weight 是字符串 → 全部修复 |
| test_degradation_summary | 触发多次降级 → summary 正确汇总 |
| test_real_prompt_e2e | "两层住宅" 完整管线不崩溃 |

---

## 关键文件清单

| 文件 | 操作 | Phase |
|---|---|---|
| `backend/core/prompts/building_prompts.py` | 拆为两步 prompt + 精简 schema 到 5 字段 | 1 |
| `backend/core/geometry/room_defaults.py` | 新建 — ROOM_TYPE_DEFAULTS + ADJACENCY_FORBIDDEN_RULES + SIZE_HINT_WEIGHTS | 1 |
| `backend/core/flows/building_semantic_flow.py` | 两步 LLM 调用 + 重写解析 + 升级重试 + 旧字段兼容修正 | 1, 2 |
| `backend/models.py` | RoomAllocation 添加 size_hint 字段，验证 extra="forbid" 兼容性 | 1, 2 |
| `backend/core/geometry/island_room_assigner.py` | assign() 4 级降级（含面积下限） | 3 |
| `backend/core/geometry/layout_generator.py` | MIQP 失败 fallback + treemap_fallback() 实现 | 3 |
| `backend/core/geometry/postprocessor.py` | 新建 — 自动生成墙/门/窗 | 4 |
| `backend/core/geometry/serializers.py` | 集成后处理输出 + 降级摘要 | 4, 5 |
| `backend/core/flows/degradation_collector.py` | 新建 — 降级事件收集器 | 5 |
| `backend/tests/test_building_robustness.py` | 新建 — 19 个鲁棒性测试（含组合故障） | 6 |

---

## 执行顺序

```
Phase 1 (Prompt 改造 + size_hint) ──┐
                                     ├──→ Phase 3 (几何容错) ──→ Phase 4 (后处理) ──→ Phase 5 (可观测性) ──→ Phase 6 (测试)
Phase 2 (解析层强化)  ───────────────┘
```

Phase 1 和 Phase 2 可并行。Phase 3 依赖 Phase 2 的解析输出。Phase 4/5 可在 Phase 3 之后并行。Phase 6 最后。

---

## 验证

```bash
# Phase 2: 解析测试
python -m pytest backend/tests/test_building_robustness.py -k "test_malformed or test_missing or test_invalid or test_type or test_old_field or test_combined_zone" -v

# Phase 3: 几何容错测试
python -m pytest backend/tests/test_building_robustness.py -k "test_small_floor or test_oversized or test_zero_islands or test_miqp" -v

# Phase 4: 后处理测试
python -m pytest backend/tests/test_building_robustness.py -k "test_wall_door_window or test_connectivity" -v

# Phase 5: 可观测性测试
python -m pytest backend/tests/test_building_robustness.py -k "test_degradation" -v

# Phase 6: 组合故障 + E2E
python -m pytest backend/tests/test_building_robustness.py -k "test_combined or test_real_prompt" -v

# 回归测试
python -m pytest backend/tests/test_layout_pipeline_v2.py backend/tests/test_building_allocation_parse.py -v

# 端到端（需要 LLM key + 启动后端）
curl -X POST http://localhost:8000/api/building/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"两层住宅，一楼客厅餐厅厨房，二楼三间卧室"}'
```