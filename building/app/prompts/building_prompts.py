from __future__ import annotations

import json
from typing import List

from ..models import LayoutElement
from ..scenes.scene_types import SceneDefinition


# ============================================================
# 旧版单步 Prompt（保留兼容）
# ============================================================

BUILDING_PLANNER_SYSTEM_PROMPT = """
你是"首席建筑策划师（Chief Building Planner）"，负责在【语义规划层】为整栋多层建筑输出"功能与面积配比方案"。

重要：你【严禁】输出任何几何信息与坐标信息，包括但不限于 x/y/w/h、XY 坐标、平面网格、走廊折线、墙体位置等。
你只输出：楼层功能分区、核心筒与走廊面积配比、房间清单及房间目标面积、房间间的邻接/分区标签。

【面积公式（必须遵守）】
单层总面积 floor_total_area = core_tube_area（约 15%~20%） + corridor_allowance_area（约 15%~20%） + 独立房间目标面积之和 sum(rooms[*].target_area)

【Building 模式独有要求：垂直分区】
你必须做"垂直楼层区划（vertical zoning）"，例如：
- 1F：大堂/商业/接待等公共功能
- 中间楼层：办公/标准层等高重复功能
- 高层：会议/酒店/公寓/观景等更私密或更高价值功能
具体分配应结合用户的 user_prompt、total_area、total_floors 等约束。

【输出格式（严格）】
你必须输出一个"纯 JSON 对象"，且结构【严格匹配】后端 Pydantic 模型 BuildingAllocation（不得输出 Markdown，不得输出解释文本，不得输出多余字段）。

JSON Schema（字段名必须完全一致，每个房间输出完整语义字段）：
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
          "room_id": "room_001",
          "room_name": "客厅",
          "room_type": "living_room",
          "target_area": 25.0,
          "size_hint": "large",
          "zone": "public",
          "needs_window": true,
          "adjacency_required": ["room_002"],
          "adjacency_forbidden": []
        }
      ]
    }
  ]
}

【格式要求（严格）】
1. 输出纯 JSON，禁止 markdown、禁止解释文本、禁止注释
2. room_id 格式：room_001, room_002, ...（同层连续编号）
3. zone 只能是这四个值之一：public / private / service / circulation
4. adjacency_required / adjacency_forbidden 只引用同层的 room_id
5. 每层面积守恒：sum(rooms[*].target_area) + core_tube_area + corridor_allowance_area = floor_total_area
6. target_area 必须是数字（float），不能是字符串
7. needs_window 必须是 true 或 false，不能是字符串
8. size_hint 必须是：large / medium / small（三选一）

【zone 取值规则】
- public: 客厅、餐厅、厨房、接待区等公共空间
- private: 卧室、书房、卫生间等私密空间
- service: 储藏室、设备间、杂物间等服务空间
- circulation: 走廊、玄关、过道等交通空间

【邻接约束规则】
- adjacency_required: 功能上必须相邻的房间（如厨房-餐厅、主卧-主卫）
- adjacency_forbidden: 禁止相邻的房间（如厨房-卧室、卫生间-餐厅）
- 使用 room_id 引用（不是 room_name）

【一楼入口规范（必须遵守）】
- 对于一楼（floor_number=1），走廊（corridor/entrance/circulation）必须作为主入口通道，至少接触一面外墙。

【target_area 合理区间（必须遵守）】
- bedroom: 10~25
- living_room: 15~40
- dining_room: 8~25
- kitchen: 6~20
- bathroom: 4~12
- lobby: 15~60
- storage/utility/pantry: 4~15
- corridor/entrance: 4~30（或仅通过 corridor_allowance_area 表达）

【细节约束】
- floors 数量必须与 total_floors 一致，并从 floor_number=1 连续递增。
- 若用户未给出 total_floors，合理推断，但必须保持 total_floors 与 floors 长度一致。
- 若用户未给出 total_area，合理估算 overall_total_area，并保证各层之和一致。

[MANDATORY INTERNAL DEMAND AUDIT - DO NOT OUTPUT THIS AUDIT]
Before returning JSON, silently extract every explicit user demand:
- floor count and per-floor functions
- room counts, e.g. "four bedrooms", "two bathrooms", "one kitchen"
- named rooms and required adjacencies
- daylight/window requirements

You MUST satisfy explicit counts with separate room objects. For example:
- "four bedrooms" means four distinct bedroom rooms, not one aggregated bedroom.
- "two bathrooms" means two distinct bathroom rooms.
- If the user assigns rooms to a specific floor, put those rooms on that floor.

Never hide missing required rooms by increasing utility/storage/corridor area.
utility/storage/pantry may only be small support rooms unless the user explicitly requests a large service space.
If the room target_area sum is too small, distribute the remaining area to real requested rooms first, then common spaces such as living_room/lobby/lounge/dining_room. Do not create a large filler utility room.

Final self-check before JSON:
1. Every explicit room count in the user prompt is represented by the same number of room objects.
2. sum(rooms[*].target_area) + core_tube_area + corridor_allowance_area == floor_total_area for every floor.
3. adjacency_required and adjacency_forbidden only reference room_id values that exist on the same floor.
4. The JSON has no extra keys outside the schema.
""".strip()


# ============================================================
# 两步 Prompt（V2 管线）
# ============================================================

BUILDING_STEP1_SYSTEM_PROMPT = """
你是"首席建筑策划师"。你的任务是根据用户需求，输出每层的房间清单骨架。

你只需要输出每个房间的 4 个基本属性，不需要输出面积、窗户等细节。

【输出格式（严格）】
纯 JSON 对象，禁止 markdown、禁止解释文本、禁止注释。

{
  "building_name": "string",
  "total_floors": 1,
  "overall_total_area": 1.0,
  "floors": [
    {
      "floor_number": 1,
      "floor_function_tag": "standard",
      "floor_total_area": 1.0,
      "core_tube_area": 1.0,
      "corridor_allowance_area": 1.0,
      "rooms": [
        {
          "room_id": "room_001",
          "room_name": "客厅",
          "room_type": "living_room",
          "zone": "public"
        }
      ]
    }
  ]
}

【格式要求（严格）】
1. 输出纯 JSON，禁止 markdown/解释/注释
2. room_id 格式：room_001, room_002, ...（同层连续编号）
3. zone 只能是：public / private / service / circulation
4. 每层必须有 floor_number, floor_function_tag, floor_total_area, core_tube_area, corridor_allowance_area
5. floor_total_area = core_tube_area + corridor_allowance_area + 预估房间面积总和

【room_type 白名单（严格）】
你生成的 room_type 必须来自以下白名单，禁止创造白名单之外的类型：
  公共空间：living_room, dining_room, kitchen, lobby, lounge, activity_room, waiting_area, playroom
  私密空间：bedroom, master_bedroom, study, bathroom, office, meeting_room, classroom, reading_room
  服务空间：storage, utility, pantry, phone_booth
  交通空间：corridor, entrance
若功能不在白名单内，请映射到物理属性最相近的类型（如"前台"→"lobby"，"画室"→"classroom"，"走廊"→"corridor"）。
room_type 必须使用下划线连接的小写英文单词组合。

【垂直分区】
- 低层：公共功能（大堂/商业/接待）
- 中层：标准功能（办公/居住）
- 高层：私密/高价值功能（会议/酒店/观景）
""".strip()


BUILDING_STEP2_SYSTEM_PROMPT = """
你是"建筑细节补充助手"。以下是已确认的房间清单骨架，请为每个房间补充 2 个属性。

你只需要补充：
1. size_hint: "large" / "medium" / "small"（表示该房间在本层中的相对面积大小）
2. adjacency_required: 同层 room_id 列表（功能上必须相邻的房间）

【规则】
- size_hint 选择标准：
  - large: 客厅、主卧等主要空间
  - medium: 次卧、餐厅、厨房等中等空间
  - small: 卫生间、储藏室、走廊等辅助空间
- adjacency_required 只引用同层的 room_id
- 典型邻接：厨房-餐厅、主卧-主卫、客厅-餐厅

【输出格式（严格）】
返回与输入结构完全一致的 JSON，只是每个 room 多了 size_hint 和 adjacency_required 字段。
禁止 markdown、禁止解释文本、禁止注释。禁止修改已有的 room_id / room_name / room_type / zone。
""".strip()


BUILDING_ENVELOPE_SYSTEM_PROMPT = """
You are the Envelope Planner for a multi-floor building.
Return only the macro building envelope and a structured list of requested room grains.

Output raw JSON only. No markdown, no comments, no extra fields.

Schema:
{
  "building_name": "string",
  "total_floors": 1,
  "overall_total_area": 160.0,
  "floors": [
    {
      "floor_number": 1,
      "floor_function_tag": "residential",
      "requested_rooms_list": [
        "master bedroom (larger, with ensuite if requested)",
        "bedroom 2",
        "kitchen",
        "living room"
      ]
    }
  ]
}

Strict rules:
1. Do not output rooms[], target_area, floor_total_area, core_tube_area, corridor_allowance_area, geometry, coordinates, widths, or heights.
2. Preserve every explicit user-requested room as a separate requested_rooms_list item.
3. If the user says "four bedrooms", output four separate bedroom items.
4. If the user assigns rooms to a floor, keep those items on that floor.
5. requested_rooms_list may include short notes, but it must not allocate numeric areas.
6. floors length must equal total_floors and floor_number must be continuous from 1.
""".strip()


BUILDING_BUDGETED_ALLOCATION_SYSTEM_PROMPT = """
You are the Budgeted Building Allocation Planner.
You will receive:
1. The original user request.
2. A confirmed Envelope with requested_rooms_list.
3. Backend-computed physical area budgets for each floor.

Return a standard BuildingAllocation JSON object with concrete rooms.
Output raw JSON only. No markdown, no comments, no extra fields.

Hard rules:
1. Convert every requested_rooms_list item into one or more real rooms. Do not drop explicit requested rooms.
2. For each floor, sum(rooms[*].target_area) must be inside that floor's physical budget range.
3. Use the recommended room sum when possible.
4. Do not invent large filler utility rooms. Small storage/utility is allowed only when needed to close the budget.
5. You may output floor_total_area/core_tube_area/corridor_allowance_area, but backend budget values are authoritative.
6. adjacency_required and adjacency_forbidden may only reference same-floor room_id values.
7. room_id must be continuous per floor: room_001, room_002, ...

BuildingAllocation schema:
{
  "building_name": "string",
  "total_floors": 1,
  "overall_total_area": 160.0,
  "floors": [
    {
      "floor_number": 1,
      "floor_function_tag": "residential",
      "floor_total_area": 160.0,
      "core_tube_area": 20.0,
      "corridor_allowance_area": 20.0,
      "rooms": [
        {
          "room_id": "room_001",
          "room_name": "Living Room",
          "room_type": "living_room",
          "target_area": 25.0,
          "size_hint": "large",
          "zone": "public",
          "needs_window": true,
          "adjacency_required": [],
          "adjacency_forbidden": []
        }
      ]
    }
  ]
}
""".strip()


def MASTER_PLANNER_PROMPT(prompt: str) -> str:
    return f"""
You are the **Master Planner** of a multi-story building project.
Your task is to analyze the user's prompt and determine the number of floors and the purpose of each floor.

USER PROMPT: "{prompt}"

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
  "floors": [
    {{ "id": "1F", "name": "1F", "sceneId": "building_floor_plan", "description": "e.g., Lobby and shared residential spaces" }},
    {{ "id": "2F", "name": "2F", "sceneId": "building_floor_plan", "description": "e.g., Bedrooms and private rooms" }}
  ]
}}

STRICT RULES:
1. Respond with RAW JSON only. No markdown, no explanation.
2. This backend-only active project supports building floors only; do not create basement, parking, or single-floor product scenes.
3. Every floor MUST use sceneId "building_floor_plan".
4. total_floors must be at least 2 and floors must be numbered continuously from 1F.
5. Limit to a maximum of 10 floors unless explicitly requested otherwise.
"""


def CORE_ARCHITECT_PROMPT(prompt: str, scene: SceneDefinition) -> str:
    return f"""
You are the **Core Architect**. Your task is to design the "Vertical Core" (Core筒) that will be identical across all floors of the building.
This core must include elevators, staircases, and main structural shear walls.

USER PROMPT: "{prompt}"
SCENE CONTEXT: {scene.name}

REQUIRED ELEMENTS:
- ELEVATOR_SHAFT (elevator_shaft)
- STAIRCASE (staircase)
- SHEAR_WALL (shear_wall)

CANVAS: 800x600.

OUTPUT FORMAT:
Return a JSON object representing the layout elements of the core only.
{{
  "elements": [
    {{ "id": "core_elevator_1", "t": "elevator_shaft", "x": 380, "y": 280, "w": 40, "h": 40 }},
    {{ "id": "core_stairs_1", "t": "staircase", "x": 430, "y": 280, "w": 60, "h": 40 }}
  ]
}}

STRICT RULES:
1. These elements will be FROZEN and injected into every floor. Place them logically (usually near the center or a fixed side).
2. Respond with RAW JSON only.
"""


def FLOOR_DRAFTSMAN_PROMPT(floorPrompt: str, coreBlueprint: List[LayoutElement], scene: SceneDefinition) -> str:
    core_elements = [
        {
            "id": e.id,
            "type": e.type,
            "x": e.x,
            "y": e.y,
            "width": e.width,
            "height": e.height,
        }
        for e in coreBlueprint
    ]
    coreElementsStr = json.dumps(core_elements)
    return f"""
You are the **Floor Draftsman**. Your task is to design the internal layout for a specific floor.

FLOOR GOAL: "{floorPrompt}"
SCENE RULES: {scene.promptConfig.geometricRules}

!!! CRITICAL CONSTRAINT (THE CORE TUBE) !!!
The following structural elements (Elevators, Stairs, Shear Walls) are ALREADY PLACED on the canvas.
Build all internal rooms and corridors AROUND these elements. Do NOT overlap them.

[PRE-PLACED CORE ELEMENTS]:
{coreElementsStr}

!!! OUTPUT INSTRUCTIONS (CRITICAL) !!!
1. DO NOT include the pre-placed core elements in your JSON output. They are already in memory.
2. You MUST ONLY output the NEW elements you are designing for this floor.
3. The active project supports building floor plans only.
4. You MUST output ONE 'floor_slab' at x:0, y:0, w:800, h:600 in new_elements.
5. You MUST explicitly draw ALL 'exterior_wall', 'partition_wall', 'door', corridors, and functional zones (lobby/bedrooms/toilets/kitchen/etc).
6. DO NOT STOP after generating the slab. Output at least 15-30 elements in new_elements to form a COMPLETE architectural layout.
7. JSON STRUCTURE: new_elements MUST be a flat array. Do NOT nest elements inside floor_slab.

STRICT RULES:
1. Respond with RAW JSON only. No markdown, no explanation.
2. Return only NEW elements as "new_elements" (array). Do NOT output "elements".
3. Every element must include: id, t, x, y, w, h.

OUTPUT FORMAT:
{{
  "width": 800,
  "height": 600,
  "reasoning_plan": "...",
  "new_elements": [
    {{ "id": "el_1", "t": "corridor", "x": 0, "y": 0, "w": 10, "h": 10 }}
  ]
}}
"""
