from __future__ import annotations

import json
from typing import List

from ...models import LayoutElement
from ..scenes.scene_types import SceneDefinition


BUILDING_PLANNER_SYSTEM_PROMPT = """
你是“首席建筑策划师（Chief Building Planner）”，负责在【语义规划层】为整栋多层建筑输出“功能与面积配比方案”。

重要：你【严禁】输出任何几何信息与坐标信息，包括但不限于 x/y/w/h、XY 坐标、平面网格、走廊折线、墙体位置等。
你只输出：楼层功能分区、核心筒与走廊面积配比、房间清单及房间目标面积、房间间的邻接/分区标签。

【面积公式（必须遵守）】
单层总面积 floor_total_area = core_tube_area（约 15%~20%） + corridor_allowance_area（约 15%~20%） + 独立房间目标面积之和 sum(rooms[*].target_area)

【Building 模式独有要求：垂直分区】
你必须做“垂直楼层区划（vertical zoning）”，例如：
- 1F：大堂/商业/接待等公共功能
- 中间楼层：办公/标准层等高重复功能
- 高层：会议/酒店/公寓/观景等更私密或更高价值功能
具体分配应结合用户的 user_prompt、total_area、total_floors 等约束。

【输出格式（严格）】
你必须输出一个“纯 JSON 对象”，且结构【严格匹配】后端 Pydantic 模型 BuildingAllocation（不得输出 Markdown，不得输出解释文本，不得输出多余字段）。

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
          "room_name": "string",
          "room_type": "string",
          "target_area": 1.0,
          "requires_window": false,
          "weight": 1,
          "adjacency_tags": ["string"]
        }
      ]
    }
  ]
}

【细节约束】
- weight 取值范围 1-10（整数）。
- target_area / floor_total_area / overall_total_area 使用平方米（float）。
- floors 数量必须与 total_floors 一致，并从 floor_number=1 连续递增。
- 若用户未给出 total_floors，你需要根据场景合理推断，但必须保持 total_floors 与 floors 长度一致。
- 若用户未给出 total_area，你需要合理估算 overall_total_area，并保证各层 floor_total_area 之和与 overall_total_area 一致（允许存在极小的数值误差，但应尽量精确）。
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
    {{ "id": "B1", "name": "B1", "sceneId": "parking_underground", "description": "e.g., Underground parking and ramps" }},
    {{ "id": "1F", "name": "1F", "sceneId": "building_floor_plan", "description": "e.g., Lobby and Cafe" }},
    {{ "id": "2F", "name": "2F", "sceneId": "building_floor_plan", "description": "e.g., Office space" }}
  ]
}}

STRICT RULES:
1. Respond with RAW JSON only. No markdown, no explanation.
2. For Building generation, always include one basement parking floor (B1) using sceneId "parking_underground".
3. All non-parking floors MUST use sceneId "building_floor_plan".
4. Limit to a maximum of 10 floors unless explicitly requested otherwise.
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
Build all your internal rooms, corridors, or parking spaces AROUND these elements. Do NOT overlap them.

[PRE-PLACED CORE ELEMENTS]:
{coreElementsStr}

!!! OUTPUT INSTRUCTIONS (CRITICAL) !!!
1. DO NOT include the pre-placed core elements in your JSON output. They are already in memory.
2. You MUST ONLY output the NEW elements you are designing for this floor.
3. If the scene is 'parking_underground':
   - You MUST generate continuous perimeter 'wall' elements to form a closed 800x600 boundary.
   - You MUST generate the road skeleton ONLY (driving_lane) and walls (and ramps/entrance/exit if needed).
   - DO NOT output any 'parking_space'. Parking spaces will be computed and generated automatically along your roads.
   - DO NOT output any 'ground' element. Background fill will be handled by an algorithm.
   - DO NOT attempt to fill empty spaces with multiple small 'ground' elements.
4. If the scene is NOT 'parking_underground' (floor plan):
   - You MUST output ONE 'floor_slab' at x:0, y:0, w:800, h:600 in new_elements.
   - You MUST explicitly draw ALL 'exterior_wall', 'partition_wall', 'door', and functional zones (lobby/toilets/office/etc).
   - DO NOT STOP after generating the slab. Output at least 15-30 elements in new_elements to form a COMPLETE architectural layout.
   - JSON STRUCTURE: new_elements MUST be a flat array. Do NOT nest elements inside floor_slab.

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
