from __future__ import annotations

import json
from typing import List

from ...models import LayoutElement
from ..scenes.scene_types import SceneDefinition


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

