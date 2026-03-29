from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ...models import ConstraintViolation, ParkingLayout
from ..scenes.scene_types import SceneDefinition


PROTOCOLS = {
    "FULL_STATE": """
 ## OUTPUT FORMAT: FULL STATE 
 - Return the COMPLETE JSON object with ALL elements. 
 - DO NOT use partial updates. 
 """,
    "PATCH_ONLY": """
 ## OUTPUT FORMAT: INCREMENTAL PATCH (CRITICAL) 
 - You are operating in **Low-Bandwidth Mode**. 
 - **DO NOT** return the full JSON. 
 - **ONLY** return the elements that you modified, added, or moved. 
 - Use the structure: { "modified_elements": [...], "deleted_ids": [...] } 
 - If you change a wall's length, return the object with the SAME ID and new 'width'. 
 - Maintain ID CONSISTENCY: Never rename existing element IDs. Use the SAME ID for modifications. 
 - If you create NEW elements, include them under "new_elements"; do not mix with "modified_elements". 
 - STRICT OUTPUT: Respond with RAW JSON only. No Markdown, no code fences, no explanatory text. 
 - REQUIRED KEYS: Use exactly "modified_elements" and "deleted_ids" for patches. 
 """,
}


ROLES = {
    "GENERATOR": """
 ## YOUR ROLE: Visionary Architect 
 You are a senior spatial designer. Your goal is to create a layout from scratch. 
 Focus on creativity, flow, and spatial utilization. 
 """,
    "OPTIMIZER": """
 ## YOUR ROLE: Detail-Oriented Engineer 
 You are a facility optimization expert. The layout already exists. 
 Your goal is to ADD facilities (stairs, elevators) without breaking the existing structure. 
 Focus on precision and compliance. 
 """,
    "FIXER": """
 ## YOUR ROLE: Strict Compliance Officer (Debugger) 
 You are a code logic validator. You DO NOT design; you ONLY fix violations. 
 Your goal is to resolve overlaps and invalid placements surgically. 
 Be minimal and non-destructive. 
 """,
}


def system_prompt() -> str:
    return """You are an expert parking lot designer with extensive experience in spatial planning.

Generate a detailed parking lot layout as JSON. Your response MUST be ONLY valid JSON, nothing else.

Your response must follow this exact structure:
{
  "width": number (e.g., 800),
  "height": number (e.g., 600),
  "reasoning_plan": "STRICTLY UNDER 30 WORDS. Summary of strategy only.",
  "elements": [
    {
      "id": "unique_identifier",
      "type": "GROUND" | "ROAD" | "PARKING_SPACE" | "SIDEWALK" | "WALL" | "ENTRANCE" | "EXIT" | "PILLAR" | "RAMP" | "STAIRCASE" | "ELEVATOR" | "CHARGING_STATION" | "GUIDANCE_SIGN" | "SAFE_EXIT" | "SPEED_BUMP" | "FIRE_EXTINGUISHER" | "LANE_LINE" | "CONVEX_MIRROR",
      "x": number,
      "y": number,
      "width": number,
      "height": number,
      "rotation": number (0-360, optional),
      "label": "descriptive text" (optional)
    }
  ]
}

CRITICAL RULES:
1. ALL parking spaces must have type "PARKING_SPACE"
2. ALL roads/lanes must have type "ROAD"
3. Width and height must be positive numbers > 0
4. X and Y coordinates must be within layout bounds
5. Do NOT include any explanatory text before or after the JSON
6. Start with { and end with } directly
7. All element types must be from the list above

Design principles:
- Maximize parking efficiency
- Ensure clear traffic flow
- Separate entrance and exit clearly
- Include pedestrian paths
- Place support infrastructure logically"""


def generation(description: str, scene: SceneDefinition) -> str:
    is_parking = ("driving_lane" in scene.promptConfig.requiredElements) or ("slope" in scene.promptConfig.requiredElements)
    if is_parking:
        required = "\n".join([f"- '{e}'" for e in scene.promptConfig.requiredElements])
        return f"""
  You are an **{scene.promptConfig.roleDefinition}**.
  Generate a **COARSE-GRAINED JSON layout** (0,0 at top-left) for: "{description}".
  
  **GOAL**: Create the infrastructure skeleton (Walls, Roads, Ground, Entrances).
  **DO NOT** generate fine details like 'parking_space', 'pillar', or 'lane_line' yet. These will be added later.

  **CANVAS CONSTRAINTS**: Width: 800, Height: 600.

  **CRITICAL GEOMETRIC RULES**:
  {scene.promptConfig.geometricRules}

  **TOKEN SAVING INSTRUCTION**:
  - Keep 'reasoning_plan' extremely short (max 1 sentence).
  - Use compact keys: t/x/y/w/h where possible.

  **REQUIRED ELEMENTS**:
  {required}

  **JSON EXAMPLE**:
  {scene.promptConfig.exampleJSON}
      """
    required = "\n".join([f"- '{e}'" for e in scene.promptConfig.requiredElements])
    return f"""
  You are an **{scene.promptConfig.roleDefinition}**.
  Generate a **COMPLETE STRUCTURAL JSON layout** (0,0 at top-left) for: "{description}".
  
  **GOAL**: Create the full architectural shell, including ALL walls (exterior & interior), zones (rooms), and connectivity (doors/windows).
  **DO NOT** generate furniture yet (beds, sofas, etc.) unless explicitly asked, but **DO** generate all structural partitions.

  **CANVAS CONSTRAINTS**: Width: 800, Height: 600.

  **CRITICAL GEOMETRIC RULES**:
  {scene.promptConfig.geometricRules}

  **TOKEN SAVING INSTRUCTION**:
  - Keep 'reasoning_plan' extremely short (max 1 sentence).
  - Use compact keys: t/x/y/w/h where possible.

  **REQUIRED ELEMENTS**:
  {required}

  **JSON EXAMPLE**:
  {scene.promptConfig.exampleJSON}
    """


def refinement(simplified_layout: Any, width: int, height: int, scene: SceneDefinition) -> str:
    is_parking = ("driving_lane" in scene.promptConfig.requiredElements) or ("slope" in scene.promptConfig.requiredElements)
    if not is_parking:
        allowed = ", ".join(scene.styles.keys())
        return f"""
    You are a **2D Scene Detailer & Interior Designer**.
    Task: 
    1. If the layout is empty or missing internal walls, **BUILD THEM** based on the prompt logic.
    2. If the structure is complete, **FURNISH** the rooms with appropriate furniture/fixtures.
    
    Canvas: {width}x{height}
    Existing Elements:
    {json.dumps(simplified_layout.get('elements'))}

    Allowed types: {allowed or 'any lowercase strings'}

    Rules:
    - **RESPECT EXISTING STRUCTURE**: Do not move exterior walls unless broken.
    - **FURNISHING**: Place beds in bedrooms, sofas in living rooms, etc.
    - **COMPLETION**: If rooms are missing (e.g. only outer walls exist), please ADD partition_walls and doors to create the rooms.
    - Use compact keys: t/x/y/w/h.

    Output JSON:
    {{ "reasoning_plan":"...", "new_elements":[ {{ "t":"...", "x":0, "y":0, "w":10, "h":10 }} ] }}
      """
    return f"""
    You are a **Spatial Algorithm Engine**.
    Task: Inject NEW detailed structural and facility elements into the existing layout.

    **INPUT DATA**:
    - Canvas: {width}x{height}
    - Existing Elements:
    {json.dumps(simplified_layout.get('elements'))}

    **CRITICAL DESIGN RULES**:
    - **FACILITY PLACEMENT**: 'staircase', 'elevator', and 'safe_exit' MUST be placed on 'ground' elements. They are FORBIDDEN from being on 'driving_lane'.
    - **SPEED BUMP ORIENTATION**: 'deceleration_zone' must be PERPENDICULAR to the road direction.
    - **SIDEWALK LOGIC**: 'pedestrian_path' must cross the 'driving_lane' to connect 'ground' areas.

    **SYSTEM ARCHITECTURE (TOKEN SAVING)**:
    - **Algorithmic Spot Filler**: Do NOT generate 'parking_space'. My algorithm will fill them later.
    - **Focus**: Only generate 'pillar', 'ground_line', 'guidance_sign', 'staircase', 'elevator', 'safe_exit', 'pedestrian_path'.

    **INCREMENTAL OUTPUT MODE (CRITICAL)**:
    - **DO NOT** return the existing 'wall', 'driving_lane', or 'ground' elements provided in INPUT.
    - **ONLY** return the **NEW** elements you are creating in this step.

    **SCENE RULES**:
    {scene.promptConfig.geometricRules}

    **OUTPUT JSON FORMAT**:
    {{
      "reasoning_plan": "Added [X] pillars and [Y] signs...",
      "new_elements": [
         {{ "t": "pillar", "x": 100, "y": 100, "w": 10, "h": 10 }}
      ]
    }}
    """


def fix(layout: ParkingLayout, violations: List[ConstraintViolation], scene: SceneDefinition, options: Optional[Dict[str, Any]] = None) -> str:
    is_parking = ("driving_lane" in scene.promptConfig.requiredElements) or ("slope" in scene.promptConfig.requiredElements)
    frozen_ids = (options or {}).get("frozenIds") if options else None
    lock_warning = (
        f"""
    WARNING:
    - Pre-placed core elements (especially any element whose ID contains 'core_') are PHYSICALLY LOCKED.
    - You CANNOT change their x/y/w/h. Any attempted modifications to them will be rejected.
    - Resolve overlaps by moving/resizing OTHER elements instead.
      """
        if frozen_ids and len(frozen_ids) > 0
        else ""
    )
    if not is_parking:
        return f"""
    {ROLES['FIXER']}
    {PROTOCOLS['PATCH_ONLY']}
    You are a **Scene Constraint Fixer**.
    INPUT: {layout.width}x{layout.height} Canvas.
    VIOLATIONS: {json.dumps([v.model_dump() for v in violations])}
    {lock_warning}

    Rules:
    - Only fix elements referenced by violations.
    - Prefer moving/resizing; delete only when necessary.

    Output JSON:
    {{ "reasoning_plan":"...", "modified_elements":[...], "deleted_ids":[...] }}
      """
    return f"""
    {ROLES['FIXER']}
    {PROTOCOLS['PATCH_ONLY']}
    You are a **Topological Constraint Solver**.
    
    **INPUT**: {layout.width}x{layout.height} Canvas.
    **VIOLATIONS**: {json.dumps([v.model_dump() for v in violations])}
    {lock_warning}

    **CRITICAL RULES**:
    {scene.promptConfig.geometricRules}
    1. **ZERO-VOID / GAP FILLING**: 
       - Any narrow gap between a 'driving_lane' and a 'wall' (or another road) MUST be filled by **EXTENDING THE GROUND**, NOT by creating a new road.
       - **Action**: If you see a small gap, resize the adjacent 'ground' to touch the road. **NEVER SHRINK** 'ground' elements to fix overlaps with 'driving_lane' or 'wall' if it creates gaps.
    
    2. **FACILITY PLACEMENT**:
       - 'staircase', 'elevator', 'safe_exit' MUST sit on 'ground'.
       - They CANNOT float in 'driving_lane' or empty space.
       - They CANNOT overlap with pillars or each other.

    3. **CLEAN INTERSECTIONS**:
       - Road junctions (where two roads overlap) must be EMPTY.
       - **DELETE** any 'ground_line', 'parking_space', or 'guidance_sign' caught inside a road intersection.

    **HIERARCHY OF TRUTH**:
    1. **Immutable**: Walls, Roads, Entrances/Exits.
    2. **Flexible**: Ground (Resize to SNAP to roads/walls).
    3. **Disposable**: Parking Spaces / Pillars (Delete if bad).

    **SURGICAL EXECUTION PLAN**:
    - **Gap Fix**: Resize Ground_ID to fill gap.
    - **Intersection Clean**: Delete Element_ID inside junction.
    - **Placement Fix**: Move Facility_ID onto nearest Ground.

    **INCREMENTAL OUTPUT MODE (CRITICAL)**:
    - **DO NOT** return the full JSON. It is too large and will be truncated.
    - **ONLY** return the elements that you actually MODIFIED, RESIZED, or MOVED.
    - If you resize a ground block, return ONLY that specific ground block with new coordinates.
    - If you delete an element, do not return it (I will handle missing IDs). 
    
    **OUTPUT JSON FORMAT**: 
    {{
      "reasoning_plan": "Fixed [X] violations by resizing [ID].",
      "modified_elements": [
         {{ "id": "id_1", "t": "...", "x": ..., "y": ..., "w": ..., "h": ... }}
      ]
    }}
    """


def generateSystemPrompt(description: str, scene: SceneDefinition) -> str:
    return f"""
 {ROLES['GENERATOR']}
 {PROTOCOLS['FULL_STATE']}
 {generation(description, scene)}
 """


def optimizeSystemPrompt(simplified_layout: Any, width: int, height: int, scene: SceneDefinition) -> str:
    return f"""
 {ROLES['OPTIMIZER']}
 {PROTOCOLS['PATCH_ONLY']}
 {refinement(simplified_layout, width, height, scene)}
 """


def fixSystemPrompt(layout: ParkingLayout, violations: List[ConstraintViolation], scene: SceneDefinition) -> str:
    return f"""
 {ROLES['FIXER']}
 {PROTOCOLS['PATCH_ONLY']}
 {fix(layout, violations, scene)}
 """


def fixPrompt(violations: Any) -> str:
    return f"""
 CONTEXT: The current layout has logical errors. 
 VIOLATIONS: {json.dumps(violations)} 
 
 MISSION: 
 1. Analyze the specific IDs involved in the violations. 
 2. Apply the MINIMAL changes needed to resolve them (e.g., slight resize or move). 
 3. **DO NOT regenerate the whole map.** Only output the specific elements you touched. 
 
 REMEMBER: You are the "Compliance Officer". Do not redesign the parking lot. Just fix the bugs. 
 
 {PROTOCOLS['PATCH_ONLY']}
 """


PROMPTS: Dict[str, Any] = {
    "systemPrompt": system_prompt,
    "generation": generation,
    "refinement": refinement,
    "fix": fix,
    "generateSystemPrompt": generateSystemPrompt,
    "optimizeSystemPrompt": optimizeSystemPrompt,
    "fixSystemPrompt": fixSystemPrompt,
    "fixPrompt": fixPrompt,
}

