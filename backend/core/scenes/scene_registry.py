from __future__ import annotations

from typing import Dict

from ..geometry.ai_common_utils import (
    clean_intersections,
    cleanup_pillars,
    fill_parking_automatically,
    generate_charging_stations,
    orient_guidance_signs,
    resolve_priority_conflicts,
)
from ..geometry.floor_geometry_utils import close_exterior_boundary, enforce_orthogonal_walls, snap_doors_to_walls
from ..prompts.floor_prompts import floorPlanExamples, floorPlanSystemPrompt
from ..types import ElementType
from .scene_types import SceneDefinition, ScenePromptConfig


parking_z_order = [
    ElementType.WALL,
    ElementType.GROUND,
    ElementType.ROAD,
    ElementType.RAMP,
    ElementType.SIDEWALK,
    ElementType.PARKING_SPACE,
    ElementType.LANE_LINE,
    ElementType.SPEED_BUMP,
    ElementType.PILLAR,
    ElementType.STAIRCASE,
    ElementType.ELEVATOR,
    ElementType.SAFE_EXIT,
    ElementType.FIRE_EXTINGUISHER,
    ElementType.GUIDANCE_SIGN,
    ElementType.CHARGING_STATION,
    ElementType.CONVEX_MIRROR,
]


ParkingScene = SceneDefinition(
    id="parking_underground",
    name="Underground Parking",
    description="Automated vehicle flow and high-density parking layout.",
    promptConfig=ScenePromptConfig(
        roleDefinition="Architectural Spatial Planner specialized in Underground Parking",
        geometricRules="""
1. **CLOSED LOOP PERIMETER**: Walls MUST overlap or touch at corners. NO perimeter gaps.
2. **ADAPTABLE ROAD NETWORK (CRITICAL)**:
   - The layout of 'driving_lane' (Roads) MUST strictly follow the user's description (e.g., parallel lanes, central spine, grid, or loop).
   - All roads must be continuously connected. NO dead ends unless logically required.
3. **'ground' Elements (STRUCTURAL FILL)**:
   - **NO FLOATING ISLANDS**: Every 'ground' element MUST touch a 'driving_lane', a 'wall', or another 'ground' block on its edges.
   - **100% INTERNAL FILL**: ANY space inside the perimeter walls that is NOT occupied by 'driving_lane' MUST be completely filled with 'ground' elements.
   - **SEAMLESS TOUCHING**: If splitting the parking zones into multiple 'ground' strips, they must TOUCH (no black gaps).
4. **Boundary Snapping**:
   - 'entrance' and 'exit' MUST touch the edges of the canvas.
5. **ZERO-VOID POLICY**:
   - Any space not occupied by a 'wall' or 'driving_lane' MUST be covered by 'ground'.
6. 'slope' (40x60) connectors must join Entrance/Exit to Roads.
    """,
        requiredElements=["wall", "driving_lane", "ground", "entrance", "exit", "slope"],
        exampleJSON="""{
  "reasoning_plan": "Connected road network with ground islands and boundary gates.",
  "width": 800,
  "height": 600,
  "elements": [
    {"t":"wall","x":0,"y":0,"w":800,"h":20},
    {"t":"wall","x":0,"y":580,"w":800,"h":20},
    {"t":"wall","x":0,"y":0,"w":20,"h":600},
    {"t":"wall","x":780,"y":0,"w":20,"h":600},
    {"t":"driving_lane","x":100,"y":80,"w":600,"h":60},
    {"t":"ground","x":20,"y":20,"w":760,"h":560},
    {"t":"entrance","x":380,"y":0,"w":40,"h":20},
    {"t":"slope","x":380,"y":20,"w":40,"h":60},
    {"t":"exit","x":380,"y":580,"w":40,"h":20},
    {"t":"slope","x":380,"y":520,"w":40,"h":60}
  ]
}""",
    ),
    styles={},
    zOrder=parking_z_order,
    elementNormalization={
        "column": ElementType.PILLAR,
        "post": ElementType.PILLAR,
        "barrier": ElementType.WALL,
        "utility_box": ElementType.PILLAR,
        "parking_spot": ElementType.PARKING_SPACE,
        "parking_bay": ElementType.PARKING_SPACE,
        "road": ElementType.ROAD,
        "lane": ElementType.ROAD,
        "path": ElementType.SIDEWALK,
        "pedestrian_walkway": ElementType.SIDEWALK,
    },
    postProcessAlgorithms=[
        clean_intersections,
        fill_parking_automatically,
        clean_intersections,
        generate_charging_stations,
        cleanup_pillars,
        clean_intersections,
        lambda layout: layout.model_copy(update={"elements": resolve_priority_conflicts(layout.elements)}),
        orient_guidance_signs,
    ],
)


GenericScene = SceneDefinition(
    id="scene_generic",
    name="Generic 2D Scene",
    description="Generic semantic scene layout with flexible element types.",
    promptConfig=ScenePromptConfig(
        roleDefinition="2D Scene Composer",
        geometricRules="""
1. Use a coherent topology based on the user request (rooms/streets/islands/zones).
2. Keep elements within bounds and avoid large overlaps.
3. Use simple rectangular blocks with short keys (t/x/y/w/h).
    """,
        requiredElements=[],
        exampleJSON="""{
  "reasoning_plan":"Simple street with park and water",
  "width":800,"height":600,
  "elements":[
    {"t":"floor","x":0,"y":0,"w":800,"h":600},
    {"t":"road","x":80,"y":260,"w":640,"h":80},
    {"t":"park","x":120,"y":80,"w":260,"h":140},
    {"t":"water","x":480,"y":80,"w":200,"h":180},
    {"t":"building","x":120,"y":380,"w":220,"h":160}
  ]
}""",
    ),
    styles={},
    zOrder=["floor", "grass", "park", "water", "road", "building", "wall", "zone", "object"],
    elementNormalization={},
    postProcessAlgorithms=None,
)


FloorPlanScene = SceneDefinition(
    id="building_floor_plan",
    name="Floor Plan",
    description="Architectural floor plan with structural rules.",
    promptConfig=ScenePromptConfig(
        roleDefinition="Architectural Floor Plan Designer",
        geometricRules=floorPlanSystemPrompt["rules"],
        requiredElements=["exterior_wall", "partition_wall", "corridor", "door", "elevator_shaft"],
        exampleJSON=floorPlanExamples["complexLayout"],
    ),
    styles={},
    zOrder=[],
    elementNormalization={},
    postProcessAlgorithms=[
        enforce_orthogonal_walls,
        close_exterior_boundary,
        snap_doors_to_walls,
    ],
)


BuildingScene = SceneDefinition(
    id="building",
    name="Building",
    description="Generate a multi-story building (B1 parking + upper floor plans).",
    promptConfig=ScenePromptConfig(
        roleDefinition="Building Orchestrator",
        geometricRules="Use floor-specific rules based on per-floor scene.",
        requiredElements=[],
        exampleJSON='{"floors":[{"id":"B1","sceneId":"parking_underground"},{"id":"1F","sceneId":"building_floor_plan"}]}',
    ),
    styles={},
    zOrder=[],
    elementNormalization={},
    postProcessAlgorithms=None,
)


DEFAULT_SCENE_ID = ParkingScene.id


SCENE_REGISTRY: Dict[str, SceneDefinition] = {
    ParkingScene.id: ParkingScene,
    GenericScene.id: GenericScene,
    FloorPlanScene.id: FloorPlanScene,
    BuildingScene.id: BuildingScene,
}

