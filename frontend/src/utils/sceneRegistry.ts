import { ElementType, SceneDefinition } from '../types';
import { createFloorPlanSceneStyles, createParkingSceneStyles } from '../theme/buildingColorScheme';

const floorPlanSystemPrompt = {
  rules: `
  **PHASE 1 - CORE STRUCTURE RULES**:
  0. **STEP 1 (CRITICAL)**:
     - You MUST first generate a background element of type 'floor_slab' starting at x:0, y:0 with width:800, height:600.
     - ALL other rooms, walls, and corridors must be placed ON TOP of this slab.
  1. **DO NOT GENERATE FURNITURE**: No beds, sofas, or tables yet.
  2. **ENVELOPE**:
     - 'exterior_wall' (thickness 15): Must form a continuous, closed boundary around the entire plan.
     - 'window' (length 40-80, thickness 15): Must be embedded WITHIN 'exterior_wall' sections. Every primary room (living, bedroom) MUST have at least one window.
  3. **PARTITIONS & ZONES**:
     - 'partition_wall' (thickness 10): Define internal rooms.
     - Create solid rectangles for 'living_room', 'bedroom', 'kitchen', 'bathroom', 'corridor', 'staircase', 'elevator_shaft'.
     - All zones must be fully enclosed by walls (exterior or partition).
  4. **LOGICAL CONNECTIVITY (DOORS & REACHABILITY)**:
     - 'door' (size ~30x10 or 10x30): **CRITICAL RULE**: Every room must be accessible.
     - **DOOR PLACEMENT**: A door MUST sit ON a wall (overlap it) and connect two adjacent spaces.
     - **REACHABILITY**: All rooms must connect to the 'corridor' or 'lobby' network. No room should be isolated.
     - 'fire_door': Use for 'staircase' and 'elevator_lobby' entry points.
  5. **CIRCULATION**:
     - The 'corridor' is the backbone. It must touch the doors of all rooms.
  `
};

const floorPlanExamples = {
  complexLayout: `{
    "reasoning_plan": "Step-by-step structural logic",
    "width": 800, "height": 600,
    "elements": [
      {"t": "exterior_wall", "x": 0, "y": 0, "w": 800, "h": 15},
      {"t": "bedroom", "x": 15, "y": 15, "w": 250, "h": 200},
      {"t": "door", "x": 265, "y": 100, "w": 10, "h": 30, "l": "BEDROOM ENTRANCE"}
    ]
  }`
};

const parkingZOrder = [
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
  ElementType.CONVEX_MIRROR
];

export const ParkingScene: SceneDefinition = {
  id: 'parking_underground',
  name: 'Underground Parking',
  description: 'Automated vehicle flow and high-density parking layout.',
  promptConfig: {
    roleDefinition: 'Architectural Spatial Planner specialized in Underground Parking',
    geometricRules: `
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
    `,
    requiredElements: ['wall', 'driving_lane', 'ground', 'entrance', 'exit', 'slope'],
    exampleJSON: `{
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
}`
  },
  styles: createParkingSceneStyles(),
  background: { mode: 'fixed', color: '#020617' },
  zOrder: parkingZOrder,
  elementNormalization: {
    column: ElementType.PILLAR,
    post: ElementType.PILLAR,
    barrier: ElementType.WALL,
    utility_box: ElementType.PILLAR,
    parking_spot: ElementType.PARKING_SPACE,
    parking_bay: ElementType.PARKING_SPACE,
    road: ElementType.ROAD,
    lane: ElementType.ROAD,
    path: ElementType.SIDEWALK,
    pedestrian_walkway: ElementType.SIDEWALK
  }
};

export const GenericScene: SceneDefinition = {
  id: 'scene_generic',
  name: 'Generic 2D Scene',
  description: 'Generic semantic scene layout with flexible element types.',
  promptConfig: {
    roleDefinition: '2D Scene Composer',
    geometricRules: `
1. Use a coherent topology based on the user request (rooms/streets/islands/zones).
2. Keep elements within bounds and avoid large overlaps.
3. Use simple rectangular blocks with short keys (t/x/y/w/h).
    `,
    requiredElements: [],
    exampleJSON: `{
  "reasoning_plan":"Simple street with park and water",
  "width":800,"height":600,
  "elements":[
    {"t":"floor","x":0,"y":0,"w":800,"h":600},
    {"t":"road","x":80,"y":260,"w":640,"h":80},
    {"t":"park","x":120,"y":80,"w":260,"h":140},
    {"t":"water","x":480,"y":80,"w":200,"h":180},
    {"t":"building","x":120,"y":380,"w":220,"h":160}
  ]
}`
  },
  styles: {
    floor: { fill: '#0b1220', opacity: 1 },
    wall: { fill: '#e2e8f0', opacity: 1 },
    road: { fill: '#1e293b', opacity: 1 },
    grass: { fill: '#166534', opacity: 1 },
    park: { fill: '#14532d', opacity: 1 },
    water: { fill: '#1d4ed8', opacity: 0.9 },
    building: { fill: '#64748b', opacity: 1 },
    object: { fill: '#f59e0b', opacity: 1 },
    zone: { fill: '#7c3aed', opacity: 0.6 }
  },
  background: { mode: 'fixed', color: '#020617' },
  zOrder: ['floor', 'grass', 'park', 'water', 'road', 'building', 'wall', 'zone', 'object']
};

export const FloorPlanScene: SceneDefinition = {
  id: 'building_floor_plan',
  name: 'Floor Plan',
  description: 'Architectural floor plan with structural rules.',
  promptConfig: {
    roleDefinition: 'Architectural Floor Plan Designer',
    geometricRules: floorPlanSystemPrompt.rules,
    requiredElements: ['exterior_wall', 'partition_wall', 'corridor', 'door', 'elevator_shaft'],
    exampleJSON: floorPlanExamples.complexLayout
  },
  styles: createFloorPlanSceneStyles(),
  background: { mode: 'styleKey', styleKey: 'floor_slab', fallbackColor: '#020617' },
  customDrawers: {
      [ElementType.STAIRCASE]: (g, element, style, _context) => {
          const w = element.width;
          const h = element.height;
          const cx = w / 2;
          const cy = h / 2;
          
          // Draw base rect
          g.append("rect")
           .attr("width", w).attr("height", h)
           .attr("fill", style.fill)
           .attr("opacity", style.opacity ?? 1)
           .attr("stroke", "none");

          // Draw stair path
          g.append("path")
             .attr("d", w > h ? `M 10 ${cy} L ${w-10} ${cy} M ${w-20} ${cy-5} L ${w-10} ${cy} L ${w-20} ${cy+5}` : `M ${cx} 10 L ${cx} ${h-10} M ${cx-5} ${h-20} L ${cx} ${h-10} L ${cx+5} ${h-20}`)
             .attr("stroke", "#ffffff").attr("fill", "none").attr("stroke-width", 1.5).attr("opacity", 0.5);
      }
  },
  elementNormalization: {
      // 卫生间变体
      toilet: 'bathroom',
      restroom: 'bathroom',
      washroom: 'bathroom',
      wc: 'bathroom',
      // 办公变体
      work_area: 'office',
      workspace: 'office',
      private_office: 'office',
      // 会议变体
      conference: 'conference_room',
      meeting: 'meeting_room',
      boardroom: 'conference_room',
      // 休息/餐饮变体
      break: 'break_room',
      tea_room: 'break_room',
      cafeteria: 'break_room',
      lunchroom: 'break_room',
      // 前台/入口变体
      front_desk: 'reception',
      entrance_hall: 'lobby',
      foyer: 'lobby',
      vestibule: 'lobby',
      waiting_room: 'waiting_area',
      // 储物变体
      closet: 'storage',
      store: 'storage',
      archive: 'storage',
      store_room: 'storage',
      // 设备变体
      utility_room: 'utility',
      service_room: 'utility',
      plant_room: 'utility',
      mep_room: 'utility',
      janitor: 'utility',
      janitor_closet: 'utility',
      // 卧室变体
      master_bedroom: 'bedroom',
      guest_bedroom: 'bedroom',
      child_room: 'bedroom',
      // 餐厅变体
      dining: 'dining_room',
      // 书房变体
      study: 'study',
      library: 'study',
      // 走廊变体
      hallway: 'corridor',
      passage: 'corridor',
      hall: 'corridor',
      // 商业变体
      store_front: 'retail',
      // 墙体统一 → wall
      exterior_wall: 'wall',
      partition_wall: 'wall',
      shear_wall: 'wall',
      parapet: 'wall',
      interior_wall: 'wall',
      // 电梯统一 → elevator
      elevator_shaft: 'elevator',
      elevator_lobby: 'elevator',
      elevator_hall: 'elevator',
  },
  zOrder: [
      ElementType.SLAB,
      ElementType.LIVING_ZONE, ElementType.BEDROOM_ZONE, ElementType.KITCHEN_ZONE, ElementType.BATHROOM_ZONE, ElementType.STORAGE_ZONE,
      'office', 'open_office', 'conference_room', 'meeting_room', 'reception', 'waiting_area',
      'lounge', 'break_room', 'pantry', 'utility', 'server_room', 'electrical_room',
      'retail', 'shop', 'food_court', 'kiosk',
      ElementType.CORRIDOR,
      'cabinet', 'wardrobe', 'desk', 'dining_table', 'sofa', 'bed', 'toilet', 'sink',
      ElementType.SERVICE_SHAFT, ElementType.STAIRCASE, 'elevator',
      'wall', ElementType.PILLAR,
      ElementType.WINDOW, ElementType.DOOR
  ]
};

export const BuildingScene: SceneDefinition = {
  id: 'building',
  name: 'Building',
  description: 'Generate a multi-story building (B1 parking + upper floor plans).',
  promptConfig: {
    roleDefinition: 'Building Orchestrator',
    geometricRules: `Use floor-specific rules based on per-floor scene.`,
    requiredElements: [],
    exampleJSON: `{"floors":[{"id":"B1","sceneId":"parking_underground"},{"id":"1F","sceneId":"building_floor_plan"}]}`
  },
  styles: createFloorPlanSceneStyles(),  // 复用楼层平面样式
  zOrder: []
};

export const DEFAULT_SCENE_ID = ParkingScene.id;

export const SCENE_REGISTRY: Record<string, SceneDefinition> = {
  [ParkingScene.id]: ParkingScene,
  [GenericScene.id]: GenericScene,
  [FloorPlanScene.id]: FloorPlanScene,
  [BuildingScene.id]: BuildingScene
};

export const SCENE_SELECTION_REGISTRY: Record<string, SceneDefinition> = {
  [ParkingScene.id]: ParkingScene,
  [FloorPlanScene.id]: FloorPlanScene,
  [BuildingScene.id]: BuildingScene
};
