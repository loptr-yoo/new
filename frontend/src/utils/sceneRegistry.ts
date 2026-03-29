import { ElementType, SceneDefinition } from '../types';
import { floorPlanSystemPrompt, floorPlanExamples } from './floorPrompts';

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
  styles: {
    [ElementType.GROUND]: { fill: '#334155', opacity: 1 },
    [ElementType.ROAD]: { fill: '#1e293b', opacity: 1 },
    [ElementType.PARKING_SPACE]: { fill: '#3b82f6', opacity: 1 },
    [ElementType.SIDEWALK]: { fill: '#1e293b', opacity: 1 },
    [ElementType.RAMP]: { fill: '#c026d3', opacity: 1 },
    [ElementType.PILLAR]: { fill: '#94a3b8', opacity: 1, rx: 4 },
    [ElementType.WALL]: { fill: '#f1f5f9', opacity: 1 },
    [ElementType.ENTRANCE]: { fill: '#15803d', opacity: 1 },
    [ElementType.EXIT]: { fill: '#b91c1c', opacity: 1 },
    [ElementType.STAIRCASE]: { fill: '#b45309', opacity: 1 },
    [ElementType.ELEVATOR]: { fill: '#06b6d4', opacity: 1 },
    [ElementType.ELEVATOR_SHAFT]: { fill: '#991b1b', opacity: 1 },
    [ElementType.CHARGING_STATION]: { fill: '#22c55e', opacity: 1 },
    [ElementType.GUIDANCE_SIGN]: { fill: '#f59e0b', opacity: 1 },
    [ElementType.SAFE_EXIT]: { fill: '#10b981', opacity: 1 },
    [ElementType.SPEED_BUMP]: { fill: '#eab308', opacity: 1 },
    [ElementType.FIRE_EXTINGUISHER]: { fill: '#ef4444', opacity: 1 },
    [ElementType.LANE_LINE]: { fill: 'none', opacity: 1, stroke: '#facc15', strokeWidth: 1.5 },
    [ElementType.CONVEX_MIRROR]: { fill: '#38bdf8', opacity: 1 }
  },
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
  styles: {
    [ElementType.SLAB]: { fill: '#1e293b', opacity: 0.15 },
    [ElementType.ROAD]: { fill: '#1e293b', opacity: 1 },
    [ElementType.WALL_EXTERNAL]: { fill: '#475569', opacity: 1 },
    [ElementType.WALL_INTERNAL]: { fill: '#475569', opacity: 1 },
    [ElementType.SHEAR_WALL]: { fill: '#475569', opacity: 1 },
    [ElementType.PILLAR]: { fill: '#475569', opacity: 1 },
    [ElementType.ELEVATOR_SHAFT]: { fill: '#991b1b', opacity: 1 },
    [ElementType.STAIRCASE]: { fill: '#b45309', opacity: 1 },
    [ElementType.ELEVATOR]: { fill: '#06b6d4', opacity: 1 },
    [ElementType.SERVICE_SHAFT]: { fill: '#064e3b', opacity: 1 },
    [ElementType.LIVING_ZONE]: { fill: '#1e293b', opacity: 0.15 },
    [ElementType.BEDROOM_ZONE]: { fill: '#1e293b', opacity: 0.15 },
    [ElementType.KITCHEN_ZONE]: { fill: '#1e293b', opacity: 0.15 },
    [ElementType.BATHROOM_ZONE]: { fill: '#1e293b', opacity: 0.15 },
    [ElementType.CORRIDOR]: { fill: '#1e293b', opacity: 0.15 },
    [ElementType.STORAGE_ZONE]: { fill: '#1e293b', opacity: 0.15 },
    'bed': { fill: '#4338ca', opacity: 1 },
    'sofa': { fill: '#be123c', opacity: 1 },
    'dining_table': { fill: '#78350f', opacity: 1 },
    'desk': { fill: '#0e7490', opacity: 1 },
    'wardrobe': { fill: '#6d28d9', opacity: 1 },
    'toilet': { fill: '#f8fafc', opacity: 1 },
    'sink': { fill: '#1d4ed8', opacity: 1 },
    'cabinet': { fill: '#059669', opacity: 1 },
    [ElementType.DOOR]: { fill: '#f8fafc', opacity: 1 },
    [ElementType.WINDOW]: { fill: '#7dd3fc', opacity: 0.8 },
  },
  customDrawers: {
      [ElementType.STAIRCASE]: (g, element, style, context) => {
          const w = element.width;
          const h = element.height;
          const cx = w / 2;
          const cy = h / 2;
          
          // Draw base rect
          g.append("rect")
           .attr("width", w).attr("height", h)
           .attr("fill", style.fill)
           .attr("opacity", style.opacity)
           .attr("stroke", "none");

          // Draw stair path
          g.append("path")
             .attr("d", w > h ? `M 10 ${cy} L ${w-10} ${cy} M ${w-20} ${cy-5} L ${w-10} ${cy} L ${w-20} ${cy+5}` : `M ${cx} 10 L ${cx} ${h-10} M ${cx-5} ${h-20} L ${cx} ${h-10} L ${cx+5} ${h-20}`)
             .attr("stroke", "#ffffff").attr("fill", "none").attr("stroke-width", 1.5).attr("opacity", 0.4);
      }
  },
  zOrder: [
      ElementType.SLAB,
      ElementType.LIVING_ZONE, ElementType.BEDROOM_ZONE, ElementType.KITCHEN_ZONE, ElementType.BATHROOM_ZONE, ElementType.STORAGE_ZONE,
      ElementType.CORRIDOR, 
      'cabinet', 'wardrobe', 'desk', 'dining_table', 'sofa', 'bed', 'toilet', 'sink',
      ElementType.SERVICE_SHAFT, ElementType.STAIRCASE, ElementType.ELEVATOR_SHAFT,
      ElementType.WALL_INTERNAL, ElementType.WALL_EXTERNAL, ElementType.SHEAR_WALL, ElementType.PILLAR,
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
  styles: {},
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
