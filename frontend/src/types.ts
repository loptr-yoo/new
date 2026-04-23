export enum ElementType {
  // --- Parking / General ---
  GROUND = 'ground',
  PARKING_SPACE = 'parking_space',
  ROAD = 'driving_lane', 
  SIDEWALK = 'pedestrian_path',
  RAMP = 'slope',
  PILLAR = 'pillar',
  WALL = 'wall',
  ENTRANCE = 'entrance',
  EXIT = 'exit',
  STAIRCASE = 'staircase',
  ELEVATOR = 'elevator',
  CHARGING_STATION = 'charging_station',
  GUIDANCE_SIGN = 'guidance_sign',
  SAFE_EXIT = 'safe_exit',
  SPEED_BUMP = 'deceleration_zone',
  FIRE_EXTINGUISHER = 'fire_extinguisher',
  LANE_LINE = 'ground_line',
  CONVEX_MIRROR = 'convex_mirror',

  // --- Floor Plan Structure ---
  SLAB = 'floor_slab',
  WALL_EXTERNAL = 'exterior_wall',
  WALL_INTERNAL = 'partition_wall',
  SHEAR_WALL = 'shear_wall',
  // PILLAR exists above
  ROOF_PARAPET = 'parapet',
  
  // --- Vertical Circulation ---
  ELEVATOR_SHAFT = 'elevator_shaft',
  // STAIRCASE exists above
  SERVICE_SHAFT = 'pipe_shaft',
  ELEVATOR_HALL = 'elevator_lobby',

  // --- Horizontal Circulation ---
  CORRIDOR = 'corridor',
  LOBBY = 'lobby',
  AISLE = 'aisle',

  // --- Apertures ---
  DOOR = 'door',
  FIRE_DOOR = 'fire_door',
  WINDOW = 'window',
  // ENTRANCE/EXIT exist above

  // --- Extensions ---
  BALCONY = 'balcony',
  TERRACE = 'terrace',
  AIR_CON_LEDGE = 'ac_ledge',
  WINDOW_SILL = 'window_sill',

  // --- Functional Zones ---
  KITCHEN_ZONE = 'kitchen',
  BATHROOM_ZONE = 'bathroom',
  LIVING_ZONE = 'living_room',
  BEDROOM_ZONE = 'bedroom',
  STORAGE_ZONE = 'storage',

  // --- MEP / Safety ---
  FIRE_HYDRANT = 'fire_hydrant',
  SAFE_EXIT_SIGN = 'exit_sign',
  HVAC_VENT = 'hvac_vent',
  ELECTRICAL_BOX = 'dist_box'
}

export interface LayoutElement {
  id: string;
  type: ElementType | string; // Allow string for backward compatibility/flexibility
  x: number;
  y: number;
  width: number;
  height: number;
  rotation?: number; // Degrees
  label?: string;
  subType?: string; // For things like 'lane_line' direction
  forward?: [number, number, number]; // 车位指向行车道的方向向量，右/下/上/左为主轴
  polygon?: [number, number][]; // 精确多边形轮廓（优先于矩形渲染）
}

export interface ParkingLayout {
  width: number;
  height: number;
  elements: LayoutElement[];
  sceneId?: string;
}

export interface ConstraintViolation {
  elementId: string;
  targetId?: string; // If colliding with another element
  type: 'overlap' | 'out_of_bounds' | 'invalid_dimension' | 'placement_error' | 'connectivity_error' | 'width_mismatch';
  message: string;
}

export interface ElementStyle {
  fill: string;
  opacity: number;
  stroke?: string;
  strokeWidth?: number;
  rx?: number;
}

export interface DrawerContext {
  layout: ParkingLayout;
  violations: ConstraintViolation[];
}

export type ElementDrawer = (
  g: any,
  element: LayoutElement,
  style: ElementStyle,
  context: DrawerContext
) => void;

export type LayoutAlgorithm = (layout: ParkingLayout) => ParkingLayout;

export type SceneBackgroundConfig =
  | { mode: 'fixed'; color: string }
  | { mode: 'styleKey'; styleKey: string; fallbackColor?: string };

export interface SceneDefinition {
  id: string;
  name: string;
  description: string;
  promptConfig: {
    roleDefinition: string;
    geometricRules: string;
    requiredElements: string[];
    exampleJSON: string;
  };
  styles: Record<string, ElementStyle>;
  background?: SceneBackgroundConfig;
  customDrawers?: Record<string, ElementDrawer>;
  zOrder?: string[];
  elementNormalization?: Record<string, string>;
  postProcessAlgorithms?: LayoutAlgorithm[];
}

export interface BuildingBlueprint {
  coreElements: LayoutElement[];
}

export interface BuildingFloor {
  id: string;
  name: string;
  layout: ParkingLayout;
}

export interface BuildingData {
  blueprint: LayoutElement[];
  floors: Record<string, ParkingLayout>;
}

// ============================================================
// V2 新管线类型（/api/building/generate）
// ============================================================

export interface RoomResultV2 {
  room_id: string;
  room_type: string;
  floor_id: string;
  polygon: [number, number][];
  center: [number, number];
  area: number;
  width: number;
  depth: number;
  target_area: number;
  has_window: boolean;
}

export interface CoreTubeV2 {
  boundary: [number, number][];
  center: [number, number];
  area: number;
  elevator?: { polygon: [number, number][]; area: number };
  staircase?: { polygon: [number, number][]; area: number };
}

export interface WallV2 {
  type: string;                    // 'exterior_wall' | 'partition_wall'
  coords: [number, number][];
  polygon?: [number, number][];    // buffer 后的精确矩形 polygon
  thickness: number;
  length: number;
  room_ids: string[];
}

export interface DoorV2 {
  position: [number, number];
  width: number;
  connects: string[];
  rotation?: number;               // 0=水平, 90=垂直
  thickness?: number;
  forward?: [number, number, number];
}

export interface WindowV2 {
  position: [number, number];
  width: number;
  room_id: string;
  rotation?: number;
  thickness?: number;
  forward?: [number, number, number];
}

export interface DegradationSummary {
  total_degradations: number;
  skipped_rooms: string[];
  miqp_fallback_floors: string[];
  adjacency_dropped: number;
  unreachable_rooms: string[];
  parse_fixes: number;
}

export interface CorridorV2 {
  id: string;
  type: string;
  polygon: [number, number][];
  center: [number, number];
  width: number;
  depth: number;
  area: number;
  orientation: string;
}

export interface FloorSlabV2 {
  id: string;
  type: string;
  polygon: [number, number][];
  center: [number, number];
  width: number;
  depth: number;
  area: number;
}

export interface BuildingFloorV2 {
  floor_name: string;
  floor_slab?: FloorSlabV2;
  corridors?: CorridorV2[];
  rooms: RoomResultV2[];
  walls?: WallV2[];
  doors?: DoorV2[];
  windows?: WindowV2[];
  generation_time_ms: number;
  warnings: string[];
}

export interface BuildingDataV2 {
  building: {
    width: number;
    depth: number;
    floors: Record<string, BuildingFloorV2>;
  };
  core_tube: CoreTubeV2 | null;
  warnings: string[];
  degradation_summary?: DegradationSummary;
}
