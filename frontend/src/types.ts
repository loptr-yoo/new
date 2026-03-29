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
