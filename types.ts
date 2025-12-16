export enum ElementType {
  GROUND = 'ground',
  PARKING_SPACE = 'parking_space',
  ROAD = 'driving_lane', // Renamed for clarity based on prompt, mapped from 'road'
  SIDEWALK = 'pedestrian_path', // Renamed for clarity
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
  CONVEX_MIRROR = 'convex_mirror'
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
}

export interface ParkingLayout {
  width: number;
  height: number;
  elements: LayoutElement[];
}

export interface ConstraintViolation {
  elementId: string;
  targetId?: string; // If colliding with another element
  type: 'overlap' | 'out_of_bounds' | 'invalid_dimension' | 'placement_error' | 'connectivity_error' | 'width_mismatch';
  message: string;
}