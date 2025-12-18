import { ParkingLayout, ConstraintViolation } from '../types';

export const PROMPTS = {
  generation: (description: string) => `
  You are an **Architectural Spatial Planner**. 
  Generate a COARSE-GRAINED JSON underground parking layout (0,0 at top-left) for: "${description}".
  
  **CANVAS CONSTRAINTS**: Width: 800, Height: 600.
  
  **CRITICAL GEOMETRIC RULES**:
  1. **CLOSED LOOP PERIMETER**: Walls MUST overlap or touch at corners. NO perimeter gaps.
  2. **The "Racetrack" Pattern**:
     - Create a main loop of 'driving_lane' (Roads).
     - **MANDATORY SETBACK**: The Road Loop must be **INSET** from the perimeter walls.
  3. **'ground' Elements (CRITICAL FOR VOID FIXING)**:
     - **NO FLOATING ISLANDS**: Every 'ground' element MUST touch a 'driving_lane' or another 'ground' on all sides.
     - **INTERNAL FILL**: The empty space INSIDE the road loop (the "donut hole") must be **100% FILLED** with 'ground' strips.
     - **STRIP LOGIC**: If splitting the center into multiple 'ground' strips, they must **TOUCH** (e.g., y of Strip B = y + height of Strip A). **DO NOT leave black gaps between strips.**
  4. **Boundary Snapping**:
     - 'entrance' and 'exit' MUST touch the edges of the canvas.
  5. **ZERO-VOID POLICY**:
     - The final layout must look like a **Solid Mosaic**. 
     - Visible Background Color = ERROR. 
     - Any space not occupied by a 'wall' or 'driving_lane' MUST be covered by 'ground'.

  **REQUIRED ELEMENTS**:
  - 'wall': Perimeter boundaries.
  - 'driving_lane': Main vehicle arteries (Width ~60).
  - 'ground': Parking islands (Must fill all voids).
  - 'entrance' / 'exit': 40x20 blocks on boundary.
  - 'slope': 40x60 connectors joining Entrance/Exit to Roads.

  **JSON EXAMPLE**:
  \`\`\`json
  {
    "reasoning_plan": "Racetrack road with solid central island ground strips touching each other.",
    "width": 800, "height": 600,
    "elements": [
      {"t": "wall", "x": 0, "y": 0, "w": 800, "h": 20},
      {"t": "driving_lane", "x": 60, "y": 60, "w": 680, "h": 60},
      {"t": "ground", "x": 120, "y": 120, "w": 560, "h": 100}, 
      {"t": "ground", "x": 120, "y": 220, "w": 560, "h": 100} 
    ]
  }
  \`\`\`
  // Note in example: y:220 is exactly y:120 + h:100. They touch.
  `,

  refinement: (simplifiedLayout: any, width: number, height: number) => `
    You are a **Spatial Algorithm Engine**.
    Task: Inject NEW detailed structural and facility elements into the existing layout.

    **INPUT DATA**: 
    - Canvas: ${width}x${height}
    - Existing Elements: 
    ${JSON.stringify(simplifiedLayout.elements)}

    **CRITICAL DESIGN RULES**:
    - **FACILITY PLACEMENT**: 'staircase', 'elevator', and 'safe_exit' MUST be placed on 'ground' elements. They are FORBIDDEN from being on 'driving_lane'.
    - **SPEED BUMP ORIENTATION**: 'deceleration_zone' must be PERPENDICULAR to the road direction. If the road is horizontal (width > height), the bump must be vertical (height > width).
    - **SIDEWALKS**: 'pedestrian_path' must cross the 'driving_lane' to connect 'ground' areas.

    **CRITICAL SYSTEM ARCHITECTURE**:
    - **Algorithmic Spot Filler**: A deterministic algorithm will automatically fill all 'ground' strips with 'parking_spot' elements after your turn. 
    - **YOUR FOCUS**: You must place facilities (stairs, elevators), pillars at row ends, and road markings. DO NOT waste tokens drawing hundreds of parking spots.
    - **IMMUTABILITY RULE**: You are **FORBIDDEN** from outputting, modifying, or deleting 'wall', 'driving_lane', or 'ground' elements. Only output NEW detail elements.

    **INCREMENTAL UPDATE**:
    - Return **ONLY** the NEW elements you are creating. 

    **GENERATION TASKS**:
    1. **Layer 1: Structural Grid ('pillar')**
       - Place 'pillar' (size 10x10) at corners of 'ground' areas.
       - Max 1 pillar every 100-150 units. Sparsity is key.
       - Pillars provide structural integrity to the parking islands.
    2. **Layer 2: Road Logic**
       - 'ground_line': Dashed lines (width 2) in center of 'driving_lane' areas.
       - 'guidance_sign': (10x10) at road junctions to indicate Exit direction.
       - 'deceleration_zone': (10x40) Place near Entrances/Exits.

    3. **Layer 3: Pedestrian Paths ('pedestrian_path')**
       - Draw zebra crossings connecting 'ground' areas across roads.
    4. **Layer 4: Facilities**
       - 'staircase' (30x30) + 'safe_exit' (20x20) placed together on 'ground' areas near the corners.
       - 'elevator' (20x20), 'fire_extinguisher' (10x10) spread out.

    **OUTPUT FORMAT**:
    - JSON with 'reasoning_plan' and 'elements'.
    - Short keys: t, x, y, w, h.
  `,

  fix: (layout: ParkingLayout, violations: ConstraintViolation[]) => `
    You are a **Topological Constraint Solver**.
    
    **INPUT**: ${layout.width}x${layout.height} Canvas.
    **VIOLATIONS**: ${JSON.stringify(violations)}

    **CRITICAL RULES**:
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

    **OUTPUT**: Return the FULL JSON layout with "fix_strategy" list.
  `
};