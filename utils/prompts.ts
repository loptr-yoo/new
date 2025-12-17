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
       - Draw paths connecting 'ground' areas to 'staircase' or 'elevator' locations.
    4. **Layer 4: Facilities**
       - 'staircase' (30x30) + 'safe_exit' (20x20) placed together on 'ground' areas near the corners.
       - 'elevator' (20x20), 'fire_extinguisher' (10x10) spread out.

    **OUTPUT FORMAT**:
    - JSON with 'reasoning_plan' and 'elements'.
    - Short keys: t, x, y, w, h.
    
    **EXAMPLE**:
    \`\`\`json
    {
      "reasoning_plan": "Adding inset pillars and lane lines.",
      "elements": [
        {"t": "pillar", "x": 105, "y": 105, "w": 20, "h": 20}, 
        {"t": "ground_line", "x": 400, "y": 300, "w": 2, "h": 10}
      ]
    }
    \`\`\`
  `,

  fix: (layout: ParkingLayout, violations: ConstraintViolation[]) => `
    You are a **Topological Constraint Solver** acting as a Lead Engineer.
    Your objective is to break the loop between "Overlap Errors", "Connectivity Errors", and "Perimeter Gaps" by applying SURGICAL fixes.

    **INPUT CONTEXT**:
    - Canvas: ${layout.width}x${layout.height}
    - Violations: ${JSON.stringify(violations)}
    - Elements: ${JSON.stringify(layout.elements)}

    **CORE PHILOSOPHY**: 
    1. **Connectivity requires Touching**: It is acceptable for elements to share an edge (overlap < 2px). DO NOT move elements if the overlap is trivial.
    2. **Closed Loop**: The perimeter MUST be sealed. No "black holes" at corners.
    3. **Ground Integrity (CRITICAL)**: 'ground' elements support 'parking_space'. **NEVER SHRINK 'ground'** in a way that creates gaps between the ground and the road. The ground MUST touch the road.

    **HIERARCHY OF TRUTH (Strict Priority)**:
    1. **Immutable**: Walls (ID: wall_*) & Entrances/Exits - NEVER MOVE.
    2. **Foundation**: Ground/Islands - **RESIZE EDGE** only to align with roads. **NEVER DELETE**.
    3. **Connectors**: Ramps/Slopes - Move or Resize to maintain connection.
    4. **Flexible**: Driving Lanes - Can be shifted or resized.
    5. **Disposable**: Parking Spaces / Pillars - DELETE if they cause unresolvable conflicts.

    **SURGICAL EXECUTION PLAN**:

    **STEP 1: PERIMETER & CORNER INTEGRITY (Critical)**
    - Check for 'out_of_bounds' or 'wall_gap' violations.
    - **ACTION**: Extend 'wall' elements specifically to cover corners (0,0), (${layout.width},0), (${layout.width},${layout.height}), (0,${layout.height}) if gaps exist.

    **STEP 2: HANDLING "FAKE" OVERLAPS**
    - Analyze 'Overlap' violations.
    - **RULE**: If overlap is < 2 pixels (just touching), **IGNORE** the move. It is a valid connection.

    **STEP 3: FIXING CONNECTIVITY (The "Bridge" Logic)**
    - Target: Disconnected 'entrance' or 'exit'.
    - **ACTION**: SNAP the connecting 'slope'/'ramp' coordinates to exactly match the Entrance/Exit edge.

    **STEP 4: RESOLVING HARD COLLISIONS (Revised for Ground)**
    - **CRITICAL**: IF 'ground' overlaps 'road' -> **SNAP** the ground edge to exactly match the road edge. **DO NOT SHRINK** beyond the road boundary. Creating a gap (void) is worse than a 1px overlap.
    - IF 'parking_space' overlaps 'wall' OR 'driving_lane' -> **DELETE** the parking space immediately.
    - IF 'driving_lane' overlaps 'wall' -> **SHIFT** the lane 2 units away.

    **OUTPUT FORMAT**:
    - Return the **FULL** JSON layout (all elements).
    - Include "fix_strategy" list describing specific actions taken.

    **JSON EXAMPLE**:
    \`\`\`json
    {
      "fix_strategy": [
        "Extended Wall_Top to (800,0) to close corner gap.",
        "Ignored minor edge overlap between Slope_1 and Road_A.",
        "Aligned Ground_Central edge to Road_A (Snapping).",
        "Deleted Parking_Spot_4 due to collision."
      ],
      "width": ${layout.width}, "height": ${layout.height},
      "elements": [ ... ]
    }
    \`\`\`
  `
};