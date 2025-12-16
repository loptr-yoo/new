import { ParkingLayout, ConstraintViolation } from '../types';

export const PROMPTS = {
  generation: (description: string) => `
  You are an **Architectural Spatial Planner**. 
  Generate a COARSE-GRAINED JSON underground parking layout (0,0 at top-left) for: "${description}".
  
  **CANVAS CONSTRAINTS**: Width: 800, Height: 600.
  
  **REQUIRED ELEMENTS (Structural Foundation)**:
  - 'wall': Perimeter boundaries (Must enclose the area).
  - 'driving_lane': Main vehicle arteries. Width must be approx 60.
  - 'ground': Parking islands/areas.
  - 'entrance' / 'exit': 40x20 blocks ON the perimeter.
  - 'slope': 40x60 connectors joining Entrance/Exit to Roads.

  **CRITICAL RULES**:
  1. **Topology**: The 'driving_lane' elements MUST form a **CLOSED LOOP** or a fully connected graph. No dead ends.
  2. **Connectivity**: 
     - Entrance -> Slope -> Road -> Slope -> Exit.
     - Slopes MUST physically touch the Road (0 distance).
  3. **Pillars**: Do NOT generate pillars in this step. They will be added later.

  **REASONING PLAN**:
  1. Define the Road Loop shape (e.g., Rectangular Loop).
  2. Place Entrance/Exit on opposite walls.
  3. Fill empty spaces with 'ground'.

  **JSON EXAMPLE**:
  \`\`\`json
  {
    "reasoning_plan": "Create a central loop road. Place grounds in the middle and edges.",
    "width": 800, "height": 600,
    "elements": [
      {"t": "wall", "x": 0, "y": 0, "w": 800, "h": 20},
      {"t": "driving_lane", "x": 20, "y": 80, "w": 760, "h": 60}
      // ... other basic elements only
    ]
  }
  \`\`\`
  `,

  refinement: (simplifiedLayout: any, width: number, height: number) => `
    You are a **Spatial Algorithm Engine**.
    Task: Inject details into the layout while preserving the road network.

    **INPUT**: Canvas ${width}x${height}, Layout with ${simplifiedLayout.elements.length} elements.

    **CRITICAL RULE**: 
    - **IMMUTABILITY**: Keep ALL existing 'wall', 'driving_lane', 'slope', 'entrance', 'exit'.
    - **SUPERSET**: Add new elements ON TOP.

    **GENERATION TASKS**:

    1. **Layer 1: Structural Grid ('pillar')**
       - **Target**: Iterate through 'ground' elements.
       - **Algorithm**: Place a 'pillar' (20x20) at the 4 corners of large 'ground' islands.
       - **Constraint**: Pillars must strictly be INSIDE the 'ground' area. Do NOT place them on roads.

    2. **Layer 2: Road Markings ('ground_line')**
       - Add center lines to 'driving_lane' elements.
       - Logic: If w > h (Horizontal), line is horizontal at y+h/2. If h > w (Vertical), line is vertical at x+w/2.
       - Skip intersections.

    3. **Layer 3: Pedestrian Paths ('pedestrian_path')**
       - Connect 'ground' islands to nearest roads/exits using stripes (w=16).

    4. **Layer 4: Facilities ('staircase', 'safe_exit', 'elevator')**
       - Place in the center of the largest 'ground' islands.
       - 'staircase' (30x30) + 'safe_exit' (20x20) must be adjacent.

    **OUTPUT FORMAT**:
    - JSON with 'reasoning_plan' and 'elements'.
    - Short keys: t, x, y, w, h.
    
    **EXAMPLE**:
    \`\`\`json
    {
      "reasoning_plan": "Detected 4 grounds. Adding pillars at corners. Adding road lines...",
      "elements": [
        // ... originals ...
        {"t": "pillar", "x": 100, "y": 100, "w": 20, "h": 20}, // Added Pillar
        {"t": "ground_line", "x": 50, "y": 110, "w": 200, "h": 2}
      ]
    }
    \`\`\`
  `,

  fix: (layout: ParkingLayout, violations: ConstraintViolation[]) => `
    You are a **Topological Constraint Solver** acting as a Lead Engineer.
    Your objective is to break the loop between "Overlap Errors" and "Connectivity Errors".

    **INPUT CONTEXT**:
    - Canvas: ${layout.width}x${layout.height}
    - Violations: ${JSON.stringify(violations)}
    - Elements: ${JSON.stringify(layout.elements)}

    **CORE PHILOSOPHY**: 
    **"Connectivity requires Touching."** It is acceptable for a 'slope'/'ramp' to share an edge (overlap = 0 or 1px) with a 'driving_lane' or 'entrance'. 
    Do NOT shrink elements to create gaps if it breaks the path.

    **HIERARCHY OF TRUTH**:
    1. **Immutable**: Walls (ID: wall_*) - NEVER MOVE.
    2. **Anchors**: Entrances/Exits - NEVER MOVE.
    3. **Connectors**: Ramps/Slopes - Must strictly bridge Anchors and Roads.
    4. **Flexible**: Driving Lanes - Can be resized/moved.
    5. **Disposable**: Parking Spaces - DELETE if overlapping anything.

    **SURGICAL EXECUTION PLAN**:

    **STEP 1: HANDLING "FAKE" OVERLAPS (Adjacency)**
    - Analyze 'Overlap' violations involving 'slope' or 'ramp'.
    - **RULE**: If a Slope overlaps a Road/Entrance by a tiny margin (e.g., < 2 pixels) along the edge, **IGNORE** the move. It is a valid connection.
    - **RULE**: Only if the overlap is deep (> 5 pixels), **RESIZE** the Slope slightly to just *kiss* the edge of the Road.

    **STEP 2: FIXING CONNECTIVITY (The "Bridge" Logic)**
    - Identify disconnected 'entrance' or 'exit'.
    - **ACTION**: Find or Create a 'slope' (40x60).
    - **MATH**: 
      - If Entrance is at Top (y=0): Set Slope { x: entrance.x, y: 20, w: entrance.w, h: 60 }.
      - If Exit is at Bottom (y=Height-20): Set Slope { x: exit.x, y: exit.y - 60, w: exit.w, h: 60 }.
      - Ensure the Road connects to the *other* end of the Slope.

    **STEP 3: RESOLVING HARD COLLISIONS**
    - IF 'parking_space' overlaps 'wall' OR 'driving_lane' -> **DELETE** the parking space.
    - IF 'driving_lane' overlaps 'wall' -> **SHIFT** the lane 1-2 units away from the wall.

    **OUTPUT FORMAT**:
    - Return the **FULL** JSON layout (do not omit valid elements).
    - Provide a "fix_strategy" list explanation.

    **FEW-SHOT THINKING**:
    - "Violation says Slope overlaps Road. Checking coords... They share edge y=80. This is a valid connection. Keeping positions."
    - "Violation says Entrance disconnected. Moving Ramp to x=Entrance.x and y=Entrance.y + Height."

    **JSON EXAMPLE**:
    \`\`\`json
    {
      "fix_strategy": [
        "Ignored minor edge overlap between Slope_1 and Road_A to maintain connectivity.",
        "Deleted Parking_Spot_4 due to wall collision.",
        "Snapped Slope_2 to Exit coordinates."
      ],
      "width": 800, "height": 600,
      "elements": [ ... ]
    }
    \`\`\`
  `
};