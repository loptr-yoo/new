import { ParkingLayout, ConstraintViolation } from '../types';

export const PROMPTS = {
  generation: (description: string) => `
  You are an **Architectural Spatial Planner**. 
  Generate a COARSE-GRAINED JSON underground parking layout (0,0 at top-left) for: "${description}".
  
  **CANVAS CONSTRAINTS**: Width: 800, Height: 600.
  
  **CRITICAL GEOMETRIC RULES**:
  1. **CLOSED LOOP PERIMETER**: Walls MUST overlap or touch at corners (0,0), (800,0), (800,600), (0,600). DO NOT leave black gaps.
  2. **The "Racetrack" Pattern**:
     - Create a main loop of 'driving_lane' (Roads).
     - **MANDATORY SETBACK**: The Road Loop must be **INSET** from the perimeter walls by approx **50-60 units**.
     - This gap allows for parking 'ground' strips between the wall and the road.
  3. **'ground' Elements**:
     - Grounds must be rectangular **STRIPS** (width/height ~50-100), NOT massive blocks.
     - Never generate a 'ground' thicker than 120 units.
  4. **Boundary Snapping**:
     - 'entrance' and 'exit' MUST touch the edges of the 800x600 canvas.

  **REQUIRED ELEMENTS**:
  - 'wall': Perimeter boundaries.
  - 'driving_lane': Main vehicle arteries (Width ~60).
  - 'ground': Parking islands.
  - 'entrance' / 'exit': 40x20 blocks on boundary.
  - 'slope': 40x60 connectors joining Entrance/Exit to Roads.

  **JSON EXAMPLE**:
  \`\`\`json
  {
    "reasoning_plan": "Closed wall loop with inset racetrack driving lane.",
    "width": 800, "height": 600,
    "elements": [
      {"t": "wall", "x": 0, "y": 0, "w": 800, "h": 20},
      {"t": "driving_lane", "x": 60, "y": 60, "w": 680, "h": 60}
    ]
  }
  \`\`\`
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

    **INCREMENTAL UPDATE**:
    - Return **ONLY** the NEW elements you are creating. Existing walls/roads will be preserved.

    **GENERATION TASKS**:
    1. **Layer 1: Structural Grid ('pillar')**
       - Place 'pillar' (size 10x10) at corners of 'ground' areas.
       - Max 1 pillar every 100-150 units. Sparsity is key.
       - Pillars provide structural integrity to the parking islands.
       - **CONSTRAINT**: Pillars must be INSET by 5 units from any edge.

    2. **Layer 2: Road Logic**
       - **'ground_line'**: Dashed lines (width 2) in the center of driving lanes. Skip intersections.
       - **'guidance_sign'**: (10x10) Place at T-junctions.
       - **'deceleration_zone'**: (10x40) Place near Entrances/Exits.

    3. **Layer 3: Pedestrian Paths ('pedestrian_path')**
       - Sidewalks (w=16) connecting ground islands to exits.

    4. **Layer 4: Facilities**
       - **'staircase'** (30x30) + **'safe_exit'** (20x20) MUST be placed adjacent to each other.
       - **'elevator'** (20x20), **'fire_extinguisher'** (10x10).

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
    Your objective is to break the loop between "Overlap Errors" and "Connectivity Errors" by applying SURGICAL fixes.

    **INPUT CONTEXT**:
    - Canvas: ${layout.width}x${layout.height}
    - Violations: ${JSON.stringify(violations)}
    - Elements: ${JSON.stringify(layout.elements)}

    **CORE PHILOSOPHY**: 
    **"Connectivity requires Touching."** It is acceptable for a 'slope'/'ramp' to share an edge (overlap = 0 or 1px) with a 'driving_lane' or 'entrance'. 
    Do NOT shrink elements to create gaps if it breaks the path.

    **HIERARCHY OF TRUTH (Strict Priority)**:
    1. **Immutable**: Walls (ID: wall_*) - NEVER MOVE.
    2. **Anchors**: Entrances/Exits - NEVER MOVE.
    3. **Connectors**: Ramps/Slopes - Move or Resize to maintain connection.
    4. **Flexible**: Driving Lanes - Can be shifted.
    5. **Disposable**: Parking Spaces / Pillars - DELETE if they cause unresolvable conflicts.

    **SURGICAL EXECUTION PLAN**:

    **STEP 1: HANDLING "FAKE" OVERLAPS (Adjacency)**
    - Analyze 'Overlap' violations involving 'slope' or 'ramp'.
    - **RULE**: If elements overlap by < 2 pixels (just touching), **IGNORE** the move. It is a valid connection.
    - **RULE**: Only if overlap is deep (> 5px), **RESIZE** slightly.

    **STEP 2: FIXING CONNECTIVITY (The "Bridge" Logic)**
    - Target: Disconnected 'entrance' or 'exit'.
    - **ACTION**: Find or Create a 'slope' (40x60).
    - **MATH**: SNAP the slope's coordinate to exactly match the Entrance/Exit edge.

    **STEP 3: RESOLVING HARD COLLISIONS**
    - IF 'parking_space' overlaps 'wall' OR 'driving_lane' -> **DELETE** the parking space immediately.
    - IF 'driving_lane' overlaps 'wall' -> **SHIFT** the lane 2 units away.

    **OUTPUT FORMAT**:
    - Return the **FULL** JSON layout (all elements).
    - Include "fix_strategy" list.

    **JSON EXAMPLE**:
    \`\`\`json
    {
      "fix_strategy": [
        "Ignored minor edge overlap between Slope_1 and Road_A.",
        "Deleted Parking_Spot_4 due to wall collision.",
        "Snapped Slope_2 to Exit coordinates."
      ],
      "width": 800, "height": 600,
      "elements": [ ... ]
    }
    \`\`\`
  `
};
