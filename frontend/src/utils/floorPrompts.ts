export const floorPlanSystemPrompt = {
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

export const floorPlanExamples = {
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
