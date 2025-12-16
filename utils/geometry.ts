import { LayoutElement, ParkingLayout, ConstraintViolation, ElementType } from '../types';

// --- SPATIAL PARTITIONING SYSTEM ---
class SpatialGrid {
    private grid = new Map<string, LayoutElement[]>();
    private cellSize: number;

    constructor(cellSize = 100) {
        this.cellSize = cellSize;
    }

    private getKey(x: number, y: number): string {
        return `${Math.floor(x / this.cellSize)},${Math.floor(y / this.cellSize)}`;
    }

    // Get all cell keys an element touches
    private getKeysForElement(el: LayoutElement): string[] {
        const startX = Math.floor(el.x / this.cellSize);
        const endX = Math.floor((el.x + el.width) / this.cellSize);
        const startY = Math.floor(el.y / this.cellSize);
        const endY = Math.floor((el.y + el.height) / this.cellSize);

        const keys: string[] = [];
        for (let x = startX; x <= endX; x++) {
            for (let y = startY; y <= endY; y++) {
                keys.push(`${x},${y}`);
            }
        }
        return keys;
    }

    add(el: LayoutElement) {
        const keys = this.getKeysForElement(el);
        keys.forEach(key => {
            if (!this.grid.has(key)) {
                this.grid.set(key, []);
            }
            this.grid.get(key)!.push(el);
        });
    }

    // Get potential collision candidates (broad phase)
    getPotentialCollisions(el: LayoutElement): LayoutElement[] {
        const keys = this.getKeysForElement(el);
        const candidates = new Set<LayoutElement>();
        
        keys.forEach(key => {
            const cell = this.grid.get(key);
            if (cell) {
                cell.forEach(candidate => {
                    if (candidate.id !== el.id) {
                        candidates.add(candidate);
                    }
                });
            }
        });

        return Array.from(candidates);
    }
}

// --- CACHED GEOMETRY UTILS ---
const cornerCache = new Map<string, {x: number, y: number}[]>();

// Helper to get corners of a rotated rectangle (Cached)
function getCorners(el: LayoutElement): {x: number, y: number}[] {
    const cacheKey = `${el.id}_${el.x}_${el.y}_${el.width}_${el.height}_${el.rotation}`;
    if (cornerCache.has(cacheKey)) {
        return cornerCache.get(cacheKey)!;
    }

    const rad = ((el.rotation || 0) * Math.PI) / 180;
    const cx = el.x + el.width / 2;
    const cy = el.y + el.height / 2;

    const corners = [
        { x: el.x, y: el.y },
        { x: el.x + el.width, y: el.y },
        { x: el.x + el.width, y: el.y + el.height },
        { x: el.x, y: el.y + el.height },
    ].map((p) => {
        const dx = p.x - cx;
        const dy = p.y - cy;
        return {
        x: cx + dx * Math.cos(rad) - dy * Math.sin(rad),
        y: cy + dx * Math.sin(rad) + dy * Math.cos(rad),
        };
    });

    // Simple cache eviction if too large
    if (cornerCache.size > 2000) cornerCache.clear();
    cornerCache.set(cacheKey, corners);
    return corners;
}

// Optimized SAT Intersection
function isPolygonsIntersecting(a: {x: number, y: number}[], b: {x: number, y: number}[]) {
  if (a.length < 3 || b.length < 3) return false;

  const polygons = [a, b];
  for (let i = 0; i < polygons.length; i++) {
    const polygon = polygons[i];
    for (let j = 0; j < polygon.length; j++) {
      const k = (j + 1) % polygon.length;
      
      // Normal vector
      const nx = polygon[k].y - polygon[j].y;
      const ny = polygon[j].x - polygon[k].x;

      if (nx === 0 && ny === 0) continue;

      // Project Polygon A
      let minA = Infinity, maxA = -Infinity;
      for (let pIdx = 0; pIdx < a.length; pIdx++) {
        const projected = nx * a[pIdx].x + ny * a[pIdx].y;
        if (projected < minA) minA = projected;
        if (projected > maxA) maxA = projected;
      }

      // Project Polygon B
      let minB = Infinity, maxB = -Infinity;
      for (let pIdx = 0; pIdx < b.length; pIdx++) {
        const projected = nx * b[pIdx].x + ny * b[pIdx].y;
        if (projected < minB) minB = projected;
        if (projected > maxB) maxB = projected;
      }

      if (maxA < minB || maxB < minA) {
        return false;
      }
    }
  }
  return true;
}

const EPSILON = 0.5; // Tolerance for floating point overlap

function isOverlapping(road: LayoutElement, item: LayoutElement, pad: number = 0): boolean {
   // Fast AABB check first with EPSILON tolerance
   // If the overlap is less than EPSILON, treat it as "touching" not overlapping
   const rRect = { l: road.x - pad, r: road.x + road.width + pad, t: road.y - pad, b: road.y + road.height + pad };
   const iRect = { l: item.x, r: item.x + item.width, t: item.y, b: item.y + item.height };

   const overlapX = Math.min(rRect.r, iRect.r) - Math.max(rRect.l, iRect.l);
   const overlapY = Math.min(rRect.b, iRect.b) - Math.max(rRect.t, iRect.t);

   // If overlap depth is microscopic, ignore it
   if (overlapX < EPSILON || overlapY < EPSILON) return false;
   
   // SAT logic doesn't easily support 'padding' without inflating polygon
   // For now, if pad > 0, we trust AABB loosely, but for pad=0 strict check, we use SAT
   return isPolygonsIntersecting(getCorners(road), getCorners(item));
}

function isTouching(a: LayoutElement, b: LayoutElement): boolean {
    return isPolygonsIntersecting(getCorners(a), getCorners(b));
}

export function getIntersectionBox(r1: LayoutElement, r2: LayoutElement) {
    const x1 = Math.max(r1.x, r2.x);
    const y1 = Math.max(r1.y, r2.y);
    const x2 = Math.min(r1.x + r1.width, r2.x + r2.width);
    const y2 = Math.min(r1.y + r1.height, r2.y + r2.height);

    if (x1 < x2 && y1 < y2) {
        return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
    }
    return null;
}

// --- OPTIMIZED VALIDATION ---
export function validateLayout(layout: ParkingLayout): ConstraintViolation[] {
  const violations: ConstraintViolation[] = [];
  if (!layout || !layout.elements) return violations;

  // Initialize Spatial Grid
  const grid = new SpatialGrid(100);
  layout.elements.forEach(el => grid.add(el));

  // 1. Check Out of Bounds
  layout.elements.forEach(el => {
    // Quick AABB check first (assume 0 rotation mostly)
    if (el.x < 0 || el.x + el.width > layout.width || el.y < 0 || el.y + el.height > layout.height) {
        // Precise check
        const corners = getCorners(el);
        const isOutside = corners.some(p => p.x < 0 || p.x > layout.width || p.y < 0 || p.y > layout.height);
        if (isOutside) {
            violations.push({
                elementId: el.id,
                type: 'out_of_bounds',
                message: `Element is outside boundary.`
            });
        }
    }
  });

  // 2. Optimized Overlap Checks using Spatial Grid
  const solidTypes = new Set([
    ElementType.PARKING_SPACE, ElementType.PILLAR, ElementType.WALL, 
    ElementType.STAIRCASE, ElementType.ELEVATOR, ElementType.ROAD, ElementType.RAMP, ElementType.CHARGING_STATION,
    ElementType.ENTRANCE, ElementType.EXIT
  ]);
  
  const solids = layout.elements.filter(e => solidTypes.has(e.type as ElementType));

  solids.forEach(el1 => {
      const candidates = grid.getPotentialCollisions(el1);
      
      candidates.forEach(el2 => {
          // Prevent double checking
          if (el1.id >= el2.id) return; 
          if (!solidTypes.has(el2.type as ElementType)) return;

          // --- PERMITTED OVERLAPS ---
          
          // Parking allowed to overlap Ground and Charging Station
          if (el1.type === ElementType.PARKING_SPACE) {
              if (el2.type === ElementType.GROUND || el2.type === ElementType.CHARGING_STATION) return;
          }
          if (el2.type === ElementType.PARKING_SPACE) {
              if (el1.type === ElementType.GROUND || el1.type === ElementType.CHARGING_STATION) return;
          }
          
          // Standard Ignore cases
          if (el1.type === ElementType.WALL && el2.type === ElementType.WALL) return;
          if (el1.type === ElementType.ROAD && el2.type === ElementType.ROAD) return;
          if ((el1.type === ElementType.PILLAR && el2.type === ElementType.WALL) ||
              (el2.type === ElementType.PILLAR && el1.type === ElementType.WALL)) return;

          // SPECIAL RULE: EXEMPT MARGINAL OVERLAPS FOR CONNECTORS (RAMP <-> ROAD/ENTRANCE/EXIT)
          const isRamp = (t: string) => t === ElementType.RAMP;
          const isConnector = (t: string) => t === ElementType.ROAD || t === ElementType.ENTRANCE || t === ElementType.EXIT;
          const isConnectorPair = (t1: string, t2: string) => (isRamp(t1) && isConnector(t2)) || (isRamp(t2) && isConnector(t1));

          if (isConnectorPair(el1.type as string, el2.type as string)) {
               const box = getIntersectionBox(el1, el2);
               // ALLOW TIGHT CONTACT: If overlap is shallow (< 2px) in EITHER dimension, assume it's a tight connection and ignore it
               // Used to be 8px, but 2px is tighter and safer for "touching" vs "crashing"
               if (box && (box.width <= 2 || box.height <= 2)) return;
          }

          // Broad phase dist check
          const r1 = Math.max(el1.width, el1.height);
          const r2 = Math.max(el2.width, el2.height);
          const distSq = Math.pow(el1.x - el2.x, 2) + Math.pow(el1.y - el2.y, 2);
          
          if (distSq < Math.pow(r1 + r2, 2)) {
             // Strict check without padding, but using EPSILON internal to isOverlapping logic if adapted, 
             // OR rely on isPolygonsIntersecting.
             // Since we already filtered connector pairs above, real intersections here are likely invalid.
             if (isPolygonsIntersecting(getCorners(el1), getCorners(el2))) {
                 violations.push({
                   elementId: el1.id, targetId: el2.id, type: 'overlap',
                   message: `${el1.type} overlaps with ${el2.type}`
                 });
             }
          }
      });
  });

  // 3. Placement Constraints (Items that MUST be on a ROAD)
  const itemsOnRoad = new Set([
    ElementType.GUIDANCE_SIGN, ElementType.SIDEWALK, ElementType.SPEED_BUMP, 
    ElementType.LANE_LINE, ElementType.CONVEX_MIRROR
  ]);

  const itemsNotOnRoad = new Set([
      ElementType.STAIRCASE, ElementType.ELEVATOR, ElementType.SAFE_EXIT, 
      ElementType.FIRE_EXTINGUISHER, ElementType.ENTRANCE, ElementType.EXIT, ElementType.RAMP 
  ]);

  layout.elements.forEach(item => {
      // Must be ON Road
      if (itemsOnRoad.has(item.type as ElementType)) {
          const nearbyRoads = grid.getPotentialCollisions(item).filter(c => c.type === ElementType.ROAD);
          // Use small padding for "on top of" check
          const isOnRoad = nearbyRoads.some(road => isOverlapping(road, item, 2));
          if (!isOnRoad) {
             violations.push({
               elementId: item.id, type: 'placement_error',
               message: `${item.type} must be placed on a Driving Lane.`
             });
          }
      }

      // Must be OFF Road
      if (itemsNotOnRoad.has(item.type as ElementType)) {
          const nearbyRoads = grid.getPotentialCollisions(item).filter(c => c.type === ElementType.ROAD);
          // Use strict check for "not overlapping"
          const onRoad = nearbyRoads.some(road => isOverlapping(road, item, 0));
          if (onRoad && item.type !== ElementType.SIDEWALK) { 
              violations.push({
                  elementId: item.id, type: 'placement_error',
                  message: `${item.type} must NOT overlap a Driving Lane.`
              });
          }
      }
  });

  // 4. Connectivity Checks
  const gates = layout.elements.filter(e => e.type === ElementType.ENTRANCE || e.type === ElementType.EXIT);
  const ramps = layout.elements.filter(e => e.type === ElementType.RAMP);

  // Gates -> Ramps
  gates.forEach(gate => {
      const nearbyRamps = grid.getPotentialCollisions(gate).filter(c => c.type === ElementType.RAMP);
      if (!nearbyRamps.some(r => isTouching(r, gate))) {
           violations.push({
              elementId: gate.id, type: 'connectivity_error',
              message: `${gate.type} must be adjacent to a Ramp.`
          });
      }
  });
  
  // Ramps -> Roads
  ramps.forEach(ramp => {
      const nearbyRoads = grid.getPotentialCollisions(ramp).filter(c => c.type === ElementType.ROAD);
      if (!nearbyRoads.some(road => isTouching(road, ramp))) {
           violations.push({
               elementId: ramp.id, type: 'connectivity_error',
               message: `Ramp must connect to a Driving Lane.`
           });
      }
  });

  // Connectivity (Graph)
  const navigableTypes = new Set([ElementType.ROAD, ElementType.RAMP, ElementType.ENTRANCE, ElementType.EXIT]);
  const navigables = layout.elements.filter(e => navigableTypes.has(e.type as ElementType));

  if (navigables.length > 0) {
    const adj = new Map<string, string[]>();
    navigables.forEach(el => adj.set(el.id, []));

    navigables.forEach(el1 => {
        const potentialNeighbors = grid.getPotentialCollisions(el1).filter(e => navigableTypes.has(e.type as ElementType));
        potentialNeighbors.forEach(el2 => {
            if (el1.id < el2.id) { 
                if (isPolygonsIntersecting(getCorners(el1), getCorners(el2))) {
                    adj.get(el1.id)?.push(el2.id);
                    adj.get(el2.id)?.push(el1.id);
                }
            }
        });
    });

    const entrances = layout.elements.filter(e => e.type === ElementType.ENTRANCE);
    const exits = layout.elements.filter(e => e.type === ElementType.EXIT);

    entrances.forEach(ent => {
        const visited = new Set<string>();
        const queue = [ent.id];
        visited.add(ent.id);
        let foundExit = false;

        while(queue.length > 0) {
            const curr = queue.shift()!;
            const currEl = layout.elements.find(e => e.id === curr);
            if (currEl?.type === ElementType.EXIT) {
                foundExit = true;
                break;
            }
            
            const neighbors = adj.get(curr) || [];
            for (const n of neighbors) {
                if (!visited.has(n)) {
                    visited.add(n);
                    queue.push(n);
                }
            }
        }

        if (!foundExit && exits.length > 0) {
            violations.push({
                elementId: ent.id, type: 'connectivity_error',
                message: 'No drivable path exists from this Entrance to any Exit.'
            });
        }
    });
  }

  // Global Checks
  const entrances = layout.elements.filter(e => e.type === ElementType.ENTRANCE);
  const exits = layout.elements.filter(e => e.type === ElementType.EXIT);
  if (entrances.length < 1) violations.push({ elementId: 'global', type: 'connectivity_error', message: 'Must have at least one Entrance.' });
  if (exits.length < 1) violations.push({ elementId: 'global', type: 'connectivity_error', message: 'Must have at least one Exit.' });

  return violations;
}