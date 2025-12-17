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

// --- GEOMETRY UTILS ---
const EPSILON = 1.5; // 容差值：小于 1.5px 的重叠视为“刚好接触”而非“非法重叠”

const cornerCache = new Map<string, {x: number, y: number}[]>();

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

    if (cornerCache.size > 2000) cornerCache.clear();
    cornerCache.set(cacheKey, corners);
    return corners;
}

function isPolygonsIntersecting(a: {x: number, y: number}[], b: {x: number, y: number}[]) {
  if (a.length < 3 || b.length < 3) return false;

  const polygons = [a, b];
  for (let i = 0; i < polygons.length; i++) {
    const polygon = polygons[i];
    for (let j = 0; j < polygon.length; j++) {
      const k = (j + 1) % polygon.length;
      
      const nx = polygon[k].y - polygon[j].y;
      const ny = polygon[j].x - polygon[k].x;

      if (nx === 0 && ny === 0) continue;

      let minA = Infinity, maxA = -Infinity;
      for (let pIdx = 0; pIdx < a.length; pIdx++) {
        const projected = nx * a[pIdx].x + ny * a[pIdx].y;
        if (projected < minA) minA = projected;
        if (projected > maxA) maxA = projected;
      }

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

export function getIntersectionBox(r1: LayoutElement, r2: LayoutElement) {
    const x1 = Math.max(r1.x, r2.x);
    const y1 = Math.max(r1.y, r2.y);
    const x2 = Math.min(r1.x + r1.width, r2.x + r2.width);
    const y2 = Math.min(r1.y + r1.height, r2.y + r2.height);

    const width = x2 - x1;
    const height = y2 - y1;

    // 关键修复：如果重叠宽或高小于 EPSILON，视为“边缘接触”，不返回相交框
    if (width > EPSILON && height > EPSILON) {
        return { x: x1, y: y1, width, height };
    }
    return null;
}

function isOverlapping(road: LayoutElement, item: LayoutElement, pad: number = 0): boolean {
   const rRect = { l: road.x - pad, r: road.x + road.width + pad, t: road.y - pad, b: road.y + road.height + pad };
   const iRect = { l: item.x, r: item.x + item.width, t: item.y, b: item.y + item.height };

   const overlapX = Math.min(rRect.r, iRect.r) - Math.max(rRect.l, iRect.l);
   const overlapY = Math.min(rRect.b, iRect.b) - Math.max(rRect.t, iRect.t);

   if (overlapX < EPSILON || overlapY < EPSILON) return false;
   
   return isPolygonsIntersecting(getCorners(road), getCorners(item));
}

// 连接性检查：允许 2px 的宽限期，只要“接近”即视为连接
function isTouching(a: LayoutElement, b: LayoutElement): boolean {
    const aCorners = getCorners(a);
    const bCorners = getCorners(b);
    
    // 基础 SAT 检查
    if (isPolygonsIntersecting(aCorners, bCorners)) return true;

    // 边界框接近度检查 (用于弥补 SAT 在刚好接触时的精度损失)
    const margin = 2.0;
    const aRect = { l: a.x - margin, r: a.x + a.width + margin, t: a.y - margin, b: a.y + a.height + margin };
    const bRect = { l: b.x - margin, r: b.x + b.width + margin, t: b.y - margin, b: b.y + b.height + margin };

    const overlapX = Math.min(aRect.r, bRect.r) - Math.max(aRect.l, bRect.l);
    const overlapY = Math.min(aRect.b, bRect.b) - Math.max(aRect.t, bRect.t);

    return overlapX >= 0 && overlapY >= 0;
}

// --- VALIDATION ---
export function validateLayout(layout: ParkingLayout): ConstraintViolation[] {
  const violations: ConstraintViolation[] = [];
  if (!layout || !layout.elements) return violations;

  const grid = new SpatialGrid(100);
  layout.elements.forEach(el => grid.add(el));

  // 1. Check Out of Bounds
  layout.elements.forEach(el => {
    if (el.x < -EPSILON || el.x + el.width > layout.width + EPSILON || el.y < -EPSILON || el.y + el.height > layout.height + EPSILON) {
        violations.push({
            elementId: el.id,
            type: 'out_of_bounds',
            message: `Element is outside boundary.`
        });
    }
  });

  // 2. Overlap Checks
  const solidTypes = new Set([
    ElementType.PARKING_SPACE, ElementType.PILLAR, ElementType.WALL, 
    ElementType.STAIRCASE, ElementType.ELEVATOR, ElementType.ROAD, ElementType.RAMP, ElementType.CHARGING_STATION,
    ElementType.ENTRANCE, ElementType.EXIT
  ]);
  
  const solids = layout.elements.filter(e => solidTypes.has(e.type as ElementType));

  solids.forEach(el1 => {
      const candidates = grid.getPotentialCollisions(el1);
      
      candidates.forEach(el2 => {
          if (el1.id >= el2.id) return; 

          // 基础排除逻辑
          if (el1.type === ElementType.PARKING_SPACE && (el2.type === ElementType.GROUND || el2.type === ElementType.CHARGING_STATION)) return;
          if (el2.type === ElementType.PARKING_SPACE && (el1.type === ElementType.GROUND || el1.type === ElementType.CHARGING_STATION)) return;
          if (el1.type === ElementType.WALL && el2.type === ElementType.WALL) return;
          if (el1.type === ElementType.ROAD && el2.type === ElementType.ROAD) return;
          if ((el1.type === ElementType.WALL && (el2.type === ElementType.ENTRANCE || el2.type === ElementType.EXIT)) ||
              (el2.type === ElementType.WALL && (el1.type === ElementType.ENTRANCE || el1.type === ElementType.EXIT))) return;

          // 核心修复：执行 EPSILON 过滤
          const box = getIntersectionBox(el1, el2);
          if (box) {
              violations.push({
                elementId: el1.id, targetId: el2.id, type: 'overlap',
                message: `${el1.type} overlaps with ${el2.type} (Overlap: ${Math.round(box.width)}x${Math.round(box.height)})`
              });
          }
      });
  });

  // 3. Placement Constraints
  const itemsOnRoad = new Set([ElementType.GUIDANCE_SIGN, ElementType.SIDEWALK, ElementType.SPEED_BUMP, ElementType.LANE_LINE]);
  layout.elements.forEach(item => {
      if (itemsOnRoad.has(item.type as ElementType)) {
          const nearbyRoads = grid.getPotentialCollisions(item).filter(c => c.type === ElementType.ROAD);
          const isOnRoad = nearbyRoads.some(road => isOverlapping(road, item, 5)); // 允许 5px 边缘余量
          if (!isOnRoad) {
             violations.push({
               elementId: item.id, type: 'placement_error',
               message: `${item.type} must be on Driving Lane.`
             });
          }
      }
  });

  // 4. Connectivity
  const gates = layout.elements.filter(e => e.type === ElementType.ENTRANCE || e.type === ElementType.EXIT);
  const ramps = layout.elements.filter(e => e.type === ElementType.RAMP);

  gates.forEach(gate => {
      const nearbyRamps = grid.getPotentialCollisions(gate).filter(c => c.type === ElementType.RAMP);
      if (!nearbyRamps.some(r => isTouching(r, gate))) {
           violations.push({
              elementId: gate.id, type: 'connectivity_error',
              message: `${gate.type} needs Ramp.`
          });
      }
  });
  
  ramps.forEach(ramp => {
      const nearbyRoads = grid.getPotentialCollisions(ramp).filter(c => c.type === ElementType.ROAD);
      if (!nearbyRoads.some(road => isTouching(road, ramp))) {
           violations.push({
               elementId: ramp.id, type: 'connectivity_error',
               message: `Ramp disconnected.`
           });
      }
  });

  return violations;
}