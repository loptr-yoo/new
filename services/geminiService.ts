import { GoogleGenAI } from "@google/genai";
import { jsonrepair } from "jsonrepair";
import { z } from "zod";
import { ParkingLayout, ElementType, ConstraintViolation, LayoutElement } from "../types";
import { validateLayout, getIntersectionBox } from "../utils/geometry";
import { PROMPTS } from "../utils/prompts";

const fallbackLayout: ParkingLayout = { width: 800, height: 600, elements: [] };

const MODEL_PRIMARY = "gemini-3-pro-preview";
const MODEL_FALLBACK = "gemini-2.5-pro"; 

let cachedTier: 'HIGH' | 'LOW' | null = null;

const getApiKey = () => process.env.API_KEY;
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// --- ALGORITHM: GEOMETRIC PARKING SPOT FILLER ---
const fillParkingAutomatically = (layout: ParkingLayout): ParkingLayout => {
  const existingElements = [...layout.elements];
  const grounds = existingElements.filter(e => e.type === ElementType.GROUND);
  const roads = existingElements.filter(e => e.type === ElementType.ROAD);
  
  // Obstacles to avoid
  const obstacles = existingElements.filter(e => 
    [ElementType.WALL, ElementType.STAIRCASE, ElementType.ELEVATOR, ElementType.PILLAR,
     ElementType.ENTRANCE, ElementType.EXIT, ElementType.RAMP, ElementType.SAFE_EXIT,
     ElementType.SIDEWALK, ElementType.PARKING_SPACE].includes(e.type as ElementType)
  );
  
  const genSpots: LayoutElement[] = [];
  const SPOT_S = 24; // Width
  const SPOT_L = 48; // Length
  const GAP = 2;     // Space between spots
  const BUFFER = 4;  
  const TOLERANCE = 12; 

  const isSafe = (rect: {x: number, y: number, w: number, h: number}) => {
      const m = 1; 
      const hitObstacle = obstacles.some(o => 
        rect.x + m < o.x + o.width && rect.x + rect.w - m > o.x &&
        rect.y + m < o.y + o.height && rect.y + rect.h - m > o.y
      );
      const hitSelf = genSpots.some(o => 
        rect.x + m < o.x + o.width && rect.x + rect.w - m > o.x &&
        rect.y + m < o.y + o.height && rect.y + rect.h - m > o.y
      );
      return !hitObstacle && !hitSelf;
  };

  let t = 0; 

  roads.forEach(r => {
      const rr = { l: r.x, r: r.x + r.width, t: r.y, b: r.y + r.height };
      
      grounds.forEach(g => {
          const gr = { l: g.x, r: g.x + g.width, t: g.y, b: g.y + g.height };
          
          // Case A: Ground is below Road (Horizontal Road)
          if (Math.abs(rr.b - gr.t) < TOLERANCE && Math.min(rr.r, gr.r) > Math.max(rr.l, gr.l)) {
               const sx = Math.max(rr.l, gr.l) + BUFFER;
               const ex = Math.min(rr.r, gr.r) - BUFFER;
               const cnt = Math.floor((ex - sx) / (SPOT_S + GAP)); 
               
               for(let i=0; i<cnt; i++) {
                   const s = { x: sx + i*(SPOT_S+GAP), y: gr.t + 2, w: SPOT_S, h: SPOT_L };
                   if (isSafe(s)) {
                       genSpots.push({ 
                           id: `p_auto_${++t}`, 
                           type: ElementType.PARKING_SPACE, 
                           x: s.x, y: s.y, width: s.w, height: s.h,
                           rotation: 0 
                       });
                   }
               }
          }
          // Case B: Ground is above Road (Horizontal Road)
          else if (Math.abs(rr.t - gr.b) < TOLERANCE && Math.min(rr.r, gr.r) > Math.max(rr.l, gr.l)) {
              const sx = Math.max(rr.l, gr.l) + BUFFER;
              const ex = Math.min(rr.r, gr.r) - BUFFER;
              const cnt = Math.floor((ex - sx) / (SPOT_S + GAP));
              
              for(let i=0; i<cnt; i++) {
                   const s = { x: sx + i*(SPOT_S+GAP), y: gr.b - SPOT_L - 2, w: SPOT_S, h: SPOT_L };
                   if (isSafe(s)) {
                       genSpots.push({ 
                           id: `p_auto_${++t}`, 
                           type: ElementType.PARKING_SPACE, 
                           x: s.x, y: s.y, width: s.w, height: s.h,
                           rotation: 0 
                       });
                   }
              }
          }
          // Case C: Ground is right of Road (Vertical Road)
          else if (Math.abs(rr.r - gr.l) < TOLERANCE && Math.min(rr.b, gr.b) > Math.max(rr.t, gr.t)) {
              const sy = Math.max(rr.t, gr.t) + BUFFER;
              const ey = Math.min(rr.b, gr.b) - BUFFER;
              const cnt = Math.floor((ey - sy) / (SPOT_S + GAP));

              for(let i=0; i<cnt; i++) {
                  const s = { x: gr.l + 2, y: sy + i*(SPOT_S+GAP), w: SPOT_L, h: SPOT_S };
                  if (isSafe(s)) {
                      genSpots.push({
                          id: `p_auto_v_${++t}`,
                          type: ElementType.PARKING_SPACE,
                          x: s.x, y: s.y, width: s.w, height: s.h,
                          rotation: 0
                      });
                  }
              }
          }
          // Case D: Ground is left of Road (Vertical Road)
          else if (Math.abs(rr.l - gr.r) < TOLERANCE && Math.min(rr.b, gr.b) > Math.max(rr.t, gr.t)) {
              const sy = Math.max(rr.t, gr.t) + BUFFER;
              const ey = Math.min(rr.b, gr.b) - BUFFER;
              const cnt = Math.floor((ey - sy) / (SPOT_S + GAP));

              for(let i=0; i<cnt; i++) {
                  const s = { x: gr.r - SPOT_L - 2, y: sy + i*(SPOT_S+GAP), w: SPOT_L, h: SPOT_S };
                  if (isSafe(s)) {
                      genSpots.push({
                          id: `p_auto_v_${++t}`,
                          type: ElementType.PARKING_SPACE,
                          x: s.x, y: s.y, width: s.w, height: s.h,
                          rotation: 0
                      });
                  }
              }
          }
      });
  });

  return { ...layout, elements: [...existingElements, ...genSpots] };
};

// --- SAFETY: CLEANUP INVALID ELEMENTS ---
const cleanupPillars = (layout: ParkingLayout): ParkingLayout => {
    const roads = layout.elements.filter(e => e.type === ElementType.ROAD);
    const spots = layout.elements.filter(e => e.type === ElementType.PARKING_SPACE);
    
    return {
        ...layout,
        elements: layout.elements.filter(el => {
            if (el.type !== ElementType.PILLAR) return true;
            
            const isOnRoad = roads.some(r => 
                el.x < r.x + r.width && el.x + el.width > r.x &&
                el.y < r.y + r.height && el.y + el.height > r.y
            );
            // Pillars shouldn't be inside spots either, usually they are at the corners
            const isInsideSpot = spots.some(s => 
                el.x > s.x + 2 && el.x + el.width < s.x + s.width - 2 &&
                el.y > s.y + 2 && el.y + el.height < s.y + s.height - 2
            );

            return !isOnRoad && !isInsideSpot;
        })
    };
};

const mergeLayoutElements = (
  original: LayoutElement[], 
  updates: LayoutElement[]
): LayoutElement[] => {
  const elementMap = new Map(original.map(el => [el.id, el]));
  
  updates.forEach(update => {
    if (update.id && elementMap.has(update.id)) {
      const existing = elementMap.get(update.id)!;
      elementMap.set(update.id, { ...existing, ...update });
    } else {
      const newId = update.id || `el_${Math.random().toString(36).substr(2, 9)}`;
      elementMap.set(newId, { ...update, id: newId });
    }
  });

  return Array.from(elementMap.values());
};

const postProcessLayout = (layout: ParkingLayout): ParkingLayout => {
    return {
        ...layout,
        elements: layout.elements.map(el => {
            const rx = Math.round(el.x);
            const ry = Math.round(el.y);
            const rw = Math.round(el.width);
            const rh = Math.round(el.height);
            
            const isStructural = [ElementType.ROAD, ElementType.GROUND, ElementType.WALL].includes(el.type as ElementType);
            const pad = isStructural ? 1 : 0;

            return {
                ...el,
                x: rx,
                y: ry,
                width: rw + pad,
                height: rh + pad
            };
        })
    };
};

const resolvePriorityConflicts = (elements: LayoutElement[]): LayoutElement[] => {
    const sidewalks = elements.filter(e => e.type === ElementType.SIDEWALK);
    return elements.filter(el => {
        if (el.type === ElementType.SPEED_BUMP) {
            const hasConflict = sidewalks.some(s => {
                const intersection = getIntersectionBox(el, s);
                return intersection !== null && (intersection.width > 2 || intersection.height > 2);
            });
            return !hasConflict;
        }
        return true;
    });
};

const orientGuidanceSigns = (layout: ParkingLayout): ParkingLayout => {
    const exits = layout.elements.filter(e => e.type === ElementType.EXIT);
    const roads = layout.elements.filter(e => e.type === ElementType.ROAD);
    if (exits.length === 0) return layout;

    const updated = layout.elements.map(el => {
        if (el.type === ElementType.GUIDANCE_SIGN) {
            const parentRoad = roads.find(r => 
                el.x >= r.x - 5 && el.x + el.width <= r.x + r.width + 5 &&
                el.y >= r.y - 5 && el.y + el.height <= r.y + r.height + 5
            );

            let nearestExit = exits[0], minDist = Infinity;
            const scx = el.x + el.width / 2;
            const scy = el.y + el.height / 2;

            exits.forEach(ex => {
                const ecx = ex.x + ex.width / 2;
                const ecy = ex.y + ex.height / 2;
                const d = Math.abs(ecx - scx) + Math.abs(ecy - scy);
                if (d < minDist) { minDist = d; nearestExit = ex; }
            });

            const ecx = nearestExit.x + nearestExit.width / 2;
            const ecy = nearestExit.y + nearestExit.height / 2;

            if (parentRoad) {
                const isHorizontal = parentRoad.width > parentRoad.height;
                if (isHorizontal) {
                    return { ...el, rotation: ecx > scx ? 0 : 180 };
                } else {
                    return { ...el, rotation: ecy > scy ? 90 : 270 };
                }
            }
            
            const dx = ecx - scx;
            const dy = ecy - scy;
            if (Math.abs(dx) > Math.abs(dy)) return { ...el, rotation: dx > 0 ? 0 : 180 };
            return { ...el, rotation: dy > 0 ? 90 : 270 };
        }
        return el;
    });
    return { ...layout, elements: updated };
};

async function determineModelTier(ai: GoogleGenAI, onLog?: (m: string) => void): Promise<'HIGH' | 'LOW'> {
    if (cachedTier) return cachedTier;
    onLog?.("Checking model availability...");
    try {
        await ai.models.generateContent({ model: MODEL_PRIMARY, contents: "test", config: { maxOutputTokens: 1 } });
        cachedTier = 'HIGH';
        onLog?.("High Tier (3-Pro) detected.");
    } catch (e) {
        cachedTier = 'LOW';
        onLog?.("Standard Tier detected.");
    }
    return cachedTier;
}

const normalizeType = (t: string | undefined): string => {
  if (!t) return ElementType.WALL;
  const lower = t.toLowerCase().trim().replace(/\s+/g, '_');
  const map: Record<string, string> = {
    'ramp': ElementType.RAMP, 'slope': ElementType.RAMP,
    'speed_bump': ElementType.SPEED_BUMP, 'deceleration_zone': ElementType.SPEED_BUMP,
    'road': ElementType.ROAD, 'driving_lane': ElementType.ROAD,
    'pedestrian_path': ElementType.SIDEWALK, 'sidewalk': ElementType.SIDEWALK,
    'ground_line': ElementType.LANE_LINE, 'lane_line': ElementType.LANE_LINE,
    'parking_spot': ElementType.PARKING_SPACE, 'parking': ElementType.PARKING_SPACE,
    'charging': ElementType.CHARGING_STATION,
    'ground': ElementType.GROUND, 
    'island': ElementType.GROUND,
    'landscape': ElementType.GROUND,
    'pillar': ElementType.PILLAR,
    'staircase': ElementType.STAIRCASE,
    'elevator': ElementType.ELEVATOR,
    'safe_exit': ElementType.SAFE_EXIT,
    'fire_extinguisher': ElementType.FIRE_EXTINGUISHER,
    'guidance_sign': ElementType.GUIDANCE_SIGN,
    'parking_strip': ElementType.GROUND,
    'central_island': ElementType.GROUND,
    'green_zone': ElementType.GROUND,
    'landscape_area': ElementType.GROUND,
    'void': ElementType.GROUND, 
    'buffer': ElementType.GROUND,
    'median': ElementType.GROUND
  };
  return map[lower] || lower;
};

const cleanAndParseJSON = (text: string): any => {
  try {
    let cleanText = text.replace(/```json\s*|```/g, "").trim();
    const firstOpen = cleanText.indexOf('{');
    const lastClose = cleanText.lastIndexOf('}');
    if (firstOpen !== -1 && lastClose !== -1) cleanText = cleanText.substring(firstOpen, lastClose + 1);
    const repaired = jsonrepair(cleanText);
    const parsed = JSON.parse(repaired);
    return parsed;
  } catch (e) {
    throw new Error(`Failed to parse AI response: ${(e as Error).message}`);
  }
};

const mapToInternalLayout = (rawData: any): ParkingLayout => ({
    width: Number(rawData.width || 800),
    height: Number(rawData.height || 600),
    elements: (rawData.elements || []).map((e: any) => ({
        id: String(e.id || `el_${Math.random().toString(36).substr(2, 9)}`),
        type: normalizeType(e.t || e.type),
        x: Number(e.x || 0),
        y: Number(e.y || 0),
        width: Number(e.w ?? e.width ?? 10),
        height: Number(e.h ?? e.height ?? 10),
        rotation: Number(e.r || 0),
        label: e.l
    }))
});

const calculateScore = (violations: ConstraintViolation[]): number => {
    return violations.reduce((acc, v) => {
        if (v.type === 'overlap') return acc + 5;
        if (v.type === 'connectivity_error') return acc + 10;
        if (v.type === 'out_of_bounds') return acc + 8;
        return acc + 2;
    }, 0);
};

const generateWithRetry = async (ai: GoogleGenAI, params: any, retries = 3) => {
    for (let i = 0; i < retries; i++) {
        try {
            return await ai.models.generateContent(params);
        } catch (e: any) {
            if (i === retries - 1) throw e;
            await sleep(2000 * Math.pow(2, i));
        }
    }
    throw new Error("Failed after retries");
};

const runIterativeFix = async (
    layout: ParkingLayout, 
    ai: GoogleGenAI, 
    model: string, 
    onLog?: (m: string) => void,
    maxPasses = 6
): Promise<ParkingLayout> => {
    let currentLayout = layout;
    let lastScore = Infinity;

    for (let pass = 1; pass <= maxPasses; pass++) {
        const violations = validateLayout(currentLayout);
        const score = calculateScore(violations);
        
        if (score === 0) {
            onLog?.(`🔧 Fix Pass ${pass}: Perfect (Score: 0).`);
            break;
        }
        if (score >= lastScore && pass > 1) {
            onLog?.(`🌡️ Stagnation detected at Pass ${pass} (Score: ${score}). Stopping fix loop.`);
            break;
        }
        
        onLog?.(`🔧 Auto-fixing pass ${pass}/${maxPasses} (Score: ${score})...`);
        lastScore = score;

        const simplified = {
            width: currentLayout.width,
            height: currentLayout.height,
            elements: currentLayout.elements.map(e => ({ 
                id: e.id, 
                t: e.type, 
                x: Math.round(e.x), 
                y: Math.round(e.y), 
                w: Math.round(e.width), 
                h: Math.round(e.height), 
                r: e.rotation 
            }))
        };

        try {
            const response = await generateWithRetry(ai, { 
                model, 
                contents: PROMPTS.fix(simplified as any, violations), 
                config: { responseMimeType: "application/json", temperature: 0.1 } 
            }, 1);
            
            const rawData = cleanAndParseJSON(response.text || "{}");
            if (rawData.fix_strategy && onLog) {
                rawData.fix_strategy.forEach((s: string) => onLog(`🤖 AI Action: ${s}`));
            }
            
            const fixedLayout = mapToInternalLayout(rawData);
            
            const countDiff = fixedLayout.elements.length - currentLayout.elements.length;
            const originalGroundCount = currentLayout.elements.filter(e => e.type === ElementType.GROUND).length;
            const newGroundCount = fixedLayout.elements.filter(e => e.type === ElementType.GROUND).length;
            const groundReduced = newGroundCount < originalGroundCount;

            const shouldUseMerge = model === MODEL_FALLBACK || countDiff < 0 || groundReduced;

            if (shouldUseMerge) {
                currentLayout = {
                    ...currentLayout,
                    elements: mergeLayoutElements(currentLayout.elements, fixedLayout.elements)
                };
            } else {
                currentLayout = fixedLayout;
            }
        } catch (e: any) {
            onLog?.(`⚠️ Fix pass failed: ${e.message}. Skipping pass.`);
        }
    }
    return currentLayout;
};

export const generateParkingLayout = async (description: string, onLog?: (msg: string) => void): Promise<ParkingLayout> => {
  const apiKey = getApiKey();
  if (!apiKey) return fallbackLayout;
  const ai = new GoogleGenAI({ apiKey });
  
  let tier = await determineModelTier(ai, onLog);
  let currentModel = tier === 'HIGH' ? MODEL_PRIMARY : MODEL_FALLBACK;
  onLog?.(`Using model: ${currentModel}`);

  try {
    const response = await generateWithRetry(ai, { model: currentModel, contents: PROMPTS.generation(description), config: { responseMimeType: "application/json" } }, 2);
    const rawData = cleanAndParseJSON(response.text);
    if (rawData.reasoning_plan && onLog) onLog(`🧠 Plan: ${rawData.reasoning_plan}`);
    
    let layout = mapToInternalLayout(rawData);
    onLog?.(`Generated ${layout.elements.length} structural elements.`);
    
    layout = await runIterativeFix(layout, ai, currentModel, onLog);
    return postProcessLayout(layout);
  } catch (error: any) {
    throw error;
  }
};

export const augmentLayoutWithRoads = async (currentLayout: ParkingLayout, onLog?: (msg: string) => void): Promise<ParkingLayout> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("API Key required");
  const ai = new GoogleGenAI({ apiKey });
  
  let tier = await determineModelTier(ai, onLog);
  let currentModel = tier === 'HIGH' ? MODEL_PRIMARY : MODEL_FALLBACK;

  try {
    const simplified = currentLayout.elements.map(e => ({ id: e.id, t: e.type, x: e.x, y: e.y, w: e.width, h: e.height }));
    const response = await generateWithRetry(ai, { 
        model: currentModel, 
        contents: PROMPTS.refinement({ elements: simplified }, currentLayout.width, currentLayout.height), 
        config: { responseMimeType: "application/json" } 
    }, 2);
    
    const rawData = cleanAndParseJSON(response.text);
    if (rawData.reasoning_plan && onLog) onLog(`✨ AI Plan: ${rawData.reasoning_plan}`);

    const aiGeneratedLayout = mapToInternalLayout(rawData);
    const newElements = aiGeneratedLayout.elements;
    onLog?.(`AI suggested ${newElements.length} detailed elements.`);

    let layout: ParkingLayout = {
        width: currentLayout.width,
        height: currentLayout.height,
        elements: [...currentLayout.elements, ...newElements]
    };

    onLog?.("📐 Running Algorithmic Spot Filler...");
    layout = fillParkingAutomatically(layout);

    onLog?.("🧹 Cleaning up illegal pillars...");
    layout = cleanupPillars(layout);

    onLog?.("⚖️ Resolving pedestrian/road conflicts...");
    layout.elements = resolvePriorityConflicts(layout.elements);

    onLog?.("🧭 Snapping signs to orthogonal road directions...");
    layout = orientGuidanceSigns(layout);
    
    layout = await runIterativeFix(layout, ai, currentModel, onLog);
    return postProcessLayout(layout);
  } catch (error: any) {
    throw error;
  }
};