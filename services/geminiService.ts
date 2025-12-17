import { GoogleGenAI } from "@google/genai";
import { jsonrepair } from "jsonrepair";
import { z } from "zod";
import { ParkingLayout, ElementType, ConstraintViolation, LayoutElement } from "../types";
import { validateLayout, getIntersectionBox } from "../utils/geometry";
import { PROMPTS } from "../utils/prompts";

const fallbackLayout: ParkingLayout = { width: 800, height: 600, elements: [] };

const MODEL_PRIMARY = "gemini-3-pro-preview";
const MODEL_FALLBACK = "gemini-2.5-pro"; 

const LayoutElementSchema = z.object({
  id: z.string().optional(),
  t: z.string().optional(),
  type: z.string().optional(),
  x: z.union([z.number(), z.string()]).transform(Number),
  y: z.union([z.number(), z.string()]).transform(Number),
  w: z.union([z.number(), z.string()]).transform(Number).optional(),
  width: z.union([z.number(), z.string()]).transform(Number).optional(),
  h: z.union([z.number(), z.string()]).transform(Number).optional(),
  height: z.union([z.number(), z.string()]).transform(Number).optional(),
  r: z.union([z.number(), z.string()]).transform(Number).optional(),
  l: z.string().optional()
});

let cachedTier: 'HIGH' | 'LOW' | null = null;

const getApiKey = () => process.env.API_KEY;
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

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
            const originalGrounds = currentLayout.elements.filter(e => e.type === ElementType.GROUND).length;
            const newGrounds = fixedLayout.elements.filter(e => e.type === ElementType.GROUND).length;
            const groundReduced = newGrounds < originalGrounds;

            const shouldUseMerge = 
                model === MODEL_FALLBACK || 
                countDiff < 0 || 
                groundReduced;

            if (shouldUseMerge) {
                const missingCount = currentLayout.elements.length - fixedLayout.elements.length;
                if (missingCount > 0 || groundReduced) {
                     onLog?.(`🛡️ Strict Merge Triggered: Ground dropped from ${originalGrounds} to ${newGrounds}. Preserving original elements.`);
                }
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
    if (rawData.reasoning_plan && onLog) onLog(`✨ Refinement Plan: ${rawData.reasoning_plan}`);

    const aiGeneratedLayout = mapToInternalLayout(rawData);
    const newElements = aiGeneratedLayout.elements;
    onLog?.(`Adding ${newElements.length} detailed elements.`);

    let layout: ParkingLayout = {
        width: currentLayout.width,
        height: currentLayout.height,
        elements: [...currentLayout.elements, ...newElements]
    };

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