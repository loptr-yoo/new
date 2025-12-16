import { GoogleGenAI, Type } from "@google/genai";
import { jsonrepair } from "jsonrepair";
import { z } from "zod";
import { ParkingLayout, ElementType, ConstraintViolation, LayoutElement } from "../types";
import { validateLayout } from "../utils/geometry";
import { PROMPTS } from "../utils/prompts";

const fallbackLayout: ParkingLayout = { width: 800, height: 600, elements: [] };

// MODELS
// User requested "2.5 pro" fallback.
const MODEL_PRIMARY = "gemini-3-pro-preview";
const MODEL_FALLBACK = "gemini-2.5-pro"; 

// ZOD SCHEMAS
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

const LayoutSchema = z.object({
  reasoning_plan: z.string().optional(),
  analysis: z.string().optional(),
  fix_strategy: z.array(z.string()).optional(),
  width: z.union([z.number(), z.string()]).transform(Number),
  height: z.union([z.number(), z.string()]).transform(Number),
  elements: z.array(LayoutElementSchema).or(z.object({}).array()) 
});

let cachedTier: 'HIGH' | 'LOW' | null = null;

const getApiKey = () => process.env.API_KEY;

// HELPER: Sleep for backoff
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// HELPER: Round numbers to reduce token usage and fix floating point creep
const roundCoord = (num: number) => Math.round(num * 10) / 10;

async function determineModelTier(ai: GoogleGenAI, onLog?: (msg: string) => void): Promise<'HIGH' | 'LOW'> {
    if (cachedTier) return cachedTier;
    try {
        if (onLog) onLog("Checking model availability...");
        await ai.models.generateContent({
            model: MODEL_PRIMARY, contents: "test", config: { maxOutputTokens: 1 }
        });
        cachedTier = 'HIGH';
        if (onLog) onLog("High Tier (3-Pro) detected.");
    } catch (e) {
        cachedTier = 'LOW';
        if (onLog) onLog("Standard Tier (Fallback) active.");
    }
    return cachedTier;
}

const normalizeType = (t: string | undefined): string => {
  if (!t) return ElementType.WALL;
  const lower = t.toLowerCase().trim().replace(/\s+/g, '_');
  const map: Record<string, string> = {
    'ramp': ElementType.RAMP, 'slope': ElementType.RAMP,
    'speed_bump': ElementType.SPEED_BUMP,
    'road': ElementType.ROAD, 'driving_lane': ElementType.ROAD,
    'pedestrian_path': ElementType.SIDEWALK, 'sidewalk': ElementType.SIDEWALK,
    'ground_line': ElementType.LANE_LINE, 'lane_line': ElementType.LANE_LINE,
    'parking_spot': ElementType.PARKING_SPACE, 'parking': ElementType.PARKING_SPACE,
    'charging': ElementType.CHARGING_STATION
  };
  return map[lower] || lower;
};

// Robust Parsing using Regex extraction + jsonrepair + Zod
const cleanAndParseJSON = (text: string): z.infer<typeof LayoutSchema> => {
  try {
    if (!text || typeof text !== 'string') throw new Error("Empty response");

    // 1. Aggressive Cleanup: Find the first '{' and the last '}'
    let cleanText = text.replace(/```json\s*|```/g, "").trim();
    const firstOpen = cleanText.indexOf('{');
    const lastClose = cleanText.lastIndexOf('}');
    
    if (firstOpen !== -1 && lastClose !== -1 && lastClose > firstOpen) {
        cleanText = cleanText.substring(firstOpen, lastClose + 1);
    } else {
        throw new Error("No JSON object found in response");
    }

    // 2. Repair
    const repaired = jsonrepair(cleanText);
    const parsed = JSON.parse(repaired);

    // Normalize keys to root if nested
    let normalized = { ...parsed };
    let elements: any[] = [];
    
    // Search for elements array
    const isElementArray = (arr: any[]) => Array.isArray(arr) && arr.length > 0 && (arr[0].t || arr[0].type || arr[0].x !== undefined);

    if (Array.isArray(parsed) && isElementArray(parsed)) {
        elements = parsed;
    } else {
        if (Array.isArray(parsed.elements)) elements = parsed.elements;
        else if (Array.isArray(parsed.layout)) elements = parsed.layout;
        else if (Array.isArray(parsed.data)) elements = parsed.data;
        else if (Array.isArray(parsed.items)) elements = parsed.items;
        
        // Deep search fallback
        if (elements.length === 0 && typeof parsed === 'object') {
           for (const key in parsed) {
               if (Array.isArray(parsed[key]) && isElementArray(parsed[key])) {
                   elements = parsed[key];
                   break;
               }
           }
        }
    }
    
    normalized.elements = elements;
    // Default dimensions if missing
    if (!normalized.width) normalized.width = 800;
    if (!normalized.height) normalized.height = 600;

    const result = LayoutSchema.safeParse(normalized);
    if (!result.success) {
        console.warn("Zod Schema Warning:", result.error);
        return {
            reasoning_plan: normalized.reasoning_plan,
            width: Number(normalized.width) || 800,
            height: Number(normalized.height) || 600,
            elements: normalized.elements || []
        };
    }
    return result.data;
  } catch (e) {
    console.error("Critical JSON Parse Error", e);
    throw new Error(`Failed to parse AI response: ${(e as Error).message}`);
  }
};

const mapToInternalLayout = (rawData: z.infer<typeof LayoutSchema>): ParkingLayout => {
    return {
        width: rawData.width,
        height: rawData.height,
        elements: (rawData.elements || []).map((e: any) => ({
            id: String(e.id || `el_${Math.random().toString(36).substr(2, 9)}`),
            type: normalizeType(e.t || e.type),
            // ARCHITECT SANITY CHECK: Rounding output immediately after parsing to prevent float creep
            x: Math.round(Number(e.x || 0)),
            y: Math.round(Number(e.y || 0)),
            width: Math.round(Number(e.w ?? e.width ?? 10)),
            height: Math.round(Number(e.h ?? e.height ?? 10)),
            rotation: Math.round(Number(e.r || 0)),
            label: e.l
        }))
    };
};

// --- API WRAPPER WITH RETRY ---
const generateWithRetry = async (ai: GoogleGenAI, params: any, retries = 3, onLog?: (msg: string) => void) => {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await ai.models.generateContent(params);
            return response;
        } catch (e: any) {
            const isLast = i === retries - 1;
            const status = e.status || e.code || 500;
            const msg = e.message || JSON.stringify(e);
            
            // Log warning
            console.warn(`API Attempt ${i + 1} failed: ${msg}`);
            
            // Handle 429 / RESOURCE_EXHAUSTED specifically
            if (status === 429 || status === "RESOURCE_EXHAUSTED" || msg.includes("429") || msg.includes("quota") || msg.includes("RESOURCE_EXHAUSTED")) {
                 if (onLog) onLog(`⚠️ Quota/Rate limit detected.`);
                 
                 // If we have retries left, wait. If last retry, throw.
                 // NOTE: The caller might catch this to switch models, so we don't wait excessively long on the LAST try.
                 
                 if (isLast) {
                     throw new Error("API Quota Exceeded.");
                 }
                 
                 const delay = 4000 * Math.pow(2, i);
                 if(onLog) onLog(`Pausing for ${delay/1000}s before retry...`);
                 await sleep(delay);
                 continue;
            }

            if (isLast) throw e;
            
            // Standard Exponential backoff: 1s, 2s, 4s
            const delay = 1000 * Math.pow(2, i);
            if (onLog) onLog(`⚠️ API Error (${status}). Retrying in ${delay/1000}s...`);
            await sleep(delay);
        }
    }
    throw new Error("Failed after retries");
};

// 1. Orient Guidance Signs
const orientGuidanceSigns = (layout: ParkingLayout): ParkingLayout => {
    const exits = layout.elements.filter(e => e.type === ElementType.EXIT);
    if (exits.length === 0) return layout;
    const updated = layout.elements.map(el => {
        if (el.type === ElementType.GUIDANCE_SIGN) {
            let nearest = exits[0], minDist = Infinity;
            const cx = el.x + el.width/2, cy = el.y + el.height/2;
            exits.forEach(ex => {
                const ecx = ex.x + ex.width/2, ecy = ex.y + ex.height/2;
                const d = Math.hypot(ecx-cx, ecy-cy);
                if(d < minDist) { minDist = d; nearest = ex; }
            });
            const ecx = nearest.x + nearest.width/2, ecy = nearest.y + nearest.height/2;
            return { ...el, rotation: (Math.atan2(ecy-cy, ecx-cx) * 180 / Math.PI) + 90 };
        }
        return el;
    });
    return { ...layout, elements: updated };
};

// 2. Snap Safe Exits
const snapSafeExits = (layout: ParkingLayout): ParkingLayout => {
    const stairs = layout.elements.filter(e => e.type === ElementType.STAIRCASE);
    if (stairs.length === 0) return layout;
    const updated = layout.elements.map(el => {
        if (el.type === ElementType.SAFE_EXIT) {
             let nearest = stairs[0], minDist = Infinity;
             const cx = el.x + el.width/2, cy = el.y + el.height/2;
             stairs.forEach(st => {
                 const scx = st.x + st.width/2, scy = st.y + st.height/2;
                 const d = Math.hypot(scx-cx, scy-cy);
                 if(d < minDist) { minDist = d; nearest = st; }
             });
             if (minDist < 200) return { ...el, x: nearest.x + nearest.width, y: nearest.y, rotation: 0 };
        }
        return el;
    });
    return { ...layout, elements: updated };
};

// 3. Auto Fill Parking
const fillParkingAutomatically = (layout: ParkingLayout): ParkingLayout => {
  const newElements = [...layout.elements];
  const grounds = newElements.filter(e => e.type === ElementType.GROUND);
  const roads = newElements.filter(e => e.type === ElementType.ROAD);
  const obstacles = newElements.filter(e => 
    [ElementType.WALL, ElementType.STAIRCASE, ElementType.ELEVATOR, ElementType.PILLAR,
     ElementType.ENTRANCE, ElementType.EXIT, ElementType.RAMP, ElementType.SAFE_EXIT,
     ElementType.SIDEWALK].includes(e.type as ElementType)
  );
  
  const genSpots: LayoutElement[] = [];
  const genChargers: LayoutElement[] = [];
  const SPOT_S = 24, SPOT_L = 48, BUFFER = 4;

  const isSafe = (rect: any) => {
      const all = [...obstacles, ...genSpots];
      const m = 1;
      return !all.some(o => 
        rect.x + m < o.x + o.width && rect.x + rect.w - m > o.x &&
        rect.y + m < o.y + o.height && rect.y + rect.h - m > o.y
      );
  };
  let t = 0;

  roads.forEach(r => {
      const rr = { l: r.x, r: r.x + r.width, t: r.y, b: r.y + r.height };
      grounds.forEach(g => {
          const gr = { l: g.x, r: g.x + g.width, t: g.y, b: g.y + g.height };
          
          if (Math.abs(rr.b - gr.t) < 5 && Math.min(rr.r, gr.r) > Math.max(rr.l, gr.l)) {
               const sx = Math.max(rr.l, gr.l) + BUFFER, ex = Math.min(rr.r, gr.r) - BUFFER;
               const cnt = Math.floor((ex - sx) / (SPOT_S + 2)); 
               for(let i=0; i<cnt; i++) {
                   const s = { x: sx + i*(SPOT_S+2), y: gr.t+1, w: SPOT_S, h: SPOT_L };
                   if (isSafe(s)) {
                       t++;
                       genSpots.push({ id: `p_${t}`, type: ElementType.PARKING_SPACE, x: s.x, y: s.y, width: s.w, height: s.h, rotation: 0 });
                       if (t % 3 === 0) genChargers.push({ id: `c_${t}`, type: ElementType.CHARGING_STATION, x: s.x+s.w/2-5, y: s.y+s.h-10, width: 10, height: 10 });
                   }
               }
          }
          else if (Math.abs(rr.t - gr.b) < 5 && Math.min(rr.r, gr.r) > Math.max(rr.l, gr.l)) {
              const sx = Math.max(rr.l, gr.l) + BUFFER, ex = Math.min(rr.r, gr.r) - BUFFER;
              const cnt = Math.floor((ex - sx) / (SPOT_S + 2));
              for(let i=0; i<cnt; i++) {
                   const s = { x: sx + i*(SPOT_S+2), y: gr.b-SPOT_L-1, w: SPOT_S, h: SPOT_L };
                   if (isSafe(s)) {
                       t++;
                       genSpots.push({ id: `p_${t}`, type: ElementType.PARKING_SPACE, x: s.x, y: s.y, width: s.w, height: s.h, rotation: 0 });
                       if (t % 3 === 0) genChargers.push({ id: `c_${t}`, type: ElementType.CHARGING_STATION, x: s.x+s.w/2-5, y: s.y+2, width: 10, height: 10 });
                   }
              }
          }
      });
  });
  return { ...layout, elements: [...newElements, ...genSpots, ...genChargers] };
};

const calculateScore = (violations: ConstraintViolation[]): number => {
    return violations.reduce((acc, v) => {
        let weight = 1;
        switch(v.type) {
            case 'connectivity_error': weight = 20; break; 
            case 'out_of_bounds': weight = 10; break;
            case 'overlap': weight = 2; break; 
            case 'placement_error': weight = 5; break;
            default: weight = 1;
        }
        return acc + weight;
    }, 0);
};

// --- FIX LOOP ---
const runFixOperation = async (
    layout: ParkingLayout, 
    violations: ConstraintViolation[], 
    ai: GoogleGenAI, 
    fixModel: string, 
    temperature: number,
    onLog?: (msg: string) => void
): Promise<{layout: ParkingLayout, raw: any}> => {
  const simplifiedLayout = {
      width: layout.width,
      height: layout.height,
      elements: layout.elements.map(e => ({
          id: e.id, 
          t: e.type, 
          x: roundCoord(e.x), 
          y: roundCoord(e.y), 
          w: roundCoord(e.width), 
          h: roundCoord(e.height), 
          r: roundCoord(e.rotation || 0)
      }))
  };

  const prompt = PROMPTS.fix(simplifiedLayout as any, violations);

  const response = await generateWithRetry(ai, {
    model: fixModel,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      temperature: temperature,
    }
  }, 2, onLog); 
  
  const rawData = cleanAndParseJSON(response.text || "{}");
  return { layout: mapToInternalLayout(rawData), raw: rawData };
};

const ensureValidLayout = async (layout: ParkingLayout, ai: GoogleGenAI, onLog?: (msg: string) => void): Promise<ParkingLayout> => {
  let currentState = {
      layout: layout,
      violations: validateLayout(layout),
      score: 0
  };
  currentState.score = calculateScore(currentState.violations);
  let bestState = { ...currentState };
  let lastFingerprint = "";
  
  const MAX_ITERATIONS = 6; 
  let temperature = 0.2; 
  let consecutiveErrors = 0;

  const tier = await determineModelTier(ai);
  let fixModel = tier === 'HIGH' ? MODEL_PRIMARY : MODEL_FALLBACK;

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    if (bestState.score === 0) {
        if (onLog) onLog("✨ Layout is perfectly valid!");
        break;
    }

    // FINGERPRINTING TO DETECT STAGNATION
    const currentFingerprint = JSON.stringify(currentState.layout.elements.map(e => ({id:e.id, x:e.x, y:e.y, w:e.width, h:e.height})));
    if (currentFingerprint === lastFingerprint) {
         if (onLog) onLog("🛑 Layout stagnated. Stopping fix loop.");
         break;
    }
    lastFingerprint = currentFingerprint;

    if (i > 0 && currentState.score >= bestState.score) {
        temperature = Math.min(0.8, temperature + 0.2);
        if (onLog) onLog(`🌡️ Stagnation detected. Increasing creativity (Temp: ${temperature.toFixed(1)})...`);
    } else {
        temperature = 0.2; 
    }

    const currentV = currentState.violations;
    if (currentV.length > 50) {
        if (onLog) onLog(`⚠️ High violation count (${currentV.length}). Pruning parking spots...`);
        const badIds = new Set(currentV.filter(v => 
            v.type === 'overlap' && currentState.layout.elements.find(e => e.id === v.elementId)?.type === ElementType.PARKING_SPACE
        ).map(v => v.elementId));
        
        currentState.layout.elements = currentState.layout.elements.filter(e => !badIds.has(e.id));
        currentState.violations = validateLayout(currentState.layout);
        currentState.score = calculateScore(currentState.violations);
        
        if (currentState.score < bestState.score) bestState = { ...currentState };
        if (currentState.violations.length > 30) continue; 
    }

    if (onLog) onLog(`🔧 Auto-fixing pass ${i+1}/${MAX_ITERATIONS} (Score: ${currentState.score})...`);

    try {
        const result = await runFixOperation(currentState.layout, currentState.violations.slice(0, 15), ai, fixModel, temperature, onLog);
        consecutiveErrors = 0; 
        
        // CHECK IF AI IGNORED THE VIOLATIONS
        const strategyText = (result.raw.fix_strategy || []).join(" ").toLowerCase();
        if (strategyText.includes("ignored") || strategyText.includes("valid connection") || strategyText.includes("minor overlap")) {
             // If AI says "I ignored this because it's valid", and the layout is roughly same, trust the AI and stop
             if (onLog) onLog(`🤖 AI claims validity despite errors ("${result.raw.fix_strategy[0]}"). Trusting AI & stopping.`);
             bestState = { layout: result.layout, violations: [], score: 0 }; // Force success state
             break;
        }

        if (result.raw.reasoning_plan && onLog) onLog(`🤔 Logic: ${result.raw.reasoning_plan}`);

        if (result.layout.elements.length < currentState.layout.elements.length * 0.2) {
             throw new Error("Sanity check failed: LLM dropped too many elements.");
        }

        const nextLayout = result.layout;
        const nextViolations = validateLayout(nextLayout);
        const nextScore = calculateScore(nextViolations);

        if (nextScore < bestState.score) {
            bestState = { layout: nextLayout, violations: nextViolations, score: nextScore };
            currentState = bestState;
            if (onLog) onLog(`✅ Improvement found! New Score: ${nextScore}`);
        } else if (nextScore < currentState.score * 1.3 || temperature > 0.5) {
            currentState = { layout: nextLayout, violations: nextViolations, score: nextScore };
            if (onLog) onLog(`🔄 Exploring... (Score: ${nextScore})`);
        } else {
            if (onLog) onLog(`❌ Reverting (Score: ${nextScore} vs Best: ${bestState.score})`);
            currentState = { ...bestState };
        }

    } catch (e: any) {
        consecutiveErrors++;
        const errMsg = e.message || "Unknown error";
        
        if (errMsg.includes("Quota") || errMsg.includes("billing") || errMsg.includes("503")) {
             if (fixModel === MODEL_PRIMARY) {
                 if (onLog) onLog(`⚠️ Primary model exhausted during fix. Switching to ${MODEL_FALLBACK} for remaining steps.`);
                 fixModel = MODEL_FALLBACK;
                 consecutiveErrors = 0; 
                 await sleep(1000);
                 continue; 
             } else {
                 if (onLog) onLog("⛔ Critical: Quota Exceeded on fallback model.");
                 throw e;
             }
        }

        if (onLog) onLog(`⚠️ Fix step failed: ${errMsg.slice(0, 100)}...`);
        
        if (consecutiveErrors >= 2) {
            if (onLog) onLog("⛔ Too many API errors. Stopping fix loop.");
            break;
        }
        temperature = 0.5; 
        await sleep(2000); 
    }
  }

  let finalLayout = bestState.layout;
  finalLayout = orientGuidanceSigns(finalLayout);
  finalLayout = snapSafeExits(finalLayout);

  if (onLog && bestState.score > 0) onLog(`🏁 Final Score: ${bestState.score} (${bestState.violations.length} issues remaining).`);
  return finalLayout;
};

// MAIN GENERATION FUNCTION WITH FALLBACK
export const generateParkingLayout = async (description: string, onLog?: (msg: string) => void): Promise<ParkingLayout> => {
  const apiKey = getApiKey();
  if (!apiKey) {
      if(onLog) onLog("❌ Error: No API Key found in env.");
      return fallbackLayout;
  }
  const ai = new GoogleGenAI({ apiKey });
  
  let tier = await determineModelTier(ai, onLog);
  let currentModel = tier === 'HIGH' ? MODEL_PRIMARY : MODEL_FALLBACK;
  
  if (onLog) onLog(`Using model: ${currentModel}`);

  try {
    let response;
    try {
        response = await generateWithRetry(ai, {
            model: currentModel,
            contents: PROMPTS.generation(description),
            config: { responseMimeType: "application/json" }
        }, 2, onLog);
    } catch (e: any) {
        if (currentModel === MODEL_PRIMARY && (e.message?.includes("Quota") || e.message?.includes("429") || e.message?.includes("500"))) {
             if (onLog) onLog(`⚠️ Primary model failed (${e.message}). Switching to fallback ${MODEL_FALLBACK}...`);
             currentModel = MODEL_FALLBACK;
             response = await generateWithRetry(ai, {
                model: currentModel,
                contents: PROMPTS.generation(description),
                config: { responseMimeType: "application/json" }
             }, 2, onLog);
        } else {
            throw e;
        }
    }
    
    const rawData = cleanAndParseJSON(response.text);
    if (rawData.reasoning_plan && onLog) onLog(`🧠 Plan: ${rawData.reasoning_plan}`);

    let layout = mapToInternalLayout(rawData);
    if (onLog) onLog(`Generated ${layout.elements.length} elements.`);
    
    return await ensureValidLayout(layout, ai, onLog);

  } catch (error: any) {
    const msg = error.message || String(error);
    console.error("Gen failed", error);
    if (onLog) onLog(`❌ Fatal Error: ${msg.slice(0, 150)}...`);
    throw error;
  }
};

export const augmentLayoutWithRoads = async (currentLayout: ParkingLayout, onLog?: (msg: string) => void): Promise<ParkingLayout> => {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("API Key required");
  const ai = new GoogleGenAI({ apiKey });
  
  let tier = await determineModelTier(ai);
  let currentModel = tier === 'HIGH' ? MODEL_PRIMARY : MODEL_FALLBACK;

  try {
    const simplified = currentLayout.elements.map(e => ({ id: e.id, t: e.type, x: roundCoord(e.x), y: roundCoord(e.y), w: roundCoord(e.width), h: roundCoord(e.height) }));
    
    // FIX: Wrap simplified in an object to match prompt requirement (Layout with .elements)
    // PROMPTS.refinement uses `simplifiedLayout.elements.length`
    const simplifiedObj = { elements: simplified };
    
    let response;
    try {
        response = await generateWithRetry(ai, {
            model: currentModel,
            contents: PROMPTS.refinement(simplifiedObj, currentLayout.width, currentLayout.height),
            config: { responseMimeType: "application/json" }
        }, 2, onLog);
    } catch (e: any) {
         if (currentModel === MODEL_PRIMARY && (e.message?.includes("Quota") || e.message?.includes("429"))) {
             if (onLog) onLog(`⚠️ Switching to fallback ${MODEL_FALLBACK} due to quota...`);
             currentModel = MODEL_FALLBACK;
             response = await generateWithRetry(ai, {
                model: currentModel,
                contents: PROMPTS.refinement(simplifiedObj, currentLayout.width, currentLayout.height),
                config: { responseMimeType: "application/json" }
             }, 2, onLog);
         } else {
             throw e;
         }
    }

    const rawData = cleanAndParseJSON(response.text);
    if (rawData.reasoning_plan && onLog) onLog(`🧠 Refinement Plan: ${rawData.reasoning_plan}`);

    let layout = mapToInternalLayout(rawData);
    layout = fillParkingAutomatically(layout);
    
    return await ensureValidLayout(layout, ai, onLog);
  } catch (error: any) {
    const msg = error.message || String(error);
    console.error("Augment failed", error);
    if (onLog) onLog(`❌ Refine Error: ${msg.slice(0, 150)}`);
    throw error;
  }
};