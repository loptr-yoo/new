import { BuildingData, BuildingDataV2, LayoutElement, ParkingLayout } from '../types';
import { AIProvider } from '../utils/aiConfig';
import { sanitizeBuildingData, sanitizeLayout } from '../utils/sanitizeLayout';

export interface AIServiceOptions {
  provider: AIProvider;
  model: string;
  apiKey: string;
}

type StreamEvent =
  | { status: 'start' }
  | { status: 'ping' }
  | { status: 'progress'; msg: string }
  | { status: 'done'; data: any }
  | { status: 'error'; message: string };

const fetchSSE = async <T>(
  url: string,
  body: any,
  onProgress?: (msg: string) => void,
  externalSignal?: AbortSignal
): Promise<T> => {
  const internalController = new AbortController();
  
  const handleExternalAbort = () => internalController.abort(externalSignal?.reason);
  if (externalSignal) {
    if (externalSignal.aborted) throw new Error('AbortError');
    externalSignal.addEventListener('abort', handleExternalAbort);
  }

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: internalController.signal,
  });
  if (!res.ok) {
    if (externalSignal) externalSignal.removeEventListener('abort', handleExternalAbort);
    const txt = await res.text();
    throw new Error(`API Error (${res.status}): ${txt}`);
  }
  if (!res.body) {
    if (externalSignal) externalSignal.removeEventListener('abort', handleExternalAbort);
    throw new Error('Empty response stream.');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let donePayload: any = null;

  // Timeouts
  const TTFB_TIMEOUT_MS = 30000;
  const IDLE_TIMEOUT_MS = 15000;
  let timeoutId: any;

  const resetTimeout = (ms: number) => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      internalController.abort(new Error('Idle Timeout'));
    }, ms);
  };

  resetTimeout(TTFB_TIMEOUT_MS);

  try {
    while (true) {
      const { value, done } = await reader.read();
      resetTimeout(IDLE_TIMEOUT_MS);
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';
      for (const part of parts) {
        const line = part
          .split('\n')
          .map(s => s.trim())
          .find(s => s.startsWith('data:'));
        if (!line) continue;
        const jsonText = line.slice('data:'.length).trim();
        if (!jsonText) continue;
        let evt: StreamEvent;
        try {
          evt = JSON.parse(jsonText);
        } catch {
          continue;
        }
        if (evt.status === 'progress') onProgress?.(evt.msg);
        if (evt.status === 'error') throw new Error(evt.message);
        if (evt.status === 'done') donePayload = evt.data;
      }
    }
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
    if (externalSignal) externalSignal.removeEventListener('abort', handleExternalAbort);
  }

  if (donePayload == null) throw new Error('Stream ended without payload.');
  return donePayload as T;
};

/**
 * 生成布局入口
 */
export const generateLayout = async (
  prompt: string,
  options: AIServiceOptions,
  onProgress?: (msg: string) => void,
  sceneId?: string,
  signal?: AbortSignal
): Promise<ParkingLayout> => {
  onProgress?.('请求后端生成布局...');
  const data = sanitizeBuildingData(await fetchSSE<BuildingData>(
    `/api/generate/stream`,
    {
      prompt,
      provider: options.provider,
      model: options.model,
      sceneId,
    },
    onProgress,
    signal
  ));
  const firstFloor = Object.keys(data.floors || {})[0];
  if (!firstFloor) throw new Error('Empty floors in response.');
  return sanitizeLayout(data.floors[firstFloor]);
};

export const generateBuilding = async (
  prompt: string,
  options: AIServiceOptions,
  onProgress?: (msg: string) => void,
  sceneId?: string,
  signal?: AbortSignal
): Promise<BuildingData> => {
  if (sceneId === 'building') {
    onProgress?.('请求后端规划整栋楼宇...');
  } else {
    onProgress?.('请求后端生成基础布局...');
  }
  return sanitizeBuildingData(await fetchSSE<BuildingData>(
    `/api/generate/stream`,
    {
      prompt,
      provider: options.provider,
      model: options.model,
      sceneId,
    },
    onProgress,
    signal
  ));
};

/**
 * 细化布局入口
 */
export const augmentLayoutWithRoads = async (
  layout: ParkingLayout,
  options: AIServiceOptions,
  onProgress?: (msg: string) => void,
  sceneId?: string,
  signal?: AbortSignal
): Promise<ParkingLayout> => {
  onProgress?.('请求后端细化布局...');
  return sanitizeLayout(await fetchSSE<ParkingLayout>(
    `/api/augment/stream`,
    {
      layout,
      provider: options.provider,
      model: options.model,
      sceneId,
    },
    onProgress,
    signal
  ));
};

// --- 辅助函数保持不变 ---

export const getApiKeyFromEnv = (_provider: AIProvider): string => {
  return 'backend_env';
};

export const checkAvailableProviders = (): AIProvider[] => {
  return ['gemini', 'deepseek', 'openai'];
};

// 添加一个别名函数以保持向后兼容性
export const augmentLayout = augmentLayoutWithRoads;


// ============================================================
// V2 新管线（Building 模式专用）
// ============================================================

/**
 * 调用新管线 POST /api/building/generate（非流式，含 60s 超时）
 */
export async function generateBuildingV2(
  prompt: string,
  options: AIServiceOptions,
  signal?: AbortSignal,
): Promise<BuildingDataV2> {
  const timeoutSignal = AbortSignal.timeout(180_000);
  const combinedSignal = signal
    ? AbortSignal.any([signal, timeoutSignal])
    : timeoutSignal;

  const resp = await fetch('/api/building/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      provider: options.provider,
      model: options.model,
    }),
    signal: combinedSignal,
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`Building generation failed (${resp.status}): ${text}`);
  }

  return resp.json();
}

/**
 * 从 polygon 坐标数组计算 bounding box
 */
function getBoundsFromPolygon(pts: [number, number][]): [number, number, number, number] {
  const xs = pts.map(p => p[0]);
  const ys = pts.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

/**
 * 从墙体线段坐标计算带厚度的矩形 bounds
 * 解决垂直/水平墙 width=0 或 height=0 导致 SVG 不渲染的问题
 */
function getWallBounds(coords: [number, number][], thickness: number = 0.2) {
  const xs = coords.map(c => c[0]);
  const ys = coords.map(c => c[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  return {
    x: maxX - minX === 0 ? minX - thickness / 2 : minX,
    y: maxY - minY === 0 ? minY - thickness / 2 : minY,
    width:  maxX - minX === 0 ? thickness : maxX - minX,
    height: maxY - minY === 0 ? thickness : maxY - minY,
  };
}

/**
 * 楼层 ID 数字排序（F1, F2, F3... 而非字符串排序 F1, F10, F2）
 */
function sortFloorIds(ids: string[]): string[] {
  return [...ids].sort((a, b) => {
    const numA = parseInt(a.replace(/\D/g, ''), 10) || 0;
    const numB = parseInt(b.replace(/\D/g, ''), 10) || 0;
    return numA - numB;
  });
}

/**
 * 将 V2 新管线返回转换为现有 BuildingData 格式（复用渲染逻辑）
 * 直接透传 polygon 精确几何，矩形字段保留做 fallback
 */
export function convertV2ToBuildingData(v2: BuildingDataV2): BuildingData {
  const sortedFloorIds = sortFloorIds(Object.keys(v2.building.floors));
  const floors: Record<string, ParkingLayout> = {};

  for (const floorId of sortedFloorIds) {
    const floorData = v2.building.floors[floorId];
    const elements: LayoutElement[] = [];

    // 楼板背景（最底层）
    if (floorData.floor_slab) {
      const slab = floorData.floor_slab;
      const [minX, minY, maxX, maxY] = getBoundsFromPolygon(slab.polygon);
      elements.push({
        id: `${floorId}_floor_slab`,
        type: 'floor_slab',
        polygon: slab.polygon,
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
      });
    }

    // 走廊
    for (const corridor of (floorData.corridors || [])) {
      const [minX, minY, maxX, maxY] = getBoundsFromPolygon(corridor.polygon);
      elements.push({
        id: corridor.id,
        type: 'corridor',
        polygon: corridor.polygon,
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
      });
    }

    // ====== Z-Order: SVG 没有 z-index，DOM 插入顺序 = 层叠顺序 ======
    // push 顺序：rooms → core_tube → walls → doors → windows
    // （floor_slab 和 corridors 已在上方 push，作为最底层）

    // Layer 3: 房间（带极细线兜底）
    for (const room of floorData.rooms) {
      const [minX, minY, maxX, maxY] = getBoundsFromPolygon(room.polygon);
      elements.push({
        id: room.room_id,
        type: room.room_type,
        polygon: room.polygon,
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
        label: room.room_id,
      });
    }

    // Layer 4a: 核心筒底色（不透明，遮挡走廊穿透效果）
    if (v2.core_tube?.boundary) {
      const [minX, minY, maxX, maxY] = getBoundsFromPolygon(v2.core_tube.boundary);
      elements.push({
        id: `${floorId}_core_bg`,
        type: 'floor_slab',
        polygon: v2.core_tube.boundary,
        x: minX, y: minY,
        width: maxX - minX, height: maxY - minY,
      });
    }

    // Layer 4b: 核心筒子区域（elevator + staircase，画在底色上方）
    if (v2.core_tube?.elevator) {
      const [minX, minY, maxX, maxY] = getBoundsFromPolygon(v2.core_tube.elevator.polygon);
      elements.push({
        id: `${floorId}_elevator`,
        type: 'elevator',
        polygon: v2.core_tube.elevator.polygon,
        x: minX, y: minY,
        width: maxX - minX, height: maxY - minY,
      });
    }
    if (v2.core_tube?.staircase) {
      const [minX, minY, maxX, maxY] = getBoundsFromPolygon(v2.core_tube.staircase.polygon);
      elements.push({
        id: `${floorId}_staircase`,
        type: 'staircase',
        polygon: v2.core_tube.staircase.polygon,
        x: minX, y: minY,
        width: maxX - minX, height: maxY - minY,
      });
    }
    if (!v2.core_tube?.elevator && !v2.core_tube?.staircase && v2.core_tube?.boundary) {
      const [minX, minY, maxX, maxY] = getBoundsFromPolygon(v2.core_tube.boundary);
      elements.push({
        id: `${floorId}_core_tube`,
        type: 'elevator',
        polygon: v2.core_tube.boundary,
        x: minX, y: minY,
        width: maxX - minX, height: maxY - minY,
      });
    }

    // Layer 5: 墙体（优先 polygon 精确渲染，fallback 到 getWallBounds）
    for (const wall of (floorData.walls || [])) {
      if (wall.polygon && wall.polygon.length >= 3) {
        const [minX, minY, maxX, maxY] = getBoundsFromPolygon(wall.polygon);
        elements.push({
          id: `${floorId}_wall_${elements.length}`,
          type: wall.type,
          polygon: wall.polygon,
          x: minX, y: minY,
          width: maxX - minX, height: maxY - minY,
        });
      } else if (wall.coords && wall.coords.length >= 2) {
        elements.push({
          id: `${floorId}_wall_${elements.length}`,
          type: wall.type,
          ...getWallBounds(wall.coords, wall.thickness || 0.2),
        });
      }
    }

    // Layer 6: 门窗（最顶层）— 厚度与墙体对齐（避免 door thickness 超出墙体边界）
    const floorMinDim = Math.min(v2.building.width, v2.building.depth);
    const VISUAL_THICKNESS = Math.max(0.3, floorMinDim * 0.025);
    const EXTERIOR_THICKNESS =
      (floorData.walls || []).find(w => w.type === 'exterior_wall')?.thickness ?? 0.24;
    const PARTITION_THICKNESS = Math.max(
      0.12,
      ...(floorData.walls || [])
        .filter(w => w.type === 'partition_wall' && (w.room_ids?.length ?? 0) >= 2)
        .map(w => w.thickness ?? 0.12)
    );

    for (const door of (floorData.doors || [])) {
      const isVertical = (door.rotation ?? 0) === 90;
      const doorDepth = door.thickness ?? Math.min(VISUAL_THICKNESS, EXTERIOR_THICKNESS, PARTITION_THICKNESS);
      const halfT = doorDepth / 2;
      elements.push({
        id: `${floorId}_door_${elements.length}`,
        type: 'door',
        x: isVertical ? door.position[0] - halfT : door.position[0] - door.width / 2,
        y: isVertical ? door.position[1] - door.width / 2 : door.position[1] - halfT,
        width:  isVertical ? doorDepth : door.width,
        height: isVertical ? door.width : doorDepth,
        forward: door.forward,
      });
    }

    for (const win of (floorData.windows || [])) {
      const isVertical = (win.rotation ?? 0) === 90;
      const windowDepth = Math.min(VISUAL_THICKNESS, EXTERIOR_THICKNESS);
      const halfT = windowDepth / 2;
      elements.push({
        id: `${floorId}_window_${elements.length}`,
        type: 'window',
        x: isVertical ? win.position[0] - halfT : win.position[0] - win.width / 2,
        y: isVertical ? win.position[1] - win.width / 2 : win.position[1] - halfT,
        width:  isVertical ? windowDepth : win.width,
        height: isVertical ? win.width : windowDepth,
        forward: win.forward,
      });
    }

    floors[floorId] = {
      width: v2.building.width,
      height: v2.building.depth,
      elements,
      sceneId: 'building_floor_plan',
    };
  }

  return { blueprint: [], floors };
}
