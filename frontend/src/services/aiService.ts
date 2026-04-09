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
  const timeoutSignal = AbortSignal.timeout(60_000);
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

    for (const room of floorData.rooms) {
      elements.push({
        id: room.room_id,
        type: room.room_type,
        polygon: room.polygon,
        x: room.center[0] - room.width / 2,
        y: room.center[1] - room.depth / 2,
        width: room.width,
        height: room.depth,
        label: room.room_id,
      });
    }

    // 核心筒子区域
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

    floors[floorId] = {
      width: v2.building.width,
      height: v2.building.depth,
      elements,
      sceneId: 'building_floor_plan',
    };
  }

  return { blueprint: [], floors };
}
