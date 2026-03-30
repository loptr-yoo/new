import { BuildingData, ParkingLayout } from '../types';
import { AIProvider } from '../utils/aiConfig';

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

const safeFetchJSON = async <T>(url: string, body: any): Promise<T> => {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`API Error (${res.status}): ${txt}`);
  }
  return (await res.json()) as T;
};

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
  const data = await fetchSSE<BuildingData>(
    `/api/generate/stream`,
    {
      prompt,
      provider: options.provider,
      model: options.model,
      sceneId,
    },
    onProgress,
    signal
  );
  const firstFloor = Object.keys(data.floors || {})[0];
  if (!firstFloor) throw new Error('Empty floors in response.');
  return data.floors[firstFloor];
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
  return await fetchSSE<BuildingData>(
    `/api/generate/stream`,
    {
      prompt,
      provider: options.provider,
      model: options.model,
      sceneId,
    },
    onProgress,
    signal
  );
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
  return await fetchSSE<ParkingLayout>(
    `/api/augment/stream`,
    {
      layout,
      provider: options.provider,
      model: options.model,
      sceneId,
    },
    onProgress,
    signal
  );
};

// --- 辅助函数保持不变 ---

export const getApiKeyFromEnv = (provider: AIProvider): string => {
  return 'backend_env';
};

export const checkAvailableProviders = (): AIProvider[] => {
  return ['gemini', 'deepseek', 'openai'];
};

// 添加一个别名函数以保持向后兼容性
export const augmentLayout = augmentLayoutWithRoads;
