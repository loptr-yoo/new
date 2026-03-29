import { BuildingData, ParkingLayout } from '../types';
import { AIProvider } from '../utils/aiConfig';

export interface AIServiceOptions {
  provider: AIProvider;
  model: string;
  apiKey: string;
}

const API_BASE = 'http://localhost:8000/api';

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

/**
 * 生成布局入口
 */
export const generateLayout = async (
  prompt: string,
  options: AIServiceOptions,
  onProgress?: (msg: string) => void,
  sceneId?: string
): Promise<ParkingLayout> => {
  onProgress?.('请求后端生成布局...');
  const data = await safeFetchJSON<BuildingData>(`${API_BASE}/generate`, {
    prompt,
    provider: options.provider,
    model: options.model,
    sceneId,
  });
  const firstFloor = Object.keys(data.floors || {})[0];
  if (!firstFloor) throw new Error('Empty floors in response.');
  return data.floors[firstFloor];
};

export const generateBuilding = async (
  prompt: string,
  options: AIServiceOptions,
  onProgress?: (msg: string) => void,
  sceneId?: string
): Promise<BuildingData> => {
  onProgress?.('请求后端生成楼宇...');
  return await safeFetchJSON<BuildingData>(`${API_BASE}/generate`, {
    prompt,
    provider: options.provider,
    model: options.model,
    sceneId,
  });
};

/**
 * 细化布局入口
 */
export const augmentLayoutWithRoads = async (
  layout: ParkingLayout,
  options: AIServiceOptions,
  onProgress?: (msg: string) => void,
  sceneId?: string
): Promise<ParkingLayout> => {
  onProgress?.('请求后端细化布局...');
  return await safeFetchJSON<ParkingLayout>(`${API_BASE}/augment`, {
    layout,
    provider: options.provider,
    model: options.model,
    sceneId,
  });
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
