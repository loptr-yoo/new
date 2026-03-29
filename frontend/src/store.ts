import { create } from 'zustand';
import { ParkingLayout, ConstraintViolation, BuildingData } from './types';
import { AIProvider } from './utils/aiConfig';
import { DEFAULT_SCENE_ID, SCENE_REGISTRY } from './utils/sceneRegistry';

interface AppState {
  layout: ParkingLayout | null;
  buildingData: BuildingData | null;
  activeFloorId: string;
  violations: ConstraintViolation[];
  isGenerating: boolean;
  error: string | null;
  logs: string[];
  generationTime: number | null;

  activeSceneId: string;
  
  // AI 模型相关
  selectedProvider: AIProvider;
  selectedModel: string;
  availableModels: Array<{ id: string; name: string; provider: AIProvider }>;

  // Actions
  setLayout: (layout: ParkingLayout | null) => void;
  setBuildingData: (data: BuildingData | null) => void;
  setActiveFloor: (floorId: string) => void;
  updateFloorLayout: (floorId: string, layout: ParkingLayout) => void;
  setViolations: (violations: ConstraintViolation[]) => void;
  setIsGenerating: (isGenerating: boolean) => void;
  setError: (error: string | null) => void;
  addLog: (msg: string) => void;
  clearLogs: () => void;
  setGenerationTime: (time: number | null) => void;
  setSelectedProvider: (provider: AIProvider) => void;
  setSelectedModel: (model: string) => void;
  switchScene: (sceneId: string) => void;
}

export const useStore = create<AppState>((set) => ({
  layout: null,
  buildingData: null,
  activeFloorId: '',
  violations: [],
  isGenerating: false,
  error: null,
  logs: [],
  generationTime: null,
  activeSceneId: DEFAULT_SCENE_ID,
  selectedProvider: 'gemini' as AIProvider,
  selectedModel: 'gemini-2.5-pro',
  availableModels: [],

  setLayout: (layout) => set({ layout }),
  setBuildingData: (data) => set({ buildingData: data }),
  setActiveFloor: (floorId) => set({ activeFloorId: floorId }),
  updateFloorLayout: (floorId, layout) => set((state) => {
    if (!state.buildingData) return state;
    return {
      buildingData: {
        ...state.buildingData,
        floors: {
          ...state.buildingData.floors,
          [floorId]: layout
        }
      }
    };
  }),
  setViolations: (violations) => set({ violations }),
  setIsGenerating: (isGenerating) => set({ isGenerating }),
  setError: (error) => set({ error }),
  addLog: (msg) => set((state) => ({ logs: [...state.logs, msg] })),
  clearLogs: () => set({ logs: [] }),
  setGenerationTime: (time) => set({ generationTime: time }),
  setSelectedProvider: (provider) => set({ selectedProvider: provider }),
  setSelectedModel: (model) => set({ selectedModel: model }),
  switchScene: (sceneId) => {
    if (SCENE_REGISTRY[sceneId]) set({ activeSceneId: sceneId });
  },
}));
