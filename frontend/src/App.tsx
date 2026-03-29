
import React, { useRef, useEffect, useState } from 'react';
import MapRenderer, { MapRendererHandle } from './components/MapRenderer';
import LayoutControl from './components/LayoutControl';
import { generateBuilding, augmentLayout, getApiKeyFromEnv } from './services/aiService';
import { useStore } from './store';

// 声明 aistudio 全局接口
// 🛡️ 修复：环境已将 window.aistudio 声明为 AIStudio 类型。
// 直接增强 AIStudio 接口并移除对 Window 的重复声明，以避免属性冲突和修饰符不一致错误。
declare global {
  interface AIStudio {
    hasSelectedApiKey: () => Promise<boolean>;
    openSelectKey: () => Promise<void>;
  }
}

const App: React.FC = () => {
  const { 
    layout, buildingData, activeFloorId, violations, isGenerating, error, logs, selectedProvider, selectedModel, activeSceneId,
    setLayout, setBuildingData, setActiveFloor, updateFloorLayout, setViolations, setIsGenerating, setError, addLog, setGenerationTime, clearLogs 
  } = useStore();

  const mapRef = useRef<MapRendererHandle>(null);
  const [hasKey, setHasKey] = useState<boolean>(true);

  useEffect(() => {
    const checkKey = async () => {
      if (window.aistudio) {
        const selected = await window.aistudio.hasSelectedApiKey();
        setHasKey(selected);
      }
    };
    checkKey();
  }, []);

  const handleSelectKey = async () => {
    if (window.aistudio) {
      await window.aistudio.openSelectKey();
      setHasKey(true);
      setError(null);
    }
  };

  const handleGenerate = async (prompt: string, sceneId: string) => {
    setIsGenerating(true);
    setError(null);
    clearLogs();
    setGenerationTime(null);
    const startTime = Date.now();
    
    try {
      const apiKey = getApiKeyFromEnv(selectedProvider);
      if (!apiKey) {
        throw new Error(`No API Key configured for ${selectedProvider}. Please set it in your environment.`);
      }

      addLog(`使用 ${selectedModel} 模型...`);
      setBuildingData(null);
      setActiveFloor('');

      const newBuilding = await generateBuilding(prompt, {
        provider: selectedProvider,
        model: selectedModel,
        apiKey,
      }, addLog, sceneId);
      
      setBuildingData(newBuilding);
      const floorIds = Object.keys(newBuilding.floors);
      if (floorIds.length > 0) {
        const firstFloor = floorIds[0];
        setActiveFloor(firstFloor);
        setLayout(newBuilding.floors[firstFloor]);
      }
      
      setViolations([]); 
      addLog("Generation complete.");
    } catch (e: any) {
      console.error(e);
      const msg = e.message || "";
      if (msg.includes("429") || msg.includes("quota") || msg.includes("rate_limit")) {
        setError("API 配额已耗尽或请求过于频繁。请稍后重试。");
      } else if (msg.includes("not found")) {
        setError("无法找到指定的模型。请检查 API Key 和模型配置。");
        setHasKey(false);
      } else if (msg.includes("No API Key")) {
        setError(msg);
      } else {
        setError(msg || "生成布局失败。");
      }
    } finally {
      setIsGenerating(false);
      setGenerationTime((Date.now() - startTime) / 1000);
    }
  };

  const handleRefine = async () => {
    if (!layout) return;
    setIsGenerating(true);
    addLog("--- Refinement ---");
    const startTime = Date.now();
    
    try {
      const apiKey = getApiKeyFromEnv(selectedProvider);
      if (!apiKey) {
        throw new Error(`No API Key configured for ${selectedProvider}.`);
      }

      addLog(`使用 ${selectedModel} 细化布局...`);
      const augmented = await augmentLayout(layout, {
        provider: selectedProvider,
        model: selectedModel,
        apiKey,
      }, addLog, activeSceneId);
      
      if (augmented && augmented.elements.length > 0) {
        setLayout(augmented);
        if (activeFloorId) {
          updateFloorLayout(activeFloorId, augmented);
        }
        setViolations([]);
        addLog("Refinement complete.");
      }
    } catch (e: any) {
      const msg = e.message || "";
      if (msg.includes("429") || msg.includes("rate_limit")) {
        setError("配额不足，细化操作失败。请稍后重试。");
      } else {
        setError(msg || "细化布局失败。");
      }
    } finally {
      setIsGenerating(false);
      setGenerationTime((Date.now() - startTime) / 1000);
    }
  };

  const handleDownload = () => {
      if (mapRef.current) {
          mapRef.current.downloadJpg();
      }
  };

  const handleDownloadJson = () => {
      if (mapRef.current) {
          mapRef.current.downloadJson();
      }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans">
      <div className="flex-1 flex flex-col p-4 min-w-0">
        <header className="mb-4 flex items-center justify-between">
            <h1 className="text-2xl font-bold tracking-tight text-white">
                <span className="text-blue-500">P</span>arking<span className="text-purple-500">V</span>iz
                <span className="ml-2 text-xs font-normal text-slate-500">AI-Powered Spatial Designer</span>
            </h1>
            <div className="flex gap-4 text-xs text-slate-500">
               {!hasKey && (
                 <div className="flex items-center gap-1 text-amber-500 animate-pulse">
                   ⚠️ 请关联 API Key 以开启 Gemini 3 Pro
                 </div>
               )}
               <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500"></span> Coarse</div>
               <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500"></span> Fine</div>
            </div>
        </header>

        {error && (
            <div className="bg-red-500/10 border border-red-500/50 text-red-200 px-4 py-3 rounded mb-4 text-sm flex flex-col gap-2">
                <div className="flex justify-between items-start">
                  <span>{error}</span>
                  <button onClick={() => setError(null)} className="hover:bg-red-500/20 px-2 rounded">✕</button>
                </div>
                {(error.includes("配额") || error.includes("Key")) && (
                  <button 
                    onClick={handleSelectKey}
                    className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-xs w-fit font-bold"
                  >
                    立即设置自定义 API Key
                  </button>
                )}
            </div>
        )}

        <main className="flex-1 min-h-0 relative">
            {buildingData && Object.keys(buildingData.floors).length > 1 && (
                <div className="absolute top-4 left-4 z-10 flex flex-wrap gap-2 bg-slate-900/80 p-2 rounded-lg border border-slate-700 backdrop-blur max-w-[80%]">
                    <span className="text-slate-400 text-xs flex items-center mr-2">楼层:</span>
                    {Object.keys(buildingData.floors).map(floorId => {
                        const floorName = floorId.replace('floor_', '') + 'F';
                        return (
                            <button
                                key={floorId}
                                onClick={() => {
                                    setActiveFloor(floorId);
                                    setLayout(buildingData.floors[floorId]);
                                }}
                                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                                    activeFloorId === floorId 
                                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' 
                                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                                }`}
                            >
                                {floorName}
                            </button>
                        );
                    })}
                </div>
            )}
            {layout ? <MapRenderer ref={mapRef} /> : (
                <div className="w-full h-full flex flex-col items-center justify-center border border-slate-800 rounded-lg bg-slate-900/50 gap-4">
                    <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center">
                      <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A2 2 0 013 15.382V6.618a2 2 0 011.553-1.944L9 4m0 16l6-3m-6 3V4m6 13l5.447 2.724A2 2 0 0021 18.018V9.236a2 2 0 00-1.553-1.944L15 4m0 13V4m0 0L9 7" />
                      </svg>
                    </div>
                    <p className="text-slate-500 text-sm">Enter prompt and Generate Structure.</p>
                </div>
            )}
        </main>
      </div>

      <LayoutControl 
        onGenerate={handleGenerate} 
        onRefine={handleRefine}
        onDownload={handleDownload}
        onDownloadJson={handleDownloadJson}
        onSelectKey={handleSelectKey}
      />
    </div>
  );
};

export default App;
