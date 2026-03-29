import React, { useState, useRef, useMemo, useEffect } from 'react';
import { Grid, ChevronDown } from 'lucide-react';
import { SceneDefinition } from '../types';

interface SceneSelectorProps {
  scenes: Record<string, SceneDefinition>;
  activeSceneId: string;
  onSceneChange: (id: string) => void;
}

const ITEM_HEIGHT = 40;
const VISIBLE_ITEMS = 5;

// 虚拟滚动场景选择器，优化前端渲染耗时
export const SceneSelector: React.FC<SceneSelectorProps> = ({ scenes, activeSceneId, onSceneChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const sceneList = useMemo(() => Object.values(scenes), [scenes]);
  const activeScene = scenes[activeSceneId];

  // 计算虚拟滚动
  const startIndex = Math.floor(scrollTop / ITEM_HEIGHT);
  const endIndex = Math.min(startIndex + VISIBLE_ITEMS + 1, sceneList.length);
  const visibleScenes = sceneList.slice(startIndex, endIndex);
  const totalHeight = sceneList.length * ITEM_HEIGHT;

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  };

  // 点击外部关闭
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="space-y-2 relative" ref={containerRef}>
      <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1">
        <Grid size={12} /> Scene
      </label>
      
      <div 
        className="w-full bg-slate-800 border border-slate-700 rounded text-xs text-white p-2 cursor-pointer flex justify-between items-center"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>{activeScene?.name || 'Select Scene'}</span>
        <ChevronDown size={14} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </div>

      {isOpen && (
        <div className="absolute z-50 w-full mt-1 bg-slate-800 border border-slate-700 rounded-md shadow-xl overflow-hidden">
          <div 
            className="overflow-y-auto custom-scrollbar" 
            style={{ maxHeight: `${VISIBLE_ITEMS * ITEM_HEIGHT}px` }}
            onScroll={handleScroll}
          >
            <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
              {visibleScenes.map((scene, index) => {
                const actualIndex = startIndex + index;
                return (
                  <div
                    key={scene.id}
                    onClick={() => {
                      onSceneChange(scene.id);
                      setIsOpen(false);
                    }}
                    className={`absolute w-full px-3 flex items-center cursor-pointer hover:bg-slate-700 transition-colors ${activeSceneId === scene.id ? 'bg-blue-900/30 text-blue-400' : 'text-slate-200'}`}
                    style={{ 
                      height: `${ITEM_HEIGHT}px`, 
                      top: `${actualIndex * ITEM_HEIGHT}px` 
                    }}
                  >
                    {scene.name}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {activeScene?.description && (
        <p className="text-[10px] text-slate-500 leading-tight">
          {activeScene.description}
        </p>
      )}
    </div>
  );
};
