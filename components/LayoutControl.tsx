import React, { useState } from 'react';
import { Layers, Map as MapIcon, Sparkles, Download, Key } from 'lucide-react';
import { useStore } from '../store';

interface Props {
  onGenerate: (p: string) => void;
  onRefine: () => void;
  onDownload: () => void;
  onSelectKey: () => void;
}

const LayoutControl: React.FC<Props> = ({ onGenerate, onRefine, onDownload, onSelectKey }) => {
  const { isGenerating, violations, layout, logs } = useStore();
  const [prompt, setPrompt] = useState("Underground parking, rectangular, 2 main lanes, central islands.");
  const hasLayout = !!layout;

  return (
    <div className="w-full md:w-80 flex-shrink-0 bg-slate-900 border-l border-slate-800 p-4 overflow-y-auto flex flex-col gap-6">
      
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-400" /> Control Panel
        </h2>
        <button 
          onClick={onSelectKey}
          title="Select API Key"
          className="p-2 hover:bg-slate-800 rounded-full text-slate-400 hover:text-blue-400 transition-colors"
        >
          <Key size={18} />
        </button>
      </div>

      <div className="space-y-3">
        <label className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Design Prompt</label>
        <textarea 
          className="w-full bg-slate-800 border border-slate-700 rounded p-3 text-sm text-slate-200 h-24 resize-none focus:outline-none focus:border-blue-500 transition-colors"
          placeholder="Describe your parking layout..."
          value={prompt} onChange={(e) => setPrompt(e.target.value)}
        />
        
        <div className="grid grid-cols-1 gap-3">
            <button onClick={() => onGenerate(prompt)} disabled={isGenerating}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm font-bold py-3 rounded-lg flex items-center justify-center gap-2 shadow-lg shadow-blue-900/20 transition-all">
                {isGenerating && !hasLayout ? <span className="animate-pulse">Analyzing...</span> : <><MapIcon size={16}/> Generate Layout</>}
            </button>
            <button onClick={onRefine} disabled={isGenerating || !hasLayout}
                className="w-full border-2 border-purple-500/50 text-purple-300 hover:bg-purple-900/20 disabled:opacity-30 text-sm font-bold py-3 rounded-lg flex items-center justify-center gap-2 transition-all">
                 <Sparkles size={16}/> Smart Refinement
            </button>
            <button onClick={onDownload} disabled={!hasLayout}
                className="w-full border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-30 text-sm font-medium py-2 rounded-lg flex items-center justify-center gap-2 transition-all">
                <Download size={16}/> Export Image
            </button>
        </div>
        <p className="text-[10px] text-slate-500 text-center">
          Pro Tip: Use specific project Key to avoid quota limits.
        </p>
      </div>

      <div className="h-px bg-slate-800 my-2" />

      <div className="flex-1 min-h-0 flex flex-col">
        <div className="flex justify-between mb-2 text-sm text-slate-300">
            <h3 className="font-semibold">System Diagnostics</h3>
            <span className={`px-2 py-0.5 rounded-full font-bold text-[10px] uppercase ${violations.length ? 'bg-red-900/50 text-red-300' : 'bg-green-900/50 text-green-300'}`}>
                {violations.length ? `${violations.length} Collisions` : 'Valid'}
            </span>
        </div>
        
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
            {logs.length > 0 && (
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-[10px] font-mono text-slate-500 leading-relaxed">
                    {logs.map((l, i) => <div key={i} className="mb-1 border-b border-slate-900 pb-1 last:border-0">> {l}</div>)}
                </div>
            )}
            
            {violations.map((v, i) => (
                <div key={i} className="bg-red-950/20 border-l-2 border-red-500 p-2 text-[11px] text-red-300 rounded-r shadow-sm">
                    <span className="font-bold opacity-75 mr-1">{v.type.toUpperCase()}:</span> {v.message}
                </div>
            ))}

            {hasLayout && logs.length === 0 && violations.length === 0 && (
              <div className="text-center py-8 text-slate-600 italic text-xs">
                No active logs or violations.
              </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default LayoutControl;