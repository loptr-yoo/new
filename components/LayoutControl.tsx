import React, { useState } from 'react';
import { Layers, Map as MapIcon, Sparkles, Download } from 'lucide-react';
import { useStore } from '../store';

interface Props {
  onGenerate: (p: string) => void;
  onRefine: () => void;
  onDownload: () => void;
}

const LayoutControl: React.FC<Props> = ({ onGenerate, onRefine, onDownload }) => {
  const { isGenerating, violations, layout, logs } = useStore();
  const [prompt, setPrompt] = useState("Underground parking, rectangular, 2 main lanes, central islands.");
  const hasLayout = !!layout;

  return (
    <div className="w-full md:w-80 flex-shrink-0 bg-slate-900 border-l border-slate-800 p-4 overflow-y-auto flex flex-col gap-6">
      
      <div>
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-400" /> Control Panel
        </h2>
      </div>

      <div className="space-y-3">
        <label className="text-sm font-semibold text-slate-300">Prompt</label>
        <textarea 
          className="w-full bg-slate-800 border border-slate-700 rounded p-3 text-sm text-slate-200 h-24 resize-none"
          value={prompt} onChange={(e) => setPrompt(e.target.value)}
        />
        
        <div className="grid grid-cols-1 gap-3">
            <button onClick={() => onGenerate(prompt)} disabled={isGenerating}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm font-bold py-3 rounded-lg flex items-center justify-center gap-2">
                {isGenerating && !hasLayout ? <span className="animate-pulse">Generating...</span> : <><MapIcon size={16}/> Generate</>}
            </button>
            <button onClick={onRefine} disabled={isGenerating || !hasLayout}
                className="w-full border-2 border-purple-500/50 text-purple-300 hover:bg-purple-900/20 disabled:opacity-30 text-sm font-bold py-3 rounded-lg flex items-center justify-center gap-2">
                 <Sparkles size={16}/> Refine
            </button>
            <button onClick={onDownload} disabled={!hasLayout}
                className="w-full border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-30 text-sm font-medium py-2 rounded-lg flex items-center justify-center gap-2">
                <Download size={16}/> JPG
            </button>
        </div>
      </div>

      <div className="h-px bg-slate-800 my-2" />

      <div className="flex-1 min-h-0 flex flex-col">
        <div className="flex justify-between mb-2 text-sm text-slate-300">
            <h3>Status</h3>
            <span className={`px-2 rounded-full font-bold text-xs ${violations.length ? 'bg-red-900 text-red-300' : 'bg-green-900 text-green-300'}`}>
                {violations.length ? `${violations.length} Issues` : 'Ready'}
            </span>
        </div>
        
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
            {logs.length > 0 && (
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-xs font-mono text-slate-400">
                    {logs.map((l, i) => <div key={i}>> {l}</div>)}
                </div>
            )}
            
            {violations.map((v, i) => (
                <div key={i} className="bg-red-950/30 border-l-2 border-red-500 p-2 text-xs text-red-200">
                    <b>{v.type}</b>: {v.message}
                </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default LayoutControl;