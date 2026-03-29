import React, { useState } from 'react';
import { AVAILABLE_MODELS, AIProvider, getModelById, MODEL_BY_PROVIDER } from '../utils/aiConfig';
import { useStore } from '../store';
import { ChevronDown } from 'lucide-react';

interface ModelSelectorProps {
  onClose?: () => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ onClose }) => {
  const { selectedProvider, selectedModel, setSelectedProvider, setSelectedModel } = useStore();
  const [isOpen, setIsOpen] = useState(false);
  const [selectedTab, setSelectedTab] = useState<AIProvider>(selectedProvider);

  const currentModel = getModelById(selectedModel);
  const modelsInTab = MODEL_BY_PROVIDER[selectedTab] || [];

  const handleSelectModel = (modelId: string) => {
    const model = getModelById(modelId);
    if (model) {
      setSelectedProvider(model.provider);
      setSelectedModel(modelId);
      setIsOpen(false);
      onClose?.();
    }
  };

  return (
    <div className="w-full">
      <div className="text-xs font-semibold text-slate-400 mb-2 uppercase">AI 模型选择</div>

      {/* 选择按钮 */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2.5 rounded-lg border border-slate-700 bg-slate-900/50 hover:bg-slate-900 text-left text-sm text-slate-200 flex items-center justify-between transition-colors"
      >
        <div className="flex items-center gap-2 flex-1">
          <div className="w-2 h-2 rounded-full bg-blue-500"></div>
          <div className="flex-1">
            <div className="font-medium">{currentModel?.name}</div>
            <div className="text-xs text-slate-500">{currentModel?.provider}</div>
          </div>
        </div>
        <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* 下拉菜单 */}
      {isOpen && (
        <div className="absolute mt-1 w-80 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 overflow-hidden">
          {/* 标签页 */}
          <div className="flex border-b border-slate-700">
            <button
              onClick={() => setSelectedTab('gemini')}
              className={`flex-1 px-3 py-2 text-xs font-semibold transition-colors ${
                selectedTab === 'gemini'
                  ? 'bg-blue-900/50 text-blue-300 border-b-2 border-blue-500'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              🔵 Gemini
            </button>
            <button
              onClick={() => setSelectedTab('deepseek')}
              className={`flex-1 px-3 py-2 text-xs font-semibold transition-colors ${
                selectedTab === 'deepseek'
                  ? 'bg-purple-900/50 text-purple-300 border-b-2 border-purple-500'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              🟣 DeepSeek
            </button>
          </div>

          {/* 模型列表 */}
          <div className="max-h-96 overflow-y-auto custom-scrollbar">
            {modelsInTab.map((model) => (
              <button
                key={model.id}
                onClick={() => handleSelectModel(model.id)}
                className={`w-full px-4 py-3 text-left border-b border-slate-800 hover:bg-slate-800/50 transition-colors ${
                  selectedModel === model.id ? 'bg-slate-800 border-l-2 border-l-blue-500' : ''
                }`}
              >
                <div className="flex items-start gap-2">
                  {selectedModel === model.id && (
                    <div className="w-4 h-4 rounded-full bg-blue-500 flex items-center justify-center mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-white"></div>
                    </div>
                  )}
                  {selectedModel !== model.id && <div className="w-4 h-4 rounded-full border border-slate-600 mt-0.5"></div>}

                  <div className="flex-1">
                    <div className="text-sm font-semibold text-slate-200">{model.name}</div>
                    <div className="text-xs text-slate-400 mt-1">{model.description}</div>
                    <div className="text-xs text-slate-500 mt-1">
                      输入: ${model.costPer1kTokens.input.toFixed(4)}/1k | 输出: ${model.costPer1kTokens.output.toFixed(4)}/1k
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>

          {/* 底部关闭按钮 */}
          <div className="border-t border-slate-700 p-2">
            <button
              onClick={() => setIsOpen(false)}
              className="w-full px-3 py-1.5 rounded text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              关闭
            </button>
          </div>
        </div>
      )}

      {/* 当前选择信息 */}
      <div className="mt-3 p-2 rounded bg-slate-900/50 border border-slate-800">
        <div className="text-xs text-slate-400">
          <div className="flex justify-between mb-1">
            <span>模型ID:</span>
            <span className="font-mono text-slate-300">{selectedModel}</span>
          </div>
          <div className="flex justify-between">
            <span>提供商:</span>
            <span className="font-mono text-slate-300 capitalize">{selectedProvider}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
