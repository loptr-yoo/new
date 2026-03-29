/**
 * AI 模型配置文件
 * 支持多个 AI 提供商和模型
 */

export type AIProvider = 'gemini' | 'deepseek' | 'openai';

export interface AIModel {
  id: string;
  name: string;
  provider: AIProvider;
  description: string;
  maxTokens: number;
  costPer1kTokens: {
    input: number;
    output: number;
  };
}

export interface AIConfig {
  provider: AIProvider;
  model: string;
  apiKey: string;
}

// 支持的所有模型列表
export const AVAILABLE_MODELS: AIModel[] = [
  {
    id: 'gemini-2.5-pro',
    name: 'Gemini 2.5 Pro',
    provider: 'gemini',
    description: 'Google 最新的多模态模型，适合复杂推理任务',
    maxTokens: 100000,
    costPer1kTokens: {
      input: 0.0075,
      output: 0.03,
    },
  },
  {
    id: 'gemini-3-pro-preview',
    name: 'Gemini 3 Pro Preview',
    provider: 'gemini',
    description: 'Google 最新预览版模型',
    maxTokens: 100000,
    costPer1kTokens: {
      input: 0.0075,
      output: 0.03,
    },
  },
  {
    id: 'deepseek-chat',
    name: 'DeepSeek Chat',
    provider: 'deepseek',
    description: '开源模型 DeepSeek，成本低廉',
    maxTokens: 4096,
    costPer1kTokens: {
      input: 0.0014,
      output: 0.0028,
    },
  },
  {
    id: 'deepseek-coder',
    name: 'DeepSeek Coder',
    provider: 'deepseek',
    description: '专用编程任务模型',
    maxTokens: 4096,
    costPer1kTokens: {
      input: 0.0014,
      output: 0.0028,
    },
  },
  // GPT 模型
  {
    id: 'gpt-4',
    name: 'GPT-4',
    provider: 'openai',
    description: 'OpenAI 最强的模型，适合复杂任务',
    maxTokens: 8192,
    costPer1kTokens: {
      input: 0.03,
      output: 0.06,
    },
  },
  {
    id: 'gpt-4-turbo',
    name: 'GPT-4 Turbo',
    provider: 'openai',
    description: '高性能 GPT-4 变种，处理速度更快',
    maxTokens: 128000,
    costPer1kTokens: {
      input: 0.01,
      output: 0.03,
    },
  },
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    provider: 'openai',
    description: 'GPT-4 Omni，最新的多模态模型',
    maxTokens: 128000,
    costPer1kTokens: {
      input: 0.005,
      output: 0.015,
    },
  },
  {
    id: 'gpt-3.5-turbo',
    name: 'GPT-3.5 Turbo',
    provider: 'openai',
    description: '快速且成本有效的模型',
    maxTokens: 4096,
    costPer1kTokens: {
      input: 0.0005,
      output: 0.0015,
    },
  },
];

// 按提供商分组的模型
export const MODEL_BY_PROVIDER = AVAILABLE_MODELS.reduce(
  (acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = [];
    }
    acc[model.provider].push(model);
    return acc;
  },
  {} as Record<AIProvider, AIModel[]>
);

// 获取默认模型
export const getDefaultModel = (provider: AIProvider): AIModel => {
  const models = MODEL_BY_PROVIDER[provider];
  return models[0];
};

// 根据 ID 获取模型信息
export const getModelById = (modelId: string): AIModel | undefined => {
  return AVAILABLE_MODELS.find((m) => m.id === modelId);
};

// 环境变量配置键
export const ENV_KEYS = {
  GEMINI_API_KEY: 'VITE_GEMINI_API_KEY',
  DEEPSEEK_API_KEY: 'VITE_DEEPSEEK_API_KEY',
  SELECTED_MODEL: 'VITE_SELECTED_MODEL', // 默认选择的模型
} as const;

// 获取 API Key (根据提供商)
export const getApiKeyEnvKey = (provider: AIProvider): string => {
  switch (provider) {
    case 'gemini':
      return ENV_KEYS.GEMINI_API_KEY;
    case 'deepseek':
      return ENV_KEYS.DEEPSEEK_API_KEY;
    case 'openai':
      return 'VITE_OPENAI_API_KEY';
    default:
      throw new Error(`Unknown provider: ${provider}`);
  }
};
