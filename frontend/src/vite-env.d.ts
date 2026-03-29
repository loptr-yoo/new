/// <reference types="vite/client" />

declare global {
  interface Window {
    aistudio: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }

  namespace NodeJS {
    interface ProcessEnv {
      VITE_GEMINI_API_KEY?: string;
    }
  }
}

export {};
