import { useRef, useCallback, useEffect } from 'react';

interface ErrorResult {
  message: string;
  isUserCanceled: boolean;
}

export function useLLMStream() {
  const abortControllerRef = useRef<AbortController | null>(null);

  const translateError = (err: any): ErrorResult => {
    const msg = err?.message || String(err);
    if (msg.includes('AbortError') || err.name === 'AbortError' || msg.includes('Idle Timeout')) {
      return { message: '请求已取消或超时。', isUserCanceled: true };
    }
    if (msg.includes('401') || msg.toLowerCase().includes('auth')) {
      return { message: 'API 鉴权失败或已欠费，请检查密钥配置。', isUserCanceled: false };
    }
    if (msg.includes('429') || msg.toLowerCase().includes('rate limit') || msg.toLowerCase().includes('too many')) {
      return { message: '当前请求人数过多，触发限流，请稍后重试。', isUserCanceled: false };
    }
    if (msg.includes('400')) {
      if (msg.toLowerCase().includes('length') || msg.toLowerCase().includes('exceed') || msg.toLowerCase().includes('context')) {
        return { message: '请求内容太长，超出了模型允许的上下文限制，请精简后再试。', isUserCanceled: false };
      }
      return { message: '请求参数错误或被安全拦截。', isUserCanceled: false };
    }
    return { message: `网络断开或服务器异常，请重试。(${msg})`, isUserCanceled: false };
  };

  const wrapStreamCall = useCallback(
    async <T>(
      apiCall: (signal: AbortSignal) => Promise<T>
    ): Promise<T> => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort('New request started');
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const result = await apiCall(controller.signal);
        return result;
      } catch (err: any) {
        const translated = translateError(err);
        throw translated;
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      }
    },
    []
  );

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancel();
    };
  }, [cancel]);

  return { wrapStreamCall, cancel };
}
