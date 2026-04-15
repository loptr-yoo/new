from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import httpx
import asyncio
import logging
import os
import random


# Default bases
DEFAULT_OPENAI_BASE = "https://api.openai-proxy.org/v1"
DEFAULT_GEMINI_BASE = "https://api.openai-proxy.org/google"
DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"

# Allow override by env for third-party relay
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE).rstrip("/")
GEMINI_BASE = os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE).rstrip("/")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE).rstrip("/")


ChatRole = Literal["system", "user", "assistant"]


class NetworkAPIError(Exception):
    """Exception raised for network or HTTP-level errors."""
    pass

class LLMOutputFormatError(Exception):
    """Exception raised when the LLM response format is unexpected or blocked."""
    pass

@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True)
class LLMConfig:
    apiKey: str
    model: str
    temperature: Optional[float] = None
    maxTokens: Optional[int] = None


class LLMClient:
    providerName: str

    async def chat(self, messages: List[ChatMessage], config: LLMConfig) -> str:
        raise NotImplementedError


async def safe_fetch(url: str, payload: Any, headers: Dict[str, str], provider_tag: str) -> Any:
    logger = logging.getLogger(f"llm.{provider_tag}")
    max_attempts = 5
    base_backoff_s = 1.0
    backoff_cap_s = 20.0
    last_exc: Optional[Exception] = None
    
    safe_headers = dict(headers)
    if "Authorization" in safe_headers:
        val = safe_headers["Authorization"]
        if len(val) > 15:
            safe_headers["Authorization"] = val[:10] + "****" + val[-4:]
        else:
            safe_headers["Authorization"] = "****"

    # ==========================================================
    # 动态配置网络代理策略
    # ==========================================================
    LOCAL_PROXY = "http://127.0.0.1:10808" 
    proxy_url = None
    trust_env = True

    if provider_tag == "gemini":
        proxy_url = LOCAL_PROXY
    elif provider_tag == "deepseek":
        proxy_url = None
        trust_env = False
    elif provider_tag == "openai":
        proxy_url = LOCAL_PROXY
    # ==========================================================

    for attempt in range(max_attempts):
        try:
            # 将策略传入 AsyncClient
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=20.0, read=120.0),
                follow_redirects=True,
                trust_env=trust_env,  # 动态：决定是否读取系统环境变量
                proxy=proxy_url,      # 动态：决定是否使用本地代理
            ) as client:
                res = await client.post(url, json=payload, headers=headers)
                
                if 400 <= res.status_code < 500:
                    if res.status_code == 429:
                        retry_after = res.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            wait_time = min(int(retry_after), int(backoff_cap_s))
                            logger.warning(f"[{provider_tag}] 429 Rate Limit. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time + random.uniform(0.0, 0.3))
                            continue
                        else:
                            raise NetworkAPIError(f"[{provider_tag}] Rate Limit Exceeded (429): {res.text}")
                    # Fast fail on 400, 401, 403, 404
                    raise NetworkAPIError(f"[{provider_tag}] Client Error ({res.status_code}): {res.text}")
                    
                if res.status_code >= 500:
                    raise NetworkAPIError(f"[{provider_tag}] Server Error ({res.status_code}): {res.text}")
                    
                return res.json()
        except NetworkAPIError as e:
            if "Client Error" in str(e):
                raise  # fast fail for 4xx errors except 429
            last_exc = e
            logger.warning(f"[{provider_tag}] request failed on attempt {attempt+1}/{max_attempts}: {e}")
            if attempt < max_attempts - 1:
                delay = min(backoff_cap_s, base_backoff_s * (2 ** attempt))
                await asyncio.sleep(random.uniform(0.0, delay))
        except Exception as e:
            last_exc = e
            logger.warning(f"[{provider_tag}] network exception on attempt {attempt+1}/{max_attempts}: {e}")
            if attempt < max_attempts - 1:
                delay = min(backoff_cap_s, base_backoff_s * (2 ** attempt))
                await asyncio.sleep(random.uniform(0.0, delay))
                
    raise NetworkAPIError(f"[{provider_tag}] Max retries exceeded. Last error: {str(last_exc)}") from last_exc

class OpenAICompatibleClient(LLMClient):
    def __init__(self, provider_name: str, base_url: str):
        self.providerName = provider_name
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: List[ChatMessage], config: LLMConfig) -> str:
        url = f"{self._base_url}/chat/completions"
        is_deepseek = "deepseek" in self.providerName.lower()
        is_openai = "openai" in self.providerName.lower()
        body: Dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": config.temperature if config.temperature is not None else 0.7,
            "max_tokens": config.maxTokens if config.maxTokens is not None else 4000,
            "stream": False,
        }
        if is_deepseek or is_openai:
            body["response_format"] = {"type": "json_object"}
        data = await safe_fetch(
            url,
            body,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.apiKey}",
            },
            self.providerName,
        )
        try:
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("Missing 'choices' array")
            content = choices[0].get("message", {}).get("content")
            if content is None:
                raise ValueError("Missing 'content' in message")
            return content
        except Exception as e:
            raise LLMOutputFormatError(f"Failed to parse OpenAI format: {e}. Raw response: {data}")


class GeminiClient(LLMClient):
    def __init__(self):
        self.providerName = "gemini"
        self._base_url = GEMINI_BASE

    async def chat(self, messages: List[ChatMessage], config: LLMConfig) -> str:
        url = f"{self._base_url}/v1beta/models/{config.model}:generateContent?key={config.apiKey}"
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]
        system_msg = next((m for m in messages if m.role == "system"), None)
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": config.temperature if config.temperature is not None else 0.7,
                "maxOutputTokens": config.maxTokens if config.maxTokens is not None else 4000,
                "responseMimeType": "application/json",
            },
        }
        if system_msg is not None:
            body["systemInstruction"] = {"parts": [{"text": system_msg.content}]}
        data = await safe_fetch(
            url,
            body,
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            "gemini",
        )
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                # Check for safety filter
                prompt_feedback = data.get("promptFeedback", {})
                if prompt_feedback.get("blockReason"):
                    raise ValueError(f"Prompt blocked by safety filter: {prompt_feedback}")
                raise ValueError("Missing 'candidates' array")
            
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            if finish_reason == "SAFETY":
                raise ValueError("Content blocked by safety filter. finishReason: SAFETY")
            
            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("Missing 'parts' in candidate content")
                
            text = parts[0].get("text")
            if text is None:
                raise ValueError("Missing 'text' in parts")
                
            return text
        except Exception as e:
            raise LLMOutputFormatError(f"Failed to parse Gemini format: {e}. Raw response: {data}")


def create_llm_client(provider: Literal["gemini", "deepseek", "openai"]) -> LLMClient:
    if provider == "gemini":
        return GeminiClient()
    if provider == "openai":
        return OpenAICompatibleClient("openai", OPENAI_BASE)
    if provider == "deepseek":
        return OpenAICompatibleClient("deepseek", DEEPSEEK_BASE)
    raise ValueError(f"Unsupported provider: {provider}")
