from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import httpx
import asyncio
import logging
import os


# Default bases
DEFAULT_OPENAI_BASE = "https://api.openai-proxy.org/v1"
DEFAULT_GEMINI_BASE = "https://api.openai-proxy.org/google"
DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"

# Allow override by env for third-party relay
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE).rstrip("/")
GEMINI_BASE = os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE).rstrip("/")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE).rstrip("/")


ChatRole = Literal["system", "user", "assistant"]


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
    backoff = [0, 1.5, 3.0]
    last_exc: Optional[Exception] = None
    for i, delay in enumerate(backoff):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0, read=30.0),
                follow_redirects=True,
                trust_env=True,
            ) as client:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code < 200 or res.status_code >= 300:
                    raise RuntimeError(f"[{provider_tag}] API Error ({res.status_code}): {res.text}")
                return res.json()
        except Exception as e:
            last_exc = e
            logger.warning(f"[{provider_tag}] request failed on attempt {i+1}: {e}")
            if i < len(backoff) - 1:
                await asyncio.sleep(delay)
    raise RuntimeError(f"[{provider_tag}] Network/Request Error: {str(last_exc) if last_exc else 'unknown'}") from last_exc


class OpenAICompatibleClient(LLMClient):
    def __init__(self, provider_name: str, base_url: str):
        self.providerName = provider_name
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: List[ChatMessage], config: LLMConfig) -> str:
        url = f"{self._base_url}/chat/completions"
        is_deepseek = "deepseek" in self.providerName.lower()
        response_format = {"type": "json_object"} if is_deepseek else {"type": "json_object"}
        body = {
            "model": config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": config.temperature if config.temperature is not None else 0.7,
            "max_tokens": config.maxTokens if config.maxTokens is not None else 4000,
            "response_format": response_format,
            "stream": False,
        }
        data = await safe_fetch(
            url,
            body,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.apiKey}",
            },
            self.providerName,
        )
        return ((data or {}).get("choices") or [{}])[0].get("message", {}).get("content") or ""


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
        return (((data or {}).get("candidates") or [{}])[0].get("content") or {}).get("parts", [{}])[0].get("text") or ""


def create_llm_client(provider: Literal["gemini", "deepseek", "openai"]) -> LLMClient:
    if provider == "gemini":
        return GeminiClient()
    if provider == "openai":
        return OpenAICompatibleClient("openai", OPENAI_BASE)
    if provider == "deepseek":
        return OpenAICompatibleClient("deepseek", DEEPSEEK_BASE)
    raise ValueError(f"Unsupported provider: {provider}")
