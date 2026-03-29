from __future__ import annotations

from typing import Callable, List, Optional

from .provider import ChatMessage, LLMClient, LLMConfig
from ..utils.misc import compress_prompt, sleep


MAX_RETRIES = 3


async def call_llm_with_retry(
    client: LLMClient,
    messages: List[ChatMessage],
    config: LLMConfig,
    on_log: Optional[Callable[[str], None]] = None,
) -> str:
    last_error: Optional[Exception] = None
    compressed = [ChatMessage(role=m.role, content=compress_prompt(m.content)) for m in messages]
    for i in range(MAX_RETRIES):
        try:
            if i > 0:
                if on_log:
                    on_log(f"⏳ [{client.providerName}] 请求重试 {i}/{MAX_RETRIES}...")
                await sleep(1000 * i)
            return await client.chat(compressed, config)
        except Exception as e:
            last_error = e
            if on_log:
                on_log(f"⚠️ [{client.providerName}] 调用失败: {str(e)}")
    if last_error:
        raise last_error
    raise RuntimeError("Max retries exceeded")

