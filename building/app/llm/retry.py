from __future__ import annotations

from typing import Callable, List, Optional

from .provider import ChatMessage, LLMClient, LLMConfig, NetworkAPIError, LLMOutputFormatError
from .transcript import append_llm_response
from ..utils.misc import compress_prompt, sleep


MAX_RETRIES = 3


async def call_llm_with_retry(
    client: LLMClient,
    messages: List[ChatMessage],
    config: LLMConfig,
    on_log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    调用 LLM，并在遇到格式解析等逻辑错误时重试。
    网络层错误 (NetworkAPIError) 由 safe_fetch 处理并在抛出后直接中断重试。
    """
    last_error: Optional[Exception] = None
    compressed = [ChatMessage(role=m.role, content=compress_prompt(m.content)) for m in messages]
    for i in range(MAX_RETRIES):
        try:
            if i > 0:
                if on_log:
                    on_log(f"⏳ [{client.providerName}] 请求重试 {i}/{MAX_RETRIES}...")
                await sleep(1000 * i)
            text = await client.chat(compressed, config)
            append_llm_response(
                provider=client.providerName,
                config=config,
                attempt=i + 1,
                messages=compressed,
                response_text=text,
            )
            return text
        except NetworkAPIError as e:
            # Network errors are already retried in safe_fetch, so fail fast here
            if on_log:
                on_log(f"[{client.providerName}] 网络请求失败，终止重试。错误: {str(e)}")
            raise
        except (LLMOutputFormatError, Exception) as e:
            last_error = e
            if on_log:
                on_log(f"⚠️ [{client.providerName}] 调用失败: {str(e)}")
    if last_error:
        raise last_error
    raise RuntimeError("Max retries exceeded")
