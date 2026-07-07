from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .provider import ChatMessage, LLMConfig


_LOCK = threading.Lock()


def _default_log_path() -> Path:
    return Path.cwd() / "building" / "out" / "llm_log.txt"


def get_llm_log_path() -> Path:
    raw = (os.getenv("LLM_LOG_PATH") or "").strip()
    return Path(raw) if raw else _default_log_path()


def set_llm_log_path(path: str | os.PathLike[str]) -> None:
    os.environ["LLM_LOG_PATH"] = str(Path(path))


def _redact(text: str) -> str:
    import re

    s = str(text or "")
    s = re.sub(r"sk-[A-Za-z0-9_\-]{10,}", "sk-****", s)
    s = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "AIza****", s)
    s = re.sub(r"Bearer\s+[A-Za-z0-9._\-]{10,}", "Bearer ****", s)
    return s


def _format_messages(messages: List[ChatMessage]) -> str:
    chunks: List[str] = []
    for i, msg in enumerate(messages, start=1):
        chunks.append(f"--- message {i} role={msg.role} ---\n{_redact(msg.content)}")
    return "\n".join(chunks)


def append_llm_response(
    *,
    provider: str,
    config: LLMConfig,
    attempt: int,
    messages: List[ChatMessage],
    response_text: str,
) -> None:
    path = get_llm_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    body = "\n".join([
        "=" * 88,
        f"timestamp: {stamp}",
        f"provider: {provider}",
        f"model: {config.model}",
        f"attempt: {attempt}",
        f"temperature: {config.temperature}",
        f"maxTokens: {config.maxTokens}",
        "",
        "[messages]",
        _format_messages(messages),
        "",
        "[response]",
        _redact(response_text),
        "",
    ])
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(body)

