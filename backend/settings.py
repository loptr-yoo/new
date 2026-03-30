from __future__ import annotations

import os
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


_here = Path(__file__).resolve().parent
_root = _here.parent
for _p in (_here / ".env.local", _here / ".env", _root / ".env.local", _root / ".env"):
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=False)

def _get_key(primary: str, fallback: str) -> Optional[str]:
    v = os.getenv(primary)
    if v:
        return v
    return os.getenv(fallback)


class Settings(BaseModel):
    cors_allow_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
        ]
    )

    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: _get_key("GEMINI_API_KEY", "VITE_GEMINI_API_KEY")
    )
    deepseek_api_key: Optional[str] = Field(
        default_factory=lambda: _get_key("DEEPSEEK_API_KEY", "VITE_DEEPSEEK_API_KEY")
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: _get_key("OPENAI_API_KEY", "VITE_OPENAI_API_KEY")
    )


settings = Settings()
