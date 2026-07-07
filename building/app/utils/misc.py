from __future__ import annotations

import asyncio
import re


def compress_prompt(prompt: str) -> str:
    if not prompt:
        return prompt
    s = str(prompt)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"\n+", "\n", s)
    s = re.sub(r'":\s+', '":', s)
    s = re.sub(r'",\s+"', '","', s)
    s = re.sub(r"\}[\s\n]+,", "},", s)
    s = re.sub(r"\[\s+\{", "[{", s)
    s = re.sub(r"\}\s+\]", "}]", s)
    s = s.replace("```json\n", "").replace("```\n", "")
    return s.strip()


async def sleep(ms: int) -> None:
    await asyncio.sleep(max(0, ms) / 1000)

