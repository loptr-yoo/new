from __future__ import annotations
import asyncio

import json
from typing import Any, AsyncIterator, Dict, Optional


def _sse_data(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def sse_stream(queue: "asyncio.Queue[Optional[Dict[str, Any]]]", ping_seconds: float = 10.0) -> AsyncIterator[str]:
    import asyncio

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=ping_seconds)
        except asyncio.TimeoutError:
            yield _sse_data({"status": "ping"})
            continue
        if item is None:
            break
        yield _sse_data(item)
