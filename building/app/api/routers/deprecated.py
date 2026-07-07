from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


def _gone(use: str) -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={
            "error": "This endpoint has been removed from the active generation path.",
            "use": use,
        },
    )


@router.post("/generate")
async def deprecated_generate() -> JSONResponse:
    return _gone("/api/building/generate")


@router.post("/generate/stream")
async def deprecated_generate_stream() -> JSONResponse:
    return _gone("/api/building/generate/stream")
