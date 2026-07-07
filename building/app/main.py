from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routers import deprecated, generate
from .logger import setup_logging
from .settings import settings


def _setup_logging() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    setup_logging(force=True)


_setup_logging()
app = FastAPI(title="Backend-Only Multi-Floor Building Generator")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api")
app.include_router(deprecated.router, prefix="/api")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "[EXCEPTION] Unhandled ASGI exception | Method=%s | Path=%s | Client=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "-",
        exc_info=True,
    )
    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
