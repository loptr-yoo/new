from __future__ import annotations

import logging
import os
from logging import StreamHandler, FileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import augment, generate
from .settings import settings


def _setup_logging() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    fmt = "%(asctime)s %(levelname)s [%(name)s] [%(threadName)s] %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    has_stream = any(isinstance(h, StreamHandler) for h in root.handlers)
    if not has_stream:
        sh = StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(fmt))
        root.addHandler(sh)
    has_file = any(isinstance(h, FileHandler) for h in root.handlers)
    if not has_file:
        fh = FileHandler("backend.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


_setup_logging()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/api")
app.include_router(augment.router, prefix="/api")
