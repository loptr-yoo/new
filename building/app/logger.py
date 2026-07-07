from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
import threading
import traceback
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Iterator, Optional


_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("log_session_id", default="-")
_FLOOR_ID: contextvars.ContextVar[str] = contextvars.ContextVar("log_floor_id", default="-")
_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar("log_stage", default="-")
_TOPOLOGY_MODE: contextvars.ContextVar[str] = contextvars.ContextVar("log_topology_mode", default="-")

_SETUP_LOCK = threading.Lock()
_UNHANDLED_LOGGER_NAME = "building.unhandled"


class ContextFilter(logging.Filter):
    """Inject async-safe contextvars into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = _SESSION_ID.get()
        record.floor_id = _FLOOR_ID.get()
        record.stage = _STAGE.get()
        record.topology_mode = _TOPOLOGY_MODE.get()
        return True


class _ArchiveTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename: Path, archive_dir: Path, *args: Any, **kwargs: Any) -> None:
        self._archive_dir = archive_dir
        super().__init__(str(filename), *args, **kwargs)
        self.namer = self._archive_name

    def _archive_name(self, default_name: str) -> str:
        return str(self._archive_dir / Path(default_name).name)


class _ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")


class _DebugFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] "
            "[%(funcName)s:%(lineno)d] [thread=%(threadName)s] "
            "[session=%(session_id)s floor=%(floor_id)s stage=%(stage)s topology=%(topology_mode)s] "
            "%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_excepthook() -> None:
    def _hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        logger = logging.getLogger(_UNHANDLED_LOGGER_NAME)
        logger.critical("[EXCEPTION] Unhandled exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


def setup_logging(
    log_dir: str | os.PathLike[str] = "logs",
    *,
    session_id: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Configure dual-channel logging once.

    Console receives INFO+, latest.log receives INFO+, and latest-debug.log
    receives DEBUG+ with contextvars and source location.
    """

    with _SETUP_LOCK:
        base_dir = Path(log_dir)
        if not base_dir.is_absolute():
            base_dir = _repo_root() / base_dir
        archive_dir = base_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        if getattr(root, "_city_ge_logging_configured", False) and not force:
            if session_id is not None:
                _SESSION_ID.set(str(session_id))
            return base_dir

        if force:
            for handler in list(root.handlers):
                root.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

        root.setLevel(logging.DEBUG)
        context_filter = ContextFilter()

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(_ConsoleFormatter())
        console.addFilter(context_filter)

        basic_file = _ArchiveTimedRotatingFileHandler(
            base_dir / "latest.log",
            archive_dir,
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=False,
        )
        basic_file.setLevel(logging.INFO)
        basic_file.setFormatter(_DebugFormatter())
        basic_file.addFilter(context_filter)

        debug_file = _ArchiveTimedRotatingFileHandler(
            base_dir / "latest-debug.log",
            archive_dir,
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=False,
        )
        debug_file.setLevel(logging.DEBUG)
        debug_file.setFormatter(_DebugFormatter())
        debug_file.addFilter(context_filter)

        root.addHandler(console)
        root.addHandler(basic_file)
        root.addHandler(debug_file)
        root._city_ge_logging_configured = True  # type: ignore[attr-defined]

        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).setLevel(logging.INFO)

        if session_id is not None:
            _SESSION_ID.set(str(session_id))

        _install_excepthook()
        return base_dir


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextlib.contextmanager
def log_context(
    *,
    session_id: Optional[str] = None,
    floor_id: Optional[str] = None,
    stage: Optional[str] = None,
    topology_mode: Optional[str] = None,
) -> Iterator[None]:
    tokens: list[tuple[contextvars.ContextVar[str], contextvars.Token[str]]] = []
    try:
        if session_id is not None:
            tokens.append((_SESSION_ID, _SESSION_ID.set(str(session_id))))
        if floor_id is not None:
            tokens.append((_FLOOR_ID, _FLOOR_ID.set(str(floor_id))))
        if stage is not None:
            tokens.append((_STAGE, _STAGE.set(str(stage))))
        if topology_mode is not None:
            tokens.append((_TOPOLOGY_MODE, _TOPOLOGY_MODE.set(str(topology_mode))))
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_log_context() -> dict[str, str]:
    return {
        "session_id": _SESSION_ID.get(),
        "floor_id": _FLOOR_ID.get(),
        "stage": _STAGE.get(),
        "topology_mode": _TOPOLOGY_MODE.get(),
    }


def log_multiline_debug(
    logger: logging.Logger,
    tag: str,
    title: str,
    payload: Any,
    boundary_name: str,
) -> None:
    text = payload if isinstance(payload, str) else repr(payload)
    boundary = str(boundary_name or "PAYLOAD").upper().strip().replace(" ", "_")
    logger.debug(
        "%s %s\n========== %s START ==========\n%s\n========== %s END ==========",
        tag,
        title,
        boundary,
        text,
        boundary,
    )


def log_exception(
    logger: logging.Logger,
    stage: str,
    exc: BaseException,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    logger.error(
        "[EXCEPTION] Stage=%s Type=%s Message=%s Metadata=%s\n%s",
        stage,
        type(exc).__name__,
        str(exc),
        metadata or {},
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        exc_info=True,
    )


