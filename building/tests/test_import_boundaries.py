from __future__ import annotations

from pathlib import Path

ACTIVE_ROOTS = [Path("building/app"), Path("building/cli")]


def _read_py(root: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in root.rglob("*.py"))


def test_active_code_does_not_import_backend_or_scripts() -> None:
    text = "\n".join(_read_py(root) for root in ACTIVE_ROOTS)
    for token in ("from " + "backend", "import " + "backend", "from " + "scripts", "import " + "scripts"):
        assert token not in text


def test_active_code_does_not_call_parking_generation() -> None:
    text = "\n".join(_read_py(root) for root in ACTIVE_ROOTS)
    for token in ("parking" + "_flow", "augment" + ".router"):
        assert token not in text


def test_active_code_has_no_frontend_runtime_dependency() -> None:
    text = "\n".join(_read_py(root).lower() for root in ACTIVE_ROOTS)
    for token in ("node" + "_modules", "npm" + " run", "play" + "wright", "pupp" + "eteer", "selen" + "ium"):
        assert token not in text
