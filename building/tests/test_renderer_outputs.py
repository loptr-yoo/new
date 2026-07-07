from __future__ import annotations

from pathlib import Path

from building.app.rendering.local_renderer import render_floor_layout


def test_renderer_writes_per_floor_seg_artifact(tmp_path: Path) -> None:
    layout = {
        "width": 8.0,
        "height": 6.0,
        "elements": [
            {"id": "slab", "type": "floor_slab", "polygon": [[0, 0], [8, 0], [8, 6], [0, 6]]},
            {"id": "room", "type": "bedroom", "polygon": [[1, 1], [4, 1], [4, 4], [1, 4]]},
        ],
    }
    out = tmp_path / "render_F1_seg.png"
    render_floor_layout(layout, out, "seg")
    assert out.exists()
    assert out.stat().st_size > 100
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_renderer_has_no_frontend_contract() -> None:
    src = Path("building/app/rendering/local_renderer.py").read_text(encoding="utf-8").lower()
    assert "scripts.local_renderer" not in src
    assert "backend.core" not in src
    assert "playwright" not in src
    assert "puppeteer" not in src
    assert "selenium" not in src
    assert "node_modules" not in src
    assert "npm" not in src
