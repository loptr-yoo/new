from __future__ import annotations

from pathlib import Path

from building.cli.full_pipeline import PROJECT_ROOT, _resolve_output_dir


def test_explicit_out_dir_is_respected() -> None:
    assert _resolve_output_dir("out/test_gemini_south") == Path("out/test_gemini_south")


def test_explicit_building_outputs_dir_is_respected() -> None:
    assert _resolve_output_dir("building/outputs/test_gemini_east") == Path("building/outputs/test_gemini_east")


def test_output_dir_building_out_is_kept() -> None:
    assert _resolve_output_dir("building/out/custom") == Path("building/out/custom")


def test_default_output_dir_uses_building_out() -> None:
    assert _resolve_output_dir(None).parent == Path(PROJECT_ROOT) / "building" / "out"
