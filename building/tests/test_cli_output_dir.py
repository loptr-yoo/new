from __future__ import annotations

from pathlib import Path

from building.cli.full_pipeline import PROJECT_ROOT, _resolve_output_dir


def test_output_dir_legacy_out_is_rebased_to_building_out() -> None:
    assert _resolve_output_dir("out/test_gemini_south") == Path(PROJECT_ROOT) / "building" / "out" / "test_gemini_south"


def test_output_dir_legacy_building_outputs_is_rebased_to_building_out() -> None:
    assert _resolve_output_dir("building/outputs/test_gemini_east") == Path(PROJECT_ROOT) / "building" / "out" / "test_gemini_east"


def test_output_dir_building_out_is_kept() -> None:
    assert _resolve_output_dir("building/out/custom") == Path("building/out/custom")
