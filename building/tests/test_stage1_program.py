from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import box

from building.app.models import BuildingAllocation, FloorAllocation, RoomAllocation
from building.app.policies import load_policy
from building.app.services.building_pipeline_service import BuildingPipelineOptions, BuildingPipelineService
from building.app.stage1 import (
    Stage1ContextMismatchError,
    Stage1ProgramInfeasibleError,
    building_allocation_from_stage1,
    core_tube_from_stage1_policy,
    run_stage1_from_allocation,
    stage2_corridor_options_from_stage1,
    validate_stage1_core_context,
    validate_stage1_corridor_context,
    write_stage1_artifacts,
)
from building.app.models import GenerateSemanticsRequest, SceneType


def _allocation() -> BuildingAllocation:
    return BuildingAllocation(
        building_name="Two Floor House",
        total_floors=2,
        overall_total_area=200.0,
        floors=[
            FloorAllocation(
                floor_number=1,
                floor_function_tag="residential",
                floor_total_area=100.0,
                core_tube_area=12.0,
                corridor_allowance_area=16.0,
                rooms=[
                    RoomAllocation(room_id="F1_living", room_name="Living", room_type="living_room", target_area=30.0),
                    RoomAllocation(room_id="F1_bed", room_name="Bedroom", room_type="bedroom", target_area=18.0),
                    RoomAllocation(room_id="F1_bath", room_name="Bath", room_type="bathroom", target_area=8.0),
                ],
            ),
            FloorAllocation(
                floor_number=2,
                floor_function_tag="residential",
                floor_total_area=100.0,
                core_tube_area=12.0,
                corridor_allowance_area=16.0,
                rooms=[
                    RoomAllocation(room_id="F2_living", room_name="Family", room_type="living_room", target_area=24.0),
                    RoomAllocation(room_id="F2_bed", room_name="Bedroom", room_type="bedroom", target_area=18.0),
                    RoomAllocation(room_id="F2_bath", room_name="Bath", room_type="bathroom", target_area=8.0),
                ],
            ),
        ],
    )


def test_residential_policy_common_rooms_have_no_fallbacks() -> None:
    policy, report = load_policy("residential")
    assert report.valid
    assert report.fallback_count == 0
    for room_type in ("living_room", "bedroom", "bathroom", "kitchen"):
        assert room_type in policy["room_rules"]


def test_stage1_preserves_program_scope_and_adapter() -> None:
    result = run_stage1_from_allocation(_allocation(), source="mock", core_placement="auto")
    assert result.core_policy.selected_placement != ""
    assert all(not r.geometry_guaranteed for r in result.feasibility_reports)
    assert result.envelope.polygon is None
    assert result.core_policy.core_bbox is None

    adapted = building_allocation_from_stage1(result)
    assert adapted.total_floors == 2
    assert [f.floor_number for f in adapted.floors] == [1, 2]
    assert [len(f.rooms) for f in adapted.floors] == [3, 3]


def test_stage1_artifacts_are_stage1_only(tmp_path: Path) -> None:
    result = run_stage1_from_allocation(_allocation(), source="mock", core_placement="auto")
    paths = write_stage1_artifacts(result, tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert "building_program.json" in names
    assert "stage1_manifest.json" in names
    assert "stage1_summary.json" not in names
    assert "layout_F1.json" not in names
    assert "render_F1_seg.png" not in names

    feasibility = json.loads(Path(paths["feasibility_F1"]).read_text(encoding="utf-8"))
    assert feasibility["schema_version"] == "stage1.v1"
    assert feasibility["feasibility_level"] == "program"
    assert feasibility["geometry_guaranteed"] is False
    manifest = json.loads((tmp_path / "stage1_manifest.json").read_text(encoding="utf-8"))
    assert "stage1_manifest.json" in manifest["generated_artifacts"]
    assert manifest["forbidden_artifacts_present"] == []


def test_stage1_manifest_reports_stale_forbidden_artifacts(tmp_path: Path) -> None:
    (tmp_path / "layout_F1.json").write_text("{}", encoding="utf-8")
    result = run_stage1_from_allocation(_allocation(), source="mock", core_placement="auto")
    write_stage1_artifacts(result, tmp_path)
    manifest = json.loads((tmp_path / "stage1_manifest.json").read_text(encoding="utf-8"))
    assert manifest["forbidden_artifacts_present"] == ["layout_F1.json"]


def _infeasible_allocation() -> BuildingAllocation:
    rooms = [
        RoomAllocation(room_id=f"r{i}", room_name=f"Room {i}", room_type="bedroom", target_area=20.0)
        for i in range(10)
    ]
    return BuildingAllocation(
        building_name="Bad Program",
        total_floors=2,
        overall_total_area=40.0,
        floors=[
            FloorAllocation(
                floor_number=1,
                floor_function_tag="residential",
                floor_total_area=20.0,
                core_tube_area=2.0,
                corridor_allowance_area=2.0,
                rooms=rooms,
            ),
            FloorAllocation(
                floor_number=2,
                floor_function_tag="residential",
                floor_total_area=20.0,
                core_tube_area=2.0,
                corridor_allowance_area=2.0,
                rooms=rooms,
            ),
        ],
    )


def test_infeasible_stage1_result_blocks_adapter() -> None:
    result = run_stage1_from_allocation(_infeasible_allocation(), source="mock")
    assert result.can_enter_geometry is False
    assert {r.failure_type for r in result.feasibility_reports} == {"min_area_exceeds_usable_area"}
    with pytest.raises(Stage1ProgramInfeasibleError):
        building_allocation_from_stage1(result)


@pytest.mark.asyncio
async def test_use_stage1_program_infeasible_does_not_call_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    result = run_stage1_from_allocation(_infeasible_allocation(), source="mock")
    service = BuildingPipelineService()

    async def fake_stage1(*args, **kwargs):
        return result

    def fail_geometry(*args, **kwargs):
        raise AssertionError("_run_geometry should not be called for infeasible Stage 1")

    monkeypatch.setattr(service, "generate_stage1", fake_stage1)
    monkeypatch.setattr(service, "_run_geometry", fail_geometry)
    pipeline_result = await service.generate(
        GenerateSemanticsRequest(scene_type=SceneType.BUILDING, user_prompt="bad", total_floors=2, model="dummy"),
        options=BuildingPipelineOptions(use_stage1_program=True),
    )
    assert pipeline_result.success is False
    assert pipeline_result.artifact_valid is False
    assert pipeline_result.failure is not None
    assert pipeline_result.failure["failure_type"] == "min_area_exceeds_usable_area"


def test_stage1_core_policy_adapts_to_core_tube_with_reference() -> None:
    result = run_stage1_from_allocation(_allocation(), source="mock", core_placement="auto")
    core, metadata = core_tube_from_stage1_policy(result, box(0.0, 0.0, 12.0, 8.0))
    assert core is not None
    assert metadata["core_source"] == "stage1"
    assert metadata["stage1_core_policy_id"] == result.core_context.stage1_core_policy_id
    validate_stage1_core_context(result, metadata)


def test_core_mismatch_and_unresolved_bbox_fail_before_stage2() -> None:
    result = run_stage1_from_allocation(_allocation(), source="mock", core_placement="auto")
    _, metadata = core_tube_from_stage1_policy(result, box(0.0, 0.0, 12.0, 8.0))
    metadata["selected_placement"] = "west"
    with pytest.raises(Stage1ContextMismatchError):
        validate_stage1_core_context(result, metadata)
    assert result.core_context.bbox is None
    with pytest.raises(Stage1ContextMismatchError):
        core_tube_from_stage1_policy(result, box(0.0, 0.0, 12.0, 8.0), require_resolved_bbox=True)


def test_stage1_corridor_policy_reaches_stage2_options_and_mismatch_fails() -> None:
    result = run_stage1_from_allocation(_allocation(), source="mock", core_placement="auto")
    options = stage2_corridor_options_from_stage1(result)
    assert options["corridor_source"] == "stage1"
    assert options["corridor_layout"] == result.corridor_policy.layout == "organic"
    validate_stage1_corridor_context(result, options)
    options["corridor_layout"] = "door_side"
    with pytest.raises(Stage1ContextMismatchError):
        validate_stage1_corridor_context(result, options)
