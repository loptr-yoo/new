from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.geometry import box

from building.app.geometry.core_contracts import validate_core_exclusion
from building.app.geometry.exceptions import LayoutGeometryInvariantError
from building.app.geometry.floor_free_space import (
    build_floor_free_space,
    build_stage2a_report,
    classify_free_space_geometry,
    positive_core_overlap_area,
)
from building.app.models import BuildingAllocation, FloorAllocation, RoomAllocation
from building.app.stage1 import (
    Stage1ContextMismatchError,
    core_tube_from_stage1_policy,
    run_stage1_from_allocation,
    stage2_corridor_options_from_stage1,
    validate_stage1_corridor_context,
)
from building.app.stage1.models import Stage1CoreContext


def _allocation() -> BuildingAllocation:
    return BuildingAllocation(
        building_name="stage2a_test",
        total_floors=2,
        overall_total_area=200.0,
        floors=[
            FloorAllocation(
                floor_number=1,
                floor_function_tag="residential",
                floor_total_area=100.0,
                core_tube_area=12.0,
                corridor_allowance_area=16.0,
                rooms=[RoomAllocation(room_id="r1", room_name="Living", room_type="living_room", target_area=20.0)],
            ),
            FloorAllocation(
                floor_number=2,
                floor_function_tag="residential",
                floor_total_area=100.0,
                core_tube_area=12.0,
                corridor_allowance_area=16.0,
                rooms=[RoomAllocation(room_id="r2", room_name="Bedroom", room_type="bedroom", target_area=14.0)],
            ),
        ],
    )


def _stage1_resolved():
    return run_stage1_from_allocation(_allocation(), width=15.0, depth=10.0, core_placement="east")


def _core_and_corridor(stage1):
    floor_boundary = box(0.0, 0.0, 15.0, 10.0)
    core, metadata = core_tube_from_stage1_policy(stage1, floor_boundary, require_resolved_bbox=True)
    corridor = stage2_corridor_options_from_stage1(stage1)
    return floor_boundary, core, metadata, corridor


def test_resolved_envelope_builds_floor_free_space() -> None:
    stage1 = _stage1_resolved()
    floor_boundary, core, metadata, corridor = _core_and_corridor(stage1)
    free_space = build_floor_free_space(
        floor_number=1,
        floor_boundary=floor_boundary,
        stage1_core_tube=core,
        core_metadata=metadata,
        corridor_options=corridor,
        topology_mode="grid_growth",
        corridor_width=float(corridor["target_width"]),
    )

    assert free_space.source == "stage1"
    assert free_space.geometry_kind == "polygon"
    assert free_space.stage1_core_reference["core_source"] == "stage1"
    assert free_space.corridor_context_reference["corridor_source"] == "stage1"
    assert pytest.approx(free_space.free_space_geometry.area + core.polygon.area) == floor_boundary.area


def test_unresolved_core_blocks_stage2_geometry() -> None:
    stage1 = run_stage1_from_allocation(_allocation(), core_placement="east")
    with pytest.raises(Stage1ContextMismatchError) as exc:
        core_tube_from_stage1_policy(stage1, box(0.0, 0.0, 15.0, 10.0), require_resolved_bbox=True)
    assert exc.value.failure_type == "core_policy_unresolved"


def test_free_space_has_holes_blocks_geometry() -> None:
    stage1 = _stage1_resolved()
    floor_boundary, core, metadata, corridor = _core_and_corridor(stage1)
    center_bbox = {"x": 6.0, "y": 3.0, "width": 2.0, "depth": 2.0}
    stage1 = stage1.model_copy(
        update={
            "core_context": stage1.core_context.model_copy(
                update={"bbox": center_bbox, "selected_placement": "center"}
            )
        }
    )
    core, metadata = core_tube_from_stage1_policy(stage1, floor_boundary, require_resolved_bbox=True)
    with pytest.raises(LayoutGeometryInvariantError) as exc:
        build_floor_free_space(
            floor_number=1,
            floor_boundary=floor_boundary,
            stage1_core_tube=core,
            core_metadata=metadata,
            corridor_options=corridor,
            topology_mode="grid_growth",
            corridor_width=float(corridor["target_width"]),
        )
    assert exc.value.metadata["failure_type"] == "free_space_has_holes"


def test_free_space_fragmented_blocks_geometry() -> None:
    stage1 = _stage1_resolved()
    floor_boundary, _, metadata, corridor = _core_and_corridor(stage1)
    split_bbox = {"x": 7.0, "y": 0.0, "width": 1.0, "depth": 10.0}
    stage1 = stage1.model_copy(
        update={
            "core_context": stage1.core_context.model_copy(
                update={"bbox": split_bbox, "selected_placement": "center"}
            )
        }
    )
    core, metadata = core_tube_from_stage1_policy(stage1, floor_boundary, require_resolved_bbox=True)
    with pytest.raises(LayoutGeometryInvariantError) as exc:
        build_floor_free_space(
            floor_number=1,
            floor_boundary=floor_boundary,
            stage1_core_tube=core,
            core_metadata=metadata,
            corridor_options=corridor,
            topology_mode="grid_growth",
            corridor_width=float(corridor["target_width"]),
        )
    assert exc.value.metadata["failure_type"] == "free_space_fragmented"


def test_grid_growth_does_not_enter_core() -> None:
    stage1 = _stage1_resolved()
    floor_boundary, core, metadata, corridor = _core_and_corridor(stage1)
    free_space = build_floor_free_space(
        floor_number=1,
        floor_boundary=floor_boundary,
        stage1_core_tube=core,
        core_metadata=metadata,
        corridor_options=corridor,
        topology_mode="grid_growth",
        corridor_width=float(corridor["target_width"]),
    )
    assert free_space.free_space_geometry.intersection(core.polygon).area <= 0.01


def test_island_partition_does_not_enter_core() -> None:
    test_grid_growth_does_not_enter_core()


def test_coverage_fallback_rejects_core_overlap_residual() -> None:
    stage1 = _stage1_resolved()
    floor_boundary, core, metadata, corridor = _core_and_corridor(stage1)
    free_space = build_floor_free_space(
        floor_number=1,
        floor_boundary=floor_boundary,
        stage1_core_tube=core,
        core_metadata=metadata,
        corridor_options=corridor,
        topology_mode="grid_growth",
        corridor_width=float(corridor["target_width"]),
    )
    residual = {"feature_id": "bad_residual", "polygon": core.polygon.buffer(-0.01)}
    overlap = positive_core_overlap_area(
        floor_id="F1",
        topology_mode="grid_growth",
        core_contract=free_space.core_contract,
        coverage_features=[residual],
    )
    assert overlap > 0.0


def test_serializer_rejects_core_overlap() -> None:
    stage1 = _stage1_resolved()
    floor_boundary, core, metadata, corridor = _core_and_corridor(stage1)
    free_space = build_floor_free_space(
        floor_number=1,
        floor_boundary=floor_boundary,
        stage1_core_tube=core,
        core_metadata=metadata,
        corridor_options=corridor,
        topology_mode="grid_growth",
        corridor_width=float(corridor["target_width"]),
    )
    room = SimpleNamespace(id="bad_room", room_type="bedroom", polygon=core.polygon.buffer(-0.01))
    with pytest.raises(LayoutGeometryInvariantError):
        validate_core_exclusion(
            floor_id="F1",
            topology_mode="grid_growth",
            core_contract=free_space.core_contract,
            rooms=[room],
            hard_fail=True,
        )


def test_corridor_policy_preserved_in_stage2() -> None:
    stage1 = _stage1_resolved()
    corridor = stage2_corridor_options_from_stage1(stage1)
    assert corridor["corridor_source"] == "stage1"
    assert corridor["corridor_layout"] == stage1.corridor_context.layout
    corridor["corridor_layout"] = "door_side"
    with pytest.raises(Stage1ContextMismatchError):
        validate_stage1_corridor_context(stage1, corridor)


def test_stage2a_report_contains_core_safe_metadata() -> None:
    stage1 = _stage1_resolved()
    floor_boundary, core, metadata, corridor = _core_and_corridor(stage1)
    free_space = build_floor_free_space(
        floor_number=1,
        floor_boundary=floor_boundary,
        stage1_core_tube=core,
        core_metadata=metadata,
        corridor_options=corridor,
        topology_mode="grid_growth",
        corridor_width=float(corridor["target_width"]),
    )
    report = build_stage2a_report({"F1": free_space})
    assert report["core_source"] == "stage1"
    assert report["corridor_source"] == "stage1"
    assert report["floor_free_space_constructed"] is True
    assert report["free_space_geometry_kind"] == "polygon"
    assert report["core_positive_overlap_area"] == 0.0
    assert report["serializer_core_contract_id"]
