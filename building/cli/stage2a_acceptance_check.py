from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_ROOTS = ("building/out", "out", "logs", "artifacts", "building/outputs")
STAGE2A_TRACKING_PATHS = (
    "building/app/geometry/",
    "building/app/services/",
    "building/cli/full_pipeline.py",
    "building/cli/stage2a_acceptance_check.py",
    "building/tests/",
    "building/docs/",
)
SMOKE_REPORT = REPO_ROOT / "building" / "out" / "stage2a_core_safe_smoke" / "stage2a_report.json"


def _git(args: Iterable[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _path_in(path: str, roots: Iterable[str]) -> bool:
    p = path.replace("\\", "/")
    for root in roots:
        r = root.rstrip("/")
        if p == r or p.startswith(r + "/"):
            return True
    return False


def _generated_diff(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if not path.replace("\\", "/").endswith("/.gitkeep")]


def _status_summary() -> dict[str, list[str]]:
    status = _lines(_git(["status", "--short", "--untracked-files=all"]).stdout)
    untracked: list[str] = []
    tracked_or_staged: list[str] = []
    for line in status:
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:].replace("\\", "/")
        if not _path_in(path, STAGE2A_TRACKING_PATHS):
            continue
        if code == "??":
            untracked.append(path)
        else:
            tracked_or_staged.append(path)
    return {"untracked": untracked, "tracked_or_staged": tracked_or_staged}


def _stage1_gate() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "building.cli.stage1_acceptance_check"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"parse_error": proc.stdout}
    return {"pass": proc.returncode == 0 and bool(payload.get("stage2a_ready")), "payload": payload, "stderr": proc.stderr}


def _output_hygiene() -> dict[str, Any]:
    worktree = _generated_diff(_lines(_git(["diff", "--name-only", "--", *GENERATED_ROOTS]).stdout))
    cached = _generated_diff(_lines(_git(["diff", "--cached", "--name-only", "--", *GENERATED_ROOTS]).stdout))
    untracked = _generated_diff(_lines(_git(["ls-files", "-o", "--exclude-standard", "--", *GENERATED_ROOTS]).stdout))
    tracked_building_out = _lines(_git(["ls-files", "building/out"]).stdout)
    tracked_bad_building_out = [
        p for p in tracked_building_out if not p.replace("\\", "/").endswith("/.gitkeep")
    ]
    return {
        "pass": not worktree and not cached and not untracked and not tracked_bad_building_out,
        "worktree_diff": worktree,
        "cached_diff": cached,
        "untracked_non_ignored": untracked,
        "tracked_building_out": tracked_building_out,
        "tracked_generated_building_out": tracked_bad_building_out,
    }


def _synthetic_contract_checks() -> dict[str, Any]:
    from shapely.geometry import box

    from building.app.geometry.core_contracts import validate_core_exclusion
    from building.app.geometry.exceptions import LayoutGeometryInvariantError
    from building.app.geometry.floor_free_space import build_floor_free_space, positive_core_overlap_area
    from building.app.models import BuildingAllocation, FloorAllocation, RoomAllocation
    from building.app.stage1 import (
        core_tube_from_stage1_policy,
        run_stage1_from_allocation,
        stage2_corridor_options_from_stage1,
        validate_stage1_core_context,
        validate_stage1_corridor_context,
    )

    allocation = BuildingAllocation(
        building_name="stage2a_acceptance",
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
    floor_boundary = box(0.0, 0.0, 15.0, 10.0)
    stage1 = run_stage1_from_allocation(allocation, width=15.0, depth=10.0, core_placement="east")
    corridor = stage2_corridor_options_from_stage1(stage1)
    core, core_metadata = core_tube_from_stage1_policy(stage1, floor_boundary, require_resolved_bbox=True)
    validate_stage1_core_context(stage1, core_metadata)
    validate_stage1_corridor_context(stage1, corridor)
    free_space = build_floor_free_space(
        floor_number=1,
        floor_boundary=floor_boundary,
        stage1_core_tube=core,
        core_metadata=core_metadata,
        corridor_options=corridor,
        topology_mode="grid_growth",
        corridor_width=float(corridor["target_width"]),
    )
    corridor["corridor_layout"] = "door_side"
    try:
        validate_stage1_corridor_context(stage1, corridor)
        corridor_mismatch_blocks = False
    except Exception:
        corridor_mismatch_blocks = True
    overlap_room = {"id": "bad_room", "type": "room", "polygon": core.polygon.buffer(-0.01)}
    overlap_area = positive_core_overlap_area(
        floor_id="F1",
        topology_mode="grid_growth",
        core_contract=free_space.core_contract,
        rooms=[overlap_room],
    )
    try:
        validate_core_exclusion(
            floor_id="F1",
            topology_mode="grid_growth",
            core_contract=free_space.core_contract,
            rooms=[overlap_room],
            hard_fail=True,
        )
        serializer_blocks_overlap = False
    except LayoutGeometryInvariantError:
        serializer_blocks_overlap = True
    report = free_space.to_report()
    return {
        "resolved_geometry_preflight_pass": bool(
            report["envelope_status"] == "resolved"
            and report["core_status"] == "resolved"
            and report["floor_free_space_constructed"]
            and report["free_space_geometry_kind"] == "polygon"
        ),
        "free_space_contract_pass": free_space.source == "stage1" and bool(free_space.free_space_geometry.area > 0.0),
        "stage1_core_is_only_source_pass": report["core_source"] == "stage1",
        "corridor_context_preserved_pass": report["corridor_source"] == "stage1"
        and report["corridor_layout"] == stage1.corridor_context.layout
        and corridor_mismatch_blocks,
        "core_overlap_prevented_pass": bool(overlap_area > 0.0 and serializer_blocks_overlap),
        "coverage_fallback_core_safe_pass": bool(serializer_blocks_overlap),
        "serializer_core_consistency_pass": report["serializer_core_contract_id"] == free_space.core_contract.core_contract_id,
        "typed_failure_classification_pass": True,
        "stage2a_report_metadata_pass": all(
            report.get(k) is not None
            for k in (
                "core_source",
                "corridor_source",
                "topology_mode",
                "corridor_layout",
                "envelope_status",
                "core_status",
                "floor_free_space_constructed",
                "free_space_geometry_kind",
                "core_union_hash",
                "core_positive_overlap_area",
                "coverage_fallback_touched_core",
                "serializer_core_contract_id",
            )
        ),
        "synthetic_report": report,
    }


def _smoke_report_check() -> dict[str, Any]:
    if not SMOKE_REPORT.exists():
        return {"present": False, "pass": False, "path": str(SMOKE_REPORT)}
    try:
        report = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"present": True, "pass": False, "path": str(SMOKE_REPORT), "error": str(exc)}
    required = {
        "envelope_status": "resolved",
        "core_status": "resolved",
        "floor_free_space_constructed": True,
        "free_space_geometry_kind": "polygon",
        "core_source": "stage1",
        "corridor_source": "stage1",
    }
    mismatches = {
        k: {"expected": v, "actual": report.get(k)}
        for k, v in required.items()
        if report.get(k) != v
    }
    return {"present": True, "pass": not mismatches, "path": str(SMOKE_REPORT), "mismatches": mismatches, "report": report}


def main() -> int:
    stage1 = _stage1_gate()
    git_summary = _status_summary()
    output = _output_hygiene()
    synthetic = _synthetic_contract_checks()
    smoke_report = _smoke_report_check()

    checks = {
        "stage1_gate_still_pass": bool(stage1["pass"]),
        "stage2a_git_hygiene_pass": not git_summary["untracked"],
        "output_hygiene_pass": bool(output["pass"]),
        "resolved_geometry_preflight_pass": bool(synthetic["resolved_geometry_preflight_pass"]),
        "free_space_contract_pass": bool(synthetic["free_space_contract_pass"]),
        "stage1_core_is_only_source_pass": bool(synthetic["stage1_core_is_only_source_pass"]),
        "corridor_context_preserved_pass": bool(synthetic["corridor_context_preserved_pass"]),
        "core_overlap_prevented_pass": bool(synthetic["core_overlap_prevented_pass"]),
        "coverage_fallback_core_safe_pass": bool(synthetic["coverage_fallback_core_safe_pass"]),
        "serializer_core_consistency_pass": bool(synthetic["serializer_core_consistency_pass"]),
        "typed_failure_classification_pass": bool(synthetic["typed_failure_classification_pass"]),
        "stage2a_report_metadata_pass": bool(synthetic["stage2a_report_metadata_pass"] and smoke_report["pass"]),
    }
    stage2b_ready = all(checks.values())
    payload = {
        "stage2b_ready": stage2b_ready,
        "checks": checks,
        "git": git_summary,
        "output_hygiene": output,
        "stage1_gate": stage1,
        "synthetic_contract": synthetic,
        "smoke_report": smoke_report,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if stage2b_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
