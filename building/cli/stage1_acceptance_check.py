from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE1_TRACKING_PATHS = (
    "building/app/stage1/",
    "building/app/policies/",
    "building/app/pipeline_defaults.py",
    "building/app/diagnostics/failure_taxonomy.py",
    "building/app/api/routers/generate.py",
    "building/app/services/building_pipeline_service.py",
    "building/app/models/request.py",
    "building/cli/full_pipeline.py",
    "building/cli/stage1_acceptance_check.py",
    "building/docs/policy_inventory.md",
    "building/tests/test_stage1_api.py",
    "building/tests/test_stage1_program.py",
)

GENERATED_ROOTS = ("building/out", "out", "logs", "artifacts", "building/outputs")


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


def _stage1_status(status_lines: list[str]) -> tuple[list[str], list[str]]:
    untracked: list[str] = []
    staged_or_tracked: list[str] = []
    for line in status_lines:
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:].replace("\\", "/")
        if not _path_in(path, STAGE1_TRACKING_PATHS):
            continue
        if code == "??":
            untracked.append(path)
        else:
            staged_or_tracked.append(path)
    return untracked, staged_or_tracked


def _manifest_check(out_dir: Path) -> dict:
    manifest_path = out_dir / "stage1_manifest.json"
    if not manifest_path.exists():
        return {"present": False, "pass": False, "error": f"missing {manifest_path}"}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {
        "building_program.json",
        "core_policy.json",
        "corridor_policy.json",
        "program_repair_log.json",
        "stage1_manifest.json",
    }
    generated = set(data.get("generated_artifacts") or [])
    unexpected = []
    for name in generated:
        if name.startswith("floor_program_F") or name.startswith("feasibility_F"):
            continue
        if name not in allowed:
            return {"present": True, "pass": False, "error": f"unexpected artifact {name}", "manifest": data}
    for path in out_dir.iterdir():
        if path.is_dir():
            continue
        name = path.name
        if name in generated:
            continue
        if name.startswith("floor_program_F") or name.startswith("feasibility_F"):
            continue
        if name not in allowed:
            unexpected.append(name)
    forbidden = list(data.get("forbidden_artifacts_present") or [])
    return {
        "present": True,
        "pass": not forbidden and not unexpected,
        "forbidden_artifacts_present": forbidden,
        "unexpected_artifacts_present": unexpected,
        "manifest": data,
    }


def _code_gate_checks() -> dict[str, bool]:
    from shapely.geometry import box

    from building.app.models import BuildingAllocation, FloorAllocation, RoomAllocation
    from building.app.stage1 import (
        Stage1ContextMismatchError,
        Stage1ProgramInfeasibleError,
        building_allocation_from_stage1,
        core_tube_from_stage1_policy,
        run_stage1_from_allocation,
        stage2_corridor_options_from_stage1,
        validate_stage1_core_context,
        validate_stage1_corridor_context,
    )

    feasible = BuildingAllocation(
        building_name="ok",
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
    rooms = [
        RoomAllocation(room_id=f"bad{i}", room_name=f"Bad {i}", room_type="bedroom", target_area=20.0)
        for i in range(10)
    ]
    infeasible = BuildingAllocation(
        building_name="bad",
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
    bad_stage1 = run_stage1_from_allocation(infeasible)
    try:
        building_allocation_from_stage1(bad_stage1)
        infeasible_blocks = False
    except Stage1ProgramInfeasibleError:
        infeasible_blocks = not bad_stage1.can_enter_geometry

    ok_stage1 = run_stage1_from_allocation(feasible)
    try:
        _, core_metadata = core_tube_from_stage1_policy(ok_stage1, box(0, 0, 12, 8))
        validate_stage1_core_context(ok_stage1, core_metadata)
        core_metadata["selected_placement"] = "west"
        try:
            validate_stage1_core_context(ok_stage1, core_metadata)
            core_consistency = False
        except Stage1ContextMismatchError:
            core_consistency = True
    except Exception:
        core_consistency = False

    try:
        corridor_options = stage2_corridor_options_from_stage1(ok_stage1)
        validate_stage1_corridor_context(ok_stage1, corridor_options)
        corridor_options["corridor_layout"] = "door_side"
        try:
            validate_stage1_corridor_context(ok_stage1, corridor_options)
            corridor_consistency = False
        except Stage1ContextMismatchError:
            corridor_consistency = True
    except Exception:
        corridor_consistency = False

    return {
        "infeasible_blocks_stage2_pass": bool(infeasible_blocks),
        "core_context_consistency_pass": bool(core_consistency),
        "corridor_context_consistency_pass": bool(corridor_consistency),
    }


def main() -> int:
    status = _lines(_git(["status", "--short", "--untracked-files=all"]).stdout)
    stage1_untracked, stage1_tracked = _stage1_status(status)

    generated_worktree_diff = _generated_diff(_lines(_git(["diff", "--name-only", "--", *GENERATED_ROOTS]).stdout))
    generated_cached_diff = _generated_diff(_lines(_git(["diff", "--cached", "--name-only", "--", *GENERATED_ROOTS]).stdout))
    tracked_building_out = _lines(_git(["ls-files", "building/out"]).stdout)
    tracked_legacy_out = _lines(_git(["ls-files", "out"]).stdout)
    tracked_legacy_building_outputs = _lines(_git(["ls-files", "building/outputs"]).stdout)

    manifest = _manifest_check(REPO_ROOT / "building" / "out" / "stage1_acceptance_audit")

    code_checks = _code_gate_checks()
    checks = {
        "git_hygiene_pass": not stage1_untracked,
        "generated_outputs_excluded_pass": not generated_worktree_diff and not generated_cached_diff,
        "stage1_artifact_allowlist_pass": bool(manifest.get("pass")),
        **code_checks,
    }
    stage2a_ready = all(checks.values())
    payload = {
        "stage2a_ready": stage2a_ready,
        "checks": checks,
        "git": {
            "stage1_untracked": stage1_untracked,
            "stage1_tracked_or_staged": stage1_tracked,
            "generated_worktree_diff": generated_worktree_diff,
            "generated_cached_diff": generated_cached_diff,
            "tracked_building_out_files": tracked_building_out,
            "tracked_legacy_out_files": tracked_legacy_out,
            "tracked_legacy_building_outputs": tracked_legacy_building_outputs,
        },
        "artifact_manifest": manifest,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if stage2a_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
