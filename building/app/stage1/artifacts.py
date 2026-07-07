from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import Stage1Result


FORBIDDEN_STAGE1_PATTERNS = (
    re.compile(r"^layout_F\d+\.json$"),
    re.compile(r"^render_F\d+_seg\.png$"),
    re.compile(r"^refined_.*"),
    re.compile(r".*fallback.*layout.*", re.IGNORECASE),
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_payload(result: Stage1Result, artifact_type: str, data: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact_type": artifact_type,
        "schema_version": result.schema_version,
        "stage": result.stage,
        "source": result.source,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
    }
    payload.update(data)
    return payload


def write_stage1_artifacts(result: Stage1Result, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    generated: list[str] = []

    building_path = out_dir / "building_program.json"
    _write_json(building_path, _artifact_payload(result, "building_program", result.building_program.model_dump(mode="json")))
    paths["building_program"] = str(building_path)
    generated.append(building_path.name)

    core_path = out_dir / "core_policy.json"
    _write_json(core_path, _artifact_payload(result, "core_policy", {"core_policy": result.core_policy.model_dump(mode="json")}))
    paths["core_policy"] = str(core_path)
    generated.append(core_path.name)

    corridor_path = out_dir / "corridor_policy.json"
    _write_json(corridor_path, _artifact_payload(result, "corridor_policy", {"corridor_policy": result.corridor_policy.model_dump(mode="json")}))
    paths["corridor_policy"] = str(corridor_path)
    generated.append(corridor_path.name)

    repair_path = out_dir / "program_repair_log.json"
    _write_json(repair_path, _artifact_payload(result, "program_repair_log", result.program_repair_log.model_dump(mode="json")))
    paths["program_repair_log"] = str(repair_path)
    generated.append(repair_path.name)

    for floor in result.floor_programs:
        path = out_dir / f"floor_program_F{int(floor.floor_number)}.json"
        _write_json(path, _artifact_payload(result, "floor_program", floor.model_dump(mode="json")))
        paths[f"floor_program_F{int(floor.floor_number)}"] = str(path)
        generated.append(path.name)

    for report in result.feasibility_reports:
        path = out_dir / f"feasibility_F{int(report.floor_number)}.json"
        _write_json(path, _artifact_payload(result, "feasibility_report", report.model_dump(mode="json")))
        paths[f"feasibility_F{int(report.floor_number)}"] = str(path)
        generated.append(path.name)

    forbidden = [
        p.name
        for p in sorted(out_dir.iterdir(), key=lambda item: item.name)
        if any(pattern.match(p.name) for pattern in FORBIDDEN_STAGE1_PATTERNS)
    ]
    manifest_path = out_dir / "stage1_manifest.json"
    manifest = {
        "artifact_type": "stage1_manifest",
        "schema_version": result.schema_version,
        "stage": result.stage,
        "source": result.source,
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "out_dir": str(out_dir),
        "generated_artifacts": sorted(generated + [manifest_path.name]),
        "forbidden_artifacts_present": forbidden,
    }
    _write_json(manifest_path, manifest)
    paths["stage1_manifest"] = str(manifest_path)

    return paths
