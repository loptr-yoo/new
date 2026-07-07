from __future__ import annotations

import pytest

from building.app.models.request import BuildingGenerateRequest
from building.app.models import GenerateSemanticsRequest, SceneType


def test_generate_request_accepts_total_floors_2() -> None:
    req = BuildingGenerateRequest(prompt="two floor house", total_floors=2)
    assert req.total_floors == 2
    assert req.topology_mode == "grid_growth"
    assert req.corridor_layout == "organic"


def test_generate_request_rejects_total_floors_1() -> None:
    with pytest.raises(Exception):
        BuildingGenerateRequest(prompt="one floor", total_floors=1)


def test_generate_request_rejects_scene_type_floor() -> None:
    with pytest.raises(Exception):
        BuildingGenerateRequest(prompt="floor", total_floors=2, scene_type="floor")


def test_generate_request_rejects_scene_type_parking() -> None:
    with pytest.raises(Exception):
        BuildingGenerateRequest(prompt="parking", total_floors=2, scene_type="parking")


def test_generate_request_rejects_parking_scene_id() -> None:
    with pytest.raises(Exception):
        BuildingGenerateRequest(prompt="parking", total_floors=2, sceneId="parking" + "_" + "underground")


def test_semantics_request_rejects_explicit_single_floor() -> None:
    with pytest.raises(Exception):
        GenerateSemanticsRequest(scene_type=SceneType.BUILDING, user_prompt="x", total_floors=1)

