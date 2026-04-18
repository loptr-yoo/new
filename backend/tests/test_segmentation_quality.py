from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.geometry.postprocessor import generate_walls_from_topology
from backend.core.geometry.style_constants import SEGMENTATION_COLORS
from backend.core.interior.models import FurnitureSpec, FurnitureCategory, RoomBoundary
from backend.core.interior.refine_solver import solve_nonoverlap_layout_greedy
from scripts import full_pipeline as fp
from scripts import local_renderer as lr


def test_palette_is_distinguishable():
    lr._validate_segmentation_palette(SEGMENTATION_COLORS)


def test_seg_mode_requires_png_output():
    layout = {"width": 2.0, "height": 2.0, "elements": []}
    with pytest.raises(ValueError):
        lr._render(layout, out_path=Path("x.svg"), mode="seg")


def test_build_room_inputs_excludes_entrance():
    layout = {
        "width": 10.0,
        "height": 8.0,
        "elements": [
            {"id": "room_entrance", "type": "entrance", "polygon": [[0, 7], [10, 7], [10, 8], [0, 8], [0, 7]]},
            {"id": "room_bed", "type": "bedroom", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]]},
        ],
    }
    rooms = fp._build_room_inputs("F1", layout)
    ids = {r.room_id for r in rooms}
    assert "room_entrance" not in ids
    assert "room_bed" in ids


def test_topology_snap_augments_missing_edge():
    room_rects = {
        "room_a": (0.0, 0.0, 2.0, 2.0),
        "room_b": (2.12, 0.0, 2.0, 2.0),  # gap 0.12 < snap tolerance
    }
    walls = generate_walls_from_topology(
        room_rects=room_rects,
        edge_set={},
        floor_bounds=(0.0, 0.0, 6.0, 4.0),
        zone_types={"room_a": "room", "room_b": "room"},
    )
    between = [w for w in walls if w.type == "partition_wall" and set(w.room_ids) == {"room_a", "room_b"}]
    assert between


def test_greedy_center_validator_blocks_narrow_tail():
    room = RoomBoundary(x_min=0.0, y_min=0.0, x_max=6.0, y_max=2.0)
    furnitures = [FurnitureSpec(id="shelf_1", name="shelf", category=FurnitureCategory.CABINET, width=1.0, height=0.6)]
    refined = solve_nonoverlap_layout_greedy(
        room=room,
        furnitures=furnitures,
        obstacles=[],
        center_validator=lambda cx, cy: cx <= 2.5,
    )
    assert refined.items
    assert refined.items[0].cx <= 2.5
