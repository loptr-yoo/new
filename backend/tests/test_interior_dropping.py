from backend.core.interior.models import FurnitureCategory, FurnitureSpec, RoomBoundary
from backend.core.interior.refine_solver import solve_nonoverlap_layout_greedy


def test_greedy_drops_when_room_too_small():
    room = RoomBoundary(x_min=0.0, y_min=0.0, x_max=2.0, y_max=2.0)
    furnitures = [
        FurnitureSpec(id="bed_1", name="bed", category=FurnitureCategory.BEDDING, width=1.6, height=1.6, priority=0),
        FurnitureSpec(id="cabinet_1", name="cabinet", category=FurnitureCategory.CABINET, width=1.2, height=1.2, priority=3),
    ]
    refined = solve_nonoverlap_layout_greedy(room=room, furnitures=furnitures, obstacles=[], coarse_layout=None)
    assert len(refined.items) == 1
    assert refined.items[0].furniture_id == "bed_1"
    assert any(w.startswith("dropped_furnitures=") for w in refined.warnings)

