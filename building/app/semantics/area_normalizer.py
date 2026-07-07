from __future__ import annotations

from typing import Any, Dict, List, Tuple


def normalize_floor_room_areas(
    obj: Dict[str, Any],
    *,
    min_room_area: float = 4.0,
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    floors = obj.get("floors")
    if not isinstance(floors, list):
        return obj, warnings

    out: Dict[str, Any] = dict(obj)
    out_floors: List[Any] = []

    for floor in floors:
        if not isinstance(floor, dict):
            out_floors.append(floor)
            continue

        rooms = floor.get("rooms")
        if not isinstance(rooms, list) or not rooms:
            out_floors.append(dict(floor))
            continue

        try:
            floor_number = int(floor.get("floor_number", 0) or 0)
        except Exception:
            floor_number = 0

        def _sf(v: Any, default: float = 0.0) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        floor_total_area = _sf(floor.get("floor_total_area", 0.0))
        core_area = _sf(floor.get("core_tube_area", 0.0))
        corridor_area = _sf(floor.get("corridor_allowance_area", 0.0))
        available = float(floor_total_area - core_area - corridor_area)

        if available <= 0.0:
            out_floors.append(dict(floor))
            continue

        weights: List[float] = []
        room_dicts: List[Dict[str, Any]] = []
        for r in rooms:
            if not isinstance(r, dict):
                continue
            a = _sf(r.get("target_area", 0.0))
            weights.append(max(1e-6, float(a)))
            room_dicts.append(r)

        if not room_dicts:
            out_floors.append(dict(floor))
            continue

        total = float(sum(weights))
        if total <= available + 1e-6:
            out_floors.append(dict(floor))
            continue

        scale = float(available / total) if total > 1e-9 else 0.0
        min_a = float(min_room_area)
        n = len(room_dicts)
        remaining = float(available)

        pool = set(range(n))
        areas: List[float] = [0.0 for _ in range(n)]
        fixed_count = 0

        while pool:
            if remaining < 0.0:
                for i in range(n):
                    areas[i] = min_a
                warnings.append(
                    f"F{floor_number}: available_area={available:.2f} too small for min_room_area={min_a:.2f} (n={n}), forcing all rooms to min"
                )
                remaining = 0.0
                pool.clear()
                break

            pool_sum = float(sum(weights[i] for i in pool))
            if pool_sum <= 1e-9:
                each = float(remaining / float(len(pool))) if pool else 0.0
                for i in list(pool):
                    areas[i] = max(min_a, each)
                pool.clear()
                break

            proposed: Dict[int, float] = {}
            below: List[int] = []
            for i in pool:
                v = float(remaining) * float(weights[i]) / float(pool_sum)
                proposed[i] = v
                if v < min_a - 1e-9:
                    below.append(i)

            if not below:
                for i, v in proposed.items():
                    areas[i] = float(v)
                pool.clear()
                break

            for i in below:
                areas[i] = min_a
                remaining -= min_a
                pool.remove(i)
                fixed_count += 1

        new_rooms: List[Any] = []
        for i, r in enumerate(room_dicts):
            nr = dict(r)
            nr["target_area"] = float(max(min_a, areas[i]))
            new_rooms.append(nr)

        scaled_total = float(sum(float(rr.get("target_area", 0.0)) for rr in new_rooms))
        if scaled_total > available + 1e-4:
            warnings.append(
                f"F{floor_number}: normalized total_target_area still exceeds available_area (total={scaled_total:.2f}, available={available:.2f})"
            )

        new_floor = dict(floor)
        new_floor["rooms"] = new_rooms
        out_floors.append(new_floor)

        warnings.append(
            f"F{floor_number}: scaled room target_area to fit available_area (before={total:.2f}, available={available:.2f}, after={scaled_total:.2f}, initial_scale={scale:.4f}, fixed_count={fixed_count}, remaining={remaining:.2f})"
        )

    out["floors"] = out_floors
    return out, warnings
