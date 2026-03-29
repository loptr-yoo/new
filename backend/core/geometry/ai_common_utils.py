from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ...models import ConstraintViolation, LayoutElement, ParkingLayout
from ..types import ElementType
from .geometry import get_intersection_box, validate_layout


def subtract_rectangle(
    r1: Dict[str, float],
    r2: Dict[str, float],
) -> List[Dict[str, float]]:
    ix = max(r1["x"], r2["x"])
    iy = max(r1["y"], r2["y"])
    iw = min(r1["x"] + r1["width"], r2["x"] + r2["width"]) - ix
    ih = min(r1["y"] + r1["height"], r2["y"] + r2["height"]) - iy
    if iw <= 0 or ih <= 0:
        return [r1]

    res: List[Dict[str, float]] = []
    if r1["y"] < iy:
        res.append({"x": r1["x"], "y": r1["y"], "width": r1["width"], "height": iy - r1["y"]})
    if r1["y"] + r1["height"] > iy + ih:
        res.append(
            {
                "x": r1["x"],
                "y": iy + ih,
                "width": r1["width"],
                "height": (r1["y"] + r1["height"]) - (iy + ih),
            }
        )
    if r1["x"] < ix:
        res.append({"x": r1["x"], "y": iy, "width": ix - r1["x"], "height": ih})
    if r1["x"] + r1["width"] > ix + iw:
        res.append(
            {
                "x": ix + iw,
                "y": iy,
                "width": (r1["x"] + r1["width"]) - (ix + iw),
                "height": ih,
            }
        )
    return res


def normalize_type(t: Optional[str]) -> str:
    if not t or not isinstance(t, str):
        return ElementType.WALL
    key = t.lower().strip()
    key = "_".join(key.split())
    mapping: Dict[str, str] = {
        "ramp": ElementType.RAMP,
        "slope": ElementType.RAMP,
        "speed_bump": ElementType.SPEED_BUMP,
        "deceleration_zone": ElementType.SPEED_BUMP,
        "road": ElementType.ROAD,
        "driving_lane": ElementType.ROAD,
        "pedestrian_path": ElementType.SIDEWALK,
        "sidewalk": ElementType.SIDEWALK,
        "ground_line": ElementType.LANE_LINE,
        "lane_line": ElementType.LANE_LINE,
        "parking_spot": ElementType.PARKING_SPACE,
        "parking_space": ElementType.PARKING_SPACE,
        "parking": ElementType.PARKING_SPACE,
        "charging": ElementType.CHARGING_STATION,
        "charging_station": ElementType.CHARGING_STATION,
        "ev_charging_zone": ElementType.CHARGING_STATION,
        "charging_zone": ElementType.CHARGING_STATION,
        "ground": ElementType.GROUND,
        "island": ElementType.GROUND,
        "landscape": ElementType.GROUND,
        "landscape_area": ElementType.GROUND,
        "buffer": ElementType.GROUND,
        "median": ElementType.GROUND,
        "pillar": ElementType.PILLAR,
        "elevator_hall": ElementType.ELEVATOR,
        "elevator": ElementType.ELEVATOR,
        "staircase": ElementType.STAIRCASE,
        "stairs": ElementType.STAIRCASE,
        "fire_stairs": ElementType.STAIRCASE,
        "safe_exit": ElementType.SAFE_EXIT,
        "fire_extinguisher": ElementType.FIRE_EXTINGUISHER,
        "guidance_sign": ElementType.GUIDANCE_SIGN,
        "parking_strip": ElementType.GROUND,
        "central_island": ElementType.GROUND,
        "green_zone": ElementType.GROUND,
        "void": ElementType.GROUND,
        "wall": ElementType.WALL,
        "entrance": ElementType.ENTRANCE,
        "exit": ElementType.EXIT,
        "convex_mirror": ElementType.CONVEX_MIRROR,
    }
    return mapping.get(key, key)


def normalize_partial_patches(patches: Any) -> List[Dict[str, Any]]:
    if not isinstance(patches, list):
        return []
    out: List[Dict[str, Any]] = []
    for p in patches:
        if not isinstance(p, dict):
            continue
        q: Dict[str, Any] = {}
        if "id" in p and p["id"] is not None:
            q["id"] = str(p["id"])
        elif "old_id" in p and p["old_id"] is not None:
            q["id"] = str(p["old_id"])
        elif "original_id" in p and p["original_id"] is not None:
            q["id"] = str(p["original_id"])
        elif "ref_id" in p and p["ref_id"] is not None:
            q["id"] = str(p["ref_id"])

        if "type" in p or "t" in p:
            q["type"] = normalize_type(p.get("type") if p.get("type") is not None else p.get("t"))

        if "x" in p and p["x"] is not None:
            try:
                q["x"] = float(p["x"])
            except Exception:
                pass
        if "y" in p and p["y"] is not None:
            try:
                q["y"] = float(p["y"])
            except Exception:
                pass
        if ("width" in p and p["width"] is not None) or ("w" in p and p["w"] is not None):
            try:
                q["width"] = float(p.get("width") if p.get("width") is not None else p.get("w"))
            except Exception:
                pass
        if ("height" in p and p["height"] is not None) or ("h" in p and p["h"] is not None):
            try:
                q["height"] = float(p.get("height") if p.get("height") is not None else p.get("h"))
            except Exception:
                pass
        if ("rotation" in p and p["rotation"] is not None) or ("r" in p and p["r"] is not None):
            try:
                q["rotation"] = float(p.get("rotation") if p.get("rotation") is not None else p.get("r"))
            except Exception:
                pass
        if "label" in p and p["label"] is not None:
            q["label"] = p["label"]
        elif "l" in p and p["l"] is not None:
            q["label"] = p["l"]
        out.append(q)
    return out


def map_to_internal_layout(raw: Any) -> ParkingLayout:
    width = 800
    height = 600
    if isinstance(raw, dict):
        try:
            width = int(float(raw.get("width") or 800)) or 800
        except Exception:
            width = 800
        try:
            height = int(float(raw.get("height") or 600)) or 600
        except Exception:
            height = 600
    els: List[LayoutElement] = []
    elements = raw.get("elements") if isinstance(raw, dict) else None
    if not isinstance(elements, list):
        elements = []
    for e in elements:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("id") or f"el_{format(int(time.time() * 1000), 'x')}_{len(els)}")
        t = normalize_type(e.get("type") or e.get("t"))
        try:
            x = float(e.get("x") or 0) or 0
        except Exception:
            x = 0
        try:
            y = float(e.get("y") or 0) or 0
        except Exception:
            y = 0
        try:
            w = float(e.get("width") if e.get("width") is not None else (e.get("w") if e.get("w") is not None else 10)) or 10
        except Exception:
            w = 10
        try:
            h = float(e.get("height") if e.get("height") is not None else (e.get("h") if e.get("h") is not None else 10)) or 10
        except Exception:
            h = 10
        try:
            r = float(e.get("rotation") if e.get("rotation") is not None else (e.get("r") if e.get("r") is not None else 0)) or 0
        except Exception:
            r = 0
        label = e.get("label") if e.get("label") is not None else e.get("l")
        els.append(LayoutElement(id=eid, type=t, x=x, y=y, width=w, height=h, rotation=r, label=label))
    return ParkingLayout(width=width, height=height, elements=els)


def merge_patches_to_layout(
    current_layout: ParkingLayout,
    patches: Sequence[Dict[str, Any]],
    deleted_ids: Sequence[str] = (),
    mode: str = "strict",
) -> ParkingLayout:
    element_map: Dict[str, LayoutElement] = {el.id: el for el in current_layout.elements}
    for did in list(deleted_ids or []):
        if did in element_map:
            del element_map[did]
    if not patches:
        return ParkingLayout(width=current_layout.width, height=current_layout.height, elements=list(element_map.values()), sceneId=current_layout.sceneId)

    def prune(o: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in o.items() if v is not None}

    for raw in patches:
        if not isinstance(raw, dict):
            continue
        patch = prune(raw)
        pid = str(patch.get("id") or patch.get("element_id") or "")
        if not pid:
            continue
        existing = element_map.get(pid)
        if existing is not None:
            merged = existing.model_copy()
            for key, val in patch.items():
                if val is None:
                    continue
                if key in ("x", "y", "width", "height", "rotation"):
                    try:
                        setattr(merged, key, float(val))
                    except Exception:
                        pass
                elif key in ("type", "t"):
                    merged.type = normalize_type(patch.get("type") if patch.get("type") is not None else patch.get("t"))
                else:
                    try:
                        setattr(merged, key, val)
                    except Exception:
                        pass
            element_map[pid] = merged
        else:
            if mode == "allowCreate":
                t_val = normalize_type(patch.get("type") if patch.get("type") is not None else patch.get("t"))
                if t_val and patch.get("x") is not None and patch.get("y") is not None:
                    try:
                        x = float(patch.get("x"))
                        y = float(patch.get("y"))
                        w = float(patch.get("width") if patch.get("width") is not None else patch.get("w") or 10)
                        h = float(patch.get("height") if patch.get("height") is not None else patch.get("h") or 10)
                        r = float(patch.get("rotation") if patch.get("rotation") is not None else patch.get("r") or 0)
                    except Exception:
                        continue
                    element_map[pid] = LayoutElement(
                        id=pid,
                        type=t_val,
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        rotation=r,
                        label=patch.get("label") if patch.get("label") is not None else patch.get("l"),
                    )
            else:
                pass
    return ParkingLayout(width=current_layout.width, height=current_layout.height, elements=list(element_map.values()), sceneId=current_layout.sceneId)


def dedupe_patches_against_layout(layout: ParkingLayout, patches: Sequence[Dict[str, Any]], tolerance: float = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for patch in patches:
        t = normalize_type(patch.get("type") if patch.get("type") is not None else patch.get("t"))
        try:
            x = float(patch.get("x") or 0)
            y = float(patch.get("y") or 0)
            w = float(patch.get("width") if patch.get("width") is not None else (patch.get("w") if patch.get("w") is not None else 0))
            h = float(patch.get("height") if patch.get("height") is not None else (patch.get("h") if patch.get("h") is not None else 0))
        except Exception:
            out.append(patch)
            continue
        duplicate = False
        for el in layout.elements:
            if normalize_type(el.type) != t:
                continue
            near_pos = abs(el.x - x) <= tolerance and abs(el.y - y) <= tolerance
            near_size = True
            if w and h:
                near_size = abs(el.width - w) <= tolerance and abs(el.height - h) <= tolerance
            if near_pos and near_size:
                duplicate = True
                break
        if not duplicate:
            out.append(patch)
    return out


def infer_parking_forward(spot: LayoutElement, roads: Sequence[LayoutElement]) -> Optional[Tuple[float, float, float]]:
    if spot.type != ElementType.PARKING_SPACE or not roads:
        return None
    sx = spot.x + spot.width / 2
    sy = spot.y + spot.height / 2
    best_d: Optional[float] = None
    best_vec: Optional[Tuple[float, float, float]] = None
    for r in roads:
        rx = r.x + r.width / 2
        ry = r.y + r.height / 2
        dx = rx - sx
        dy = ry - sy
        dist = abs(dx) + abs(dy)
        if dist == 0:
            continue
        if abs(dx) >= abs(dy):
            vec: Tuple[float, float, float] = (1.0, 0.0, 0.0) if dx >= 0 else (-1.0, 0.0, 0.0)
        else:
            vec = (0.0, 1.0, 0.0) if dy >= 0 else (0.0, -1.0, 0.0)
        if best_d is None or dist < best_d:
            best_d = dist
            best_vec = vec
    return best_vec


def snap_ground_to_boundaries(layout: ParkingLayout) -> ParkingLayout:
    SNAP_TOLERANCE = 15
    elements = list(layout.elements)
    structural = [e for e in elements if e.type in (ElementType.ROAD, ElementType.WALL)]
    updated: List[LayoutElement] = []
    for el in elements:
        if el.type != ElementType.GROUND:
            updated.append(el)
            continue
        new_x = el.x
        new_y = el.y
        new_w = el.width
        new_h = el.height
        for target in structural:
            if max(el.y, target.y) < min(el.y + el.height, target.y + target.height):
                if el.x > (target.x + target.width) and abs(el.x - (target.x + target.width)) <= SNAP_TOLERANCE:
                    diff = el.x - (target.x + target.width)
                    new_x -= diff
                    new_w += diff
                if (el.x + el.width) < target.x and abs((el.x + el.width) - target.x) <= SNAP_TOLERANCE:
                    new_w += target.x - (el.x + el.width)
            if max(el.x, target.x) < min(el.x + el.width, target.x + target.width):
                if el.y > (target.y + target.height) and abs(el.y - (target.y + target.height)) <= SNAP_TOLERANCE:
                    diff = el.y - (target.y + target.height)
                    new_y -= diff
                    new_h += diff
                if (el.y + el.height) < target.y and abs((el.y + el.height) - target.y) <= SNAP_TOLERANCE:
                    new_h += target.y - (el.y + el.height)
        updated.append(el.model_copy(update={"x": new_x, "y": new_y, "width": new_w, "height": new_h}))
    return ParkingLayout(width=layout.width, height=layout.height, elements=updated, sceneId=layout.sceneId)


def post_process_layout(layout: ParkingLayout) -> ParkingLayout:
    processed = []
    for el in layout.elements:
        rx = int(round(el.x))
        ry = int(round(el.y))
        rw = int(round(el.width))
        rh = int(round(el.height))
        is_structural = el.type in (ElementType.ROAD, ElementType.GROUND, ElementType.WALL)
        pad = 1 if is_structural else 0
        cx = max(0, rx)
        cy = max(0, ry)
        cw = max(1, rw + pad)
        ch = max(1, rh + pad)
        if cx + cw > layout.width:
            cw = max(1, int(layout.width - cx))
        if cy + ch > layout.height:
            ch = max(1, int(layout.height - cy))
        processed.append(el.model_copy(update={"x": cx, "y": cy, "width": cw, "height": ch}))
    snap = snap_ground_to_boundaries(ParkingLayout(width=layout.width, height=layout.height, elements=processed, sceneId=layout.sceneId))
    return snap


def calculate_score(violations: Sequence[ConstraintViolation]) -> int:
    score = 0
    connectivity = 0
    for v in violations:
        if v.type == "overlap":
            score += 5
        elif v.type == "out_of_bounds":
            score += 8
        elif v.type == "placement_error":
            score += 4
        elif v.type == "connectivity_error":
            score += 12
            connectivity += 1
        else:
            score += 2
    score += max(0, connectivity - 1) * 5
    return score


def fill_parking_automatically(layout: ParkingLayout) -> ParkingLayout:
    existing = list(layout.elements)
    grounds = [e for e in existing if e.type == ElementType.GROUND]
    roads = [e for e in existing if e.type == ElementType.ROAD]
    obstacles = [
        e
        for e in existing
        if e.type
        in (
            ElementType.WALL,
            ElementType.STAIRCASE,
            ElementType.ELEVATOR,
            ElementType.PILLAR,
            ElementType.ENTRANCE,
            ElementType.EXIT,
            ElementType.RAMP,
            ElementType.SAFE_EXIT,
            ElementType.SIDEWALK,
            ElementType.PARKING_SPACE,
        )
    ]

    gen_spots: List[LayoutElement] = []
    SPOT_S = 24
    SPOT_L = 48
    GAP = 2
    BUFFER = 4
    TOLERANCE = 12

    def is_safe(rect: Dict[str, float]) -> bool:
        m = 1
        for o in obstacles:
            if rect["x"] + m < o.x + o.width and rect["x"] + rect["w"] - m > o.x and rect["y"] + m < o.y + o.height and rect["y"] + rect["h"] - m > o.y:
                return False
        for o in gen_spots:
            if rect["x"] + m < o.x + o.width and rect["x"] + rect["w"] - m > o.x and rect["y"] + m < o.y + o.height and rect["y"] + rect["h"] - m > o.y:
                return False
        return True

    t = 0
    for r in roads:
        rr = {"l": r.x, "r": r.x + r.width, "t": r.y, "b": r.y + r.height}
        for g in grounds:
            gr = {"l": g.x, "r": g.x + g.width, "t": g.y, "b": g.y + g.height}

            if abs(rr["b"] - gr["t"]) < TOLERANCE and min(rr["r"], gr["r"]) > max(rr["l"], gr["l"]):
                sx = max(rr["l"], gr["l"]) + BUFFER
                ex = min(rr["r"], gr["r"]) - BUFFER
                cnt = int(math.floor((ex - sx) / (SPOT_S + GAP)))
                for i in range(cnt):
                    s = {"x": sx + i * (SPOT_S + GAP), "y": gr["t"] + 1, "w": SPOT_S, "h": SPOT_L}
                    if is_safe(s):
                        t += 1
                        gen_spots.append(
                            LayoutElement(
                                id=f"p_auto_{t}",
                                type=ElementType.PARKING_SPACE,
                                x=s["x"],
                                y=s["y"],
                                width=s["w"],
                                height=s["h"],
                                rotation=0,
                                forward=(0.0, -1.0, 0.0),
                            )
                        )
            elif abs(rr["t"] - gr["b"]) < TOLERANCE and min(rr["r"], gr["r"]) > max(rr["l"], gr["l"]):
                sx = max(rr["l"], gr["l"]) + BUFFER
                ex = min(rr["r"], gr["r"]) - BUFFER
                cnt = int(math.floor((ex - sx) / (SPOT_S + GAP)))
                for i in range(cnt):
                    s = {"x": sx + i * (SPOT_S + GAP), "y": gr["b"] - SPOT_L - 1, "w": SPOT_S, "h": SPOT_L}
                    if is_safe(s):
                        t += 1
                        gen_spots.append(
                            LayoutElement(
                                id=f"p_auto_{t}",
                                type=ElementType.PARKING_SPACE,
                                x=s["x"],
                                y=s["y"],
                                width=s["w"],
                                height=s["h"],
                                rotation=0,
                                forward=(0.0, 1.0, 0.0),
                            )
                        )
            elif abs(rr["r"] - gr["l"]) < TOLERANCE and min(rr["b"], gr["b"]) > max(rr["t"], gr["t"]):
                sy = max(rr["t"], gr["t"]) + BUFFER
                ey = min(rr["b"], gr["b"]) - BUFFER
                cnt = int(math.floor((ey - sy) / (SPOT_S + GAP)))
                for i in range(cnt):
                    s = {"x": gr["l"] + 1, "y": sy + i * (SPOT_S + GAP), "w": SPOT_L, "h": SPOT_S}
                    if is_safe(s):
                        t += 1
                        gen_spots.append(
                            LayoutElement(
                                id=f"p_auto_v_{t}",
                                type=ElementType.PARKING_SPACE,
                                x=s["x"],
                                y=s["y"],
                                width=s["w"],
                                height=s["h"],
                                rotation=0,
                                forward=(-1.0, 0.0, 0.0),
                            )
                        )
            elif abs(rr["l"] - gr["r"]) < TOLERANCE and min(rr["b"], gr["b"]) > max(rr["t"], gr["t"]):
                sy = max(rr["t"], gr["t"]) + BUFFER
                ey = min(rr["b"], gr["b"]) - BUFFER
                cnt = int(math.floor((ey - sy) / (SPOT_S + GAP)))
                for i in range(cnt):
                    s = {"x": gr["r"] - SPOT_L - 1, "y": sy + i * (SPOT_S + GAP), "w": SPOT_L, "h": SPOT_S}
                    if is_safe(s):
                        t += 1
                        gen_spots.append(
                            LayoutElement(
                                id=f"p_auto_v_{t}",
                                type=ElementType.PARKING_SPACE,
                                x=s["x"],
                                y=s["y"],
                                width=s["w"],
                                height=s["h"],
                                rotation=0,
                                forward=(1.0, 0.0, 0.0),
                            )
                        )
    return ParkingLayout(width=layout.width, height=layout.height, elements=existing + gen_spots, sceneId=layout.sceneId)


def clean_intersections(layout: ParkingLayout) -> ParkingLayout:
    roads = [e for e in layout.elements if e.type == ElementType.ROAD]
    to_remove: Set[str] = set()
    for i in range(len(roads)):
        for j in range(i + 1, len(roads)):
            r1 = roads[i]
            r2 = roads[j]
            intersection = get_intersection_box(r1, r2)
            if intersection and intersection["width"] > 20 and intersection["height"] > 20:
                for el in layout.elements:
                    if el.id in to_remove:
                        continue
                    if el.type not in (ElementType.LANE_LINE, ElementType.PARKING_SPACE, ElementType.SPEED_BUMP, ElementType.GUIDANCE_SIGN):
                        continue
                    cx = el.x + el.width / 2
                    cy = el.y + el.height / 2
                    if cx > intersection["x"] and cx < intersection["x"] + intersection["width"] and cy > intersection["y"] and cy < intersection["y"] + intersection["height"]:
                        to_remove.add(el.id)
    if to_remove:
        return ParkingLayout(width=layout.width, height=layout.height, elements=[e for e in layout.elements if e.id not in to_remove], sceneId=layout.sceneId)
    return layout


def auto_remove_overlapping_spots(layout: ParkingLayout, threshold: float = 0.2) -> ParkingLayout:
    spots = [e for e in layout.elements if e.type == ElementType.PARKING_SPACE]
    blockers = [
        e
        for e in layout.elements
        if e.type
        in (
            ElementType.WALL,
            ElementType.ROAD,
            ElementType.PILLAR,
            ElementType.STAIRCASE,
            ElementType.ELEVATOR,
            ElementType.RAMP,
            ElementType.ENTRANCE,
            ElementType.EXIT,
            ElementType.CHARGING_STATION,
            ElementType.FIRE_EXTINGUISHER,
            ElementType.SAFE_EXIT,
        )
    ]
    if not spots or not blockers:
        return layout

    to_remove: Set[str] = set()
    updates: Dict[str, LayoutElement] = {}

    def compute_max_ratio(spot: LayoutElement) -> float:
        area = spot.width * spot.height
        if area <= 0:
            return 0
        max_ratio = 0.0
        for b in blockers:
            box = get_intersection_box(spot, b)
            if not box:
                continue
            overlap_area = box["width"] * box["height"]
            max_ratio = max(max_ratio, overlap_area / area)
        return max_ratio

    def try_shift(spot: LayoutElement) -> LayoutElement:
        base_ratio = compute_max_ratio(spot)
        if base_ratio <= threshold:
            return spot
        best = spot
        best_ratio = base_ratio
        steps = [4, 8, 12, 16, 20, 24]
        for step in steps:
            candidates = [
                {"x": spot.x + step, "y": spot.y},
                {"x": spot.x - step, "y": spot.y},
                {"x": spot.x, "y": spot.y + step},
                {"x": spot.x, "y": spot.y - step},
            ]
            for c in candidates:
                nx = min(max(0, c["x"]), layout.width - spot.width)
                ny = min(max(0, c["y"]), layout.height - spot.height)
                moved = spot.model_copy(update={"x": nx, "y": ny})
                ratio = compute_max_ratio(moved)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best = moved
                if best_ratio <= threshold:
                    return best
        return best

    for spot in spots:
        if spot.width * spot.height <= 0:
            continue
        max_ratio = compute_max_ratio(spot)
        if max_ratio > threshold:
            shifted = try_shift(spot)
            shifted_ratio = compute_max_ratio(shifted)
            if shifted_ratio > threshold:
                to_remove.add(spot.id)
            else:
                updates[spot.id] = shifted

    if not to_remove and not updates:
        return layout
    next_elements = []
    for e in layout.elements:
        if e.id in to_remove:
            continue
        if e.id in updates:
            next_elements.append(updates[e.id])
        else:
            next_elements.append(e)
    return ParkingLayout(width=layout.width, height=layout.height, elements=next_elements, sceneId=layout.sceneId)


def auto_snap_road_items(layout: ParkingLayout) -> ParkingLayout:
    roads = [e for e in layout.elements if e.type == ElementType.ROAD]
    if not roads:
        return layout
    items_on_road: Set[str] = {ElementType.GUIDANCE_SIGN, ElementType.LANE_LINE, ElementType.SPEED_BUMP, ElementType.SIDEWALK}
    updated: List[LayoutElement] = []
    for el in layout.elements:
        if el.type not in items_on_road:
            updated.append(el)
            continue
        best = None
        for r in roads:
            if el.x >= r.x - 5 and el.x + el.width <= r.x + r.width + 5 and el.y >= r.y - 5 and el.y + el.height <= r.y + r.height + 5:
                best = r
                break
        nearest = best
        if nearest is None:
            min_d = None
            cx = el.x + el.width / 2
            cy = el.y + el.height / 2
            for r in roads:
                rcx = r.x + r.width / 2
                rcy = r.y + r.height / 2
                d = abs(cx - rcx) + abs(cy - rcy)
                if min_d is None or d < min_d:
                    min_d = d
                    nearest = r
        if nearest is None:
            updated.append(el)
            continue
        cx = el.x + el.width / 2
        cy = el.y + el.height / 2
        if el.type == ElementType.GUIDANCE_SIGN:
            edge = 4
            left = nearest.x + edge
            right = nearest.x + nearest.width - edge
            top = nearest.y + edge
            bottom = nearest.y + nearest.height - edge
            dx = min(abs(cx - left), abs(cx - right))
            dy = min(abs(cy - top), abs(cy - bottom))
            nx = cx
            ny = cy
            if dx < dy:
                nx = left if abs(cx - left) < abs(cx - right) else right
            else:
                ny = top if abs(cy - top) < abs(cy - bottom) else bottom
            updated.append(el.model_copy(update={"x": int(round(nx - el.width / 2)), "y": int(round(ny - el.height / 2))}))
            continue
        if el.type == ElementType.LANE_LINE:
            is_horizontal = nearest.width >= nearest.height
            if is_horizontal:
                ny = nearest.y + nearest.height / 2 - el.height / 2
                updated.append(el.model_copy(update={"y": int(round(ny)), "x": int(round(nearest.x))}))
            else:
                nx = nearest.x + nearest.width / 2 - el.width / 2
                updated.append(el.model_copy(update={"x": int(round(nx)), "y": int(round(nearest.y))}))
            continue
        if el.type == ElementType.SPEED_BUMP:
            nx = min(max(cx, nearest.x + 4), nearest.x + nearest.width - 4)
            ny = min(max(cy, nearest.y + 4), nearest.y + nearest.height - 4)
            updated.append(el.model_copy(update={"x": int(round(nx - el.width / 2)), "y": int(round(ny - el.height / 2))}))
            continue
        nx = min(max(cx, nearest.x + 2), nearest.x + nearest.width - 2)
        ny = min(max(cy, nearest.y + 2), nearest.y + nearest.height - 2)
        updated.append(el.model_copy(update={"x": int(round(nx - el.width / 2)), "y": int(round(ny - el.height / 2))}))
    return ParkingLayout(width=layout.width, height=layout.height, elements=updated, sceneId=layout.sceneId)


def generate_charging_stations(layout: ParkingLayout) -> ParkingLayout:
    spots = [e for e in layout.elements if e.type == ElementType.PARKING_SPACE]
    roads = [e for e in layout.elements if e.type == ElementType.ROAD]
    stations: List[LayoutElement] = []
    sorted_spots = sorted(spots, key=lambda a: (a.y if abs(a.y) > -1 else 0, a.x))
    station_count = 0
    STATION_SIZE = 10
    OFFSET = 2
    for idx, spot in enumerate(sorted_spots):
        if (idx + 1) % 3 != 0:
            continue
        forward = spot.forward
        if forward is None:
            forward = infer_parking_forward(spot, roads)
        if forward is None:
            continue
        dx, dy, _ = forward
        cx = 0.0
        cy = 0.0
        if abs(dx) < 0.1 and dy < -0.9:
            cx = spot.x + spot.width / 2 - STATION_SIZE / 2
            cy = spot.y + spot.height - STATION_SIZE - OFFSET
        elif abs(dx) < 0.1 and dy > 0.9:
            cx = spot.x + spot.width / 2 - STATION_SIZE / 2
            cy = spot.y + OFFSET
        elif dx < -0.9 and abs(dy) < 0.1:
            cx = spot.x + spot.width - STATION_SIZE - OFFSET
            cy = spot.y + spot.height / 2 - STATION_SIZE / 2
        elif dx > 0.9 and abs(dy) < 0.1:
            cx = spot.x + OFFSET
            cy = spot.y + spot.height / 2 - STATION_SIZE / 2
        if cx != 0 or cy != 0:
            station_count += 1
            stations.append(
                LayoutElement(
                    id=f"charging_{station_count}",
                    type=ElementType.CHARGING_STATION,
                    x=cx,
                    y=cy,
                    width=STATION_SIZE,
                    height=STATION_SIZE,
                    rotation=0,
                )
            )
    return ParkingLayout(width=layout.width, height=layout.height, elements=list(layout.elements) + stations, sceneId=layout.sceneId)


def cleanup_pillars(layout: ParkingLayout) -> ParkingLayout:
    roads = [e for e in layout.elements if e.type == ElementType.ROAD]
    spots = [e for e in layout.elements if e.type == ElementType.PARKING_SPACE]
    kept: List[LayoutElement] = []
    for el in layout.elements:
        if el.type != ElementType.PILLAR:
            kept.append(el)
            continue
        is_on_road = any(el.x < r.x + r.width and el.x + el.width > r.x and el.y < r.y + r.height and el.y + el.height > r.y for r in roads)
        is_inside_spot = any(el.x > s.x + 2 and el.x + el.width < s.x + s.width - 2 and el.y > s.y + 2 and el.y + el.height < s.y + s.height - 2 for s in spots)
        if not is_on_road and not is_inside_spot:
            kept.append(el)
    return ParkingLayout(width=layout.width, height=layout.height, elements=kept, sceneId=layout.sceneId)


def resolve_priority_conflicts(elements: Sequence[LayoutElement]) -> List[LayoutElement]:
    sidewalks = [e for e in elements if e.type == ElementType.SIDEWALK]
    out: List[LayoutElement] = []
    for el in elements:
        if el.type == ElementType.SPEED_BUMP:
            has_conflict = False
            for s in sidewalks:
                intersection = get_intersection_box(el, s)
                if intersection and (intersection["width"] > 2 or intersection["height"] > 2):
                    has_conflict = True
                    break
            if not has_conflict:
                out.append(el)
        else:
            out.append(el)
    return out


def orient_guidance_signs(layout: ParkingLayout) -> ParkingLayout:
    exits = [e for e in layout.elements if e.type == ElementType.EXIT]
    roads = [e for e in layout.elements if e.type == ElementType.ROAD]
    if not exits:
        return layout
    updated: List[LayoutElement] = []
    for el in layout.elements:
        if el.type != ElementType.GUIDANCE_SIGN:
            updated.append(el)
            continue
        parent_road = None
        for r in roads:
            if el.x >= r.x - 5 and el.x + el.width <= r.x + r.width + 5 and el.y >= r.y - 5 and el.y + el.height <= r.y + r.height + 5:
                parent_road = r
                break
        nearest_exit = exits[0]
        min_dist = None
        scx = el.x + el.width / 2
        scy = el.y + el.height / 2
        for ex in exits:
            ecx = ex.x + ex.width / 2
            ecy = ex.y + ex.height / 2
            d = abs(ecx - scx) + abs(ecy - scy)
            if min_dist is None or d < min_dist:
                min_dist = d
                nearest_exit = ex
        ecx = nearest_exit.x + nearest_exit.width / 2
        ecy = nearest_exit.y + nearest_exit.height / 2
        rot = el.rotation or 0
        if parent_road is not None:
            is_horizontal = parent_road.width > parent_road.height
            if is_horizontal:
                rot = 0 if ecx > scx else 180
            else:
                rot = 90 if ecy > scy else 270
        else:
            dx = ecx - scx
            dy = ecy - scy
            if abs(dx) > abs(dy):
                rot = 0 if dx > 0 else 180
            else:
                rot = 90 if dy > 0 else 270
        updated.append(el.model_copy(update={"rotation": rot}))
    return ParkingLayout(width=layout.width, height=layout.height, elements=updated, sceneId=layout.sceneId)


def generate_auto_connectivity_patches(layout: ParkingLayout) -> List[Dict[str, Any]]:
    violations = validate_layout(layout)
    ramps = [e for e in layout.elements if e.type == ElementType.RAMP]
    roads = [e for e in layout.elements if e.type == ElementType.ROAD]
    gates = [e for e in layout.elements if e.type == ElementType.ENTRANCE or e.type == ElementType.EXIT]
    patches: List[Dict[str, Any]] = []

    def add_ramp_for_gate(gate: LayoutElement) -> None:
        w, h = 40, 60
        rx = gate.x + gate.width / 2 - w / 2
        ry = gate.y
        if gate.y <= 5:
            ry = gate.y + gate.height
        elif gate.y + gate.height >= layout.height - 5:
            ry = gate.y - h
        elif gate.x <= 5:
            rx = gate.x + gate.width
        elif gate.x + gate.width >= layout.width - 5:
            rx = gate.x - w
        patches.append({"t": "RAMP", "type": ElementType.RAMP, "x": int(round(rx)), "y": int(round(ry)), "w": w, "h": h})

    def touch_road(ramp: LayoutElement) -> None:
        best = None
        for rd in roads:
            cx1 = ramp.x + ramp.width / 2
            cy1 = ramp.y + ramp.height / 2
            cx2 = rd.x + rd.width / 2
            cy2 = rd.y + rd.height / 2
            d = abs(cx1 - cx2) + abs(cy1 - cy2)
            if best is None or d < (abs(best["dx"]) + abs(best["dy"])):
                best = {"road": rd, "dx": cx2 - cx1, "dy": cy2 - cy1}
        if best is None:
            return
        x, y, w, h = ramp.x, ramp.y, ramp.width, ramp.height
        if abs(best["dx"]) > abs(best["dy"]):
            if best["dx"] > 0:
                w = max(w, best["road"].x - ramp.x)
            else:
                w = max(w, ramp.x + ramp.width - (best["road"].x + best["road"].width))
        else:
            if best["dy"] > 0:
                h = max(h, best["road"].y - ramp.y)
            else:
                h = max(h, ramp.y + ramp.height - (best["road"].y + best["road"].height))
        patches.append({"id": ramp.id, "t": "RAMP", "type": ElementType.RAMP, "x": x, "y": y, "w": w, "h": h})

    for v in violations:
        if v.type == "connectivity_error" and "needs Ramp" in v.message:
            gate = next((g for g in gates if g.id == v.elementId), None)
            if gate is not None:
                add_ramp_for_gate(gate)
        if v.type == "connectivity_error" and "Ramp disconnected" in v.message:
            ramp = next((r for r in ramps if r.id == v.elementId), None)
            if ramp is not None:
                touch_road(ramp)
    return patches


def fix_small_geometry(layout: ParkingLayout) -> ParkingLayout:
    min_size = 4
    updated = [el.model_copy(update={"width": max(min_size, int(round(el.width or 0))), "height": max(min_size, int(round(el.height or 0)))}) for el in layout.elements]
    return ParkingLayout(width=layout.width, height=layout.height, elements=updated, sceneId=layout.sceneId)


def fill_voids_with_ground(layout: ParkingLayout) -> ParkingLayout:
    clean_elements = [el for el in layout.elements if not el.id.startswith("auto_ground_void_")]
    step = 10
    width = max(1, int(round(layout.width)))
    height = max(1, int(round(layout.height)))
    cols = max(1, int(math.ceil(width / step)))
    rows = max(1, int(math.ceil(height / step)))
    occupied = [False] * (rows * cols)

    solid_types: Set[str] = {
        ElementType.WALL,
        ElementType.ROAD,
        ElementType.RAMP,
        ElementType.ENTRANCE,
        ElementType.EXIT,
        ElementType.GROUND,
        ElementType.STAIRCASE,
        ElementType.ELEVATOR,
        ElementType.PILLAR,
    }

    def mark_occupied(el: LayoutElement) -> None:
        x1 = max(0, int(math.floor(el.x / step)))
        y1 = max(0, int(math.floor(el.y / step)))
        x2 = min(cols - 1, int(math.floor((el.x + el.width - 1) / step)))
        y2 = min(rows - 1, int(math.floor((el.y + el.height - 1) / step)))
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                occupied[y * cols + x] = True

    for el in clean_elements:
        t = normalize_type(el.type)
        if t not in solid_types:
            continue
        mark_occupied(el)

    segments_by_row: List[List[Dict[str, int]]] = [[] for _ in range(rows)]
    for y in range(rows):
        segs: List[Dict[str, int]] = []
        start = -1
        for x in range(cols):
            is_empty = not occupied[y * cols + x]
            if is_empty and start == -1:
                start = x
            if (not is_empty) and start != -1:
                segs.append({"x1": start, "x2": x - 1})
                start = -1
        if start != -1:
            segs.append({"x1": start, "x2": cols - 1})
        segments_by_row[y] = segs

    rects: List[Dict[str, int]] = []
    prev: Dict[str, Dict[str, int]] = {}
    for y in range(rows):
        nxt: Dict[str, Dict[str, int]] = {}
        for seg in segments_by_row[y]:
            key = f"{seg['x1']}-{seg['x2']}"
            existing = prev.get(key)
            if existing is not None:
                existing["y2"] = y
                nxt[key] = existing
            else:
                nxt[key] = {"x1": seg["x1"], "x2": seg["x2"], "y1": y, "y2": y}
        for key, r in prev.items():
            if key not in nxt:
                rects.append(r)
        prev = nxt
    rects.extend(list(prev.values()))

    ts = int(time.time() * 1000)
    new_grounds: List[LayoutElement] = []
    for i, r in enumerate(rects):
        gx = r["x1"] * step
        gy = r["y1"] * step
        gw = min(width - gx, (r["x2"] - r["x1"] + 1) * step)
        gh = min(height - gy, (r["y2"] - r["y1"] + 1) * step)
        if gw >= 5 and gh >= 5:
            new_grounds.append(
                LayoutElement(
                    id=f"auto_ground_void_{ts}_{i}",
                    type=ElementType.GROUND,
                    x=gx,
                    y=gy,
                    width=gw,
                    height=gh,
                    rotation=0,
                )
            )

    if not new_grounds:
        return ParkingLayout(width=layout.width, height=layout.height, elements=clean_elements, sceneId=layout.sceneId)
    return ParkingLayout(width=layout.width, height=layout.height, elements=new_grounds + clean_elements, sceneId=layout.sceneId)


async def enhance_layout_with_geometry(layout: ParkingLayout, on_log: Optional[Callable[[str], None]] = None) -> ParkingLayout:
    current = layout
    if on_log:
        on_log("📐 执行几何填充算法...")
    current = fill_parking_automatically(current)
    current = auto_remove_overlapping_spots(current, 0.2)
    if on_log:
        on_log("🧹 清理交叉口...")
    current = clean_intersections(current)
    if on_log:
        on_log("⚡ 生成充电桩...")
    current = generate_charging_stations(current)
    if on_log:
        on_log("🧹 清理非法柱子...")
    current = cleanup_pillars(current)
    current = clean_intersections(current)
    if on_log:
        on_log("⚖️ 解决优先级冲突...")
    current = ParkingLayout(
        width=current.width,
        height=current.height,
        elements=resolve_priority_conflicts(current.elements),
        sceneId=current.sceneId,
    )
    if on_log:
        on_log("🧭 调整指示牌方向...")
    current = orient_guidance_signs(current)
    current = auto_snap_road_items(current)
    return current


def apply_scene_post_process(layout: ParkingLayout, scene: Any, on_log: Optional[Callable[[str], None]] = None) -> ParkingLayout:
    processed = post_process_layout(layout)
    algos = scene.postProcessAlgorithms if hasattr(scene, "postProcessAlgorithms") else None
    if isinstance(algos, list):
        for algo in algos:
            try:
                processed = algo(processed)
            except Exception as e:
                if on_log:
                    on_log(f"⚠️ 后处理算法失败: {str(e)}")
    return processed

