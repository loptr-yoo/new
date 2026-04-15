from __future__ import annotations

from typing import List

from .models import FurnitureCategory, FurnitureSpec


def furnitures_for_room(room_type: str) -> List[FurnitureSpec]:
    t = (room_type or "").strip().lower()

    if t in {"corridor", "hallway", "passage"}:
        return []

    if t in {"bedroom", "master_bedroom"}:
        return [
            FurnitureSpec(id="bed_1", name="床", category=FurnitureCategory.BEDDING, width=1.8, height=2.0),
            FurnitureSpec(id="wardrobe_1", name="衣柜", category=FurnitureCategory.CABINET, width=1.5, height=0.6),
            FurnitureSpec(id="nightstand_1", name="床头柜", category=FurnitureCategory.CABINET, width=0.5, height=0.5),
        ]

    if t in {"living_room", "living", "reception"}:
        return [
            FurnitureSpec(id="sofa_1", name="沙发", category=FurnitureCategory.SEATING, width=2.2, height=0.9),
            FurnitureSpec(id="coffee_table_1", name="茶几", category=FurnitureCategory.TABLE, width=1.2, height=0.6),
            FurnitureSpec(id="tv_stand_1", name="电视柜", category=FurnitureCategory.CABINET, width=1.6, height=0.4),
        ]

    if t in {"kitchen"}:
        return [
            FurnitureSpec(id="fridge_1", name="冰箱", category=FurnitureCategory.APPLIANCE, width=0.8, height=0.8),
            FurnitureSpec(id="counter_1", name="灶台/橱柜", category=FurnitureCategory.CABINET, width=1.8, height=0.6),
        ]

    if t in {"bathroom", "toilet"}:
        return [
            FurnitureSpec(id="toilet_1", name="马桶", category=FurnitureCategory.APPLIANCE, width=0.7, height=0.7),
            FurnitureSpec(id="sink_1", name="洗手台", category=FurnitureCategory.CABINET, width=0.9, height=0.5),
        ]

    if t in {"dining_room", "dining"}:
        return [
            FurnitureSpec(id="dining_table_1", name="餐桌", category=FurnitureCategory.TABLE, width=1.4, height=0.8),
            FurnitureSpec(id="chair_1", name="椅子", category=FurnitureCategory.CHAIR, width=0.45, height=0.45),
            FurnitureSpec(id="chair_2", name="椅子", category=FurnitureCategory.CHAIR, width=0.45, height=0.45),
            FurnitureSpec(id="chair_3", name="椅子", category=FurnitureCategory.CHAIR, width=0.45, height=0.45),
            FurnitureSpec(id="chair_4", name="椅子", category=FurnitureCategory.CHAIR, width=0.45, height=0.45),
        ]

    if t in {"study"}:
        return [
            FurnitureSpec(id="desk_1", name="书桌", category=FurnitureCategory.TABLE, width=1.2, height=0.6),
            FurnitureSpec(id="chair_1", name="椅子", category=FurnitureCategory.CHAIR, width=0.45, height=0.45),
            FurnitureSpec(id="bookshelf_1", name="书柜", category=FurnitureCategory.CABINET, width=1.2, height=0.35),
        ]

    if t in {"storage", "laundry", "utility"}:
        return [
            FurnitureSpec(id="cabinet_1", name="柜子", category=FurnitureCategory.CABINET, width=1.0, height=0.5),
        ]

    if t in {"entrance"}:
        return [
            FurnitureSpec(id="shoe_cabinet_1", name="鞋柜", category=FurnitureCategory.CABINET, width=1.0, height=0.35),
        ]

    return [
        FurnitureSpec(id="table_1", name="桌子", category=FurnitureCategory.TABLE, width=0.8, height=0.8),
    ]
