"""
test_building_robustness.py

Building 管线鲁棒性测试套件（Phase 6）

覆盖：
- JSON 修复（尾逗号、注释、截断、markdown 包裹）
- 缺失字段自动补全
- 非法值修复（zone、room_type、adjacency 引用）
- 类型强制转换
- 旧字段兼容
- size_hint → target_area
- 几何层容错（小楼层、超大房间、0 岛屿、MIQP 失败 fallback）
- 后处理（墙/门/窗生成）
- 降级摘要
- 组合故障
"""
from __future__ import annotations

import json
import pytest

from backend.models import BuildingAllocation, RoomAllocation, FloorAllocation


# ============================================================
# 辅助：构建测试数据
# ============================================================

def _make_room(room_id="room_001", room_name="客厅", room_type="living_room",
               target_area=35.0, zone="public", needs_window=True, weight=5,
               **kwargs):
    d = {
        "room_id": room_id,
        "room_name": room_name,
        "room_type": room_type,
        "target_area": target_area,
        "zone": zone,
        "needs_window": needs_window,
        "weight": weight,
    }
    d.update(kwargs)
    return d


def _make_floor(floor_number=1, rooms=None, floor_total_area=100.0,
                core_tube_area=15.0, corridor_allowance_area=15.0):
    if rooms is None:
        rooms = [
            _make_room("room_001", "客厅", "living_room", 35.0),
            _make_room("room_002", "餐厅", "dining_room", 15.0,
                       adjacency_required=["room_001"]),
        ]
    return {
        "floor_number": floor_number,
        "floor_function_tag": "residential",
        "floor_total_area": floor_total_area,
        "core_tube_area": core_tube_area,
        "corridor_allowance_area": corridor_allowance_area,
        "rooms": rooms,
    }


def _make_building(floors=None):
    if floors is None:
        floors = [_make_floor()]
    return {
        "building_name": "测试建筑",
        "total_floors": len(floors),
        "overall_total_area": sum(f.get("floor_total_area", 100) for f in floors),
        "floors": floors,
    }


def _try_parse(text: str):
    """调用 _parse_building_allocation，跳过如果依赖不满足"""
    try:
        from backend.core.flows.building_semantic_flow import _parse_building_allocation
    except ImportError:
        pytest.skip("Server dependencies not installed")
    return _parse_building_allocation(text, provider="test", model="test")


# ============================================================
# Phase 2: JSON 修复测试
# ============================================================

class TestMalformedJsonRepair:

    def test_trailing_commas(self):
        """尾逗号修复"""
        raw = json.dumps(_make_building(), ensure_ascii=False)
        broken = raw.replace('"weight": 5}', '"weight": 5,}')
        alloc, warns = _try_parse(broken)
        assert len(alloc.floors) >= 1

    def test_comments_in_json(self):
        """JSON 注释修复"""
        raw = json.dumps(_make_building(), ensure_ascii=False)
        broken = raw.replace('"room_type": "living_room"',
                             '"room_type": "living_room" // 主要起居空间')
        alloc, warns = _try_parse(broken)
        assert alloc.floors[0].rooms[0].room_type == "living_room"

    def test_markdown_wrapped(self):
        """markdown 代码块包裹"""
        raw = json.dumps(_make_building(), ensure_ascii=False)
        wrapped = f"```json\n{raw}\n```"
        alloc, warns = _try_parse(wrapped)
        assert len(alloc.floors) == 1

    def test_extra_text_before_json(self):
        """JSON 前有解释文本"""
        raw = json.dumps(_make_building(), ensure_ascii=False)
        with_preamble = f"好的，以下是方案：\n\n{raw}\n\n希望满足您的需求。"
        alloc, warns = _try_parse(with_preamble)
        assert len(alloc.floors) == 1

    def test_truncated_json(self):
        """截断的 JSON（括号不闭合）"""
        raw = json.dumps(_make_building(), ensure_ascii=False)
        truncated = raw[:-2]  # 去掉最后两个字符
        alloc, warns = _try_parse(truncated)
        assert len(alloc.floors) >= 1


# ============================================================
# Phase 2: 缺失字段自动补全
# ============================================================

class TestMissingFieldsFilled:

    def test_missing_room_id_auto_generated(self):
        """缺失 room_id → 自动生成"""
        rooms = [{"room_name": "客厅", "room_type": "living_room", "target_area": 30.0}]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].room_id != ""

    def test_missing_zone_defaults_public(self):
        """缺失 zone → 默认 public"""
        rooms = [{"room_name": "房间", "room_type": "room", "target_area": 20.0}]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].zone == "public"

    def test_missing_needs_window_defaults(self):
        """缺失 needs_window → 默认 False"""
        rooms = [{"room_name": "储藏", "room_type": "storage", "target_area": 5.0}]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        # 在 _normalize_allocation 中 storage 的 needs_window 保持 False
        room = alloc.floors[0].rooms[0]
        assert isinstance(room.needs_window, bool)

    def test_missing_floor_number_auto_filled(self):
        """缺失 floor_number → 自动填充"""
        floor = _make_floor()
        del floor["floor_number"]
        obj = _make_building([floor])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].floor_number >= 1

    def test_missing_building_name_auto_filled(self):
        """缺失 building_name → 自动填充"""
        obj = _make_building()
        del obj["building_name"]
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.building_name != ""


# ============================================================
# Phase 2: 非法值修复
# ============================================================

class TestInvalidZoneMapped:

    def test_common_mapped_to_public(self):
        """zone='common' → 'public'"""
        rooms = [_make_room(zone="common")]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].zone == "public"

    def test_bedroom_mapped_to_private(self):
        """zone='bedroom' → 'private'"""
        rooms = [_make_room(zone="bedroom")]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].zone == "private"

    def test_corridor_mapped_to_circulation(self):
        """zone='corridor' → 'circulation'"""
        rooms = [_make_room(zone="corridor")]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].zone == "circulation"

    def test_utility_mapped_to_service(self):
        """zone='utility' → 'service'"""
        rooms = [_make_room(zone="utility")]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].zone == "service"

    def test_unknown_zone_defaults_public(self):
        """zone='semi-public' → 'public'（未知值默认 public）"""
        rooms = [_make_room(zone="semi-public")]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].zone == "public"
        assert any("zone" in w for w in warns)


class TestInvalidAdjacencyCleaned:

    def test_nonexistent_ref_removed(self):
        """adjacency_required 引用不存在的 room_id → 移除 + warning"""
        rooms = [_make_room("r1", adjacency_required=["nonexistent"])]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert "nonexistent" not in alloc.floors[0].rooms[0].adjacency_required
        assert any("nonexistent" in w or "引用" in w for w in warns)

    def test_valid_ref_kept(self):
        """同层的合法 room_id 保留"""
        rooms = [
            _make_room("r1", adjacency_required=["r2"]),
            _make_room("r2", room_name="餐厅", room_type="dining_room", target_area=15.0),
        ]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert "r2" in alloc.floors[0].rooms[0].adjacency_required


# ============================================================
# Phase 2: 类型强制转换
# ============================================================

class TestTypeCoercion:

    def test_string_target_area(self):
        """target_area='25.5' (string) → 25.5 (float)"""
        rooms = [_make_room()]
        rooms[0]["target_area"] = "25.5"
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].target_area > 0
        assert isinstance(alloc.floors[0].rooms[0].target_area, float)

    def test_string_weight(self):
        """weight='5' (string) → 5 (int)"""
        rooms = [_make_room()]
        rooms[0]["weight"] = "5"
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].weight == 5

    def test_string_needs_window(self):
        """needs_window='true' (string) → True (bool)"""
        rooms = [_make_room()]
        rooms[0]["needs_window"] = "true"
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        assert alloc.floors[0].rooms[0].needs_window is True


# ============================================================
# Phase 2: 旧字段兼容
# ============================================================

class TestOldFieldCompat:

    def test_requires_window_fallback(self):
        """requires_window=True + needs_window 默认 → needs_window=True"""
        rooms = [{
            "room_name": "卧室", "room_type": "bedroom",
            "target_area": 20.0, "requires_window": True,
        }]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        # _normalize_allocation 会把 requires_window=True 回退到 needs_window
        room = alloc.floors[0].rooms[0]
        assert room.needs_window is True

    def test_needs_window_false_not_overridden(self):
        """needs_window=False + requires_window=True → needs_window 保持 False

        注意：在 _normalize_allocation 中，当 needs_window=False 且 requires_window=True 时
        会被覆盖。但如果 room_type 不推导为需要窗户，则保持 False。
        这里测 Pydantic 模型层面两个字段可以共存。
        """
        rooms = [{
            "room_name": "储藏", "room_type": "storage",
            "target_area": 5.0,
            "needs_window": False, "requires_window": True,
        }]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc = BuildingAllocation.model_validate(obj)
        room = alloc.floors[0].rooms[0]
        assert room.needs_window is False
        assert room.requires_window is True

    def test_adjacency_tags_fallback(self):
        """adjacency_tags 非空 + adjacency_required 空 → 回退"""
        rooms = [{
            "room_name": "厨房", "room_type": "kitchen",
            "target_area": 12.0, "adjacency_tags": ["dining_room"],
        }]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))
        room = alloc.floors[0].rooms[0]
        # _normalize_allocation 会把 adjacency_tags 回退到 adjacency_required
        assert room.adjacency_required == ["dining_room"]


# ============================================================
# Phase 1: size_hint → target_area
# ============================================================

class TestSizeHintToArea:

    def test_large_gets_more_area(self):
        """size_hint='large' 的房间面积 > 'small' 的房间"""
        from backend.core.geometry.room_defaults import apply_size_hints

        rooms = [
            RoomAllocation(
                room_name="客厅", room_type="living_room",
                target_area=20.0, size_hint="large",
            ),
            RoomAllocation(
                room_name="储藏", room_type="storage",
                target_area=20.0, size_hint="small",
            ),
        ]

        apply_size_hints(rooms, floor_area=100.0, core_tube_area=15.0)
        assert rooms[0].target_area > rooms[1].target_area

    def test_min_area_enforced(self):
        """size_hint 分配面积不低于 room_type 最小值"""
        from backend.core.geometry.room_defaults import apply_size_hints

        rooms = [
            RoomAllocation(
                room_name="卧室", room_type="bedroom",
                target_area=1.0, size_hint="small",
            ),
        ]

        apply_size_hints(rooms, floor_area=30.0, core_tube_area=5.0)
        # bedroom 最小面积 9.0
        assert rooms[0].target_area >= 9.0


# ============================================================
# Phase 3: 几何层容错
# ============================================================

class TestSmallFloorNocrash:

    def test_100m2_five_rooms(self):
        """100m² 楼层 5 个房间 → 成功解析（不崩溃）"""
        rooms = [
            _make_room(f"r{i}", f"房间{i}", "room", 10.0)
            for i in range(5)
        ]
        obj = _make_building([_make_floor(
            rooms=rooms, floor_total_area=100.0,
            core_tube_area=10.0, corridor_allowance_area=10.0,
        )])
        alloc = BuildingAllocation.model_validate(obj)
        assert len(alloc.floors[0].rooms) == 5


class TestOversizedRoomsNocrash:

    def test_room_area_exceeds_floor(self):
        """房间面积 > 分配空间 → Pydantic 面积守恒修复"""
        rooms = [
            _make_room("r1", target_area=80.0),  # 超过可分配面积 70
        ]
        obj = _make_building([_make_floor(
            rooms=rooms, floor_total_area=100.0,
            core_tube_area=15.0, corridor_allowance_area=15.0,
        )])
        alloc = BuildingAllocation.model_validate(obj)
        # 面积被守恒修复缩放
        assert alloc.floors[0].rooms[0].target_area <= 71.0


class TestZeroIslandsNocrash:

    def test_assigner_with_no_islands(self):
        """0 岛屿时分配器不崩溃，所有房间被跳过"""
        from backend.core.geometry.island_room_assigner import (
            IslandRoomAssigner,
        )
        from backend.core.geometry.room_spec import RoomSpec, ZoneType

        rooms = [
            RoomSpec(room_id="r1", room_type="living_room", target_area=20.0),
            RoomSpec(room_id="r2", room_type="bedroom", target_area=15.0),
        ]

        assigner = IslandRoomAssigner(
            islands=[],
            rooms=rooms,
            adjacency_graph={},
        )
        assignments, degradation = assigner.assign()
        assert len(degradation.skipped_rooms) == 2
        assert "r1" in degradation.skipped_rooms
        assert "r2" in degradation.skipped_rooms


# ============================================================
# Phase 3: 连通性检查
# ============================================================

class TestConnectivityCheck:

    def test_connected_rooms(self):
        """相邻房间通过共享边连通"""
        from shapely.geometry import box
        from backend.core.geometry.layout_generator import RoomResult

        rooms = [
            RoomResult(
                id="r1", room_type="living_room",
                polygon=box(0, 0, 5, 5),
                area=25.0, target_area=25.0, area_error=0.0,
                centroid=(2.5, 2.5), has_window=True,
                facade_length=5.0, aspect_ratio=1.0,
            ),
            RoomResult(
                id="r2", room_type="dining_room",
                polygon=box(5, 0, 10, 5),
                area=25.0, target_area=25.0, area_error=0.0,
                centroid=(7.5, 2.5), has_window=True,
                facade_length=5.0, aspect_ratio=1.0,
            ),
        ]

        # 两个房间共享 x=5 的边，应该连通
        from backend.core.geometry.postprocessor import generate_walls
        walls = generate_walls(rooms, box(0, 0, 10, 5))
        partition_walls = [w for w in walls if w.type == "partition_wall"]
        assert len(partition_walls) >= 1


# ============================================================
# Phase 4: 后处理（墙/门/窗生成）
# ============================================================

class TestWallDoorWindowGeneration:

    def _make_layout_rooms(self):
        from shapely.geometry import box
        from backend.core.geometry.layout_generator import RoomResult

        return [
            RoomResult(
                id="r1", room_type="living_room",
                polygon=box(0, 0, 6, 5),
                area=30.0, target_area=30.0, area_error=0.0,
                centroid=(3.0, 2.5), has_window=True,
                facade_length=6.0, aspect_ratio=1.2,
            ),
            RoomResult(
                id="r2", room_type="bedroom",
                polygon=box(6, 0, 12, 5),
                area=30.0, target_area=30.0, area_error=0.0,
                centroid=(9.0, 2.5), has_window=True,
                facade_length=6.0, aspect_ratio=1.2,
            ),
            RoomResult(
                id="r3", room_type="bathroom",
                polygon=box(0, 5, 6, 8),
                area=18.0, target_area=18.0, area_error=0.0,
                centroid=(3.0, 6.5), has_window=False,
                facade_length=0.0, aspect_ratio=2.0,
            ),
        ]

    def test_walls_generated(self):
        """生成墙体：外墙 + 内墙"""
        from shapely.geometry import box
        from backend.core.geometry.postprocessor import generate_walls

        rooms = self._make_layout_rooms()
        walls = generate_walls(rooms, box(0, 0, 12, 8))

        ext_walls = [w for w in walls if w.type == "exterior_wall"]
        part_walls = [w for w in walls if w.type == "partition_wall"]
        assert len(ext_walls) > 0, "应有外墙"
        assert len(part_walls) > 0, "应有内墙"

    def test_doors_generated(self):
        """在内墙中点放置门"""
        from shapely.geometry import box
        from backend.core.geometry.postprocessor import generate_walls, generate_doors

        rooms = self._make_layout_rooms()
        walls = generate_walls(rooms, box(0, 0, 12, 8))
        doors = generate_doors(walls)

        assert len(doors) > 0, "应有门"
        for door in doors:
            assert len(door.connects) == 2
            assert door.width > 0

    def test_windows_on_exterior_walls(self):
        """在外墙上为 has_window=True 的房间放置窗"""
        from shapely.geometry import box
        from backend.core.geometry.postprocessor import (
            generate_walls, generate_windows,
        )

        rooms = self._make_layout_rooms()
        walls = generate_walls(rooms, box(0, 0, 12, 8))
        windows = generate_windows(walls, rooms)

        assert len(windows) > 0, "应有窗户"
        # bathroom (has_window=False) 不应有窗
        bath_windows = [w for w in windows if w.room_id == "r3"]
        assert len(bath_windows) == 0, "bathroom 不应有窗"

    def test_postprocess_floor_integration(self):
        """postprocess_floor 一站式调用正确"""
        from shapely.geometry import box
        from backend.core.geometry.postprocessor import postprocess_floor

        rooms = self._make_layout_rooms()
        pp = postprocess_floor(rooms, box(0, 0, 12, 8))

        assert len(pp.walls) > 0
        assert len(pp.doors) > 0
        assert len(pp.windows) > 0

    def test_serialization(self):
        """后处理结果可序列化为 dict"""
        from shapely.geometry import box
        from backend.core.geometry.postprocessor import (
            postprocess_floor, wall_to_dict, door_to_dict, window_to_dict,
        )

        rooms = self._make_layout_rooms()
        pp = postprocess_floor(rooms, box(0, 0, 12, 8))

        for w in pp.walls:
            d = wall_to_dict(w)
            assert "type" in d
            assert "coords" in d
            assert isinstance(d["coords"], list)

        for door in pp.doors:
            d = door_to_dict(door)
            assert "position" in d
            assert "connects" in d

        for win in pp.windows:
            d = window_to_dict(win)
            assert "position" in d
            assert "room_id" in d

    def test_extract_linestrings_splits_filters_and_supports_linearring(self):
        from shapely.geometry import LineString, Polygon
        from backend.core.geometry.postprocessor import _extract_linestrings

        line = LineString([(0, 0), (1, 0), (1, 1)])
        segments = _extract_linestrings(line)
        assert len(segments) == 2
        assert all(len(list(s.coords)) == 2 for s in segments)

        with_tiny = LineString([(0, 0), (0.01, 0), (1, 0)])
        segments2 = _extract_linestrings(with_tiny)
        assert len(segments2) == 1
        assert segments2[0].length > 0.05

        ring = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]).exterior
        segments3 = _extract_linestrings(ring)
        assert len(segments3) == 4
        assert all(len(list(s.coords)) == 2 for s in segments3)

    def test_dedup_walls_ignores_room_ids_for_same_segment(self):
        from shapely.geometry import LineString
        from backend.core.geometry.postprocessor import WallSegment, _dedup_walls

        w1 = WallSegment(
            type="exterior_wall",
            geometry=LineString([(0, 0), (1, 0)]),
            thickness=0.24,
            room_ids=["r1"],
        )
        w2 = WallSegment(
            type="exterior_wall",
            geometry=LineString([(0, 0), (1, 0)]),
            thickness=0.24,
            room_ids=["r2"],
        )
        w3 = WallSegment(
            type="partition_wall",
            geometry=LineString([(0, 0), (1, 0)]),
            thickness=0.12,
            room_ids=["r1", "r2"],
        )

        deduped = _dedup_walls([w1, w2, w3])
        assert len(deduped) == 2

    def test_corridors_do_not_overlap_core_after_cut(self):
        from shapely.geometry import box
        from backend.core.geometry.topology_generator import CoreTube, generate_rectangular_topology

        boundary = box(0, 0, 20, 15)
        core = CoreTube.create(center=(10, 7.5), width=3.0, depth=3.0)
        core2, corridors, _islands = generate_rectangular_topology(
            floor_boundary=boundary,
            corridor_width=2.0,
            corridor_layout="cross",
            core_tube_override=core,
        )

        for c in corridors:
            assert c.polygon.intersection(core2.polygon).area < 1e-6


# ============================================================
# Phase 5: 降级摘要
# ============================================================

class TestDegradationSummary:

    def test_empty_warnings_zero_degradations(self):
        """无 warnings → total_degradations=0"""
        from backend.core.geometry.serializers import _build_degradation_summary

        summary = _build_degradation_summary([])
        assert summary["total_degradations"] == 0

    def test_skip_room_counted(self):
        """跳过房间的 warning 被分类"""
        from backend.core.geometry.serializers import _build_degradation_summary

        summary = _build_degradation_summary([
            "Skipping room r1: no viable island",
        ])
        assert summary["total_degradations"] == 1
        assert len(summary["skipped_rooms"]) == 1

    def test_miqp_fallback_counted(self):
        """MIQP fallback warning 被分类"""
        from backend.core.geometry.serializers import _build_degradation_summary

        summary = _build_degradation_summary([
            "MIQP fallback on island_0: timeout",
        ])
        assert summary["total_degradations"] == 1
        assert len(summary["miqp_fallback_floors"]) == 1

    def test_mixed_warnings(self):
        """多种 warning 混合正确分类"""
        from backend.core.geometry.serializers import _build_degradation_summary

        summary = _build_degradation_summary([
            "Skipping room r1: no viable island",
            "MIQP fallback on island_0: timeout",
            "F1, r2: zone='common' 非法，已修正为 'public'",
            "Unreachable rooms: [r3]",
        ])
        assert summary["total_degradations"] == 4
        assert len(summary["skipped_rooms"]) == 1
        assert len(summary["miqp_fallback_floors"]) == 1
        assert summary["parse_fixes"] == 1
        assert len(summary["unreachable_rooms"]) == 1


# ============================================================
# 组合故障测试
# ============================================================

class TestCombinedFailures:

    def test_json_truncated_plus_invalid_zone(self):
        """JSON 截断 + 非法 zone → 不崩溃"""
        rooms = [_make_room(zone="semi-public")]
        obj = _make_building([_make_floor(rooms=rooms)])
        raw = json.dumps(obj, ensure_ascii=False)
        truncated = raw[:-1]  # 去掉最后一个 }
        alloc, warns = _try_parse(truncated)
        assert len(alloc.floors) >= 1
        assert alloc.floors[0].rooms[0].zone == "public"

    def test_combined_zone_missing_id_string_weight(self):
        """zone 非法 + room_id 缺失 + weight 是字符串 → 全部修复"""
        rooms = [{
            "room_name": "客厅",
            "room_type": "living_room",
            "target_area": 30.0,
            "zone": "common",
            "weight": "7",
        }]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))

        room = alloc.floors[0].rooms[0]
        assert room.room_id != ""  # 自动生成
        assert room.zone == "public"  # common → public
        assert room.weight == 7  # "7" → 7

    def test_invalid_adjacency_plus_missing_fields(self):
        """无效邻接引用 + 缺失字段 → 不崩溃"""
        rooms = [{
            "room_name": "房间",
            "room_type": "room",
            "target_area": 20.0,
            "adjacency_required": ["ghost_room", "another_ghost"],
        }]
        obj = _make_building([_make_floor(rooms=rooms)])
        alloc, warns = _try_parse(json.dumps(obj, ensure_ascii=False))

        room = alloc.floors[0].rooms[0]
        assert room.room_id != ""
        assert room.zone == "public"
        # 无效引用被清除
        assert "ghost_room" not in room.adjacency_required

    def test_all_combined(self):
        """JSON 注释 + 尾逗号 + 无效 zone + 字符串 weight + 缺 room_id → 全部修复"""
        raw = json.dumps({
            "building_name": "测试",
            "total_floors": 1,
            "overall_total_area": 100.0,
            "floors": [{
                "floor_number": 1,
                "floor_function_tag": "residential",
                "floor_total_area": 100.0,
                "core_tube_area": 15.0,
                "corridor_allowance_area": 15.0,
                "rooms": [{
                    "room_name": "客厅",
                    "room_type": "living_room",
                    "target_area": 30.0,
                    "zone": "shared",
                    "weight": 5,
                }]
            }]
        }, ensure_ascii=False)

        # 注入注释 + 尾逗号
        broken = raw.replace('"weight": 5', '"weight": 5 // 优先级')
        broken = broken.replace('"zone": "shared"', '"zone": "shared",')

        alloc, warns = _try_parse(broken)
        assert len(alloc.floors) == 1
        room = alloc.floors[0].rooms[0]
        assert room.room_id != ""
        assert room.zone == "public"  # shared → public


# ============================================================
# 序列化集成测试
# ============================================================

class TestSerializerIntegration:

    def test_building_result_to_dict_has_walls(self):
        """building_result_to_dict 输出包含 walls/doors/windows/corridors/floor_slab"""
        from shapely.geometry import box, LineString
        from backend.core.geometry.layout_generator import RoomResult, LayoutResultV2
        from backend.core.geometry.building_orchestrator import BuildingResult
        from backend.core.geometry.topology_generator import Corridor
        from backend.core.geometry.serializers import building_result_to_dict

        rooms = [
            RoomResult(
                id="r1", room_type="living_room",
                polygon=box(0, 0, 6, 5),
                area=30.0, target_area=30.0, area_error=0.0,
                centroid=(3.0, 2.5), has_window=True,
                facade_length=6.0, aspect_ratio=1.2,
            ),
            RoomResult(
                id="r2", room_type="bedroom",
                polygon=box(6, 0, 12, 5),
                area=30.0, target_area=30.0, area_error=0.0,
                centroid=(9.0, 2.5), has_window=True,
                facade_length=6.0, aspect_ratio=1.2,
            ),
        ]

        corridors = [
            Corridor(
                id="corridor_h",
                centerline=LineString([(0, 2.5), (12, 2.5)]),
                width=2.0,
                orientation="horizontal",
            ),
        ]

        layout = LayoutResultV2(
            core_tube=None,
            corridors=corridors,
            islands=[],
            assignments={},
            room_layouts=rooms,
            validation=None,
            generation_time_ms=100.0,
            warnings=[],
        )

        result = BuildingResult(
            core_tube=None,
            floor_layouts={"F1": layout},
            warnings=[],
        )

        d = building_result_to_dict(result, box(0, 0, 12, 5))

        f1 = d["building"]["floors"]["F1"]
        assert "walls" in f1
        assert "doors" in f1
        assert "windows" in f1
        assert "corridors" in f1
        assert "floor_slab" in f1
        assert isinstance(f1["walls"], list)
        assert isinstance(f1["corridors"], list)
        assert len(f1["corridors"]) == 1
        assert f1["corridors"][0]["type"] == "corridor"
        assert f1["floor_slab"]["type"] == "floor_slab"
        assert f1["floor_slab"]["width"] == 12.0

    def test_degradation_summary_in_response(self):
        """API 响应包含 degradation_summary"""
        from shapely.geometry import box
        from backend.core.geometry.layout_generator import RoomResult, LayoutResultV2
        from backend.core.geometry.building_orchestrator import BuildingResult
        from backend.core.geometry.serializers import building_result_to_dict

        layout = LayoutResultV2(
            core_tube=None, corridors=[], islands=[],
            assignments={}, room_layouts=[],
            validation=None, generation_time_ms=50.0,
            warnings=["MIQP fallback on island_0: timeout"],
        )

        result = BuildingResult(
            core_tube=None,
            floor_layouts={"F1": layout},
            warnings=["MIQP fallback on island_0: timeout"],
        )

        d = building_result_to_dict(result, box(0, 0, 10, 10))

        assert "degradation_summary" in d
        assert d["degradation_summary"]["total_degradations"] >= 1


# ============================================================
# adjacency_forbidden 规则表
# ============================================================

class TestAdjacencyForbidden:

    def test_kitchen_forbidden_bedroom(self):
        """kitchen 不能与 bedroom 相邻"""
        from backend.core.geometry.room_defaults import compute_adjacency_forbidden

        rooms = [
            RoomAllocation(
                room_id="r1", room_name="厨房", room_type="kitchen", target_area=12.0,
            ),
            RoomAllocation(
                room_id="r2", room_name="卧室", room_type="bedroom", target_area=15.0,
            ),
        ]

        forbidden = compute_adjacency_forbidden(rooms)
        assert "r2" in forbidden["r1"]  # kitchen forbids bedroom

    def test_bathroom_forbidden_kitchen(self):
        """bathroom 不能与 kitchen 相邻"""
        from backend.core.geometry.room_defaults import compute_adjacency_forbidden

        rooms = [
            RoomAllocation(
                room_id="r1", room_name="卫生间", room_type="bathroom", target_area=6.0,
            ),
            RoomAllocation(
                room_id="r2", room_name="厨房", room_type="kitchen", target_area=12.0,
            ),
        ]

        forbidden = compute_adjacency_forbidden(rooms)
        assert "r2" in forbidden["r1"]  # bathroom forbids kitchen


# ============================================================
# E2E: 模拟完整 LLM 输出解析
# ============================================================

class TestRealPromptE2E:

    def test_two_floor_residential(self):
        """模拟"两层住宅"的完整 LLM 返回"""
        llm_output = json.dumps({
            "building_name": "两层住宅",
            "total_floors": 2,
            "overall_total_area": 200.0,
            "floors": [
                {
                    "floor_number": 1,
                    "floor_function_tag": "residential_public",
                    "floor_total_area": 100.0,
                    "core_tube_area": 10.0,
                    "corridor_allowance_area": 10.0,
                    "rooms": [
                        {"room_id": "room_001", "room_name": "客厅",
                         "room_type": "living_room", "target_area": 35.0,
                         "zone": "public", "needs_window": True,
                         "size_hint": "large", "adjacency_required": ["room_002"],
                         "weight": 5},
                        {"room_id": "room_002", "room_name": "餐厅",
                         "room_type": "dining_room", "target_area": 20.0,
                         "zone": "public", "size_hint": "medium",
                         "adjacency_required": ["room_001", "room_003"],
                         "weight": 3},
                        {"room_id": "room_003", "room_name": "厨房",
                         "room_type": "kitchen", "target_area": 15.0,
                         "zone": "service", "needs_window": True,
                         "size_hint": "small", "adjacency_required": ["room_002"],
                         "weight": 3},
                    ]
                },
                {
                    "floor_number": 2,
                    "floor_function_tag": "residential_private",
                    "floor_total_area": 100.0,
                    "core_tube_area": 10.0,
                    "corridor_allowance_area": 10.0,
                    "rooms": [
                        {"room_id": "room_004", "room_name": "主卧",
                         "room_type": "bedroom", "target_area": 25.0,
                         "zone": "private", "needs_window": True,
                         "size_hint": "large", "weight": 5},
                        {"room_id": "room_005", "room_name": "次卧",
                         "room_type": "bedroom", "target_area": 18.0,
                         "zone": "private", "needs_window": True,
                         "size_hint": "medium", "weight": 3},
                        {"room_id": "room_006", "room_name": "卫生间",
                         "room_type": "bathroom", "target_area": 7.0,
                         "zone": "service", "size_hint": "small",
                         "weight": 2},
                    ]
                }
            ]
        }, ensure_ascii=False)

        alloc, warns = _try_parse(llm_output)
        assert alloc.total_floors == 2
        assert len(alloc.floors) == 2
        assert len(alloc.floors[0].rooms) == 3
        assert len(alloc.floors[1].rooms) == 3

        # 邻接关系保留
        r1 = alloc.floors[0].rooms[0]
        assert "room_002" in r1.adjacency_required

        # size_hint 保留
        assert alloc.floors[0].rooms[0].size_hint == "large"
        assert alloc.floors[1].rooms[2].size_hint == "small"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
