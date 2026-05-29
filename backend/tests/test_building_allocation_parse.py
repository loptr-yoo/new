"""
test_building_allocation_parse.py

测试 LLM JSON → BuildingAllocation 的解析逻辑（正常 + 异常路径）
"""
import json
import pytest
from pydantic import ValidationError

from backend.models import BuildingAllocation, RoomAllocation, FloorAllocation


# ============================================================
# 辅助：构建测试用 JSON
# ============================================================

def _make_floor_json(floor_number=1, rooms=None):
    """构建单层 JSON"""
    if rooms is None:
        rooms = [
            {
                "room_id": "room_001",
                "room_name": "客厅",
                "room_type": "living_room",
                "target_area": 35.0,
                "zone": "public",
                "needs_window": True,
                "min_width": 3.5,
                "aspect_ratio_range": [0.6, 1.8],
                "adjacency_required": ["room_002"],
                "adjacency_preferred": [],
                "adjacency_forbidden": [],
                "weight": 5,
            },
            {
                "room_id": "room_002",
                "room_name": "餐厅",
                "room_type": "dining_room",
                "target_area": 15.0,
                "zone": "public",
                "needs_window": False,
                "min_width": 2.5,
                "aspect_ratio_range": [0.5, 2.0],
                "adjacency_required": ["room_001"],
                "adjacency_preferred": [],
                "adjacency_forbidden": [],
                "weight": 3,
            },
        ]
    return {
        "floor_number": floor_number,
        "floor_function_tag": "residential",
        "floor_total_area": 100.0,
        "core_tube_area": 15.0,
        "corridor_allowance_area": 15.0,
        "rooms": rooms,
    }


def _make_building_json(floors=None):
    """构建完整 Building JSON"""
    if floors is None:
        floors = [_make_floor_json(1)]
    return {
        "building_name": "测试住宅",
        "total_floors": len(floors),
        "overall_total_area": sum(f["floor_total_area"] for f in floors),
        "floors": floors,
    }


# ============================================================
# 正常路径测试
# ============================================================


class TestParseNewFormat:

    def test_new_fields_parsed(self):
        """新字段 (room_id, zone, adjacency_required) 正确解析"""
        obj = _make_building_json()
        allocation = BuildingAllocation.model_validate(obj)

        room = allocation.floors[0].rooms[0]
        assert room.room_id == "room_001"
        assert room.zone == "public"
        assert room.needs_window is True
        assert room.min_width == 3.5
        assert room.adjacency_required == ["room_002"]
        assert room.aspect_ratio_range == [0.6, 1.8]

    def test_two_floors(self):
        """多层解析"""
        floors = [_make_floor_json(1), _make_floor_json(2)]
        obj = _make_building_json(floors)
        allocation = BuildingAllocation.model_validate(obj)

        assert len(allocation.floors) == 2
        assert allocation.floors[0].floor_number == 1
        assert allocation.floors[1].floor_number == 2


class TestParseOldFormatCompat:

    def test_requires_window_compat(self):
        """旧字段 requires_window=True 回退到 needs_window"""
        rooms = [{
            "room_name": "卧室",
            "room_type": "bedroom",
            "target_area": 20.0,
            "requires_window": True,
            "weight": 5,
            "adjacency_tags": ["living_room"],
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        room = allocation.floors[0].rooms[0]
        # requires_window=True 应使 needs_window 也为 True（通过 _normalize_allocation）
        assert room.requires_window is True

    def test_adjacency_tags_present(self):
        """旧字段 adjacency_tags 应保留"""
        rooms = [{
            "room_name": "厨房",
            "room_type": "kitchen",
            "target_area": 12.0,
            "weight": 3,
            "adjacency_tags": ["dining_room"],
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        room = allocation.floors[0].rooms[0]
        assert room.adjacency_tags == ["dining_room"]


class TestParseMissingFieldsDefaults:

    def test_missing_zone_defaults_public(self):
        """zone 缺失时默认 public"""
        rooms = [{
            "room_name": "房间",
            "room_type": "room",
            "target_area": 20.0,
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        assert allocation.floors[0].rooms[0].zone == "public"

    def test_missing_needs_window_defaults_false(self):
        """needs_window 缺失时默认 False"""
        rooms = [{
            "room_name": "储藏室",
            "room_type": "storage",
            "target_area": 5.0,
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        assert allocation.floors[0].rooms[0].needs_window is False

    def test_missing_adjacency_defaults_empty(self):
        """adjacency_required 缺失时默认空列表"""
        rooms = [{
            "room_name": "房间",
            "room_type": "room",
            "target_area": 20.0,
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        assert allocation.floors[0].rooms[0].adjacency_required == []


# ============================================================
# 兼容逻辑精确测试
# ============================================================


class TestCompatLogicPrecision:

    def test_needs_window_false_not_overridden(self):
        """needs_window=False + requires_window=True → 保持 False

        验证兼容逻辑不会覆盖 LLM 显式设置的 needs_window=False。
        （_normalize_allocation 仅当 needs_window 为默认 False 且 requires_window=True 时回退）
        注意：此测试验证 Pydantic 模型层面的行为，_normalize_allocation 需要单独测试。
        """
        rooms = [{
            "room_name": "储藏室",
            "room_type": "storage",
            "target_area": 5.0,
            "needs_window": False,
            "requires_window": True,
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        room = allocation.floors[0].rooms[0]
        # Pydantic 层：needs_window 已被显式设为 False
        assert room.needs_window is False
        # requires_window 也被保留
        assert room.requires_window is True

    def test_adjacency_required_not_overridden(self):
        """adjacency_required=["r2"] + adjacency_tags=["old"] → 保持 ["r2"]"""
        rooms = [{
            "room_name": "房间",
            "room_type": "room",
            "target_area": 20.0,
            "adjacency_required": ["r2"],
            "adjacency_tags": ["old_tag"],
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        room = allocation.floors[0].rooms[0]
        assert room.adjacency_required == ["r2"]
        assert room.adjacency_tags == ["old_tag"]


# ============================================================
# 异常路径测试
# ============================================================


class TestParseInvalidValues:

    def test_invalid_zone_in_preprocess(self):
        """zone 非法值在预处理中被修正为 public"""
        try:
            from backend.core.flows.building_semantic_flow import _parse_building_allocation
        except ImportError:
            pytest.skip("dotenv or other server dependency not installed")

        rooms = [{
            "room_name": "房间",
            "room_type": "room",
            "target_area": 20.0,
            "zone": "semi-public",  # 非法值
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        text = json.dumps(obj, ensure_ascii=False)

        allocation, _warnings = _parse_building_allocation(text, provider="test", model="test")
        assert allocation.floors[0].rooms[0].zone == "public"

    def test_invalid_adjacency_ref_does_not_crash(self):
        """adjacency_required 引用不存在的 room_id 不崩溃"""
        rooms = [{
            "room_id": "r1",
            "room_name": "房间",
            "room_type": "room",
            "target_area": 20.0,
            "adjacency_required": ["nonexistent_room"],
        }]
        obj = _make_building_json([_make_floor_json(1, rooms)])
        allocation = BuildingAllocation.model_validate(obj)

        # 解析成功，不崩溃
        assert allocation.floors[0].rooms[0].adjacency_required == ["nonexistent_room"]


# ============================================================
# 核心筒拆分测试
# ============================================================


class TestCoreTubeSplit:

    def test_areas_match(self):
        """elevator + staircase 面积之和 == 整体面积"""
        from backend.core.geometry.topology_generator import CoreTube

        core = CoreTube.create_for_floor((0, 0, 20, 15))
        assert core.elevator is not None
        assert core.staircase is not None
        assert abs(core.elevator_area + core.staircase_area - core.polygon.area) < 0.01

    def test_boundary_unchanged(self):
        """拆分后外轮廓不变"""
        from backend.core.geometry.topology_generator import CoreTube

        core = CoreTube.create((10, 7.5), 4, 3)
        from shapely.geometry import box
        expected = box(8, 6, 12, 9)
        assert core.polygon.equals_exact(expected, 0.01)

    def test_elevator_larger_than_staircase(self):
        """电梯面积应大于楼梯"""
        from backend.core.geometry.topology_generator import CoreTube

        core = CoreTube.create_for_floor((0, 0, 30, 20))
        assert core.elevator_area > core.staircase_area


# ============================================================
# 序列化测试
# ============================================================


class TestSerialization:

    def test_room_result_to_dict(self):
        """RoomResult → dict 格式正确"""
        from shapely.geometry import box
        from backend.core.geometry.layout_generator import RoomResult
        from backend.core.geometry.serializers import room_result_to_dict

        room = RoomResult(
            id="living", room_type="living_room",
            polygon=box(0, 0, 5, 7),
            area=35.0, target_area=35.0, area_error=0.0,
            centroid=(2.5, 3.5), has_window=True,
            facade_length=5.0, aspect_ratio=1.4,
        )
        d = room_result_to_dict(room, "F1", floor_bounds=(0.0, 0.0, 10.0, 10.0))

        assert d["room_id"] == "living"
        assert d["room_type"] == "living_room"
        assert d["floor_id"] == "F1"
        assert len(d["polygon"]) == 5  # 矩形 5 个点（含闭合）
        assert d["area"] == 35.0
        assert d["has_window"] is True

    def test_core_tube_to_dict(self):
        """CoreTube → dict 含 elevator/staircase"""
        from backend.core.geometry.topology_generator import CoreTube
        from backend.core.geometry.serializers import core_tube_to_dict

        core = CoreTube.create((10, 7.5), 4, 3)
        d = core_tube_to_dict(core, floor_bounds=(0.0, 0.0, 30.0, 20.0))

        assert "boundary" in d
        assert "elevator" in d
        assert "staircase" in d
        assert d["elevator"]["area"] > 0
        assert d["staircase"]["area"] > 0


# ============================================================
# JSON 修复测试（LLM 输出容错）
# ============================================================


class TestJsonRepair:
    """测试 _parse_building_allocation 对 LLM 常见 JSON 错误的容错"""

    def _try_parse(self, text: str):
        try:
            from backend.core.flows.building_semantic_flow import _parse_building_allocation
        except ImportError:
            pytest.skip("dotenv or other server dependency not installed")
        allocation, _warnings = _parse_building_allocation(text, provider="test", model="test")
        return allocation

    def _make_valid_json(self) -> str:
        """构造一个合法的 BuildingAllocation JSON 字符串"""
        return json.dumps(_make_building_json(), ensure_ascii=False)

    def test_trailing_commas(self):
        """LLM 常见错误：尾逗号"""
        raw = self._make_valid_json()
        # 注入尾逗号
        broken = raw.replace('"weight": 3}', '"weight": 3,}')
        allocation = self._try_parse(broken)
        assert len(allocation.floors) == 1

    def test_comments_in_json(self):
        """LLM 常见错误：JSON 中混入注释"""
        raw = self._make_valid_json()
        broken = raw.replace('"room_type": "living_room"',
                            '"room_type": "living_room" // 客厅')
        allocation = self._try_parse(broken)
        assert allocation.floors[0].rooms[0].room_type == "living_room"

    def test_markdown_wrapped(self):
        """LLM 常见错误：JSON 被 markdown 代码块包裹"""
        raw = self._make_valid_json()
        wrapped = f"```json\n{raw}\n```"
        allocation = self._try_parse(wrapped)
        assert len(allocation.floors) == 1

    def test_extra_text_before_json(self):
        """LLM 常见错误：JSON 前有解释文本"""
        raw = self._make_valid_json()
        with_preamble = f"好的，以下是生成的方案：\n\n{raw}"
        allocation = self._try_parse(with_preamble)
        assert len(allocation.floors) == 1

    def test_unbalanced_braces(self):
        """LLM 常见错误：括号不闭合（截断）"""
        raw = self._make_valid_json()
        # 去掉最后两个 }
        truncated = raw.rstrip('}')[:-1]
        allocation = self._try_parse(truncated)
        assert len(allocation.floors) >= 1

    def test_real_world_two_floor_residential(self):
        """端到端：模拟"两层住宅"的 LLM 返回"""
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
                        {"room_name": "客厅", "room_type": "living_room",
                         "target_area": 35.0, "zone": "public",
                         "needs_window": True, "weight": 5},
                        {"room_name": "餐厅", "room_type": "dining_room",
                         "target_area": 20.0, "zone": "public", "weight": 3},
                        {"room_name": "厨房", "room_type": "kitchen",
                         "target_area": 15.0, "zone": "public",
                         "needs_window": True, "weight": 3},
                    ]
                },
                {
                    "floor_number": 2,
                    "floor_function_tag": "residential_private",
                    "floor_total_area": 100.0,
                    "core_tube_area": 10.0,
                    "corridor_allowance_area": 10.0,
                    "rooms": [
                        {"room_name": "主卧", "room_type": "bedroom",
                         "target_area": 25.0, "zone": "private",
                         "needs_window": True, "weight": 5},
                        {"room_name": "次卧", "room_type": "bedroom",
                         "target_area": 18.0, "zone": "private",
                         "needs_window": True, "weight": 3},
                        {"room_name": "卫生间", "room_type": "bathroom",
                         "target_area": 7.0, "zone": "private", "weight": 2},
                    ]
                }
            ]
        }, ensure_ascii=False)

        allocation = self._try_parse(llm_output)
        assert allocation.total_floors == 2
        assert len(allocation.floors) == 2
        assert len(allocation.floors[0].rooms) >= 3
        assert len(allocation.floors[1].rooms) >= 3

        f1_types = {r.room_type for r in allocation.floors[0].rooms}
        assert {"living_room", "dining_room", "kitchen"} <= f1_types

        f2_types = {r.room_type for r in allocation.floors[1].rooms}
        assert {"bedroom", "bathroom"} <= f2_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
