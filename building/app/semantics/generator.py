from __future__ import annotations

import json
import logging
import re
import hashlib
from asyncio import sleep
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from ..models import BuildingAllocation, BuildingEnvelopeBase, GenerateSemanticsRequest
from ..settings import settings
from ..geometry.room_spec import ROOM_TYPE_DEFAULTS
from ..logger import log_multiline_debug
from ..llm.provider import ChatMessage, LLMConfig, create_llm_client
from ..llm.response_parser import ParseOptions, extract_json, parse_ai_response
from ..llm.retry import call_llm_with_retry
from ..prompts.building_prompts import (
    BUILDING_BUDGETED_ALLOCATION_SYSTEM_PROMPT,
    BUILDING_ENVELOPE_SYSTEM_PROMPT,
    BUILDING_PLANNER_SYSTEM_PROMPT,
    BUILDING_STEP1_SYSTEM_PROMPT,
    BUILDING_STEP2_SYSTEM_PROMPT,
)
from .area_normalizer import normalize_floor_room_areas
from ..geometry.topology_snapshot import (
    BuildingAreaBudget,
    LayoutFailureReport,
    TopologySnapshot,
    compute_building_area_budget,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_BY_PROVIDER: Dict[str, str] = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
}

BUILDING_SEMANTIC_MAX_TOKENS = 12000


class BudgetValidationError(RuntimeError):
    """Raised when LLM room allocation violates physical floor budgets."""


@dataclass
class BudgetedBuildingSemanticResult:
    allocation: BuildingAllocation
    warnings: List[str]
    envelope: BuildingEnvelopeBase
    budget: BuildingAreaBudget
    topology_snapshot: Optional[TopologySnapshot]


def _normalize_room_type(raw_type: str) -> str:
    """
    将 LLM 输出的非标房间类型映射为标准白名单类型。
    在房间对象初始化/预处理时调用。

    注意：此映射表仅处理"相似但不规范的词" → "官方新词"的映射。
    Task 1 已将 phone_booth、waiting_area、utility 等加入 ROOM_TYPE_DEFAULTS，
    因此本函数只需将变体形式映射到这些已识别的标准类型。
    已删除所有 key == value 的自我映射条目（未映射类型会通过兜底逻辑原样返回）。
    """
    if not raw_type:
        return "unknown"

    raw = str(raw_type).strip()
    raw_l = raw.lower().strip()

    has_zh = any("\u4e00" <= ch <= "\u9fff" for ch in raw_l)
    if has_zh:
        cleaned = raw_l.replace("间", "").replace("室", "").replace("区", "")
        raw_l = cleaned if cleaned else raw_l

    mapping = {
        # office
        "open_office": "office",
        "open plan office": "office",
        "open-plan office": "office",
        "office_space": "office",
        "office area": "office",
        "admin_office": "office",
        "management_office": "office",
        # meeting
        "small_meeting": "meeting_room",
        "small meeting": "meeting_room",
        "meeting": "meeting_room",
        "meeting room": "meeting_room",
        "conference": "meeting_room",
        "conference_room": "meeting_room",
        "large_meeting": "meeting_room",
        "discussion_room": "meeting_room",
        "negotiation_room": "meeting_room",
        "visitor_meeting": "meeting_room",
        # phone booth
        "booth": "phone_booth",
        "phone booth": "phone_booth",
        "phone_booth": "phone_booth",
        # pantry
        "tea_room": "pantry",
        "tea room": "pantry",
        "pantry_room": "pantry",
        # waiting / lobby / reception
        "waiting": "waiting_area",
        "waiting area": "waiting_area",
        "waiting_area": "waiting_area",
        "foyer": "lobby",
        "lobby_area": "lobby",
        "reception": "reception",
        "front_desk": "reception",
        "front desk": "reception",
        "reception_desk": "reception",
        "reception_hall": "reception",
        # lounge
        "lounge_area": "lounge",
        "lounge area": "lounge",
        "coffee_lounge": "lounge",
        "cafe": "lounge",
        "cafe_lounge": "lounge",
        # classroom / reading / activity / play
        "art_room": "classroom",
        "art room": "classroom",
        "art_studio": "classroom",
        "training_room": "classroom",
        "children_classroom": "classroom",
        "reading": "reading_room",
        "activity": "activity_room",
        "play room": "playroom",
        # bathroom / storage
        "restroom": "bathroom",
        "toilet": "bathroom",
        "wc": "bathroom",
        "storage_room": "storage",
        "storage room": "storage",
        # utility / back-of-house
        "linen_room": "utility",
        "back_of_house": "utility",
    }

    if raw_l in mapping:
        mapped = mapping[raw_l]
    else:
        if has_zh:
            zh_map = {
                "茶水": "pantry",
                "电话": "phone_booth",
                "前台": "reception",
                "接待": "reception",
                "大堂": "lobby",
                "门厅": "lobby",
                "玄关": "entrance",
                "办公": "office",
                "会议": "meeting_room",
                "洽谈": "meeting_room",
                "咖啡": "lounge",
                "休闲": "lounge",
                "后勤": "utility",
                "设备": "utility",
                "布草": "utility",
                "阅读": "reading_room",
                "活动": "activity_room",
            }
            mapped = None
            for k, v in zh_map.items():
                if k in raw_l:
                    mapped = v
                    break
            if mapped is None:
                mapped = raw_l.replace(" ", "_").replace("-", "_")
        else:
            mapped = raw_l.replace(" ", "_").replace("-", "_")

    if mapped == "reception" and "reception" not in ROOM_TYPE_DEFAULTS:
        return "lobby"

    return mapped


def _pick_provider_model_and_key(request: GenerateSemanticsRequest) -> Tuple[str, str, str]:
    def _key_for(p: str) -> Optional[str]:
        if p == "openai":
            return settings.openai_api_key
        if p == "gemini":
            return settings.gemini_api_key
        if p == "deepseek":
            return settings.deepseek_api_key
        return None

    provider = request.provider
    if provider is None:
        for cand in ("openai", "gemini", "deepseek"):
            if _key_for(cand):
                provider = cand  # type: ignore[assignment]
                break
    if provider is None:
        raise RuntimeError("未配置任何可用的 LLM API Key，无法执行语义规划。")

    api_key = _key_for(provider)
    if not api_key:
        raise RuntimeError(f"缺少 provider={provider} 的 API Key。")

    model = request.model or _DEFAULT_MODEL_BY_PROVIDER.get(provider)
    if not model:
        raise RuntimeError(f"provider={provider} 未提供 model。")

    return provider, model, api_key


# ============================================================
# User prompt 构建
# ============================================================

def _build_user_prompt(request: GenerateSemanticsRequest) -> str:
    lines: List[str] = []
    lines.append("请基于以下需求输出 BuildingAllocation 的纯 JSON。")
    lines.append("")
    lines.append(f"用户需求：{request.user_prompt}")
    if request.total_area is not None:
        lines.append(f"总建筑面积（overall_total_area）：{request.total_area} 平方米")
    if request.total_floors is not None:
        lines.append(f"总楼层数（total_floors）：{request.total_floors}")
    lines.append("")
    lines.append("输出要求：只输出一个 JSON 对象，不要 markdown，不要解释，不要额外字段。")
    return "\n".join(lines)


def _build_step1_user_prompt(request: GenerateSemanticsRequest) -> str:
    lines: List[str] = []
    lines.append("请基于以下需求输出每层的房间清单骨架（纯 JSON）。")
    lines.append(f"用户需求：{request.user_prompt}")
    if request.total_area is not None:
        lines.append(f"总建筑面积：{request.total_area} 平方米")
    if request.total_floors is not None:
        lines.append(f"总楼层数：{request.total_floors}")
    lines.append("")
    lines.append("每个房间只输出 4 个字段：room_id, room_name, room_type, zone")
    lines.append("楼层需包含 floor_total_area, core_tube_area, corridor_allowance_area 的合理估值。")
    return "\n".join(lines)


def _build_step2_user_prompt(step1_json: str) -> str:
    return (
        f"以下是已确认的房间清单骨架：\n\n{step1_json}\n\n"
        "请为每个房间补充 size_hint 和 adjacency_required 字段，"
        "保持其他字段不变。输出完整 JSON。"
    )


# ============================================================
# 多层防御解析
# ============================================================

def _build_envelope_user_prompt(request: GenerateSemanticsRequest) -> str:
    lines: List[str] = [
        "Extract only the building envelope as JSON.",
        f"User request: {request.user_prompt}",
    ]
    if request.total_area is not None:
        lines.append(f"User-provided overall_total_area: {request.total_area} m2")
    if request.total_floors is not None:
        lines.append(f"User-provided total_floors: {request.total_floors}")
    lines.extend(
        [
            "",
            "Do not output rooms, room target areas, core budget, corridor budget, or geometry.",
            "Preserve explicit room counts in requested_rooms_list as separate items.",
        ]
    )
    return "\n".join(lines)


def _budget_prompt_payload(
    envelope: BuildingEnvelopeBase,
    budget: BuildingAreaBudget,
) -> Dict[str, Any]:
    floors_payload: List[Dict[str, Any]] = []
    envelope_by_floor = {int(f.floor_number): f for f in envelope.floors}
    for fid in sorted(budget.floors.keys()):
        b = budget.floors[fid]
        env_floor = envelope_by_floor.get(int(b.floor_number))
        floors_payload.append(
            {
                "floor_id": fid,
                "floor_number": int(b.floor_number),
                "floor_function_tag": getattr(env_floor, "floor_function_tag", "standard"),
                "requested_rooms_list": list(getattr(env_floor, "requested_rooms_list", []) or []),
                "floor_total_area": round(float(b.floor_total_area), 2),
                "core_tube_area": round(float(b.core_tube_area), 2),
                "corridor_allowance_area": round(float(b.corridor_allowance_area), 2),
                "total_island_area": round(float(b.total_island_area), 2),
                "room_sum_min": round(float(b.room_sum_min), 2),
                "room_sum_recommended": round(float(b.room_sum_recommended), 2),
                "room_sum_max": round(float(b.room_sum_max), 2),
            }
        )
    return {
        "building_name": envelope.building_name,
        "total_floors": int(envelope.total_floors),
        "overall_total_area": round(float(envelope.overall_total_area), 2),
        "floors": floors_payload,
    }


def _build_budgeted_allocation_user_prompt(
    request: GenerateSemanticsRequest,
    envelope: BuildingEnvelopeBase,
    budget: BuildingAreaBudget,
    repair_instruction: Optional[str] = None,
) -> str:
    payload = _budget_prompt_payload(envelope, budget)
    lines = [
        "Generate the final BuildingAllocation JSON using the physical budgets below.",
        f"Original user request: {request.user_prompt}",
        "",
        "Envelope and physical budgets:",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ]
    if repair_instruction:
        lines.extend(["", "Previous attempt failed. Mandatory repair instruction:", str(repair_instruction)])
    return "\n".join(lines)


_VALID_ZONES = {"public", "private", "service", "circulation"}
_ZONE_ALIASES = {
    "common": "public", "shared": "public", "living": "public",
    "bedroom": "private", "sleeping": "private", "personal": "private",
    "bath": "private", "toilet": "private",
    "storage": "service", "utility": "service", "mep": "service",
    "mechanical": "service",
    "corridor": "circulation", "hallway": "circulation",
}


def _robust_json_loads(text: str, provider: str = "", model: str = "") -> Optional[dict]:
    """多策略 JSON 提取"""
    extracted = extract_json(text)

    # 预处理：在所有策略前先清理注释和尾逗号
    cleaned = re.sub(r'//[^\n]*', '', extracted)
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)

    # 策略 1: 直接解析（先试清理后版本，再试原始版本）
    for candidate in (cleaned, extracted):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except JSONDecodeError:
            pass

    # 策略 2: json_repair（用清理后的文本）
    try:
        import json_repair
        repaired = json_repair.repair_json(cleaned)
        obj = json.loads(repaired) if isinstance(repaired, str) else repaired
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 策略 3: 手动修复（括号补全）
    try:
        fixed = cleaned
        open_b = fixed.count('{') - fixed.count('}')
        if 0 < open_b < 20:
            fixed += '}' * open_b
        open_s = fixed.count('[') - fixed.count(']')
        if 0 < open_s < 20:
            fixed += ']' * open_s
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 策略 4: 兜底
    try:
        obj = parse_ai_response(text, ParseOptions(provider=provider, model=model))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def _parse_building_allocation(
    text: str, provider: str, model: str,
) -> Tuple[BuildingAllocation, List[str]]:
    """
    将 LLM 返回文本解析为 BuildingAllocation。

    返回 (allocation, parse_warnings)。永不因字段问题崩溃——
    通过多层修复 + 智能预处理确保 Pydantic 校验前数据已尽量合规。
    """
    parse_warnings: List[str] = []

    # --- Stage 1: 提取 JSON ---
    obj = _robust_json_loads(text, provider, model)
    if obj is None:
        raise RuntimeError("无法从 LLM 响应中提取有效 JSON")

    # --- Stage 2: Pydantic 前智能预处理 ---
    if isinstance(obj, dict) and "floors" in obj:
        _preprocess_allocation(obj, parse_warnings)
        obj, area_warnings = normalize_floor_room_areas(obj, min_room_area=4.0)
        parse_warnings.extend(area_warnings)

    # --- Stage 3: Pydantic 校验 ---
    try:
        allocation = BuildingAllocation.model_validate(obj)
    except ValidationError as e:
        logger.error(f"Pydantic validation failed after preprocessing: {e}")
        raise
    _validate_building_floor_count(allocation)

    # --- Stage 4: 后处理 ---
    _normalize_allocation(allocation)

    return allocation, parse_warnings


def _validate_building_floor_count(allocation: BuildingAllocation) -> None:
    expected = int(getattr(allocation, "total_floors", 0) or 0)
    actual = len(getattr(allocation, "floors", []) or [])
    if actual != expected:
        raise RuntimeError(
            "BuildingAllocation floor count mismatch: "
            f"total_floors={expected}, floors_len={actual}. "
            "The model output is incomplete or inconsistent and must be regenerated."
        )


def _preprocess_allocation(obj: dict, parse_warnings: List[str]) -> None:
    """Pydantic 前智能预处理"""
    def _safe_float(val, default: float = 0.0) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    def _estimate_core_area_ratio(total_area: float) -> float:
        return 0.08 if total_area < 80.0 else 0.12

    # 顶层字段补全
    obj.setdefault("building_name", "Building")
    obj.setdefault("total_floors", len(obj.get("floors", [])))
    if "overall_total_area" not in obj:
        obj["overall_total_area"] = sum(
            f.get("floor_total_area", 100) for f in obj.get("floors", [])
            if isinstance(f, dict)
        )

    for fi, floor in enumerate(obj.get("floors", [])):
        if not isinstance(floor, dict):
            continue

        # 楼层字段补全
        floor.setdefault("floor_number", fi + 1)
        floor.setdefault("floor_function_tag", "standard")
        floor.setdefault("core_tube_area", floor.get("floor_total_area", 100) * 0.15)
        floor.setdefault("corridor_allowance_area", floor.get("floor_total_area", 100) * 0.15)

        rooms = floor.get("rooms", [])
        if not isinstance(rooms, list):
            rooms = []
            floor["rooms"] = rooms

        # 收集本层 room_id 用于引用清洗
        for i, room in enumerate(rooms):
            if not isinstance(room, dict):
                continue
            if not room.get("room_id"):
                room["room_id"] = f"room_{i + 1:03d}"

        room_ids_in_floor = {
            r.get("room_id", "") for r in rooms if isinstance(r, dict)
        }

        for i, room in enumerate(rooms):
            if not isinstance(room, dict):
                continue

            # 必填字段补全
            room.setdefault("room_name", room.get("room_type", f"room_{i}"))
            room.setdefault("room_type", "room")
            room.setdefault("target_area", 20.0)
            room.setdefault("zone", "public")
            room.setdefault("needs_window", False)

            # 规范化 room_type（仅当 LLM 给出了非默认值时）
            if room.get("room_type") and room["room_type"] != "room":
                normalized = _normalize_room_type(room["room_type"])
                if normalized != room["room_type"]:
                    parse_warnings.append(
                        f"F{floor.get('floor_number', '?')}, {room.get('room_id', '?')}: "
                        f"room_type '{room['room_type']}' → '{normalized}'"
                    )
                    room["room_type"] = normalized
                if normalized and normalized not in ROOM_TYPE_DEFAULTS:
                    parse_warnings.append(
                        f"F{floor.get('floor_number', '?')}, {room.get('room_id', '?')}: "
                        f"room_type '{normalized}' 不在 ROOM_TYPE_DEFAULTS 中，可能发生语义退化"
                    )

            # 类型修复
            if isinstance(room.get("target_area"), str):
                try:
                    room["target_area"] = float(room["target_area"])
                except ValueError:
                    room["target_area"] = 20.0

            if isinstance(room.get("weight"), str):
                try:
                    room["weight"] = int(room["weight"])
                except ValueError:
                    room["weight"] = 5

            if isinstance(room.get("needs_window"), str):
                room["needs_window"] = room["needs_window"].lower() in ("true", "1", "yes")

            # weight 范围
            if "weight" in room:
                room["weight"] = max(1, min(10, int(room.get("weight", 5))))

            # zone 枚举修复
            zone_val = str(room.get("zone", "public")).lower().strip()
            if zone_val not in _VALID_ZONES:
                mapped = _ZONE_ALIASES.get(zone_val, "public")
                parse_warnings.append(
                    f"F{floor.get('floor_number', '?')}, {room.get('room_id', '?')}: "
                    f"zone='{zone_val}' 非法，已修正为 '{mapped}'"
                )
                room["zone"] = mapped
            else:
                room["zone"] = zone_val

            # aspect_ratio_range 修复
            if "aspect_ratio_range" in room:
                arr = room["aspect_ratio_range"]
                if not isinstance(arr, list) or len(arr) != 2:
                    room["aspect_ratio_range"] = [0.5, 2.0]
                else:
                    try:
                        room["aspect_ratio_range"] = [float(arr[0]), float(arr[1])]
                    except (ValueError, TypeError):
                        room["aspect_ratio_range"] = [0.5, 2.0]

            # 邻接字段规范化
            for adj_key in ["adjacency_required", "adjacency_preferred",
                            "adjacency_forbidden", "adjacency_tags"]:
                if adj_key in room:
                    val = room[adj_key]
                    if isinstance(val, str):
                        room[adj_key] = [val] if val else []
                    elif not isinstance(val, list):
                        room[adj_key] = []

            # 引用清洗（带 warning）
            for adj_key in ["adjacency_required", "adjacency_forbidden"]:
                if adj_key in room and isinstance(room[adj_key], list):
                    invalid = [ref for ref in room[adj_key] if ref not in room_ids_in_floor]
                    if invalid:
                        parse_warnings.append(
                            f"F{floor.get('floor_number', '?')}, {room.get('room_id', '?')}: "
                            f"{adj_key} 引用 {invalid} 不存在，已移除"
                        )
                        room[adj_key] = [ref for ref in room[adj_key] if ref in room_ids_in_floor]

        floor_total_area = _safe_float(floor.get("floor_total_area"))
        if floor_total_area > 0:
            corridor_area = _safe_float(floor.get("corridor_allowance_area", 0.0))
            floor["corridor_allowance_area"] = corridor_area

            min_corridor = floor_total_area * 0.10
            max_corridor = floor_total_area * 0.30
            if corridor_area < min_corridor or corridor_area > max_corridor:
                corridor_area = floor_total_area * 0.20
                floor["corridor_allowance_area"] = corridor_area
                parse_warnings.append(
                    f"F{floor.get('floor_number', '?')}: corridor_allowance_area 离谱，已修正为 {corridor_area:.1f}㎡"
                )

            core_ratio = _estimate_core_area_ratio(floor_total_area)
            core_area_est = floor_total_area * core_ratio
            core_area_raw = _safe_float(floor.get("core_tube_area", core_area_est), default=core_area_est)
            if abs(core_area_raw - core_area_est) > max(2.0, floor_total_area * 0.10):
                floor["core_tube_area"] = core_area_est
                parse_warnings.append(
                    f"F{floor.get('floor_number', '?')}: core_tube_area 偏差较大，已按几何规则修正为 {core_area_est:.1f}㎡"
                )
            else:
                floor["core_tube_area"] = core_area_raw

            for room in rooms:
                if not isinstance(room, dict):
                    continue
                room["target_area"] = _safe_float(room.get("target_area", 0.0))

            total_target_area = sum(
                _safe_float(r.get("target_area", 0.0))
                for r in rooms
                if isinstance(r, dict)
            )

            core_area = _safe_float(floor.get("core_tube_area", 0.0))
            available_area = floor_total_area - core_area - corridor_area
            gap = available_area - total_target_area

            if abs(gap) > 1.0:
                parse_warnings.append(
                    f"F{floor.get('floor_number', '?')}: 面积不闭环。 "
                    f"目标 {total_target_area:.1f}㎡ vs 可用 {available_area:.1f}㎡ (差值 {gap:.1f}㎡)"
                )

            if gap > 5.0:
                existing_ids = {
                    r.get("room_id", "") for r in rooms if isinstance(r, dict)
                }
                gap_seed = round(float(gap), 2)
                floor_seed = str(floor.get("floor_number", "?"))
                idx = 0
                seed = f"{floor_seed}_filler_utility_{gap_seed}_{idx}"
                new_room_id = f"room_filler_utility_{hashlib.md5(seed.encode('utf-8')).hexdigest()[:6]}"
                while new_room_id in existing_ids:
                    idx += 1
                    seed = f"{floor_seed}_filler_utility_{gap_seed}_{idx}"
                    new_room_id = f"room_filler_utility_{hashlib.md5(seed.encode('utf-8')).hexdigest()[:6]}"

                rooms.append(
                    {
                        "room_id": new_room_id,
                        "room_name": "设备间",
                        "room_type": "utility",
                        "target_area": round(gap, 2),
                        "zone": "service",
                        "needs_window": False,
                    }
                )
                parse_warnings.append(f"F{floor.get('floor_number', '?')}: 已动态注入 {gap:.1f}㎡ 的 utility 占位空间。")
                gap = 0.0
            elif (1.0 <= gap <= 5.0) or (gap < -1.0):
                flexible_types = {
                    "lobby",
                    "lounge",
                    "office",
                    "activity_room",
                    "corridor",
                    "living_room",
                    "dining_room",
                }
                flex_rooms = [
                    r
                    for r in rooms
                    if isinstance(r, dict) and r.get("room_type") in flexible_types
                ]

                flex_total = sum(
                    _safe_float(r.get("target_area", 0.0)) for r in flex_rooms
                )
                if flex_total > 0:
                    flex_scale = (flex_total + gap) / flex_total
                    flex_safe_scale = max(0.7, min(1.3, flex_scale))
                    for r in flex_rooms:
                        r["target_area"] = _safe_float(r.get("target_area", 0.0)) * flex_safe_scale

                total_target_area = sum(
                    _safe_float(r.get("target_area", 0.0))
                    for r in rooms
                    if isinstance(r, dict)
                )
                gap = available_area - total_target_area

                if abs(gap) > 1.0:
                    current_total = total_target_area
                    if current_total > 0:
                        scale = available_area / current_total
                        safe_scale = max(0.8, min(1.2, scale))
                        for r in rooms:
                            if not isinstance(r, dict):
                                continue
                            r["target_area"] = _safe_float(r.get("target_area", 0.0)) * safe_scale
                    else:
                        parse_warnings.append(
                            f"F{floor.get('floor_number', '?')}: 房间总面积为 0，跳过全体缩放，交由模型层兜底"
                        )

                total_target_area = sum(
                    _safe_float(r.get("target_area", 0.0))
                    for r in rooms
                    if isinstance(r, dict)
                )
                gap = available_area - total_target_area

                if abs(gap) > 2.0:
                    parse_warnings.append(
                        f"F{floor.get('floor_number', '?')}: 残余面积偏差过大({gap:.1f}㎡)，放弃强行补差，交由模型层全局缩放。"
                    )
                elif abs(gap) > 1e-3:
                    if rooms:
                        biggest_room = (
                            max(flex_rooms, key=lambda x: _safe_float(x.get("target_area", 0.0)))
                            if flex_rooms
                            else max(
                                (r for r in rooms if isinstance(r, dict)),
                                key=lambda x: _safe_float(x.get("target_area", 0.0)),
                            )
                        )
                        candidate = _safe_float(biggest_room.get("target_area", 0.0)) + gap
                        if candidate < 2.0:
                            parse_warnings.append(
                                f"F{floor.get('floor_number', '?')}: 补差会导致房间面积低于 2.0㎡，放弃补差并交由模型层兜底"
                            )
                        else:
                            biggest_room["target_area"] = candidate

        # floor_total_area 补全（从 rooms 反推）
        if "floor_total_area" not in floor:
            room_sum = sum(r.get("target_area", 0) for r in rooms if isinstance(r, dict))
            floor["floor_total_area"] = room_sum + floor.get("core_tube_area", 0) + floor.get("corridor_allowance_area", 0)


def _normalize_allocation(allocation: BuildingAllocation) -> None:
    """后处理：旧字段兼容 + needs_window 推导"""
    from ..geometry.room_defaults import infer_needs_window

    for floor in allocation.floors:
        for i, room in enumerate(floor.rooms):
            if not room.room_id:
                room.room_id = f"F{floor.floor_number}_{room.room_type}_{i}"

            # 旧字段兼容：仅当 needs_window 为默认 False 且旧字段为 True 时
            if not room.needs_window and room.requires_window is True:
                room.needs_window = True

            # 旧字段兼容：adjacency_tags → adjacency_required
            if not room.adjacency_required and room.adjacency_tags:
                room.adjacency_required = list(room.adjacency_tags)

            # needs_window 后端推导（当 LLM 未显式设置时）
            if not room.needs_window:
                room.needs_window = infer_needs_window(room.room_type)


def _apply_size_hints_to_allocation(allocation: BuildingAllocation) -> None:
    """如果房间有 size_hint，用 apply_size_hints 计算 target_area"""
    from ..geometry.room_defaults import apply_size_hints

    for floor in allocation.floors:
        has_hints = any(room.size_hint for room in floor.rooms)
        if not has_hints:
            continue
        apply_size_hints(
            floor.rooms,
            floor.floor_total_area,
            floor.core_tube_area,
        )


def _apply_adjacency_forbidden(allocation: BuildingAllocation) -> None:
    """用规则表计算 adjacency_forbidden"""
    from ..geometry.room_defaults import compute_adjacency_forbidden

    for floor in allocation.floors:
        forbidden_map = compute_adjacency_forbidden(floor.rooms)
        for room in floor.rooms:
            if not room.adjacency_forbidden:
                room.adjacency_forbidden = forbidden_map.get(room.room_id, [])


# ============================================================
# 错误反馈辅助
# ============================================================

def _extract_validation_errors(e: Exception) -> str:
    """从 Pydantic ValidationError 提取字段级错误，构造针对性重试提示"""
    if isinstance(e, ValidationError):
        field_errors = []
        for err in e.errors()[:5]:
            path = " -> ".join(str(p) for p in err["loc"])
            field_errors.append(f"  - {path}: {err['msg']}")
        details = "\n".join(field_errors)
    else:
        details = str(e)[:300]

    return (
        f"你上一次的输出校验失败，具体错误：\n{details}\n\n"
        "请修正以上问题并重新输出纯 JSON。要求：\n"
        "1. 所有 room 必须有 room_id, room_name, room_type, target_area, zone\n"
        "2. target_area 必须是数字，zone 只能是 public/private/service/circulation\n"
        "3. adjacency_required/adjacency_forbidden 必须是 room_id 列表\n"
        "4. 不要输出 markdown 或解释文本"
    )


# ============================================================
# 两步 LLM 调用
# ============================================================

async def _call_llm_step(
    client,
    messages: List[ChatMessage],
    config: LLMConfig,
    step_name: str,
    max_retries: int = 3,
) -> str:
    """单步 LLM 调用（带重试）"""
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                await sleep(2.0 * attempt)

            text = await call_llm_with_retry(client, messages, config, on_log=None)
            return text

        except Exception as e:
            last_error = e
            logger.warning(f"{step_name} attempt {attempt + 1}/{max_retries} failed: {e}")
            continue

    if last_error:
        raise last_error
    raise RuntimeError(f"{step_name} 失败")


def _parse_building_envelope(text: str, provider: str, model: str) -> BuildingEnvelopeBase:
    obj = _robust_json_loads(text, provider, model)
    if obj is None:
        raise RuntimeError("Envelope pass: cannot extract valid JSON")
    forbidden = {"rooms", "target_area", "floor_total_area", "core_tube_area", "corridor_allowance_area"}
    found: List[str] = []

    def _scan(value: Any, path: str = "$") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if k in forbidden:
                    found.append(f"{path}.{k}")
                _scan(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                _scan(item, f"{path}[{i}]")

    _scan(obj)
    if found:
        raise RuntimeError(
            "Envelope pass must not output rooms, areas, core/corridor budgets, or target_area. "
            f"Forbidden fields: {', '.join(found[:8])}"
        )
    return BuildingEnvelopeBase.model_validate(obj)


def _preprocess_budgeted_allocation_obj(obj: dict, parse_warnings: List[str]) -> None:
    """Budget V2 cleanup without area scaling or filler-room injection."""
    floors = obj.get("floors", [])
    if not isinstance(floors, list):
        return
    adjacency_errors: List[str] = []
    for floor in floors:
        if not isinstance(floor, dict):
            continue
        rooms = floor.get("rooms", [])
        if not isinstance(rooms, list):
            floor["rooms"] = []
            rooms = []
        room_ids = {str(r.get("room_id", "")) for r in rooms if isinstance(r, dict)}
        for idx, room in enumerate(rooms):
            if not isinstance(room, dict):
                continue
            if not room.get("room_id"):
                room["room_id"] = f"room_{idx + 1:03d}"
            if "room_type" in room:
                room["room_type"] = _normalize_room_type(str(room.get("room_type") or ""))
            zone_val = str(room.get("zone", "public") or "public").lower().strip()
            room["zone"] = zone_val if zone_val in _VALID_ZONES else _ZONE_ALIASES.get(zone_val, "public")
            try:
                room["target_area"] = float(room.get("target_area", 0.0))
            except Exception:
                room["target_area"] = 0.0
            if "needs_window" in room and isinstance(room["needs_window"], str):
                room["needs_window"] = room["needs_window"].strip().lower() in {"true", "yes", "1"}
            for adj_key in ["adjacency_required", "adjacency_preferred", "adjacency_forbidden", "adjacency_tags"]:
                val = room.get(adj_key, [])
                if isinstance(val, str):
                    room[adj_key] = [val] if val else []
                elif not isinstance(val, list):
                    room[adj_key] = []
            for adj_key in ["adjacency_required", "adjacency_preferred", "adjacency_forbidden", "adjacency_tags"]:
                refs = list(room.get(adj_key, []) or [])
                invalid = [r for r in refs if r not in room_ids]
                if invalid:
                    adjacency_errors.append(
                        "Topology Validation Error: "
                        f"F{floor.get('floor_number', '?')} room {room.get('room_id', '?')} "
                        f"{adj_key} references missing room_id(s) {invalid}. "
                        "Only reference room_ids that exist on the same floor. "
                        "Do not modify any target_area values."
                    )
    if adjacency_errors:
        raise BudgetValidationError("\n".join(adjacency_errors))


def _parse_budgeted_allocation(text: str, provider: str, model: str) -> Tuple[BuildingAllocation, List[str]]:
    parse_warnings: List[str] = []
    obj = _robust_json_loads(text, provider, model)
    if obj is None:
        raise RuntimeError("Budgeted allocation pass: cannot extract valid JSON")
    if isinstance(obj, dict) and "floors" in obj:
        _preprocess_budgeted_allocation_obj(obj, parse_warnings)
    allocation = BuildingAllocation.model_validate(obj)
    _validate_building_floor_count(allocation)
    _normalize_allocation(allocation)
    _apply_adjacency_forbidden(allocation)
    return allocation, parse_warnings


def _apply_budget_to_allocation(allocation: BuildingAllocation, budget: BuildingAreaBudget) -> None:
    for floor in allocation.floors:
        fid = f"F{int(floor.floor_number)}"
        if fid not in budget.floors:
            continue
        b = budget.floors[fid]
        floor.floor_total_area = float(b.floor_total_area)
        floor.core_tube_area = float(b.core_tube_area)
        floor.corridor_allowance_area = float(b.corridor_allowance_area)
    allocation.overall_total_area = float(sum(f.floor_total_area for f in allocation.floors))


def validate_allocation_against_budget(
    allocation: BuildingAllocation,
    budget: BuildingAreaBudget,
    *,
    tolerance: float = 0.05,
) -> None:
    errors: List[str] = []
    for floor in allocation.floors:
        fid = f"F{int(floor.floor_number)}"
        if fid not in budget.floors:
            errors.append(f"{fid}: missing backend budget")
            continue
        b = budget.floors[fid]
        room_sum = float(sum(max(0.0, float(r.target_area)) for r in floor.rooms))
        if room_sum > float(b.room_sum_max) + tolerance:
            delta = room_sum - float(b.room_sum_max)
            logger.error(
                "[VALIDATION] Rule: BUDGET_RANGE | Result: FAIL | Floor=%s | Current=%.2fm2 | Max=%.2fm2 | Delta=%.2fm2",
                fid,
                room_sum,
                float(b.room_sum_max),
                delta,
            )
            errors.append(
                f"Budget Validation Error: {fid} budget overload: current room total is {room_sum:.2f}m2, "
                f"but physical max is {float(b.room_sum_max):.2f}m2. "
                f"You must reduce at least {delta:.2f}m2 in one revision. "
                "Shrink secondary bedrooms, bathrooms, and storage first, or remove non-required storage. "
                "Do not delete explicit user-requested rooms."
            )
        elif room_sum < float(b.room_sum_min) - tolerance:
            delta = float(b.room_sum_min) - room_sum
            logger.error(
                "[VALIDATION] Rule: BUDGET_RANGE | Result: FAIL | Floor=%s | Current=%.2fm2 | Min=%.2fm2 | Delta=%.2fm2",
                fid,
                room_sum,
                float(b.room_sum_min),
                delta,
            )
            errors.append(
                f"Budget Validation Error: {fid} budget underfill: current room total is {room_sum:.2f}m2, "
                f"but physical min is {float(b.room_sum_min):.2f}m2. "
                f"You must add at least {delta:.2f}m2. "
                "Reasonably enlarge real requested rooms or add small storage/utility rooms. "
                "Do not create a huge filler utility room."
            )
        else:
            logger.info(
                "[VALIDATION] Rule: BUDGET_RANGE | Result: PASS | Floor=%s | Current=%.2fm2 | Min=%.2fm2 | Max=%.2fm2",
                fid,
                room_sum,
                float(b.room_sum_min),
                float(b.room_sum_max),
            )
    if errors:
        raise BudgetValidationError("\n".join(errors))


def validate_allocation_adjacency_ids(allocation: BuildingAllocation) -> None:
    errors: List[str] = []
    for floor in allocation.floors:
        fid = f"F{int(floor.floor_number)}"
        room_ids = {str(r.room_id) for r in floor.rooms}
        for room in floor.rooms:
            rid = str(room.room_id)
            for adj_key in ("adjacency_required", "adjacency_preferred", "adjacency_forbidden", "adjacency_tags"):
                refs = list(getattr(room, adj_key, []) or [])
                invalid = [str(ref) for ref in refs if str(ref) not in room_ids]
                if invalid:
                    logger.error(
                        "[VALIDATION] Rule: ADJACENCY_IDS | Result: FAIL | Floor=%s | Room=%s | Field=%s | Missing=%s",
                        fid,
                        rid,
                        adj_key,
                        invalid,
                    )
                    errors.append(
                        "Topology Validation Error: "
                        f"{fid} room {rid} {adj_key} references missing room_id(s) {invalid}. "
                        "Only reference room_ids that exist on the same floor. "
                        "Do not modify any target_area values."
                    )
    if not errors:
        logger.info("[VALIDATION] Rule: ADJACENCY_IDS | Result: PASS | Floors=%d", len(allocation.floors))
    if errors:
        raise BudgetValidationError("\n".join(errors))


async def generate_building_envelope(
    request: GenerateSemanticsRequest,
    on_llm_output: Optional[Callable[[str, str], None]] = None,
) -> BuildingEnvelopeBase:
    provider, model, api_key = _pick_provider_model_and_key(request)
    client = create_llm_client(provider)  # type: ignore[arg-type]
    llm_config = LLMConfig(apiKey=api_key, model=model, temperature=0.1, maxTokens=BUILDING_SEMANTIC_MAX_TOKENS)
    base_messages = [
        ChatMessage(role="system", content=BUILDING_ENVELOPE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=_build_envelope_user_prompt(request)),
    ]
    log_multiline_debug(
        logger,
        "[LLM]",
        "Envelope prompt messages.",
        json.dumps([{"role": m.role, "content": m.content} for m in base_messages], ensure_ascii=False, indent=2),
        "PROMPT",
    )
    last_error: Optional[Exception] = None
    for attempt in range(3):
        messages = list(base_messages)
        if attempt > 0 and last_error is not None:
            feedback = _extract_validation_errors(last_error)
            log_multiline_debug(logger, "[LLM]", f"Envelope validation feedback attempt={attempt + 1}.", feedback, "VALIDATION_FEEDBACK")
            messages.append(ChatMessage(role="user", content=feedback))
            await sleep(2.0 * attempt)
        try:
            text = await call_llm_with_retry(client, messages, llm_config, on_log=None)
            log_multiline_debug(logger, "[LLM]", f"Envelope raw response attempt={attempt + 1}.", text, "LLM_JSON")
            if on_llm_output is not None:
                on_llm_output(f"building_semantics.envelope.attempt_{attempt + 1}", text)
            return _parse_building_envelope(text, provider, model)
        except (JSONDecodeError, ValidationError, RuntimeError) as e:
            last_error = e
            logger.warning(f"Envelope attempt {attempt + 1}/3 failed: {type(e).__name__}: {str(e)[:240]}")
            continue
    if last_error:
        raise last_error
    raise RuntimeError("Envelope generation failed")


async def generate_budgeted_building_semantics(
    request: GenerateSemanticsRequest,
    *,
    floor_boundary,
    corridor_width: float,
    core_area_ratio: float,
    corridor_layout: str,
    base_seed: Optional[int] = None,
    fixed_core_tube: Optional[Any] = None,
    envelope: Optional[BuildingEnvelopeBase] = None,
    budget: Optional[BuildingAreaBudget] = None,
    repair_instruction: Optional[str] = None,
    on_llm_output: Optional[Callable[[str, str], None]] = None,
) -> BudgetedBuildingSemanticResult:
    provider, model, api_key = _pick_provider_model_and_key(request)
    envelope = envelope or await generate_building_envelope(request, on_llm_output=on_llm_output)
    if budget is None:
        budget = compute_building_area_budget(
            floor_boundary=floor_boundary,
            floors=envelope.floors,
            corridor_width=float(corridor_width),
            core_area_ratio=float(core_area_ratio),
            corridor_layout=corridor_layout,
            base_seed=base_seed,
            fixed_core_tube=fixed_core_tube,
        )
    client = create_llm_client(provider)  # type: ignore[arg-type]
    llm_config = LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=BUILDING_SEMANTIC_MAX_TOKENS)
    base_messages = [
        ChatMessage(role="system", content=BUILDING_BUDGETED_ALLOCATION_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=_build_budgeted_allocation_user_prompt(request, envelope, budget, repair_instruction=repair_instruction),
        ),
    ]
    log_multiline_debug(
        logger,
        "[LLM]",
        "Budgeted allocation prompt messages.",
        json.dumps([{"role": m.role, "content": m.content} for m in base_messages], ensure_ascii=False, indent=2),
        "PROMPT",
    )
    parse_warnings: List[str] = []
    last_error: Optional[Exception] = None
    for attempt in range(3):
        messages = list(base_messages)
        if attempt > 0 and last_error is not None:
            feedback = str(last_error)
            log_multiline_debug(
                logger,
                "[LLM]",
                f"Budgeted allocation validation feedback attempt={attempt + 1}.",
                feedback,
                "VALIDATION_FEEDBACK",
            )
            messages.append(ChatMessage(role="user", content=feedback))
            await sleep(2.0 * attempt)
        try:
            text = await call_llm_with_retry(client, messages, llm_config, on_log=None)
            log_multiline_debug(
                logger,
                "[LLM]",
                f"Budgeted allocation raw response attempt={attempt + 1}.",
                text,
                "LLM_JSON",
            )
            if on_llm_output is not None:
                on_llm_output(f"building_semantics.budgeted_allocation.attempt_{attempt + 1}", text)
            allocation, warnings = _parse_budgeted_allocation(text, provider, model)
            parse_warnings.extend(warnings)
            _apply_budget_to_allocation(allocation, budget)
            validate_allocation_adjacency_ids(allocation)
            validate_allocation_against_budget(allocation, budget)
            return BudgetedBuildingSemanticResult(
                allocation=allocation,
                warnings=parse_warnings,
                envelope=envelope,
                budget=budget,
                topology_snapshot=budget.topology_snapshot,
            )
        except (JSONDecodeError, ValidationError, RuntimeError, BudgetValidationError) as e:
            last_error = e
            logger.warning(f"Budgeted allocation attempt {attempt + 1}/3 failed: {type(e).__name__}: {str(e)[:400]}")
            continue
    if last_error:
        raise last_error
    raise RuntimeError("Budgeted allocation generation failed")


async def _two_step_generate(
    request: GenerateSemanticsRequest,
    provider: str,
    model: str,
    api_key: str,
    on_llm_output: Optional[Callable[[str, str], None]] = None,
) -> Tuple[BuildingAllocation, List[str]]:
    """
    两步 LLM 调用：

    Step 1: 房间清单骨架（4 字段/room）
    Step 2: 补充 size_hint + adjacency_required（2 字段/room）

    每步独立校验 + 重试。Step 2 失败不回退 Step 1。
    """
    client = create_llm_client(provider)  # type: ignore[arg-type]
    llm_config = LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=BUILDING_SEMANTIC_MAX_TOKENS)
    parse_warnings: List[str] = []

    # ---- Step 1: 房间清单骨架 ----
    step1_messages = [
        ChatMessage(role="system", content=BUILDING_STEP1_SYSTEM_PROMPT),
        ChatMessage(role="user", content=_build_step1_user_prompt(request)),
    ]

    last_error: Optional[Exception] = None
    step1_obj: Optional[dict] = None

    for attempt in range(3):
        try:
            if attempt > 0 and last_error:
                step1_messages.append(ChatMessage(
                    role="user",
                    content=_extract_validation_errors(last_error),
                ))
                await sleep(2.0 * attempt)

            text = await call_llm_with_retry(client, step1_messages, llm_config, on_log=None)
            if on_llm_output is not None:
                on_llm_output(f"building_semantics.step1.attempt_{attempt + 1}", text)
            obj = _robust_json_loads(text, provider, model)
            if obj is None:
                raise RuntimeError("Step 1: 无法提取 JSON")

            # 预处理 + 基本校验
            if "floors" in obj:
                _preprocess_allocation(obj, parse_warnings)

            # 验证骨架可以通过 Pydantic（target_area 可能是默认值 20.0）
            _validate_building_floor_count(BuildingAllocation.model_validate(obj))
            step1_obj = obj
            break

        except (JSONDecodeError, ValidationError, RuntimeError) as e:
            last_error = e
            logger.warning(f"Step 1 attempt {attempt + 1}/3 failed: {type(e).__name__}: {str(e)[:200]}")
            continue

    if step1_obj is None:
        if last_error:
            raise last_error
        raise RuntimeError("Step 1 语义规划失败")

    # ---- Step 2: 补充 size_hint + adjacency_required ----
    step1_json_str = json.dumps(step1_obj, ensure_ascii=False, indent=2)

    step2_messages = [
        ChatMessage(role="system", content=BUILDING_STEP2_SYSTEM_PROMPT),
        ChatMessage(role="user", content=_build_step2_user_prompt(step1_json_str)),
    ]

    last_error = None
    step2_obj: Optional[dict] = None

    for attempt in range(3):
        try:
            if attempt > 0 and last_error:
                step2_messages.append(ChatMessage(
                    role="user",
                    content=(
                        f"上一次输出有误：{str(last_error)[:200]}\n"
                        "请重新输出完整 JSON，确保每个 room 都有 size_hint 和 adjacency_required 字段。"
                    ),
                ))
                await sleep(2.0 * attempt)

            text = await call_llm_with_retry(client, step2_messages, llm_config, on_log=None)
            if on_llm_output is not None:
                on_llm_output(f"building_semantics.step2.attempt_{attempt + 1}", text)
            obj = _robust_json_loads(text, provider, model)
            if obj is None:
                raise RuntimeError("Step 2: 无法提取 JSON")

            if "floors" in obj:
                _preprocess_allocation(obj, parse_warnings)

            _validate_building_floor_count(BuildingAllocation.model_validate(obj))
            step2_obj = obj
            break

        except (JSONDecodeError, ValidationError, RuntimeError) as e:
            last_error = e
            logger.warning(f"Step 2 attempt {attempt + 1}/3 failed: {type(e).__name__}: {str(e)[:200]}")
            continue

    # Step 2 失败时退回 Step 1 结果（降级）
    if step2_obj is None:
        logger.warning("Step 2 failed after 3 attempts, using Step 1 result with defaults")
        parse_warnings.append("Step 2 失败，使用 Step 1 骨架 + 默认值")
        final_obj = step1_obj
    else:
        final_obj = step2_obj

    # --- 最终解析 ---
    if isinstance(final_obj, dict) and "floors" in final_obj:
        final_obj, area_warnings = normalize_floor_room_areas(final_obj, min_room_area=4.0)
        parse_warnings.extend(area_warnings)
    allocation = BuildingAllocation.model_validate(final_obj)
    _validate_building_floor_count(allocation)
    _normalize_allocation(allocation)
    _apply_size_hints_to_allocation(allocation)
    _apply_adjacency_forbidden(allocation)

    return allocation, parse_warnings


# ============================================================
# 主入口
# ============================================================

async def generate_building_semantics(
    request: GenerateSemanticsRequest,
    use_two_step: bool = False,
    on_llm_output: Optional[Callable[[str, str], None]] = None,
) -> Tuple[BuildingAllocation, List[str]]:
    """
    Building 模式：生成 BuildingAllocation + parse_warnings。

    Args:
        request: 语义规划请求
        use_two_step: 是否使用两步 LLM 调用（默认 False）

    Returns:
        (allocation, parse_warnings)
    """
    provider, model, api_key = _pick_provider_model_and_key(request)

    if use_two_step:
        return await _two_step_generate(request, provider, model, api_key, on_llm_output=on_llm_output)

    # ---- 单步模式（兼容旧管线）----
    client = create_llm_client(provider)  # type: ignore[arg-type]

    base_messages = [
        ChatMessage(role="system", content=BUILDING_PLANNER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=_build_user_prompt(request)),
    ]

    last_error: Optional[Exception] = None

    for attempt in range(3):
        messages = list(base_messages)

        if attempt > 0 and last_error:
            messages.append(ChatMessage(
                role="user",
                content=_extract_validation_errors(last_error),
            ))

        try:
            if attempt > 0:
                await sleep(2.0 * attempt)

            text = await call_llm_with_retry(
                client,
                messages,
                LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=BUILDING_SEMANTIC_MAX_TOKENS),
                on_log=None,
            )
            if on_llm_output is not None:
                on_llm_output(f"building_semantics.single_step.attempt_{attempt + 1}", text)
            allocation, warnings = _parse_building_allocation(
                text, provider=client.providerName, model=model,
            )
            _apply_adjacency_forbidden(allocation)
            return allocation, warnings

        except (JSONDecodeError, ValidationError, RuntimeError) as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1}/3 failed: {type(e).__name__}: {str(e)[:200]}")
            continue

    if last_error:
        raise last_error
    raise RuntimeError("Building 语义规划失败。")
