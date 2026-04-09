from __future__ import annotations

import json
import logging
import re
from asyncio import sleep
from json import JSONDecodeError
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from ...models import BuildingAllocation, GenerateSemanticsRequest
from ...settings import settings
from ..llm.provider import ChatMessage, LLMConfig, create_llm_client
from ..llm.response_parser import ParseOptions, extract_json, parse_ai_response
from ..llm.retry import call_llm_with_retry
from ..prompts.building_prompts import (
    BUILDING_PLANNER_SYSTEM_PROMPT,
    BUILDING_STEP1_SYSTEM_PROMPT,
    BUILDING_STEP2_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_BY_PROVIDER: Dict[str, str] = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
}


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

    # --- Stage 3: Pydantic 校验 ---
    try:
        allocation = BuildingAllocation.model_validate(obj)
    except ValidationError as e:
        logger.error(f"Pydantic validation failed after preprocessing: {e}")
        raise

    # --- Stage 4: 后处理 ---
    _normalize_allocation(allocation)

    return allocation, parse_warnings


def _preprocess_allocation(obj: dict, parse_warnings: List[str]) -> None:
    """Pydantic 前智能预处理"""
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


async def _two_step_generate(
    request: GenerateSemanticsRequest,
    provider: str,
    model: str,
    api_key: str,
) -> Tuple[BuildingAllocation, List[str]]:
    """
    两步 LLM 调用：

    Step 1: 房间清单骨架（4 字段/room）
    Step 2: 补充 size_hint + adjacency_required（2 字段/room）

    每步独立校验 + 重试。Step 2 失败不回退 Step 1。
    """
    client = create_llm_client(provider)  # type: ignore[arg-type]
    llm_config = LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=4096)
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
            obj = _robust_json_loads(text, provider, model)
            if obj is None:
                raise RuntimeError("Step 1: 无法提取 JSON")

            # 预处理 + 基本校验
            if "floors" in obj:
                _preprocess_allocation(obj, parse_warnings)

            # 验证骨架可以通过 Pydantic（target_area 可能是默认值 20.0）
            BuildingAllocation.model_validate(obj)
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
            obj = _robust_json_loads(text, provider, model)
            if obj is None:
                raise RuntimeError("Step 2: 无法提取 JSON")

            if "floors" in obj:
                _preprocess_allocation(obj, parse_warnings)

            BuildingAllocation.model_validate(obj)
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
    allocation = BuildingAllocation.model_validate(final_obj)
    _normalize_allocation(allocation)
    _apply_size_hints_to_allocation(allocation)
    _apply_adjacency_forbidden(allocation)

    return allocation, parse_warnings


# ============================================================
# 主入口
# ============================================================

async def generate_building_semantics(
    request: GenerateSemanticsRequest,
    use_two_step: bool = True,
) -> Tuple[BuildingAllocation, List[str]]:
    """
    Building 模式：生成 BuildingAllocation + parse_warnings。

    Args:
        request: 语义规划请求
        use_two_step: 是否使用两步 LLM 调用（默认 True）

    Returns:
        (allocation, parse_warnings)
    """
    provider, model, api_key = _pick_provider_model_and_key(request)

    if use_two_step:
        return await _two_step_generate(request, provider, model, api_key)

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
                LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=4096),
                on_log=None,
            )
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
