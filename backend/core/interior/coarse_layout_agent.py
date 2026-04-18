from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from pydantic import ValidationError

from .models import FurnitureSpec, LLMCoarseLayout, Obstacle, RoomBoundary
from ..llm.provider import ChatMessage, LLMConfig, LLMOutputFormatError
from ..llm.response_parser import extract_json
from ..llm.retry import call_llm_with_retry
from ...settings import settings


class AsyncOpenAI(Protocol):
    providerName: str

    async def chat(self, messages: List[ChatMessage], config: LLMConfig) -> str:
        raise NotImplementedError


_DEFAULT_MODEL_BY_PROVIDER: Dict[str, str] = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
}


def _key_for(provider: str) -> Optional[str]:
    if provider == "openai":
        return settings.openai_api_key
    if provider == "gemini":
        return settings.gemini_api_key
    if provider == "deepseek":
        return settings.deepseek_api_key
    return None


def _is_finite_number(x: float) -> bool:
    return math.isfinite(float(x))


def _point_in_rect(cx: float, cy: float, x_min: float, y_min: float, x_max: float, y_max: float) -> bool:
    return x_min <= cx <= x_max and y_min <= cy <= y_max


def _shrink_room(room: RoomBoundary, margin: float) -> Optional[Tuple[float, float, float, float]]:
    x0 = room.x_min + margin
    y0 = room.y_min + margin
    x1 = room.x_max - margin
    y1 = room.y_max - margin
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def _build_system_prompt() -> str:
    return "\n".join([
        "你是一个顶级的室内空间规划 AI 代理。",
        "",
        "【当前管线状态 (Pipeline Context)】",
        "目前系统已经完成了第一阶段的“楼层划分”与“门窗生成”。房间的物理边界（墙体）和障碍物（门窗开启区域）已经是锁定不可变的绝对真理。",
        "你现在的任务处于管线的第二阶段：房间内家具填充。你需要根据传入的房间边界和障碍物，为给定的家具列表分配初始的中心点坐标 (cx, cy) 和正交旋转角度 (rotation)。",
        "",
        "【你的定位与后处理机制】",
        "你输出的是启发式热启动初值，稍后将被送入后端的 MIQP 连续空间物理引擎进行亚毫米级微调。",
        "因此，你必须输出一个物理可行的初值：不重叠、不越界、避让门窗。后端只做小幅微调，不负责修复重大碰撞。",
        "",
        "【绝对规则】",
        "1) 坐标系：左下角为 (0.0, 0.0)。输入坐标是全局绝对坐标。",
        "2) 锚点法则：坐标 (cx, cy) 代表家具的几何中心点，禁止把中心点放到墙外或明显贴在墙边。",
        "3) 类别限制：家具必须归属于【床具, 坐具, 电器, 柜子, 桌子, 椅子, 挂件, 摆件】八个分类之一；禁止臆造任何清单之外的家具或建筑元素。",
        "4) 门窗避让：家具中心点不得落在任何障碍物区域内（Obstacle）。边缘轻微蹭到由 MIQP 推挤修复。",
        "5) rotation 只能是 0, 90, 180, 270 四个值之一。",
        "6) 面积自检：你在输出前必须自行计算所有家具面积之和，确保 sum(area_furniture) <= area_room * 0.4（预留过道/开门空间）。",
        "7) 不重叠：任意两件家具的矩形外接框不得相交（允许贴边，不允许重叠）。",
        "",
        "【输出格式】",
        "只输出一个 JSON 对象，禁止 markdown、禁止额外字段。",
        "reasoning 字段必须是 3-6 条要点（单个字符串，多行以 - 开头）。",
        "items 必须覆盖所有输入的 furniture_id，且不得新增、不得遗漏、不得重复。",
    ])


def _build_user_prompt(room: RoomBoundary, furnitures: Sequence[FurnitureSpec], obstacles: Sequence[Obstacle]) -> str:
    lines: List[str] = []
    lines.append("请为以下房间生成 LLMCoarseLayout 的 JSON：")
    lines.append("")
    lines.append(f"RoomBoundary: x:[{room.x_min}, {room.x_max}], y:[{room.y_min}, {room.y_max}]")
    lines.append("")
    lines.append("Furnitures (必须全部摆放，且只摆放这些)：")
    for f in furnitures:
        lines.append(f"- id={f.id} | name={f.name} | category={f.category.value} | size={f.width}x{f.height}")
    lines.append("")
    lines.append("Obstacles (中心点不得落入以下区域)：")
    if obstacles:
        for o in obstacles:
            lines.append(f"- name={o.name} | x:[{o.x_min}, {o.x_max}] y:[{o.y_min}, {o.y_max}]")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("输出 JSON 字段：")
    lines.append('{"reasoning": "...", "items": [{"furniture_id": "...", "cx": 0.0, "cy": 0.0, "rotation": 0}] }')
    return "\n".join(lines)


def _validate_layout_coarsely(
    layout: LLMCoarseLayout,
    room: RoomBoundary,
    furnitures: Sequence[FurnitureSpec],
    obstacles: Sequence[Obstacle],
    soft_margin: float = 0.5,
) -> List[str]:
    warnings: List[str] = []
    expected_ids = [f.id for f in furnitures]
    expected_set = set(expected_ids)
    got_ids = [it.furniture_id for it in layout.items]
    got_set = set(got_ids)

    if len(layout.items) != len(expected_ids):
        raise LLMOutputFormatError(f"items 数量不匹配：expected={len(expected_ids)} got={len(layout.items)}")
    if got_set != expected_set or len(got_ids) != len(got_set):
        raise LLMOutputFormatError("furniture_id 集合不一致或存在重复/遗漏")

    safe_inner = _shrink_room(room, soft_margin)

    for it in layout.items:
        if not (_is_finite_number(it.cx) and _is_finite_number(it.cy)):
            raise LLMOutputFormatError("cx/cy 非法（NaN/Inf）")

        if not _point_in_rect(it.cx, it.cy, room.x_min, room.y_min, room.x_max, room.y_max):
            raise LLMOutputFormatError(f"中心点严重出界：furniture_id={it.furniture_id} cx={it.cx} cy={it.cy}")

        for o in obstacles:
            if _point_in_rect(it.cx, it.cy, o.x_min, o.y_min, o.x_max, o.y_max):
                raise LLMOutputFormatError(f"中心点堵塞障碍物：furniture_id={it.furniture_id} obstacle={o.name}")

        if safe_inner is not None:
            x0, y0, x1, y1 = safe_inner
            if not _point_in_rect(it.cx, it.cy, x0, y0, x1, y1):
                warnings.append(f"soft-boundary: {it.furniture_id} center near boundary (margin={soft_margin})")

    return warnings


async def generate_coarse_layout(
    room: RoomBoundary,
    furnitures: list[FurnitureSpec],
    obstacles: list[Obstacle],
    client: AsyncOpenAI,
    model: Optional[str] = None,
) -> LLMCoarseLayout:
    provider = getattr(client, "providerName", None) or "openai"
    provider = str(provider).lower()
    api_key = _key_for(provider)
    if not api_key:
        raise RuntimeError(f"缺少 provider={provider} 的 API Key，无法生成 coarse layout。")

    picked_model = (model or "").strip() or _DEFAULT_MODEL_BY_PROVIDER.get(provider)
    if not picked_model:
        raise RuntimeError(f"provider={provider} 未配置默认 model。")

    llm_config = LLMConfig(apiKey=api_key, model=picked_model, temperature=0.2, maxTokens=2048)

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(room, furnitures, obstacles)

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    raw = await call_llm_with_retry(client, messages, llm_config)
    text = extract_json(raw)
    try:
        obj = json.loads(text)
    except Exception as e:
        raise LLMOutputFormatError(f"JSON 解析失败：{type(e).__name__}: {e}") from e

    try:
        layout = LLMCoarseLayout.model_validate(obj)
    except ValidationError as e:
        raise LLMOutputFormatError(f"LLMCoarseLayout 校验失败：{e}") from e

    warnings = _validate_layout_coarsely(layout, room, furnitures, obstacles, soft_margin=0.5)
    if warnings and warnings[0]:
        layout.reasoning = (layout.reasoning.rstrip() + "\n\n" + "\n".join(f"- {w}" for w in warnings)).strip()

    return layout
