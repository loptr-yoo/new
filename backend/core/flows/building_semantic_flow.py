from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from ...models import BuildingAllocation, GenerateSemanticsRequest
from ...settings import settings
from ..llm.provider import ChatMessage, LLMConfig, create_llm_client
from ..llm.response_parser import ParseOptions, extract_json, parse_ai_response
from ..llm.retry import call_llm_with_retry
from ..prompts.building_prompts import BUILDING_PLANNER_SYSTEM_PROMPT


_DEFAULT_MODEL_BY_PROVIDER: Dict[str, str] = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-pro",
    "deepseek": "deepseek-chat",
}


def _pick_provider_model_and_key(request: GenerateSemanticsRequest) -> Tuple[str, str, str]:
    """
    为 Building 语义规划挑选 provider/model/api_key。

    约束：
    - request.provider / request.model 可选；未提供则按后端可用 key 自动挑选。
    - 若指定了 provider 但 key 缺失，直接报错（避免悄悄降级导致结果不可控）。
    """

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
        raise RuntimeError("未配置任何可用的 LLM API Key（OPENAI/GEMINI/DEEPSEEK），无法执行语义规划。")

    api_key = _key_for(provider)
    if not api_key:
        raise RuntimeError(f"缺少 provider={provider} 的 API Key，无法执行语义规划。")

    model = request.model or _DEFAULT_MODEL_BY_PROVIDER.get(provider)
    if not model:
        raise RuntimeError(f"provider={provider} 未提供 model，且后端缺少默认 model 映射。")

    return provider, model, api_key


def _build_user_prompt(request: GenerateSemanticsRequest) -> str:
    """
    将请求体转换为 LLM 的 user prompt（只注入约束，不引入几何坐标概念）。
    """

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


def _parse_building_allocation(text: str, provider: str, model: str) -> BuildingAllocation:
    """
    将 LLM 返回文本解析为 BuildingAllocation。

    解析策略：
    1) 优先 extract_json + json.loads
    2) 失败时走 parse_ai_response（包含简单修复策略）
    3) 最终用 Pydantic 做 schema 校验
    """

    extracted = extract_json(text)
    try:
        obj = json.loads(extracted)
    except JSONDecodeError:
        obj = parse_ai_response(text, ParseOptions(provider=provider, model=model))
    return BuildingAllocation.model_validate(obj)


async def generate_building_semantics(request: GenerateSemanticsRequest) -> BuildingAllocation:
    """
    Building 模式专属：生成“纯语义”的 BuildingAllocation。

    注意：
    - 这里输出只包含逻辑/面积配比，不包含任何几何坐标（x/y/w/h）。
    - 解析阶段可能出现两类常见错误：
      1) JSONDecodeError：模型输出混入解释文本/截断
      2) ValidationError：字段缺失、类型错误、额外字段等导致 schema 不通过
    - 我们会在上述错误出现时进行有限次数的“提示修复重试”。
    """

    provider, model, api_key = _pick_provider_model_and_key(request)
    client = create_llm_client(provider)  # type: ignore[arg-type]

    base_messages = [
        ChatMessage(role="system", content=BUILDING_PLANNER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=_build_user_prompt(request)),
    ]

    last_error: Optional[Exception] = None
    for attempt in range(2):
        messages = list(base_messages)
        if attempt > 0:
            # 结构化重试提示（仅注释说明：真实产品可进一步加入“错误摘要→约束强化→重试”策略）
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "你上一次的输出无法被后端解析/校验。请严格按 BuildingAllocation JSON Schema 输出："
                        "只输出 JSON 对象；不要 markdown；不要解释；不要多余字段；数值字段必须是数字。"
                    ),
                )
            )

        try:
            text = await call_llm_with_retry(
                client,
                messages,
                LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=4096),
                on_log=None,
            )
            return _parse_building_allocation(text, provider=client.providerName, model=model)
        except (JSONDecodeError, ValidationError, RuntimeError) as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise RuntimeError("Building 语义规划失败：未知错误。")

