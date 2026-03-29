from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional


Provider = Literal["gemini", "deepseek", "openai"]


@dataclass(frozen=True)
class ParseOptions:
    provider: Provider
    model: str
    strictMode: bool = False


TYPE_MAPPING: Dict[str, str] = {
    "GROUND": "ground",
    "ground": "ground",
    "ROAD": "driving_lane",
    "road": "driving_lane",
    "PARKING_SPACE": "parking_space",
    "parking_space": "parking_space",
    "SIDEWALK": "pedestrian_path",
    "sidewalk": "pedestrian_path",
    "pedestrian_path": "pedestrian_path",
    "RAMP": "slope",
    "ramp": "slope",
    "slope": "slope",
    "PILLAR": "pillar",
    "pillar": "pillar",
    "WALL": "wall",
    "wall": "wall",
    "ENTRANCE": "entrance",
    "entrance": "entrance",
    "EXIT": "exit",
    "exit": "exit",
    "STAIRCASE": "staircase",
    "staircase": "staircase",
    "ELEVATOR": "elevator",
    "elevator": "elevator",
    "CHARGING_STATION": "charging_station",
    "charging_station": "charging_station",
    "GUIDANCE_SIGN": "guidance_sign",
    "guidance_sign": "guidance_sign",
    "SAFE_EXIT": "safe_exit",
    "safe_exit": "safe_exit",
    "SPEED_BUMP": "deceleration_zone",
    "speed_bump": "deceleration_zone",
    "deceleration_zone": "deceleration_zone",
    "FIRE_EXTINGUISHER": "fire_extinguisher",
    "fire_extinguisher": "fire_extinguisher",
    "LANE_LINE": "ground_line",
    "lane_line": "ground_line",
    "ground_line": "ground_line",
    "CONVEX_MIRROR": "convex_mirror",
    "convex_mirror": "convex_mirror",
    "driving_lane": "driving_lane",
}


def normalize_layout_element_types(layout: Any) -> Any:
    if not isinstance(layout, dict):
        return layout
    elements = layout.get("elements")
    if not isinstance(elements, list):
        return layout
    out = dict(layout)
    out["elements"] = [dict(el, type=TYPE_MAPPING.get(el.get("type"), el.get("type"))) if isinstance(el, dict) else el for el in elements]
    return out


def extract_json(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        cleaned = cleaned[first : last + 1]
    return cleaned


def _balance_braces(s: str) -> str:
    open_count = 0
    for ch in s:
        if ch == "{":
            open_count += 1
        elif ch == "}":
            open_count -= 1
    if open_count > 0 and open_count < 50:
        return s + ("}" * open_count)
    return s


def _strip_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _try_parse_json(text: str) -> Any:
    return json.loads(text)


def _try_parse_literal(text: str) -> Any:
    s = text
    s = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfalse\b", "False", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnull\b", "None", s, flags=re.IGNORECASE)
    return ast.literal_eval(s)


def _parse_with_repair(text: str) -> Any:
    candidate = extract_json(text)
    candidate = _balance_braces(candidate)
    candidate = _strip_trailing_commas(candidate)
    try:
        return _try_parse_json(candidate)
    except Exception:
        pass
    try:
        return _try_parse_literal(candidate)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidate2 = _balance_braces(_strip_trailing_commas(m.group(0)))
        try:
            return _try_parse_json(candidate2)
        except Exception:
            return _try_parse_literal(candidate2)
    raise ValueError("Failed to parse response")


def parse_ai_response(text: str, options: ParseOptions) -> Any:
    if not isinstance(text, str):
        raise ValueError("Invalid response: expected string")
    try:
        if options.provider == "deepseek":
            return _parse_with_repair(text)
        if options.provider == "gemini":
            return _parse_with_repair(text)
        if options.provider == "openai":
            return _parse_with_repair(text)
        return _parse_with_repair(text)
    except Exception as e:
        raise RuntimeError(str(e)) from e


def validate_parking_layout(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if "width" not in data or "height" not in data or "elements" not in data:
        return False
    if not isinstance(data.get("width"), (int, float)) or not isinstance(data.get("height"), (int, float)):
        return False
    if not isinstance(data.get("elements"), list):
        return False
    if len(data["elements"]) > 0:
        first = data["elements"][0]
        if not isinstance(first, dict):
            return False
        if "x" not in first or "y" not in first or ("type" not in first and "t" not in first):
            return False
    return True


def safe_parse_response(
    text: str,
    options: ParseOptions,
    on_log: Optional[Callable[[str], None]] = None,
) -> Any:
    if on_log:
        on_log(f"[{options.provider}] 开始解析响应...")
        on_log(f"[{options.provider}] 响应长度: {len(text)} 字符")
    try:
        try:
            extracted = extract_json(text)
            snippet = extracted[:800] + "..." if len(extracted) > 800 else extracted
            if on_log:
                on_log(f"[{options.provider}] 提取 JSON 长度: {len(extracted)}")
                on_log(f"[{options.provider}] 提取 JSON 摘要: {snippet}")
        except Exception:
            pass

        parsed = parse_ai_response(text, options)
        if on_log:
            on_log(f"[{options.provider}] 解析成功")

        normalized = normalize_layout_element_types(parsed)
        if on_log:
            on_log(f"[{options.provider}] 类型转换完成")

        has_elements = isinstance(normalized, dict) and isinstance(normalized.get("elements"), list)
        has_patch = isinstance(normalized, dict) and (
            isinstance(normalized.get("modified_elements"), list)
            or isinstance(normalized.get("new_elements"), list)
            or isinstance(normalized.get("deleted_ids"), list)
        )

        if has_elements and not has_patch:
            if not validate_parking_layout(normalized):
                snapshot = json.dumps(normalized)[:2000]
                raise RuntimeError(f"数据结构不完整: {snapshot}")
        elif not has_elements and not has_patch:
            snapshot = json.dumps(normalized)[:2000]
            raise RuntimeError(f"响应缺少 elements/patch 字段: {snapshot}")

        return normalized
    except Exception as e:
        if on_log:
            on_log(f"[{options.provider}] 解析失败: {str(e)}")
            if len(text) > 1000:
                on_log(f"[{options.provider}] 响应摘要: {text[:200]}...")
            else:
                on_log(f"[{options.provider}] 完整响应: {text}")
        raise

