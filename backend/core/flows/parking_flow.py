from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from ...models import BuildingData, ConstraintViolation, LayoutElement, ParkingLayout
from ..geometry.ai_common_utils import (
    apply_scene_post_process,
    calculate_score,
    dedupe_patches_against_layout,
    enhance_layout_with_geometry,
    fill_voids_with_ground,
    fix_small_geometry,
    generate_auto_connectivity_patches,
    map_to_internal_layout,
    merge_patches_to_layout,
    normalize_partial_patches,
    normalize_type,
    post_process_layout,
)
from ..geometry.geometry import calculate_void_ratio, get_intersection_box, validate_layout
from ..llm.provider import ChatMessage, LLMClient, LLMConfig
from ..llm.response_parser import ParseOptions, safe_parse_response
from ..llm.retry import call_llm_with_retry
from ..prompts.building_prompts import CORE_ARCHITECT_PROMPT, FLOOR_DRAFTSMAN_PROMPT, MASTER_PLANNER_PROMPT
from ..prompts.prompts import PROMPTS
from ..scenes.scene_registry import DEFAULT_SCENE_ID, SCENE_REGISTRY
from ..scenes.scene_types import SceneDefinition
from ..types import ElementType
from ..utils.misc import sleep


MAX_FIX_PASSES = 4
_generation_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_MS = 1000 * 60 * 60


def get_active_scene(scene_id: Optional[str]) -> SceneDefinition:
    key = scene_id or DEFAULT_SCENE_ID
    return SCENE_REGISTRY.get(key) or SCENE_REGISTRY[DEFAULT_SCENE_ID]


def apply_scene_type_normalization(raw: Any, scene_id: Optional[str]) -> Any:
    scene = get_active_scene(scene_id)
    mapping = scene.elementNormalization or {}

    def normalize_one(el: Any) -> Any:
        if not isinstance(el, dict):
            return el
        raw_type = el.get("t") if el.get("t") is not None else el.get("type")
        if not raw_type:
            return el
        key = str(raw_type).lower()
        mapped = mapping.get(key)
        if not mapped:
            return el
        if "t" in el and el.get("t") is not None:
            out = dict(el)
            out["t"] = mapped
            return out
        out = dict(el)
        out["type"] = mapped
        return out

    def normalize_list(arr: Any) -> Any:
        if isinstance(arr, list):
            return [normalize_one(x) for x in arr]
        return arr

    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    out["elements"] = normalize_list(raw.get("elements"))
    out["new_elements"] = normalize_list(raw.get("new_elements"))
    out["modified_elements"] = normalize_list(raw.get("modified_elements"))
    return out


def _filter_parking_violations_by_scene(layout: ParkingLayout, violations: List[ConstraintViolation], scene: SceneDefinition) -> List[ConstraintViolation]:
    if scene.id == DEFAULT_SCENE_ID:
        v2 = [v for v in violations if v.elementId != "global_ground_check"]
        v2 = [v for v in v2 if ("auto_ground_void_" not in str(v.elementId or "")) and ("auto_ground_void_" not in str(v.targetId or ""))]
        return v2
    out: List[ConstraintViolation] = []
    for v in violations:
        if v.type != "overlap":
            out.append(v)
            continue
        if not v.targetId:
            out.append(v)
            continue
        a = next((e for e in layout.elements if e.id == v.elementId), None)
        b = next((e for e in layout.elements if e.id == v.targetId), None)
        if a is None or b is None:
            out.append(v)
            continue
        box = get_intersection_box(a, b)
        if not box:
            out.append(v)
            continue
        if min(box["width"], box["height"]) > 9:
            out.append(v)
    return out


async def run_iterative_fix(
    layout: ParkingLayout,
    client: LLMClient,
    config: LLMConfig,
    scene: SceneDefinition,
    on_log: Optional[Callable[[str], None]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> ParkingLayout:
    options = options or {}
    current = layout
    last_score = float("inf")
    structural_types: Set[str] = {
        ElementType.ROAD,
        ElementType.GROUND,
        ElementType.RAMP,
        ElementType.ENTRANCE,
        ElementType.EXIT,
        ElementType.WALL_EXTERNAL,
        ElementType.SHEAR_WALL,
        ElementType.ELEVATOR_SHAFT,
        ElementType.ELEVATOR,
        ElementType.STAIRCASE,
    }

    for p in range(1, MAX_FIX_PASSES + 1):
        violations = validate_layout(current)
        violations = _filter_parking_violations_by_scene(current, violations, scene)
        score = calculate_score(violations)
        if violations and on_log:
            counts: Dict[str, int] = {}
            for v in violations:
                counts[v.type] = counts.get(v.type, 0) + 1
            summary = ", ".join([f"{k}={v}" for k, v in counts.items()])
            samples = ", ".join([f"{v.type}:{v.elementId}" for v in violations[:6]])
            on_log(f"🔎 违规统计: {summary}")
            on_log(f"🔎 违规样例: {samples}")

        if score == 0:
            if on_log:
                on_log(f"✅ [Fix Loop] 布局完美 (Pass {p}, Score 0)")
            break

        if score >= last_score and p > 1:
            if options.get("freezeStructural"):
                if on_log:
                    on_log(f"🛑 [Fix Loop] 优化停滞 (Score {score}), 停止修复")
                break
            auto_patches = generate_auto_connectivity_patches(current)
            if auto_patches:
                if on_log:
                    on_log(f"🧩 触发自动补丁: {len(auto_patches)} 个")
                cleaned = normalize_partial_patches(auto_patches)
                current = merge_patches_to_layout(current, cleaned, [], mode="strict")
                continue
            if on_log:
                on_log(f"🛑 [Fix Loop] 优化停滞 (Score {score}), 停止修复")
            break

        if on_log:
            on_log(f"🔧 [Fix Loop] 修复轮次 {p}/{MAX_FIX_PASSES} (Score: {score}, Violations: {len(violations)})")
        last_score = score

        simplified = [
            {
                "id": e.id,
                "t": e.type,
                "x": int(round(e.x)),
                "y": int(round(e.y)),
                "w": int(round(e.width)),
                "h": int(round(e.height)),
                "r": e.rotation,
            }
            for e in current.elements
        ]
        compact_layout: Dict[str, Any] = {"width": current.width, "height": current.height, "elements": simplified}

        try:
            fix_response = await call_llm_with_retry(
                client,
                [
                    ChatMessage(
                        role="user",
                        content=PROMPTS["fix"](compact_layout, violations, scene, {"frozenIds": options.get("frozenIds")}),
                    )
                ],
                LLMConfig(apiKey=config.apiKey, model=config.model, temperature=0.1, maxTokens=config.maxTokens),
                on_log,
            )
            parsed = safe_parse_response(
                fix_response,
                ParseOptions(provider=client.providerName, model=config.model),
                on_log,
            )

            patches: List[Dict[str, Any]] = []
            deleted_ids: List[str] = []
            new_patches: List[Dict[str, Any]] = []

            if isinstance(parsed, dict) and isinstance(parsed.get("modified_elements"), list):
                patches = parsed.get("modified_elements") or []
                deleted_ids = parsed.get("deleted_ids") or []
                raw_new = parsed.get("new_elements")
                new_patches = list(raw_new) if isinstance(raw_new, list) else []
                if on_log:
                    on_log(f"🔧 应用 {len(patches)} 个修复补丁，删除 {len(deleted_ids)} 个元素...")
            elif isinstance(parsed, dict) and isinstance(parsed.get("elements"), list):
                patches = parsed.get("elements") or []
                if on_log:
                    on_log(f"🔧 应用全量更新 ({len(patches)} 个元素)...")
            else:
                if on_log:
                    on_log("⚠️ AI 未返回有效修复，跳过此轮")
                continue

            if patches:
                if isinstance(parsed, dict) and ("modified_elements" in parsed) and ("elements" not in parsed):
                    cleaned = normalize_partial_patches(patches)
                    filtered_deletes = list(deleted_ids or [])
                    frozen_ids = options.get("frozenIds") or []
                    if frozen_ids:
                        frozen_set = set([str(x) for x in frozen_ids])
                        cleaned = [p for p in cleaned if str(p.get("id") or p.get("element_id") or "") not in frozen_set]
                        filtered_deletes = [d for d in filtered_deletes if d not in frozen_set]
                    elif options.get("freezeStructural"):
                        def is_allowed_patch(patch: Dict[str, Any]) -> bool:
                            pid = str(patch.get("id") or patch.get("element_id") or "")
                            existing = next((el for el in current.elements if el.id == pid), None)
                            t = normalize_type((existing.type if existing is not None else None) or (patch.get("type") or patch.get("t")))
                            return t not in structural_types
                        cleaned = [p for p in cleaned if is_allowed_patch(p)]
                        filtered_deletes2: List[str] = []
                        for did in filtered_deletes:
                            existing = next((el for el in current.elements if el.id == did), None)
                            t = normalize_type(existing.type if existing is not None else None)
                            if t not in structural_types:
                                filtered_deletes2.append(did)
                        filtered_deletes = filtered_deletes2

                    current = merge_patches_to_layout(current, cleaned, filtered_deletes, mode="strict")
                    if new_patches:
                        adds_layout = map_to_internal_layout({"width": current.width, "height": current.height, "elements": new_patches})
                        adds = [e.model_dump() for e in adds_layout.elements]
                        frozen_ids = options.get("frozenIds") or []
                        if frozen_ids:
                            frozen_set = set([str(x) for x in frozen_ids])
                            adds = [p for p in adds if str(p.get("id")) not in frozen_set]
                        elif options.get("freezeStructural"):
                            adds = [p for p in adds if normalize_type(p.get("type")) not in structural_types]
                        if adds:
                            current = merge_patches_to_layout(current, adds, [], mode="allowCreate")
                else:
                    if options.get("freezeStructural"):
                        if on_log:
                            on_log("⚠️ [Fix Loop] 忽略全量修复以保护骨架")
                    else:
                        full = map_to_internal_layout({"width": current.width, "height": current.height, "elements": patches})
                        current = ParkingLayout(width=current.width, height=current.height, elements=full.elements, sceneId=current.sceneId)
            else:
                if options.get("freezeStructural"):
                    continue
                auto_patches = generate_auto_connectivity_patches(current)
                if auto_patches:
                    if on_log:
                        on_log(f"🧩 模型返回空补丁，应用自动补丁 {len(auto_patches)} 个")
                    cleaned = normalize_partial_patches(auto_patches)
                    unique_auto = dedupe_patches_against_layout(current, cleaned, 5)
                    if unique_auto:
                        current = merge_patches_to_layout(current, unique_auto, [], mode="allowCreate")
        except Exception as e:
            if on_log:
                on_log(f"⚠️ 修复出错: {str(e)}")
            break

    return current


def validate_generated_content(layout: ParkingLayout) -> None:
    if len(layout.elements) < 5:
        raise RuntimeError(f"Incomplete generation: Only {len(layout.elements)} elements found. AI stopped early.")
    has_road = any((e.type == "driving_lane" or e.type == "ROAD") for e in layout.elements)
    if not has_road:
        raise RuntimeError("Incomplete generation: No roads detected. AI failed to generate layout interior.")


def validate_floor_plan_delta(new_elements: Sequence[LayoutElement], delta_layout: ParkingLayout) -> None:
    if len(new_elements) < 15:
        raise RuntimeError(f"Incomplete floor plan: Only {len(new_elements)} new elements found. AI stopped early.")
    has_walls = any(normalize_type(e.type) in (ElementType.WALL_EXTERNAL, ElementType.WALL_INTERNAL, ElementType.WALL) for e in delta_layout.elements)
    if not has_walls:
        raise RuntimeError("Incomplete floor plan: No walls detected.")
    has_door = any(normalize_type(e.type) == ElementType.DOOR for e in delta_layout.elements)
    if not has_door:
        raise RuntimeError("Incomplete floor plan: No doors detected.")


async def execute_floor_generation_with_core(
    prompt: str,
    core_blueprint: List[LayoutElement],
    client: LLMClient,
    api_key: str,
    model: str,
    on_log: Optional[Callable[[str], None]] = None,
    scene_id: Optional[str] = None,
) -> ParkingLayout:
    scene = get_active_scene(scene_id)
    is_parking_scene = scene.id == DEFAULT_SCENE_ID
    config = LLMConfig(apiKey=api_key, model=model, temperature=0.9, maxTokens=8192)
    if on_log:
        on_log(f" [{client.providerName}] 开始生成楼层...")

    floor_layout: Optional[ParkingLayout] = None
    last_error: Optional[Exception] = None
    MAX_GEN_RETRIES = 3
    for attempt in range(1, MAX_GEN_RETRIES + 1):
        try:
            if attempt > 1:
                if on_log:
                    on_log(f"🔄 生成重试 {attempt}/{MAX_GEN_RETRIES}: 上次结果不完整...")
                await sleep(1000)
            response_text = await call_llm_with_retry(
                client,
                [ChatMessage(role="user", content=FLOOR_DRAFTSMAN_PROMPT(prompt, core_blueprint, scene))],
                config,
                on_log,
            )
            raw_parsed = safe_parse_response(
                response_text,
                ParseOptions(provider=client.providerName, model=model),
                on_log,
            )
            raw_data = apply_scene_type_normalization(raw_parsed, scene_id)
            if isinstance(raw_data, dict) and raw_data.get("reasoning_plan") and on_log:
                on_log(f"🧠 Plan: {raw_data.get('reasoning_plan')}")

            delta_elements = []
            if isinstance(raw_data, dict) and isinstance(raw_data.get("new_elements"), list):
                delta_elements = raw_data.get("new_elements") or []
            elif isinstance(raw_data, dict) and isinstance(raw_data.get("elements"), list):
                delta_elements = raw_data.get("elements") or []

            delta_layout = map_to_internal_layout(
                {
                    "width": (raw_data.get("width") if isinstance(raw_data, dict) else None) or 800,
                    "height": (raw_data.get("height") if isinstance(raw_data, dict) else None) or 600,
                    "elements": delta_elements,
                }
            )
            core_ids = [e.id for e in core_blueprint]
            core_set = set(core_ids)
            seen: Set[str] = set()
            new_elements = []
            for e in delta_layout.elements:
                if not e.id:
                    continue
                if e.id in core_set:
                    continue
                if e.id in seen:
                    continue
                seen.add(e.id)
                new_elements.append(e)
            merged = ParkingLayout(width=delta_layout.width, height=delta_layout.height, elements=new_elements + core_blueprint)
            if is_parking_scene:
                validate_generated_content(merged)
            else:
                validate_floor_plan_delta(new_elements, delta_layout)
            floor_layout = merged
            if on_log:
                on_log(f"✅ 初步生成成功: 包含 {len(new_elements)} 个新增元素")
            break
        except Exception as e:
            last_error = e
            if on_log:
                on_log(f"⚠️ 生成质量校验失败: {str(e)}")
            if attempt == MAX_GEN_RETRIES:
                raise RuntimeError(f"生成失败，模型连续 {MAX_GEN_RETRIES} 次输出不完整: {str(e)}") from e

    if floor_layout is None:
        raise last_error or RuntimeError("Unknown floor generation error")

    frozen_ids = [e.id for e in core_blueprint]
    if not is_parking_scene:
        has_slab = any(e.type == ElementType.SLAB for e in floor_layout.elements)
        with_slab = (
            floor_layout
            if has_slab
            else ParkingLayout(
                width=floor_layout.width,
                height=floor_layout.height,
                sceneId=floor_layout.sceneId,
                elements=[
                    LayoutElement(id="floor_slab", type=ElementType.SLAB, x=0, y=0, width=800, height=600),
                    *floor_layout.elements,
                ],
            )
        )
        final_with_slab = post_process_layout(with_slab)
        return apply_scene_post_process(final_with_slab, scene, on_log, _core_elements=core_blueprint)

    fixed = await run_iterative_fix(
        floor_layout,
        client,
        LLMConfig(apiKey=api_key, model=model, temperature=0.1, maxTokens=8192),
        scene,
        on_log,
        {"freezeStructural": True, "frozenIds": frozen_ids},
    )
    fixed = fill_voids_with_ground(fixed)
    final_fixed = post_process_layout(fixed)
    return apply_scene_post_process(final_fixed, scene, on_log, _core_elements=core_blueprint)


async def execute_generation(
    prompt: str,
    client: LLMClient,
    api_key: str,
    model: str,
    on_log: Optional[Callable[[str], None]] = None,
    scene_id: Optional[str] = None,
) -> ParkingLayout:
    config = LLMConfig(apiKey=api_key, model=model, temperature=1.2, maxTokens=8192)
    if on_log:
        on_log(f" [{client.providerName}] 开始生成...")
    scene = get_active_scene(scene_id)
    is_parking_scene = scene.id == DEFAULT_SCENE_ID

    layout: Optional[ParkingLayout] = None
    last_error: Optional[Exception] = None
    MAX_GEN_RETRIES = 3
    for attempt in range(1, MAX_GEN_RETRIES + 1):
        try:
            if attempt > 1:
                if on_log:
                    on_log(f"🔄 生成重试 {attempt}/{MAX_GEN_RETRIES}: 上次结果不完整...")
                await sleep(1000)
            response_text = await call_llm_with_retry(
                client,
                [ChatMessage(role="user", content=PROMPTS["generation"](prompt, scene))],
                config,
                on_log,
            )
            raw_parsed = safe_parse_response(
                response_text,
                ParseOptions(provider=client.providerName, model=model),
                on_log,
            )
            raw_data = apply_scene_type_normalization(raw_parsed, scene_id)
            layout = map_to_internal_layout(raw_data if isinstance(raw_data, dict) else {"elements": []})
            if isinstance(raw_data, dict) and raw_data.get("reasoning_plan") and on_log:
                on_log(f"🧠 Plan: {raw_data.get('reasoning_plan')}")
            if is_parking_scene:
                validate_generated_content(layout)
            if on_log:
                on_log(f"✅ 初步生成成功: 包含 {len(layout.elements)} 个元素")
            break
        except Exception as e:
            last_error = e
            if on_log:
                on_log(f"⚠️ 生成质量校验失败: {str(e)}")
            if attempt == MAX_GEN_RETRIES:
                raise RuntimeError(f"生成失败，模型连续 {MAX_GEN_RETRIES} 次输出不完整: {str(e)}") from e

    if layout is None:
        raise last_error or RuntimeError("Unknown generation error")

    if not is_parking_scene:
        return layout

    layout = await run_iterative_fix(layout, client, config, scene, on_log)
    layout = fill_voids_with_ground(layout)
    void_ratio = calculate_void_ratio(layout)
    if void_ratio > 0.005 and on_log:
        on_log(f"❌ Void ratio too high before refinement: {(void_ratio * 100):.3f}%")
    return layout


async def execute_refinement(
    current_layout: ParkingLayout,
    client: LLMClient,
    api_key: str,
    model: str,
    on_log: Optional[Callable[[str], None]] = None,
    scene_id: Optional[str] = None,
) -> ParkingLayout:
    scene = get_active_scene(scene_id)
    is_parking_scene = scene.id == DEFAULT_SCENE_ID
    config = LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=8192)
    void_ratio = calculate_void_ratio(current_layout)
    if is_parking_scene and void_ratio > 0.005 and on_log:
        on_log(f"❌ Void ratio too high before refinement: {(void_ratio * 100):.3f}%")

    if on_log:
        on_log(f"✨ [{client.providerName}] 开始细化布局 (增量模式)...")

    if is_parking_scene:
        structural_elements = [e for e in current_layout.elements if e.type in ("WALL", "ROAD", "driving_lane", "GROUND", "ground", "wall")]
    else:
        structural_elements = list(current_layout.elements)

    simplified = [
        {"id": e.id, "t": e.type, "x": int(round(e.x)), "y": int(round(e.y)), "w": int(round(e.width)), "h": int(round(e.height))}
        for e in structural_elements
    ]

    try:
        response_text = await call_llm_with_retry(
            client,
            [
                ChatMessage(
                    role="user",
                    content=PROMPTS["optimizeSystemPrompt"]({"elements": simplified}, int(current_layout.width), int(current_layout.height), scene),
                )
            ],
            config,
            on_log,
        )
        parsed = safe_parse_response(
            response_text,
            ParseOptions(provider=client.providerName, model=model),
            on_log,
        )
        raw_data = apply_scene_type_normalization(parsed, scene_id)

        layout = current_layout
        if is_parking_scene and isinstance(raw_data, dict) and isinstance(raw_data.get("modified_elements"), list):
            if on_log:
                on_log(f"🩹 应用修改补丁: {len(raw_data.get('modified_elements') or [])} 个")
            updates = normalize_partial_patches(raw_data.get("modified_elements") or [])
            structural_types: Set[str] = {
                ElementType.WALL,
                ElementType.ROAD,
                ElementType.GROUND,
                ElementType.RAMP,
                ElementType.ENTRANCE,
                ElementType.EXIT,
                ElementType.WALL_EXTERNAL,
                ElementType.SHEAR_WALL,
                ElementType.ELEVATOR_SHAFT,
                ElementType.ELEVATOR,
                ElementType.STAIRCASE,
            }

            def allow_patch(patch: Dict[str, Any]) -> bool:
                pid = str(patch.get("id") or patch.get("element_id") or "")
                existing = next((el for el in current_layout.elements if el.id == pid), None)
                t = normalize_type((existing.type if existing is not None else None) or (patch.get("type") or patch.get("t")))
                return t not in structural_types

            updates = [p for p in updates if allow_patch(p)]

            filtered_deletes: List[str] = []
            for did in (raw_data.get("deleted_ids") or []):
                existing = next((el for el in current_layout.elements if el.id == did), None)
                t = normalize_type(existing.type if existing is not None else None)
                if t not in structural_types:
                    filtered_deletes.append(did)

            layout = merge_patches_to_layout(current_layout, updates, filtered_deletes, mode="strict")
        elif isinstance(raw_data, dict) and isinstance(raw_data.get("new_elements"), list):
            if on_log:
                on_log(f"➕ 应用新增元素: {len(raw_data.get('new_elements') or [])} 个")
            adds_layout = map_to_internal_layout({"width": current_layout.width, "height": current_layout.height, "elements": raw_data.get("new_elements") or []})
            adds = [e.model_dump() for e in adds_layout.elements]
            layout = merge_patches_to_layout(current_layout, adds, [], mode="allowCreate")
        elif isinstance(raw_data, dict) and isinstance(raw_data.get("elements"), list):
            if on_log:
                on_log(f"📥 应用全量更新: {len(raw_data.get('elements') or [])} 个元素")
            expanded = normalize_partial_patches(raw_data.get("elements") or [])
            normalized = map_to_internal_layout(
                {
                    "width": raw_data.get("width") or current_layout.width,
                    "height": raw_data.get("height") or current_layout.height,
                    "elements": expanded,
                }
            )
            layout = ParkingLayout(
                width=raw_data.get("width") or current_layout.width,
                height=raw_data.get("height") or current_layout.height,
                elements=normalized.elements,
                sceneId=current_layout.sceneId,
            )
        else:
            if on_log:
                on_log("⚠️ AI 未返回有效元素，跳过更新")
            return current_layout

        layout = fix_small_geometry(layout)
        if is_parking_scene:
            layout = await enhance_layout_with_geometry(layout, on_log)

        if on_log:
            on_log("🔧 执行最终冲突微调 (增量安全模式)...")
        if is_parking_scene:
            layout = await run_iterative_fix(
                layout,
                client,
                LLMConfig(apiKey=api_key, model=model, temperature=0.0, maxTokens=8192),
                scene,
                on_log,
                {"freezeStructural": True},
            )
        if on_log:
            on_log("✅ 细化流程全部完成")
        final_layout = post_process_layout(layout)
        return apply_scene_post_process(final_layout, scene, on_log)
    except Exception as e:
        if on_log:
            on_log(f"❌ 细化阶段出错: {str(e)}")
        return current_layout


async def generate_building(
    prompt: str,
    provider: str,
    model: str,
    api_key: str,
    on_log: Optional[Callable[[str], None]] = None,
    scene_id: Optional[str] = None,
) -> BuildingData:
    from ..llm.provider import create_llm_client

    scene = get_active_scene(scene_id)
    cache_key = f"{prompt.strip().lower()}|{model}|{scene_id or DEFAULT_SCENE_ID}"
    cached = _generation_cache.get(cache_key)
    now = int(__import__("time").time() * 1000)
    if cached and (now - int(cached.get("timestamp") or 0) < _CACHE_TTL_MS):
        if on_log:
            on_log("命中热点场景缓存，极速返回结果...")
        await sleep(500)
        return BuildingData.model_validate(cached["data"])

    client = create_llm_client(provider)  # type: ignore[arg-type]

    if scene.id != "building":
        layout = await execute_generation(prompt, client, api_key, model, on_log, scene_id)
        floor_id = "B1" if scene.id == DEFAULT_SCENE_ID else "1F"
        single = BuildingData(blueprint=[], floors={floor_id: layout.model_copy(update={"sceneId": scene.id})})
        _generation_cache[cache_key] = {"data": single.model_dump(), "timestamp": now}
        return single

    if on_log:
        on_log("正在规划楼层结构...")
    planner_text = await call_llm_with_retry(
        client,
        [ChatMessage(role="user", content=MASTER_PLANNER_PROMPT(prompt))],
        LLMConfig(apiKey=api_key, model=model, temperature=0.2, maxTokens=2048),
        on_log,
    )
    planner_parsed = safe_parse_response(planner_text, ParseOptions(provider=client.providerName, model=model), on_log)
    floors_raw = planner_parsed.get("floors") if isinstance(planner_parsed, dict) else None
    floors_raw = floors_raw if isinstance(floors_raw, list) else []
    floors = []
    for idx, f in enumerate(floors_raw):
        if not isinstance(f, dict):
            continue
        floors.append(
            {
                "id": str(f.get("id") or f"floor_{idx + 1}"),
                "name": str(f.get("name") or f"{idx + 1}F"),
                "sceneId": str(DEFAULT_SCENE_ID if f.get("id") == "B1" else (f.get("sceneId") or "building_floor_plan")),
                "description": str(f.get("description") or ""),
            }
        )
    floors = floors[:10]
    if not floors:
        floors.append({"id": "B1", "name": "B1", "sceneId": DEFAULT_SCENE_ID, "description": "Underground parking"})
        floors.append({"id": "1F", "name": "1F", "sceneId": "building_floor_plan", "description": ""})
    if not any(f.get("id") == "B1" for f in floors):
        floors.insert(0, {"id": "B1", "name": "B1", "sceneId": DEFAULT_SCENE_ID, "description": "Underground parking"})

    if on_log:
        on_log("正在设计核心筒基础图纸...")
    core_text = await call_llm_with_retry(
        client,
        [ChatMessage(role="user", content=CORE_ARCHITECT_PROMPT(prompt, get_active_scene("building_floor_plan")))],
        LLMConfig(apiKey=api_key, model=model, temperature=0.3, maxTokens=4096),
        on_log,
    )
    core_parsed = safe_parse_response(core_text, ParseOptions(provider=client.providerName, model=model), on_log)
    core_elements_raw = core_parsed.get("elements") if isinstance(core_parsed, dict) else None
    core_elements_raw = core_elements_raw if isinstance(core_elements_raw, list) else []
    core_layout = map_to_internal_layout({"width": 800, "height": 600, "elements": core_elements_raw})
    core_blueprint = core_layout.elements

    floor_layouts: Dict[str, ParkingLayout] = {}
    for floor in floors:
        if on_log:
            on_log(f"正在生成楼层: {floor['name']}...")
            on_log(f"楼层场景: {floor['sceneId']}")
        floor_prompt = " / ".join([x for x in [prompt, floor.get("description")] if x])
        generated = await execute_floor_generation_with_core(
            floor_prompt,
            core_blueprint,
            client,
            api_key,
            model,
            on_log,
            floor.get("sceneId"),
        )
        floor_layouts[str(floor["id"])] = generated.model_copy(update={"sceneId": str(floor["sceneId"])})

    final_data = BuildingData(blueprint=core_blueprint, floors=floor_layouts)
    _generation_cache[cache_key] = {"data": final_data.model_dump(), "timestamp": now}
    return final_data
