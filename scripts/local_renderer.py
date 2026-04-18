#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
local_renderer.py

纯本地/无头（headless）CAD 风格渲染器：
- 读取 layout.json（width/height/elements[]）
- 使用 matplotlib Agg 后端生成 .png 或 .svg

关键目标：
1) 完全绕开浏览器 SVG 的 miter/抗锯齿/alpha 混色 bug
2) 强制墙体永远不透明（alpha=1.0），并在房间之上“盖章”
3) 通过 zOrder 字段确保图层是数据固有属性（渲染器只读，不猜）

锚点（Anchor）规范：
- 对于离散型实体（door/window/家具等）采用 rect 表达时，JSON 的 x,y 必须是中心点 (cx, cy)
- Matplotlib Rectangle 需要左下角坐标，因此渲染时换算为 (cx-w/2, cy-h/2)
- rotation 围绕中心点旋转（rotate_deg_around）

兼容旧 JSON（可选）：
- 若 elem 显式声明 anchor="min"，则把 x,y 解释为左下角
- 若未声明 anchor 且缺少 zOrder（常见于旧导出），默认按左下角处理
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

try:
    import matplotlib  # type: ignore[import-not-found]
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    import matplotlib.patches as patches  # type: ignore[import-not-found]
    import matplotlib.transforms as transforms  # type: ignore[import-not-found]
except ImportError as e:
    raise SystemExit(
        "缺少依赖：matplotlib。请先执行：pip install matplotlib\n"
        f"原始错误：{type(e).__name__}: {e}"
    )

matplotlib.use("Agg")

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.core.geometry.style_constants import SEGMENTATION_COLORS  # type: ignore[import-not-found]
except Exception:
    SEGMENTATION_COLORS = {}


COLOR_MAP: Dict[str, str] = {
    "floor_slab": "#1e293b",
    "corridor": "#cbd5e1",
    "living_room": "#a78bfa",
    "bedroom": "#fda4af",
    "kitchen": "#fdba74",
    "bathroom": "#93c5fd",
    "dining_room": "#fbbf24",
    "elevator": "#06b6d4",
    "staircase": "#d2b48c",
    "door": "#fbbf24",
    "window": "#38bdf8",
    "wall": "#334155",
    "partition_wall": "#334155",
    "exterior_wall": "#334155",
}


DEFAULT_ZORDER: Dict[str, int] = {
    "floor_slab": 10,
    "corridor": 20,
    "elevator": 30,
    "elevator_hall": 30,
    "elevator_shaft": 30,
    "staircase": 30,
    "wall": 80,
    "partition_wall": 80,
    "exterior_wall": 80,
    "door": 90,
    "window": 90,
}


def _warn(msg: str) -> None:
    print(f"[local_renderer] {msg}", file=sys.stderr)


def _is_wall_type(t: str) -> bool:
    return t in {"wall", "partition_wall", "exterior_wall"} or "wall" in t


def _zorder(elem: Dict[str, Any]) -> int:
    z = elem.get("zOrder")
    if isinstance(z, (int, float)):
        return int(z)
    t = str(elem.get("type") or "")
    if _is_wall_type(t):
        return 80
    return DEFAULT_ZORDER.get(t, 20)


def _close_polygon(poly: List[List[float]]) -> List[List[float]]:
    if not poly:
        return poly
    if poly[0] != poly[-1]:
        return poly + [poly[0]]
    return poly


def _draw_polygon(ax: Any, elem: Dict[str, Any]) -> None:
    t = str(elem.get("type") or "")
    poly = elem.get("polygon") or []
    if not isinstance(poly, list) or len(poly) < 3:
        _warn(f"Skip polygon element (invalid polygon): id={elem.get('id')} type={t}")
        return
    poly = _close_polygon(poly)
    points: List[List[float]] = [[float(x), float(y)] for x, y in poly]

    face = "#334155" if _is_wall_type(t) else COLOR_MAP.get(t, "#94a3b8")
    z = _zorder(elem)

    patch = patches.Polygon(
        cast(Any, points),
        closed=True,
        facecolor=face,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        alpha=1.0,
        zorder=z,
    )
    ax.add_patch(patch)


def _seg_canonical_type(elem: Dict[str, Any]) -> str:
    t = str(elem.get("type") or "")
    t_lower = t.lower()
    if t == "furniture":
        cat = elem.get("category")
        if isinstance(cat, str) and cat:
            return cat
        return "__furniture_unknown__"
    if "wall" in t_lower:
        return "wall"
    if t in {"elevator", "elevator_shaft"}:
        return "elevator"
    if t == "staircase":
        return "staircase"
    if t == "floor_slab":
        return "floor_slab"
    if t == "door":
        return "door"
    if t == "window":
        return "window"
    if t == "column":
        return "column"
    return "room_default"


def _seg_facecolor(elem: Dict[str, Any]) -> str:
    ct = _seg_canonical_type(elem)
    if ct == "__furniture_unknown__":
        return "#FF00FF"
    if ct == "__unknown__":
        return "#FF00FF"
    if isinstance(SEGMENTATION_COLORS, dict) and ct in SEGMENTATION_COLORS:
        return str(SEGMENTATION_COLORS[ct])
    return "#FF00FF"


def _draw_polygon_seg(ax: Any, elem: Dict[str, Any]) -> None:
    poly = elem.get("polygon") or []
    if not isinstance(poly, list) or len(poly) < 3:
        return
    poly = _close_polygon(poly)
    points: List[List[float]] = [[float(x), float(y)] for x, y in poly]
    z = _zorder(elem)
    face = _seg_facecolor(elem)
    patch = patches.Polygon(
        cast(Any, points),
        closed=True,
        facecolor=face,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        alpha=1.0,
        zorder=z,
    )
    ax.add_patch(patch)


def _rect_anchor_mode(elem: Dict[str, Any]) -> str:
    """
    返回 'center' 或 'min'。
    - 新契约：默认 center
    - 兼容旧 JSON：若缺 zOrder 且未显式 anchor，则默认 min（旧导出通常是左下角坐标）
    """
    a = elem.get("anchor")
    if isinstance(a, str) and a in {"center", "min"}:
        return a
    if "zOrder" not in elem:
        return "min"
    return "center"


def _draw_rect(ax: Any, elem: Dict[str, Any], facecolor: str) -> None:
    t = str(elem.get("type") or "")
    try:
        x_raw = elem.get("x")
        y_raw = elem.get("y")
        w_raw = elem.get("width")
        h_raw = elem.get("height")
        if x_raw is None or y_raw is None or w_raw is None or h_raw is None:
            raise ValueError("missing required rect fields")
        x = float(x_raw)
        y = float(y_raw)
        w = float(w_raw)
        h = float(h_raw)
    except Exception:
        _warn(f"Skip rect element (missing x/y/width/height): id={elem.get('id')} type={t}")
        return

    rotation = float(elem.get("rotation") or 0.0)
    z = _zorder(elem)

    anchor = _rect_anchor_mode(elem)
    if anchor == "center":
        cx, cy = x, y
        blx = cx - w / 2
        bly = cy - h / 2
    else:
        blx, bly = x, y
        cx, cy = blx + w / 2, bly + h / 2

    rect = patches.Rectangle(
        (blx, bly),
        w,
        h,
        facecolor=facecolor,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        alpha=1.0,
        zorder=z,
    )
    if abs(rotation) > 1e-6:
        rect.set_transform(transforms.Affine2D().rotate_deg_around(cx, cy, rotation) + ax.transData)
    ax.add_patch(rect)


def _draw_rect_seg(ax: Any, elem: Dict[str, Any]) -> None:
    face = _seg_facecolor(elem)
    _draw_rect(ax, elem, facecolor=face)


def _draw_door(ax: Any, elem: Dict[str, Any]) -> None:
    try:
        x_raw = elem.get("x")
        y_raw = elem.get("y")
        w_raw = elem.get("width")
        h_raw = elem.get("height")
        if x_raw is None or y_raw is None or w_raw is None or h_raw is None:
            raise ValueError("missing required rect fields")
        x = float(x_raw)
        y = float(y_raw)
        w = float(w_raw)
        h = float(h_raw)
    except Exception:
        _warn(f"Skip door element (missing x/y/width/height): id={elem.get('id')}")
        return

    rotation = float(elem.get("rotation") or 0.0)
    z = _zorder(elem)

    anchor = _rect_anchor_mode(elem)
    if anchor == "center":
        cx, cy = x, y
        blx = cx - w / 2
        bly = cy - h / 2
    else:
        blx, bly = x, y
        cx, cy = blx + w / 2, bly + h / 2

    t = transforms.Affine2D().rotate_deg_around(cx, cy, rotation) + ax.transData

    eraser = patches.Rectangle(
        (blx, bly),
        w,
        h,
        facecolor="#ffffff",
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        alpha=1.0,
        zorder=z,
    )
    eraser.set_transform(t)
    ax.add_patch(eraser)

    outline = patches.Rectangle(
        (blx, bly),
        w,
        h,
        facecolor="none",
        edgecolor="#334155",
        linewidth=1.2,
        antialiased=False,
        alpha=1.0,
        zorder=z + 1,
    )
    outline.set_transform(t)
    ax.add_patch(outline)


def _draw_window(ax: Any, elem: Dict[str, Any]) -> None:
    try:
        x_raw = elem.get("x")
        y_raw = elem.get("y")
        w_raw = elem.get("width")
        h_raw = elem.get("height")
        if x_raw is None or y_raw is None or w_raw is None or h_raw is None:
            raise ValueError("missing required rect fields")
        x = float(x_raw)
        y = float(y_raw)
        w = float(w_raw)
        h = float(h_raw)
    except Exception:
        _warn(f"Skip window element (missing x/y/width/height): id={elem.get('id')}")
        return

    rotation = float(elem.get("rotation") or 0.0)
    z = _zorder(elem)

    anchor = _rect_anchor_mode(elem)
    if anchor == "center":
        cx, cy = x, y
        blx = cx - w / 2
        bly = cy - h / 2
    else:
        blx, bly = x, y
        cx, cy = blx + w / 2, bly + h / 2

    t = transforms.Affine2D().rotate_deg_around(cx, cy, rotation) + ax.transData

    eraser = patches.Rectangle(
        (blx, bly),
        w,
        h,
        facecolor="#ffffff",
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        alpha=1.0,
        zorder=z,
    )
    eraser.set_transform(t)
    ax.add_patch(eraser)

    if w >= h:
        x0, y0 = blx, cy
        x1, y1 = blx + w, cy
    else:
        x0, y0 = cx, bly
        x1, y1 = cx, bly + h
    ax.plot([x0, x1], [y0, y1], color="#334155", linewidth=1.6, zorder=z + 1, transform=t, antialiased=False)


CUSTOM_DRAW_HOOKS: Dict[str, Callable[[Any, Dict[str, Any]], None]] = {
    "door": _draw_door,
    "window": _draw_window,
}


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _validate_segmentation_palette(colors: Dict[str, str]) -> None:
    keys = [k for k in colors.keys() if isinstance(colors.get(k), str)]
    if len(keys) != len(set(keys)):
        raise ValueError("SEGMENTATION_COLORS key 重复")

    rgb_map: Dict[str, Tuple[int, int, int]] = {k: _hex_to_rgb(colors[k]) for k in keys}
    min_rgb_dist = 60.0
    min_hue_gap = 0.08
    allowed_same = {frozenset({"floor_slab", "room_default"})}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            if frozenset({a, b}) in allowed_same:
                continue
            ra, ga, ba = rgb_map[a]
            rb, gb, bb = rgb_map[b]
            dist = math.sqrt((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2)
            ha = colorsys.rgb_to_hsv(ra / 255.0, ga / 255.0, ba / 255.0)[0]
            hb = colorsys.rgb_to_hsv(rb / 255.0, gb / 255.0, bb / 255.0)[0]
            hue_gap = min(abs(ha - hb), 1.0 - abs(ha - hb))
            if dist < min_rgb_dist and hue_gap < min_hue_gap:
                raise ValueError(f"SEGMENTATION_COLORS 色差不足: {a} vs {b}")


def _load_layout(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "elements" in data:
        return data
    raise ValueError("输入 JSON 不包含 elements[]，请使用 cli_runner.py 生成的 layout.json 或导出前端 layout.json。")


def _render(layout: Dict[str, Any], out_path: Path, mode: str) -> None:
    width = float(layout.get("width") or 0.0)
    height = float(layout.get("height") or 0.0)
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height 非法：width={width} height={height}")

    elements = layout.get("elements") or []
    if not isinstance(elements, list):
        raise ValueError("elements 必须是数组")

    fig_w = max(6.0, width / 2.0)
    fig_h = max(4.0, height / 2.0)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if mode == "seg":
        if out_path.suffix.lower() != ".png":
            raise ValueError("seg 模式仅允许输出 PNG（无损）")
        _validate_segmentation_palette(SEGMENTATION_COLORS if isinstance(SEGMENTATION_COLORS, dict) else {})
        matplotlib.rcParams["lines.antialiased"] = False
        matplotlib.rcParams["patch.antialiased"] = False
        ax.set_xlim(0.0, width)
        ax.set_ylim(0.0, height)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        ax.set_position([0.0, 0.0, 1.0, 1.0])
    else:
        padding = 0.5
        ax.set_xlim(0.0 - padding, width + padding)
        ax.set_ylim(0.0 - padding, height + padding)
    ax.set_aspect("equal")
    ax.axis("off")

    def sort_key(e: Dict[str, Any]) -> Tuple[int, str]:
        return (_zorder(e), str(e.get("id") or ""))

    if mode == "seg":
        matplotlib.rcParams["path.simplify"] = False

    for elem in sorted((e for e in elements if isinstance(e, dict)), key=sort_key):
        t = str(elem.get("type") or "")

        if mode != "seg" and t in CUSTOM_DRAW_HOOKS:
            CUSTOM_DRAW_HOOKS[t](ax, elem)
            continue

        poly = elem.get("polygon")
        if isinstance(poly, list) and len(poly) >= 3:
            if mode == "seg":
                _draw_polygon_seg(ax, elem)
            else:
                _draw_polygon(ax, elem)
            continue

        if all(k in elem for k in ("x", "y", "width", "height")):
            if mode == "seg":
                _draw_rect_seg(ax, elem)
            else:
                if _is_wall_type(t):
                    _draw_rect(ax, elem, facecolor="#334155")
                else:
                    _draw_rect(ax, elem, facecolor=COLOR_MAP.get(t, "#94a3b8"))
            continue

        _warn(f"Skip element (unknown schema): id={elem.get('id')} type={t}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".png":
        if mode == "seg":
            fig.savefig(str(out_path), dpi=300, bbox_inches=None, pad_inches=0)
        else:
            fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless local renderer (Matplotlib CAD style)")
    p.add_argument("-i", "--input", required=True, help="输入 layout.json（width/height/elements[]）")
    p.add_argument("-o", "--output", required=True, help="输出文件路径（.png 或 .svg）")
    p.add_argument("--mode", choices=["seg", "cad"], default="seg", help="渲染模式：seg=语义分割（离散色表，无AA），cad=CAD 风格")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    layout = _load_layout(Path(args.input))
    _render(layout, Path(args.output), str(args.mode))
    print(f"[local_renderer] Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
