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
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    points = [(float(x), float(y)) for x, y in poly]

    face = "#334155" if _is_wall_type(t) else COLOR_MAP.get(t, "#94a3b8")
    z = _zorder(elem)

    patch = patches.Polygon(points, closed=True, facecolor=face, edgecolor="none", alpha=1.0, zorder=z)  # type: ignore[arg-type]
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

    rect = patches.Rectangle((blx, bly), w, h, facecolor=facecolor, edgecolor="none", alpha=1.0, zorder=z)
    if abs(rotation) > 1e-6:
        rect.set_transform(transforms.Affine2D().rotate_deg_around(cx, cy, rotation) + ax.transData)
    ax.add_patch(rect)


def _draw_door(ax: Any, elem: Dict[str, Any]) -> None:
    """
    门洞绘制 Hook：
    - 当前只画 Bounding Box（黄色块）
    - 预留 swing_angle / swing_dir 字段，以便未来画扇形轨迹（CAD 风格）
    """
    _draw_rect(ax, elem, facecolor=COLOR_MAP.get("door", "#fbbf24"))


def _draw_window(ax: Any, elem: Dict[str, Any]) -> None:
    _draw_rect(ax, elem, facecolor=COLOR_MAP.get("window", "#38bdf8"))


CUSTOM_DRAW_HOOKS: Dict[str, Callable[[Any, Dict[str, Any]], None]] = {
    "door": _draw_door,
    "window": _draw_window,
}


def _load_layout(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "elements" in data:
        return data
    raise ValueError("输入 JSON 不包含 elements[]，请使用 cli_runner.py 生成的 layout.json 或导出前端 layout.json。")


def _render(layout: Dict[str, Any], out_path: Path) -> None:
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

    # 给四周留出 0.5 米的物理边距，防止外墙及可能悬挑的元素被切除
    padding = 0.5
    ax.set_xlim(0.0 - padding, width + padding)
    ax.set_ylim(0.0 - padding, height + padding)
    ax.set_aspect("equal")
    ax.axis("off")

    def sort_key(e: Dict[str, Any]) -> Tuple[int, str]:
        return (_zorder(e), str(e.get("id") or ""))

    for elem in sorted((e for e in elements if isinstance(e, dict)), key=sort_key):
        t = str(elem.get("type") or "")

        if t in CUSTOM_DRAW_HOOKS:
            CUSTOM_DRAW_HOOKS[t](ax, elem)
            continue

        poly = elem.get("polygon")
        if isinstance(poly, list) and len(poly) >= 3:
            _draw_polygon(ax, elem)
            continue

        if all(k in elem for k in ("x", "y", "width", "height")):
            if _is_wall_type(t):
                _draw_rect(ax, elem, facecolor="#334155")
            else:
                _draw_rect(ax, elem, facecolor=COLOR_MAP.get(t, "#94a3b8"))
            continue

        _warn(f"Skip element (unknown schema): id={elem.get('id')} type={t}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".png":
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless local renderer (Matplotlib CAD style)")
    p.add_argument("-i", "--input", required=True, help="输入 layout.json（width/height/elements[]）")
    p.add_argument("-o", "--output", required=True, help="输出文件路径（.png 或 .svg）")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()
    layout = _load_layout(Path(args.input))
    _render(layout, Path(args.output))
    print(f"[local_renderer] Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
