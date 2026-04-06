# 紧急修复指令：修复掩膜回退与正交化拓扑灾难

当前系统出现了严重的图形崩溃。请严格执行以下两项修复，不要修改其他无关代码：

### 任务 1：修复 `power_diagram.py` 的掩膜逻辑回退 (Regression)
你之前在修改 `power_diagram.py` 加入 `buffer(1e-4)` 时，丢失了极其重要的 `mask` 维度修复代码！请立刻将 `_mask_points_in_boundary` 函数恢复为以下安全版本：
```python
def _mask_points_in_boundary(boundary, x, y):
    orig_shape = x.shape
    flat_x = x.ravel()
    flat_y = y.ravel()
    try:
        from shapely import covers_xy
        mask_flat = covers_xy(boundary, flat_x, flat_y)
    except Exception:
        from shapely.prepared import prep
        from shapely.geometry import Point
        prepared_b = prep(boundary)
        mask_flat = np.array([prepared_b.covers(Point(px, py)) for px, py in zip(flat_x, flat_y)])
    return mask_flat.reshape(orig_shape)
```

### 任务 2：修复 orthogonalization.py 的线段交叉与无主碎片丢弃问题
在正交化模块中，曼哈顿阶梯化（L型折线）导致了线段交叉，进而在 polygonize 时产生了大量细小碎片。请对 orthogonalize_layout 进行以下强健性修复：

安全降噪：在提取 internal_lines 后，简化力度稍微加大一点，过滤掉微小线段，防止 L 型折线互相干涉。internal_lines.simplify(1.0, preserve_topology=True)

安全的正交化逻辑 (只拉直，不折叠)：
不要使用“曼哈顿 L 型折线”了！它会导致线段自交。请将 _force_orthogonal_segments 的逻辑改为：
```
遍历所有线段 (P1, P2)。

如果 dx > dy（趋于水平）：直接把 P1.y 和 P2.y 强制设为 (P1.y + P2.y) / 2。

如果 dy >= dx（趋于垂直）：直接把 P1.x 和 P2.x 强制设为 (P1.x + P2.x) / 2。

不要新增顶点，只移动端点！ 这样可以最大程度避免线段在重构时互相交叉打结。
```
彻底的间隙填充 (Gap Filling)：
在最后 seeds 认领完多边形后（此时 final_cells 只有几个带红点的小多边形），必须把剩下大楼面积全部分配掉！

```Python
claimed_union = unary_union([c for c in final_cells if not c.is_empty])
leftover_area = boundary.difference(claimed_union)

# 把 leftover_area (可能是一个 MultiPolygon) 切碎，分给离它最近的 cell
if not leftover_area.is_empty:
    leftover_polys = leftover_area.geoms if hasattr(leftover_area, 'geoms') else [leftover_area]
    for piece in leftover_polys:
        if piece.area < 0.1: continue
        # 找到离这个碎片最近的种子/房间
        best_idx = 0
        min_dist = float('inf')
        for i, (sx, sy) in enumerate(seeds):
            dist = piece.distance(Point(sx, sy))
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        # 合并给最近的房间
        final_cells[best_idx] = final_cells[best_idx].union(piece)
```
请按照以上明确的逻辑，修复 power_diagram.py 并重构 orthogonalization.py 中的正交化与填充代码。