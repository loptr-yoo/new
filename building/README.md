# Backend-Only Multi-Floor Building Generator

Active project boundary: `building/`.

This project is backend-only, multi-floor-only, and backend-rendered. It does not require the frontend, Node, or a browser for active generation or rendering.

## Active API

Start the ASGI app:

```bash
python -m uvicorn building.app.main:app --reload
```

Active endpoints:

```text
POST /api/building/generate
POST /api/building/generate/stream
POST /api/v1/generate/semantics
```

Deprecated endpoints return `410 Gone`:

```text
POST /api/generate
POST /api/generate/stream
```

## Active CLI

```bash
python -m building.cli.full_pipeline -p "生成一个二层住宅区 一楼包含四个卧室、厨房、客厅、两个卫生间 二楼包含四个卧室和一个卫生间" -m "gemini-3.1-pro-preview" -c east --corridor-mode organic --seed 123 --out-dir building/out/test_gemini_east --render-mode seg --seg-target refined --provider gemini --log-llm --topology-mode grid_growth
```

No-LLM geometry smoke:

```bash
python -m building.cli.geometry_smoke_fixed_allocation --mode both --out building/out/test_gemini_east/geometry_smoke_fixed_allocation_result.json
```

## Product Boundary

Supported:

- backend-only multi-floor building generation
- per-floor internal geometry as part of multi-floor generation
- backend local rendering
- no-LLM fixed-allocation smoke diagnostics

Rejected:

- single-floor product mode
- parking product mode
- `scene_type=floor`
- `scene_type=parking`
- `sceneId=parking_underground`
- `total_floors=1`

## Output And Log Hygiene

When `--out-dir` is omitted, `full_pipeline` writes each run under `building/out/<timestamp>_full`. Legacy relative paths such as `out/test_gemini_south` and `building/outputs/test_gemini_east` are rebased under `building/out/`.

Stable backend render artifacts use names such as `layout_F1.json`, `layout_F2.json`, `render_F1_seg.png`, and `render_F2_seg.png`. Existing debug artifacts such as `refined_layout_F1.json` may still be emitted by the full pipeline.

`--log-llm` and `--log-llm-max-chars` remain supported. The LLM transcript path defaults to the current run directory as `llm_log.txt`, unless `LLM_LOG_PATH` is explicitly set.

Generated outputs, logs, and artifacts are local verification products and should not be committed.
