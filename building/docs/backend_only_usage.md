# Backend-Only Usage

Use `building.app.main:app` as the ASGI entrypoint and `building/out/` for generated artifacts.

No frontend, Node, browser renderer, or parking product route is part of the active path.

Run tests:

```bash
python -m pytest building/tests -q
```

Run smoke:

```bash
python -m building.cli.geometry_smoke_fixed_allocation --mode both --out building/out/geometry_smoke_fixed_allocation_result.json
```

## Outputs

Default CLI runs write to `building/out/<timestamp>_full`. Legacy relative `--out-dir` values under `out/` or `building/outputs/` are rebased under `building/out/`. Generated `layout_F{n}.json`, `render_F{n}_seg.png`, logs, smoke JSON, and other artifacts are ignored by git except for `building/out/.gitkeep`.
