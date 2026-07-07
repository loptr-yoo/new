# Policy Inventory

This inventory tracks rules that currently live in prompts, geometry helpers,
post-processing, rendering, or schemas and whether they should move into the
Stage 1 policy layer.

| rule_name | current_location | current_owner | target_policy_file | stage_owner | migration_status | risk |
| --- | --- | --- | --- | --- | --- | --- |
| room min/target/max area | `building/app/geometry/room_spec.py`, `room_defaults.py`, prompts | semantics/geometry | `policies/common/room_rules.yaml` | Stage 1 | partial | Hidden fallbacks can make CLI/API programs diverge |
| room needs_window | `room_spec.py`, `room_defaults.py`, postprocessor | semantics/geometry/postprocess | `policies/common/room_rules.yaml`, `window_rules.yaml` | Stage 1/3 | partial | Stage 2 may over-assign facade demand |
| forbidden adjacency | `room_defaults.py`, prompts | semantics | `policies/common/adjacency_rules.yaml` | Stage 1 | partial | LLM may emit invalid same-floor references |
| corridor layout default | API schema, CLI, service, topology snapshot | API/CLI/geometry | `pipeline_defaults.py`, `corridor_rules.yaml` | Stage 1/2 | partial | `door_side` vs `organic` mismatch changes budgets |
| corridor reserve ratio | topology budget, prompts | semantics/geometry | `policies/common/corridor_rules.yaml` | Stage 1 | partial | Program area can pass upstream but fail geometry capacity |
| wall reserve ratio | implicit geometry/postprocess | geometry/postprocess | `policies/common/corridor_rules.yaml` | Stage 1/3 | new | Stage 1 may overestimate usable area |
| core default location | API schema, CLI, service, topology generator | API/CLI/geometry | `pipeline_defaults.py`, `vertical_core_rules.yaml` | Stage 1/2 | partial | Different stages can validate different cores |
| core area ratio | service/topology budget/fixed smoke | service/geometry | `policies/common/vertical_core_rules.yaml` | Stage 1 | partial | Core overlap and capacity failures surface too late |
| door default width | `SolverConfig`, postprocessor | geometry/postprocess | `policies/common/door_rules.yaml` | Stage 3 | pending | Door reachability failures appear after layout |
| window required room types | `room_spec.py`, postprocessor | geometry/postprocess | `policies/common/window_rules.yaml` | Stage 1/3 | partial | Window shortage can be hidden until rendering |
| renderer style constants | `style_constants.py`, renderer | rendering | future style policy | Stage 3 | pending | Visual output can drift from semantic room type |

