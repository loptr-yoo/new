from building.app.geometry.grid_growth_planner import plan_grid_growth_topology


def test_grid_growth_planner_imports_from_building_package() -> None:
    assert callable(plan_grid_growth_topology)
