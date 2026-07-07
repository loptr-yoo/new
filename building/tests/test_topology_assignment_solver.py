from building.app.geometry.topology_assignment_solver import TopologyAssignmentSolver


def test_topology_assignment_solver_imports_from_building_package() -> None:
    assert TopologyAssignmentSolver is not None
