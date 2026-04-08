from .island_partition_solver import (
    IslandPartitionSolver,
    RoomResult,
    RoomSpec,
    SemanticIslandPartitionSolver,
    partition_island,
    partition_island_semantic,
)
from .constraint_validator import (
    ConstraintValidator,
    SemanticConstraintValidator,
    SemanticValidationReport,
    ValidationReport,
    validate_layout,
)
from .layout_generator import (
    GeneratorConfig,
    LayoutGenerator,
    LayoutResult,
    LayoutResultV2,
    LayoutGenerationError,
    TopologyError,
    PartitionError,
    generate_layout,
    generate_layout_semantic,
    generate_layout_simple,
    generate_layout_v2,
)
from .layout_generator import RoomSpec as LayoutRoomSpec
from .room_spec import (
    IslandContext,
    SolverConfig,
    WarmStartRect,
    ZoneType,
)
from .room_spec import RoomSpec as SemanticRoomSpec

# 拓扑生成
from .topology_generator import (
    CoreTube,
    Corridor,
    Island,
    RectangularTopologyGenerator,
    generate_rectangular_topology,
)

# 房间-岛屿分配
from .island_room_assigner import (
    AssignmentResult,
    AssignerConfig,
    AssignmentError,
    IslandRoomAssigner,
    assign_rooms_to_islands,
)

# 多层编排器
from .building_orchestrator import (
    BuildingOrchestrator,
    BuildingResult,
)

# 工具
from .axis_align import snap_to_grid


__all__ = [
    # Old solver
    "IslandPartitionSolver",
    "RoomResult",
    "RoomSpec",
    "partition_island",
    # Semantic solver
    "SemanticIslandPartitionSolver",
    "partition_island_semantic",
    "SemanticRoomSpec",
    "ZoneType",
    "IslandContext",
    "SolverConfig",
    "WarmStartRect",
    # Validators
    "ConstraintValidator",
    "ValidationReport",
    "SemanticConstraintValidator",
    "SemanticValidationReport",
    "validate_layout",
    # Generator (v1)
    "GeneratorConfig",
    "LayoutGenerator",
    "LayoutResult",
    "LayoutRoomSpec",
    "generate_layout",
    "generate_layout_simple",
    "generate_layout_semantic",
    # Generator (v2 multi-island)
    "LayoutResultV2",
    "LayoutGenerationError",
    "TopologyError",
    "PartitionError",
    "generate_layout_v2",
    # Topology
    "CoreTube",
    "Corridor",
    "Island",
    "RectangularTopologyGenerator",
    "generate_rectangular_topology",
    # Assignment
    "AssignmentResult",
    "AssignerConfig",
    "AssignmentError",
    "IslandRoomAssigner",
    "assign_rooms_to_islands",
    # Building orchestrator
    "BuildingOrchestrator",
    "BuildingResult",
    # Tools
    "snap_to_grid",
]
