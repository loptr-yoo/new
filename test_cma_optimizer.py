import os

from shapely.geometry import Polygon

from backend.core.geometry.layout_optimizer import optimize_island
from backend.models import RoomAllocation
from test_power_diagram import export_power_diagram_svg


def main() -> None:
    boundary = Polygon(
        [
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, 40.0),
            (0.0, 40.0),
            (0.0, 25.0),
            (15.0, 25.0),
            (15.0, 15.0),
            (0.0, 15.0),
        ]
    )
    # test_cma_optimizer.py 修改方案

    # 1. 算出大楼的总物理面积
    total_usable_area = float(boundary.area)  # 我们之前算过大约是 1450.0

    # 2. 定义你要的比例 1:2:4:8
    ratios = [1.0, 2.0, 4.0, 8.0]
    total_ratio = sum(ratios)  # 1+2+4+8 = 15.0

    # 3. 按比例分配真实的 target_area，确保面积物理守恒！
    rooms = [
        RoomAllocation(
            room_name="R1",
            room_type="room",
            target_area=(ratios[0] / total_ratio) * total_usable_area,
            requires_window=False,
            adjacency_tags=["cluster:a"],
        ),
        RoomAllocation(
            room_name="R2",
            room_type="room",
            target_area=(ratios[1] / total_ratio) * total_usable_area,
            requires_window=False,
            adjacency_tags=["cluster:a"],
        ),
        RoomAllocation(
            room_name="R3",
            room_type="room",
            target_area=(ratios[2] / total_ratio) * total_usable_area,
            requires_window=True,
            adjacency_tags=["cluster:b"],
        ),
        RoomAllocation(
            room_name="R4",
            room_type="room",
            target_area=(ratios[3] / total_ratio) * total_usable_area,
            requires_window=True,
            adjacency_tags=["cluster:b"],
        ),
    ]



    seeds, weights, cells = optimize_island(boundary=boundary, rooms=rooms, resolution=1.0)

    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "optimized_layout.svg")
    export_power_diagram_svg(boundary=boundary, cells=cells, seeds=seeds, filepath=output_filepath)
    print(f"已生成 SVG: {os.path.abspath(output_filepath)}")
    print(f"Best weights: {weights}")


if __name__ == "__main__":
    main()
