import os
import math
from backend.models import FloorAllocation, RoomAllocation
from backend.core.geometry.topology_generator import generate_floor_skeleton
from backend.core.geometry.debug_exporter import export_skeleton_to_svg

def main():
    print("开始测试 Building 模式拓扑骨架生成...")

    # 1. 模拟第一步 LLM 输出的语义数据
    mock_floor_data = FloorAllocation(
        floor_number=1,
        floor_function_tag="office_floor",
        floor_total_area=1000.0,
        core_tube_area=100.0,
        corridor_allowance_area=150.0,
        rooms=[
            RoomAllocation(
                room_name="开放办公区", 
                room_type="office",
                target_area=450.0, 
                weight=10, 
                requires_window=True
            ),
            RoomAllocation(
                room_name="大会议室", 
                room_type="meeting_room",
                target_area=300.0, 
                weight=5, 
                requires_window=False
            )
        ]
    )

    # 2. 模拟前端传来的建筑外轮廓点阵 (标准矩形地块 40m x 25m)
    # 面积 = 40 * 25 = 1000 ㎡
    mock_outline_points = [
        (0.0, 0.0),    # 左下角
        (40.0, 0.0),   # 右下角
        (40.0, 25.0),  # 右上角
        (0.0, 25.0)    # 左上角
    ]

    try:
        # 3. 运行核心几何算法
        print(" -> 正在执行 Shapely 布尔切割与几何对齐...")
        skeleton = generate_floor_skeleton(mock_floor_data, outline_points=mock_outline_points)

        # 4. 导出为 SVG 图纸
        output_dir = "artifacts"
        os.makedirs(output_dir, exist_ok=True)
        # 改个名字，防止覆盖之前的 L 型
        output_filepath = os.path.join(output_dir, "debug_skeleton_rectangle.svg")
        
        print(" -> 正在渲染 SVG 图纸...")
        export_skeleton_to_svg(skeleton, output_filepath)
        
        print(f"\n✅ 测试成功！请用双击打开或用浏览器查看: {os.path.abspath(output_filepath)}")
        
    except Exception as e:
        print(f"\n❌ 测试失败，捕获到异常: {e}")

if __name__ == "__main__":
    main()