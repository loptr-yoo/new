from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LayoutElement(BaseModel):
    id: str
    type: str
    x: float = Field(description="屏幕坐标系：原点左上，x 向右为正。")
    y: float = Field(description="屏幕坐标系：原点左上，y 向下为正。")
    width: float
    height: float
    rotation: Optional[float] = Field(default=None, description="旋转角度（度），顺时针为正，绕元素中心。")
    label: Optional[str] = None
    subType: Optional[str] = None
    forward: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="朝向单位向量；坐标系：X-Y 为平面，Z 为竖直；水平朝向通常 z=0。",
    )

    @model_validator(mode="after")
    def _validate_forward(self) -> "LayoutElement":
        if self.forward is None:
            return self
        if len(self.forward) != 3:
            raise ValueError("forward 必须是长度为 3 的向量")
        if abs(float(self.forward[2])) > 1e-6:
            raise ValueError("在当前契约下，水平朝向 forward 的 Z 分量必须为 0")
        return self


class ParkingLayout(BaseModel):
    width: float
    height: float
    elements: List[LayoutElement]
    sceneId: Optional[str] = None


class ConstraintViolation(BaseModel):
    elementId: str
    targetId: Optional[str] = None
    type: Literal[
        "overlap",
        "out_of_bounds",
        "invalid_dimension",
        "placement_error",
        "connectivity_error",
        "width_mismatch",
    ]
    message: str


class BuildingData(BaseModel):
    blueprint: List[LayoutElement]
    floors: Dict[str, ParkingLayout]


class GenerateRequest(BaseModel):
    prompt: str
    provider: Literal["gemini", "deepseek", "openai"]
    model: str
    sceneId: Optional[str] = None
    floorCount: Optional[int] = Field(default=None)


class AugmentRequest(BaseModel):
    layout: ParkingLayout
    provider: Literal["gemini", "deepseek", "openai"]
    model: str
    sceneId: Optional[str] = None


class SceneType(str, Enum):
    """
    场景生成类型（语义规划层入口分流枚举）。

    重要约定：
    - floor：单层室内布局模式
    - parking：单层停车场模式
    - building：整栋多层建筑模式
    """

    FLOOR = "floor"
    PARKING = "parking"
    BUILDING = "building"


class GenerateSemanticsRequest(BaseModel):
    """
    语义规划层统一入口请求体。

    该请求体用于新接口 `/api/v1/generate/semantics`：
    - building：返回纯语义 `BuildingAllocation`
    - floor/parking：黑盒透传旧管线，返回旧结构 `BuildingData`
    """

    model_config = ConfigDict(extra="forbid")

    scene_type: SceneType = Field(..., description="场景类型：floor/parking/building")
    user_prompt: str = Field(..., min_length=1, description="用户自然语言需求")

    total_area: Optional[float] = Field(default=None, gt=0, description="总建筑面积（平方米）")
    total_floors: Optional[int] = Field(default=None, ge=1, description="总楼层数（building 模式常用）")

    provider: Optional[Literal["gemini", "deepseek", "openai"]] = Field(
        default=None, description="LLM 提供商；不传则由后端自动选择"
    )
    model: Optional[str] = Field(default=None, min_length=1, description="LLM 模型 ID；不传则由后端选择默认值")


class RoomAllocation(BaseModel):
    """
    Building 模式：单个房间/功能空间的目标配额。

    新增语义字段（zone, needs_window, adjacency_required 等）以适配
    SemanticRoomSpec 和 Treemap+MIQP 求解器。旧字段 requires_window /
    adjacency_tags 保留做兼容。
    """

    model_config = ConfigDict(extra="forbid")

    # 基础属性
    room_id: str = Field(default="", description="唯一标识（LLM 生成，如 room_001）")
    room_name: str = Field(..., min_length=1, description="房间名称（人类可读）")
    room_type: str = Field(..., min_length=1, description="房间类型（用于后续管线识别/映射）")
    target_area: float = Field(..., gt=0, description="目标面积（平方米）")

    # 语义属性（新增）
    zone: str = Field(default="public", description="功能分区: public|private|service|circulation")
    needs_window: bool = Field(default=False, description="是否需要采光（外墙）")
    min_width: float = Field(default=2.5, ge=0, description="最小开间（米）")
    aspect_ratio_range: List[float] = Field(default=[0.5, 2.0], description="宽高比范围 [min, max]")

    # 邻接约束（新增）
    adjacency_required: List[str] = Field(default_factory=list, description="必须相邻的房间ID列表")
    adjacency_preferred: List[str] = Field(default_factory=list, description="偏好相邻的房间ID列表")
    adjacency_forbidden: List[str] = Field(default_factory=list, description="禁止相邻的房间ID列表")

    # size_hint（两步 LLM 模式：Step 2 输出，后端转为 target_area）
    size_hint: Optional[str] = Field(default=None, description="面积提示: large/medium/small（有值时覆盖 target_area）")

    # 优先级
    weight: int = Field(default=5, ge=1, le=10, description="优先级/权重（1-10）")

    # 旧字段（兼容，deprecated）
    requires_window: Optional[bool] = Field(default=None, description="[deprecated] 使用 needs_window")
    adjacency_tags: List[str] = Field(default_factory=list, description="[deprecated] 使用 adjacency_required/preferred/forbidden")


class FloorAllocation(BaseModel):
    """
    Building 模式：单层的功能划分与面积配额。

    面积守恒要求（用于容错修复）：
    单层总面积 = 核心筒面积 + 走廊预留面积 + 独立房间面积之和
    """

    model_config = ConfigDict(extra="forbid")

    floor_number: int = Field(..., ge=1, description="楼层编号（从 1 开始）")
    floor_function_tag: str = Field(..., min_length=1, description="楼层功能标签（如 lobby/office/hotel 等）")
    floor_total_area: float = Field(..., gt=0, description="该层总面积（平方米）")
    core_tube_area: float = Field(..., ge=0, description="核心筒面积（平方米）")
    corridor_allowance_area: float = Field(..., ge=0, description="走廊预留面积（平方米）")
    rooms: List[RoomAllocation] = Field(default_factory=list, description="房间列表（目标面积配额）")

    @model_validator(mode="after")
    def _normalize_room_areas_if_needed(self) -> "FloorAllocation":
        # 容错逻辑：当 LLM 的面积分配与 floor_total_area 偏差超过 5% 时，
        # 按比例缩放 rooms[*].target_area，以保证单层面积严格守恒。
        #
        # 约束：
        # - core_tube_area / corridor_allowance_area 视为“固定预留”，不参与缩放
        # - 仅对 rooms 的 target_area 做比例调整
        room_area_sum = sum(r.target_area for r in self.rooms)
        current_total = room_area_sum + self.core_tube_area + self.corridor_allowance_area

        if self.floor_total_area <= 0:
            return self

        diff_ratio = abs(current_total - self.floor_total_area) / self.floor_total_area
        if diff_ratio <= 0.05:
            return self

        target_room_sum = self.floor_total_area - self.core_tube_area - self.corridor_allowance_area
        if target_room_sum <= 0:
            raise ValueError(
                "楼层面积配额不合法：floor_total_area 小于核心筒面积+走廊预留面积，无法分配房间面积。"
            )

        if room_area_sum <= 0:
            raise ValueError("楼层面积配额不合法：房间 target_area 之和为 0，无法按比例缩放。")

        scale = target_room_sum / room_area_sum

        if not self.rooms:
            raise ValueError("楼层面积配额不合法：rooms 为空，无法进行面积守恒修复。")

        new_rooms: List[RoomAllocation] = []
        allocated = 0.0
        for idx, room in enumerate(self.rooms):
            if idx < len(self.rooms) - 1:
                new_area = room.target_area * scale
                allocated += new_area
            else:
                # 最后一个房间用补差方式兜底，确保浮点累加后严格守恒：
                # sum(new_rooms.target_area) == target_room_sum
                new_area = target_room_sum - allocated
            if new_area <= 0:
                raise ValueError("面积守恒修复失败：缩放后出现非正房间面积，请检查输入面积与预留比例。")
            new_rooms.append(room.model_copy(update={"target_area": float(new_area)}))

        self.rooms = new_rooms
        return self


class BuildingAllocation(BaseModel):
    """
    Building 模式：整栋建筑的语义规划产物（纯语义，不含几何坐标）。
    """

    model_config = ConfigDict(extra="forbid")

    building_name: str = Field(..., min_length=1, description="建筑名称")
    total_floors: int = Field(..., ge=1, description="总楼层数")
    overall_total_area: float = Field(..., gt=0, description="总建筑面积（平方米）")
    floors: List[FloorAllocation] = Field(default_factory=list, description="逐层的面积与功能配额")
