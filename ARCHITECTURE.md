# 项目架构与开发指南
- 生成 layout（每层一个）： python scripts/cli_runner.py -p "...prompt..." -m ... -c north -o out/layout.json
- 输出分割 mask： python scripts/local_renderer.py -i out/layout_F1.json -o out/mask_F1.png --mode seg
## 1. 核心架构

本项目采用 **多模型适配 (Multi-Model Adapter)** + **双阶段生成 (Two-Stage Generation)** 的架构设计。

*   **服务层 (Services)**:
    *   `parkingFlow.ts`: 业务编排核心。负责协调生成、细化、修复全流程，统一处理重试机制和错误兜底。
    *   `llmProvider.ts`: 模型抽象层。屏蔽 Gemini, DeepSeek, OpenAI 的接口差异。
    *   `responseParser.ts`: 响应解析器。处理 AI 返回的非标 JSON、Markdown 包裹等格式问题。

*   **场景注册 (Scene Registry)**:
    *   `sceneRegistry.ts`: 场景配置中心。通过配置对象定义不同场景的 Prompt 规则、渲染样式和后处理算法。

*   **几何算法 (Geometry)**:
    *   `aiCommonUtils.ts`: 通用几何工具（碰撞检测、自动补丁合并）。
    *   `floorGeometryUtils.ts`: 楼层专用几何算法（正交化、吸附）。

## 2. 核心方法

*   `executeGeneration` (parkingFlow.ts): 生成入口。执行 "生成 -> 校验 -> 自动修复" 循环。
*   `executeRefinement` (parkingFlow.ts): 细化入口。基于已有布局进行增量生成（如添加家具或设施）。
*   `runIterativeFix` (parkingFlow.ts): 自动修复循环。基于 `validateLayout` 返回的违规项，调用 AI 进行针对性微调。
*   `validateLayout` (geometry.ts): 约束校验器。输出重叠、越界、连通性等错误列表。

## 3. 注册新场景

在 `utils/sceneRegistry.ts` 中添加新的 `SceneDefinition` 对象：

```typescript
export const MyNewScene: SceneDefinition = {
  id: 'my_new_scene',
  promptConfig: {
    roleDefinition: '场景角色定义',
    geometricRules: '核心几何约束规则',
    requiredElements: ['必须包含的元素类型'],
    exampleJSON: 'Few-Shot 示例'
  },
  styles: { ... }, // 定义元素颜色与透明度
  postProcessAlgorithms: [ ... ] // 挂载专用后处理算法
};
```

最后将其加入 `SCENE_REGISTRY` 字典即可。

## 4. 楼层 vs 停车场：生成策略差异

系统根据场景类型（通过 `requiredElements` 自动判定）采用截然不同的生成策略：

| 维度 | 停车场场景 (Parking Scene) | 楼层平面图 (Floor Plan) |
| :--- | :--- | :--- |
| **生成目标** | **Coarse-Grained (骨架优先)** | **Complete Structural (结构完整)** |
| **第一阶段** | 仅生成道路、外墙、地块。**禁止**生成车位与细节。 | 生成所有外墙、内墙、房间分区、门窗。**禁止**生成家具。 |
| **第二阶段** | **算法主导**。调用 `fillParkingAutomatically` 算法自动计算车位，AI 仅辅助生成柱子/标线。 | **语义主导**。AI 根据房间功能填充家具（床、沙发等）。 |
| **后处理** | 强几何清洗。自动移除路口重叠、吸附地块缝隙。 | 强拓扑修正。`enforceOrthogonalWalls` (正交化), `snapDoorsToWalls` (门吸附)。 |
| **核心难点** | 交通流线连通性、车位排布效率。 | 房间拓扑关系、墙体闭合性。 |
