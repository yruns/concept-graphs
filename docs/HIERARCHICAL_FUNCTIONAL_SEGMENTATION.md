# 层次化功能区域划分 (Hierarchical Functional Segmentation)

## 概述

本算法是对原有 **Step 4.5 (区域感知场景划分)** 的升级版本，核心改进是引入了 **基于功能可用性的三层层次化场景图结构**，并通过 **LLM迭代推理** 实现对场景功能区域的智能划分。

### 核心创新

| 特性 | 原方法 (Step 4.5) | 新方法 (Step 4.5b) |
|------|-------------------|---------------------|
| **划分依据** | 视觉相似度 + 变化信号 | 物体功能 + 视觉上下文 + 轨迹行为 |
| **输出结构** | 扁平化区域列表 | **三层层次结构** (空间单元→功能区域→物体群组) |
| **物体分配** | 基于可见性统计 | **LLM推理** + 功能匹配 |
| **可解释性** | 基于信号的数值分析 | 自然语言推理证据 |
| **下游支持** | 纯视觉划分 | **面向任务的接口** (导航、搜索、规划) |

### 设计目标

1. **功能可解释性**: 每个区域有明确的功能定义和支持的活动类型
2. **层次化表示**: 支持从粗到细的空间理解
3. **任务对齐**: 输出直接服务于机器人导航、物体搜索等下游任务
4. **多模态融合**: 综合利用3D物体、视频帧、相机轨迹

---

## Pipeline 位置

```
Step 1:   2D分割 (SAM + CLIP)
              ↓
Step 2:   3D物体地图
              ↓
Step 4:   物体描述 (Vision LLM)
              ↓
Step 4.5: 区域感知场景划分 (原方法，基于视觉信号)
              ↓
Step 4.5b: ★ 层次化功能区域划分 ← 本算法
              ↓
Step 5:   细化描述
              ↓
Step 6:   场景图
```

**注意**: Step 4.5 和 Step 4.5b 可以独立运行，建议使用 4.5b 获得更好的功能划分结果。

---

## 算法架构

### 7步流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: 加载场景数据                                                │
│  ────────────────────                                                │
│  输入: traj.txt, pcd_saves/*.pkl.gz                                  │
│  输出: 相机位姿 (N×7), 3D物体列表 (M个)                              │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: 提取物体Affordance                                          │
│  ───────────────────────────                                         │
│  为每个物体提取增强功能属性:                                          │
│  - action: 主要动作 (sit, eat, store, display...)                   │
│  - context: 使用场景 (dining, relaxation, work...)                  │
│  - posture: 交互姿态 (sitting, standing, bending...)                │
│  - duration: 交互时长 (momentary, short, extended...)               │
│  - co_objects: 协同物体 (chair需要table, lamp需要power outlet...)   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: 选取关键帧                                                  │
│  ─────────────────────                                               │
│  基于物体可见性变化选取代表性关键帧:                                  │
│  - 可见性变化检测 (新物体出现/消失)                                  │
│  - 稳定片段采样 (变化小的区间取中点)                                 │
│  - 覆盖率最大化 (确保所有物体被至少一帧覆盖)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: 分析轨迹行为                                                │
│  ─────────────────────                                               │
│  从相机轨迹中提取行为模式:                                           │
│  - 停留点 (Dwell Points): 相机长时间停留的位置                       │
│  - 环顾事件 (Look-Around): 原地旋转观察的行为                        │
│  - 快速穿越 (Traverse): 快速移动的路径段                             │
│  输出: 重要性热力图                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: VLM分析关键帧 (可选)                                        │
│  ───────────────────────────                                         │
│  使用视觉语言模型分析关键帧:                                          │
│  - 单帧分析: 识别功能物体组合                                        │
│  - 帧对对比: 识别功能区域边界                                        │
│  - 片段总结: 提取视觉上下文                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6: LLM推理功能区域 ★核心★                                      │
│  ─────────────────────────────                                       │
│  三步迭代推理:                                                       │
│  6.1 区域推理: 基于VLM分析+物体affordance+轨迹→推理功能区域          │
│  6.2 物体分配: 基于区域定义+物体affordance→分配物体到区域            │
│  6.3 验证修正: 检查一致性，提出修正建议                              │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 7: 构建层次化场景图                                            │
│  ───────────────────────────                                         │
│  组装最终输出:                                                       │
│  - 三层层次结构 (SpatialUnit → FunctionalZone → ObjectCluster)      │
│  - 任务接口 (导航目标、物体搜索提示、任务区域)                       │
│  - 可视化输出 (仪表盘、俯视图)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三层层次结构

### 数据模型

```
📍 SpatialUnit (空间单元)
│   - 最粗粒度的空间划分
│   - 通常对应一个房间或独立空间
│   - 例: "living_room", "kitchen", "bedroom"
│
└─── 🎯 FunctionalZone (功能区域)
     │   - 基于功能用途的子区域划分
     │   - 包含支持的活动类型
     │   - 例: "seating_area", "dining_area", "work_desk"
     │
     └─── 📦 ObjectCluster (物体群组)
          - 功能相关的物体组合
          - 例: "dining_set" (table + chairs + tableware)
```

### 物体-区域关系类型

| 关系类型 | 说明 | 示例 |
|----------|------|------|
| **Defining** | 定义性物体，决定区域功能 | 餐桌定义"用餐区" |
| **Supporting** | 支持性物体，辅助区域功能 | 餐椅支持"用餐区" |
| **Shared** | 共享物体，跨区域使用 | 台灯可能被多个区域共享 |
| **Boundary** | 边界物体，标记区域分界 | 书架分隔"客厅"和"书房" |

### 示例输出

```json
{
  "scene_id": "room0",
  "spatial_units": [
    {
      "unit_id": "su_0",
      "unit_name": "room0_room",
      "unit_type": "room",
      "functional_zones": ["fz_0", "fz_1", "fz_2"]
    }
  ],
  "functional_zones": [
    {
      "zone_id": "fz_0",
      "zone_name": "seating_and_social_area",
      "primary_activity": "休闲、社交和休息",
      "supported_activities": ["sit", "relax", "chat", "read"],
      "objects": [...],
      "defining_evidence": {
        "video": "多帧中出现沙发、扶手椅、茶几...",
        "objects": "沙发、单椅、茶几、地毯...",
        "trajectory": "相机多次停留且环顾于沙发区..."
      },
      "confidence": 0.95
    }
  ]
}
```

---

## 模块详解

### 1. 数据结构模块 (`data_structures.py`)

定义所有核心数据类：

```python
@dataclass
class HierarchicalSceneGraph:
    """层次化场景图"""
    scene_id: str
    spatial_units: List[SpatialUnit]
    functional_zones: List[FunctionalZone]
    object_clusters: List[ObjectCluster]
    zone_relations: List[ZoneRelation]
    task_affordances: TaskAffordances
    metadata: Dict[str, Any]

@dataclass
class FunctionalZone:
    """功能区域"""
    zone_id: str
    zone_name: str
    parent_unit: str
    primary_activity: str
    supported_activities: List[str]
    affordances: List[str]
    spatial: SpatialInfo
    objects: List[ObjectInfo]
    defining_evidence: Dict[str, str]
    confidence: float

@dataclass
class EnhancedAffordance:
    """增强功能属性"""
    action: str          # 主要动作
    context: str         # 使用场景
    duration: str        # 交互时长
    co_objects: List[str] # 协同物体
    posture: str         # 交互姿态
    frequency: str       # 使用频率
```

### 2. Affordance提取模块 (`enhanced_affordance.py`)

为每个物体提取功能属性：

```python
class EnhancedAffordanceExtractor:
    """增强Affordance提取器"""
    
    # 预定义的Affordance映射表
    DEFAULT_AFFORDANCE_MAP = {
        "chair": EnhancedAffordance(
            action="sit", context="seating", duration="extended",
            co_objects=["table", "desk"], posture="sitting", frequency="frequent"
        ),
        "table": EnhancedAffordance(
            action="place_items", context="dining", duration="extended",
            co_objects=["chair"], posture="sitting", frequency="frequent"
        ),
        "sofa": EnhancedAffordance(
            action="sit", context="relaxation", duration="extended",
            co_objects=["coffee_table", "lamp"], posture="sitting", frequency="frequent"
        ),
        # ... 更多物体
    }
    
    def extract(self, object_tag: str) -> EnhancedAffordance:
        """提取物体affordance，优先使用预定义，否则调用LLM"""
        if object_tag in self.DEFAULT_AFFORDANCE_MAP:
            return self.DEFAULT_AFFORDANCE_MAP[object_tag]
        return self._llm_extract(object_tag)
```

### 3. 关键帧选取模块 (`visibility_keyframe.py`)

基于物体可见性变化选取代表性帧：

```python
class VisibilityBasedKeyframeSelector:
    """基于可见性的关键帧选择器"""
    
    def select_keyframes(self, visibility_matrix: np.ndarray, 
                         n_keyframes: int = 15) -> List[KeyframeInfo]:
        """
        选取关键帧的三步策略:
        1. 可见性变化点: 检测物体出现/消失的帧
        2. 稳定段采样: 在变化小的区间取中点
        3. 覆盖率最大化: 确保所有物体被覆盖
        """
        keyframes = []
        
        # Step 1: 检测可见性变化点
        change_frames = self._detect_visibility_changes(visibility_matrix)
        keyframes.extend(change_frames)
        
        # Step 2: 稳定段采样
        stable_frames = self._sample_stable_segments(visibility_matrix)
        keyframes.extend(stable_frames)
        
        # Step 3: 覆盖率最大化
        while not self._all_objects_covered(keyframes, visibility_matrix):
            best_frame = self._find_best_coverage_frame(keyframes, visibility_matrix)
            keyframes.append(best_frame)
        
        return sorted(keyframes)[:n_keyframes]
```

### 4. 轨迹行为分析模块 (`trajectory_behavior.py`)

从相机轨迹中提取行为模式：

```python
class TrajectoryBehaviorAnalyzer:
    """轨迹行为分析器"""
    
    def analyze(self, poses: np.ndarray) -> TrajectoryBehaviorAnalysis:
        """
        分析相机轨迹，提取:
        - 停留点 (Dwell Points): 速度 < 阈值持续 N 帧
        - 环顾事件 (Look-Around): 位移小但旋转大
        - 快速穿越 (Traverse): 速度 > 阈值
        """
        analysis = TrajectoryBehaviorAnalysis()
        
        # 计算速度和角速度
        velocities = self._compute_velocities(poses)
        angular_velocities = self._compute_angular_velocities(poses)
        
        # 检测停留点
        dwell_mask = velocities < self.dwell_threshold
        analysis.dwell_points = self._extract_dwell_points(poses, dwell_mask)
        
        # 检测环顾事件
        look_around_mask = (velocities < self.dwell_threshold) & \
                          (angular_velocities > self.look_around_threshold)
        analysis.look_around_events = self._extract_look_around(poses, look_around_mask)
        
        # 生成重要性热力图
        analysis.importance_heatmap = self._compute_importance_heatmap(
            poses, analysis.dwell_points, analysis.look_around_events
        )
        
        return analysis
```

### 5. VLM功能分析模块 (`vlm_functional_analyzer.py`)

使用视觉语言模型分析关键帧：

```python
class VLMFunctionalAnalyzer:
    """VLM功能组合分析器"""
    
    def analyze_frame(self, image_path: str) -> FrameAnalysis:
        """
        单帧分析，识别功能物体组合
        
        Prompt设计:
        "分析这张室内场景图片，识别功能相关的物体组合。
         对于每个组合，说明:
         1. 组合名称 (如 dining_set, seating_area)
         2. 主要功能 (用餐、休息、工作等)
         3. 包含的物体列表
         4. 支持的活动类型"
        """
        response = self._call_vlm(image_path, self.FRAME_ANALYSIS_PROMPT)
        return self._parse_frame_analysis(response)
    
    def compare_frames(self, frame1_path: str, frame2_path: str) -> BoundaryAnalysis:
        """
        帧对对比，识别功能区域边界
        
        Prompt设计:
        "比较这两张图片，判断它们是否属于不同的功能区域。
         如果是，说明:
         1. 区域变化的类型 (如从客厅到餐厅)
         2. 边界指示物 (如门、走廊、家具分隔)
         3. 功能变化的证据"
        """
        response = self._call_vlm([frame1_path, frame2_path], self.BOUNDARY_PROMPT)
        return self._parse_boundary_analysis(response)
```

### 6. LLM区域推理模块 (`llm_zone_inference.py`) ★核心★

三步迭代推理机制：

```python
class LLMZoneInference:
    """LLM区域推理器"""
    
    def step1_infer_zones(self, vlm_analysis: List[FrameAnalysis],
                          affordances: List[EnhancedAffordance],
                          trajectory: TrajectoryBehaviorAnalysis) -> ZoneInferenceResult:
        """
        Step 1: 功能区域推理
        
        输入:
        - VLM帧分析结果 (功能组合、边界指示)
        - 物体affordance列表
        - 轨迹行为分析 (停留点、环顾事件)
        
        Prompt:
        "基于以下信息，推理该场景的功能区域划分:
         
         1. 视觉分析结果:
         {vlm_analysis}
         
         2. 物体功能属性:
         {affordances_summary}
         
         3. 相机轨迹行为:
         - 停留点: {dwell_points}
         - 环顾位置: {look_around_positions}
         
         请输出:
         - 功能区域列表 (名称、主要功能、支持的活动)
         - 每个区域的定义性证据
         - 区域边界描述"
        """
        prompt = self._build_zone_inference_prompt(vlm_analysis, affordances, trajectory)
        response = self._call_llm(prompt)
        return self._parse_zone_result(response)
    
    def step2_assign_objects(self, zones: List[FunctionalZone],
                              objects: List[ObjectInfo]) -> ObjectAssignmentResult:
        """
        Step 2: 物体-区域分配
        
        Prompt:
        "将以下物体分配到对应的功能区域:
         
         功能区域:
         {zones_description}
         
         待分配物体:
         {objects_with_affordances}
         
         对于每个物体，输出:
         - 所属区域ID
         - 关系类型 (defining/supporting/shared/boundary)
         - 分配理由"
        """
        prompt = self._build_assignment_prompt(zones, objects)
        response = self._call_llm(prompt)
        return self._parse_assignment_result(response)
    
    def step3_validate_and_refine(self, zones: ZoneInferenceResult,
                                   assignments: ObjectAssignmentResult) -> ValidationResult:
        """
        Step 3: 验证与修正
        
        检查:
        - 每个区域是否有定义性物体
        - 物体分配是否与affordance一致
        - 空间位置是否合理
        
        如有问题，提出修正建议
        """
        prompt = self._build_validation_prompt(zones, assignments)
        response = self._call_llm(prompt)
        return self._parse_validation_result(response)
```

### 7. 层次化构建器 (`hierarchical_builder.py`)

整合所有模块：

```python
class HierarchicalSceneBuilder:
    """层次化场景图构建器"""
    
    def build(self) -> HierarchicalSceneGraph:
        """构建完整的层次化场景图"""
        
        # Step 1: 加载数据
        self._load_data()
        print(f"  位姿: {len(self.poses)} 帧")
        print(f"  3D物体: {len(self.objects)} 个")
        
        # Step 2: 提取Affordance
        self._extract_affordances()
        
        # Step 3: 选取关键帧
        self._select_keyframes()
        
        # Step 4: 分析轨迹
        self._analyze_trajectory()
        
        # Step 5: VLM分析 (可选)
        if self.use_vlm:
            self._run_vlm_analysis()
        
        # Step 6: LLM推理
        if self.use_llm:
            self._run_llm_inference()
        
        # Step 7: 组装场景图
        return self._build_scene_graph()
```

---

## 任务接口

### TaskInterface (`task_interface.py`)

提供面向下游任务的查询接口：

```python
class TaskInterface:
    """任务接口"""
    
    def get_navigation_goals(self) -> List[NavigationGoal]:
        """获取导航目标点列表"""
        return [
            NavigationGoal(
                zone_id=zone.zone_id,
                zone_name=zone.zone_name,
                position=zone.spatial.center,
                activity=zone.primary_activity
            )
            for zone in self.scene_graph.functional_zones
        ]
    
    def find_object(self, object_query: str) -> List[ObjectSearchHint]:
        """搜索物体，返回可能的区域"""
        hints = []
        for zone in self.scene_graph.functional_zones:
            for obj in zone.objects:
                if self._match_query(obj.object_tag, object_query):
                    hints.append(ObjectSearchHint(
                        object_tag=obj.object_tag,
                        zone_name=zone.zone_name,
                        position=obj.position,
                        confidence=obj.confidence
                    ))
        return sorted(hints, key=lambda x: -x.confidence)
    
    def get_task_zone(self, task: str) -> Optional[FunctionalZone]:
        """根据任务获取对应区域"""
        task_activity_map = {
            "eat": ["dining", "eating"],
            "rest": ["relaxation", "休息", "休闲"],
            "work": ["work", "study", "工作"],
            "cook": ["cooking", "kitchen"],
        }
        activities = task_activity_map.get(task, [task])
        for zone in self.scene_graph.functional_zones:
            if any(act in zone.primary_activity.lower() for act in activities):
                return zone
        return None
```

---

## 可视化输出

### 1. 仪表盘 (`hierarchical_dashboard.png`)

四面板布局:

| 位置 | 内容 |
|------|------|
| 左上 | **俯视图**: 物体分布 + 区域边界框 + 区域标签 |
| 右上 | **层次树**: SpatialUnit → FunctionalZone → Objects |
| 左下 | **柱状图**: 每个区域的物体数量 |
| 右下 | **饼图**: 物体-区域关系类型分布 |

### 2. 俯视图 (`zone_map_topdown.png`)

- 不同颜色表示不同功能区域
- 圆点表示物体位置
- 矩形框表示区域边界
- 文字标签显示区域名称

### 3. 场景摘要 (`scene_summary.json`)

```json
{
  "scene_id": "room0",
  "n_spatial_units": 1,
  "n_functional_zones": 3,
  "n_objects": 75,
  "zones_summary": [
    {
      "name": "seating_and_social_area",
      "activity": "休闲、社交和休息",
      "n_objects": 7,
      "confidence": 0.95
    }
  ]
}
```

---

## 使用方法

### Bash 脚本

```bash
# 基本用法
bash bashes/4.5b_hierarchical_segmentation.sh room0

# 禁用VLM (仅使用LLM)
bash bashes/4.5b_hierarchical_segmentation.sh room0 --no_vlm

# 禁用LLM (仅使用规则)
bash bashes/4.5b_hierarchical_segmentation.sh room0 --no_llm

# 自定义LLM地址
bash bashes/4.5b_hierarchical_segmentation.sh room0 --llm_url http://localhost:8000

# 调整采样步长
bash bashes/4.5b_hierarchical_segmentation.sh room0 --stride 10
```

### Python 直接调用

```bash
python -m conceptgraph.segmentation.hierarchical_builder \
    --scene_path /path/to/Replica/room0 \
    --output /path/to/output/hierarchical_scene_graph.json \
    --stride 5 \
    --llm_url http://10.21.231.7:8006
```

### 环境变量

```bash
# 设置LLM服务地址 (推荐)
export LLM_BASE_URL="http://10.21.231.7:8006"

# 设置数据集根目录
export REPLICA_ROOT="$HOME/Datasets/Replica/Replica"
```

---

## 配置参数

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scene_path` | (必需) | 场景数据路径 |
| `--output` | `{scene}/hierarchical_segmentation/` | 输出目录 |
| `--stride` | 5 | 帧采样步长 |
| `--n_keyframes` | 15 | 关键帧数量 |
| `--no_vlm` | False | 禁用VLM视觉分析 |
| `--no_llm` | False | 禁用LLM推理 (使用规则) |
| `--llm_url` | `$LLM_BASE_URL` | LLM服务地址 |

### 内部配置

```python
# 轨迹分析参数
DWELL_THRESHOLD = 0.02        # 停留点速度阈值 (m/frame)
DWELL_MIN_FRAMES = 10         # 停留点最小帧数
LOOK_AROUND_THRESHOLD = 0.1   # 环顾角速度阈值 (rad/frame)

# 关键帧选取参数
VISIBILITY_CHANGE_THRESHOLD = 3  # 可见性变化阈值 (物体数)
MIN_KEYFRAME_DISTANCE = 20       # 关键帧最小间距

# LLM参数
LLM_MODEL = "gpt-5.2-2025-12-11"           # 或 "qwen2.5:72b" 等
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096
```

---

## 输出文件

```
{scene}/hierarchical_segmentation/
├── hierarchical_scene_graph.json   # 完整层次化场景图
├── hierarchical_dashboard.png      # 多面板可视化仪表盘
├── zone_map_topdown.png            # 俯视图可视化
└── scene_summary.json              # 场景摘要统计
```

---

## 代码文件列表

```
conceptgraph/segmentation/
├── __init__.py
├── data_structures.py          # 数据结构定义
├── enhanced_affordance.py      # Affordance提取
├── visibility_keyframe.py      # 关键帧选取
├── trajectory_behavior.py      # 轨迹行为分析
├── vlm_functional_analyzer.py  # VLM功能分析
├── llm_zone_inference.py       # LLM区域推理 ★
├── hierarchical_builder.py     # 主构建器
├── hierarchical_visualizer.py  # 可视化生成
├── task_interface.py           # 任务接口
└── object_region_relation.py   # 物体-区域关系分类

bashes/
└── 4.5b_hierarchical_segmentation.sh  # 执行脚本
```

---

## 与原方法的对比实验

### room0 场景结果

| 指标 | 原方法 (4.5) | 新方法 (4.5b) |
|------|--------------|---------------|
| 区域数量 | 3 (自动检测) | 3 (LLM推理) |
| 区域命名 | region_0, region_1, region_2 | seating_and_social_area, display_and_storage_area, dining_area |
| 功能描述 | 无 | 休闲社交、装饰展示、用餐 |
| 物体分配 | 基于可见性统计 | 基于功能匹配 |
| 可解释性 | 信号数值 | 自然语言证据 |
| 任务支持 | 无 | 导航、搜索、规划接口 |

---

## 限制与未来工作

### 当前限制

1. **依赖LLM质量**: 区域推理质量受LLM能力影响
2. **单房间假设**: 当前假设输入是单个空间单元
3. **物体标签依赖**: 需要准确的物体语义标签

### 未来改进方向

1. **多房间支持**: 自动检测和分割多个空间单元
2. **时序动态**: 支持场景变化的增量更新
3. **交互式修正**: 允许用户反馈修正区域划分
4. **评估指标**: 设计功能划分质量的评估方法

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2025-01 | 初始实现：三层层次结构 + LLM推理 |

---

## 参考资料

1. ConceptGraphs: Open-Vocabulary 3D Scene Graphs (2023)
2. ScanNet: Richly-annotated 3D Reconstructions (2017)
3. Replica Dataset: A Photorealistic Indoor Dataset (2019)
