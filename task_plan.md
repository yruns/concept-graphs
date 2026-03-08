# Task Plan: KeyframeSelector 机制研究

## Goal
深入理解 `conceptgraph/query_scene/keyframe_selector.py` 的工作机制

## Status: in_progress

## Phases

### Phase 1: 核心数据结构 [complete]
- SceneObject: 场景物体表示
- KeyframeResult: 关键帧选择结果

### Phase 2: 场景加载流程 [complete]
- 从 pkl.gz 加载 3D 物体
- 加载相机位姿
- 构建可见性索引

### Phase 3: 查询解析机制 [pending]
- LLM 解析查询
- Hypothesis 多假设系统

### Phase 4: 物体匹配机制 [pending]
- 字符串匹配
- CLIP 语义匹配

### Phase 5: 关键帧选择策略 [pending]
- joint_coverage 策略
- 空间关系过滤

## Key Files
- `keyframe_selector.py` - 主文件 (1829 行)
- `query_structures.py` - 查询结构定义
- `query_parser.py` - 查询解析器
- `query_executor.py` - 查询执行器
- `spatial_relations.py` - 空间关系检查

## Errors Encountered
(none yet)
