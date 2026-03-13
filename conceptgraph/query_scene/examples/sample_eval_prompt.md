# LLMEvaluator v2 完整评测 Prompt 示例

> 本文档展示发送给 Gemini 的完整评测 Prompt，包括文本和图像占位符。

---

## 发送给 Gemini 的内容结构

```
[Content 1: Text Prompt]
[Content 2: Image - Keyframe 0]
[Content 3: Image - Keyframe 1]
[Content 4: Image - Keyframe 2]
[Content 5: Image - BEV 俯视图]
```

---

## Content 1: Text Prompt

```
# Keyframe Selection Evaluation (Blind Mode)

## Task
Evaluate whether the selected keyframes adequately support answering the given query.
You are evaluating the SELECTOR's choices, not the query parsing.

## Original Query
"the pillow on the sofa near the window"

## Parsed Query Structure
The query was parsed as:
- **Target Categories**: ['pillow', 'throw_pillow']
- **Anchor Categories**: ['sofa', 'window']
- **Spatial Relation**: on
- **Hypothesis Kind**: direct

## Selected Keyframes
3 keyframes were selected (Images 1-3).
The last image (Image 4) is a Bird's Eye View showing the spatial layout.

## Evaluation Dimensions

For EACH keyframe (by index 0 to 2), score:

1. **target_visibility** (0-10): Can you see objects matching the target categories?
   - Look for: ['pillow', 'throw_pillow']

2. **target_completeness** (0-10): Are the target objects fully visible, not cropped/occluded?

3. **spatial_context** (0-10 or null): Can you verify the spatial relation "on"?
   - Score null if no spatial relation in query

4. **image_quality** (0-10): Overall image quality for this evaluation task

## Important
- Focus on whether the keyframes SHOW the query targets, not whether parsing is correct
- Be strict: if you cannot clearly identify the target object, score low
- Consider if a human could use these frames to answer the query
- Output ONLY index numbers (0, 1, 2...), NOT file paths or view IDs

## Response Format (JSON only)
{
  "per_keyframe_evals": [
    {
      "keyframe_idx": 0,
      "target_visibility": <0-10>,
      "target_completeness": <0-10>,
      "spatial_context": <0-10 or null>,
      "image_quality": <0-10>,
      "observations": "<what you see>"
    }
  ],
  "target_visibility": <avg across frames>,
  "target_completeness": <avg across frames>,
  "spatial_context": <avg or null>,
  "image_quality": <avg>,
  "selector_score": <weighted average>,
  "best_keyframe_idx": <index>,
  "can_answer_query": true/false,
  "reasoning": "<explanation>",
  "issues": ["issue1", ...]
}
```

---

## Content 2: Image - Keyframe 0 (view_id=45)

```
[IMAGE: /Users/bytedance/Replica/room0/results/frame000225.jpg]

这是 Selector 选择的第一个关键帧，view_id=45 (frame_idx = 45 * 5 = 225)
展示了从某个视角看到的场景，Gemini 需要判断是否能看到 pillow 和 sofa
```

---

## Content 3: Image - Keyframe 1 (view_id=52)

```
[IMAGE: /Users/bytedance/Replica/room0/results/frame000260.jpg]

这是 Selector 选择的第二个关键帧，view_id=52 (frame_idx = 52 * 5 = 260)
展示了另一个视角
```

---

## Content 4: Image - Keyframe 2 (view_id=67)

```
[IMAGE: /Users/bytedance/Replica/room0/results/frame000335.jpg]

这是 Selector 选择的第三个关键帧，view_id=67 (frame_idx = 67 * 5 = 335)
展示了第三个视角
```

---

## Content 5: Image - BEV 俯视图

```
[IMAGE: /Users/bytedance/Replica/room0/bev/scene_bev_e9ff6a93.png]

Bird's Eye View 俯视图，由 KeyframeSelector._generate_scene_images() 生成
- 保存路径: {scene_path}/bev/scene_bev_{config_hash}.png
- 配置: perspective=True, show_objects=False, show_labels=False
- 使用场景 mesh 渲染，纯视觉无标注
- 用于验证 "on the sofa near the window" 这样的空间关系


```

---

## Gemini 预期返回的 JSON 示例

```json
{
  "per_keyframe_evals": [
    {
      "keyframe_idx": 0,
      "target_visibility": 8.5,
      "target_completeness": 7.0,
      "spatial_context": 8.0,
      "image_quality": 8.5,
      "observations": "I can see a decorative pillow on a gray sofa. The pillow is clearly visible and the sofa is near a window on the right side of the frame."
    },
    {
      "keyframe_idx": 1,
      "target_visibility": 6.0,
      "target_completeness": 5.5,
      "spatial_context": 7.0,
      "image_quality": 7.5,
      "observations": "The pillow is partially visible, slightly occluded by another cushion. The sofa and window relationship is visible but not as clear."
    },
    {
      "keyframe_idx": 2,
      "target_visibility": 9.0,
      "target_completeness": 9.0,
      "spatial_context": 9.0,
      "image_quality": 8.0,
      "observations": "Excellent view of the pillow on the sofa. The window is clearly visible in the background, making the spatial relationship very clear."
    }
  ],
  "target_visibility": 7.83,
  "target_completeness": 7.17,
  "spatial_context": 8.0,
  "image_quality": 8.0,
  "selector_score": 7.73,
  "best_keyframe_idx": 2,
  "can_answer_query": true,
  "reasoning": "The selected keyframes adequately show the target pillow on the sofa. Keyframe 2 provides the best view with clear visibility of both the pillow and the spatial relationship with the window. A human could confidently identify the pillow described in the query using these frames.",
  "issues": [
    "Keyframe 1 has partial occlusion of the target pillow"
  ]
}
```

---

## 评分计算流程

### Stage 1: Parse Evaluation (代码计算，无 Gemini)

```python
# GT (来自 EvaluationCase)
gt_target_categories = ["pillow"]
gt_anchor_categories = ["sofa"]
gt_spatial_relation = "on"

# Parsed (来自 QueryParser)
parsed_target_categories = ["pillow", "throw_pillow"]
parsed_anchor_categories = ["sofa", "window"]
parsed_spatial_relation = "on"

# 计算匹配
target_match = compute_category_match(gt_target_categories, parsed_target_categories)
# -> match_score = 1.0 (pillow 和 throw_pillow 通过别名映射为同一类)

anchor_match = compute_category_match(gt_anchor_categories, parsed_anchor_categories)
# -> match_score = 0.5 (sofa 匹配, window 是 extra)

spatial_correct = (gt_spatial_relation == parsed_spatial_relation)
# -> True

# 动态权重计算
parse_score = 0.5 * 10.0 + 0.3 * 5.0 + 0.2 * 10.0
           = 5.0 + 1.5 + 2.0 = 8.5
```

### Stage 2: Selector Evaluation (Gemini 视觉评估)

```python
# Gemini 返回的分数
target_visibility = 7.83
target_completeness = 7.17
spatial_context = 8.0
image_quality = 8.0

# 动态权重计算 (有 spatial_context)
selector_score = 0.35 * 7.83 + 0.25 * 7.17 + 0.25 * 8.0 + 0.15 * 8.0
              = 2.74 + 1.79 + 2.0 + 1.2 = 7.73
```

### Stage 3: GT Comparison (可选诊断)

```python
# GT 目标对象
gt_target_obj_ids = [12, 15]  # 两个 pillow

# Selector 匹配到的对象
matched_obj_ids = [12, 15, 23]  # 多匹配了一个

# 计算覆盖率
gt_found = [12, 15]
gt_missed = []
extra_matched = [23]
coverage = 2/2 = 1.0
```

### Overall Score

```python
# 有 GT 比对时
overall_score = 0.30 * parse_score + 0.50 * selector_score + 0.20 * (coverage * 10)
             = 0.30 * 8.5 + 0.50 * 7.73 + 0.20 * 10.0
             = 2.55 + 3.87 + 2.0 = 8.42

# 无 GT 比对时
overall_score = 0.375 * parse_score + 0.625 * selector_score
             = 0.375 * 8.5 + 0.625 * 7.73
             = 3.19 + 4.83 = 8.02
```

---

## 关键设计点

### 1. Blind Mode - 无 GT 信息泄露

发送给 Gemini 的信息 **不包含**：
- ❌ `gt_target_obj_ids` - 防止答案泄露
- ❌ `matched_obj_ids` - 防止偏置评分
- ❌ GT source frame 标注图 - 防止锚定效应

只包含：
- ✅ 原始 query
- ✅ parsed 结构 (categories, relation)
- ✅ 选中的 keyframes
- ✅ BEV 俯视图

### 2. Index-Only Output

Prompt 明确要求 Gemini **只输出 index**，不输出 view_id 或 path：
- 防止 Gemini 幻觉 metadata
- 代码端根据 index 映射回 view_id 和 path

### 3. 动态权重归一化

当某些维度不存在时（如无 spatial_relation），自动归一化剩余权重：
- 有 spatial: `target=35%, completeness=25%, spatial=25%, quality=15%`
- 无 spatial: `target=45%, completeness=35%, quality=20%`

### 4. 分阶段评估

| 阶段 | 计算方式 | 目的 |
|------|----------|------|
| Stage 1 Parse | 代码精确计算 | 评估 QueryParser |
| Stage 2 Selector | Gemini 视觉评估 | 评估 KeyframeSelector |
| Stage 3 GT | 代码集合比较 | 诊断 GT 覆盖率 |
