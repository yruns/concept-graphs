# Open-World Query Parsing + Keyframe Retrieval 详细执行计划（v2）

## 1. 一句话目标
训练一个仅负责 `query -> 结构化表达` 的 Qwen 解析器，同时用**规则执行器**完成 keyframe 选择，并在 30% 漏检 hard-case 下仍返回有价值证据。

## 2. 固定约束与关键决策
1. 数据分布固定：`40% direct + 30% soft + 30% hard`。
2. hard-case 只占 30%，且通过“故意隐去真实存在类别”构造。
3. Qwen 只学解析，不学 keyframe 规则。
4. 解析输出必须兼容：有些 query 只需单解，有些 query 需要多猜想。
5. 先进 API 用于样本构造：`gpt-5.2-2025-12-11` 与 `gemini-3-pro-preview-new`。

## 3. 交付物清单（必须落地）
1. `parser_sft.jsonl`：Qwen 训练集，仅解析标签，无 keyframe。
2. `retrieval_eval.jsonl`：规则选帧评测集，含 `gold_keyframes` 与状态标签。
3. `scene_manifest.jsonl`：每个 scene 的 3D 类别、2D 类别、对象统计。
4. `generation_report.md`：样本比例、过滤率、重复率、失败样本原因。
5. `open_world_query_keyframe_examples.md`：10 个可执行 case（含样本 JSON 与规则执行结果）。

## 4. 数据格式定义（先定协议，再写脚本）

### 4.1 Qwen训练样本：`parser_sft.jsonl`
每行一个样本，字段如下：
```json
{
  "sample_id": "room0_direct_000123",
  "bucket": "direct",
  "scene_id": "room0",
  "scene_categories": ["throw_pillow", "sofa", "door", "armchair", "side_table"],
  "user_query": "the throw pillow on the sofa",
  "target_mode": "single_or_multi",
  "hypotheses": [
    {
      "kind": "direct",
      "confidence": 1.0,
      "grounding_query": {
        "raw_query": "the throw pillow on the sofa",
        "root": {
          "categories": ["throw_pillow", "pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "on",
              "anchors": [
                {"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}
              ]
            }
          ],
          "select_constraint": null,
          "node_id": ""
        },
        "expect_unique": true
      }
    }
  ]
}
```

### 4.2 规则评测样本：`retrieval_eval.jsonl`
每行一个样本，字段如下：
```json
{
  "sample_id": "room0_hard_000031",
  "bucket": "hard",
  "scene_id": "room0",
  "mask_spec": {"type": "M1+M2", "hidden_categories": ["throw_pillow", "pillow"]},
  "scene_categories_masked": ["sofa", "door", "armchair", "side_table", "throw_blanket", "sofa_seat_cushion"],
  "user_query": "find the cushion on the couch closest to the door",
  "qwen_target_output": {
    "target_mode": "single_or_multi",
    "hypotheses": [
      {"kind": "direct", "confidence": 0.44, "grounding_query": {"raw_query": "find the cushion on the couch closest to the door", "root": {"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [{"relation": "on", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": {"constraint_type": "superlative", "metric": "distance", "order": "min", "reference": {"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}, "position": null}, "node_id": ""}]}], "select_constraint": null, "node_id": ""}, "expect_unique": true}},
      {"kind": "proxy", "confidence": 0.37, "grounding_query": {"raw_query": "proxy cushion near sofa", "root": {"categories": ["sofa_seat_cushion", "throw_blanket"], "attributes": [], "spatial_constraints": [{"relation": "on", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}], "select_constraint": null, "node_id": ""}, "expect_unique": false}},
      {"kind": "context", "confidence": 0.19, "grounding_query": {"raw_query": "sofa near door", "root": {"categories": ["sofa"], "attributes": [], "spatial_constraints": [{"relation": "near", "anchors": [{"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}], "select_constraint": null, "node_id": ""}, "expect_unique": false}}
    ]
  },
  "gold_status": "proxy_grounded",
  "gold_keyframes": [88, 95, 103]
}
```

## 5. 样本构造流程（每一步“输入-动作-输出”）

### Step 1: 构建 scene manifest
1. 输入：`room*/pcd_saves/*_post.pkl.gz`、`gsa_detections_*/*.pkl.gz`、`gsa_classes_*.json`。
2. 动作：统计 3D object_tag 分布、2D class 分布、每类实例数。
3. 输出：`scene_manifest.jsonl`。
4. 验收：每个 scene 至少包含 `scene_id/scene_categories/object_counts/classes_2d`。

### Step 2: 生成结构化程序骨架（不写自然语言）
1. 输入：`scene_manifest.jsonl`。
2. 动作：按 L0-L4 程序化生成 `GroundingQuery` 树。
3. 输出：`query_program_pool.jsonl`。
4. 验收：每条程序可通过 schema 校验。

### Step 3: 桶采样（40/30/30）
1. 输入：`query_program_pool.jsonl`。
2. 动作：按比例抽样 direct/soft/hard。
3. 输出：`bucketed_program_pool.jsonl`。
4. 验收：比例偏差 <= 1%。

### Step 4: hard-case 掩蔽（仅 hard 桶）
1. 输入：hard 桶程序 + scene manifest。
2. 动作：执行 M1 或 M1+M2。
3. 输出：带 `mask_spec` 的 hard 程序集。
4. 验收：hard 样本目标类在 `scene_categories_masked` 中确实缺失。

### Step 5: 生成自由 query（先进 API）
1. 输入：程序骨架 + scene categories。
2. 动作：GPT 生成 canonical；Gemini 做 paraphrase。
3. 输出：`program_to_query_candidates.jsonl`。
4. 验收：每个程序至少 4 条 query 候选。

### Step 6: 结构化标签组装（兼容单解与多猜想）
1. 输入：query 候选 + 程序骨架 + mask 信息。
2. 动作：组装 `hypotheses`。
3. 输出：`parser_sft_raw.jsonl`。
4. 验收：direct 样本 `len(hypotheses)=1`；hard 样本 `len(hypotheses)>=2`。

### Step 7: 自动过滤
1. 输入：`parser_sft_raw.jsonl`。
2. 动作：schema 校验、执行一致性、去重。
3. 输出：`parser_sft.jsonl`。
4. 验收：无 schema failure，重复率可控（<5%）。

### Step 8: 生成规则评测集
1. 输入：hard + soft + direct 子集。
2. 动作：附加 `gold_status` 与 `gold_keyframes`。
3. 输出：`retrieval_eval.jsonl`。
4. 验收：每条样本包含 `gold_status`，且 keyframe 数量满足 `k` 设定。

## 6. 推理规则设计（Qwen输出如何被 KeyframeSelector 执行）

### 6.1 解析输出适配层（兼容单解/多猜想）
```python
def normalize_qwen_output(x: dict) -> list:
    if "hypotheses" in x:
        return sorted(x["hypotheses"], key=lambda h: h.get("confidence", 0), reverse=True)
    if "grounding_query" in x:
        return [{"kind": "direct", "confidence": 1.0, "grounding_query": x["grounding_query"]}]
    raise ValueError("invalid parser output")
```

### 6.2 执行逻辑（规则，不依赖学习）
```python
def select_keyframes_from_hypotheses(hypotheses, executor, vis_index, k=3):
    executed = []
    for h in hypotheses:
        result = executor.execute(h["grounding_query"])
        executed.append((h, result))
        if h["kind"] == "direct" and h["confidence"] >= 0.78 and not result.is_empty:
            return build_evidence("direct_grounded", h, result, vis_index, k)
        if not result.is_empty:
            return build_evidence("proxy_grounded" if h["kind"]=="proxy" else "context_only", h, result, vis_index, k)
    return build_no_evidence(executed)
```

### 6.3 候选视角打分
`S(view) = 0.50*object_coverage + 0.30*anchor_support + 0.20*text_evidence`
1. `object_coverage`：目标/代理对象在 view 的可见性得分。
2. `anchor_support`：anchor 对象在 view 的可见性得分。
3. `text_evidence`：2D 检测类命中 + CLIP query 相似度。

## 7. 三个核心 Case（摘要版，完整10例见补充文档）

完整案例文件：`plans/open_world_query_keyframe_examples.md`

### Case A: direct（40%）
1. scene categories：`["throw_pillow","sofa","door","armchair"]`
2. user query：`"the throw pillow on the sofa"`
3. qwen output：1条 `direct` hypothesis。
4. 执行：direct 命中 throw_pillow。
5. 输出：`status=direct_grounded`，返回目标+sofa联合覆盖的 top-k 视角。

### Case B: soft（30%）
1. scene categories：`["throw_pillow","sofa","door","armchair"]`
2. user query：`"find the cushion by the couch"`
3. qwen output：通常1条 direct，`categories=["throw_pillow","pillow"]`。
4. 执行：direct 命中。
5. 输出：`status=direct_grounded`，验证同义词鲁棒性。

### Case C: hard（30%）
1. 原始存在类：`throw_pillow`。
2. 掩蔽：M1+M2，删除 `throw_pillow/pillow`。
3. user query：`"find the cushion on the couch closest to the door"`
4. qwen output：`direct(UNKNOW) + proxy + context`。
5. 执行：direct 失败，proxy 命中，context 兜底。
6. 输出：`status=proxy_grounded`，返回 `sofa/door + proxy object` 证据帧。

## 8. 训练与评测执行清单（可打勾）
1. [ ] 生成 `scene_manifest.jsonl`（room0 先跑通）。
2. [ ] 生成 `query_program_pool.jsonl`（覆盖 L0-L4）。
3. [ ] 完成 40/30/30 分桶与 hard 掩蔽。
4. [ ] 完成 GPT+Gemini query 扩写与过滤。
5. [ ] 导出 `parser_sft.jsonl`（无 keyframe 字段）。
6. [ ] 训练 Qwen parser（只吃 parser_sft）。
7. [ ] 导出 `retrieval_eval.jsonl`（含 gold_keyframes）。
8. [ ] 接入解析适配层（兼容单解和多猜想）。
9. [ ] 接入规则选帧与 Evidence Pack 输出。
10. [ ] 分桶评测 direct/soft/hard 并输出报告。

## 9. 验收指标（分桶报告必须有）
1. Parser 分布正确率：40/30/30 偏差 <= 1%。
2. direct 桶：过度猜想率低，单解命中率高。
3. soft 桶：同义词表达下结构解析准确率稳定。
4. hard 桶：`proxy_grounded + context_only` 覆盖率显著高于空返回基线。
5. keyframe：hard 桶 `Recall@3` 比旧流程有明确提升。

## 10. 实施注意事项
1. 当前本机已完成 3D 产物的 Replica 场景以 `room0` 为主，其他场景先补 pipeline 再扩展。
2. 若后续改动 query_scene 行为或 pipeline 脚本，必须同步更新 `memory/query_scene_knowledge.md` 与 `memory/bash_scripts_index.md`。
