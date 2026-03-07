# Open-World Query Parsing + Keyframe Retrieval 执行计划（v4）

## 1. 执行策略（强顺序）
1. 先用 `GPT-5.2` 在 `room0` 做真实分布 query 构建与 selector 端到端验证。
2. 仅在第 1 阶段通过后，启动 `Qwen3` 训练数据构造与训练。
3. 最后用 `Qwen3` 替换 `GPT-5.2` 做 query 解析，并做同集对比回归。

说明：本计划当前不做离线检索评测集，不产出 `retrieval_eval.jsonl`。

## 2. 阶段 A：GPT-5.2 + room0 先验证 selector（当前最高优先级）

### A1. 构建 room0 真实分布 query 集
目标：得到一批接近真实用户查询分布的文本 query，用于 selector 端到端验证。

输入资产：
1. `plans/generated_open_world/scene_manifest.jsonl`
2. `plans/generated_open_world/query_program_pool.jsonl`
3. `room0/indices/visibility_index.pkl`（若不存在先构建）

query 分布规则（room0）：
1. `present_normal`（目标在场景中）占 70%。
2. `hard_missing`（目标类不可见或被掩蔽）占 30%。
3. `present_normal` 内部意图分布：
   - direct 指认：40%
   - spatial relation：35%
   - superlative/ordinal：15%
   - attribute + relation：10%
4. 同义表达要求：
   - 至少 50% query 含同义/近义表达（如 `cushion/couch/lamp/footstool`）。
5. 长度分布要求：
   - 短句（4-8 词）30%
   - 中句（9-14 词）50%
   - 长句（15-24 词）20%

建议样本规模（第一轮）：
1. `N=200`（可先 smoke: `N=40`）。

产物：
1. `plans/generated_room0_gpt52_eval/room0_query_benchmark.jsonl`
2. `plans/generated_room0_gpt52_eval/query_generation_report.md`

单条样本格式（示例）：
```json
{
  "query_id": "room0_q_000123",
  "scene_id": "room0",
  "bucket": "present_normal",
  "intent_type": "spatial_relation",
  "user_query": "find the cushion on the couch nearest the door",
  "program_hash": "ab12cd34ef56",
  "source_program_type": "spatial",
  "target_presence": "present",
  "teacher_model": "gpt-5.2-2025-12-11",
  "prompt_version": "p_selector_eval_room0_v1_20260307"
}
```

### A2. 用 GPT-5.2 解析并运行 selector 端到端
执行链路：
1. query text -> `KeyframeSelector.parse_query_hypotheses`（模型：`gpt-5.2-2025-12-11`）
2. `execute_hypotheses` 执行
3. `select_keyframes_v2` 选帧
4. 记录每条 query 的运行日志

运行输出（每条 query）必须记录：
1. `parse_ok`（结构化输出是否合法）
2. `final_status`（`direct_grounded/proxy_grounded/context_only/no_evidence`）
3. `first_hit_kind`（`direct/proxy/context/none`）
4. `selected_frame_ids`（top-k）
5. `resolved_image_paths`（确保文件存在）
6. `latency_ms_parse` / `latency_ms_total`

产物：
1. `plans/generated_room0_gpt52_eval/selector_run.jsonl`
2. `plans/generated_room0_gpt52_eval/selector_summary.md`

### A3. 阶段 A gate（通过后才进入 Qwen3）
必须同时满足：
1. `parse_ok_rate >= 99%`
2. `resolved_image_exists_rate = 100%`
3. `present_normal` 中 `no_evidence_rate <= 15%`
4. `hard_missing` 中 `proxy_grounded + context_only >= 60%`
5. 无结构化字段越界（`HypothesisOutputV1` 校验通过）

## 3. 阶段 B：Qwen3 数据构造与训练（仅在 A 通过后）

### B1. 构造 Qwen3 训练数据
数据来源：
1. 阶段 A 的 query 集（真实文本分布）
2. `query_program_pool` 的结构意图
3. GPT-5.2 产出的高质量结构化解析结果（经 schema + 类别约束过滤）

训练集格式：
1. 仅保留 `parser_sft.jsonl`，字段为 `user_query + target_output(HypothesisOutputV1)`。
2. 不包含 `gold_keyframes/gold_status`。

分桶策略：
1. `direct/soft/hard = 40/30/30`（hard 为训练增强桶，不等同线上真实占比）。

切分策略：
1. 优先 scene 互斥（多场景时）。
2. 同 `program_hash` 不跨 split。
3. paraphrase 变体不跨 split。

### B2. Qwen3 训练与离线解析质量检查
检查项：
1. schema 通过率（`target_output`）
2. 类别约束通过率（in scene or `UNKNOW`）
3. hard 掩蔽泄漏率（=0）

产物：
1. `plans/generated_open_world_qwen3/parser_sft_train.jsonl`
2. `plans/generated_open_world_qwen3/parser_sft_val.jsonl`
3. `plans/generated_open_world_qwen3/training_report.md`

## 4. 阶段 C：Qwen3 替换 GPT-5.2 解析器

### C1. 推理接入
1. 在 query parser 配置中新增 `qwen3` 解析后端。
2. 默认策略：`qwen3` 主用，`gpt-5.2` 作为可切换回退。

### C2. 同集回归（使用阶段 A 同一 query 基准集）
对比对象：
1. `GPT-5.2 baseline`（阶段 A 结果）
2. `Qwen3 candidate`

回归判定：
1. `parse_ok_rate` 不低于 baseline - 1%
2. `present_normal no_evidence_rate` 不高于 baseline + 5%
3. `resolved_image_exists_rate` 保持 100%
4. 关键错误类型（结构错误/类别越界）不新增

通过后动作：
1. 将默认解析器切到 `qwen3`。
2. 保留 `gpt-5.2` 紧急回退开关。

## 5. 当前执行清单（按顺序）
1. [ ] 生成 `room0_query_benchmark.jsonl`（真实分布，N=200，含 30% hard）。
2. [ ] 跑 `GPT-5.2` + selector 端到端批处理，输出 `selector_run.jsonl`。
3. [ ] 产出 `selector_summary.md`，执行阶段 A gate 判定。
4. [ ] 若 A 通过：构造 `Qwen3 parser_sft` 训练数据（40/30/30）。
5. [ ] 训练 Qwen3 并产出 `training_report.md`。
6. [ ] 接入 Qwen3 解析器，跑同集回归与替换。

## 6. 建议命令（阶段 A）
1. 构建可见性索引（若缺失）：
   - `bash bashes/6b_build_visibility_index.sh room0`
2. 先确保 room0 资产存在：
   - `python conceptgraph/scripts/build_open_world_dataset_assets.py --scene room0=/abs/path/to/room0 --output_dir plans/generated_open_world`
3. 生成 query benchmark（待实现/扩展脚本）：
   - `python conceptgraph/scripts/build_room0_query_benchmark.py --scene_manifest plans/generated_open_world/scene_manifest.jsonl --query_program_pool plans/generated_open_world/query_program_pool.jsonl --output_dir plans/generated_room0_gpt52_eval --num_queries 200 --llm_model gpt-5.2-2025-12-11`
4. 批量运行 selector（待实现/扩展脚本）：
   - `python conceptgraph/scripts/run_selector_e2e_benchmark.py --scene_id room0 --scene_path /abs/path/to/room0 --query_file plans/generated_room0_gpt52_eval/room0_query_benchmark.jsonl --llm_model gpt-5.2-2025-12-11 --top_k 3 --output_dir plans/generated_room0_gpt52_eval`

## 7. 与案例文件关系
1. 本文件定义“执行顺序 + gate + 产物”。
2. 具体样本格式与 query 案例放在 `plans/open_world_query_keyframe_examples.md`。
3. 若案例与本计划冲突，以本文件（v4）为准并同步修订案例文件。
