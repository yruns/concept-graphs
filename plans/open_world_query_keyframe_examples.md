# Open-World Query + Keyframe 具体案例（10例）

## 1. 目的
本文件给出可直接照抄的样本格式与执行结果案例，覆盖：
1. `direct`（目标可见，单解）
2. `soft`（同义词/语序变化，可见但表达自由）
3. `hard`（目标被掩蔽，需多猜想）

主计划文件：`plans/open_world_query_keyframe_plan.md`  
本文件用于补充主计划第 7 节。

## 2. 案例分布
1. Direct: D01-D04（4例）
2. Soft: S01-S03（3例）
3. Hard: H01-H03（3例）

## 3. Direct 案例（4例）

### D01
用户查询：
```text
the throw pillow on the sofa
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_direct_D01",
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

规则执行期望：
1. first-hit hypothesis: `direct`
2. 结果状态: `direct_grounded`
3. 示例 keyframes: `[64, 88, 95]`

### D02
用户查询：
```text
the armchair nearest the door
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_direct_D02",
  "bucket": "direct",
  "scene_id": "room0",
  "scene_categories": ["armchair", "door", "sofa", "ottoman"],
  "user_query": "the armchair nearest the door",
  "target_mode": "single_or_multi",
  "hypotheses": [
    {
      "kind": "direct",
      "confidence": 1.0,
      "grounding_query": {
        "raw_query": "the armchair nearest the door",
        "root": {
          "categories": ["armchair"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "superlative",
            "metric": "distance",
            "order": "min",
            "reference": {"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""},
            "position": null
          },
          "node_id": ""
        },
        "expect_unique": true
      }
    }
  ]
}
```

规则执行期望：
1. 结果状态: `direct_grounded`
2. 示例 keyframes: `[22, 31, 47]`

### D03
用户查询：
```text
all throw_pillows near the sofa
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_direct_D03",
  "bucket": "direct",
  "scene_id": "room0",
  "scene_categories": ["throw_pillow", "sofa", "armchair", "window_blinds"],
  "user_query": "all throw_pillows near the sofa",
  "target_mode": "single_or_multi",
  "hypotheses": [
    {
      "kind": "direct",
      "confidence": 1.0,
      "grounding_query": {
        "raw_query": "all throw_pillows near the sofa",
        "root": {
          "categories": ["throw_pillow", "pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "near",
              "anchors": [
                {"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}
              ]
            }
          ],
          "select_constraint": null,
          "node_id": ""
        },
        "expect_unique": false
      }
    }
  ]
}
```

规则执行期望：
1. 结果状态: `direct_grounded`
2. 示例 keyframes: `[55, 74, 88]`

### D04
用户查询：
```text
the second largest side_table
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_direct_D04",
  "bucket": "direct",
  "scene_id": "room0",
  "scene_categories": ["side_table", "sofa", "door"],
  "user_query": "the second largest side_table",
  "target_mode": "single_or_multi",
  "hypotheses": [
    {
      "kind": "direct",
      "confidence": 1.0,
      "grounding_query": {
        "raw_query": "the second largest side_table",
        "root": {
          "categories": ["side_table"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "ordinal",
            "metric": "size",
            "order": "desc",
            "reference": null,
            "position": 2
          },
          "node_id": ""
        },
        "expect_unique": true
      }
    }
  ]
}
```

规则执行期望：
1. 结果状态: `direct_grounded`
2. 示例 keyframes: `[12, 28, 41]`

## 4. Soft 案例（3例）

### S01
用户查询：
```text
find the cushion by the couch
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_soft_S01",
  "bucket": "soft",
  "scene_id": "room0",
  "scene_categories": ["throw_pillow", "sofa", "door", "armchair"],
  "user_query": "find the cushion by the couch",
  "target_mode": "single_or_multi",
  "hypotheses": [
    {
      "kind": "direct",
      "confidence": 0.94,
      "grounding_query": {
        "raw_query": "find the cushion by the couch",
        "root": {
          "categories": ["throw_pillow", "pillow"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "near",
              "anchors": [
                {"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}
              ]
            }
          ],
          "select_constraint": null,
          "node_id": ""
        },
        "expect_unique": false
      }
    }
  ]
}
```

规则执行期望：
1. 结果状态: `direct_grounded`
2. 备注: 检验 `cushion/couch` 同义映射是否稳定。

### S02
用户查询：
```text
please locate the lamp next to the armchair
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_soft_S02",
  "bucket": "soft",
  "scene_id": "room0",
  "scene_categories": ["floor_lamp", "wall_sconce", "armchair", "sofa"],
  "user_query": "please locate the lamp next to the armchair",
  "target_mode": "single_or_multi",
  "hypotheses": [
    {
      "kind": "direct",
      "confidence": 0.92,
      "grounding_query": {
        "raw_query": "please locate the lamp next to the armchair",
        "root": {
          "categories": ["floor_lamp", "wall_sconce"],
          "attributes": [],
          "spatial_constraints": [
            {
              "relation": "next_to",
              "anchors": [
                {"categories": ["armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}
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

规则执行期望：
1. 结果状态: `direct_grounded`
2. 示例 keyframes: `[39, 63, 90]`

### S03
用户查询：
```text
which ottoman is closest to the sofa
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_soft_S03",
  "bucket": "soft",
  "scene_id": "room0",
  "scene_categories": ["ottoman", "sofa", "armchair", "side_table"],
  "user_query": "which ottoman is closest to the sofa",
  "target_mode": "single_or_multi",
  "hypotheses": [
    {
      "kind": "direct",
      "confidence": 0.9,
      "grounding_query": {
        "raw_query": "which ottoman is closest to the sofa",
        "root": {
          "categories": ["ottoman"],
          "attributes": [],
          "spatial_constraints": [],
          "select_constraint": {
            "constraint_type": "superlative",
            "metric": "distance",
            "order": "min",
            "reference": {"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""},
            "position": null
          },
          "node_id": ""
        },
        "expect_unique": true
      }
    }
  ]
}
```

规则执行期望：
1. 结果状态: `direct_grounded`
2. 备注: 检验问句/选择句式与 declarative 句式一致性。

## 5. Hard 案例（3例）

### H01
用户查询：
```text
find the cushion on the couch closest to the door
```

`retrieval_eval.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_hard_H01",
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

规则执行期望：
1. direct 失败 -> proxy 命中 -> context 不执行或仅作补充。
2. 最终状态：`proxy_grounded`。

### H02
用户查询：
```text
show me the table lamp beside the armchair
```

`retrieval_eval.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_hard_H02",
  "bucket": "hard",
  "scene_id": "room0",
  "mask_spec": {"type": "M1", "hidden_categories": ["floor_lamp", "wall_sconce"]},
  "scene_categories_masked": ["armchair", "sofa", "side_table", "door"],
  "user_query": "show me the table lamp beside the armchair",
  "qwen_target_output": {
    "target_mode": "single_or_multi",
    "hypotheses": [
      {"kind": "direct", "confidence": 0.41, "grounding_query": {"raw_query": "show me the table lamp beside the armchair", "root": {"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [{"relation": "beside", "anchors": [{"categories": ["armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}], "select_constraint": null, "node_id": ""}, "expect_unique": true}},
      {"kind": "proxy", "confidence": 0.39, "grounding_query": {"raw_query": "proxy lighting near armchair", "root": {"categories": ["wall_sconce", "ceiling_light"], "attributes": [], "spatial_constraints": [{"relation": "near", "anchors": [{"categories": ["armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}], "select_constraint": null, "node_id": ""}, "expect_unique": false}},
      {"kind": "context", "confidence": 0.2, "grounding_query": {"raw_query": "armchair zone", "root": {"categories": ["armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}, "expect_unique": false}}
    ]
  },
  "gold_status": "context_only",
  "gold_keyframes": [40, 61, 79]
}
```

规则执行期望：
1. proxy 不稳定时可退化到 context。
2. 最终状态允许 `context_only`，但不能直接 `no_evidence`。

### H03
用户查询：
```text
find the footstool between the sofa and the side table
```

`retrieval_eval.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_hard_H03",
  "bucket": "hard",
  "scene_id": "room0",
  "mask_spec": {"type": "M1+M2", "hidden_categories": ["ottoman"]},
  "scene_categories_masked": ["sofa", "side_table", "armchair", "door", "area_rug"],
  "user_query": "find the footstool between the sofa and the side table",
  "qwen_target_output": {
    "target_mode": "single_or_multi",
    "hypotheses": [
      {"kind": "direct", "confidence": 0.46, "grounding_query": {"raw_query": "find the footstool between the sofa and the side table", "root": {"categories": ["UNKNOW"], "attributes": [], "spatial_constraints": [{"relation": "between", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}, {"categories": ["side_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}], "select_constraint": null, "node_id": ""}, "expect_unique": true}},
      {"kind": "proxy", "confidence": 0.35, "grounding_query": {"raw_query": "proxy ottoman-like near sofa and side_table", "root": {"categories": ["stool", "area_rug"], "attributes": [], "spatial_constraints": [{"relation": "near", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}, {"relation": "near", "anchors": [{"categories": ["side_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}], "select_constraint": null, "node_id": ""}, "expect_unique": false}},
      {"kind": "context", "confidence": 0.19, "grounding_query": {"raw_query": "between sofa and side table area", "root": {"categories": ["sofa", "side_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}, "expect_unique": false}}
    ]
  },
  "gold_status": "proxy_grounded",
  "gold_keyframes": [33, 57, 71]
}
```

规则执行期望：
1. 强制测试 `between` 关系在 hard 桶下的降级策略。
2. direct 失败时必须仍返回 3 帧证据。

## 6. 快速使用说明（如何在实现中消费这些case）
1. 训练 Qwen 时仅使用 `parser_sft` 格式字段。
2. 调试规则执行器时使用 `retrieval_eval` 样本，读取 `qwen_target_output` 执行。
3. 对每条 hard case 记录：first-hit hypothesis、final status、top-k frame ids。
4. 若 final status 为 `no_evidence`，该样本进入 hard-case 回归集。

