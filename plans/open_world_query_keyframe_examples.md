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

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_hard_H01",
  "bucket": "hard",
  "scene_id": "room0",
  "mask_spec": {"type": "M1+M2", "hidden_categories": ["throw_pillow", "pillow"]},
  "scene_categories_masked": ["sofa", "door", "armchair", "side_table", "throw_blanket", "sofa_seat_cushion"],
  "user_query": "find the cushion on the couch closest to the door",
  "target_output": {
    "format_version": "hypothesis_output_v1",
    "parse_mode": "multi",
    "hypotheses": [
      {
        "kind": "direct",
        "rank": 1,
        "grounding_query": {
          "raw_query": "find the cushion on the couch closest to the door",
          "root": {
            "categories": ["UNKNOW"],
            "attributes": [],
            "spatial_constraints": [{"relation": "on", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": {"constraint_type": "superlative", "metric": "distance", "order": "min", "reference": {"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}, "position": null}, "node_id": ""}]}],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": true
        },
        "lexical_hints": ["cushion", "couch"]
      },
      {
        "kind": "proxy",
        "rank": 2,
        "grounding_query": {
          "raw_query": "proxy cushion near sofa",
          "root": {
            "categories": ["sofa_seat_cushion", "throw_blanket"],
            "attributes": [],
            "spatial_constraints": [{"relation": "on", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": false
        },
        "lexical_hints": ["seat cushion"]
      },
      {
        "kind": "context",
        "rank": 3,
        "grounding_query": {
          "raw_query": "sofa near door",
          "root": {
            "categories": ["sofa"],
            "attributes": [],
            "spatial_constraints": [{"relation": "near", "anchors": [{"categories": ["door"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": false
        },
        "lexical_hints": ["seating area"]
      }
    ]
  }
}
```

运行期行为约束：
1. `direct` 失败后按 `rank` 回退到 `proxy`。
2. hard 桶不得出现 `hidden_categories` 泄漏。

### H02
用户查询：
```text
show me the table lamp beside the armchair
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_hard_H02",
  "bucket": "hard",
  "scene_id": "room0",
  "mask_spec": {"type": "M1", "hidden_categories": ["floor_lamp", "wall_sconce"]},
  "scene_categories_masked": ["armchair", "sofa", "side_table", "door", "ceiling_light"],
  "user_query": "show me the table lamp beside the armchair",
  "target_output": {
    "format_version": "hypothesis_output_v1",
    "parse_mode": "multi",
    "hypotheses": [
      {
        "kind": "direct",
        "rank": 1,
        "grounding_query": {
          "raw_query": "show me the table lamp beside the armchair",
          "root": {
            "categories": ["UNKNOW"],
            "attributes": [],
            "spatial_constraints": [{"relation": "beside", "anchors": [{"categories": ["armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": true
        },
        "lexical_hints": ["table lamp"]
      },
      {
        "kind": "proxy",
        "rank": 2,
        "grounding_query": {
          "raw_query": "proxy lighting near armchair",
          "root": {
            "categories": ["ceiling_light"],
            "attributes": [],
            "spatial_constraints": [{"relation": "near", "anchors": [{"categories": ["armchair"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": false
        },
        "lexical_hints": ["lighting"]
      },
      {
        "kind": "context",
        "rank": 3,
        "grounding_query": {
          "raw_query": "armchair zone",
          "root": {
            "categories": ["armchair"],
            "attributes": [],
            "spatial_constraints": [],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": false
        },
        "lexical_hints": ["reading corner"]
      }
    ]
  }
}
```

运行期行为约束：
1. `proxy` 不命中时允许回退到 `context`。
2. 最终状态可为 `context_only`，但不能直接 `no_evidence`。

### H03
用户查询：
```text
find the footstool between the sofa and the side table
```

`parser_sft.jsonl` 样本（单行）：
```json
{
  "sample_id": "room0_hard_H03",
  "bucket": "hard",
  "scene_id": "room0",
  "mask_spec": {"type": "M1+M2", "hidden_categories": ["ottoman"]},
  "scene_categories_masked": ["sofa", "side_table", "armchair", "door", "area_rug", "stool"],
  "user_query": "find the footstool between the sofa and the side table",
  "target_output": {
    "format_version": "hypothesis_output_v1",
    "parse_mode": "multi",
    "hypotheses": [
      {
        "kind": "direct",
        "rank": 1,
        "grounding_query": {
          "raw_query": "find the footstool between the sofa and the side table",
          "root": {
            "categories": ["UNKNOW"],
            "attributes": [],
            "spatial_constraints": [{"relation": "between", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}, {"categories": ["side_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": true
        },
        "lexical_hints": ["footstool"]
      },
      {
        "kind": "proxy",
        "rank": 2,
        "grounding_query": {
          "raw_query": "proxy stool-like near sofa and side table",
          "root": {
            "categories": ["stool", "area_rug"],
            "attributes": [],
            "spatial_constraints": [{"relation": "near", "anchors": [{"categories": ["sofa"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}, {"relation": "near", "anchors": [{"categories": ["side_table"], "attributes": [], "spatial_constraints": [], "select_constraint": null, "node_id": ""}]}],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": false
        },
        "lexical_hints": ["stool"]
      },
      {
        "kind": "context",
        "rank": 3,
        "grounding_query": {
          "raw_query": "between sofa and side table area",
          "root": {
            "categories": ["sofa", "side_table"],
            "attributes": [],
            "spatial_constraints": [],
            "select_constraint": null,
            "node_id": ""
          },
          "expect_unique": false
        },
        "lexical_hints": ["in-between area"]
      }
    ]
  }
}
```

运行期行为约束：
1. 强制覆盖 `between` 关系在 hard 桶下的降级路径。
2. direct 失败时仍需返回可解释的 `proxy/context` 支撑证据。

## 6. 快速使用说明（如何在实现中消费这些 case）
1. 训练 Qwen 仅使用 `parser_sft` 样本字段（`user_query` + `target_output`）。
2. 在线推理时，Qwen 输出必须符合 `HypothesisOutputV1`，由 KeyframeSelector 按 `rank` 依次执行。
3. 执行日志只记录运行行为（first-hit hypothesis、final status、top-k frame ids），不维护离线指标汇总。
4. 若 hard case 频繁出现 `no_evidence`，将样本回流到数据修订队列并重生 query/program。
