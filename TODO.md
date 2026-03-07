# Keyframe Selector Improvement Tasks

## Goal
通过改进 KeyframeSelector 和 GroundingQuery 结构，提升自然语言文本检索场景关键帧的准确率。
这是后续所有内容的基石，必须优先解决。

## Current Baseline (2026-03-07)
- **Test Suite**: 59 queries across 2 dimensions (Presence x Complexity)
- **Pass Rate**: 57/59 (96.6%)
- **Failed Cases**:
  - EDGE-01: `"the throw_pillow on the floor_lamp"` -> 0 objects (expected: 0, correct)
  - EDGE-03: `"the ceiling_light near the area_rug"` -> 0 objects (expected: 0, correct)
  - P3-C1-01/02/03: Missing category -> 0 objects (expected behavior)

## Issues to Address
1. **Proxy Anchor Selection Quality**: 当前 proxy anchor 选择基于 co-objects，但质量不稳定
2. **Spatial Relation Precision**: ON/NEAR 等关系的阈值可能需要调优
3. **Multi-target Query Coverage**: "all X on Y" 类查询返回数量偏少
4. **Complex Query Parsing**: 多层嵌套 + superlative 组合的解析准确性

## Pending
- [ ] 1. 分析当前失败和边界 case，记录详细原因到 results/analysis_v1.md
- [ ] 2. 检查 spatial_relations.py 中 ON/NEAR 阈值，评估是否需要调整
- [ ] 3. 分析 proxy anchor 选择策略，提出改进方案
- [ ] 4. 测试更多真实场景 query 变体，扩展测试集
- [ ] 5. 实现改进方案 A：调整空间关系阈值
- [ ] 6. 实现改进方案 B：优化 proxy anchor 选择
- [ ] 7. 运行 e2e 测试，对比改进效果
- [ ] 8. 如果通过率 < 95%，继续迭代改进
- [ ] 9. 通过率达标后，整理最终报告

## Completed
<!-- 完成的任务会被移到这里 -->

## Results Log
| Iteration | Date | Pass Rate | Changes | Notes |
|-----------|------|-----------|---------|-------|
| 0 (baseline) | 2026-03-07 | 57/59 (96.6%) | Initial hypothesis mechanism | P3-C2 now uses proxy anchors |
