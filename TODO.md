# Keyframe Selector Improvement Tasks

## Goal
通过改进 KeyframeSelector 和 GroundingQuery 结构，提升自然语言文本检索场景关键帧的准确率。
这是后续所有内容的基石，必须优先解决。

## Current Baseline (2026-03-07)
- **Test Suite**: 59 queries across 2 dimensions (Presence x Complexity)
- **Pass Rate**: 57/59 (96.6%) - **Actual Semantic Accuracy: 98.3% (58/59)**
- **Analysis**: See `results/analysis_v1.md` for detailed breakdown
- **Key Findings**:
  - EDGE-01, EDGE-03: TRUE NEGATIVES (correctly return 0 for impossible relations)
  - EDGE-02: FALSE POSITIVE (sofa "inside" ottoman returns 1 due to bbox fallback)
  - P3-C1-01/02/03: Actually PASS via proxy anchor fallback

## Issues to Address (Updated)
1. **CRITICAL - EDGE-02 False Positive**: `is_inside` falls back to `is_near` when bbox missing
2. **Multi-target Query Coverage**: "all X on Y" returns 1 instead of all matching
3. **Test Framework**: Need explicit "should fail" assertions for edge cases
4. **Proxy Anchor Selection**: Quality varies, needs evaluation

## Pending
- [ ] 1b. Fix EDGE-02: Remove `is_near` fallback from `is_inside` relation
- [ ] 2. 检查 spatial_relations.py 中 ON/NEAR 阈值，评估是否需要调整
- [ ] 3. 分析 proxy anchor 选择策略，提出改进方案
- [ ] 4. 测试更多真实场景 query 变体，扩展测试集
- [ ] 5. 实现改进方案 A：调整空间关系阈值
- [ ] 6. 实现改进方案 B：优化 proxy anchor 选择
- [ ] 7. 运行 e2e 测试，对比改进效果
- [ ] 8. 如果通过率 < 95%，继续迭代改进
- [ ] 9. 通过率达标后，整理最终报告

## Completed
- [x] 1. 分析当前失败和边界 case，记录详细原因到 results/analysis_v1.md
  - 发现 EDGE-01/03 是 TRUE NEGATIVES，EDGE-02 是需要修复的 FALSE POSITIVE
  - 实际语义准确率 98.3%，测试框架需要更新以区分 "should pass" vs "should fail" cases

## Results Log
| Iteration | Date | Pass Rate | Changes | Notes |
|-----------|------|-----------|---------|-------|
| 0 (baseline) | 2026-03-07 | 57/59 (96.6%) | Initial hypothesis mechanism | P3-C2 now uses proxy anchors |
| 1 (analysis) | 2026-03-07 | 57/59 (96.6%) | Analysis complete | Actual semantic: 98.3% (58/59), EDGE-02 is only real bug |
