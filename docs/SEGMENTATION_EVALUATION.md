# 时序场景分段评估

## 当前结果 (Replica room0)

### 基本统计
- 总帧数: 400 (stride=5)
- 分段数: **16 个区域**
- 平均每段: 25.0 帧
- 标准差: 8.77 帧

### 空间质量
- 区域间平均距离: **1.49m**
- 区域间最小距离: 0.27m
- 区域内紧凑度: 0.31m
- 分离度/紧凑度比: **4.8** (>2 为良好, >4 为优秀)

## 评估指标解释

| 指标 | 当前值 | 说明 |
|------|--------|------|
| n_segments | 16 | 检测到的区域数 |
| avg_inter_region_distance | 1.49m | 区域中心间距离 |
| avg_region_compactness | 0.31m | 区域内轨迹离散度 |
| avg_velocity_variance | 9.3e-05 | 区域内运动一致性 |

## 可视化文件

```
sg_cache/segmentation/
├── trajectory_3d.png       # 3D轨迹分段
├── trajectory_topdown.png  # 俯视图
├── segment_statistics.png  # 统计图表
└── segmentation_signals.png # 信号可视化
```

## 参数调优

分段过多: 增加 min_segment_frames, peak_prominence
分段过少: 减少 min_segment_frames, peak_prominence
