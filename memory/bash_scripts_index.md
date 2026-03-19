# Bash Scripts Index

## Query Scene 运行提示
- Darwin 上推荐直接用 `.venv/bin/python -m ...` 运行 query scene 模块；若直接运行 bash 包装脚本，要注意它们多数仍假设 conda 环境。
- `6b`、`7b`、`run_e2e_query_test` 这三个 query scene 相关脚本当前都不会自动 source `env_vars.bash`。
- `7b_query_scene.sh` 的默认 `REPLICA_ROOT` 与 `6b`/`run_full_detect_pipeline_to_6b.sh` 不一致，运行前最好显式 `export REPLICA_ROOT=/abs/path/to/Replica`。

## Pipeline 脚本职责
- `bashes/1_extract_2d_segmentation.sh`: 类别无关 2D 分割 + CLIP 特征提取（SAM）。
- `bashes/1b_extract_2d_segmentation_detect.sh`: 类别感知分割（RAM + GroundingDINO + SAM）。
- `bashes/2_build_3d_object_map.sh`: 类别无关结果融合为 3D 对象地图。
- `bashes/2b_build_3d_object_map_detect.sh`: 类别感知结果融合为 3D 对象地图（产物路径字段写为相对 `REPLICA_ROOT`，不写绝对路径）。
- `bashes/3_visualize_object_map.sh`: 交互式可视化 3D 对象地图。
- `bashes/4_extract_object_captions.sh`: 生成对象初始描述（`sg_cache`）。
- `bashes/4b_extract_object_captions_detect.sh`: 类别感知版本对象描述（`sg_cache_detect`，默认 `gemini-2.5-pro + GeminiClientPool`，`NUM_WORKERS` 可配置并发）。
- `bashes/5_refine_object_captions.sh`: 细化对象描述（`sg_cache`）。
- `bashes/5b_refine_object_captions_detect.sh`: 细化对象描述（`sg_cache_detect`）。
- `bashes/5b_refine_with_affordance.sh`: 细化描述并提取 affordance（图像+文本）。
- `bashes/6_build_scene_graph.sh`: 场景图构建与关系输出。
- `bashes/6b_build_visibility_index.sh`: 预计算双向可见性索引（query 加速；metadata 中场景/pcd 路径为相对 `REPLICA_ROOT`）。当前脚本默认 `REPLICA_ROOT=$HOME/Datasets/Replica`，并直接 `conda activate conceptgraph`。
- `bashes/7_visualize_scene_graph.sh`: 交互式场景图可视化。
- `bashes/7_visualize_scene_graph_offscreen.sh`: 离线导出场景图可视化（images/ply/html）。
- `bashes/7b_query_scene.sh`: 文本查询驱动关键帧选择主入口，底层调用 `conceptgraph.query_scene.examples.query_keyframes`。当前脚本默认 `REPLICA_ROOT=$HOME/Datasets/Replica/Replica`，与 `6b`/`run_full_detect_pipeline_to_6b.sh` 不一致，且不 source `env_vars.bash`。
- `bashes/run_e2e_query_test.sh`: query scene 端到端测试并输出日志到 `docs/e2e_query_test_run.log`。当前脚本直接 `conda activate conceptgraph`，再以文件路径运行 `conceptgraph/query_scene/examples/e2e_query_test.py`。
- `bashes/run_full_detect_pipeline_to_6b.sh`: 针对单个 scene 的 detect 全链路总控（清理旧产物，保留 `results/result + traj.txt`，串行执行 `1b→2b→4b→5b+→6b` 并做产物校验）。
