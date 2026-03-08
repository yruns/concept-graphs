# room0 Artifact Lineage

本文梳理 `room0/` 中主要文件的来源步骤（哪一步生成）和核心内容（里面装了什么）。

## 0) 原始输入（数据集自带，不由本仓库生成）
- `room0/results/frame*.jpg`（当前 2000 帧）  
  来源：Replica 场景原始 RGB 序列。
- `room0/traj.txt`  
  来源：Replica 相机轨迹（4x4 位姿矩阵序列）。

## 1) Step 1B: 2D 检测与分割
- 脚本：`bash bashes/1b_extract_2d_segmentation_detect.sh`
- 代码入口：`conceptgraph/scripts/generate_gsa_results.py`
- 产物：
  - `room0/gsa_detections_ram_withbg_allclasses/*.pkl.gz`（当前 400 个，stride=5）
  - `room0/gsa_vis_ram_withbg_allclasses/*.jpg`
  - `room0/gsa_classes_ram_withbg_allclasses.json`
- 主要内容：
  - 每帧 `.pkl.gz` 含 `xyxy/confidence/class_id/mask`、CLIP 图像与文本特征、`frame_clip_feat`。
  - 注：脚本注释里写过 `gsa_results_*`，实际代码输出目录是 `gsa_detections_*`。

## 2) Step 2B: 3D 对象融合建图
- 脚本：`bash bashes/2b_build_3d_object_map_detect.sh room0`
- 代码入口：`conceptgraph/slam/cfslam_pipeline_batch.py`
- 产物：
  - `room0/pcd_saves/full_pcd_ram_withbg_allclasses_... .pkl.gz`
  - `room0/pcd_saves/full_pcd_ram_withbg_allclasses_... _post.pkl.gz`
  - `room0/gsa_classes_ram_withbg_allclasses_colors.json`
  - `room0/pcd_saves/ply_export_detect/all_objects_colored.ply`
  - `room0/pcd_saves/ply_export_detect/objects_info.json`
- 主要内容：
  - `full_pcd*_post.pkl.gz` 内含 `objects/bg_objects/cfg/class_names/class_colors`。
  - `objects[i]` 包含 `pcd_np/bbox_np/image_idx/mask/xyxy/class_name` 等多视角融合属性。
  - 路径字段（如 `objects[*].color_path`、`cfg` 内路径）为相对 `REPLICA_ROOT` 的相对路径，不再写绝对路径。

## 3) Step 4B: 物体多视角 caption
- 脚本：`bash bashes/4b_extract_object_captions_detect.sh room0`
- 代码入口：`scenegraph/build_scenegraph_cfslam.py --mode extract-node-captions`
- 产物：
  - `room0/sg_cache_detect/cfslam_llava_captions.json`
  - `room0/sg_cache_detect/cfslam_feat_llava/*.pt`
  - `room0/sg_cache_detect/cfslam_captions_llava_debug/*.png`
- 主要内容：
  - 每个物体多视角文字描述（captions）+ 视觉特征缓存 + 调试截图。

## 4) Step 5B+: affordance 精炼
- 脚本：`bash bashes/5b_refine_with_affordance.sh room0`
- 代码入口：`conceptgraph/query_scene/refine_with_affordance.py`
- 产物：`room0/sg_cache_detect/object_affordances.json`
- 主要内容：
  - 每个物体的 `object_tag/summary/category/affordances/co_objects`。
  - Query scene 在加载对象后会优先合并该文件增强语义。

## 5) Step 6B: 可见性索引
- 脚本：`bash bashes/6b_build_visibility_index.sh room0`
- 代码入口：`conceptgraph/scripts/build_visibility_index.py`
- 产物：`room0/indices/visibility_index.pkl`
- 主要内容：
  - `object_to_views` 与 `view_to_objects` 双向索引 + `metadata`（stride、对象数、映射数等）。
  - `metadata.scene_path / metadata.pcd_file` 为相对 `REPLICA_ROOT` 的相对路径。

## 6) Query / 可视化实验输出
- `room0/vis_index_debug/`  
  来源：`python -m conceptgraph.query_scene.examples.visualize_visibility_index`。  
  内容：对象-视角 TopK 可视化图、标注图、局部/全局 PLY。
- `room0/query_visualizations_base/`、`room0/query_visualizations_improved/`、`room0/query_visualizations_mutlicategory/`  
  来源：`conceptgraph/query_scene/examples/e2e_query_test.py` 的多轮实验结果归档（默认脚本输出名是 `query_visualizations/`，这里是对比实验的分组目录）。  
  内容：每个 query 子目录含 `00_initial_candidates.ply`、`01_final_candidates.ply`、`final_combined.ply`、`color_legend.txt`、`keyframes/*.jpg`；根目录含 `test_results.json`。
