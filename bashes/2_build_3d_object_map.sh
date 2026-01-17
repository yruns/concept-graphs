#!/bin/bash

################################################################################
# 步骤 2: 构建 3D 对象地图
################################################################################
#
# 作用：
#   - 将 2D 分割结果投影到 3D 空间
#   - 跨帧关联和匹配同一物体
#   - 融合多视图点云构建 3D 物体表示
#   - 合并重复检测的物体
#   - 计算物体的边界框和空间关系
#
# 输入：
#   - 2D 分割结果: $REPLICA_ROOT/$SCENE_NAME/gsa_results_none/*.pkl.gz
#   - 相机位姿: $REPLICA_ROOT/$SCENE_NAME/traj.txt
#   - RGB-D 数据: $REPLICA_ROOT/$SCENE_NAME/results/
#
# 输出：
#   - 3D 对象地图: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz
#   - 中间结果: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph
export PYTHONPATH="/home/shyue/codebase/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# 进入工作目录
cd /home/shyue/codebase/concept-graphs/conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=room0
THRESHOLD=1.2

echo "================================================"
echo "步骤 2: 构建 3D 对象地图"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "相似度阈值: ${THRESHOLD}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_results_none/"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/"
echo "================================================"
echo ""

# 运行 3D 对象映射 (ConceptGraphs - 类别无关模式)
python slam/cfslam_pipeline_batch.py \
    dataset_root=$REPLICA_ROOT \
    dataset_config=$REPLICA_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.95 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=none \
    class_agnostic=True \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 3D 对象地图构建完成"
    echo ""
    echo "输出文件:"
    PKL_BASE="full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub"
    echo "  - 后处理版本 (推荐): ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_BASE}_post.pkl.gz"
    echo "  - 原始版本: ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_BASE}.pkl.gz"
    echo ""
    echo "下一步使用: ${PKL_BASE}_post.pkl.gz"
    echo ""
else
    echo ""
    echo "✗ 3D 对象地图构建失败"
    exit 1
fi

