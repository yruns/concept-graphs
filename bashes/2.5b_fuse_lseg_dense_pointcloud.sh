#!/bin/bash

################################################################################
# 步骤 2.5b: 生成全场景稠密点云 + LSeg 特征
################################################################################
#
# 作用：
#   - 从深度图生成整个场景的稠密点云（不限于物体区域）
#   - 使用 LSeg 模型为每个点提取 512 维 CLIP 对齐特征
#   - 多视角融合 + 体素下采样
#
# 输入：
#   - RGB 图像: $REPLICA_ROOT/$SCENE_NAME/results/frame*.jpg
#   - 深度图像: $REPLICA_ROOT/$SCENE_NAME/results/depth*.png
#   - 相机位姿: $REPLICA_ROOT/$SCENE_NAME/traj.txt
#
# 输出：
#   - 稠密点云: $REPLICA_ROOT/$SCENE_NAME/dense_pcd_lseg.npz
#   - 包含: points (N,3), lseg_features (N,512), colors (N,3)
#
# 注意：
#   - 必须在 lseg conda 环境中运行
#   - 需要 GPU
#
################################################################################

# 配置
LSEG_PROJECT_DIR="/home/shyue/codebase/lseg_feature_extraction"
CONCEPT_GRAPH_DIR="/home/shyue/codebase/concept-graphs"
STRIDE=8
VOXEL_SIZE=0.025  # 2.5cm 体素大小 (与 ConceptGraphs 一致)
LSEG_IMG_SIZE=640  # LSeg 输入图像长边 (越小越快，默认 480)

# 激活 lseg 环境
source /home/shyue/anaconda3/bin/activate lseg

# 加载 ConceptGraphs 环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=${1:-room0}
OUTPUT_FILE=${2:-"dense_pcd_lseg.npz"}

echo "================================================"
echo "步骤 2.5b: 生成全场景稠密点云 + LSeg 特征"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "Stride: ${STRIDE}"
echo "体素大小: ${VOXEL_SIZE}m"
echo "LSeg 图像尺寸: ${LSEG_IMG_SIZE}px"
echo ""
echo "输入数据:"
echo "  - RGB: ${REPLICA_ROOT}/${SCENE_NAME}/results/frame*.jpg"
echo "  - Depth: ${REPLICA_ROOT}/${SCENE_NAME}/results/depth*.png"
echo "  - Poses: ${REPLICA_ROOT}/${SCENE_NAME}/traj.txt"
echo ""
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/${OUTPUT_FILE}"
echo "================================================"
echo ""

# 进入 lseg 项目目录
cd "${LSEG_PROJECT_DIR}"

# 运行稠密点云生成
python "${CONCEPT_GRAPH_DIR}/conceptgraph/slam/fuse_lseg_dense_pointcloud.py" \
    --scene_path "${REPLICA_ROOT}/${SCENE_NAME}" \
    --output_file "${OUTPUT_FILE}" \
    --stride ${STRIDE} \
    --voxel_size ${VOXEL_SIZE} \
    --lseg_img_size ${LSEG_IMG_SIZE}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 全场景稠密点云生成完成"
    echo ""
    
    OUTPUT_PATH="${REPLICA_ROOT}/${SCENE_NAME}/${OUTPUT_FILE}"
    if [ -f "${OUTPUT_PATH}" ]; then
        FILE_SIZE=$(du -h "${OUTPUT_PATH}" | cut -f1)
        echo "输出文件: ${OUTPUT_PATH}"
        echo "文件大小: ${FILE_SIZE}"
        
        # 显示点云信息
        python3 -c "
import numpy as np
data = np.load('${OUTPUT_PATH}', allow_pickle=True)
print(f'点数: {len(data[\"points\"]):,}')
print(f'点云形状: {data[\"points\"].shape}')
print(f'LSeg 特征形状: {data[\"lseg_features\"].shape}')
print(f'颜色形状: {data[\"colors\"].shape}')
"
    fi
    echo ""
else
    echo ""
    echo "✗ 稠密点云生成失败"
    exit 1
fi
