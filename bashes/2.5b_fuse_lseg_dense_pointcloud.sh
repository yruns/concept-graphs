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
STRIDE=8
VOXEL_SIZE=0.025  # 2.5cm 体素大小 (与 ConceptGraphs 一致)
LSEG_IMG_SIZE=640  # LSeg 输入图像长边 (越小越快，默认 480)
MAX_DEPTH=10.0
DEPTH_SCALE=6553.5
RESIZE_LONG_SIDE=640

# 激活 lseg 环境
source /home/shyue/anaconda3/bin/activate lseg

# 加载 ConceptGraphs 环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=${1:-room0}
OUTPUT_DIR=${2:-""}
OUTPUT_PATH=""
if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_PATH="${OUTPUT_DIR}/dense_pcd_lseg.npz"
else
  OUTPUT_PATH="${REPLICA_ROOT}/${SCENE_NAME}/dense_pcd_lseg.npz"
fi

echo "================================================"
echo "步骤 2.5b: 生成全场景稠密点云 + LSeg 特征"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "Stride: ${STRIDE}"
echo "体素大小: ${VOXEL_SIZE}m"
echo "最大深度: ${MAX_DEPTH}m"
echo "LSeg 图像尺寸: ${LSEG_IMG_SIZE}px"
echo ""
echo "输入数据:"
echo "  - RGB: ${REPLICA_ROOT}/${SCENE_NAME}/results/frame*.jpg"
echo "  - Depth: ${REPLICA_ROOT}/${SCENE_NAME}/results/depth*.png"
echo "  - Poses: ${REPLICA_ROOT}/${SCENE_NAME}/traj.txt"
echo ""
echo "输出: ${OUTPUT_PATH}"
echo "================================================"
echo ""

# 运行 LSeg 融合（切到 LSeg 目录以确保 data/ 路径正确）
pushd "${CG_FOLDER}/lseg_feature_extraction" >/dev/null
python fusion_replica.py \
  --scene_path "${REPLICA_ROOT}/${SCENE_NAME}" \
  --output_path "${OUTPUT_PATH}" \
  --lseg_model "checkpoints/demo_e200.ckpt" \
  --stride "${STRIDE}" \
  --voxel_size "${VOXEL_SIZE}" \
  --max_depth "${MAX_DEPTH}" \
  --depth_scale "${DEPTH_SCALE}" \
  --lseg_img_long_side "${LSEG_IMG_SIZE}" \
  --resize_long_side "${RESIZE_LONG_SIDE}"
popd >/dev/null

