#!/bin/bash

################################################################################
# 步骤 1C: 提取 LSeg 像素级特征
################################################################################
#
# 作用：
#   - 使用 LSeg 模型提取每帧图像的像素级 CLIP 对齐特征
#   - 为步骤 2b 中的点云提供点级语义特征
#   - stride 与 ConceptGraphs 保持一致
#
# 输入：
#   - RGB 图像: $REPLICA_ROOT/$SCENE_NAME/results/frame*.jpg
#
# 输出：
#   - LSeg 特征: $REPLICA_ROOT/$SCENE_NAME/lseg_features/*.pt
#   - 每个文件形状: (H, W, 512), dtype: float16
#
# 注意：
#   - 必须在 lseg conda 环境中运行
#   - LSeg 模型需要 GPU
#
################################################################################

# 配置
LSEG_PROJECT_DIR="/home/shyue/codebase/lseg_feature_extraction"
LSEG_MODEL="checkpoints/demo_e200.ckpt"
STRIDE=5
SAVE_FORMAT="pt"  # pt or npy

# 激活 lseg 环境
source /home/shyue/anaconda3/bin/activate lseg

# 加载 ConceptGraphs 环境变量获取数据集路径
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=${1:-room0}

echo "================================================"
echo "步骤 1C: 提取 LSeg 像素级特征"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "Stride: ${STRIDE}"
echo "LSeg 模型: ${LSEG_MODEL}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/results/"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/lseg_features/"
echo "================================================"
echo ""

# 进入 LSeg 项目目录
cd "${LSEG_PROJECT_DIR}"

# 运行特征提取
python extract_replica_features.py \
    --scene_path "${REPLICA_ROOT}/${SCENE_NAME}" \
    --output_dir "${REPLICA_ROOT}/${SCENE_NAME}/lseg_features" \
    --stride ${STRIDE} \
    --lseg_model "${LSEG_MODEL}" \
    --save_format ${SAVE_FORMAT}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ LSeg 特征提取完成"
    echo ""
    echo "输出目录: ${REPLICA_ROOT}/${SCENE_NAME}/lseg_features/"
    
    # 统计输出文件
    NUM_FILES=$(ls -1 "${REPLICA_ROOT}/${SCENE_NAME}/lseg_features/"*.${SAVE_FORMAT} 2>/dev/null | wc -l)
    echo "生成文件数: ${NUM_FILES}"
    
    # 显示存储大小
    TOTAL_SIZE=$(du -sh "${REPLICA_ROOT}/${SCENE_NAME}/lseg_features/" 2>/dev/null | cut -f1)
    echo "总存储大小: ${TOTAL_SIZE}"
    echo ""
else
    echo ""
    echo "✗ LSeg 特征提取失败"
    echo ""
    echo "请确保:"
    echo "  1. 已激活 lseg conda 环境"
    echo "  2. LSeg 模型文件存在: ${LSEG_PROJECT_DIR}/${LSEG_MODEL}"
    echo "  3. GPU 可用"
    exit 1
fi
