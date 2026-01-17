#!/bin/bash

################################################################################
# 步骤 1: 提取 2D 分割和 CLIP 特征
################################################################################
#
# 作用：
#   - 使用 SAM (Segment Anything Model) 进行 2D 图像分割
#   - 提取每个分割区域的 CLIP 视觉特征
#   - 类别无关模式（不需要 RAM 模型）
#
# 输入：
#   - RGB 图像: $REPLICA_ROOT/$SCENE_NAME/results/color/*.jpg
#   - 深度图像: $REPLICA_ROOT/$SCENE_NAME/results/depth/*.png
#
# 输出：
#   - 分割结果: $REPLICA_ROOT/$SCENE_NAME/gsa_results_none/*.pkl.gz
#   - 可视化: $REPLICA_ROOT/$SCENE_NAME/gsa_vis_none/*.jpg
#   - CLIP 特征: 保存在 .pkl.gz 文件中
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

echo "================================================"
echo "步骤 1: 提取 2D 分割和 CLIP 特征"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "模式: ConceptGraphs (类别无关)"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/results/"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_results_none/"
echo "      ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/"
echo "================================================"
echo ""

# 运行分割提取 (ConceptGraphs - 类别无关模式)
python scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set none \
    --stride 5

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 2D 分割和特征提取完成"
    echo ""
    echo "结果保存在:"
    echo "  - 分割数据: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_results_none/"
    echo "  - 可视化: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/"
    echo ""
else
    echo ""
    echo "✗ 2D 分割提取失败"
    exit 1
fi

