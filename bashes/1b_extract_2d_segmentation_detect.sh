#!/bin/bash

################################################################################
# 步骤 1B: 提取 2D 分割和 CLIP 特征 (ConceptGraphs-Detect 模式)
################################################################################
#
# 作用：
#   - 使用标记模型 (RAM) 和检测模型 (Grounding-DINO) 先检测物体
#   - 使用 SAM 对检测到的物体进行精确分割
#   - 提取每个分割区域的 CLIP 视觉特征
#   - 类别感知模式，包含背景类别
#
# 输入：
#   - RGB 图像: $REPLICA_ROOT/$SCENE_NAME/results/color/*.jpg
#   - 深度图像: $REPLICA_ROOT/$SCENE_NAME/results/depth/*.png
#
# 输出：
#   - 分割结果: $REPLICA_ROOT/$SCENE_NAME/gsa_results_ram_withbg_allclasses/*.pkl.gz
#   - 可视化: $REPLICA_ROOT/$SCENE_NAME/gsa_vis_ram_withbg_allclasses/*.jpg
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
CLASS_SET=ram

echo "================================================"
echo "步骤 1B: 提取 2D 分割和 CLIP 特征 (Detect 模式)"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "模式: ConceptGraphs-Detect (类别感知)"
echo "检测模型: RAM + Grounding-DINO"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/results/"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_results_ram_withbg_allclasses/"
echo "      ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_ram_withbg_allclasses/"
echo "================================================"
echo ""

# 运行分割提取 (ConceptGraphs-Detect - 类别感知模式)
python scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride 5 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 2D 分割和特征提取完成 (Detect 模式)"
    echo ""
    echo "结果保存在:"
    echo "  - 分割数据: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_results_ram_withbg_allclasses/"
    echo "  - 可视化: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_ram_withbg_allclasses/"
    echo ""
else
    echo ""
    echo "✗ 2D 分割提取失败 (Detect 模式)"
    exit 1
fi

