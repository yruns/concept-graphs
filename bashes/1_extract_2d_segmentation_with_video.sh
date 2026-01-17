#!/bin/bash

################################################################################
# 步骤 1 (增强版): 提取 2D 分割、CLIP 特征并生成视频
################################################################################
#
# 作用：
#   - 使用 SAM (Segment Anything Model) 进行 2D 图像分割
#   - 提取每个分割区域的 CLIP 视觉特征
#   - 类别无关模式（不需要 RAM 模型）
#   - ✨ 生成 2D 分割结果的视频动画
#
# 输入：
#   - RGB 图像: $REPLICA_ROOT/$SCENE_NAME/results/color/*.jpg
#   - 深度图像: $REPLICA_ROOT/$SCENE_NAME/results/depth/*.png
#
# 输出：
#   - 分割结果: $REPLICA_ROOT/$SCENE_NAME/gsa_results_none/*.pkl.gz
#   - 可视化图像: $REPLICA_ROOT/$SCENE_NAME/gsa_vis_none/*.jpg
#   - ✨ 视频: $REPLICA_ROOT/$SCENE_NAME/gsa_vis_none.mp4
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
echo "步骤 1 (增强版): 提取 2D 分割 + 生成视频"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "模式: ConceptGraphs (类别无关)"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/results/"
echo "输出:"
echo "  - 分割数据: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_results_none/"
echo "  - 可视化: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/"
echo "  - ✨ 视频: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.mp4"
echo "================================================"
echo ""

# 运行分割提取 (ConceptGraphs - 类别无关模式) + 保存视频
python scripts/generate_gsa_results.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set none \
    --stride 5 \
    --save_video

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 2D 分割、特征提取和视频生成完成"
    echo ""
    echo "结果保存在:"
    echo "  - 分割数据: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_results_none/"
    echo "  - 可视化图像: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/"
    echo "  - ✨ 视频文件: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.mp4"
    echo ""
    echo "播放视频命令:"
    echo "  vlc ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.mp4"
    echo "  # 或"
    echo "  ffplay ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.mp4"
    echo ""
else
    echo ""
    echo "✗ 2D 分割提取失败"
    exit 1
fi

