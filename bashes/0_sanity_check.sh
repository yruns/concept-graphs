#!/bin/bash

################################################################################
# 步骤 0: 3D 重建健全性检查 (可选)
################################################################################
#
# 作用：
#   - 使用 GradSLAM 进行常规 RGB 3D 重建
#   - 可视化重建结果，验证输入数据质量
#   - 确保相机位姿和 RGB-D 数据正常
#
# 输入：
#   - RGB-D 图像: $REPLICA_ROOT/$SCENE_NAME/results/*.jpg, *.png
#   - 相机位姿: $REPLICA_ROOT/$SCENE_NAME/traj.txt
#
# 输出：
#   - 实时可视化窗口 (需要 GUI)
#   - 不保存文件
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate scgh
export PYTHONPATH="/home/shyue/codebase/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# 进入工作目录
cd /home/shyue/codebase/concept-graphs/conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=room0

echo "================================================"
echo "步骤 0: 3D 重建健全性检查"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "数据路径: ${REPLICA_ROOT}/${SCENE_NAME}"
echo ""
echo "这将打开一个可视化窗口，按 ESC 关闭"
echo "================================================"
echo ""

# 运行 3D RGB 重建
python scripts/run_slam_rgb.py \
    --dataset_root $REPLICA_ROOT \
    --dataset_config $REPLICA_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --image_height 480 \
    --image_width 640 \
    --stride 5 \
    --visualize

echo ""
echo "✓ 3D 重建检查完成"
echo ""

