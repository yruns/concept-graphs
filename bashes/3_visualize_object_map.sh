#!/bin/bash

################################################################################
# 步骤 3: 可视化 3D 对象地图 (可选)
################################################################################
#
# 作用：
#   - 交互式可视化 3D 对象地图
#   - 验证映射质量
#   - 检查物体分割和点云融合效果
#
# 输入：
#   - 3D 对象地图: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz
#
# 输出：
#   - 交互式 Open3D 可视化窗口
#
# 快捷键：
#   b - 切换背景点云显示
#   c - 按类别着色
#   r - 按 RGB 着色
#   f - 按 CLIP 文本相似度着色 (输入查询)
#   i - 按实例 ID 着色
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
PKL_FILENAME=full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz

echo "================================================"
echo "步骤 3: 可视化 3D 对象地图"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "文件: ${PKL_FILENAME}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"
echo ""
echo "可视化快捷键:"
echo "  [b] - 切换背景点云"
echo "  [c] - 按类别着色"
echo "  [r] - 按 RGB 着色"
echo "  [f] - 按文本相似度着色"
echo "  [i] - 按实例 ID 着色"
echo "================================================"
echo ""

# 检查文件是否存在
if [ ! -f "${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}" ]; then
    echo "✗ 错误: 文件不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"
    echo ""
    echo "请先运行步骤 2 (2_build_3d_object_map.sh)"
    exit 1
fi

# 可视化对象地图
python scripts/visualize_cfslam_results.py \
    --result_path ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

echo ""
echo "✓ 可视化完成"
echo ""

