#!/bin/bash

################################################################################
# 步骤 7: 可视化场景图
################################################################################
#
# 作用：
#   - 交互式可视化带有场景图的 3D 地图
#   - 显示物体节点和关系边
#   - 支持多种查看模式
#
# 输入：
#   - 场景图地图: $REPLICA_ROOT/$SCENE_NAME/sg_cache/map/scene_map_cfslam_pruned.pkl.gz
#   - 物体关系: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_object_relations.json
#
# 输出：
#   - 交互式 Open3D 可视化窗口
#
# 快捷键：
#   g - 显示/隐藏场景图
#   b - 切换背景点云
#   c - 按类别着色
#   r - 按 RGB 着色
#   f - 按 CLIP 文本相似度着色
#   i - 按实例 ID 着色
#   + - 增大点云点大小
#   - - 减小点云点大小
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
echo "步骤 7: 可视化场景图"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo ""
echo "输入文件:"
echo "  - ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz"
echo "  - ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json"
echo ""
echo "可视化快捷键:"
echo "  [g]   - 显示/隐藏场景图 ⭐"
echo "  [b]   - 切换背景点云"
echo "  [c]   - 按类别着色"
echo "  [r]   - 按 RGB 着色"
echo "  [f]   - 按文本相似度着色"
echo "  [i]   - 按实例 ID 着色"
echo "  [+/-] - 调整点云大小"
echo "================================================"
echo ""

# 检查输入文件
if [ ! -f "${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz" ]; then
    echo "✗ 错误: 场景图地图文件不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz"
    echo ""
    echo "请先运行步骤 6 (6_build_scene_graph.sh)"
    exit 1
fi

if [ ! -f "${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json" ]; then
    echo "✗ 错误: 物体关系文件不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json"
    echo ""
    echo "请先运行步骤 6 (6_build_scene_graph.sh)"
    exit 1
fi

# 可视化场景图
python scripts/visualize_cfslam_results.py \
    --result_path ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz \
    --edge_file ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json

echo ""
echo "✓ 可视化完成"
echo ""

