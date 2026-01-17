#!/bin/bash

################################################################################
# 步骤 7: 可视化场景图（离线版本，适用于无显示器的服务器）
################################################################################
#
# 作用：
#   - 在无显示器环境下将场景图可视化为文件
#   - 支持多种输出格式：多视角图像、3D模型(PLY)、交互式HTML
#   - 不需要 X11 或显示服务器
#
# 输入：
#   - 场景图地图: $REPLICA_ROOT/$SCENE_NAME/sg_cache/map/scene_map_cfslam_pruned.pkl.gz
#   - 物体关系: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_object_relations.json
#
# 输出：
#   - images/: 多个视角的 PNG 图像
#   - ply/: 可用其他软件打开的 3D 模型文件
#   - html/: 可在浏览器中交互查看的 HTML 文件
#
# 提示：
#   - 修改 OUTPUT_FORMAT 来选择输出格式
#   - 修改 NUM_VIEWS 来控制生成的图像数量
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

# 输出设置
OUTPUT_DIR="${REPLICA_ROOT}/${SCENE_NAME}/visualization"
OUTPUT_FORMAT="all"  # 可选: images, ply, html, all
NUM_VIEWS=8          # 生成多少个视角的图像
IMAGE_WIDTH=1920     # 图像宽度
IMAGE_HEIGHT=1080    # 图像高度

# 原始mesh文件路径（Replica提供的ground truth场景）
ORIGINAL_MESH="${REPLICA_ROOT}/${SCENE_NAME}_mesh.ply"

echo "================================================"
echo "步骤 7: 可视化场景图（离线版本）"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo ""
echo "输入文件:"
echo "  - ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz"
echo "  - ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json"
echo ""
echo "输出设置:"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - 输出格式: ${OUTPUT_FORMAT}"
echo "  - 视角数量: ${NUM_VIEWS}"
echo "  - 图像尺寸: ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
echo ""

# 检查原始mesh文件
if [ -f "${ORIGINAL_MESH}" ]; then
    echo "原始场景mesh: ${ORIGINAL_MESH}"
    echo "  ✓ 将在可视化中叠加显示原始场景"
else
    echo "原始场景mesh: 未找到 (${ORIGINAL_MESH})"
    echo "  ⚠ 将只显示重建的场景图"
fi
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

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行离线可视化
echo "正在生成可视化..."
echo ""

# 构建命令
CMD="python scripts/visualize_cfslam_results_offscreen.py \
    --result_path \"${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz\" \
    --edge_file \"${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --output_format \"${OUTPUT_FORMAT}\" \
    --num_views \"${NUM_VIEWS}\" \
    --image_width \"${IMAGE_WIDTH}\" \
    --image_height \"${IMAGE_HEIGHT}\""

# 如果原始mesh存在，添加参数
if [ -f "${ORIGINAL_MESH}" ]; then
    CMD="${CMD} --original_mesh \"${ORIGINAL_MESH}\""
fi

# 执行命令
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 可视化完成"
    echo ""
    echo "================================================"
    echo "输出文件位置: ${OUTPUT_DIR}"
    echo "================================================"
    
    # 列出生成的文件
    if [ -d "${OUTPUT_DIR}/images" ]; then
        echo ""
        echo "📸 多视角图像:"
        ls -lh "${OUTPUT_DIR}/images/"*.png 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    fi
    
    if [ -d "${OUTPUT_DIR}/ply" ]; then
        echo ""
        echo "🎨 3D 模型文件 (可用 MeshLab/CloudCompare/Blender 打开):"
        ls -lh "${OUTPUT_DIR}/ply/"*.ply 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    fi
    
    if [ -d "${OUTPUT_DIR}/html" ]; then
        echo ""
        echo "🌐 HTML 可视化 (在浏览器中打开):"
        ls -lh "${OUTPUT_DIR}/html/"*.html 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    fi
    
    echo ""
    echo "================================================"
    echo "使用提示:"
    echo "================================================"
    echo "1. 查看图像: 将 ${OUTPUT_DIR}/images/ 下的 PNG 文件下载到本地查看"
    echo "2. 3D 模型: 将 PLY 文件下载后用 MeshLab、CloudCompare 或 Blender 打开"
    echo "3. HTML: 将 HTML 文件下载后在浏览器中打开（支持 3D 交互旋转）"
    echo ""
    echo "如需下载文件到本地，可使用 scp 命令:"
    echo "  scp -r user@server:${OUTPUT_DIR} ."
    echo ""
else
    echo ""
    echo "✗ 可视化失败"
    exit 1
fi

