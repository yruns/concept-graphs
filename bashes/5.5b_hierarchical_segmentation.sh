#!/bin/bash
# =============================================================================
# Step 5.5b: 层次化功能区域划分（使用预提取的Affordances）
# =============================================================================
# 基于步骤5b+生成的object_affordances.json进行层次化场景图构建
#
# 前置步骤:
#   1b -> 2b -> 4b -> 5b+ (refine_with_affordance)
#
# 输入:
#   - 3D物体地图 (pcd_saves/*.pkl.gz)
#   - 物体Affordances (sg_cache_detect/object_affordances.json) [来自5b+]
#   - 相机轨迹 (traj.txt)
#
# 输出:
#   - hierarchical_scene_graph.json: 层次化场景图
#   - hierarchical_dashboard.png: 可视化面板
#   - zone_map_topdown.png: 俯视图
#   - scene_summary.json: 场景摘要
#
# 使用方法:
#   ./5.5b_hierarchical_segmentation.sh [SCENE_NAME]
#
# 示例:
#   ./5.5b_hierarchical_segmentation.sh room0
# =============================================================================

set -e

# 加载环境变量
if [ -f "./env_vars.bash" ]; then
    source ./env_vars.bash
fi

# 设置LLM服务地址
export LLM_BASE_URL="${LLM_BASE_URL:-http://10.21.231.7:8006}"
export LLM_MODEL="${LLM_MODEL:-gemini-3-flash-preview}"

# 场景设置
SCENE_NAME="${1:-room0}"
STRIDE=5
N_KEYFRAMES=15
THRESHOLD=1.2

# 数据集路径
REPLICA_ROOT="${REPLICA_ROOT:-/path/to/replica}"
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"
CACHE_DIR="${SCENE_PATH}/sg_cache_detect"
OUTPUT_DIR="${SCENE_PATH}/hierarchical_segmentation_detect"
PKL_FILE="full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"

echo "============================================================"
echo "Step 5.5b: 层次化功能区域划分"
echo "============================================================"
echo "场景: ${SCENE_NAME}"
echo "场景路径: ${SCENE_PATH}"
echo "缓存目录: ${CACHE_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "LLM服务: ${LLM_BASE_URL}"
echo "LLM模型: ${LLM_MODEL}"
echo "============================================================"

# 检查场景路径
if [ ! -d "${SCENE_PATH}" ]; then
    echo "错误: 场景路径不存在: ${SCENE_PATH}"
    exit 1
fi

# 检查物体地图文件
if [ ! -f "${SCENE_PATH}/pcd_saves/${PKL_FILE}" ]; then
    echo "错误: 未找到物体地图文件: ${SCENE_PATH}/pcd_saves/${PKL_FILE}"
    echo "请先运行: ./2b_build_3d_object_map_detect.sh ${SCENE_NAME}"
    exit 1
fi

# 检查object_affordances.json（来自5b+）
if [ ! -f "${CACHE_DIR}/object_affordances.json" ]; then
    echo "错误: 未找到affordances文件: ${CACHE_DIR}/object_affordances.json"
    echo "请先运行: ./5b_refine_with_affordance.sh ${SCENE_NAME}"
    exit 1
fi

echo ""
echo "✓ 找到 object_affordances.json (来自步骤5b+)"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 构建Python命令
PYTHON_CMD="python -m conceptgraph.segmentation.hierarchical_builder"
PYTHON_CMD="${PYTHON_CMD} --scene_path ${SCENE_PATH}"
PYTHON_CMD="${PYTHON_CMD} --pcd_file ${SCENE_PATH}/pcd_saves/${PKL_FILE}"
PYTHON_CMD="${PYTHON_CMD} --cache_dir ${CACHE_DIR}"
PYTHON_CMD="${PYTHON_CMD} --output ${OUTPUT_DIR}/hierarchical_scene_graph.json"
PYTHON_CMD="${PYTHON_CMD} --stride ${STRIDE}"
PYTHON_CMD="${PYTHON_CMD} --llm_url ${LLM_BASE_URL}"
PYTHON_CMD="${PYTHON_CMD} --no_vlm"  # 跳过VLM分析（已在5b+中完成）

echo ""
echo "执行命令:"
echo "${PYTHON_CMD}"
echo ""

# 运行主程序
cd "$(dirname "$0")/.."
eval ${PYTHON_CMD}

# 生成可视化
echo ""
echo "生成可视化..."
python -m conceptgraph.segmentation.hierarchical_visualizer \
    --scene_graph "${OUTPUT_DIR}/hierarchical_scene_graph.json" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "完成! 输出文件:"
echo "  - ${OUTPUT_DIR}/hierarchical_scene_graph.json"
echo "  - ${OUTPUT_DIR}/hierarchical_dashboard.png"
echo "  - ${OUTPUT_DIR}/zone_map_topdown.png"
echo "  - ${OUTPUT_DIR}/scene_summary.json"
echo "============================================================"
