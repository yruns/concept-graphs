#!/bin/bash
# =============================================================================
# Step 4.5b: 层次化功能区域划分
# =============================================================================
# 基于物体功能性和视频行为分析的三层层次化场景图构建
#
# 输入:
#   - 3D物体地图 (pcd_saves/*.pkl.gz)
#   - 物体描述 (captions.json)
#   - 相机轨迹 (traj.txt)
#   - RGB图像 (rgb/*.png) [可选，用于VLM分析]
#
# 输出:
#   - hierarchical_scene_graph.json: 层次化场景图
#   - hierarchical_dashboard.png: 可视化面板
#   - zone_map_topdown.png: 俯视图
#   - scene_summary.json: 场景摘要
#
# 使用方法:
#   ./4.5b_hierarchical_segmentation.sh [SCENE_NAME] [MODE]
#
# 示例:
#   ./4.5b_hierarchical_segmentation.sh room0 detect   # 类别感知模式 (推荐)
#   ./4.5b_hierarchical_segmentation.sh room0 none     # 类别无关模式
#   ./4.5b_hierarchical_segmentation.sh room0          # 默认使用类别感知模式
# =============================================================================

set -e

# 加载环境变量
if [ -f "./env_vars.bash" ]; then
    source ./env_vars.bash
fi

# 设置LLM服务地址 (如果未设置则使用默认值)
export LLM_BASE_URL="${LLM_BASE_URL:-http://10.21.231.7:8006}"
export LLM_MODEL="gemini-3-flash-preview"

# 默认参数
SCENE_NAME="${1:-room0}"
MODE="${2:-detect}"  # detect (类别感知) 或 none (类别无关)
STRIDE=5
N_KEYFRAMES=15

# 解析选项
USE_VLM="true"
USE_LLM="true"

# 数据集路径
REPLICA_ROOT="${REPLICA_ROOT:-/path/to/replica}"
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

# 根据模式设置文件路径
THRESHOLD=1.2
if [ "${MODE}" = "detect" ]; then
    PKL_FILE="full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
    CACHE_DIR="${SCENE_PATH}/sg_cache_detect"
    OUTPUT_DIR="${SCENE_PATH}/hierarchical_segmentation_detect"
else
    PKL_FILE="full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
    CACHE_DIR="${SCENE_PATH}/sg_cache"
    OUTPUT_DIR="${SCENE_PATH}/hierarchical_segmentation"
fi

echo "============================================================"
echo "Step 4.5b: 层次化功能区域划分"
echo "============================================================"
echo "场景: ${SCENE_NAME}"
echo "模式: ${MODE} ($([ "${MODE}" = "detect" ] && echo "类别感知" || echo "类别无关"))"
echo "场景路径: ${SCENE_PATH}"
echo "物体地图: ${PKL_FILE}"
echo "缓存目录: ${CACHE_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "使用VLM: ${USE_VLM}"
echo "使用LLM: ${USE_LLM}"
echo "LLM服务: ${LLM_BASE_URL}"
echo "============================================================"

# 检查场景路径
if [ ! -d "${SCENE_PATH}" ]; then
    echo "错误: 场景路径不存在: ${SCENE_PATH}"
    exit 1
fi

# 检查物体地图文件
if [ ! -f "${SCENE_PATH}/pcd_saves/${PKL_FILE}" ]; then
    echo "错误: 未找到物体地图文件: ${SCENE_PATH}/pcd_saves/${PKL_FILE}"
    if [ "${MODE}" = "detect" ]; then
        echo "请先运行: ./2b_build_3d_object_map_detect.sh ${SCENE_NAME}"
    else
        echo "请先运行: ./2_build_3d_object_map.sh"
    fi
    exit 1
fi

# 检查captions文件
if [ ! -f "${CACHE_DIR}/cfslam_llava_captions.json" ]; then
    echo "错误: 未找到captions文件: ${CACHE_DIR}/cfslam_llava_captions.json"
    if [ "${MODE}" = "detect" ]; then
        echo "请先运行: ./4b_extract_object_captions_detect.sh ${SCENE_NAME}"
    else
        echo "请先运行: ./4_extract_object_captions.sh"
    fi
    exit 1
fi

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

if [ "${USE_VLM}" = "false" ]; then
    PYTHON_CMD="${PYTHON_CMD} --no_vlm"
fi

if [ "${USE_LLM}" = "false" ]; then
    PYTHON_CMD="${PYTHON_CMD} --no_llm"
fi

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
