#!/bin/bash

################################################################################
# 步骤 4B: 提取物体描述 (使用类别感知模式的检测结果)
################################################################################
#
# 作用：
#   - 使用视觉-语言模型为类别感知模式检测到的物体生成文本描述
#   - 处理每个物体的多个观察视角
#   - 生成初始的自然语言标注
#
# 输入：
#   - 3D 对象地图: ram_withbg_allclasses 版本
#
# 输出：
#   - 物体描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache_detect/cfslam_llava_captions.json
#
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 激活环境
if [ -f "${HOME}/anaconda3/bin/activate" ]; then
    source "${HOME}/anaconda3/bin/activate" conceptgraph
elif [ -f "${HOME}/miniconda3/bin/activate" ]; then
    source "${HOME}/miniconda3/bin/activate" conceptgraph
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate conceptgraph
else
    echo "⚠ 未找到 conda 激活脚本，继续使用当前环境"
fi

# 加载环境变量
source "${ROOT_DIR}/env_vars.bash"
if [ -n "${GSA_PATH}" ]; then
    export PYTHONPATH="${GSA_PATH}/GroundingDINO:${PYTHONPATH}"
fi

# 进入工作目录
cd "${ROOT_DIR}/conceptgraph"

# 配置 LLM 客户端
export LLM_MODEL="${LLM_MODEL:-gpt-5.2-2025-12-11}"
export NUM_WORKERS=10

# 场景设置
SCENE_NAME=${1:-room0}
THRESHOLD=1.2
PKL_FILENAME=full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz

# 使用独立的缓存目录以避免覆盖
CACHE_DIR="${REPLICA_ROOT}/${SCENE_NAME}/sg_cache_detect"

echo "================================================"
echo "步骤 4B: 提取物体描述 (类别感知模式)"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "LLM 客户端: llm_client"
echo "模型: ${LLM_MODEL}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"
echo "输出: ${CACHE_DIR}/cfslam_llava_captions.json"
echo "================================================"
echo ""

# 创建缓存目录
mkdir -p "${CACHE_DIR}"

# LLM 客户端由 llm_client 统一管理
echo "✓ LLM 客户端已配置"
echo ""

# 检查输入文件
if [ ! -f "${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}" ]; then
    echo "✗ 错误: 输入文件不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"
    echo ""
    echo "请先运行步骤 2B (2b_build_3d_object_map_detect.sh)"
    exit 1
fi

echo "python scenegraph/build_scenegraph_cfslam.py \
    --mode extract-node-captions \
    --cachedir ${CACHE_DIR} \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"

# 提取物体描述
python scenegraph/build_scenegraph_cfslam.py \
    --mode extract-node-captions \
    --cachedir ${CACHE_DIR} \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 物体描述提取完成 (类别感知模式)"
    echo ""
    echo "输出文件:"
    echo "  - 描述 JSON: ${CACHE_DIR}/cfslam_llava_captions.json"
    echo "  - 特征文件: ${CACHE_DIR}/cfslam_feat_llava/"
    echo "  - 调试图像: ${CACHE_DIR}/cfslam_captions_llava_debug/"
    echo ""
else
    echo ""
    echo "✗ 物体描述提取失败"
    exit 1
fi
