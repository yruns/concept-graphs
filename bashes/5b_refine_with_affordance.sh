#!/bin/bash

################################################################################
# 步骤 5B+: 精炼物体描述并提取Affordance（带图像）
################################################################################
#
# 作用：
#   - 合并步骤5（精炼描述）和步骤4.5b（affordance提取）
#   - 使用VLM同时分析图像和多视角描述
#   - 生成统一的物体标签、精炼描述和功能性属性
#
# 输入：
#   - 原始描述: sg_cache_detect/cfslam_llava_captions.json
#   - 物体图像: sg_cache_detect/cfslam_captions_llava_debug/*.png
#
# 输出：
#   - sg_cache_detect/object_affordances.json
#     包含: object_tag, summary, category, affordances
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
cd "${ROOT_DIR}"

# 配置LLM (使用VLM模型，需要能处理图像)
export LLM_MODEL="gpt-5.2-2025-12-11"

# 场景设置
SCENE_NAME=${1:-room0}
IMAGE_NUM=${2:-4}  # 每个物体使用的图像数量，默认1
MAX_WORKERS=${3:-10}  # 并行worker数量，默认20
THRESHOLD=1.2
CACHE_DIR="${REPLICA_ROOT}/${SCENE_NAME}/sg_cache_detect"
PCD_FILE="${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"

echo "================================================"
echo "步骤 5B+: 精炼描述 + Affordance提取（带图像）"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "每物体图像数: ${IMAGE_NUM}"
echo "并行workers: ${MAX_WORKERS}"
echo "LLM 服务器: ${LLM_BASE_URL}"
echo "模型: ${LLM_MODEL}"
echo ""
echo "输入:"
echo "  - Captions: ${CACHE_DIR}/cfslam_llava_captions.json"
echo "  - PCD文件: ${PCD_FILE}"
echo ""
echo "输出:"
echo "  - ${CACHE_DIR}/object_affordances.json"
echo "================================================"
echo ""

# 检查输入文件
if [ ! -f "${CACHE_DIR}/cfslam_llava_captions.json" ]; then
    echo "✗ 错误: Captions文件不存在"
    echo "   请先运行: ./4b_extract_object_captions_detect.sh ${SCENE_NAME}"
    exit 1
fi

# 检查pcd文件
if [ ! -f "${PCD_FILE}" ]; then
    echo "⚠ 警告: PCD文件不存在，将只使用文本描述"
    echo "   ${PCD_FILE}"
fi

SCRIPT_FILE="${ROOT_DIR}/conceptgraph/query_scene/refine_with_affordance.py"

echo "python ${SCRIPT_FILE} \\
    --cache_dir \"${CACHE_DIR}\" \\
    --pcd_file \"${PCD_FILE}\" \\
    --image_num ${IMAGE_NUM} \\
    --max_workers ${MAX_WORKERS} \\
    --output \"${CACHE_DIR}/object_affordances.json\""

# 运行
python "${SCRIPT_FILE}" \
    --cache_dir "${CACHE_DIR}" \
    --pcd_file "${PCD_FILE}" \
    --image_num ${IMAGE_NUM} \
    --max_workers ${MAX_WORKERS} \
    --output "${CACHE_DIR}/object_affordances.json"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 精炼描述 + Affordance提取完成"
    echo ""
    echo "输出文件: ${CACHE_DIR}/object_affordances.json"
    echo ""
    
    # 显示统计
    echo "物体类别分布:"
    python3 -c "
import json
from collections import Counter
with open('${CACHE_DIR}/object_affordances.json') as f:
    data = json.load(f)
cats = Counter(d.get('category', '未知') for d in data)
for cat, cnt in cats.most_common():
    print(f'  {cat}: {cnt}')
print()
print('物体标签示例:')
for d in data[:10]:
    print(f\"  {d['id']}: {d.get('object_tag', 'N/A')}\")
"
else
    echo ""
    echo "✗ 处理失败"
    exit 1
fi
