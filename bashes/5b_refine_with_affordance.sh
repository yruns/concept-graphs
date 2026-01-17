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

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph
export PYTHONPATH="/home/shyue/codebase/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# 进入工作目录
cd /home/shyue/codebase/concept-graphs

# 加载环境变量
source ./env_vars.bash

# 配置LLM (使用VLM模型，需要能处理图像)
export LLM_BASE_URL="http://10.21.231.7:8006"
export LLM_MODEL="gemini-3-flash-preview"

# 场景设置
SCENE_NAME=${1:-room0}
THRESHOLD=1.2
CACHE_DIR="${REPLICA_ROOT}/${SCENE_NAME}/sg_cache_detect"
PCD_FILE="${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"

echo "================================================"
echo "步骤 5B+: 精炼描述 + Affordance提取（带图像）"
echo "================================================"
echo "场景: ${SCENE_NAME}"
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

# 检查 LLM 服务
if ! curl -s ${LLM_BASE_URL}/healthz > /dev/null 2>&1; then
    echo "✗ 错误: LLM 服务未运行"
    exit 1
fi
echo "✓ LLM 服务运行正常"
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

echo "python -m conceptgraph.segmentation.refine_with_affordance \
    --cache_dir \"${CACHE_DIR}\" \
    --pcd_file \"${PCD_FILE}\" \
    --output \"${CACHE_DIR}/object_affordances.json\""

# 运行
python -m conceptgraph.segmentation.refine_with_affordance \
    --cache_dir "${CACHE_DIR}" \
    --pcd_file "${PCD_FILE}" \
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
