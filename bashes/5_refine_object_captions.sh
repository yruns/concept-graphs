#!/bin/bash

################################################################################
# 步骤 5: 细化物体描述 (使用 Ollama GPT)
################################################################################
#
# 作用：
#   - 使用大语言模型 (Ollama GPT) 整合每个物体的多视图描述
#   - 识别和解决描述冲突
#   - 生成统一的物体语义标签
#   - 过滤无效或低质量的检测
#
# 输入：
#   - 原始描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_llava_captions.json
#   - 3D 对象地图: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz
#
# 输出：
#   - 精炼描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_gpt-4_responses/*.json
#   - 汇总文件: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_gpt-4_responses.pkl
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph
export PYTHONPATH="/home/shyue/codebase/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# 进入工作目录
cd /home/shyue/codebase/concept-graphs/conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 配置统一LLM客户端
export LLM_BASE_URL="http://10.21.231.7:8000"
export LLM_MODEL="gpt-4o-2024-08-06"

# 场景设置
SCENE_NAME=room0
THRESHOLD=1.2
PKL_FILENAME=full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz

echo "================================================"
echo "步骤 5: 细化物体描述"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "LLM 服务器: ${LLM_BASE_URL}"
echo "模型: ${LLM_MODEL}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_llava_captions.json"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_gpt-4_responses/"
echo "================================================"
echo ""

# 检查 LLM 服务
if ! curl -s ${LLM_BASE_URL}/healthz > /dev/null 2>&1; then
    echo "✗ 错误: LLM 服务未运行"
    echo ""
    echo "请检查 LLM 服务器: ${LLM_BASE_URL}"
    echo ""
    exit 1
fi

echo "✓ LLM 服务运行正常"
echo ""

# 检查输入文件
if [ ! -f "${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_llava_captions.json" ]; then
    echo "✗ 错误: 输入文件不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_llava_captions.json"
    echo ""
    echo "请先运行步骤 4 (4_extract_object_captions.sh)"
    exit 1
fi

# 细化物体描述
python scenegraph/build_scenegraph_cfslam.py \
    --mode refine-node-captions \
    --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 物体描述细化完成"
    echo ""
    echo "输出文件:"
    echo "  - 每个物体的精炼描述: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_gpt-4_responses/"
    echo "  - 汇总文件: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_gpt-4_responses.pkl"
    echo ""
else
    echo ""
    echo "✗ 物体描述细化失败"
    exit 1
fi

