#!/bin/bash

################################################################################
# 步骤 6: 构建场景图 (使用 Ollama GPT)
################################################################################
#
# 作用：
#   - 分析物体之间的空间关系 (在...上、在...里等)
#   - 使用大语言模型推理物体关系
#   - 构建场景图 (节点=物体, 边=关系)
#   - 使用最小生成树优化图结构
#
# 输入：
#   - 精炼描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_<model>_responses/
#   - 3D 对象地图: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz
#
# 输出：
#   - 场景图地图: $REPLICA_ROOT/$SCENE_NAME/sg_cache/map/scene_map_cfslam_pruned.pkl.gz
#   - 物体关系: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_object_relations.json
#   - 关系查询: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_object_relation_queries.json
#   - 场景图边: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_scenegraph_edges.pkl
#   - JSON 摘要: $REPLICA_ROOT/$SCENE_NAME/sg_cache/scene_graph.json
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

# 配置统一LLM客户端
export LLM_BASE_URL="http://10.21.231.7:8006"
export LLM_MODEL="gpt-5.2-2025-12-11"
SAFE_LLM_MODEL="${LLM_MODEL//\//_}"
SAFE_LLM_MODEL="${SAFE_LLM_MODEL//:/_}"
RESPONSES_DIR="cfslam_${SAFE_LLM_MODEL}_responses"

# 场景设置
SCENE_NAME=room0
THRESHOLD=1.2
PKL_FILENAME=full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz

echo "================================================"
echo "步骤 6: 构建场景图"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "LLM 服务器: ${LLM_BASE_URL}"
echo "模型: ${LLM_MODEL}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/${RESPONSES_DIR}/"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz"
echo "      ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json"
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
if [ ! -f "${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/${RESPONSES_DIR}.pkl" ]; then
    echo "✗ 错误: 输入文件不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/${RESPONSES_DIR}.pkl"
    echo ""
    echo "请先运行步骤 5 (5_refine_object_captions.sh)"
    exit 1
fi

# 构建场景图
python scenegraph/build_scenegraph_cfslam.py \
    --mode build-scenegraph \
    --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 场景图构建完成"
    echo ""
    echo "输出文件:"
    echo "  - 场景图地图: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/map/scene_map_cfslam_pruned.pkl.gz"
    echo "  - 物体关系: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relations.json"
    echo "  - 关系查询: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_object_relation_queries.json"
    echo "  - 场景图边: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_scenegraph_edges.pkl"
    echo ""
else
    echo ""
    echo "✗ 场景图构建失败"
    exit 1
fi

