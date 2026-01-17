#!/bin/bash

################################################################################
# 步骤 5B: 细化物体描述 (使用类别感知模式的检测结果)
################################################################################
#
# 作用：
#   - 使用大语言模型整合每个物体的多视图描述
#   - 识别和解决描述冲突
#   - 生成统一的物体语义标签
#   - 过滤无效或低质量的检测
#
# 输入：
#   - 原始描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache_detect/cfslam_llava_captions.json
#   - 3D 对象地图: ram_withbg_allclasses 版本
#
# 输出：
#   - 精炼描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache_detect/cfslam_gpt-4_responses/*.json
#   - 汇总文件: $REPLICA_ROOT/$SCENE_NAME/sg_cache_detect/cfslam_gpt-4_responses.pkl
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
export LLM_BASE_URL="http://10.21.231.7:8006"
export LLM_MODEL="gemini-3-flash-preview"

# 场景设置
SCENE_NAME=${1:-room0}
THRESHOLD=1.2
PKL_FILENAME=full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz

# 使用类别感知的缓存目录
CACHE_DIR="${REPLICA_ROOT}/${SCENE_NAME}/sg_cache_detect"

echo "================================================"
echo "步骤 5B: 细化物体描述 (类别感知模式)"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "LLM 服务器: ${LLM_BASE_URL}"
echo "模型: ${LLM_MODEL}"
echo ""
echo "输入: ${CACHE_DIR}/cfslam_llava_captions.json"
echo "输出: ${CACHE_DIR}/cfslam_gpt-4_responses/"
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
if [ ! -f "${CACHE_DIR}/cfslam_llava_captions.json" ]; then
    echo "✗ 错误: 输入文件不存在"
    echo "   ${CACHE_DIR}/cfslam_llava_captions.json"
    echo ""
    echo "请先运行步骤 4B (4b_extract_object_captions_detect.sh)"
    exit 1
fi

# 生成具体python命令
echo "python scenegraph/build_scenegraph_cfslam.py \
    --mode refine-node-captions \
    --cachedir ${CACHE_DIR} \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"

# 细化物体描述
python scenegraph/build_scenegraph_cfslam.py \
    --mode refine-node-captions \
    --cachedir ${CACHE_DIR} \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 物体描述细化完成 (类别感知模式)"
    echo ""
    echo "输出文件:"
    echo "  - 每个物体的精炼描述: ${CACHE_DIR}/cfslam_gpt-4_responses/"
    echo "  - 汇总文件: ${CACHE_DIR}/cfslam_gpt-4_responses.pkl"
    echo ""
else
    echo ""
    echo "✗ 物体描述细化失败"
    exit 1
fi
