#!/bin/bash

################################################################################
# 步骤 4: 提取物体描述 (使用 Ollama Vision)
################################################################################
#
# 作用：
#   - 使用视觉-语言模型 (Ollama Vision) 为每个物体生成文本描述
#   - 处理每个物体的多个观察视角
#   - 生成初始的自然语言标注
#
# 输入：
#   - 3D 对象地图: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz
#   - 物体裁剪图像: 从原始 RGB 图像中提取
#
# 输出：
#   - 物体描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_llava_captions.json
#   - CLIP 特征: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_feat_llava/*.pt
#   - 调试可视化: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_captions_llava_debug/*.png
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
export NUM_WORKERS=4

# 场景设置
SCENE_NAME=room0
THRESHOLD=1.2
PKL_FILENAME=full_pcd_none_overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz

echo "================================================"
echo "步骤 4: 提取物体描述"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "LLM 服务器: ${LLM_BASE_URL}"
echo "模型: ${LLM_MODEL}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_llava_captions.json"
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
if [ ! -f "${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}" ]; then
    echo "✗ 错误: 输入文件不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"
    echo ""
    echo "请先运行步骤 2 (2_build_3d_object_map.sh)"
    exit 1
fi

# 提取物体描述
python scenegraph/build_scenegraph_cfslam.py \
    --mode extract-node-captions \
    --cachedir ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache \
    --mapfile ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 物体描述提取完成"
    echo ""
    echo "输出文件:"
    echo "  - 描述 JSON: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_llava_captions.json"
    echo "  - 特征文件: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_feat_llava/"
    echo "  - 调试图像: ${REPLICA_ROOT}/${SCENE_NAME}/sg_cache/cfslam_captions_llava_debug/"
    echo ""
else
    echo ""
    echo "✗ 物体描述提取失败"
    exit 1
fi

