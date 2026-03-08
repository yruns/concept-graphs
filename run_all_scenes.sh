#!/bin/bash
################################################################################
# 批量运行 4b + 5b 脚本处理所有 Replica 场景
# 使用 GeminiClientPool 提升并发
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 激活 .venv 环境
source .venv/bin/activate

# 加载环境变量
source env_vars.bash

# 配置 GeminiClientPool
export USE_GEMINI_POOL=true
export LLM_MODEL=gemini-2.5-pro
export NUM_WORKERS=8  # 4B 步骤的并行 worker 数量，减少以避免 429 限流
export PYTHONPATH="${CG_FOLDER}:${GSA_PATH}/GroundingDINO:${PYTHONPATH}"

# 使用 .venv 的 python
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# 需要处理的场景 (所有场景)
SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")

THRESHOLD=1.2
PKL_FILENAME="full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"

echo "========================================"
echo "批量处理 Replica 场景"
echo "使用: GeminiClientPool (${LLM_MODEL})"
echo "场景: ${SCENES[*]}"
echo "========================================"
echo ""

for SCENE_NAME in "${SCENES[@]}"; do
    echo ""
    echo "========================================"
    echo "处理场景: ${SCENE_NAME}"
    echo "========================================"

    CACHE_DIR="${REPLICA_ROOT}/${SCENE_NAME}/sg_cache_detect"
    PCD_FILE="${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_FILENAME}"

    # 检查输入文件
    if [ ! -f "${PCD_FILE}" ]; then
        echo "✗ 跳过 ${SCENE_NAME}: PCD 文件不存在"
        continue
    fi

    mkdir -p "${CACHE_DIR}"

    # 删除旧的输出文件以强制重新运行
    rm -f "${CACHE_DIR}/cfslam_llava_captions.json"
    rm -f "${CACHE_DIR}/object_affordances.json"

    # ============ 步骤 4B: 提取物体描述 ============
    CAPTION_FILE="${CACHE_DIR}/cfslam_llava_captions.json"
    echo ""
    echo ">>> 运行步骤 4B: 提取物体描述..."
    ${PYTHON} conceptgraph/scenegraph/build_scenegraph_cfslam.py \
        --mode extract-node-captions \
        --cachedir "${CACHE_DIR}" \
        --mapfile "${PCD_FILE}"

    if [ $? -eq 0 ]; then
        echo "✓ 步骤 4B 完成"
    else
        echo "✗ 步骤 4B 失败，跳过此场景"
        continue
    fi

    # ============ 步骤 5B: 精炼描述 + Affordance ============
    AFFORD_FILE="${CACHE_DIR}/object_affordances.json"
    echo ""
    echo ">>> 运行步骤 5B: 精炼描述 + Affordance..."
    ${PYTHON} conceptgraph/query_scene/refine_with_affordance.py \
        --cache_dir "${CACHE_DIR}" \
        --pcd_file "${PCD_FILE}" \
        --image_num 4 \
        --max_workers 10 \
        --output "${CACHE_DIR}/object_affordances.json"

    if [ $? -eq 0 ]; then
        echo "✓ 步骤 5B 完成"
    else
        echo "✗ 步骤 5B 失败"
    fi

    echo ""
    echo "✓ 场景 ${SCENE_NAME} 处理完成"
done

echo ""
echo "========================================"
echo "所有场景处理完成"
echo "========================================"
