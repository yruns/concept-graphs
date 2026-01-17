#!/bin/bash

################################################################################
# 批量运行 Step 1: 提取所有场景的 2D 分割和 CLIP 特征
################################################################################
#
# 作用：
#   - 对所有场景运行 Step 1（SAM 分割 + CLIP 特征提取）
#   - 包括帧级 CLIP 特征（用于场景分割）
#
# 场景列表：
#   - room0, room1, room2
#   - office0, office1, office2, office3, office4
#
################################################################################

set -e

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph
export PYTHONPATH="/home/shyue/codebase/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# 进入工作目录
cd /home/shyue/codebase/concept-graphs/conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 所有场景列表
SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")

# 可选：只运行指定场景
if [ $# -gt 0 ]; then
    SCENES=("$@")
fi

echo "========================================================"
echo "批量运行 Step 1: 提取 2D 分割和 CLIP 特征"
echo "========================================================"
echo "场景列表: ${SCENES[*]}"
echo "总数: ${#SCENES[@]} 个场景"
echo "========================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 遍历所有场景
for i in "${!SCENES[@]}"; do
    SCENE_NAME="${SCENES[$i]}"
    
    echo ""
    echo "========================================================"
    echo "[$(($i+1))/${#SCENES[@]}] 处理场景: $SCENE_NAME"
    echo "========================================================"
    
    # 检查场景是否存在
    if [ ! -d "$REPLICA_ROOT/$SCENE_NAME" ]; then
        echo "⚠ 警告: 场景目录不存在，跳过: $REPLICA_ROOT/$SCENE_NAME"
        continue
    fi
    
    # 检查是否已经有帧级 CLIP 特征
    FIRST_FILE=$(ls "$REPLICA_ROOT/$SCENE_NAME/gsa_detections_none/"*.pkl.gz 2>/dev/null | head -1)
    if [ -n "$FIRST_FILE" ]; then
        HAS_CLIP=$(python3 -c "
import gzip, pickle
with gzip.open('$FIRST_FILE', 'rb') as f:
    d = pickle.load(f)
print('yes' if 'frame_clip_feat' in d else 'no')
" 2>/dev/null || echo "no")
        
        if [ "$HAS_CLIP" = "yes" ]; then
            echo "✓ 场景 $SCENE_NAME 已有帧级 CLIP 特征，跳过"
            continue
        fi
    fi
    
    echo "开始处理..."
    SCENE_START=$(date +%s)
    
    # 运行分割提取
    python scripts/generate_gsa_results.py \
        --dataset_root "$REPLICA_ROOT" \
        --dataset_config "$REPLICA_CONFIG_PATH" \
        --scene_id "$SCENE_NAME" \
        --class_set none \
        --stride 5
    
    SCENE_END=$(date +%s)
    SCENE_DURATION=$((SCENE_END - SCENE_START))
    
    if [ $? -eq 0 ]; then
        echo "✓ 场景 $SCENE_NAME 完成 (耗时: ${SCENE_DURATION}秒)"
    else
        echo "✗ 场景 $SCENE_NAME 失败"
    fi
done

# 统计总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "========================================================"
echo "全部完成！"
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo "========================================================"

# 验证结果
echo ""
echo "验证各场景帧级 CLIP 特征:"
for SCENE_NAME in "${SCENES[@]}"; do
    if [ -d "$REPLICA_ROOT/$SCENE_NAME/gsa_detections_none" ]; then
        FIRST_FILE=$(ls "$REPLICA_ROOT/$SCENE_NAME/gsa_detections_none/"*.pkl.gz 2>/dev/null | head -1)
        if [ -n "$FIRST_FILE" ]; then
            HAS_CLIP=$(python3 -c "
import gzip, pickle
with gzip.open('$FIRST_FILE', 'rb') as f:
    d = pickle.load(f)
print('yes' if 'frame_clip_feat' in d else 'no')
" 2>/dev/null || echo "error")
            if [ "$HAS_CLIP" = "yes" ]; then
                echo "  ✓ $SCENE_NAME: 有帧级 CLIP 特征"
            else
                echo "  ✗ $SCENE_NAME: 无帧级 CLIP 特征"
            fi
        else
            echo "  ✗ $SCENE_NAME: 无检测文件"
        fi
    else
        echo "  ✗ $SCENE_NAME: 目录不存在"
    fi
done
