#!/bin/bash
################################################################################
# 批量运行场景分段 (步骤 1b + 2.5)
################################################################################
# 
# 运行流程:
#   1b. RAM 模式 2D 分割 (提取物体类别)
#   2.5 时序场景分段 (多模态融合)
#
# 输出:
#   - GSA 检测: $REPLICA_ROOT/$SCENE/gsa_detections_ram_withbg_allclasses/
#   - 分段结果: $REPLICA_ROOT/$SCENE/sg_cache/segmentation_balanced/
#
################################################################################

set -e  # 遇错即停

source /home/shyue/anaconda3/bin/activate conceptgraph
cd /home/shyue/codebase/concept-graphs
source env_vars.bash

export PYTHONPATH="/home/shyue/codebase/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# 场景列表 (8个场景)
SCENES=(
    "room0"
    "room1"
    "room2"
    "office0"
    "office1"
    "office2"
    "office3"
    "office4"
)

# 参数
STRIDE=5
GSA_MODE="ram_withbg_allclasses"

echo "============================================================"
echo "批量场景分段"
echo "============================================================"
echo "场景数: ${#SCENES[@]}"
echo "步骤: 1b (RAM 2D分割) → 2.5 (时序分段)"
echo "============================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

for i in "${!SCENES[@]}"; do
    SCENE="${SCENES[$i]}"
    idx=$((i + 1))
    
    echo ""
    echo "============================================================"
    echo "[$idx/${#SCENES[@]}] 处理场景: $SCENE"
    echo "============================================================"
    
    SCENE_PATH="$REPLICA_ROOT/$SCENE"
    GSA_DIR="$SCENE_PATH/gsa_detections_$GSA_MODE"
    SEG_DIR="$SCENE_PATH/sg_cache/segmentation_balanced"
    
    # 检查是否已完成
    if [ -f "$SEG_DIR/trajectory_segments.json" ]; then
        echo "  ✓ 已完成，跳过"
        continue
    fi
    
    # 步骤 1b: RAM 模式 2D 分割
    echo ""
    echo "--- 步骤 1b: RAM 2D 分割 ---"
    
    if [ -d "$GSA_DIR" ] && [ "$(ls -1 $GSA_DIR/*.pkl.gz 2>/dev/null | wc -l)" -ge 100 ]; then
        echo "  ✓ GSA 结果已存在，跳过"
    else
        echo "  运行 RAM 检测..."
        cd /home/shyue/codebase/concept-graphs/conceptgraph
        
        python scripts/generate_gsa_results.py \
            --dataset_root $REPLICA_ROOT \
            --dataset_config $REPLICA_CONFIG_PATH \
            --scene_id $SCENE \
            --class_set ram \
            --box_threshold 0.2 \
            --text_threshold 0.2 \
            --stride $STRIDE \
            --add_bg_classes \
            --accumu_classes \
            --exp_suffix withbg_allclasses
        
        echo "  ✓ RAM 检测完成"
    fi
    
    # 步骤 2.5: 时序场景分段
    echo ""
    echo "--- 步骤 2.5: 时序分段 ---"
    
    cd /home/shyue/codebase/concept-graphs/conceptgraph
    python segmentation/balanced_segmenter.py \
        --dataset_root $REPLICA_ROOT \
        --scene $SCENE \
        --gsa_mode $GSA_MODE \
        --stride $STRIDE \
        --min_frames 40 \
        --target_regions 6
    
    echo "  ✓ 时序分段完成"
    
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "全部完成!"
echo "============================================================"
echo "处理场景: ${#SCENES[@]} 个"
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo ""
echo "结果位置:"
for SCENE in "${SCENES[@]}"; do
    SEG_FILE="$REPLICA_ROOT/$SCENE/sg_cache/segmentation_balanced/trajectory_segments.json"
    if [ -f "$SEG_FILE" ]; then
        N_REGIONS=$(python -c "import json; print(len(json.load(open('$SEG_FILE'))))")
        echo "  $SCENE: $N_REGIONS 个区域"
    fi
done
