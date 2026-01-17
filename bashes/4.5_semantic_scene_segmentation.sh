#!/bin/bash
################################################################################
# Step 4.5: Region-Aware Scene Segmentation (区域感知场景分割)
################################################################################
#
# 作用：
#   - 两阶段场景划分：时序分割 + 片段聚类
#   - 支持识别不连续但属于同一区域的片段（如相机多次访问同一房间）
#   - 自动检测最佳区域数量
#
# 输入：
#   - 3D物体地图: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/*.pkl.gz
#   - 物体描述: $REPLICA_ROOT/$SCENE_NAME/sg_cache/cfslam_llava_captions.json
#   - 相机位姿: $REPLICA_ROOT/$SCENE_NAME/traj.txt
#
# 输出：
#   - 区域数据: $REPLICA_ROOT/$SCENE_NAME/sg_cache/segmentation_regions/regions.json
#   - 信号分析图: segmentation.png
#   - 区域总览: regions_overview.jpg
#   - 区域GIF: region_gifs/region_XX.gif
#   - 关键帧: region_keyframes/region_XX_seg_YY.jpg
#   - 区域点云: region_pointclouds/
#
################################################################################

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env_vars.bash"

# 参数
SCENE_NAME="${1:-room0}"
STRIDE="${2:-5}"
MAX_REGIONS="${3:-10}"

echo "================================================"
echo "Step 4.5: Region-Aware Scene Segmentation"
echo "================================================"
echo "场景: $SCENE_NAME"
echo "采样步长: $STRIDE"
echo "最大区域数: $MAX_REGIONS (自动检测)"
echo ""

SCENE_PATH="$REPLICA_ROOT/$SCENE_NAME"

# Check prerequisites
if ! ls "$SCENE_PATH/pcd_saves/"*.pkl.gz 1>/dev/null 2>&1; then
    echo "✗ 错误: 3D物体地图不存在"
    echo "  请先运行 Step 2 (2_build_3d_object_map.sh)"
    exit 1
fi

if [ ! -f "$SCENE_PATH/sg_cache/cfslam_llava_captions.json" ]; then
    echo "✗ 错误: 物体描述文件不存在"
    echo "  请先运行 Step 4 (4_extract_object_captions.sh)"
    exit 1
fi

echo "输入:"
echo "  - 3D物体地图: $SCENE_PATH/pcd_saves/"
echo "  - 物体描述: $SCENE_PATH/sg_cache/cfslam_llava_captions.json"
echo ""
echo "输出: $SCENE_PATH/sg_cache/segmentation_regions/"
echo "================================================"
echo ""

cd "$SCRIPT_DIR/.."
python -m conceptgraph.segmentation.region_aware_segmenter \
    --dataset_root "$REPLICA_ROOT" \
    --scene "$SCENE_NAME" \
    --stride "$STRIDE" \
    --max_regions "$MAX_REGIONS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 区域感知场景分割完成"
    echo ""
    echo "输出文件:"
    echo "  - 区域数据: $SCENE_PATH/sg_cache/segmentation_regions/regions.json"
    echo "  - 信号分析: $SCENE_PATH/sg_cache/segmentation_regions/segmentation.png"
    echo "  - 区域总览: $SCENE_PATH/sg_cache/segmentation_regions/regions_overview.jpg"
    echo "  - 区域GIF:  $SCENE_PATH/sg_cache/segmentation_regions/region_gifs/"
    echo ""
    echo "可解释性可视化:"
    echo "  - 分割原因: segmentation_reasons.png"
    echo "  - 区域语义: region_semantics.png"
    echo "  - 物体时间线: object_visibility_timeline_full.png"
    echo ""
else
    echo ""
    echo "✗ 场景分割失败"
    exit 1
fi