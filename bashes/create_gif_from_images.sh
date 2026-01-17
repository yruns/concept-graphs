#!/bin/bash

################################################################################
# 从可视化图像直接生成 GIF (不需要视频)
################################################################################
#
# 作用：
#   - 直接从 JPG 图像生成 GIF
#   - 跳过视频生成步骤
#   - 提供自定义选项
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 参数设置
SCENE_NAME=room0
FPS=10
SCALE=800  # 宽度像素
COLORS=128  # 调色板颜色数 (越少文件越小)

IMAGE_DIR="${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none"
OUTPUT_GIF="${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.gif"

echo "================================================"
echo "从图像直接生成 GIF"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "图像目录: ${IMAGE_DIR}"
echo "输出: ${OUTPUT_GIF}"
echo ""
echo "参数:"
echo "  - 帧率: ${FPS} fps"
echo "  - 宽度: ${SCALE} 像素"
echo "  - 颜色数: ${COLORS}"
echo "================================================"
echo ""

# 检查目录
if [ ! -d "${IMAGE_DIR}" ]; then
    echo "✗ 错误: 图像目录不存在: ${IMAGE_DIR}"
    exit 1
fi

# 检查图像数量
IMAGE_COUNT=$(ls ${IMAGE_DIR}/*.jpg 2>/dev/null | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then
    echo "✗ 错误: 没有找到图像文件"
    exit 1
fi

echo "找到 ${IMAGE_COUNT} 张图像"
echo ""
echo "开始生成 GIF..."
echo ""

# 使用 ffmpeg 生成 GIF
ffmpeg -framerate ${FPS} \
    -pattern_type glob \
    -i "${IMAGE_DIR}/*.jpg" \
    -vf "fps=${FPS},scale=${SCALE}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=${COLORS}[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
    -y "${OUTPUT_GIF}" 2>&1 | grep -E "frame=|video:"

if [ $? -eq 0 ] && [ -f "${OUTPUT_GIF}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_GIF}" | cut -f1)
    DURATION=$(echo "scale=2; ${IMAGE_COUNT}/${FPS}" | bc)
    
    echo ""
    echo "✅ GIF 生成成功！"
    echo ""
    echo "文件信息:"
    echo "  - 路径: ${OUTPUT_GIF}"
    echo "  - 大小: ${FILE_SIZE}"
    echo "  - 总帧数: ${IMAGE_COUNT}"
    echo "  - 时长: ${DURATION} 秒"
    echo "  - 帧率: ${FPS} fps"
    echo ""
    echo "预览命令:"
    echo "  firefox ${OUTPUT_GIF}"
    echo ""
else
    echo ""
    echo "✗ GIF 生成失败"
    exit 1
fi

