#!/bin/bash

################################################################################
# 将 2D 分割视频转换为 GIF (适合 PPT 演示)
################################################################################
#
# 作用：
#   - 将 MP4 视频转换为 GIF 动图
#   - 提供多种质量/大小选项
#   - 优化 GIF 文件大小
#
# 输入：
#   - 视频文件: gsa_vis_none.mp4
#
# 输出：
#   - GIF 文件: gsa_vis_none.gif (多种版本)
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=room0
INPUT_VIDEO="${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.mp4"

echo "================================================"
echo "将视频转换为 GIF 动图"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "输入: ${INPUT_VIDEO}"
echo ""
echo "将生成多个不同质量的 GIF 版本："
echo "  1. 高质量 (较大, 适合全屏演示)"
echo "  2. 中等质量 (推荐, 平衡质量和大小)"
echo "  3. 低质量 (最小, 适合快速加载)"
echo "  4. 快速版本 (帧率加倍, 15 fps)"
echo "================================================"
echo ""

# 检查输入文件
if [ ! -f "${INPUT_VIDEO}" ]; then
    echo "✗ 错误: 视频文件不存在"
    echo "   ${INPUT_VIDEO}"
    echo ""
    echo "请先运行: bash create_2d_segmentation_video.sh"
    exit 1
fi

# 检查 ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "✗ 错误: 未找到 ffmpeg"
    exit 1
fi

# 输出目录
OUTPUT_DIR="${REPLICA_ROOT}/${SCENE_NAME}"

echo "开始转换..."
echo ""

# ============================================================================
# 方案 1: 高质量 GIF (分辨率: 1200x680, 帧率: 10fps)
# ============================================================================
echo "🎨 [1/4] 生成高质量 GIF..."
OUTPUT_HQ="${OUTPUT_DIR}/gsa_vis_none_hq.gif"

ffmpeg -i "${INPUT_VIDEO}" \
    -vf "fps=10,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
    -y "${OUTPUT_HQ}" 2>/dev/null

if [ -f "${OUTPUT_HQ}" ]; then
    SIZE_HQ=$(du -h "${OUTPUT_HQ}" | cut -f1)
    echo "   ✓ 完成: ${OUTPUT_HQ}"
    echo "   文件大小: ${SIZE_HQ}"
else
    echo "   ✗ 失败"
fi
echo ""

# ============================================================================
# 方案 2: 中等质量 GIF (分辨率: 800x~, 帧率: 10fps) 推荐用于PPT
# ============================================================================
echo "📊 [2/4] 生成中等质量 GIF (推荐用于PPT)..."
OUTPUT_MQ="${OUTPUT_DIR}/gsa_vis_none_medium.gif"

ffmpeg -i "${INPUT_VIDEO}" \
    -vf "fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
    -y "${OUTPUT_MQ}" 2>/dev/null

if [ -f "${OUTPUT_MQ}" ]; then
    SIZE_MQ=$(du -h "${OUTPUT_MQ}" | cut -f1)
    echo "   ✓ 完成: ${OUTPUT_MQ}"
    echo "   文件大小: ${SIZE_MQ}"
else
    echo "   ✗ 失败"
fi
echo ""

# ============================================================================
# 方案 3: 低质量 GIF (分辨率: 600x~, 帧率: 5fps)
# ============================================================================
echo "💾 [3/4] 生成低质量 GIF (最小文件)..."
OUTPUT_LQ="${OUTPUT_DIR}/gsa_vis_none_low.gif"

ffmpeg -i "${INPUT_VIDEO}" \
    -vf "fps=5,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=2" \
    -y "${OUTPUT_LQ}" 2>/dev/null

if [ -f "${OUTPUT_LQ}" ]; then
    SIZE_LQ=$(du -h "${OUTPUT_LQ}" | cut -f1)
    echo "   ✓ 完成: ${OUTPUT_LQ}"
    echo "   文件大小: ${SIZE_LQ}"
else
    echo "   ✗ 失败"
fi
echo ""

# ============================================================================
# 方案 4: 快速版 GIF (分辨率: 800x~, 帧率: 15fps) - 播放更流畅
# ============================================================================
echo "⚡ [4/4] 生成快速版 GIF (流畅播放)..."
OUTPUT_FAST="${OUTPUT_DIR}/gsa_vis_none_fast.gif"

ffmpeg -i "${INPUT_VIDEO}" \
    -vf "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
    -y "${OUTPUT_FAST}" 2>/dev/null

if [ -f "${OUTPUT_FAST}" ]; then
    SIZE_FAST=$(du -h "${OUTPUT_FAST}" | cut -f1)
    echo "   ✓ 完成: ${OUTPUT_FAST}"
    echo "   文件大小: ${SIZE_FAST}"
else
    echo "   ✗ 失败"
fi
echo ""

# ============================================================================
# 汇总结果
# ============================================================================
echo "================================================"
echo "✅ GIF 生成完成！"
echo "================================================"
echo ""
echo "生成的 GIF 文件："
echo ""

if [ -f "${OUTPUT_HQ}" ]; then
    echo "📌 高质量版 (1200x680, 10fps):"
    echo "   文件: ${OUTPUT_HQ}"
    echo "   大小: $(du -h "${OUTPUT_HQ}" | cut -f1)"
    echo "   适合: 全屏演示、高清展示"
    echo ""
fi

if [ -f "${OUTPUT_MQ}" ]; then
    echo "⭐ 中等质量版 (800x~, 10fps) [推荐]:"
    echo "   文件: ${OUTPUT_MQ}"
    echo "   大小: $(du -h "${OUTPUT_MQ}" | cut -f1)"
    echo "   适合: PPT 演示、文档嵌入"
    echo ""
fi

if [ -f "${OUTPUT_LQ}" ]; then
    echo "💡 低质量版 (600x~, 5fps):"
    echo "   文件: ${OUTPUT_LQ}"
    echo "   大小: $(du -h "${OUTPUT_LQ}" | cut -f1)"
    echo "   适合: 网页展示、邮件发送"
    echo ""
fi

if [ -f "${OUTPUT_FAST}" ]; then
    echo "⚡ 快速版 (800x~, 15fps):"
    echo "   文件: ${OUTPUT_FAST}"
    echo "   大小: $(du -h "${OUTPUT_FAST}" | cut -f1)"
    echo "   适合: 流畅演示、动态展示"
    echo ""
fi

echo "================================================"
echo "💡 使用建议"
echo "================================================"
echo ""
echo "对于 PPT 演示，推荐使用: gsa_vis_none_medium.gif"
echo ""
echo "插入 PPT 步骤："
echo "  1. 打开 PowerPoint"
echo "  2. 插入 -> 图片 -> 此设备"
echo "  3. 选择 GIF 文件"
echo "  4. GIF 会自动播放（幻灯片放映时）"
echo ""
echo "在 LibreOffice Impress 中："
echo "  1. 插入 -> 图像"
echo "  2. 选择 GIF 文件"
echo "  3. 右键 -> 交互 -> 播放动画"
echo ""

echo "================================================"
echo "🎬 预览 GIF"
echo "================================================"
echo ""
echo "在浏览器中预览："
echo "  firefox ${OUTPUT_MQ}"
echo ""
echo "在图像查看器中预览："
echo "  eog ${OUTPUT_MQ}  # Eye of GNOME"
echo "  gwenview ${OUTPUT_MQ}  # KDE"
echo ""

