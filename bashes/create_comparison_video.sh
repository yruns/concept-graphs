#!/bin/bash

################################################################################
# 创建左右对比视频：原始图像 vs 2D分割结果
################################################################################
#
# 作用：
#   - 左侧：原始 RGB 图像
#   - 右侧：2D 分割可视化结果
#   - 并排展示，便于对比
#
# 输入：
#   - 原始图像: $REPLICA_ROOT/$SCENE_NAME/results/frame*.jpg
#   - 分割结果: $REPLICA_ROOT/$SCENE_NAME/gsa_vis_none/frame*.jpg
#
# 输出：
#   - 对比视频: $REPLICA_ROOT/$SCENE_NAME/comparison_video.mp4
#   - 对比 GIF: $REPLICA_ROOT/$SCENE_NAME/comparison_*.gif
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=room0
FPS=10

# 路径设置
ORIGINAL_DIR="${REPLICA_ROOT}/${SCENE_NAME}/results"
SEGMENTATION_DIR="${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none"
OUTPUT_DIR="${REPLICA_ROOT}/${SCENE_NAME}"
TEMP_DIR="${OUTPUT_DIR}/temp_comparison"

echo "================================================"
echo "创建左右对比视频"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo ""
echo "输入:"
echo "  - 原始图像: ${ORIGINAL_DIR}/frame*.jpg"
echo "  - 分割结果: ${SEGMENTATION_DIR}/frame*.jpg"
echo ""
echo "输出:"
echo "  - 视频: ${OUTPUT_DIR}/comparison_video.mp4"
echo "  - GIF: ${OUTPUT_DIR}/comparison_*.gif"
echo "================================================"
echo ""

# 检查输入目录
if [ ! -d "${ORIGINAL_DIR}" ]; then
    echo "✗ 错误: 原始图像目录不存在: ${ORIGINAL_DIR}"
    exit 1
fi

if [ ! -d "${SEGMENTATION_DIR}" ]; then
    echo "✗ 错误: 分割结果目录不存在: ${SEGMENTATION_DIR}"
    echo "请先运行步骤 1: bash 1_extract_2d_segmentation.sh"
    exit 1
fi

# 创建临时目录
mkdir -p "${TEMP_DIR}"

# 统计文件数量
ORIGINAL_COUNT=$(ls ${ORIGINAL_DIR}/frame*.jpg 2>/dev/null | wc -l)
SEGMENTATION_COUNT=$(ls ${SEGMENTATION_DIR}/frame*.jpg 2>/dev/null | wc -l)

echo "找到原始图像: ${ORIGINAL_COUNT} 张"
echo "找到分割结果: ${SEGMENTATION_COUNT} 张"
echo ""

if [ ${SEGMENTATION_COUNT} -eq 0 ]; then
    echo "✗ 错误: 没有找到分割结果图像"
    exit 1
fi

# 使用 Python 脚本合并图像
echo "开始合并图像..."
echo ""

# 导出环境变量供 Python 使用
export ORIGINAL_DIR="${ORIGINAL_DIR}"
export SEGMENTATION_DIR="${SEGMENTATION_DIR}"
export TEMP_DIR="${TEMP_DIR}"

python3 << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import natsort

# 路径
original_dir = Path(os.environ['ORIGINAL_DIR'])
segmentation_dir = Path(os.environ['SEGMENTATION_DIR'])
temp_dir = Path(os.environ['TEMP_DIR'])

# 获取所有分割结果文件（因为分割结果可能是stride采样的）
seg_files = sorted(segmentation_dir.glob('frame*.jpg'))
seg_files = natsort.natsorted(seg_files)

print(f"处理 {len(seg_files)} 张图像...")

for idx, seg_path in enumerate(tqdm(seg_files, desc="合并图像")):
    # 获取对应的原始图像文件名
    frame_name = seg_path.name
    orig_path = original_dir / frame_name
    
    if not orig_path.exists():
        print(f"警告: 原始图像不存在: {orig_path}")
        continue
    
    # 读取图像
    img_orig = Image.open(orig_path)
    img_seg = Image.open(seg_path)
    
    # 确保两个图像高度相同（调整原始图像到分割图像的高度）
    if img_orig.size[1] != img_seg.size[1]:
        # 按比例缩放原始图像
        ratio = img_seg.size[1] / img_orig.size[1]
        new_width = int(img_orig.size[0] * ratio)
        img_orig = img_orig.resize((new_width, img_seg.size[1]), Image.LANCZOS)
    
    # 创建并排图像 (左: 原始, 右: 分割)
    total_width = img_orig.size[0] + img_seg.size[0]
    height = img_orig.size[1]
    
    combined = Image.new('RGB', (total_width, height))
    combined.paste(img_orig, (0, 0))
    combined.paste(img_seg, (img_orig.size[0], 0))
    
    # 添加文字标签
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    
    # 使用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # 添加标签背景
    label_height = 40
    # 左侧标签（使用英文避免乱码）
    draw.rectangle([(0, 0), (img_orig.size[0], label_height)], fill=(0, 0, 0, 180))
    draw.text((10, 5), "Original Image", fill=(255, 255, 255), font=font)
    
    # 右侧标签（使用英文避免乱码）
    draw.rectangle([(img_orig.size[0], 0), (total_width, label_height)], fill=(0, 0, 0, 180))
    draw.text((img_orig.size[0] + 10, 5), "2D Segmentation", fill=(255, 255, 255), font=font)
    
    # 保存合并后的图像
    output_path = temp_dir / f"comparison_{idx:06d}.jpg"
    combined.save(output_path, quality=95)

print("\n✓ 图像合并完成")
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 图像合并失败"
    exit 1
fi

echo ""
echo "生成对比视频..."
echo ""

# 生成视频
VIDEO_OUTPUT="${OUTPUT_DIR}/comparison_video.mp4"

ffmpeg -framerate ${FPS} \
    -pattern_type glob \
    -i "${TEMP_DIR}/comparison_*.jpg" \
    -c:v libopenh264 \
    -pix_fmt yuv420p \
    -y \
    "${VIDEO_OUTPUT}" 2>&1 | grep -E "frame=|video:"

if [ $? -ne 0 ]; then
    echo "尝试使用 mpeg4 编码器..."
    ffmpeg -framerate ${FPS} \
        -pattern_type glob \
        -i "${TEMP_DIR}/comparison_*.jpg" \
        -c:v mpeg4 \
        -q:v 5 \
        -y \
        "${VIDEO_OUTPUT}" 2>&1 | grep -E "frame=|video:"
fi

if [ -f "${VIDEO_OUTPUT}" ]; then
    VIDEO_SIZE=$(du -h "${VIDEO_OUTPUT}" | cut -f1)
    echo ""
    echo "✓ 对比视频生成成功！"
    echo "  文件: ${VIDEO_OUTPUT}"
    echo "  大小: ${VIDEO_SIZE}"
    echo ""
else
    echo ""
    echo "✗ 视频生成失败"
    exit 1
fi

# 生成 GIF (中等质量，适合 PPT)
echo "生成对比 GIF (适合 PPT)..."
echo ""

GIF_OUTPUT="${OUTPUT_DIR}/comparison_medium.gif"

ffmpeg -framerate ${FPS} \
    -pattern_type glob \
    -i "${TEMP_DIR}/comparison_*.jpg" \
    -vf "fps=${FPS},scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
    -y "${GIF_OUTPUT}" 2>&1 | grep -E "frame=|Output"

if [ -f "${GIF_OUTPUT}" ]; then
    GIF_SIZE=$(du -h "${GIF_OUTPUT}" | cut -f1)
    echo ""
    echo "✓ 对比 GIF 生成成功！"
    echo "  文件: ${GIF_OUTPUT}"
    echo "  大小: ${GIF_SIZE}"
    echo ""
fi

# 可选：生成低质量 GIF (更小)
echo "生成对比 GIF (低质量版，文件更小)..."
echo ""

GIF_LOW_OUTPUT="${OUTPUT_DIR}/comparison_low.gif"

ffmpeg -framerate 5 \
    -pattern_type glob \
    -i "${TEMP_DIR}/comparison_*.jpg" \
    -vf "fps=5,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=2" \
    -y "${GIF_LOW_OUTPUT}" 2>&1 | grep -E "frame=|Output"

if [ -f "${GIF_LOW_OUTPUT}" ]; then
    GIF_LOW_SIZE=$(du -h "${GIF_LOW_OUTPUT}" | cut -f1)
    echo ""
    echo "✓ 对比 GIF (低质量) 生成成功！"
    echo "  文件: ${GIF_LOW_OUTPUT}"
    echo "  大小: ${GIF_LOW_SIZE}"
    echo ""
fi

# 清理临时文件
echo "清理临时文件..."
rm -rf "${TEMP_DIR}"
echo ""

# 汇总
echo "================================================"
echo "✅ 全部完成！"
echo "================================================"
echo ""
echo "生成的文件:"
echo ""

if [ -f "${VIDEO_OUTPUT}" ]; then
    echo "📹 对比视频:"
    echo "   ${VIDEO_OUTPUT}"
    echo "   大小: $(du -h "${VIDEO_OUTPUT}" | cut -f1)"
    echo ""
fi

if [ -f "${GIF_OUTPUT}" ]; then
    echo "🎨 对比 GIF (中等质量，推荐用于PPT):"
    echo "   ${GIF_OUTPUT}"
    echo "   大小: $(du -h "${GIF_OUTPUT}" | cut -f1)"
    echo ""
fi

if [ -f "${GIF_LOW_OUTPUT}" ]; then
    echo "💾 对比 GIF (低质量，最小文件):"
    echo "   ${GIF_LOW_OUTPUT}"
    echo "   大小: $(du -h "${GIF_LOW_OUTPUT}" | cut -f1)"
    echo ""
fi

echo "================================================"
echo "播放/预览命令:"
echo "================================================"
echo ""
echo "视频:"
echo "  vlc ${VIDEO_OUTPUT}"
echo "  ffplay ${VIDEO_OUTPUT}"
echo ""
echo "GIF:"
echo "  firefox ${GIF_OUTPUT}"
echo "  eog ${GIF_OUTPUT}"
echo ""

