#!/bin/bash

################################################################################
# 从已有的 2D 分割可视化图像生成视频
################################################################################
#
# 作用：
#   - 从已经生成的 2D 分割可视化图像创建视频
#   - 适用于已经运行过步骤 1 的情况
#   - 无需重新运行耗时的分割过程
#
# 输入：
#   - 可视化图像: $REPLICA_ROOT/$SCENE_NAME/gsa_vis_none/*.jpg
#
# 输出：
#   - 视频: $REPLICA_ROOT/$SCENE_NAME/gsa_vis_none.mp4
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=room0
FPS=10  # 帧率：每秒显示多少帧

echo "================================================"
echo "从已有图像生成 2D 分割视频"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "帧率: ${FPS} fps"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/*.jpg"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.mp4"
echo "================================================"
echo ""

# 检查输入目录是否存在
if [ ! -d "${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none" ]; then
    echo "✗ 错误: 可视化图像目录不存在"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none"
    echo ""
    echo "请先运行步骤 1 (1_extract_2d_segmentation.sh)"
    exit 1
fi

# 检查是否有图像文件
IMAGE_COUNT=$(ls ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/*.jpg 2>/dev/null | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then
    echo "✗ 错误: 没有找到可视化图像"
    echo "   ${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/*.jpg"
    echo ""
    echo "请先运行步骤 1 生成可视化图像"
    exit 1
fi

echo "找到 ${IMAGE_COUNT} 张图像"
echo ""

# 使用 ffmpeg 生成视频
# 注意: 需要安装 ffmpeg (sudo apt install ffmpeg)
OUTPUT_PATH="${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none.mp4"

if command -v ffmpeg &> /dev/null; then
    echo "使用 ffmpeg 生成视频..."
    echo ""
    
    # 使用 ffmpeg 从图像序列创建视频
    # -framerate: 输入帧率
    # -pattern_type glob: 使用通配符匹配文件
    # -i: 输入文件模式
    # -c:v libx264: 使用 H.264 编码
    # -pix_fmt yuv420p: 像素格式 (兼容性好)
    # -vf: 视频滤镜
    # -y: 覆盖输出文件
    
    # 尝试使用不同的编码器
    # 1. 尝试 libopenh264 (通常在conda环境中可用)
    ffmpeg -framerate ${FPS} \
        -pattern_type glob \
        -i "${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/*.jpg" \
        -c:v libopenh264 \
        -pix_fmt yuv420p \
        -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        -y \
        "${OUTPUT_PATH}" 2>/dev/null
    
    # 如果 libopenh264 失败，尝试 mpeg4
    if [ $? -ne 0 ]; then
        echo "libopenh264 不可用，尝试 mpeg4..."
        ffmpeg -framerate ${FPS} \
            -pattern_type glob \
            -i "${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none/*.jpg" \
            -c:v mpeg4 \
            -pix_fmt yuv420p \
            -q:v 5 \
            -y \
            "${OUTPUT_PATH}"
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 视频生成成功！"
        echo ""
        echo "视频文件: ${OUTPUT_PATH}"
        echo "总帧数: ${IMAGE_COUNT}"
        echo "帧率: ${FPS} fps"
        echo "时长: $(echo "scale=2; ${IMAGE_COUNT}/${FPS}" | bc) 秒"
        echo ""
        echo "播放视频命令:"
        echo "  vlc ${OUTPUT_PATH}"
        echo "  # 或"
        echo "  ffplay ${OUTPUT_PATH}"
        echo ""
    else
        echo ""
        echo "✗ ffmpeg 生成视频失败"
        exit 1
    fi
else
    echo "✗ 错误: 未找到 ffmpeg"
    echo ""
    echo "请安装 ffmpeg:"
    echo "  sudo apt install ffmpeg"
    echo ""
    echo "或者使用 Python 脚本生成视频 (运行下面的命令):"
    echo ""
    echo "  python3 << 'PYTHON_SCRIPT'"
    echo "import imageio"
    echo "import glob"
    echo "import natsort"
    echo "from pathlib import Path"
    echo ""
    echo "vis_dir = Path('${REPLICA_ROOT}/${SCENE_NAME}/gsa_vis_none')"
    echo "output = Path('${OUTPUT_PATH}')"
    echo ""
    echo "images = glob.glob(str(vis_dir / '*.jpg'))"
    echo "images = natsort.natsorted(images)"
    echo ""
    echo "print(f'找到 {len(images)} 张图像')"
    echo "frames = [imageio.imread(img) for img in images]"
    echo "imageio.mimsave(output, frames, fps=${FPS})"
    echo "print(f'视频已保存: {output}')"
    echo "PYTHON_SCRIPT"
    echo ""
    exit 1
fi

