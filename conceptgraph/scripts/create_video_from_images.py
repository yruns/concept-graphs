#!/usr/bin/env python3
"""
从已有的 2D 分割可视化图像生成视频

使用方法:
    python scripts/create_video_from_images.py \
        --image_dir /path/to/gsa_vis_none \
        --output_path /path/to/output.mp4 \
        --fps 10
"""

import argparse
import glob
from pathlib import Path
import natsort
import imageio
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="从图像序列生成视频")
    
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="可视化图像目录路径"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出视频路径 (默认: image_dir.mp4)"
    )
    
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="*.jpg",
        help="图像文件匹配模式 (默认: *.jpg)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="视频帧率 (默认: 10)"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        help="视频质量 (1-10, 10最高) (默认: 8)"
    )
    
    return parser


def main(args):
    image_dir = Path(args.image_dir)
    
    # 检查目录是否存在
    if not image_dir.exists():
        print(f"❌ 错误: 目录不存在: {image_dir}")
        return
    
    # 查找所有图像
    image_pattern = str(image_dir / args.image_pattern)
    images = glob.glob(image_pattern)
    images = natsort.natsorted(images)
    
    if len(images) == 0:
        print(f"❌ 错误: 没有找到匹配的图像")
        print(f"   搜索路径: {image_pattern}")
        return
    
    print(f"✅ 找到 {len(images)} 张图像")
    
    # 设置输出路径
    if args.output_path is None:
        # 默认在父目录生成视频文件
        output_path = image_dir.parent / f"{image_dir.name}.mp4"
    else:
        output_path = Path(args.output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📹 开始生成视频...")
    print(f"   输出路径: {output_path}")
    print(f"   帧率: {args.fps} fps")
    print(f"   时长: {len(images) / args.fps:.2f} 秒")
    print()
    
    # 读取图像并生成视频
    frames = []
    for img_path in tqdm(images, desc="读取图像"):
        frame = imageio.imread(img_path)
        frames.append(frame)
    
    print(f"💾 保存视频...")
    
    # 使用项目中验证过的方法保存视频
    # mimwrite 比 mimsave 更适合视频格式
    imageio.mimwrite(output_path, frames, fps=args.fps)
    
    print()
    print(f"✅ 视频生成成功!")
    print(f"   文件: {output_path}")
    print(f"   大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print(f"播放视频命令:")
    print(f"   vlc {output_path}")
    print(f"   # 或")
    print(f"   ffplay {output_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

