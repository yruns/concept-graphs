#!/usr/bin/env python3
"""
区域可视化生成器

为每个分段区域生成:
1. 关键帧图像
2. 区域 GIF 动画
3. 区域概览拼图
"""

import numpy as np
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio
from natsort import natsorted


def load_segments(segments_path):
    """加载分段结果"""
    with open(segments_path, 'r') as f:
        return json.load(f)


def get_image_files(image_dir, extension='jpg'):
    """获取图像文件列表"""
    image_dir = Path(image_dir)
    files = list(image_dir.glob('*.%s' % extension))
    if not files:
        files = list(image_dir.glob('*.png'))
    return natsorted(files)


def add_text_overlay(image, text, position='top', font_size=24):
    """在图像上添加文字标注"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 计算文字位置
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    if position == 'top':
        x = (img.width - text_width) // 2
        y = 10
    else:
        x = (img.width - text_width) // 2
        y = img.height - text_height - 10
    
    # 绘制背景框
    padding = 5
    draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                   fill=(0, 0, 0, 180))
    
    # 绘制文字
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    
    return img


def generate_region_keyframes(segments, image_files, output_dir, stride=5, n_keyframes=3):
    """为每个区域生成关键帧图像"""
    output_dir = Path(output_dir)
    keyframes_dir = output_dir / 'region_keyframes'
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    
    print('Generating keyframes for %d regions...' % len(segments))
    
    all_keyframes = []
    
    for seg in segments:
        region_id = seg['region_id']
        start_frame = seg['start_frame']
        end_frame = seg['end_frame']
        n_frames = seg['n_frames']
        
        # 计算关键帧索引（均匀采样）
        if n_frames <= n_keyframes:
            keyframe_indices = list(range(start_frame, end_frame))
        else:
            step = n_frames // n_keyframes
            keyframe_indices = [start_frame + i * step for i in range(n_keyframes)]
        
        # 转换为原始帧索引
        original_indices = [idx * stride for idx in keyframe_indices]
        
        region_keyframes = []
        
        for i, (seg_idx, orig_idx) in enumerate(zip(keyframe_indices, original_indices)):
            if orig_idx < len(image_files):
                img_path = image_files[orig_idx]
                img = Image.open(img_path)
                
                # 添加标注
                text = 'Region %d | Frame %d' % (region_id, seg_idx)
                img_annotated = add_text_overlay(img, text)
                
                # 保存关键帧
                output_path = keyframes_dir / ('region_%02d_keyframe_%d.jpg' % (region_id, i))
                img_annotated.save(output_path, quality=90)
                
                region_keyframes.append(img_annotated)
        
        all_keyframes.append((region_id, region_keyframes))
        print('  Region %d: %d keyframes' % (region_id, len(region_keyframes)))
    
    return all_keyframes, keyframes_dir


def generate_region_gifs(segments, image_files, output_dir, stride=5, fps=5, max_frames=30):
    """为每个区域生成 GIF 动画"""
    output_dir = Path(output_dir)
    gifs_dir = output_dir / 'region_gifs'
    gifs_dir.mkdir(parents=True, exist_ok=True)
    
    print('Generating GIFs for %d regions...' % len(segments))
    
    for seg in segments:
        region_id = seg['region_id']
        start_frame = seg['start_frame']
        end_frame = seg['end_frame']
        n_frames = min(seg['n_frames'], max_frames)
        
        # 计算采样步长
        total_seg_frames = seg['n_frames']
        if total_seg_frames > max_frames:
            sample_step = total_seg_frames // max_frames
        else:
            sample_step = 1
        
        frames = []
        frame_indices = range(start_frame, end_frame, sample_step)
        
        for seg_idx in frame_indices:
            orig_idx = seg_idx * stride
            if orig_idx < len(image_files):
                img_path = image_files[orig_idx]
                img = Image.open(img_path)
                
                # 缩小尺寸以减小GIF大小
                img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                # 添加标注
                text = 'Region %d | Frame %d/%d' % (region_id, seg_idx - start_frame, total_seg_frames)
                img_annotated = add_text_overlay(img, text)
                
                frames.append(np.array(img_annotated))
        
        if frames:
            # 保存 GIF
            gif_path = gifs_dir / ('region_%02d.gif' % region_id)
            imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
            print('  Region %d: %d frames -> %s' % (region_id, len(frames), gif_path.name))
    
    return gifs_dir


def generate_overview_grid(all_keyframes, output_path, cols=4):
    """生成所有区域的概览拼图"""
    if not all_keyframes:
        return
    
    # 获取第一个关键帧的尺寸
    first_img = all_keyframes[0][1][0] if all_keyframes[0][1] else None
    if first_img is None:
        return
    
    thumb_width = 320
    thumb_height = 240
    
    n_regions = len(all_keyframes)
    rows = (n_regions + cols - 1) // cols
    
    # 创建拼图
    grid_width = cols * thumb_width
    grid_height = rows * thumb_height
    grid_img = Image.new('RGB', (grid_width, grid_height), (30, 30, 30))
    
    for i, (region_id, keyframes) in enumerate(all_keyframes):
        if not keyframes:
            continue
        
        # 使用第一个关键帧
        img = keyframes[0].copy()
        img.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        
        # 计算位置
        row = i // cols
        col = i % cols
        x = col * thumb_width + (thumb_width - img.width) // 2
        y = row * thumb_height + (thumb_height - img.height) // 2
        
        grid_img.paste(img, (x, y))
    
    grid_img.save(output_path, quality=95)
    print('Overview grid saved to: %s' % output_path)


def generate_trajectory_gif(segments, image_files, output_path, stride=5, fps=10, sample_every=2):
    """生成完整轨迹的 GIF，用颜色标注区域"""
    print('Generating full trajectory GIF...')
    
    # 为每个区域分配颜色
    import colorsys
    n_regions = len(segments)
    colors = []
    for i in range(n_regions):
        hue = i / n_regions
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    frames = []
    
    for seg in segments:
        region_id = seg['region_id']
        start_frame = seg['start_frame']
        end_frame = seg['end_frame']
        color = colors[region_id]
        
        for seg_idx in range(start_frame, end_frame, sample_every):
            orig_idx = seg_idx * stride
            if orig_idx < len(image_files):
                img = Image.open(image_files[orig_idx])
                img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                # 添加彩色边框表示区域
                bordered = Image.new('RGB', (img.width + 10, img.height + 10), color)
                bordered.paste(img, (5, 5))
                
                # 添加文字
                text = 'Region %d' % region_id
                bordered = add_text_overlay(bordered, text)
                
                frames.append(np.array(bordered))
    
    if frames:
        imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
        print('Full trajectory GIF saved to: %s (%d frames)' % (output_path, len(frames)))


def main():
    parser = argparse.ArgumentParser(description='区域可视化生成器')
    parser.add_argument('--segments', type=str, required=True, help='分段结果JSON')
    parser.add_argument('--image_dir', type=str, required=True, help='原始图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--stride', type=int, default=5, help='帧步长')
    parser.add_argument('--fps', type=int, default=5, help='GIF帧率')
    parser.add_argument('--n_keyframes', type=int, default=3, help='每区域关键帧数')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print('Region Visualization Generator')
    print('=' * 60)
    
    # 加载数据
    segments = load_segments(args.segments)
    print('Loaded %d segments' % len(segments))
    
    image_files = get_image_files(args.image_dir)
    print('Found %d images' % len(image_files))
    
    if not image_files:
        print('Error: No images found in %s' % args.image_dir)
        return
    
    # 生成关键帧
    print('\n--- Generating Keyframes ---')
    all_keyframes, keyframes_dir = generate_region_keyframes(
        segments, image_files, output_dir, 
        stride=args.stride, n_keyframes=args.n_keyframes
    )
    
    # 生成区域 GIF
    print('\n--- Generating Region GIFs ---')
    gifs_dir = generate_region_gifs(
        segments, image_files, output_dir,
        stride=args.stride, fps=args.fps
    )
    
    # 生成概览拼图
    print('\n--- Generating Overview Grid ---')
    generate_overview_grid(
        all_keyframes, 
        output_dir / 'regions_overview.jpg'
    )
    
    # 生成完整轨迹 GIF
    print('\n--- Generating Full Trajectory GIF ---')
    generate_trajectory_gif(
        segments, image_files,
        output_dir / 'full_trajectory.gif',
        stride=args.stride, fps=10, sample_every=3
    )
    
    print('\n' + '=' * 60)
    print('Visualization complete!')
    print('Output directory: %s' % output_dir)
    print('=' * 60)


if __name__ == '__main__':
    main()
