"""
查询结果可视化
==============

支持:
- 3D点云可视化 (PLY输出)
- 目标物体高亮显示
- 参照物显示
- 场景全景显示
"""

import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 颜色定义
COLORS = {
    "target": [255, 0, 0],       # 红色 - 目标物体
    "reference": [0, 255, 0],    # 绿色 - 参照物
    "context": [0, 0, 255],      # 蓝色 - 上下文物体
    "background": [180, 180, 180], # 灰色 - 背景
}


def visualize_query_result(
    pcd_file: str,
    target_ids: List[int],
    reference_ids: List[int] = None,
    context_ids: List[int] = None,
    output_path: str = None,
    show_all: bool = True,
) -> str:
    """
    可视化查询结果
    
    Args:
        pcd_file: 原始pcd文件路径
        target_ids: 目标物体ID列表
        reference_ids: 参照物ID列表
        context_ids: 上下文物体ID列表
        output_path: 输出PLY文件路径
        show_all: 是否显示所有物体 (False则只显示相关物体)
    
    Returns:
        输出文件路径
    """
    reference_ids = reference_ids or []
    context_ids = context_ids or []
    
    # 加载原始数据
    with gzip.open(pcd_file, 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    
    # 收集点云和颜色
    all_points = []
    all_colors = []
    
    for i, obj in enumerate(objects):
        pcd_np = obj.get('pcd_np')
        if pcd_np is None or len(pcd_np) == 0:
            continue
        
        # 只取xyz
        if pcd_np.shape[1] > 3:
            points = pcd_np[:, :3]
        else:
            points = pcd_np
        
        # 确定颜色
        if i in target_ids:
            color = COLORS["target"]
        elif i in reference_ids:
            color = COLORS["reference"]
        elif i in context_ids:
            color = COLORS["context"]
        else:
            if not show_all:
                continue
            color = COLORS["background"]
        
        all_points.append(points)
        all_colors.append(np.tile(color, (len(points), 1)))
    
    if not all_points:
        print("No points to visualize")
        return None
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors).astype(np.uint8)
    
    # 确定输出路径
    if output_path is None:
        pcd_dir = Path(pcd_file).parent.parent
        output_path = str(pcd_dir / "query_result.ply")
    
    # 写入PLY
    write_ply(output_path, all_points, all_colors)
    
    print(f"Visualization saved to: {output_path}")
    print(f"  Total points: {len(all_points)}")
    print(f"  Target objects (RED): {target_ids}")
    print(f"  Reference objects (GREEN): {reference_ids}")
    
    return output_path


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """写入PLY文件"""
    n_points = len(points)
    
    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(path, 'w') as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def visualize_all_objects(
    pcd_file: str,
    output_path: str = None,
    color_by: str = "id",
) -> str:
    """
    可视化所有物体 (不同颜色区分)
    
    Args:
        pcd_file: pcd文件路径
        output_path: 输出路径
        color_by: 着色方式 ("id" 或 "tag")
    """
    with gzip.open(pcd_file, 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    
    # 生成颜色调色板
    n_objects = len(objects)
    palette = generate_color_palette(n_objects)
    
    all_points = []
    all_colors = []
    
    for i, obj in enumerate(objects):
        pcd_np = obj.get('pcd_np')
        if pcd_np is None or len(pcd_np) == 0:
            continue
        
        if pcd_np.shape[1] > 3:
            points = pcd_np[:, :3]
        else:
            points = pcd_np
        
        color = palette[i % len(palette)]
        all_points.append(points)
        all_colors.append(np.tile(color, (len(points), 1)))
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors).astype(np.uint8)
    
    if output_path is None:
        pcd_dir = Path(pcd_file).parent.parent
        output_path = str(pcd_dir / "all_objects.ply")
    
    write_ply(output_path, all_points, all_colors)
    print(f"Saved all objects visualization to: {output_path}")
    
    return output_path


def generate_color_palette(n: int) -> List[List[int]]:
    """生成n个不同的颜色"""
    colors = []
    for i in range(n):
        h = i / n
        r, g, b = hsv_to_rgb(h, 0.8, 0.9)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """HSV转RGB"""
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def visualize_topdown(
    pcd_file: str,
    target_ids: List[int],
    reference_ids: List[int] = None,
    output_path: str = None,
    labels: Dict[int, str] = None,
) -> str:
    """
    生成俯视图2D可视化 (PNG)
    
    Args:
        pcd_file: pcd文件路径
        target_ids: 目标物体ID
        reference_ids: 参照物ID
        output_path: 输出PNG路径
        labels: 物体标签字典 {id: tag}
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except ImportError:
        print("matplotlib not available, skipping topdown visualization")
        return None
    
    reference_ids = reference_ids or []
    labels = labels or {}
    
    # 加载数据
    with gzip.open(pcd_file, 'rb') as f:
        data = pickle.load(f)
    
    objects = data.get('objects', [])
    
    # 创建图
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 计算场景边界
    all_positions = []
    for obj in objects:
        pcd_np = obj.get('pcd_np')
        if pcd_np is not None and len(pcd_np) > 0:
            pos = pcd_np.mean(axis=0)
            all_positions.append(pos[:2])
    
    if not all_positions:
        return None
    
    all_positions = np.array(all_positions)
    x_min, y_min = all_positions.min(axis=0) - 0.5
    x_max, y_max = all_positions.max(axis=0) + 0.5
    
    # 绘制所有物体
    for i, obj in enumerate(objects):
        pcd_np = obj.get('pcd_np')
        if pcd_np is None or len(pcd_np) == 0:
            continue
        
        pos = pcd_np.mean(axis=0)[:2]
        bbox_min = pcd_np.min(axis=0)[:2]
        bbox_max = pcd_np.max(axis=0)[:2]
        
        # 确定颜色和大小
        if i in target_ids:
            color = 'red'
            alpha = 0.8
            zorder = 10
            size = 150
        elif i in reference_ids:
            color = 'green'
            alpha = 0.7
            zorder = 5
            size = 120
        else:
            color = 'gray'
            alpha = 0.3
            zorder = 1
            size = 50
        
        # 绘制点
        ax.scatter(pos[0], pos[1], c=color, s=size, alpha=alpha, zorder=zorder)
        
        # 绘制边界框
        width = bbox_max[0] - bbox_min[0]
        height = bbox_max[1] - bbox_min[1]
        rect = Rectangle((bbox_min[0], bbox_min[1]), width, height,
                         fill=False, edgecolor=color, alpha=alpha*0.5, linewidth=1)
        ax.add_patch(rect)
        
        # 添加标签
        if i in target_ids or i in reference_ids:
            label = labels.get(i, f"obj_{i}")
            ax.annotate(f"{label}\n(id={i})", (pos[0], pos[1]), 
                       fontsize=9, ha='center', va='bottom',
                       color=color, fontweight='bold')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Query Result - Top-down View\n(RED=target, GREEN=reference)')
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Target'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Reference'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Background'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 保存
    if output_path is None:
        pcd_dir = Path(pcd_file).parent.parent
        output_path = str(pcd_dir / "query_result_topdown.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Top-down view saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("pcd_file", help="Input pcd file")
    parser.add_argument("--targets", type=int, nargs="+", default=[], help="Target object IDs")
    parser.add_argument("--refs", type=int, nargs="+", default=[], help="Reference object IDs")
    parser.add_argument("--output", "-o", help="Output PLY file")
    parser.add_argument("--all-objects", action="store_true", help="Visualize all objects")
    
    args = parser.parse_args()
    
    if args.all_objects:
        visualize_all_objects(args.pcd_file, args.output)
    else:
        visualize_query_result(
            args.pcd_file,
            target_ids=args.targets,
            reference_ids=args.refs,
            output_path=args.output,
        )
