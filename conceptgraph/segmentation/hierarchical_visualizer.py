#!/usr/bin/env python3
"""
层次化场景图可视化器
==================

支持多种可视化模式：
1. 静态图片：生成PNG展示面板
2. 3D点云PLY：按区域着色的点云导出
3. 关键帧可视化：显示关键帧及物体标注
"""
import json
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import distinctipy
import cv2
from PIL import Image

from .data_structures import HierarchicalSceneGraph, ObjectRegionRelation


class HierarchicalSceneVisualizer:
    """层次化场景可视化器"""
    
    def __init__(self, scene_graph: HierarchicalSceneGraph):
        self.sg = scene_graph
        self.zone_colors = self._generate_zone_colors()
        self.relation_icons = {
            ObjectRegionRelation.DEFINING: "★",
            ObjectRegionRelation.SUPPORTING: "●",
            ObjectRegionRelation.SHARED: "◆",
            ObjectRegionRelation.BOUNDARY: "▲"
        }
    
    def _generate_zone_colors(self) -> Dict[str, tuple]:
        """为每个区域生成不同颜色"""
        n_zones = len(self.sg.functional_zones)
        if n_zones == 0:
            return {}
        colors = distinctipy.get_colors(n_zones, pastel_factor=0.5)
        return {z.zone_id: colors[i] for i, z in enumerate(self.sg.functional_zones)}
    
    def generate_dashboard(self, output_path: str, figsize=(16, 12)):
        """生成4合1展示面板"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        # 左上：俯视图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_topdown_zones(ax1)
        
        # 右上：层次树
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_hierarchy_tree(ax2)
        
        # 左下：统计
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_statistics(ax3)
        
        # 右下：物体-区域关系
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_relation_distribution(ax4)
        
        plt.suptitle(f"Hierarchical Scene Graph: {self.sg.scene_id}", fontsize=14, fontweight='bold')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Dashboard saved to: {output_path}")
    
    def _plot_topdown_zones(self, ax):
        """绘制功能区域俯视图"""
        ax.set_title("Functional Zones (Top-Down View)", fontsize=11)
        
        for zone in self.sg.functional_zones:
            color = self.zone_colors.get(zone.zone_id, (0.5, 0.5, 0.5))
            
            # 绘制物体位置
            for obj in zone.objects:
                if obj.position:
                    marker = 's' if obj.relation_type == ObjectRegionRelation.DEFINING else 'o'
                    size = 100 if obj.relation_type == ObjectRegionRelation.DEFINING else 50
                    ax.scatter(obj.position[0], obj.position[1], c=[color], s=size, 
                              marker=marker, edgecolors='black', linewidths=0.5, alpha=0.8)
            
            # 绘制区域中心和标签
            if zone.spatial:
                center = zone.spatial.center
                ax.annotate(zone.zone_name.replace('_', '\n'), xy=(center[0], center[1]),
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
    
    def _plot_hierarchy_tree(self, ax):
        """绘制层次结构树"""
        ax.set_title("Scene Hierarchy", fontsize=11)
        ax.axis('off')
        
        y = 0.95
        line_height = 0.08
        
        for unit in self.sg.spatial_units:
            ax.text(0.05, y, f"📍 {unit.unit_name}", fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
            y -= line_height * 0.8
            
            for zone_id in unit.functional_zones:
                zone = self.sg.get_zone_by_id(zone_id)
                if zone:
                    color = self.zone_colors.get(zone_id, (0.5, 0.5, 0.5))
                    n_objs = len(zone.objects)
                    n_def = len([o for o in zone.objects if o.relation_type == ObjectRegionRelation.DEFINING])
                    
                    ax.text(0.1, y, f"└─ 🎯 {zone.zone_name}", fontsize=9, color='black',
                           transform=ax.transAxes)
                    ax.text(0.5, y, f"[{n_objs} objs, {n_def}★]", fontsize=8, color='gray',
                           transform=ax.transAxes)
                    
                    # 显示前几个物体
                    y -= line_height * 0.6
                    objs_shown = 0
                    for obj in zone.objects[:4]:
                        icon = self.relation_icons.get(obj.relation_type, "○")
                        ax.text(0.15, y, f"   {icon} {obj.object_tag}", fontsize=8,
                               transform=ax.transAxes)
                        y -= line_height * 0.5
                        objs_shown += 1
                    
                    if len(zone.objects) > 4:
                        ax.text(0.15, y, f"   ... +{len(zone.objects)-4} more", fontsize=7, color='gray',
                               transform=ax.transAxes)
                        y -= line_height * 0.5
                    
                    y -= line_height * 0.3
    
    def _plot_statistics(self, ax):
        """绘制统计图表"""
        ax.set_title("Objects per Zone", fontsize=11)
        
        if not self.sg.functional_zones:
            ax.text(0.5, 0.5, "No zones", ha='center', va='center')
            return
        
        zone_names = [z.zone_name for z in self.sg.functional_zones]
        obj_counts = [len(z.objects) for z in self.sg.functional_zones]
        colors = [self.zone_colors.get(z.zone_id, (0.5, 0.5, 0.5)) for z in self.sg.functional_zones]
        
        y_pos = np.arange(len(zone_names))
        ax.barh(y_pos, obj_counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n.replace('_zone', '') for n in zone_names], fontsize=9)
        ax.set_xlabel('Number of Objects')
        ax.invert_yaxis()
    
    def _plot_relation_distribution(self, ax):
        """绘制物体-区域关系分布"""
        ax.set_title("Object-Zone Relation Types", fontsize=11)
        
        relation_counts = {r: 0 for r in ObjectRegionRelation}
        for zone in self.sg.functional_zones:
            for obj in zone.objects:
                relation_counts[obj.relation_type] += 1
        
        labels = [r.value for r in relation_counts.keys()]
        sizes = list(relation_counts.values())
        
        if sum(sizes) == 0:
            ax.text(0.5, 0.5, "No objects", ha='center', va='center')
            return
        
        colors_pie = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        explode = [0.05] * len(labels)
        
        ax.pie(sizes, labels=labels, colors=colors_pie, explode=explode,
               autopct=lambda p: f'{p:.0f}%' if p > 5 else '',
               startangle=90, textprops={'fontsize': 9})
        ax.axis('equal')
    
    def generate_topdown_map(self, output_path: str, figsize=(10, 10)):
        """生成单独的俯视图"""
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_topdown_zones(ax)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Topdown map saved to: {output_path}")
    
    def generate_3d_ply(self, pcd_file: str, output_path: str):
        """Generate zone-colored 3D point cloud PLY file (binary format)"""
        
        # Load point cloud data
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        objects = data.get('objects', [])
        
        if not objects:
            print("Warning: No object data")
            return
        
        # Build object-to-zone mapping
        obj_to_zone = {}
        for zone in self.sg.functional_zones:
            for obj in zone.objects:
                obj_to_zone[obj.object_id] = zone.zone_id
        
        # Merge all point clouds with colors
        all_points = []
        all_colors = []
        
        # Default color for unassigned (gray)
        default_color = np.array([128, 128, 128], dtype=np.uint8)
        
        for i, obj in enumerate(objects):
            pcd_np = obj.get('pcd_np', None)
            if pcd_np is None or len(pcd_np) == 0:
                continue
            
            # Get zone color (convert to 0-255 uint8)
            zone_id = obj_to_zone.get(i)
            if zone_id and zone_id in self.zone_colors:
                color = (np.array(self.zone_colors[zone_id]) * 255).astype(np.uint8)
            else:
                color = default_color
            
            all_points.append(pcd_np.astype(np.float32))
            all_colors.append(np.tile(color, (len(pcd_np), 1)))
        
        if not all_points:
            print("Warning: No point cloud data")
            return
        
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        # Write PLY file (binary format for better compatibility)
        self._write_ply_binary(output_path, points, colors)
        print(f"3D PLY saved to: {output_path} ({len(points)} points)")
        
        # Generate color legend
        self._generate_color_legend(output_path.replace('.ply', '_legend.png'))
    
    def _write_ply_binary(self, output_path: str, points: np.ndarray, colors: np.ndarray):
        """Write PLY file in binary format"""
        import struct
        import sys
        
        n_points = len(points)
        
        # Determine endianness
        if sys.byteorder == 'little':
            ply_format = 'binary_little_endian'
        else:
            ply_format = 'binary_big_endian'
        
        with open(output_path, 'wb') as f:
            # Write header
            header = f"""ply
format {ply_format} 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write binary data
            for i in range(n_points):
                # Pack: 3 floats + 3 unsigned chars
                f.write(struct.pack('fffBBB', 
                    float(points[i, 0]), float(points[i, 1]), float(points[i, 2]),
                    int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
    
    def _generate_color_legend(self, output_path: str):
        """Generate zone color legend"""
        n_zones = len(self.sg.functional_zones)
        if n_zones == 0:
            return
        
        fig, ax = plt.subplots(figsize=(8, max(2, n_zones * 0.6)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_zones + 1)
        ax.axis('off')
        ax.set_title("Zone Color Legend", fontsize=12, fontweight='bold')
        
        for i, zone in enumerate(self.sg.functional_zones):
            y = n_zones - i
            color = self.zone_colors.get(zone.zone_id, (0.5, 0.5, 0.5))
            ax.add_patch(plt.Rectangle((0.02, y - 0.4), 0.08, 0.8, facecolor=color, edgecolor='black'))
            # Use zone_name (English) instead of primary_activity (may contain Chinese)
            ax.text(0.12, y, f"{zone.zone_name} ({len(zone.objects)} objects)", fontsize=10, va='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Color legend saved to: {output_path}")
    
    def generate_keyframe_visualization(self, scene_path: str, pcd_file: str, output_dir: str, 
                                        n_keyframes: int = 6, stride: int = 5):
        """Generate keyframe visualization with object annotations"""
        scene_path = Path(scene_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load point cloud data
        with gzip.open(pcd_file, 'rb') as f:
            data = pickle.load(f)
        objects = data.get('objects', [])
        
        # Build object-to-zone mapping
        obj_to_zone = {}
        for zone in self.sg.functional_zones:
            for obj in zone.objects:
                obj_to_zone[obj.object_id] = (zone.zone_id, zone.zone_name)
        
        # Find RGB image directory - prefer frame*.jpg files (not depth*.png)
        rgb_dir = scene_path / 'results'
        if not rgb_dir.exists():
            rgb_dir = scene_path / 'rgb'
        
        if not rgb_dir.exists():
            print(f"Warning: RGB directory not found")
            return
        
        # Get RGB image list (prioritize frame*.jpg, exclude depth images)
        image_files = sorted(rgb_dir.glob('frame*.jpg'))
        if not image_files:
            image_files = sorted(rgb_dir.glob('rgb*.png'))
        if not image_files:
            # Fallback: get all images but exclude depth
            all_images = sorted(rgb_dir.glob('*.jpg')) + sorted(rgb_dir.glob('*.png'))
            image_files = [f for f in all_images if 'depth' not in f.name.lower()]
        
        if not image_files:
            print("Warning: No RGB images found")
            return
        
        # 按stride选取关键帧
        total_frames = len(image_files)
        keyframe_indices = np.linspace(0, total_frames - 1, n_keyframes, dtype=int)
        
        # 为每个关键帧生成可视化
        keyframe_images = []
        for idx in keyframe_indices:
            img_path = image_files[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 查找此帧中可见的物体
            visible_objects = self._find_visible_objects_in_frame(objects, idx * stride, obj_to_zone)
            
            # 在图像上绘制物体标注
            img_annotated = self._draw_object_annotations(img, visible_objects)
            keyframe_images.append((idx, img_annotated))
        
        # 生成关键帧拼接图
        if keyframe_images:
            self._save_keyframe_mosaic(keyframe_images, output_dir / "keyframe_mosaic.png")
        
        # 生成按区域分组的关键帧
        self._generate_zone_keyframes(scene_path, objects, obj_to_zone, image_files, 
                                      stride, output_dir)
    
    def _find_visible_objects_in_frame(self, objects: list, frame_idx: int, 
                                       obj_to_zone: dict) -> List[Dict]:
        """查找在指定帧中可见的物体"""
        visible = []
        for i, obj in enumerate(objects):
            # 检查是否有此帧的检测
            color_path = obj.get('color_path', [])
            bbox_2d = obj.get('bbox_2d', [])
            
            for j, path in enumerate(color_path):
                # 从路径中提取帧号
                try:
                    if 'frame' in str(path):
                        frame_num = int(Path(path).stem.replace('frame', ''))
                    else:
                        frame_num = int(Path(path).stem)
                except:
                    continue
                
                if abs(frame_num - frame_idx) < 10 and j < len(bbox_2d):
                    zone_info = obj_to_zone.get(i, (None, "unassigned"))
                    visible.append({
                        'object_id': i,
                        'object_tag': self._get_object_tag(i),
                        'bbox': bbox_2d[j],
                        'zone_id': zone_info[0],
                        'zone_name': zone_info[1] if isinstance(zone_info, tuple) and len(zone_info) > 1 else "unassigned"
                    })
                    break
        return visible
    
    def _get_object_tag(self, obj_id: int) -> str:
        """获取物体标签"""
        for zone in self.sg.functional_zones:
            for obj in zone.objects:
                if obj.object_id == obj_id:
                    return obj.object_tag
        return f"object_{obj_id}"
    
    def _draw_object_annotations(self, img: np.ndarray, visible_objects: List[Dict]) -> np.ndarray:
        """在图像上绘制物体标注"""
        img = img.copy()
        h, w = img.shape[:2]
        
        for obj in visible_objects:
            bbox = obj['bbox']
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # 获取区域颜色
            zone_id = obj['zone_id']
            if zone_id and zone_id in self.zone_colors:
                color = tuple(int(c * 255) for c in self.zone_colors[zone_id])
            else:
                color = (128, 128, 128)
            
            # 绘制边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{obj['object_tag']}"
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), thickness)
        
        return img
    
    def _save_keyframe_mosaic(self, keyframe_images: List[Tuple[int, np.ndarray]], output_path: str):
        """保存关键帧拼接图"""
        n = len(keyframe_images)
        if n == 0:
            return
        
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, (frame_idx, img) in enumerate(keyframe_images):
            r, c = i // cols, i % cols
            axes[r][c].imshow(img)
            axes[r][c].set_title(f"Frame {frame_idx}", fontsize=10)
            axes[r][c].axis('off')
        
        # 隐藏空白子图
        for i in range(n, rows * cols):
            r, c = i // cols, i % cols
            axes[r][c].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Keyframe mosaic saved to: {output_path}")
    
    def _generate_zone_keyframes(self, scene_path: Path, objects: list, 
                                 obj_to_zone: dict, image_files: list, 
                                 stride: int, output_dir: Path):
        """为每个区域生成代表性关键帧"""
        zone_frames = {zone.zone_id: [] for zone in self.sg.functional_zones}
        
        # 收集每个区域的可见帧
        for i, obj in enumerate(objects):
            zone_info = obj_to_zone.get(i)
            if not zone_info:
                continue
            zone_id = zone_info[0] if isinstance(zone_info, tuple) else zone_info
            
            color_paths = obj.get('color_path', [])
            for path in color_paths[:3]:  # 每个物体最多3帧
                try:
                    if 'frame' in str(path):
                        frame_num = int(Path(path).stem.replace('frame', ''))
                    else:
                        frame_num = int(Path(path).stem)
                    zone_frames[zone_id].append(frame_num)
                except:
                    continue
        
        # 为每个区域生成可视化
        for zone in self.sg.functional_zones:
            frames = zone_frames.get(zone.zone_id, [])
            if not frames:
                continue
            
            # 选择出现最多的帧
            from collections import Counter
            frame_counts = Counter(frames)
            top_frames = [f for f, _ in frame_counts.most_common(4)]
            
            zone_images = []
            for frame_idx in top_frames:
                # 找到对应的图像文件
                img_idx = frame_idx // stride
                if img_idx >= len(image_files):
                    continue
                
                img_path = image_files[img_idx]
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 查找此帧中属于该区域的物体
                visible = []
                for obj_id, (zid, zname) in obj_to_zone.items():
                    if zid != zone.zone_id:
                        continue
                    if obj_id >= len(objects):
                        continue
                    obj = objects[obj_id]
                    color_paths = obj.get('color_path', [])
                    bbox_2d = obj.get('bbox_2d', [])
                    
                    for j, path in enumerate(color_paths):
                        try:
                            if 'frame' in str(path):
                                fn = int(Path(path).stem.replace('frame', ''))
                            else:
                                fn = int(Path(path).stem)
                            if abs(fn - frame_idx) < 10 and j < len(bbox_2d):
                                visible.append({
                                    'object_id': obj_id,
                                    'object_tag': self._get_object_tag(obj_id),
                                    'bbox': bbox_2d[j],
                                    'zone_id': zone.zone_id,
                                    'zone_name': zone.zone_name
                                })
                                break
                        except:
                            continue
                
                img_annotated = self._draw_object_annotations(img, visible)
                zone_images.append((frame_idx, img_annotated))
            
            if zone_images:
                # 保存该区域的可视化
                zone_output = output_dir / f"zone_{zone.zone_name}.png"
                self._save_zone_visualization(zone, zone_images, zone_output)
    
    def _save_zone_visualization(self, zone, zone_images: List[Tuple[int, np.ndarray]], output_path: str):
        """Save single zone visualization"""
        n = min(4, len(zone_images))
        
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.7])
        
        # Draw images
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            if i < len(zone_images):
                frame_idx, img = zone_images[i]
                ax.imshow(img)
                ax.set_title(f"Frame {frame_idx}", fontsize=9)
            ax.axis('off')
        
        # Draw zone info (English only)
        ax_info = fig.add_subplot(gs[0, 4])
        ax_info.axis('off')
        
        color = self.zone_colors.get(zone.zone_id, (0.5, 0.5, 0.5))
        ax_info.add_patch(plt.Rectangle((0.05, 0.88), 0.9, 0.08, facecolor=color, edgecolor='black'))
        
        # Extract supported activities (English)
        activities = zone.supported_activities[:3] if zone.supported_activities else []
        activities_str = ", ".join(activities) if activities else "general"
        
        info_text = f"{zone.zone_name}\n\n"
        info_text += f"Activities: {activities_str}\n\n"
        info_text += f"Objects ({len(zone.objects)}):\n"
        
        # List defining objects
        defining = [o for o in zone.objects if o.relation_type == ObjectRegionRelation.DEFINING]
        if defining:
            info_text += "* Defining:\n"
            for o in defining[:5]:
                info_text += f"  - {o.object_tag}\n"
        
        # List some supporting objects
        supporting = [o for o in zone.objects if o.relation_type != ObjectRegionRelation.DEFINING][:3]
        if supporting:
            info_text += "* Supporting:\n"
            for o in supporting:
                info_text += f"  - {o.object_tag}\n"
        
        ax_info.text(0.05, 0.80, info_text, fontsize=8, va='top', transform=ax_info.transAxes,
                    family='monospace')
        
        plt.suptitle(f"Zone: {zone.zone_name}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Zone visualization saved to: {output_path}")


def visualize_hierarchical_scene(scene_graph_path: str, output_dir: str, 
                                  scene_path: str = None, pcd_file: str = None):
    """便捷函数：生成所有可视化"""
    sg = HierarchicalSceneGraph.load(scene_graph_path)
    vis = HierarchicalSceneVisualizer(sg)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础可视化
    vis.generate_dashboard(str(output_dir / "hierarchical_dashboard.png"))
    vis.generate_topdown_map(str(output_dir / "zone_map_topdown.png"))
    
    # 3D点云可视化（如果提供了pcd文件）
    if pcd_file and Path(pcd_file).exists():
        try:
            vis.generate_3d_ply(pcd_file, str(output_dir / "zones_colored.ply"))
        except Exception as e:
            print(f"Warning: Failed to generate 3D PLY: {e}")
    
    # 关键帧可视化（如果提供了场景路径）
    if scene_path and pcd_file and Path(scene_path).exists() and Path(pcd_file).exists():
        try:
            keyframe_dir = output_dir / "keyframes"
            vis.generate_keyframe_visualization(scene_path, pcd_file, str(keyframe_dir))
        except Exception as e:
            print(f"Warning: Failed to generate keyframe visualization: {e}")
    
    # 保存JSON摘要
    summary = {
        "scene_id": sg.scene_id,
        "n_units": len(sg.spatial_units),
        "n_zones": len(sg.functional_zones),
        "zones": [{"id": z.zone_id, "name": z.zone_name, "n_objects": len(z.objects),
                  "primary_activity": z.primary_activity}
                 for z in sg.functional_zones]
    }
    with open(output_dir / "scene_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Visualization complete. Output: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hierarchical Scene Graph Visualizer")
    parser.add_argument("--scene_graph", type=str, required=True, help="Path to scene graph JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--scene_path", type=str, default=None, help="Scene data path (for keyframes)")
    parser.add_argument("--pcd_file", type=str, default=None, help="PCD file path (for 3D PLY)")
    args = parser.parse_args()
    
    visualize_hierarchical_scene(args.scene_graph, args.output_dir, 
                                 args.scene_path, args.pcd_file)
