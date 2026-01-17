#!/usr/bin/env python3
"""
增强可视化器 - 生成更直观的层次化场景分割可视化
=================================================

输出:
1. 区域代表性图像 - 每个功能区域的代表帧 + 物体标注
2. 3D点云PLY - 带区域颜色的点云文件
3. 物体裁剪图拼接 - 每个区域包含的物体图像
4. 综合面板 - 汇总展示
"""

import os
import json
import gzip
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
import distinctipy


class EnhancedHierarchicalVisualizer:
    """增强层次化场景可视化器"""
    
    def __init__(self, scene_path: str, scene_graph_path: str):
        self.scene_path = Path(scene_path)
        self.scene_graph_path = Path(scene_graph_path)
        
        with open(scene_graph_path, 'r') as f:
            self.scene_graph = json.load(f)
        
        self.objects_data = None
        self._load_point_cloud()
        
        n_zones = len(self.scene_graph.get('functional_zones', []))
        self.zone_colors = distinctipy.get_colors(max(n_zones, 1), pastel_factor=0.3)
        self.zone_color_map = {}
        for i, zone in enumerate(self.scene_graph.get('functional_zones', [])):
            self.zone_color_map[zone['zone_id']] = self.zone_colors[i]
        
        self.rgb_dir = self.scene_path / "results"
        self.crop_dir = self.scene_path / "sg_cache" / "cfslam_captions_llava_debug"
    
    def _load_point_cloud(self):
        pcd_dir = self.scene_path / "pcd_saves"
        if not pcd_dir.exists():
            return
        pcd_files = list(pcd_dir.glob("*_post.pkl.gz"))
        if not pcd_files:
            pcd_files = list(pcd_dir.glob("*.pkl.gz"))
        if pcd_files:
            with gzip.open(pcd_files[0], 'rb') as f:
                self.objects_data = pickle.load(f)
            print(f"  加载点云: {pcd_files[0].name}")
    
    def generate_all(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n生成增强可视化...")
        self.generate_zone_keyframes(output_dir / "zone_keyframes")
        self.generate_zone_pointclouds(output_dir / "zone_pointclouds")
        self.generate_zone_object_grids(output_dir / "zone_objects")
        self.generate_comprehensive_panel(output_dir / "comprehensive_panel.png")
        print(f"\n可视化完成! 输出目录: {output_dir}")
    
    def _find_frame_image(self, frame_idx: int) -> Optional[Path]:
        patterns = [
            f"frame_{frame_idx:05d}.jpg", f"frame_{frame_idx:05d}.png",
            f"frame{frame_idx:05d}.jpg", f"frame{frame_idx:05d}.png",
            f"{frame_idx:05d}.jpg", f"{frame_idx:05d}.png",
            f"frame_{frame_idx:06d}.jpg", f"frame_{frame_idx:06d}.png",
        ]
        for pattern in patterns:
            path = self.rgb_dir / pattern
            if path.exists():
                return path
        return None
    
    def generate_zone_keyframes(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("  生成区域代表性图像...")
        
        if not self.rgb_dir.exists() or self.objects_data is None:
            return
        
        objects = self.objects_data.get('objects', [])
        
        for zone in self.scene_graph.get('functional_zones', []):
            zone_id = zone['zone_id']
            zone_name = zone['zone_name']
            color = self.zone_color_map.get(zone_id, (0.5, 0.5, 0.5))
            zone_obj_ids = [obj['object_id'] for obj in zone.get('objects', [])]
            
            visible_frames = set()
            for obj_id in zone_obj_ids:
                if obj_id < len(objects):
                    obj_data = objects[obj_id]
                    if 'image_idx' in obj_data:
                        visible_frames.update(obj_data['image_idx'])
            
            if not visible_frames:
                continue
            
            sorted_frames = sorted(visible_frames)
            representative_frame = sorted_frames[len(sorted_frames) // 2]
            
            frame_path = self._find_frame_image(representative_frame)
            if frame_path is None:
                continue
            
            img = Image.open(frame_path)
            draw = ImageDraw.Draw(img)
            
            for obj_id in zone_obj_ids:
                if obj_id < len(objects):
                    obj_data = objects[obj_id]
                    if representative_frame in obj_data.get('image_idx', []):
                        frame_list = list(obj_data['image_idx'])
                        if representative_frame in frame_list:
                            mask_idx = frame_list.index(representative_frame)
                            if mask_idx < len(obj_data.get('xyxy', [])):
                                xyxy = obj_data['xyxy'][mask_idx]
                                rgb_color = tuple(int(c * 255) for c in color)
                                draw.rectangle(xyxy.tolist(), outline=rgb_color, width=3)
                                label = obj_data.get('class_name', f'obj_{obj_id}')
                                draw.text((xyxy[0], xyxy[1] - 15), label, fill=rgb_color)
            
            title = f"{zone_name} ({len(zone_obj_ids)} objects)"
            draw.rectangle([0, 0, img.width, 30], fill=tuple(int(c * 255) for c in color))
            draw.text((10, 5), title, fill=(255, 255, 255))
            
            img.save(output_dir / f"{zone_id}_{zone_name}.jpg")
        print(f"    保存到: {output_dir}")
