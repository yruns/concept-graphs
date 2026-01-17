#!/usr/bin/env python3
"""
层次化场景图可视化器
==================

支持两种可视化模式：
1. 静态图片：生成PNG展示面板
2. 交互式3D：Open3D窗口
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
from typing import List, Dict, Optional
import distinctipy

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


def visualize_hierarchical_scene(scene_graph_path: str, output_dir: str):
    """便捷函数：生成所有可视化"""
    sg = HierarchicalSceneGraph.load(scene_graph_path)
    vis = HierarchicalSceneVisualizer(sg)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis.generate_dashboard(str(output_dir / "hierarchical_dashboard.png"))
    vis.generate_topdown_map(str(output_dir / "zone_map_topdown.png"))
    
    # 保存JSON摘要
    summary = {
        "scene_id": sg.scene_id,
        "n_units": len(sg.spatial_units),
        "n_zones": len(sg.functional_zones),
        "zones": [{"id": z.zone_id, "name": z.zone_name, "n_objects": len(z.objects)}
                 for z in sg.functional_zones]
    }
    with open(output_dir / "scene_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Visualization complete. Output: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_graph", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    visualize_hierarchical_scene(args.scene_graph, args.output_dir)
