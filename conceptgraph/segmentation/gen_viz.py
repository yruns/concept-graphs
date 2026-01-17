#!/usr/bin/env python3
"""生成增强可视化"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import distinctipy

def main():
    scene_path = Path('/home/shyue/Datasets/Replica/Replica/room0')
    sg_path = scene_path / 'hierarchical_segmentation/hierarchical_scene_graph.json'
    output_dir = scene_path / 'hierarchical_segmentation'
    crop_dir = scene_path / 'sg_cache' / 'cfslam_captions_llava_debug'
    
    with open(sg_path, 'r') as f:
        sg = json.load(f)
    
    zones = sg.get('functional_zones', [])
    zone_colors = distinctipy.get_colors(len(zones), pastel_factor=0.3)
    
    obj_grid_dir = output_dir / 'zone_objects'
    obj_grid_dir.mkdir(exist_ok=True)
    
    for i, zone in enumerate(zones):
        zone_id = zone['zone_id']
        zone_name = zone['zone_name']
        zone_objs = zone.get('objects', [])
        
        images, labels = [], []
        for obj in zone_objs:
            obj_id = obj['object_id']
            crop_path = crop_dir / f'{obj_id}.png'
            if crop_path.exists():
                img = Image.open(crop_path)
                img.thumbnail((150, 150))
                images.append(img)
                labels.append(obj.get('object_tag', f'obj_{obj_id}'))
        
        if not images:
            continue
        
        n_cols = min(5, len(images))
        n_rows = (len(images) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5+0.8))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        fig.suptitle(f'{zone_name}', fontsize=11, fontweight='bold')
        
        for idx in range(len(images)):
            r, c = idx // n_cols, idx % n_cols
            axes[r][c].imshow(images[idx])
            axes[r][c].set_title(labels[idx], fontsize=8)
            axes[r][c].axis('off')
        
        for idx in range(len(images), n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes[r][c].axis('off')
        
        plt.tight_layout()
        plt.savefig(obj_grid_dir / f'{zone_id}_{zone_name}_objects.png', dpi=100)
        plt.close()
        print(f'{zone_name}: {len(images)} objects')
    
    print('Done')

if __name__ == '__main__':
    main()
