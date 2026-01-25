"""
可视化脚本: 点级别和物体级别特征热力图

针对查询 "the sofa near the door" 生成:
1. 点级别 CLIP 相似度热力图
2. 物体级别 CLIP 相似度热力图
3. 俯视图 (BEV) 热力图
"""

import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 尝试导入 CLIP
try:
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    logger.warning("open_clip not available")


def load_pcd_data(pcd_path: str) -> Dict:
    """加载 pcd 数据文件"""
    with gzip.open(pcd_path, 'rb') as f:
        data = pickle.load(f)
    return data


def encode_text_query(query: str, model=None, tokenizer=None) -> np.ndarray:
    """编码文本查询为 CLIP 特征"""
    if not HAS_CLIP:
        raise RuntimeError("CLIP not available")
    
    if model is None:
        model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    with torch.no_grad():
        tokens = tokenizer([query])
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten()


def compute_object_similarities(
    objects: List[Dict],
    query_feature: np.ndarray
) -> List[Tuple[int, str, float, np.ndarray]]:
    """计算所有物体与查询的相似度
    
    Returns:
        List of (obj_id, category, similarity, centroid)
    """
    results = []
    
    for i, obj in enumerate(objects):
        clip_ft = obj.get('clip_ft')
        if clip_ft is None:
            continue
        
        if hasattr(clip_ft, 'cpu'):
            clip_ft = clip_ft.cpu().numpy()
        clip_ft = np.asarray(clip_ft).flatten()
        
        # Normalize
        clip_ft = clip_ft / (np.linalg.norm(clip_ft) + 1e-8)
        
        # Cosine similarity
        similarity = float(np.dot(clip_ft, query_feature))
        
        # Get centroid
        pcd_np = obj.get('pcd_np')
        if pcd_np is not None and len(pcd_np) > 0:
            centroid = np.mean(pcd_np, axis=0)
        else:
            centroid = np.zeros(3)
        
        # Get category
        class_names = obj.get('class_name', [])
        if class_names and class_names[0]:
            category = class_names[0]
        else:
            category = f"object_{i}"
        
        results.append((i, category, similarity, centroid))
    
    return sorted(results, key=lambda x: x[2], reverse=True)


def create_bev_heatmap(
    objects: List[Dict],
    similarities: List[Tuple[int, str, float, np.ndarray]],
    size: Tuple[int, int] = (600, 600),
    query: str = "",
) -> np.ndarray:
    """创建俯视图热力图"""
    # 收集所有点
    all_points = []
    for obj in objects:
        pcd_np = obj.get('pcd_np')
        if pcd_np is not None and len(pcd_np) > 0:
            all_points.append(np.asarray(pcd_np)[:, :2])  # XY only
    
    if not all_points:
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    all_points = np.vstack(all_points)
    
    # Compute bounds
    min_pt = all_points.min(axis=0) - 0.5
    max_pt = all_points.max(axis=0) + 0.5
    
    scale = min(size[0], size[1]) / max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]) * 0.9
    offset = np.array([
        (size[0] - (max_pt[0] - min_pt[0]) * scale) / 2,
        (size[1] - (max_pt[1] - min_pt[1]) * scale) / 2
    ])
    
    # Create heatmap
    heatmap = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240
    
    # Build similarity lookup
    sim_lookup = {obj_id: sim for obj_id, _, sim, _ in similarities}
    
    # Draw objects with color based on similarity
    for obj_id, category, similarity, centroid in similarities:
        # Map similarity to color (blue -> red)
        # Normalize similarity to [0, 1]
        normalized_sim = (similarity + 1) / 2  # from [-1,1] to [0,1]
        normalized_sim = max(0, min(1, normalized_sim))
        
        # Color: blue (low) -> yellow -> red (high)
        if normalized_sim < 0.5:
            r = int(255 * normalized_sim * 2)
            g = int(255 * normalized_sim * 2)
            b = int(255 * (1 - normalized_sim * 2))
        else:
            r = 255
            g = int(255 * (1 - (normalized_sim - 0.5) * 2))
            b = 0
        
        color = (r, g, b)
        
        # Convert centroid to pixel
        x = int((centroid[0] - min_pt[0]) * scale + offset[0])
        y = int((centroid[1] - min_pt[1]) * scale + offset[1])
        
        # Draw circle
        radius = max(8, int(15 * normalized_sim))
        if 0 <= x < size[0] and 0 <= y < size[1]:
            cv2.circle(heatmap, (x, y), radius, color, -1)
            cv2.circle(heatmap, (x, y), radius, (50, 50, 50), 1)
            
            # Add label for top objects
            if similarity > 0.2:
                label = f"#{obj_id}: {category[:15]}"
                cv2.putText(heatmap, label, (x + radius + 5, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add title and colorbar
    cv2.putText(heatmap, f'Query: "{query}"', (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(heatmap, "Object-Level CLIP Similarity Heatmap", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Draw colorbar
    cb_x, cb_y, cb_w, cb_h = size[0] - 80, 60, 20, 150
    for i in range(cb_h):
        ratio = i / cb_h
        if ratio < 0.5:
            r = int(255 * ratio * 2)
            g = int(255 * ratio * 2)
            b = int(255 * (1 - ratio * 2))
        else:
            r = 255
            g = int(255 * (1 - (ratio - 0.5) * 2))
            b = 0
        cv2.line(heatmap, (cb_x, cb_y + cb_h - i), (cb_x + cb_w, cb_y + cb_h - i), (r, g, b), 1)
    
    cv2.putText(heatmap, "High", (cb_x - 5, cb_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(heatmap, "Low", (cb_x - 5, cb_y + cb_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    return heatmap


def create_point_heatmap(
    objects: List[Dict],
    query_feature: np.ndarray,
    size: Tuple[int, int] = (600, 600),
    query: str = "",
) -> np.ndarray:
    """创建点级别热力图 (使用物体特征近似)"""
    # 收集所有点和对应的相似度
    all_points = []
    all_sims = []
    
    for obj in objects:
        clip_ft = obj.get('clip_ft')
        pcd_np = obj.get('pcd_np')
        
        if clip_ft is None or pcd_np is None or len(pcd_np) == 0:
            continue
        
        if hasattr(clip_ft, 'cpu'):
            clip_ft = clip_ft.cpu().numpy()
        clip_ft = np.asarray(clip_ft).flatten()
        clip_ft = clip_ft / (np.linalg.norm(clip_ft) + 1e-8)
        
        similarity = float(np.dot(clip_ft, query_feature))
        
        # 每个点继承物体的相似度
        pcd_np = np.asarray(pcd_np)
        all_points.append(pcd_np[:, :2])  # XY
        all_sims.extend([similarity] * len(pcd_np))
    
    if not all_points:
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    all_points = np.vstack(all_points)
    all_sims = np.array(all_sims)
    
    # Compute bounds
    min_pt = all_points.min(axis=0) - 0.5
    max_pt = all_points.max(axis=0) + 0.5
    
    scale = min(size[0], size[1]) / max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]) * 0.9
    offset = np.array([
        (size[0] - (max_pt[0] - min_pt[0]) * scale) / 2,
        (size[1] - (max_pt[1] - min_pt[1]) * scale) / 2
    ])
    
    # Create heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Transform points to pixel space
    px = (all_points[:, 0] - min_pt[0]) * scale + offset[0]
    py = (all_points[:, 1] - min_pt[1]) * scale + offset[1]
    
    # Normalize similarities for coloring
    norm_sims = (all_sims - all_sims.min()) / (all_sims.max() - all_sims.min() + 1e-8)
    
    # Scatter plot
    scatter = ax.scatter(px, py, c=norm_sims, cmap='RdYlBu_r', s=1, alpha=0.6)
    
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Point-Level CLIP Similarity\nQuery: "{query}"')
    
    plt.colorbar(scatter, ax=ax, label='Similarity')
    
    # Convert to image - save to temp file and read back
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.close(fig)
    
    return img


def visualize_query(
    pcd_path: str,
    query: str,
    output_dir: str,
    target_terms: List[str] = None,
    anchor_terms: List[str] = None,
):
    """主可视化函数
    
    Args:
        pcd_path: PCD数据文件路径
        query: 查询文本
        output_dir: 输出目录
        target_terms: 目标物体的关键词列表 (用于高亮)
        anchor_terms: 参照物的关键词列表 (用于高亮)
    """
    logger.info(f"Loading data from: {pcd_path}")
    data = load_pcd_data(pcd_path)
    objects = data.get('objects', [])
    logger.info(f"Found {len(objects)} objects")
    
    # Encode query
    logger.info(f"Encoding query: '{query}'")
    query_feature = encode_text_query(query)
    
    # Compute similarities
    logger.info("Computing object similarities...")
    similarities = compute_object_similarities(objects, query_feature)
    
    # Print top results
    logger.info("Top 10 objects by similarity:")
    for i, (obj_id, category, sim, centroid) in enumerate(similarities[:10]):
        pos_str = f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
        logger.info(f"  {i+1}. #{obj_id}: {category:20s} sim={sim:.4f}  @ {pos_str}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate BEV heatmap
    logger.info("Generating BEV heatmap...")
    bev_heatmap = create_bev_heatmap(objects, similarities, query=query)
    bev_path = output_dir / "object_heatmap_bev.png"
    cv2.imwrite(str(bev_path), cv2.cvtColor(bev_heatmap, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved to: {bev_path}")
    
    # Generate point heatmap
    logger.info("Generating point-level heatmap...")
    point_heatmap = create_point_heatmap(objects, query_feature, query=query)
    point_path = output_dir / "point_heatmap.png"
    cv2.imwrite(str(point_path), cv2.cvtColor(point_heatmap, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved to: {point_path}")
    
    # Also encode sub-queries for target and anchor
    if target_terms or anchor_terms:
        logger.info("Generating separate heatmaps for target and anchor...")
        
        if target_terms:
            target_query = " ".join(target_terms)
            target_feature = encode_text_query(target_query)
            target_sims = compute_object_similarities(objects, target_feature)
            target_heatmap = create_bev_heatmap(objects, target_sims, query=f"target: {target_query}")
            target_path = output_dir / "target_heatmap.png"
            cv2.imwrite(str(target_path), cv2.cvtColor(target_heatmap, cv2.COLOR_RGB2BGR))
            logger.info(f"Target heatmap: {target_path}")
        
        if anchor_terms:
            anchor_query = " ".join(anchor_terms)
            anchor_feature = encode_text_query(anchor_query)
            anchor_sims = compute_object_similarities(objects, anchor_feature)
            anchor_heatmap = create_bev_heatmap(objects, anchor_sims, query=f"anchor: {anchor_query}")
            anchor_path = output_dir / "anchor_heatmap.png"
            cv2.imwrite(str(anchor_path), cv2.cvtColor(anchor_heatmap, cv2.COLOR_RGB2BGR))
            logger.info(f"Anchor heatmap: {anchor_path}")
    
    # Generate combined visualization
    logger.info("Generating combined visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Full query
    axes[0].imshow(bev_heatmap)
    axes[0].set_title(f'Full Query: "{query}"')
    axes[0].axis('off')
    
    # Point level
    axes[1].imshow(point_heatmap)
    axes[1].set_title('Point-Level Similarity')
    axes[1].axis('off')
    
    # Top objects table
    axes[2].axis('off')
    table_data = []
    for i, (obj_id, category, sim, centroid) in enumerate(similarities[:15]):
        table_data.append([f"#{obj_id}", category[:20], f"{sim:.3f}"])
    
    table = axes[2].table(
        cellText=table_data,
        colLabels=['ID', 'Category', 'Similarity'],
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[2].set_title('Top Objects by CLIP Similarity')
    
    plt.tight_layout()
    combined_path = output_dir / "combined_visualization.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Combined: {combined_path}")
    
    logger.success("Visualization complete!")
    return {
        "bev_heatmap": str(bev_path),
        "point_heatmap": str(point_path),
        "combined": str(combined_path),
        "top_objects": similarities[:10],
    }


def main():
    """运行示例: the sofa near the door"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize CLIP similarity heatmap")
    parser.add_argument("--pcd", type=str, 
                       default="/home/shyue/Datasets/Replica/Replica/room0/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz",
                       help="Path to pcd file")
    parser.add_argument("--query", type=str, default="the sofa near the door",
                       help="Query text")
    parser.add_argument("--output", type=str, 
                       default="/home/shyue/Datasets/Replica/Replica/room0/query_visualizations",
                       help="Output directory")
    parser.add_argument("--target", type=str, default="sofa", help="Target object")
    parser.add_argument("--anchor", type=str, default="door", help="Anchor object")
    
    args = parser.parse_args()
    
    visualize_query(
        pcd_path=args.pcd,
        query=args.query,
        output_dir=args.output,
        target_terms=[args.target],
        anchor_terms=[args.anchor],
    )


if __name__ == "__main__":
    main()
