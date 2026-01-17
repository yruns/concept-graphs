#!/bin/bash

################################################################################
# 步骤 2B: 构建 3D 对象地图 (使用类别感知检测结果)
################################################################################
#
# 作用：
#   - 使用步骤1B (ram_withbg_allclasses) 的检测结果
#   - 将 2D 分割结果投影到 3D 空间
#   - 跨帧关联和匹配同一物体
#   - 融合多视图点云构建 3D 物体表示
#   - 保留物体类别信息
#
# 输入：
#   - 2D 分割结果: $REPLICA_ROOT/$SCENE_NAME/gsa_detections_ram_withbg_allclasses/*.pkl.gz
#   - 相机位姿: $REPLICA_ROOT/$SCENE_NAME/traj.txt
#
# 输出：
#   - 3D 对象地图: $REPLICA_ROOT/$SCENE_NAME/pcd_saves/full_pcd_ram_withbg_allclasses_*_post.pkl.gz
#
################################################################################

# 激活环境
source /home/shyue/anaconda3/bin/activate conceptgraph
export PYTHONPATH="/home/shyue/codebase/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# 进入工作目录
cd /home/shyue/codebase/concept-graphs/conceptgraph

# 加载环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash

# 场景设置
SCENE_NAME=${1:-room0}
THRESHOLD=1.2

echo "================================================"
echo "步骤 2B: 构建 3D 对象地图 (类别感知模式)"
echo "================================================"
echo "场景: ${SCENE_NAME}"
echo "相似度阈值: ${THRESHOLD}"
echo ""
echo "输入: ${REPLICA_ROOT}/${SCENE_NAME}/gsa_detections_ram_withbg_allclasses/"
echo "输出: ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/"
echo "================================================"
echo ""

# 运行 3D 对象映射 (ConceptGraphs-Detect - 类别感知模式)
python slam/cfslam_pipeline_batch.py \
    dataset_root=$REPLICA_ROOT \
    dataset_config=$REPLICA_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=ram_withbg_allclasses \
    class_agnostic=False \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 3D 对象地图构建完成 (类别感知模式)"
    echo ""
    PKL_BASE="full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub"
    PKL_FILE="${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_BASE}_post.pkl.gz"
    
    echo "输出文件:"
    echo "  - 后处理版本 (推荐): ${PKL_FILE}"
    echo "  - 原始版本: ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/${PKL_BASE}.pkl.gz"
    echo ""
    
    # 生成PLY可视化文件
    echo "================================================"
    echo "生成PLY可视化文件..."
    echo "================================================"
    
    python -c "
import gzip
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import distinctipy

pcd_path = '${PKL_FILE}'
output_dir = Path('${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/ply_export_detect')
output_dir.mkdir(exist_ok=True)

print(f'加载: {pcd_path}')
with gzip.open(pcd_path, 'rb') as f:
    data = pickle.load(f)

objects = data.get('objects', [])
print(f'物体数量: {len(objects)}')

if not objects:
    print('无物体，跳过PLY生成')
    exit(0)

# 为每个物体分配颜色
colors = distinctipy.get_colors(len(objects), pastel_factor=0.2)

all_points, all_colors = [], []
object_info = []

for i, obj in enumerate(objects):
    if 'pcd_np' not in obj or len(obj['pcd_np']) == 0:
        continue
    
    points = obj['pcd_np']
    color = colors[i]
    obj_colors = np.tile(np.array(color), (len(points), 1))
    
    # 获取类别名称
    class_names = obj.get('class_name', [])
    if class_names:
        valid = [n for n in class_names if n and n.lower() != 'item']
        if valid:
            tag = Counter(valid).most_common(1)[0][0]
        else:
            tag = class_names[0] if class_names[0] else f'object_{i}'
    else:
        tag = f'object_{i}'
    
    object_info.append({
        'id': i,
        'tag': tag,
        'n_points': len(points),
        'color': color
    })
    
    all_points.append(points)
    all_colors.append(obj_colors)

# 合并并保存PLY
all_pts = np.vstack(all_points)
all_cls = np.vstack(all_colors)

ply_path = output_dir / 'all_objects_colored.ply'
with open(ply_path, 'w') as f:
    f.write('ply\nformat ascii 1.0\n')
    f.write(f'element vertex {len(all_pts)}\n')
    f.write('property float x\nproperty float y\nproperty float z\n')
    f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    for j in range(len(all_pts)):
        x, y, z = all_pts[j]
        r, g, b = (all_cls[j] * 255).astype(int)
        f.write(f'{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n')
print(f'保存PLY: {ply_path}')

# 保存物体信息JSON
import json
info_path = output_dir / 'objects_info.json'
save_info = []
for obj in object_info:
    save_info.append({
        'id': obj['id'],
        'tag': obj['tag'],
        'n_points': obj['n_points'],
        'color_rgb': [int(c*255) for c in obj['color']]
    })
with open(info_path, 'w') as f:
    json.dump(save_info, f, indent=2, ensure_ascii=False)
print(f'保存物体信息: {info_path}')

# 打印物体统计
print(f'\n物体类别统计:')
tag_counts = Counter([o['tag'] for o in object_info])
for tag, count in tag_counts.most_common(15):
    print(f'  {tag}: {count}')
"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ PLY可视化文件生成完成"
        echo "  - ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/ply_export_detect/all_objects_colored.ply"
        echo "  - ${REPLICA_ROOT}/${SCENE_NAME}/pcd_saves/ply_export_detect/objects_info.json"
    fi
    echo ""
else
    echo ""
    echo "✗ 3D 对象地图构建失败"
    exit 1
fi
