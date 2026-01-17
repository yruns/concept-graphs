#!/usr/bin/env python3
"""检查 ConceptGraph PKL 文件的内容"""
import gzip
import pickle
import sys
import numpy as np

if len(sys.argv) < 2:
    print("用法: python inspect_pkl.py <pkl文件路径>")
    sys.exit(1)

pkl_path = sys.argv[1]

print("=" * 70)
print(f"加载文件: {pkl_path}")
print("=" * 70)

with gzip.open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 顶层结构
print("\n【文件顶层结构】")
for key in data.keys():
    val = data[key]
    if val is None:
        print(f"  {key}: None")
    elif isinstance(val, list):
        print(f"  {key}: 列表，包含 {len(val)} 个元素")
    elif isinstance(val, dict):
        print(f"  {key}: 字典，包含 {len(val)} 个键")
    else:
        print(f"  {key}: {type(val).__name__}")

# 物体信息
objects = data.get('objects', [])
print(f"\n【物体信息】共 {len(objects)} 个物体")

if len(objects) > 0:
    print(f"\n物体 #0 包含的属性:")
    obj = objects[0]
    
    for key in sorted(obj.keys()):
        val = obj[key]
        if val is None:
            info = "None"
        elif isinstance(val, np.ndarray):
            info = f"NumPy数组 shape={val.shape}"
        elif hasattr(val, 'points'):
            # Open3D 点云
            points = np.asarray(val.points)
            info = f"Open3D点云 ({len(points)} 个点)"
        elif isinstance(val, (int, float)):
            info = f"{type(val).__name__} = {val}"
        elif isinstance(val, str):
            info = f'字符串 = "{val[:30]}..."' if len(val) > 30 else f'"{val}"'
        elif isinstance(val, (list, tuple)):
            info = f"{type(val).__name__} (长度: {len(val)})"
        else:
            info = f"{type(val).__name__}"
        
        print(f"  • {key:20s} : {info}")

# 检查是否有图结构信息
print("\n【场景图结构检查】")
graph_keys = ['edges', 'relationships', 'scene_graph', 'connections', 'graph', 'adjacency']
found_graph_data = False

for gkey in graph_keys:
    if gkey in data:
        print(f"  找到图结构数据: {gkey}")
        val = data[gkey]
        if isinstance(val, list):
            print(f"    类型: 列表，包含 {len(val)} 个元素")
        elif isinstance(val, dict):
            print(f"    类型: 字典，包含 {len(val)} 个键")
            if len(val) > 0:
                print(f"    键: {list(val.keys())[:5]}...")
        else:
            print(f"    类型: {type(val).__name__}")
        found_graph_data = True

if not found_graph_data:
    print("  ❌ 未找到明显的场景图结构数据")
    print("  这可能是因为:")
    print("    1. ConceptGraphs在此配置下只生成物体检测，不生成关系")
    print("    2. 场景图信息存储在其他文件中")
    print("    3. 需要额外的后处理步骤生成关系")

# 检查配置信息
print("\n【配置信息】")
cfg = data.get('cfg', {})
if cfg:
    print("  关键配置参数:")
    important_keys = ['gsa_variant', 'class_agnostic', 'spatial_sim_type', 'match_method']
    for key in important_keys:
        if hasattr(cfg, key):
            val = getattr(cfg, key)
            print(f"    {key}: {val}")

print("\n" + "=" * 70)
print("PKL 文件说明:")
print("  这是一个 Python Pickle 压缩文件，包含 3D 物体建图的所有信息")
print("  - objects: 检测到的所有前景物体")
print("  - 每个物体包含: 点云、包围盒、CLIP特征、颜色等")
if found_graph_data:
    print("  - 包含场景图结构信息")
else:
    print("  - ⚠️  不包含场景图关系信息")
print("  - 可用于查询、可视化和下游任务")
print("=" * 70)




