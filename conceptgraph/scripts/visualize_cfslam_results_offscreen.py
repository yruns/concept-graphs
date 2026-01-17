#!/usr/bin/env python3
"""
离线可视化场景图（无需显示器）

这个脚本可以在无显示器的服务器上将场景图可视化保存为图像文件。
支持多种输出格式：
1. 静态图像（从多个预设视角）
2. 3D 模型文件（PLY 格式）
3. HTML 交互式可视化（使用 matplotlib 3D）
"""

import copy
import json
import os
import pickle
import gzip
import argparse
import sys

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d

import distinctipy

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh


def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """创建彩色球体网格"""
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def get_parser():
    parser = argparse.ArgumentParser(description="离线可视化场景图")
    parser.add_argument("--result_path", type=str, required=True,
                        help="场景图地图文件路径")
    parser.add_argument("--edge_file", type=str, default=None,
                        help="物体关系文件路径")
    parser.add_argument("--output_dir", type=str, default="./visualization_output",
                        help="输出目录")
    parser.add_argument("--output_format", type=str, default="images",
                        choices=["images", "ply", "html", "all"],
                        help="输出格式: images=多视角图像, ply=3D模型, html=HTML可视化, all=全部")
    parser.add_argument("--image_width", type=int, default=1920,
                        help="输出图像宽度")
    parser.add_argument("--image_height", type=int, default=1080,
                        help="输出图像高度")
    parser.add_argument("--num_views", type=int, default=8,
                        help="生成的视角数量")
    parser.add_argument("--original_mesh", type=str, default=None,
                        help="原始场景mesh文件路径（Replica提供的ground truth）")
    return parser


def load_result(result_path):
    """加载场景图结果"""
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    
    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])

        class_colors = results['class_colors']
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)

        bg_objects = None
        class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
        class_colors = {str(i): c for i, c in enumerate(class_colors)}
    else:
        raise ValueError("Unknown results type: ", type(results))
        
    return objects, bg_objects, class_colors


def create_scene_graph_geometries(objects, edges, class_colors):
    """创建场景图的几何对象（节点和边）"""
    scene_graph_geometries = []
    
    classes = objects.get_most_common_class()
    colors = [class_colors[str(c)] for c in classes]
    obj_centers = []
    
    # 创建节点（球体）
    for obj, c in zip(objects, colors):
        pcd = obj['pcd']
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        radius = 0.10
        obj_centers.append(center)

        ball = create_ball_mesh(center, radius, c)
        scene_graph_geometries.append(ball)
    
    # 创建边（连接线）
    for edge in edges:
        if edge['object_relation'] == "none of these":
            continue
        id1 = edge["object1"]['id']
        id2 = edge["object2"]['id']

        line_mesh = LineMesh(
            points = np.array([obj_centers[id1], obj_centers[id2]]),
            lines = np.array([[0, 1]]),
            colors = [1, 0, 0],
            radius=0.02
        )

        scene_graph_geometries.extend(line_mesh.cylinder_segments)
    
    return scene_graph_geometries, obj_centers


def save_as_images_matplotlib(objects, scene_graph_geometries, output_dir, num_views,
                             image_width, image_height, original_mesh_path=None):
    """使用 matplotlib 生成多视角图像（后备方案）"""
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 加载原始mesh（如果提供）
    original_mesh_points = None
    if original_mesh_path and os.path.exists(original_mesh_path):
        print(f"  加载原始mesh: {original_mesh_path}")
        try:
            original_mesh = o3d.io.read_triangle_mesh(original_mesh_path)
            if len(original_mesh.vertices) > 0:
                # 采样mesh表面点
                original_pcd = original_mesh.sample_points_uniformly(number_of_points=50000)
                original_mesh_points = np.asarray(original_pcd.points)
                print(f"  ✓ 已加载原始mesh ({len(original_mesh_points)} 个点)")
        except Exception as e:
            print(f"  ✗ 加载原始mesh失败: {e}")
    
    # 收集所有点云数据
    all_points = []
    all_colors = []
    for obj in objects:
        pcd = obj['pcd']
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # 采样以减少点数
        if len(points) > 500:
            indices = np.random.choice(len(points), 500, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        all_points.append(points)
        all_colors.append(colors)
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    # 计算场景边界
    center = np.mean(all_points, axis=0)
    max_bound = np.max(all_points, axis=0)
    min_bound = np.min(all_points, axis=0)
    scene_size = np.linalg.norm(max_bound - min_bound)
    
    # 提取场景图节点和边
    node_positions = []
    edge_lines = []
    
    for geom in scene_graph_geometries:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            # 球体节点
            vertices = np.asarray(geom.vertices)
            if len(vertices) > 0:
                node_center = np.mean(vertices, axis=0)
                node_positions.append(node_center)
    
    # 生成多个视角
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    
    for i, angle in enumerate(angles):
        fig = plt.figure(figsize=(image_width/100, image_height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原始mesh（如果有）
        if original_mesh_points is not None:
            ax.scatter(original_mesh_points[:, 0], original_mesh_points[:, 1], original_mesh_points[:, 2],
                      c='lightgray', s=0.5, alpha=0.3, label='Original Scene')
        
        # 绘制点云
        ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                  c=all_colors, s=1, alpha=0.6, label='Objects')
        
        # 绘制场景图节点
        if len(node_positions) > 0:
            nodes = np.array(node_positions)
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                      c='yellow', s=100, marker='o', edgecolors='black', linewidths=2)
        
        # 设置视角
        distance = scene_size * 1.5
        elev = 20  # 仰角
        azim = np.degrees(angle)  # 方位角
        ax.view_init(elev=elev, azim=azim)
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Scene Graph View {i+1}/{num_views}')
        
        # 设置白色背景
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 保存图像
        image_path = os.path.join(images_dir, f"scene_graph_view_{i:02d}.png")
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ 保存视角 {i+1}/{num_views}: {image_path}")
    
    print(f"\n✓ 所有图像已保存到: {images_dir}")


def save_as_images(objects, bg_objects, scene_graph_geometries, output_dir, 
                   image_width, image_height, num_views):
    """保存多视角图像"""
    print(f"\n正在生成 {num_views} 个视角的图像...")
    
    # 准备几何对象
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # 降采样点云以加快渲染
    for i in range(len(objects)):
        pcds[i] = pcds[i].voxel_down_sample(0.05)
    
    # 尝试创建离线渲染器
    try:
        vis = o3d.visualization.Visualizer()
        success = vis.create_window(width=image_width, height=image_height, visible=False)
        
        if not success:
            raise RuntimeError("Failed to create Open3D window")
        
        # 添加几何对象
        for geometry in pcds + bboxes + scene_graph_geometries:
            vis.add_geometry(geometry)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        if opt is None:
            raise RuntimeError("Failed to get render options")
            
        opt.background_color = np.asarray([1, 1, 1])  # 白色背景
        opt.point_size = 2.0
        
        # 获取视图控制器
        view_ctrl = vis.get_view_control()
        if view_ctrl is None:
            raise RuntimeError("Failed to get view control")
            
    except Exception as e:
        print(f"  ✗ Open3D 离线渲染失败: {e}")
        print(f"  提示: 在无显示器环境下，Open3D 需要 OSMesa 或 EGL 支持")
        print(f"  改用 matplotlib 生成静态图像...")
        
        # 使用 matplotlib 作为后备方案
        original_mesh = getattr(save_as_images, 'original_mesh_path', None)
        save_as_images_matplotlib(objects, scene_graph_geometries, output_dir, num_views, 
                                 image_width, image_height, original_mesh)
        return
    
    # 计算场景边界
    all_points = []
    for pcd in pcds:
        all_points.extend(np.asarray(pcd.points))
    all_points = np.array(all_points)
    center = np.mean(all_points, axis=0)
    max_bound = np.max(all_points, axis=0)
    min_bound = np.min(all_points, axis=0)
    scene_size = np.linalg.norm(max_bound - min_bound)
    
    # 生成多个视角
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for i, angle in enumerate(angles):
        # 设置相机位置（围绕场景旋转）
        distance = scene_size * 1.5
        eye = center + np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            scene_size * 0.5
        ])
        
        # 更新视图
        view_ctrl.set_lookat(center)
        view_ctrl.set_front((center - eye) / np.linalg.norm(center - eye))
        view_ctrl.set_up([0, 0, 1])
        view_ctrl.set_zoom(0.7)
        
        # 渲染并保存
        vis.poll_events()
        vis.update_renderer()
        
        image_path = os.path.join(images_dir, f"scene_graph_view_{i:02d}.png")
        vis.capture_screen_image(image_path, do_render=True)
        print(f"  ✓ 保存视角 {i+1}/{num_views}: {image_path}")
    
    vis.destroy_window()
    print(f"\n✓ 所有图像已保存到: {images_dir}")


def save_as_ply(objects, scene_graph_geometries, output_dir):
    """保存为 PLY 3D 模型文件"""
    print("\n正在保存 PLY 文件...")
    
    # 合并所有几何对象
    pcds = objects.get_values("pcd")
    
    # 合并所有点云
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined_pcd += pcd
    
    # 降采样
    combined_pcd = combined_pcd.voxel_down_sample(0.02)
    
    # 保存点云
    ply_dir = os.path.join(output_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    
    pcd_path = os.path.join(ply_dir, "scene_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, combined_pcd)
    print(f"  ✓ 点云已保存: {pcd_path}")
    
    # 保存场景图几何（合并网格）
    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in scene_graph_geometries:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            combined_mesh += geom
    
    if len(combined_mesh.vertices) > 0:
        mesh_path = os.path.join(ply_dir, "scene_graph.ply")
        o3d.io.write_triangle_mesh(mesh_path, combined_mesh)
        print(f"  ✓ 场景图网格已保存: {mesh_path}")
    
    print(f"\n✓ PLY 文件已保存到: {ply_dir}")
    print(f"  提示: 可使用 MeshLab, CloudCompare 或 Blender 打开这些文件")


def save_as_html(objects, obj_centers, edges, class_colors, output_dir):
    """保存为交互式 HTML 可视化"""
    print("\n正在生成 HTML 可视化...")
    
    try:
        import plotly.graph_objects as go
        use_plotly = True
    except ImportError:
        print("  警告: plotly 未安装，使用 matplotlib 替代")
        use_plotly = False
    
    html_dir = os.path.join(output_dir, "html")
    os.makedirs(html_dir, exist_ok=True)
    
    if use_plotly:
        # 使用 Plotly 创建交互式 3D 图
        fig = go.Figure()
        
        # 添加物体点云（采样）
        classes = objects.get_most_common_class()
        colors_rgb = [class_colors[str(c)] for c in classes]
        
        for i, (obj, color, center) in enumerate(zip(objects, colors_rgb, obj_centers)):
            pcd = obj['pcd']
            points = np.asarray(pcd.points)
            
            # 采样点以减小文件大小
            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                points = points[indices]
            
            # 获取物体类别
            obj_classes = np.asarray(obj['class_id'])
            values, counts = np.unique(obj_classes, return_counts=True)
            obj_class = values[np.argmax(counts)]
            
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})',
                    opacity=0.8
                ),
                name=f'物体 {i} (类别 {obj_class})',
                hovertext=f'物体 ID: {i}<br>类别: {obj_class}',
            ))
        
        # 添加场景图的边
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']
            relation = edge['object_relation']
            
            fig.add_trace(go.Scatter3d(
                x=[obj_centers[id1][0], obj_centers[id2][0]],
                y=[obj_centers[id1][1], obj_centers[id2][1]],
                z=[obj_centers[id1][2], obj_centers[id2][2]],
                mode='lines',
                line=dict(color='red', width=5),
                name=f'关系: {relation}',
                hovertext=f'{id1} -> {id2}: {relation}',
                showlegend=False
            ))
        
        # 添加节点（物体中心）
        centers_array = np.array(obj_centers)
        fig.add_trace(go.Scatter3d(
            x=centers_array[:, 0],
            y=centers_array[:, 1],
            z=centers_array[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color='yellow',
                symbol='diamond',
                line=dict(color='black', width=2)
            ),
            name='物体中心',
            hovertext=[f'物体 {i}' for i in range(len(obj_centers))],
        ))
        
        # 设置布局
        fig.update_layout(
            title='场景图 3D 可视化（可交互）',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        html_path = os.path.join(html_dir, "scene_graph_interactive.html")
        fig.write_html(html_path)
        print(f"  ✓ 交互式 HTML 已保存: {html_path}")
        
    else:
        # 使用 matplotlib 创建静态 3D 图
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        classes = objects.get_most_common_class()
        colors_rgb = [class_colors[str(c)] for c in classes]
        
        # 绘制物体点云（采样）
        for i, (obj, color) in enumerate(zip(objects, colors_rgb)):
            pcd = obj['pcd']
            points = np.asarray(pcd.points)
            
            # 采样点
            if len(points) > 500:
                indices = np.random.choice(len(points), 500, replace=False)
                points = points[indices]
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=[color], s=1, alpha=0.6)
        
        # 绘制场景图的边
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']
            
            ax.plot([obj_centers[id1][0], obj_centers[id2][0]],
                   [obj_centers[id1][1], obj_centers[id2][1]],
                   [obj_centers[id1][2], obj_centers[id2][2]],
                   'r-', linewidth=2)
        
        # 绘制节点
        centers_array = np.array(obj_centers)
        ax.scatter(centers_array[:, 0], centers_array[:, 1], centers_array[:, 2],
                  c='yellow', s=100, marker='D', edgecolors='black', linewidths=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('场景图 3D 可视化')
        
        html_path = os.path.join(html_dir, "scene_graph_static.png")
        plt.savefig(html_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 静态图像已保存: {html_path}")
        plt.close()
    
    print(f"\n✓ HTML 可视化已保存到: {html_dir}")
    print(f"  提示: 在浏览器中打开 HTML 文件即可交互查看")


def main(args):
    print("=" * 80)
    print("离线场景图可视化")
    print("=" * 80)
    
    # 检查输入文件
    if not os.path.exists(args.result_path):
        print(f"✗ 错误: 场景图文件不存在: {args.result_path}")
        sys.exit(1)
    
    if args.edge_file and not os.path.exists(args.edge_file):
        print(f"✗ 错误: 物体关系文件不存在: {args.edge_file}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n输出目录: {args.output_dir}")
    
    # 加载数据
    print(f"\n加载场景图: {args.result_path}")
    objects, bg_objects, class_colors = load_result(args.result_path)
    print(f"  ✓ 加载了 {len(objects)} 个物体")
    
    # 降采样点云
    print("\n降采样点云...")
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        pcd = pcd.voxel_down_sample(0.05)
        objects[i]['pcd'] = pcd
    print("  ✓ 降采样完成")
    
    # 加载场景图数据
    scene_graph_geometries = []
    obj_centers = []
    edges = []
    
    if args.edge_file:
        print(f"\n加载物体关系: {args.edge_file}")
        with open(args.edge_file, "r") as f:
            edges = json.load(f)
        print(f"  ✓ 加载了 {len(edges)} 个关系")
        
        # 创建场景图几何对象
        scene_graph_geometries, obj_centers = create_scene_graph_geometries(
            objects, edges, class_colors
        )
        print(f"  ✓ 创建了场景图几何对象")
    
    # 根据输出格式生成可视化
    if args.output_format in ["images", "all"]:
        # 传递original_mesh_path给save_as_images（用于后备方案）
        save_as_images.original_mesh_path = args.original_mesh
        save_as_images(objects, bg_objects, scene_graph_geometries, 
                      args.output_dir, args.image_width, args.image_height, 
                      args.num_views)
    
    if args.output_format in ["ply", "all"]:
        save_as_ply(objects, scene_graph_geometries, args.output_dir)
    
    if args.output_format in ["html", "all"] and args.edge_file:
        save_as_html(objects, obj_centers, edges, class_colors, args.output_dir)
    
    print("\n" + "=" * 80)
    print("✓ 可视化完成!")
    print("=" * 80)
    print(f"\n所有输出文件保存在: {args.output_dir}")
    
    # 生成摘要信息
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("场景图可视化摘要\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"输入文件:\n")
        f.write(f"  - 场景图: {args.result_path}\n")
        f.write(f"  - 关系: {args.edge_file}\n\n")
        f.write(f"统计信息:\n")
        f.write(f"  - 物体数量: {len(objects)}\n")
        f.write(f"  - 关系数量: {len(edges)}\n\n")
        f.write(f"输出格式: {args.output_format}\n")
        f.write(f"输出目录: {args.output_dir}\n")
    
    print(f"\n摘要信息已保存: {summary_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

