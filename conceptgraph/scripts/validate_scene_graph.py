#!/usr/bin/env python3
"""Validate scene graph outputs with point-cloud and index visualizations."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--top_views", type=int, default=3)
    parser.add_argument("--top_objects", type=int, default=5)
    parser.add_argument("--max_objects_per_view", type=int, default=12)
    parser.add_argument("--max_views_per_object", type=int, default=6)
    parser.add_argument("--points_per_object", type=int, default=1500)
    return parser.parse_args()


def load_scene_graph(scene_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pcd_file = next(scene_path.joinpath("pcd_saves").glob("*_post.pkl.gz"))
    with gzip.open(pcd_file, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected scene graph format: {type(data)}")

    index_file = scene_path / "indices" / "visibility_index.pkl"
    with open(index_file, "rb") as handle:
        visibility = pickle.load(handle)

    build_info_file = scene_path / "indices" / "build_info.json"
    build_info = {}
    if build_info_file.exists():
        build_info = json.loads(build_info_file.read_text(encoding="utf-8"))

    return data["objects"], visibility, data, build_info


def get_process_stride(build_info: dict[str, Any]) -> int:
    stride = int(build_info.get("process_stride", 1) or 1)
    return max(stride, 1)


def resolve_view_image_paths(scene_path: Path, process_stride: int, max_view_id: int) -> list[Path]:
    candidates = [
        sorted(scene_path.glob("*-rgb.png")),
        sorted(scene_path.glob("results/frame*.jpg")),
        sorted(scene_path.glob("results/*.png")),
        sorted(scene_path.glob("color/*.jpg")),
        sorted(scene_path.glob("color/*.png")),
    ]
    all_paths = next((paths for paths in candidates if paths), [])
    if not all_paths:
        raise FileNotFoundError(f"No RGB frames found under {scene_path}")

    strided = all_paths[::process_stride]
    if max_view_id < len(strided):
        return strided
    if max_view_id < len(all_paths):
        return all_paths
    raise ValueError(
        f"Cannot map view ids to frames: max_view_id={max_view_id}, "
        f"all_paths={len(all_paths)}, process_stride={process_stride}"
    )


def object_label(obj: dict[str, Any], obj_id: int) -> str:
    raw_names = obj.get("class_name", [])
    if isinstance(raw_names, str):
        raw_names = [raw_names]
    names = [
        str(name)
        for name in raw_names
        if name is not None and str(name).strip().lower() not in {"", "item", "object", "none"}
    ]
    if names:
        return Counter(names).most_common(1)[0][0]
    return f"object_{obj_id}"


def object_color(obj_id: int, total_objects: int) -> np.ndarray:
    hue = (obj_id / max(total_objects, 1)) % 1.0
    rgb = matplotlib.colors.hsv_to_rgb((hue, 0.75, 0.95))
    return rgb.astype(np.float32)


def sample_points(points: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = rng.choice(len(points), size=limit, replace=False)
    return points[indices]


def best_bbox_for_view(obj: dict[str, Any], view_id: int) -> list[int] | None:
    image_indices = obj.get("image_idx", [])
    xyxy_list = obj.get("xyxy", [])
    candidates: list[tuple[float, list[int]]] = []
    for idx, image_idx in enumerate(image_indices):
        if image_idx != view_id or idx >= len(xyxy_list):
            continue
        xyxy = xyxy_list[idx]
        if xyxy is None or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        candidates.append((float(area), [x1, y1, x2, y2]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_object_frame_map(
    objects: list[dict[str, Any]],
    visibility: dict[str, Any],
    view_paths: list[Path],
    process_stride: int,
) -> dict[str, Any]:
    object_to_views = visibility.get("object_to_views", {})
    view_to_objects = visibility.get("view_to_objects", {})

    object_to_frames: dict[str, Any] = {}
    for obj_id, views in object_to_views.items():
        obj_id_int = int(obj_id)
        obj = objects[obj_id_int]
        entries = []
        for view_id, score in views:
            view_id_int = int(view_id)
            if view_id_int >= len(view_paths):
                continue
            entries.append(
                {
                    "view_id": view_id_int,
                    "frame_name": view_paths[view_id_int].name,
                    "score": float(score),
                    "bbox_xyxy": best_bbox_for_view(obj, view_id_int),
                }
            )
        object_to_frames[str(obj_id_int)] = {
            "object_id": obj_id_int,
            "class_name": object_label(obj, obj_id_int),
            "num_views": len(entries),
            "frames": entries,
        }

    frame_to_objects: dict[str, Any] = {}
    for view_id, obj_scores in view_to_objects.items():
        view_id_int = int(view_id)
        if view_id_int >= len(view_paths):
            continue
        objects_here = []
        for obj_id, score in obj_scores:
            obj_id_int = int(obj_id)
            obj = objects[obj_id_int]
            objects_here.append(
                {
                    "object_id": obj_id_int,
                    "class_name": object_label(obj, obj_id_int),
                    "score": float(score),
                    "bbox_xyxy": best_bbox_for_view(obj, view_id_int),
                }
            )
        frame_to_objects[str(view_id_int)] = {
            "view_id": view_id_int,
            "frame_name": view_paths[view_id_int].name,
            "num_objects": len(objects_here),
            "objects": objects_here,
        }

    return {
        "metadata": {
            "process_stride": process_stride,
            "num_objects": len(object_to_frames),
            "num_views": len(frame_to_objects),
        },
        "object_to_frames": object_to_frames,
        "frame_to_objects": frame_to_objects,
    }


def save_colored_scene_point_cloud(
    objects: list[dict[str, Any]],
    output_dir: Path,
    points_per_object: int,
    highlight_object_ids: list[int],
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    all_points = []
    all_colors = []
    centroids = []
    legend = []

    for obj_id, obj in enumerate(objects):
        points = np.asarray(obj.get("pcd_np"))
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            continue
        sampled = sample_points(points, points_per_object, rng)
        rgb = object_color(obj_id, len(objects))
        colors = np.repeat(rgb[None, :], len(sampled), axis=0)
        all_points.append(sampled)
        all_colors.append(colors)
        centroid = sampled.mean(axis=0)
        centroids.append((obj_id, centroid))
        legend.append(
            {
                "object_id": obj_id,
                "class_name": object_label(obj, obj_id),
                "rgb": [int(channel * 255) for channel in rgb],
                "num_points": int(len(sampled)),
            }
        )

    if not all_points:
        raise ValueError("No object point clouds available for validation")

    points_np = np.vstack(all_points)
    colors_np = np.vstack(all_colors)

    output_dir.mkdir(parents=True, exist_ok=True)
    if o3d is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_np)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_np)
        o3d.io.write_point_cloud(str(output_dir / "scene_objects_colored.ply"), point_cloud)

    fig = plt.figure(figsize=(16, 5), dpi=140)
    view_angles = [(20, 45), (25, 135), (75, -90)]
    top_label_ids = set(highlight_object_ids)
    for axis_idx, (elev, azim) in enumerate(view_angles, start=1):
        ax = fig.add_subplot(1, 3, axis_idx, projection="3d")
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors_np, s=0.8, alpha=0.9)
        for obj_id, centroid in centroids:
            if obj_id not in top_label_ids:
                continue
            ax.text(
                centroid[0],
                centroid[1],
                centroid[2],
                f"{obj_id}",
                fontsize=7,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1},
            )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {axis_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.tight_layout()
    fig.savefig(output_dir / "scene_objects_colored_preview.png", bbox_inches="tight")
    plt.close(fig)

    (output_dir / "scene_objects_colored_legend.json").write_text(
        json.dumps(legend, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "preview_png": str(output_dir / "scene_objects_colored_preview.png"),
        "ply_path": str(output_dir / "scene_objects_colored.ply") if o3d is not None else None,
    }


def draw_text_block(canvas: np.ndarray, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
    current_y = y
    for line in lines:
        cv2.putText(canvas, line, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        current_y += 22


def visualize_top_views(
    objects: list[dict[str, Any]],
    visibility: dict[str, Any],
    view_paths: list[Path],
    output_dir: Path,
    top_views: int,
    max_objects_per_view: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    view_to_objects = visibility.get("view_to_objects", {})
    ranked_views = sorted(
        ((int(view_id), obj_scores) for view_id, obj_scores in view_to_objects.items()),
        key=lambda item: (len(item[1]), sum(score for _, score in item[1])),
        reverse=True,
    )
    selected = []
    palette = [
        (240, 80, 80),
        (80, 180, 80),
        (80, 120, 240),
        (220, 180, 60),
        (180, 80, 200),
        (60, 180, 200),
        (255, 140, 70),
        (120, 120, 255),
        (255, 100, 180),
        (120, 200, 120),
        (200, 200, 90),
        (90, 200, 200),
    ]

    for view_id, obj_scores in ranked_views[:top_views]:
        if view_id >= len(view_paths):
            continue
        image = cv2.imread(str(view_paths[view_id]))
        if image is None:
            continue
        annotated = image.copy()
        panel = np.full((image.shape[0], 480, 3), 255, dtype=np.uint8)
        lines = [f"view_id={view_id}", f"frame={view_paths[view_id].name}", f"num_objects={len(obj_scores)}", ""]

        for rank, (obj_id, score) in enumerate(obj_scores[:max_objects_per_view]):
            obj_id_int = int(obj_id)
            obj = objects[obj_id_int]
            color = palette[rank % len(palette)]
            bbox = best_bbox_for_view(obj, view_id)
            label = object_label(obj, obj_id_int)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"{obj_id_int}:{label[:18]}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            lines.append(f"{obj_id_int:02d} {label[:24]} score={score:.2f}")

        draw_text_block(panel, lines, 18, 30, (20, 20, 20))
        combined = np.concatenate([annotated, panel], axis=1)
        out_path = output_dir / f"view_{view_id:04d}_objects.png"
        cv2.imwrite(str(out_path), combined)
        selected.append({"view_id": view_id, "frame_name": view_paths[view_id].name, "output_path": str(out_path)})

    return selected


def make_grid(images: list[np.ndarray], columns: int) -> np.ndarray:
    if not images:
        raise ValueError("No images to grid")
    rows = math.ceil(len(images) / columns)
    height, width = images[0].shape[:2]
    blank = np.full_like(images[0], 255)
    padded = images + [blank] * (rows * columns - len(images))
    row_images = []
    for row_idx in range(rows):
        row = padded[row_idx * columns : (row_idx + 1) * columns]
        row_images.append(np.concatenate(row, axis=1))
    return np.concatenate(row_images, axis=0)


def visualize_top_objects(
    objects: list[dict[str, Any]],
    visibility: dict[str, Any],
    view_paths: list[Path],
    output_dir: Path,
    top_objects: int,
    max_views_per_object: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    object_to_views = visibility.get("object_to_views", {})
    ranked_objects = sorted(
        ((int(obj_id), views) for obj_id, views in object_to_views.items()),
        key=lambda item: (len(item[1]), len(objects[item[0]].get("image_idx", []))),
        reverse=True,
    )
    selected = []

    for obj_id, views in ranked_objects[:top_objects]:
        obj = objects[obj_id]
        label = object_label(obj, obj_id)
        tiles = []
        used_views = []
        for view_id, score in views[:max_views_per_object]:
            view_id_int = int(view_id)
            if view_id_int >= len(view_paths):
                continue
            image = cv2.imread(str(view_paths[view_id_int]))
            if image is None:
                continue
            bbox = best_bbox_for_view(obj, view_id_int)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (30, 30, 240), 3)
            header = np.full((70, image.shape[1], 3), 255, dtype=np.uint8)
            draw_text_block(
                header,
                [
                    f"object={obj_id} class={label[:26]}",
                    f"view={view_id_int} frame={view_paths[view_id_int].name}",
                    f"score={score:.3f}",
                ],
                12,
                22,
                (20, 20, 20),
            )
            tile = np.concatenate([header, image], axis=0)
            tile = cv2.resize(tile, (560, 420))
            tiles.append(tile)
            used_views.append(view_id_int)

        if not tiles:
            continue

        grid = make_grid(tiles, columns=3)
        out_path = output_dir / f"object_{obj_id:03d}_views.png"
        cv2.imwrite(str(out_path), grid)
        selected.append(
            {
                "object_id": obj_id,
                "class_name": label,
                "view_ids": used_views,
                "output_path": str(out_path),
            }
        )

    return selected


def main() -> None:
    args = parse_args()
    scene_path = args.scene_path.resolve()
    output_dir = (args.output_dir or scene_path / "checks" / "03_scene_graph_validation").resolve()

    objects, visibility, _, build_info = load_scene_graph(scene_path)
    process_stride = get_process_stride(build_info)
    max_view_id = max((int(view_id) for view_id in visibility.get("view_to_objects", {}).keys()), default=-1)
    view_paths = resolve_view_image_paths(scene_path, process_stride, max_view_id)

    ranked_object_ids = sorted(
        range(len(objects)),
        key=lambda obj_id: len(objects[obj_id].get("image_idx", [])),
        reverse=True,
    )[: args.top_objects]

    point_cloud_info = save_colored_scene_point_cloud(
        objects=objects,
        output_dir=output_dir / "pointcloud",
        points_per_object=args.points_per_object,
        highlight_object_ids=ranked_object_ids,
    )

    top_view_outputs = visualize_top_views(
        objects=objects,
        visibility=visibility,
        view_paths=view_paths,
        output_dir=output_dir / "view_to_objects",
        top_views=args.top_views,
        max_objects_per_view=args.max_objects_per_view,
    )

    top_object_outputs = visualize_top_objects(
        objects=objects,
        visibility=visibility,
        view_paths=view_paths,
        output_dir=output_dir / "object_to_views",
        top_objects=args.top_objects,
        max_views_per_object=args.max_views_per_object,
    )

    object_frame_map = build_object_frame_map(
        objects=objects,
        visibility=visibility,
        view_paths=view_paths,
        process_stride=process_stride,
    )
    object_frame_map_path = scene_path / "indices" / "object_frame_map.json"
    object_frame_map_path.write_text(
        json.dumps(object_frame_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "scene_path": str(scene_path),
        "process_stride": process_stride,
        "num_objects": len(objects),
        "num_views": len(view_paths),
        "pointcloud": point_cloud_info,
        "top_view_outputs": top_view_outputs,
        "top_object_outputs": top_object_outputs,
        "object_frame_map_path": str(object_frame_map_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
