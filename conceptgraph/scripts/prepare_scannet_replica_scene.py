#!/usr/bin/env python3
"""Prepare a ScanNet scene in a Replica-like layout for ConceptGraph.

The output scene keeps native ScanNet inputs for the dataset loader:
- color/frameXXXXXX.jpg
- depth/depthXXXXXX.png
- pose/frameXXXXXX.txt
- intrinsic/intrinsic_color.txt

And also creates Replica-compatible paths for downstream Stage 1 tooling:
- results/frameXXXXXX.jpg
- results/depthXXXXXX.png
- traj.txt
- <scene_id>_mesh.ply (placed in output_root)
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw


def _load_sensor_data_class(repo_root: Path):
    candidate = repo_root / "data" / "benchmark" / "open-eqa" / "data" / "scannet" / "SensorData.py"
    if not candidate.exists():
        raise FileNotFoundError(
            "Missing SensorData.py. Run the public benchmark asset download first: "
            f"{candidate}"
        )
    spec = importlib.util.spec_from_file_location("scannet_sensor_data", candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import SensorData from {candidate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SensorData


def _find_raw_scene_dir(scannet_root: Path, scene_id: str) -> Path:
    for folder in ("scans", "scans_test"):
        candidate = scannet_root / folder / scene_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Raw ScanNet scene not found under {scannet_root}/scans or scans_test: {scene_id}"
    )


def _extract_frame_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse frame id from {path}")


def _write_intrinsics(target_dir: Path, sensor_data) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(target_dir / "intrinsic_color.txt", sensor_data.intrinsic_color, fmt="%.8f")
    np.savetxt(target_dir / "extrinsic_color.txt", sensor_data.extrinsic_color, fmt="%.8f")
    np.savetxt(target_dir / "intrinsic_depth.txt", sensor_data.intrinsic_depth, fmt="%.8f")
    np.savetxt(target_dir / "extrinsic_depth.txt", sensor_data.extrinsic_depth, fmt="%.8f")


def _sample_frame_indices(total_frames: int, frame_skip: int, max_frames: int | None) -> List[int]:
    indices = list(range(0, total_frames, frame_skip))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def _link_or_copy(src: Path, dst: Path, prefer_symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def _write_traj(poses: Sequence[np.ndarray], traj_path: Path) -> None:
    with traj_path.open("w", encoding="utf-8") as f:
        for pose in poses:
            flat = pose.reshape(-1)
            f.write(" ".join(f"{value:.8f}" for value in flat))
            f.write("\n")


def _save_contact_sheet(image_paths: Sequence[Path], output_path: Path) -> None:
    if not image_paths:
        return
    picked = [image_paths[0]]
    if len(image_paths) > 2:
        picked.append(image_paths[len(image_paths) // 2])
    if len(image_paths) > 1:
        picked.append(image_paths[-1])

    tiles = []
    labels = ["first", "middle", "last"][: len(picked)]
    for label, path in zip(labels, picked):
        image = Image.open(path).convert("RGB")
        image.thumbnail((640, 480))
        canvas = Image.new("RGB", (image.width, image.height + 28), "white")
        canvas.paste(image, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"{label}: {path.name}", fill="black")
        tiles.append(canvas)

    total_width = sum(tile.width for tile in tiles)
    max_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (total_width, max_height), "white")

    cursor = 0
    for tile in tiles:
        sheet.paste(tile, (cursor, 0))
        cursor += tile.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _extract_scene(
    sensor_data,
    scene_dir: Path,
    results_dir: Path,
    frame_indices: Iterable[int],
    prefer_symlink: bool,
) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for frame_id in frame_indices:
        frame = sensor_data.frames[frame_id]

        color = frame.decompress_color(sensor_data.color_compression_type)
        color_path = color_dir / f"frame{frame_id:06d}.jpg"
        if not color_path.exists():
            Image.fromarray(color).save(color_path, quality=95)

        depth_bytes = frame.decompress_depth(sensor_data.depth_compression_type)
        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
            sensor_data.depth_height,
            sensor_data.depth_width,
        )
        depth_path = depth_dir / f"depth{frame_id:06d}.png"
        if not depth_path.exists():
            Image.fromarray(depth).save(depth_path)

        pose = np.asarray(frame.camera_to_world, dtype=np.float32)
        poses.append(pose)
        pose_path = pose_dir / f"frame{frame_id:06d}.txt"
        if not pose_path.exists():
            np.savetxt(pose_path, pose, fmt="%.8f")

        _link_or_copy(color_path, results_dir / color_path.name, prefer_symlink)
        _link_or_copy(depth_path, results_dir / depth_path.name, prefer_symlink)

    return poses


def _copy_preextracted_scene(
    raw_scene_dir: Path,
    scene_dir: Path,
    results_dir: Path,
    frame_skip: int,
    max_frames: int | None,
    prefer_symlink: bool,
) -> List[np.ndarray]:
    color_candidates = sorted(list((raw_scene_dir / "color").glob("*.jpg")) + list((raw_scene_dir / "color").glob("*.png")))
    depth_candidates = sorted((raw_scene_dir / "depth").glob("*.png"))
    pose_candidates = sorted((raw_scene_dir / "pose").glob("*.txt"))

    if not color_candidates or not depth_candidates or not pose_candidates:
        raise FileNotFoundError(
            "Expected pre-extracted ScanNet folders color/, depth/, pose/ under "
            f"{raw_scene_dir}"
        )

    sampled_positions = _sample_frame_indices(len(color_candidates), frame_skip, max_frames)

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    poses: List[np.ndarray] = []
    for pos in sampled_positions:
        color_src = color_candidates[pos]
        depth_src = depth_candidates[pos]
        pose_src = pose_candidates[pos]
        frame_id = _extract_frame_number(color_src)

        color_dst = color_dir / f"frame{frame_id:06d}.jpg"
        if not color_dst.exists():
            image = Image.open(color_src).convert("RGB")
            image.save(color_dst, quality=95)

        depth_dst = depth_dir / f"depth{frame_id:06d}.png"
        if not depth_dst.exists():
            _link_or_copy(depth_src, depth_dst, prefer_symlink)

        pose = np.loadtxt(pose_src).reshape(4, 4)
        poses.append(pose.astype(np.float32))
        pose_dst = pose_dir / f"frame{frame_id:06d}.txt"
        if not pose_dst.exists():
            np.savetxt(pose_dst, pose, fmt="%.8f")

        _link_or_copy(color_dst, results_dir / color_dst.name, prefer_symlink)
        _link_or_copy(depth_dst, results_dir / depth_dst.name, prefer_symlink)

    for intrinsic_name in (
        "intrinsic_color.txt",
        "extrinsic_color.txt",
        "intrinsic_depth.txt",
        "extrinsic_depth.txt",
    ):
        src = raw_scene_dir / "intrinsic" / intrinsic_name
        if src.exists():
            _link_or_copy(src, intrinsic_dir / intrinsic_name, prefer_symlink)

    return poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_id", required=True, help="ScanNet scene id, e.g. scene0709_00")
    parser.add_argument(
        "--scannet_root",
        type=Path,
        required=True,
        help="Root containing ScanNet scans/ and scans_test/ folders.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Target root for Replica-like wrapped ScanNet scenes.",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Frame stride during extraction. Use 1 for full Replica-like parity.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on the number of extracted frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing wrapped scene before regenerating it.",
    )
    parser.add_argument(
        "--copy_results",
        action="store_true",
        help="Copy files into results/ instead of symlinking them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    raw_scene_dir = _find_raw_scene_dir(args.scannet_root, args.scene_id)
    sens_path = raw_scene_dir / f"{args.scene_id}.sens"
    has_preextracted = all(
        (raw_scene_dir / name).exists() for name in ("color", "depth", "pose", "intrinsic")
    )
    if not sens_path.exists() and not has_preextracted:
        raise FileNotFoundError(
            f"Missing ScanNet inputs. Expected {sens_path} or pre-extracted "
            f"color/depth/pose/intrinsic under {raw_scene_dir}."
        )

    scene_dir = args.output_root / args.scene_id
    results_dir = scene_dir / "results"
    checks_dir = scene_dir / "checks"
    mesh_src = raw_scene_dir / f"{args.scene_id}_vh_clean_2.ply"
    mesh_dst = args.output_root / f"{args.scene_id}_mesh.ply"

    if args.overwrite and scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    if sens_path.exists():
        SensorData = _load_sensor_data_class(repo_root)
        sensor_data = SensorData(str(sens_path))
        frame_indices = _sample_frame_indices(len(sensor_data.frames), args.frame_skip, args.max_frames)
        poses = _extract_scene(
            sensor_data=sensor_data,
            scene_dir=scene_dir,
            results_dir=results_dir,
            frame_indices=frame_indices,
            prefer_symlink=not args.copy_results,
        )
        _write_intrinsics(scene_dir / "intrinsic", sensor_data)
    else:
        poses = _copy_preextracted_scene(
            raw_scene_dir=raw_scene_dir,
            scene_dir=scene_dir,
            results_dir=results_dir,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            prefer_symlink=not args.copy_results,
        )
    _write_traj(poses, scene_dir / "traj.txt")

    if mesh_src.exists() and not mesh_dst.exists():
        _link_or_copy(mesh_src, mesh_dst, prefer_symlink=not args.copy_results)

    result_frames = sorted(results_dir.glob("frame*.jpg"))
    _save_contact_sheet(result_frames, checks_dir / "00_scene_wrapper_preview.jpg")

    print(
        json.dumps(
            {
                "scene_id": args.scene_id,
                "raw_scene_dir": str(raw_scene_dir),
                "output_scene_dir": str(scene_dir),
                "frame_count": len(result_frames),
                "frame_skip": args.frame_skip,
                "traj_file": str(scene_dir / "traj.txt"),
                "mesh_path": str(mesh_dst) if mesh_dst.exists() else None,
                "preview": str(checks_dir / "00_scene_wrapper_preview.jpg"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
