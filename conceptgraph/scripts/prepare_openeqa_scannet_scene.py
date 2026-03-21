#!/usr/bin/env python3
"""Prepare an OpenEQA ScanNet clip scene for ConceptGraph.

The prepared layout is:

OpenEQA/scannet/<clip_id>/
  conceptgraph/
    000000-rgb.png -> source frame
    000000-depth.png -> source frame
    000000.txt -> source pose
    intrinsic_color.txt -> source intrinsic
    intrinsic_depth.txt -> source intrinsic
    extrinsic_color.txt -> source extrinsic
    extrinsic_depth.txt -> source extrinsic
    mesh.ply -> optional ScanNet raw mesh
    checks/00_clip_wrapper_preview.jpg
    scene_info.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


def _parse_scene_id(clip_id: str) -> str:
    match = re.match(r"^\d+-scannet-(scene\d+_\d+)$", clip_id)
    if not match:
        raise ValueError(
            f"Clip id does not match expected OpenEQA ScanNet format: {clip_id}"
        )
    return match.group(1)


def _find_raw_scene_dir(scannet_root: Path | None, scene_id: str) -> Path | None:
    if scannet_root is None:
        return None
    for folder in ("scans", "scans_test"):
        candidate = scannet_root / folder / scene_id
        if candidate.exists():
            return candidate
    return None


def _link_or_copy(src: Path, dst: Path, prefer_symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def _gather_input_files(source_dir: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    rgb_files = sorted(source_dir.glob("*-rgb.png"))
    depth_files = sorted(source_dir.glob("*-depth.png"))
    pose_files = sorted(source_dir.glob("[0-9]*.txt"))
    aux_files = sorted(source_dir.glob("intrinsic*.txt")) + sorted(source_dir.glob("extrinsic*.txt"))
    if not rgb_files or not depth_files or not pose_files:
        raise FileNotFoundError(
            f"Expected *-rgb.png, *-depth.png, and [0-9]*.txt under {source_dir}"
        )
    return rgb_files, depth_files, pose_files, aux_files


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

    sheet = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
    cursor = 0
    for tile in tiles:
        sheet.paste(tile, (cursor, 0))
        cursor += tile.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _write_traj_from_pose_files(pose_files: Sequence[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pose_file in pose_files:
            pose = np.loadtxt(pose_file).reshape(4, 4)
            handle.write(" ".join(f"{value:.8f}" for value in pose.reshape(-1)))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip_id", required=True, help="OpenEQA clip id, e.g. 002-scannet-scene0709_00")
    parser.add_argument(
        "--frames_root",
        type=Path,
        required=True,
        help="Root containing OpenEQA ScanNet clip folders.",
    )
    parser.add_argument(
        "--scannet_root",
        type=Path,
        default=None,
        help="Optional root containing ScanNet scans/ and scans_test/ folders.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Target root containing OpenEQA/scannet/<clip_id>/conceptgraph.",
    )
    parser.add_argument(
        "--prefer_symlink",
        action="store_true",
        help="Use symlinks instead of copying source files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_id = args.clip_id
    source_dir = args.frames_root / clip_id
    if not source_dir.exists():
        raise FileNotFoundError(f"Clip source directory not found: {source_dir}")

    clip_dir = args.output_root / clip_id
    scene_dir = clip_dir / "conceptgraph"
    scene_dir.mkdir(parents=True, exist_ok=True)

    scene_id = _parse_scene_id(clip_id)
    raw_scene_dir = _find_raw_scene_dir(args.scannet_root, scene_id)
    mesh_src = None if raw_scene_dir is None else raw_scene_dir / f"{scene_id}_vh_clean_2.ply"
    mesh_dst = scene_dir / "mesh.ply"

    rgb_files, depth_files, pose_files, aux_files = _gather_input_files(source_dir)
    for path in rgb_files + depth_files + pose_files + aux_files:
        _link_or_copy(path, scene_dir / path.name, args.prefer_symlink)
    if mesh_src is not None and mesh_src.exists():
        _link_or_copy(mesh_src, mesh_dst, args.prefer_symlink)
    _write_traj_from_pose_files(pose_files, scene_dir / "traj.txt")

    preview_path = scene_dir / "checks" / "00_clip_wrapper_preview.jpg"
    _save_contact_sheet(rgb_files, preview_path)

    info = {
        "clip_id": clip_id,
        "scene_id": scene_id,
        "source_dir": str(source_dir),
        "scene_dir": str(scene_dir),
        "mesh_path": str(mesh_dst) if mesh_dst.exists() else None,
        "mesh_available": mesh_dst.exists(),
        "num_rgb_frames": len(rgb_files),
        "num_depth_frames": len(depth_files),
        "num_pose_files": len(pose_files),
        "linked_inputs": args.prefer_symlink,
    }
    (scene_dir / "scene_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[PREVIEW] {preview_path}")


if __name__ == "__main__":
    main()
