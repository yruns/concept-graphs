#!/usr/bin/env python3
"""Build a ScanNet scene manifest for multi-benchmark evaluation.

This script consolidates the ScanNet scene requirements across:
- OpenEQA (ScanNet split only)
- SQA3D (val/test)
- ScanRefer (val/test when available; otherwise fall back to ScanNet splits)

Outputs:
- scannet_scene_manifest.json
- scannet_union_val_test.txt
- benchmark_scene_coverage.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_text_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_openeqa_scannet_scenes(openeqa_json: Path) -> Set[str]:
    data = _read_json(openeqa_json)
    scenes = set()
    for item in data:
        episode_history = item.get("episode_history", "")
        if not episode_history.startswith("scannet-v0/"):
            continue
        episode_dir = episode_history.split("/")[-1]
        scene_id = episode_dir.split("-")[-1]
        if scene_id:
            scenes.add(scene_id)
    return scenes


def _load_sqa3d_split_scenes(scene_split_json: Path) -> Dict[str, Set[str]]:
    data = _read_json(scene_split_json)
    return {
        "val": set(data.get("val", [])),
        "test": set(data.get("test", [])),
    }


def _load_scanrefer_scenes(scanrefer_root: Path) -> Dict[str, Set[str] | str]:
    scanrefer_data_dir = scanrefer_root / "data"
    val_json = scanrefer_data_dir / "ScanRefer_filtered_val.json"
    test_json = scanrefer_data_dir / "ScanRefer_filtered_test.json"

    result: Dict[str, Set[str] | str] = {
        "val_source": "",
        "test_source": "",
    }

    if val_json.exists():
        val_data = _read_json(val_json)
        result["val"] = {item["scene_id"] for item in val_data if item.get("scene_id")}
        result["val_source"] = str(val_json)
    else:
        val_txt = (
            scanrefer_root
            / "data"
            / "scannet"
            / "meta_data"
            / "scannetv2_val.txt"
        )
        result["val"] = set(_read_text_lines(val_txt))
        result["val_source"] = (
            f"{val_txt} (fallback: official ScanNet val split, "
            "ScanRefer val annotations not present)"
        )

    if test_json.exists():
        test_data = _read_json(test_json)
        result["test"] = {item["scene_id"] for item in test_data if item.get("scene_id")}
        result["test_source"] = str(test_json)
    else:
        test_txt = (
            scanrefer_root
            / "data"
            / "scannet"
            / "meta_data"
            / "scannetv2_test.txt"
        )
        result["test"] = set(_read_text_lines(test_txt))
        result["test_source"] = (
            f"{test_txt} (fallback: official ScanNet test split, "
            "ScanRefer test annotations not present)"
        )

    return result


def _sorted_list(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


def _save_text_list(path: Path, values: Iterable[str]) -> None:
    path.write_text("\n".join(_sorted_list(values)) + "\n", encoding="utf-8")


def _write_summary_plot(output_path: Path, counts: Dict[str, int]) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(labels, values, color=["#2A6F97", "#61A5C2", "#89C2D9", "#014F86", "#D90429"])
    ax.set_title("ScanNet Scene Coverage Across Benchmarks")
    ax.set_ylabel("Unique Scene Count")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(values) * 0.02,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_manifest(benchmark_root: Path, output_dir: Path) -> Path:
    scanrefer_root = benchmark_root / "ScanRefer"
    sqa3d_root = benchmark_root / "SQA3D"
    openeqa_root = benchmark_root / "open-eqa"

    openeqa_json = openeqa_root / "data" / "open-eqa-v0.json"
    sqa3d_scene_split = sqa3d_root / "assets" / "data" / "scene_split.json"

    openeqa_scannet = _load_openeqa_scannet_scenes(openeqa_json)
    sqa3d = _load_sqa3d_split_scenes(sqa3d_scene_split)
    scanrefer = _load_scanrefer_scenes(scanrefer_root)

    union_val_test = (
        set(scanrefer["val"])
        | set(scanrefer["test"])
        | sqa3d["val"]
        | sqa3d["test"]
        | openeqa_scannet
    )

    manifest = {
        "sources": {
            "scanrefer_val": scanrefer["val_source"],
            "scanrefer_test": scanrefer["test_source"],
            "sqa3d_scene_split": str(sqa3d_scene_split),
            "openeqa_dataset": str(openeqa_json),
        },
        "scene_sets": {
            "scanrefer_val": _sorted_list(scanrefer["val"]),
            "scanrefer_test": _sorted_list(scanrefer["test"]),
            "sqa3d_val": _sorted_list(sqa3d["val"]),
            "sqa3d_test": _sorted_list(sqa3d["test"]),
            "openeqa_scannet": _sorted_list(openeqa_scannet),
            "union_val_test": _sorted_list(union_val_test),
        },
        "counts": {
            "scanrefer_val": len(scanrefer["val"]),
            "scanrefer_test": len(scanrefer["test"]),
            "sqa3d_val": len(sqa3d["val"]),
            "sqa3d_test": len(sqa3d["test"]),
            "openeqa_scannet": len(openeqa_scannet),
            "union_val_test": len(union_val_test),
        },
        "download_plan": {
            "scannet_required_file_types": [".sens", "_vh_clean_2.ply"],
            "notes": [
                "OpenEQA ScanNet scenes are a strict subset of ScanRefer val/test ScanNet scenes.",
                "SQA3D val/test scenes are a strict subset of ScanRefer val/test ScanNet scenes.",
                "Downloading ScanNet val+test scenes is sufficient for multi-benchmark scene graph preprocessing.",
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "scannet_scene_manifest.json"
    union_txt_path = output_dir / "scannet_union_val_test.txt"
    plot_path = output_dir / "benchmark_scene_coverage.png"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    _save_text_list(union_txt_path, union_val_test)
    _write_summary_plot(
        plot_path,
        {
            "ScanRefer val": len(scanrefer["val"]),
            "ScanRefer test": len(scanrefer["test"]),
            "SQA3D val+test": len(sqa3d["val"] | sqa3d["test"]),
            "OpenEQA ScanNet": len(openeqa_scannet),
            "Union": len(union_val_test),
        },
    )

    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("data/benchmark"),
        help="Root directory containing ScanRefer / SQA3D / open-eqa.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/benchmark/manifests"),
        help="Directory for manifest outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = build_manifest(args.benchmark_root, args.output_dir)
    print(f"[OK] Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
