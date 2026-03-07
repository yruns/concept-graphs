#!/usr/bin/env python
"""Build scene_manifest.jsonl and query_program_pool.jsonl for open-world training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from conceptgraph.query_scene.open_world_dataset import (
    build_scene_manifest_entry,
    generate_query_program_pool,
    write_jsonl,
)


def _parse_scene_arg(scene_arg: str) -> Tuple[str, Path]:
    """Parse `scene_id=scene_path` or plain `scene_path`."""
    if "=" in scene_arg:
        scene_id, scene_path = scene_arg.split("=", 1)
        return scene_id.strip(), Path(scene_path).expanduser().resolve()

    scene_path = Path(scene_arg).expanduser().resolve()
    return scene_path.name, scene_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build open-world dataset assets")
    parser.add_argument(
        "--scene",
        action="append",
        required=True,
        help="Scene spec: scene_id=/abs/path/to/scene or /abs/path/to/scene",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("plans/generated_open_world"),
        help="Output directory for generated JSONL files",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max_programs_per_scene", type=int, default=300)
    args = parser.parse_args()

    scenes: List[Tuple[str, Path]] = [_parse_scene_arg(s) for s in args.scene]

    manifest_records = []
    all_programs = []

    for scene_id, scene_path in scenes:
        entry = build_scene_manifest_entry(
            scene_path=scene_path,
            scene_id=scene_id,
            stride=args.stride,
        )
        manifest_records.append(entry)

        programs = generate_query_program_pool(
            scene_manifest=entry,
            max_programs_per_scene=args.max_programs_per_scene,
        )
        for idx, program in enumerate(programs):
            program["program_id"] = f"{scene_id}_prog_{idx:06d}"
        all_programs.extend(programs)

    output_dir = args.output_dir.resolve()
    scene_manifest_path = output_dir / "scene_manifest.jsonl"
    query_pool_path = output_dir / "query_program_pool.jsonl"
    report_path = output_dir / "generation_report.md"

    manifest_count = write_jsonl(manifest_records, scene_manifest_path)
    program_count = write_jsonl(all_programs, query_pool_path)

    summary = {
        "scenes": len(manifest_records),
        "scene_manifest_records": manifest_count,
        "query_program_records": program_count,
        "output_dir": str(output_dir),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Open-World Asset Generation Report\n\n")
        for key, value in summary.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n## Scene Summary\n")
        for entry in manifest_records:
            f.write(
                f"- {entry['scene_id']}: categories={entry['num_categories']}, "
                f"objects={entry['num_objects']}, frames={entry['num_frames']}\n"
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
