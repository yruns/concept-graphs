#!/usr/bin/env python
"""Build parser_sft.jsonl and retrieval_eval.jsonl from open-world assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conceptgraph.query_scene.open_world_sample_builder import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_TEACHER_MODELS,
    build_samples_from_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build open-world training/eval samples")
    parser.add_argument(
        "--scene_manifest",
        type=Path,
        default=Path("plans/generated_open_world/scene_manifest.jsonl"),
    )
    parser.add_argument(
        "--query_program_pool",
        type=Path,
        default=Path("plans/generated_open_world/query_program_pool.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("plans/generated_open_world"),
    )
    parser.add_argument("--samples_per_scene", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_teacher_llm",
        action="store_true",
        help="Enable dual-teacher LLM query generation with cache/retry",
    )
    parser.add_argument(
        "--teacher_models",
        type=str,
        default=",".join(DEFAULT_TEACHER_MODELS),
        help="Comma-separated teacher models",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default=DEFAULT_PROMPT_VERSION,
    )
    parser.add_argument("--teacher_temperature", type=float, default=0.2)
    parser.add_argument("--teacher_max_retries", type=int, default=2)
    parser.add_argument(
        "--teacher_cache",
        type=Path,
        default=None,
        help="Optional cache path, default: <output_dir>/teacher_query_cache.jsonl",
    )
    args = parser.parse_args()

    teacher_models = [m.strip() for m in args.teacher_models.split(",") if m.strip()]

    summary = build_samples_from_assets(
        scene_manifest_path=args.scene_manifest,
        query_program_pool_path=args.query_program_pool,
        output_dir=args.output_dir,
        samples_per_scene=args.samples_per_scene,
        seed=args.seed,
        use_teacher_llm=args.use_teacher_llm,
        teacher_models=teacher_models,
        prompt_version=args.prompt_version,
        teacher_temperature=args.teacher_temperature,
        teacher_max_retries=args.teacher_max_retries,
        teacher_cache_path=args.teacher_cache,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
