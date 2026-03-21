#!/usr/bin/env python3
"""Prune heavy per-frame GSA detection caches after 3D mapping is complete."""

from __future__ import annotations

import argparse
import gzip
import io
import pickle
from pathlib import Path


DEFAULT_KEEP_KEYS = (
    "xyxy",
    "confidence",
    "class_id",
    "classes",
    "frame_clip_feat",
    "tagging_caption",
    "tagging_text_prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections_dir",
        type=Path,
        required=True,
        help="Directory containing per-frame *.pkl.gz detection caches.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Estimate size reduction without rewriting files.",
    )
    parser.add_argument(
        "--keep_keys",
        nargs="*",
        default=list(DEFAULT_KEEP_KEYS),
        help="Keys to keep in each detection payload.",
    )
    return parser.parse_args()


def _prune_payload(payload: object, keep_keys: set[str]) -> dict:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload).__name__}")
    return {key: value for key, value in payload.items() if key in keep_keys}


def _estimate_pruned_size(pruned: dict) -> int:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
        pickle.dump(pruned, gz, protocol=pickle.HIGHEST_PROTOCOL)
    return len(buffer.getvalue())


def _rewrite_payload(path: Path, pruned: dict) -> int:
    tmp_path = path.with_name(path.name + ".tmp")
    with gzip.open(tmp_path, "wb") as f:
        pickle.dump(pruned, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_size = tmp_path.stat().st_size
    tmp_path.replace(path)
    return tmp_size


def main() -> None:
    args = parse_args()
    keep_keys = set(args.keep_keys)

    files = sorted(args.detections_dir.glob("*.pkl.gz"))
    if not files:
        raise FileNotFoundError(f"No *.pkl.gz files found in {args.detections_dir}")

    original_total = 0
    pruned_total = 0
    rewritten = 0

    for path in files:
        original_total += path.stat().st_size
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        pruned = _prune_payload(payload, keep_keys)

        if args.dry_run:
            pruned_total += _estimate_pruned_size(pruned)
            continue

        pruned_total += _rewrite_payload(path, pruned)
        rewritten += 1

    reduction = 100.0 * (1.0 - pruned_total / original_total)
    mode = "DRY RUN" if args.dry_run else "DONE"
    print(
        {
            "mode": mode,
            "detections_dir": str(args.detections_dir),
            "num_files": len(files),
            "rewritten": rewritten,
            "keep_keys": sorted(keep_keys),
            "original_bytes": original_total,
            "pruned_bytes": pruned_total,
            "original_gib": round(original_total / 1024**3, 3),
            "pruned_gib": round(pruned_total / 1024**3, 3),
            "reduction_pct": round(reduction, 2),
        }
    )


if __name__ == "__main__":
    main()
