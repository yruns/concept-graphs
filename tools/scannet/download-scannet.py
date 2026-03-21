#!/usr/bin/env python3
"""Download a minimal public ScanNet subset with the official script interface.

This is a small compatibility wrapper for the non-public official
``download-scannet.py`` script. It supports the arguments used in this repo:

    python download-scannet.py -o /path/to/ScanNet --id scene0011_00 --type .sens
    python download-scannet.py -o /path/to/ScanNet --id scene0011_00 --type _vh_clean_2.ply

Files are written to:
    <out_dir>/scans/<scene_id>/<scene_id><file_type>
or, when only the test URL exists:
    <out_dir>/scans_test/<scene_id>/<scene_id><file_type>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


BASE_URL = "https://kaldir.vc.in.tum.de/scannet"
TOS_URL = f"{BASE_URL}/ScanNet_TOS.pdf"
FILETYPES = {
    ".aggregation.json",
    ".sens",
    ".txt",
    "_vh_clean.ply",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.ply",
    "_vh_clean.segs.json",
    "_vh_clean.aggregation.json",
    "_vh_clean_2.labels.ply",
    "_2d-instance.zip",
    "_2d-instance-filt.zip",
    "_2d-label.zip",
    "_2d-label-filt.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        help="Directory in which to download ScanNet data.",
    )
    parser.add_argument("--id", required=True, help="Specific ScanNet scene id to download.")
    parser.add_argument(
        "--type",
        required=True,
        dest="file_type",
        help="Specific file type to download, e.g. .sens or _vh_clean_2.ply.",
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Prefer v1 URLs. Useful for older assets.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=5,
        help="Retry count for each candidate URL.",
    )
    return parser.parse_args()


def build_candidates(scene_id: str, file_type: str, use_v1: bool) -> list[tuple[str, str]]:
    rel_paths: list[str] = []
    if use_v1:
        rel_paths.extend(
            [
                f"v1/scans/{scene_id}/{scene_id}{file_type}",
                f"v1/scans_test/{scene_id}/{scene_id}{file_type}",
            ]
        )
    else:
        if file_type == ".sens":
            rel_paths.extend(
                [
                    f"v1/scans/{scene_id}/{scene_id}{file_type}",
                    f"v2/scans/{scene_id}/{scene_id}{file_type}",
                    f"v2/scans_test/{scene_id}/{scene_id}{file_type}",
                ]
            )
        else:
            rel_paths.extend(
                [
                    f"v2/scans/{scene_id}/{scene_id}{file_type}",
                    f"v2/scans_test/{scene_id}/{scene_id}{file_type}",
                    f"v1/scans/{scene_id}/{scene_id}{file_type}",
                    f"v1/scans_test/{scene_id}/{scene_id}{file_type}",
                ]
            )

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rel_path in rel_paths:
        if rel_path in seen:
            continue
        seen.add(rel_path)
        subset = "scans_test" if "/scans_test/" in rel_path else "scans"
        deduped.append((f"{BASE_URL}/{rel_path}", subset))
    return deduped


def run_curl(url: str, out_path: Path, retry: int) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_name(out_path.name + ".part")

    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(retry),
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--speed-time",
        "30",
        "--speed-limit",
        "1024",
        "--continue-at",
        "-",
        "-o",
        str(partial_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if partial_path.exists() and partial_path.stat().st_size == 0:
            partial_path.unlink()
        return False, (result.stderr or result.stdout).strip()

    partial_path.replace(out_path)
    return True, ""


def download_one(scene_id: str, out_root: Path, file_type: str, use_v1: bool, retry: int) -> Path:
    errors: list[str] = []
    for url, subset in build_candidates(scene_id, file_type, use_v1):
        out_path = out_root / subset / scene_id / f"{scene_id}{file_type}"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[SKIP] {out_path}")
            return out_path

        print(f"[TRY] {url}")
        ok, error = run_curl(url=url, out_path=out_path, retry=retry)
        if ok:
            print(f"[OK] {out_path}")
            return out_path

        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
        errors.append(f"{url}: {error}")

    joined = "\n".join(errors)
    raise RuntimeError(
        f"Failed to download {scene_id}{file_type} from all candidate URLs.\n{joined}"
    )


def main() -> int:
    args = parse_args()

    if shutil.which("curl") is None:
        raise RuntimeError("curl is required but was not found in PATH.")

    if args.file_type not in FILETYPES:
        raise ValueError(f"Unsupported file type: {args.file_type}")

    print(f"[INFO] ScanNet terms of use: {TOS_URL}")
    print(f"[INFO] Downloading {args.id}{args.file_type}")
    out_path = download_one(
        scene_id=args.id,
        out_root=Path(args.out_dir),
        file_type=args.file_type,
        use_v1=args.v1,
        retry=args.retry,
    )
    print(f"[DONE] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
