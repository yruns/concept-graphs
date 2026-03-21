#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OPENEQA_SCANNET_FRAMES_ROOT="${OPENEQA_SCANNET_FRAMES_ROOT:-${HOME}/Datasets/open-eqa/data/frames/scannet-v0}"
SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
POLL_SECS="${POLL_SECS:-60}"

while true; do
    missing_count="$(
        python - "${OPENEQA_SCANNET_FRAMES_ROOT}" "${SCANNET_RAW_ROOT}" <<'PY'
import re
import sys
from pathlib import Path

frames_root = Path(sys.argv[1])
raw_root = Path(sys.argv[2])
missing = 0
for clip_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
    match = re.match(r"^\d+-scannet-(scene\d+_\d+)$", clip_dir.name)
    if not match:
        continue
    scene_id = match.group(1)
    mesh_found = any(
        (raw_root / folder / scene_id / f"{scene_id}_vh_clean_2.ply").exists()
        for folder in ("scans", "scans_test")
    )
    if not mesh_found:
        missing += 1
print(missing)
PY
    )"

    echo "[WAIT] missing_mesh_scenes=${missing_count}"
    if [[ "${missing_count}" == "0" ]]; then
        break
    fi
    sleep "${POLL_SECS}"
done

CLIP_LIMIT="${CLIP_LIMIT:-10}" \
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}" \
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-15000}" \
SESSION_PREFIX="${SESSION_PREFIX:-openeqa_scannet_batch_missing}" \
bash "${SCRIPT_DIR}/launch_multi_gpu_batch_to_6c.sh"
