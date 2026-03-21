#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

OPENEQA_SCANNET_FRAMES_ROOT="${OPENEQA_SCANNET_FRAMES_ROOT:-${HOME}/Datasets/open-eqa/data/frames/scannet-v0}"
SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
DOWNLOAD_SCRIPT="${SCANNET_DOWNLOAD_SCRIPT:-${ROOT_DIR}/tools/scannet/download-scannet.py}"

if [[ ! -f "${DOWNLOAD_SCRIPT}" ]]; then
    echo "[ERROR] Download script not found: ${DOWNLOAD_SCRIPT}"
    exit 1
fi

python - "${OPENEQA_SCANNET_FRAMES_ROOT}" "${SCANNET_RAW_ROOT}" <<'PY' | while read -r scene_id; do
import re
import sys
from pathlib import Path

frames_root = Path(sys.argv[1])
raw_root = Path(sys.argv[2])

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
        print(scene_id)
PY
    echo "[SCENE] ${scene_id}"
    python "${DOWNLOAD_SCRIPT}" -o "${SCANNET_RAW_ROOT}" --id "${scene_id}" --type _vh_clean_2.ply
done

echo "[DONE] Missing OpenEQA ScanNet meshes synced into ${SCANNET_RAW_ROOT}"
