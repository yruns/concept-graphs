#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST_PATH="${1:-${ROOT_DIR}/data/benchmark/manifests/scannet_scene_manifest.json}"
SET_NAME="${2:-union_val_test}"

DEFAULT_SCANNET_DOWNLOAD_SCRIPT="${ROOT_DIR}/tools/scannet/download-scannet.py"
SCANNET_DOWNLOAD_SCRIPT="${SCANNET_DOWNLOAD_SCRIPT:-${DEFAULT_SCANNET_DOWNLOAD_SCRIPT}}"
SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
SCANNET_DOWNLOADER_PYTHON="${SCANNET_DOWNLOADER_PYTHON:-python}"
SCANNET_FILE_TYPES="${SCANNET_FILE_TYPES:-.sens,_vh_clean_2.ply}"

if [[ -z "${SCANNET_DOWNLOAD_SCRIPT}" ]]; then
    echo "[ERROR] Set SCANNET_DOWNLOAD_SCRIPT to a compatible download-scannet.py path."
    exit 1
fi

if [[ ! -f "${SCANNET_DOWNLOAD_SCRIPT}" ]]; then
    echo "[ERROR] SCANNET_DOWNLOAD_SCRIPT not found: ${SCANNET_DOWNLOAD_SCRIPT}"
    exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "[ERROR] Manifest not found: ${MANIFEST_PATH}"
    exit 1
fi

mkdir -p "${SCANNET_RAW_ROOT}"

mapfile -t SCENES < <(
    python - "${MANIFEST_PATH}" "${SET_NAME}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
set_name = sys.argv[2]
for item in manifest["scene_sets"][set_name]:
    print(item)
PY
)

IFS=',' read -r -a TYPES <<< "${SCANNET_FILE_TYPES}"

echo "[INFO] Downloading ${#SCENES[@]} ScanNet scenes into ${SCANNET_RAW_ROOT}"
echo "[INFO] File types: ${SCANNET_FILE_TYPES}"

for scene_id in "${SCENES[@]}"; do
    echo "================================================"
    echo "[SCENE] ${scene_id}"
    echo "================================================"
    for file_type in "${TYPES[@]}"; do
        echo "[TYPE] ${file_type}"
        "${SCANNET_DOWNLOADER_PYTHON}" "${SCANNET_DOWNLOAD_SCRIPT}" \
            -o "${SCANNET_RAW_ROOT}" \
            --id "${scene_id}" \
            --type "${file_type}"
    done
done

echo "[DONE] Raw ScanNet subset download finished"
