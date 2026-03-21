#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/run_after_1b_to_6b.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCANNET_PROCESS_STRIDE="${SCANNET_PROCESS_STRIDE:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
CUDA_DEVICE="${CUDA_DEVICE:-}"
SCANNET_PRUNE_DETECTIONS="${SCANNET_PRUNE_DETECTIONS:-1}"

SCENE_PATH="${SCANNET_SCENE_ROOT}/${SCENE_NAME}"
DETECTION_DIR="${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"

if [[ ! -d "${SCENE_PATH}/color" ]]; then
    echo "[ERROR] Missing wrapped scene color directory: ${SCENE_PATH}/color"
    exit 1
fi

TARGET_DETECTIONS="$(
    python - "${SCENE_PATH}" "${SCANNET_PROCESS_STRIDE}" <<'PY'
import sys
from pathlib import Path

scene_path = Path(sys.argv[1])
stride = int(sys.argv[2])
num_frames = len(list((scene_path / "color").glob("frame*.jpg")))
print((num_frames + stride - 1) // stride)
PY
)"

echo "[WAIT] scene=${SCENE_NAME} target_detections=${TARGET_DETECTIONS} stride=${SCANNET_PROCESS_STRIDE}"

while true; do
    count="$(
        find "${DETECTION_DIR}" -maxdepth 1 -name '*.pkl.gz' 2>/dev/null \
            | wc -l \
            | tr -d ' '
    )"
    proc="$(pgrep -f "generate_gsa_results.py.*--scene_id ${SCENE_NAME}" || true)"
    echo "[WAIT] detections=${count}/${TARGET_DETECTIONS} proc=${proc:-none}"
    if [[ "${count}" -ge "${TARGET_DETECTIONS}" ]] && [[ -z "${proc}" ]]; then
        break
    fi
    sleep "${WAIT_SECONDS}"
done

if [[ -n "${CUDA_DEVICE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
fi
export SCANNET_SCENE_ROOT
export SCANNET_PROCESS_STRIDE

bash "${SCRIPT_DIR}/2b_build_3d_object_map_detect.sh" "${SCENE_NAME}"
bash "${SCRIPT_DIR}/6b_build_visibility_index.sh" "${SCENE_NAME}"

if [[ "${SCANNET_PRUNE_DETECTIONS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    bash "${SCRIPT_DIR}/6c_prune_detection_cache.sh" "${SCENE_NAME}"
fi

echo "[DONE] Continued ScanNet pipeline to 6b for ${SCENE_NAME}"
