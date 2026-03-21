#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST_PATH="${1:-${ROOT_DIR}/data/benchmark/manifests/scannet_scene_manifest.json}"
SET_NAME="${2:-scanrefer_val}"

SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-5}"
SCANNET_PROCESS_STRIDE="${SCANNET_PROCESS_STRIDE:-1}"
SCANNET_PRUNE_DETECTIONS="${SCANNET_PRUNE_DETECTIONS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-15000}"
GPU_WAIT_SECS="${GPU_WAIT_SECS:-30}"
SCENE_LIMIT="${SCENE_LIMIT:-}"
SCENE_START_AFTER="${SCENE_START_AFTER:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
WORKER_MODE="${WORKER_MODE:-queue}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/bashes/logs/scannet_batch}"
QUEUE_ROOT="${QUEUE_ROOT:-${LOG_DIR}/queues}"
QUEUE_DIR="${QUEUE_DIR:-}"
GPU_LIST="${GPU_LIST:-}"
SESSION_PREFIX="${SESSION_PREFIX:-scannet_batch}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${QUEUE_ROOT}"

if [[ -z "${QUEUE_DIR}" ]]; then
    QUEUE_DIR="${QUEUE_ROOT}/${SET_NAME}_${RUN_STAMP}"
fi
mkdir -p "${QUEUE_DIR}"

if [[ -n "${GPU_LIST}" ]]; then
    IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
else
    mapfile -t GPUS < <(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F',' -v min_mib="${FREE_MEM_MIN_MIB}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($2 + 0) >= (min_mib + 0)) print $1}'
    )
fi

if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "[ERROR] No GPUs selected."
    exit 1
fi

echo "[INFO] Launching ${#GPUS[@]} workers on GPUs: ${GPUS[*]}"
echo "[INFO] worker_mode=${WORKER_MODE}"
echo "[INFO] queue_dir=${QUEUE_DIR}"
echo "[INFO] free_mem_min_mib=${FREE_MEM_MIN_MIB} gpu_wait_secs=${GPU_WAIT_SECS}"

for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    session_name="${SESSION_PREFIX}_${SET_NAME}_g${gpu}"
    tmux kill-session -t "${session_name}" 2>/dev/null || true
    tmux new-session -d -s "${session_name}" \
        "bash -lc 'cd ${ROOT_DIR} && \
        SCANNET_RAW_ROOT=${SCANNET_RAW_ROOT} \
        SCANNET_SCENE_ROOT=${SCANNET_SCENE_ROOT} \
        SCANNET_FRAME_SKIP=${SCANNET_FRAME_SKIP} \
        SCANNET_PROCESS_STRIDE=${SCANNET_PROCESS_STRIDE} \
        SCANNET_PRUNE_DETECTIONS=${SCANNET_PRUNE_DETECTIONS} \
        MIN_FREE_GB=${MIN_FREE_GB} \
        FREE_MEM_MIN_MIB=${FREE_MEM_MIN_MIB} \
        GPU_WAIT_SECS=${GPU_WAIT_SECS} \
        SCENE_LIMIT=${SCENE_LIMIT} \
        SCENE_START_AFTER=${SCENE_START_AFTER} \
        STOP_ON_ERROR=${STOP_ON_ERROR} \
        WORKER_MODE=${WORKER_MODE} \
        QUEUE_DIR=${QUEUE_DIR} \
        RUN_STAMP=${RUN_STAMP} \
        CUDA_DEVICE=${gpu} \
        WORKER_INDEX=${idx} \
        WORKER_COUNT=${#GPUS[@]} \
        WORKER_TAG=g${gpu} \
        LOG_DIR=${LOG_DIR} \
        bash bashes/scannet_benchmark/run_batch_detect_pipeline_to_6c.sh ${MANIFEST_PATH} ${SET_NAME}'"
    echo "[LAUNCHED] ${session_name} -> GPU ${gpu}"
done

tmux ls | grep "${SESSION_PREFIX}_${SET_NAME}" || true
