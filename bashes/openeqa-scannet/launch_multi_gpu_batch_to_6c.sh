#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OPENEQA_SCANNET_FRAMES_ROOT="${OPENEQA_SCANNET_FRAMES_ROOT:-${HOME}/Datasets/open-eqa/data/frames/scannet-v0}"
OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
OPENEQA_SCANNET_CONFIG_PATH="${OPENEQA_SCANNET_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml}"
OPENEQA_PROCESS_STRIDE="${OPENEQA_PROCESS_STRIDE:-2}"
OPENEQA_PRUNE_DETECTIONS="${OPENEQA_PRUNE_DETECTIONS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-15000}"
GPU_WAIT_SECS="${GPU_WAIT_SECS:-30}"
CLIP_LIMIT="${CLIP_LIMIT:-}"
CLIP_START_AFTER="${CLIP_START_AFTER:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
WORKER_MODE="${WORKER_MODE:-queue}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/bashes/logs/openeqa_scannet_batch}"
QUEUE_ROOT="${QUEUE_ROOT:-${LOG_DIR}/queues}"
QUEUE_DIR="${QUEUE_DIR:-}"
GPU_LIST="${GPU_LIST:-}"
SESSION_PREFIX="${SESSION_PREFIX:-openeqa_scannet_batch}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${QUEUE_ROOT}"

if [[ -z "${QUEUE_DIR}" ]]; then
    QUEUE_DIR="${QUEUE_ROOT}/openeqa_scannet_${RUN_STAMP}"
fi
mkdir -p "${QUEUE_DIR}"

if [[ -n "${GPU_LIST}" ]]; then
    IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
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
    session_name="${SESSION_PREFIX}_g${gpu}"
    tmux kill-session -t "${session_name}" 2>/dev/null || true
    tmux new-session -d -s "${session_name}" \
        "bash -lc 'cd ${ROOT_DIR} && \
        OPENEQA_SCANNET_FRAMES_ROOT=${OPENEQA_SCANNET_FRAMES_ROOT} \
        OPENEQA_SCANNET_ROOT=${OPENEQA_SCANNET_ROOT} \
        SCANNET_RAW_ROOT=${SCANNET_RAW_ROOT} \
        OPENEQA_SCANNET_CONFIG_PATH=${OPENEQA_SCANNET_CONFIG_PATH} \
        OPENEQA_PROCESS_STRIDE=${OPENEQA_PROCESS_STRIDE} \
        OPENEQA_PRUNE_DETECTIONS=${OPENEQA_PRUNE_DETECTIONS} \
        MIN_FREE_GB=${MIN_FREE_GB} \
        FREE_MEM_MIN_MIB=${FREE_MEM_MIN_MIB} \
        GPU_WAIT_SECS=${GPU_WAIT_SECS} \
        CLIP_LIMIT=${CLIP_LIMIT} \
        CLIP_START_AFTER=${CLIP_START_AFTER} \
        STOP_ON_ERROR=${STOP_ON_ERROR} \
        WORKER_MODE=${WORKER_MODE} \
        QUEUE_DIR=${QUEUE_DIR} \
        RUN_STAMP=${RUN_STAMP} \
        CUDA_DEVICE=${gpu} \
        WORKER_INDEX=${idx} \
        WORKER_COUNT=${#GPUS[@]} \
        WORKER_TAG=g${gpu} \
        LOG_DIR=${LOG_DIR} \
        bash bashes/openeqa-scannet/run_batch_detect_pipeline_to_6c.sh'"
    echo "[LAUNCHED] ${session_name} -> GPU ${gpu}"
done

tmux ls | grep "${SESSION_PREFIX}" || true
