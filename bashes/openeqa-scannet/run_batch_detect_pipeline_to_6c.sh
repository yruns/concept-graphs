#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

OPENEQA_SCANNET_FRAMES_ROOT="${OPENEQA_SCANNET_FRAMES_ROOT:-${HOME}/Datasets/open-eqa/data/frames/scannet-v0}"
OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
OPENEQA_SCANNET_CONFIG_PATH="${OPENEQA_SCANNET_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml}"
OPENEQA_PROCESS_STRIDE="${OPENEQA_PROCESS_STRIDE:-2}"
OPENEQA_PRUNE_DETECTIONS="${OPENEQA_PRUNE_DETECTIONS:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CLIP_LIMIT="${CLIP_LIMIT:-}"
CLIP_START_AFTER="${CLIP_START_AFTER:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/bashes/logs/openeqa_scannet_batch}"
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-15000}"
GPU_WAIT_SECS="${GPU_WAIT_SECS:-30}"
WORKER_MODE="${WORKER_MODE:-static}"
QUEUE_DIR="${QUEUE_DIR:-}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
WORKER_TAG="${WORKER_TAG:-g${CUDA_DEVICE}}"

mkdir -p "${LOG_DIR}" "${OPENEQA_SCANNET_ROOT}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAIN_LOG="${LOG_DIR}/openeqa_scannet_${RUN_STAMP}_${WORKER_TAG}.log"
FAIL_LOG="${LOG_DIR}/openeqa_scannet_${RUN_STAMP}_${WORKER_TAG}_failures.log"

load_clip_subset() {
    python - "${OPENEQA_SCANNET_FRAMES_ROOT}" "${CLIP_LIMIT}" "${CLIP_START_AFTER}" "${WORKER_INDEX}" "${WORKER_COUNT}" <<'PY'
import sys
from pathlib import Path

frames_root = Path(sys.argv[1])
limit_raw = sys.argv[2]
start_after = sys.argv[3]
worker_index = int(sys.argv[4])
worker_count = int(sys.argv[5])
clips = sorted(path.name for path in frames_root.iterdir() if path.is_dir())
if start_after:
    try:
        idx = clips.index(start_after)
        clips = clips[idx + 1 :]
    except ValueError:
        pass
if limit_raw:
    clips = clips[: int(limit_raw)]
if worker_count < 1:
    raise ValueError("WORKER_COUNT must be >= 1")
if worker_index < 0 or worker_index >= worker_count:
    raise ValueError("WORKER_INDEX must be in [0, WORKER_COUNT)")
for clip in clips[worker_index::worker_count]:
    print(clip)
PY
}

count_clip_subset() {
    python - "${OPENEQA_SCANNET_FRAMES_ROOT}" "${CLIP_LIMIT}" "${CLIP_START_AFTER}" <<'PY'
import sys
from pathlib import Path

frames_root = Path(sys.argv[1])
limit_raw = sys.argv[2]
start_after = sys.argv[3]
clips = sorted(path.name for path in frames_root.iterdir() if path.is_dir())
if start_after:
    try:
        idx = clips.index(start_after)
        clips = clips[idx + 1 :]
    except ValueError:
        pass
if limit_raw:
    clips = clips[: int(limit_raw)]
print(len(clips))
PY
}

claim_next_clip() {
    python - "${OPENEQA_SCANNET_FRAMES_ROOT}" "${CLIP_LIMIT}" "${CLIP_START_AFTER}" "${QUEUE_DIR}" <<'PY'
import fcntl
import json
import sys
from pathlib import Path

frames_root, limit_raw, start_after, queue_dir_raw = sys.argv[1:5]
queue_dir = Path(queue_dir_raw)
queue_dir.mkdir(parents=True, exist_ok=True)
lock_path = queue_dir / "queue.lock"
state_path = queue_dir / "state.json"
clips = sorted(path.name for path in Path(frames_root).iterdir() if path.is_dir())
if start_after:
    try:
        idx = clips.index(start_after)
        clips = clips[idx + 1 :]
    except ValueError:
        pass
if limit_raw:
    clips = clips[: int(limit_raw)]

with lock_path.open("a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"next_index": 0}
    next_index = int(state.get("next_index", 0))
    if next_index >= len(clips):
        print("")
    else:
        clip = clips[next_index]
        state["next_index"] = next_index + 1
        state["total"] = len(clips)
        state["last_clip"] = clip
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        print(clip)
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
PY
}

clip_done() {
    local clip_id="$1"
    local scene_path="${OPENEQA_SCANNET_ROOT}/${clip_id}/conceptgraph"
    local pcd_file="${scene_path}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz"
    local vis_file="${scene_path}/indices/visibility_index.pkl"
    local build_info="${scene_path}/indices/build_info.json"
    if [[ ! -f "${pcd_file}" || ! -f "${vis_file}" || ! -f "${build_info}" ]]; then
        return 1
    fi
    python - "${build_info}" "${OPENEQA_PROCESS_STRIDE}" <<'PY'
import json
import sys
from pathlib import Path

build_info = json.loads(Path(sys.argv[1]).read_text())
expected_stride = int(sys.argv[2])
if int(build_info.get("process_stride", -1)) != expected_stride:
    raise SystemExit(1)
PY
}

prune_if_needed() {
    local clip_id="$1"
    if [[ ! "${OPENEQA_PRUNE_DETECTIONS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
        return 0
    fi
    OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT}" \
        bash "${SCRIPT_DIR}/6c_prune_detection_cache.sh" "${clip_id}" >> "${MAIN_LOG}" 2>&1
}

wait_for_gpu_memory() {
    if [[ -z "${FREE_MEM_MIN_MIB}" || "${FREE_MEM_MIN_MIB}" == "0" ]]; then
        return 0
    fi

    local free_mib
    while true; do
        free_mib="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F',' -v target="${CUDA_DEVICE}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($1 + 0) == (target + 0)) {print $2; exit}}')"
        free_mib="${free_mib:-0}"
        echo "[GPU] cuda=${CUDA_DEVICE} free=${free_mib}MiB threshold=${FREE_MEM_MIN_MIB}MiB" | tee -a "${MAIN_LOG}"
        if (( free_mib >= FREE_MEM_MIN_MIB )); then
            return 0
        fi
        sleep "${GPU_WAIT_SECS}"
    done
}

check_free_space() {
    local free_kb
    free_kb="$(df -Pk "${OPENEQA_SCANNET_ROOT}" | awk 'NR==2 {print $4}')"
    local free_gb=$((free_kb / 1024 / 1024))
    echo "[SPACE] free=${free_gb}GB threshold=${MIN_FREE_GB}GB" | tee -a "${MAIN_LOG}"
    if (( free_gb < MIN_FREE_GB )); then
        echo "[STOP] Free space below threshold; stopping batch run." | tee -a "${MAIN_LOG}"
        exit 2
    fi
}

process_clip() {
    local clip_id="$1"
    check_free_space
    echo "================================================" | tee -a "${MAIN_LOG}"
    echo "[CLIP] ${clip_id}" | tee -a "${MAIN_LOG}"
    echo "================================================" | tee -a "${MAIN_LOG}"

    if clip_done "${clip_id}"; then
        echo "[SKIP] clip already completed" | tee -a "${MAIN_LOG}"
        prune_if_needed "${clip_id}"
        return 0
    fi

    if ! OPENEQA_SCANNET_FRAMES_ROOT="${OPENEQA_SCANNET_FRAMES_ROOT}" \
        OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT}" \
        SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT}" \
        OPENEQA_SCANNET_CONFIG_PATH="${OPENEQA_SCANNET_CONFIG_PATH}" \
        OPENEQA_PROCESS_STRIDE="${OPENEQA_PROCESS_STRIDE}" \
        OPENEQA_PRUNE_DETECTIONS="${OPENEQA_PRUNE_DETECTIONS}" \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        bash "${SCRIPT_DIR}/run_full_detect_pipeline_to_6c.sh" "${clip_id}" >> "${MAIN_LOG}" 2>&1; then
        echo "[FAIL] ${clip_id}" | tee -a "${MAIN_LOG}" | tee -a "${FAIL_LOG}"
        if [[ "${STOP_ON_ERROR}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
            exit 1
        fi
        return 1
    fi

    echo "[DONE] ${clip_id}" | tee -a "${MAIN_LOG}"
}

TOTAL_CLIPS="$(count_clip_subset)"
echo "[START] Batch OpenEQA ScanNet pipeline" | tee -a "${MAIN_LOG}"
echo "[INFO] frames_root=${OPENEQA_SCANNET_FRAMES_ROOT}" | tee -a "${MAIN_LOG}"
echo "[INFO] clips=${TOTAL_CLIPS}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_mode=${WORKER_MODE}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_tag=${WORKER_TAG}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_index=${WORKER_INDEX} worker_count=${WORKER_COUNT}" | tee -a "${MAIN_LOG}"
echo "[INFO] output_root=${OPENEQA_SCANNET_ROOT}" | tee -a "${MAIN_LOG}"
echo "[INFO] raw_root=${SCANNET_RAW_ROOT}" | tee -a "${MAIN_LOG}"
echo "[INFO] process_stride=${OPENEQA_PROCESS_STRIDE} cuda=${CUDA_DEVICE}" | tee -a "${MAIN_LOG}"
echo "[INFO] gpu_wait_secs=${GPU_WAIT_SECS} free_mem_min_mib=${FREE_MEM_MIN_MIB}" | tee -a "${MAIN_LOG}"

if [[ "${WORKER_MODE}" == "queue" ]]; then
    if [[ -z "${QUEUE_DIR}" ]]; then
        echo "[ERROR] WORKER_MODE=queue requires QUEUE_DIR" | tee -a "${MAIN_LOG}"
        exit 1
    fi
    echo "[INFO] queue_dir=${QUEUE_DIR}" | tee -a "${MAIN_LOG}"
    while true; do
        check_free_space
        wait_for_gpu_memory
        clip_id="$(claim_next_clip)"
        if [[ -z "${clip_id}" ]]; then
            break
        fi
        process_clip "${clip_id}" || true
    done
else
    mapfile -t CLIPS < <(load_clip_subset)
    for clip_id in "${CLIPS[@]}"; do
        wait_for_gpu_memory
        process_clip "${clip_id}" || true
    done
fi

echo "[DONE] Batch OpenEQA ScanNet pipeline finished" | tee -a "${MAIN_LOG}"
echo "[LOG] ${MAIN_LOG}"
echo "[FAILURES] ${FAIL_LOG}"
