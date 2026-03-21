#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST_PATH="${1:-${ROOT_DIR}/data/benchmark/manifests/scannet_scene_manifest.json}"
SET_NAME="${2:-scanrefer_val}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "[ERROR] Manifest not found: ${MANIFEST_PATH}"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT:-${HOME}/Datasets/ScanNet}"
SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP:-5}"
SCANNET_PROCESS_STRIDE="${SCANNET_PROCESS_STRIDE:-1}"
SCANNET_PRUNE_DETECTIONS="${SCANNET_PRUNE_DETECTIONS:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SCENE_LIMIT="${SCENE_LIMIT:-}"
SCENE_START_AFTER="${SCENE_START_AFTER:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/bashes/logs/scannet_batch}"
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-15000}"
GPU_WAIT_SECS="${GPU_WAIT_SECS:-30}"
WORKER_MODE="${WORKER_MODE:-static}"
QUEUE_DIR="${QUEUE_DIR:-}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
WORKER_TAG="${WORKER_TAG:-g${CUDA_DEVICE}}"

mkdir -p "${LOG_DIR}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAIN_LOG="${LOG_DIR}/${SET_NAME}_${RUN_STAMP}_${WORKER_TAG}.log"
FAIL_LOG="${LOG_DIR}/${SET_NAME}_${RUN_STAMP}_${WORKER_TAG}_failures.log"

load_scene_subset() {
    python - "${MANIFEST_PATH}" "${SET_NAME}" "${SCENE_LIMIT}" "${SCENE_START_AFTER}" "${WORKER_INDEX}" "${WORKER_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
set_name = sys.argv[2]
limit_raw = sys.argv[3]
start_after = sys.argv[4]
worker_index = int(sys.argv[5])
worker_count = int(sys.argv[6])
scenes = list(manifest["scene_sets"][set_name])
if start_after:
    try:
        idx = scenes.index(start_after)
        scenes = scenes[idx + 1 :]
    except ValueError:
        pass
if limit_raw:
    scenes = scenes[: int(limit_raw)]
if worker_count < 1:
    raise ValueError("WORKER_COUNT must be >= 1")
if worker_index < 0 or worker_index >= worker_count:
    raise ValueError("WORKER_INDEX must be in [0, WORKER_COUNT)")
for scene in scenes[worker_index::worker_count]:
    print(scene)
PY
}

count_scene_subset() {
    python - "${MANIFEST_PATH}" "${SET_NAME}" "${SCENE_LIMIT}" "${SCENE_START_AFTER}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
set_name = sys.argv[2]
limit_raw = sys.argv[3]
start_after = sys.argv[4]
scenes = list(manifest["scene_sets"][set_name])
if start_after:
    try:
        idx = scenes.index(start_after)
        scenes = scenes[idx + 1 :]
    except ValueError:
        pass
if limit_raw:
    scenes = scenes[: int(limit_raw)]
print(len(scenes))
PY
}

claim_next_scene() {
    python - "${MANIFEST_PATH}" "${SET_NAME}" "${SCENE_LIMIT}" "${SCENE_START_AFTER}" "${QUEUE_DIR}" <<'PY'
import fcntl
import json
import sys
from pathlib import Path

manifest_path, set_name, limit_raw, start_after, queue_dir_raw = sys.argv[1:6]
queue_dir = Path(queue_dir_raw)
queue_dir.mkdir(parents=True, exist_ok=True)
lock_path = queue_dir / "queue.lock"
state_path = queue_dir / "state.json"
manifest = json.loads(Path(manifest_path).read_text())
scenes = list(manifest["scene_sets"][set_name])

if start_after:
    try:
        idx = scenes.index(start_after)
        scenes = scenes[idx + 1 :]
    except ValueError:
        pass
if limit_raw:
    scenes = scenes[: int(limit_raw)]

with lock_path.open("a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"next_index": 0}
    next_index = int(state.get("next_index", 0))
    if next_index >= len(scenes):
        print("")
    else:
        scene = scenes[next_index]
        state["next_index"] = next_index + 1
        state["total"] = len(scenes)
        state["last_scene"] = scene
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        print(scene)
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
PY
}

scene_done() {
    local scene_name="$1"
    local scene_path="${SCANNET_SCENE_ROOT}/${scene_name}"
    local pcd_file="${scene_path}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz"
    local vis_file="${scene_path}/indices/visibility_index.pkl"
    [[ -f "${pcd_file}" && -f "${vis_file}" ]]
}

prune_if_needed() {
    local scene_name="$1"
    if [[ ! "${SCANNET_PRUNE_DETECTIONS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
        return 0
    fi
    SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT}" \
        bash "${SCRIPT_DIR}/6c_prune_detection_cache.sh" "${scene_name}" >> "${MAIN_LOG}" 2>&1
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
    free_kb="$(df -Pk "${SCANNET_SCENE_ROOT}" | awk 'NR==2 {print $4}')"
    local free_gb=$((free_kb / 1024 / 1024))
    echo "[SPACE] free=${free_gb}GB threshold=${MIN_FREE_GB}GB" | tee -a "${MAIN_LOG}"
    if (( free_gb < MIN_FREE_GB )); then
        echo "[STOP] Free space below threshold; stopping batch run." | tee -a "${MAIN_LOG}"
        exit 2
    fi
}

process_scene() {
    local scene_name="$1"
    check_free_space
    echo "================================================" | tee -a "${MAIN_LOG}"
    echo "[SCENE] ${scene_name}" | tee -a "${MAIN_LOG}"
    echo "================================================" | tee -a "${MAIN_LOG}"

    if scene_done "${scene_name}"; then
        echo "[SKIP] scene already completed" | tee -a "${MAIN_LOG}"
        prune_if_needed "${scene_name}"
        return 0
    fi

    if ! SCANNET_RAW_ROOT="${SCANNET_RAW_ROOT}" \
        SCANNET_SCENE_ROOT="${SCANNET_SCENE_ROOT}" \
        SCANNET_FRAME_SKIP="${SCANNET_FRAME_SKIP}" \
        SCANNET_PROCESS_STRIDE="${SCANNET_PROCESS_STRIDE}" \
        SCANNET_PRUNE_DETECTIONS="${SCANNET_PRUNE_DETECTIONS}" \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        bash "${SCRIPT_DIR}/run_full_detect_pipeline_to_6b.sh" "${scene_name}" >> "${MAIN_LOG}" 2>&1; then
        echo "[FAIL] ${scene_name}" | tee -a "${MAIN_LOG}" | tee -a "${FAIL_LOG}"
        if [[ "${STOP_ON_ERROR}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
            exit 1
        fi
        return 1
    fi

    echo "[DONE] ${scene_name}" | tee -a "${MAIN_LOG}"
}

TOTAL_SCENES="$(count_scene_subset)"
echo "[START] Batch ScanNet pipeline" | tee -a "${MAIN_LOG}"
echo "[INFO] manifest=${MANIFEST_PATH}" | tee -a "${MAIN_LOG}"
echo "[INFO] set=${SET_NAME} scenes=${TOTAL_SCENES}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_mode=${WORKER_MODE}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_tag=${WORKER_TAG}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_index=${WORKER_INDEX} worker_count=${WORKER_COUNT}" | tee -a "${MAIN_LOG}"
echo "[INFO] raw_root=${SCANNET_RAW_ROOT}" | tee -a "${MAIN_LOG}"
echo "[INFO] scene_root=${SCANNET_SCENE_ROOT}" | tee -a "${MAIN_LOG}"
echo "[INFO] frame_skip=${SCANNET_FRAME_SKIP} process_stride=${SCANNET_PROCESS_STRIDE} cuda=${CUDA_DEVICE}" | tee -a "${MAIN_LOG}"
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
        scene_name="$(claim_next_scene)"
        if [[ -z "${scene_name}" ]]; then
            break
        fi
        process_scene "${scene_name}" || true
    done
else
    mapfile -t SCENES < <(load_scene_subset)
    for scene_name in "${SCENES[@]}"; do
        wait_for_gpu_memory
        process_scene "${scene_name}" || true
    done
fi

echo "[DONE] Batch ScanNet pipeline finished" | tee -a "${MAIN_LOG}"
echo "[LOG] ${MAIN_LOG}"
echo "[FAILURES] ${FAIL_LOG}"
