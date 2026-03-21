#!/usr/bin/env bash

# ============================================================================
# run_full_detect_pipeline_to_6b.sh
# End-to-end detect pipeline runner:
#   clean scene outputs -> 1b -> 2b -> 4b -> 5b+ -> 6b
#
# Safety:
#   - Always preserves: scene/results/, scene/result/, scene/traj.txt
#   - Deletes other files/directories directly under the scene folder
#
# Usage:
#   bash bashes/run_full_detect_pipeline_to_6b.sh room1
# ============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/run_full_detect_pipeline_to_6b.sh <scene_name>"
    exit 1
fi

REPLICA_ROOT="${REPLICA_ROOT:-${HOME}/Datasets/Replica}"
SCENE_PATH="${REPLICA_ROOT}/${SCENE_NAME}"

THRESHOLD="1.2"
PCD_POST_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
CAPTION_FILE="${SCENE_PATH}/sg_cache_detect/cfslam_llava_captions.json"
AFFORDANCE_FILE="${SCENE_PATH}/sg_cache_detect/object_affordances.json"
INDEX_FILE="${SCENE_PATH}/indices/visibility_index.pkl"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_full_detect_to_6b_${SCENE_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

die() {
    echo "[ERROR] $*"
    exit 1
}

info() {
    echo "[INFO] $*"
}

run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "============================================================"
    echo "[STEP] ${step_name}"
    echo "============================================================"
    "$@"
}

assert_file_exists() {
    local file_path="$1"
    local hint="$2"
    [[ -f "${file_path}" ]] || die "Missing file: ${file_path}. ${hint}"
}

assert_dir_exists() {
    local dir_path="$1"
    local hint="$2"
    [[ -d "${dir_path}" ]] || die "Missing directory: ${dir_path}. ${hint}"
}

assert_glob_nonempty() {
    local glob_pattern="$1"
    local hint="$2"
    shopt -s nullglob
    local matches=(${glob_pattern})
    shopt -u nullglob
    (( ${#matches[@]} > 0 )) || die "No files matched: ${glob_pattern}. ${hint}"
}

clean_scene_outputs() {
    info "Cleaning scene outputs under: ${SCENE_PATH}"
    assert_dir_exists "${SCENE_PATH}" "Scene path does not exist."

    if [[ ! -d "${SCENE_PATH}/results" && ! -d "${SCENE_PATH}/result" ]]; then
        die "Neither ${SCENE_PATH}/results nor ${SCENE_PATH}/result exists."
    fi
    assert_file_exists "${SCENE_PATH}/traj.txt" "traj.txt must exist and will be preserved."

    mapfile -t cleanup_targets < <(
        find "${SCENE_PATH}" -mindepth 1 -maxdepth 1 \
            ! -name "results" \
            ! -name "result" \
            ! -name "traj.txt" \
            -print
    )

    if (( ${#cleanup_targets[@]} == 0 )); then
        info "No stale outputs found. Nothing to clean."
        return 0
    fi

    info "Deleting ${#cleanup_targets[@]} stale paths (preserving results/result + traj.txt)."
    printf '  - %s\n' "${cleanup_targets[@]}"
    rm -rf -- "${cleanup_targets[@]}"
}

check_json_nonempty() {
    local json_path="$1"
    python - "${json_path}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, (list, dict)) and len(data) == 0:
    raise SystemExit(f"JSON exists but empty: {path}")
print(f"[CHECK] JSON ok: {path} (type={type(data).__name__}, size={len(data) if hasattr(data, '__len__') else 'n/a'})")
PY
}

check_visibility_index() {
    local index_path="$1"
    python - "${index_path}" <<'PY'
import pickle
import sys
path = sys.argv[1]
with open(path, "rb") as f:
    data = pickle.load(f)
if not isinstance(data, dict):
    raise SystemExit("visibility_index.pkl is not a dict")
obj_to_views = data.get("object_to_views", {})
view_to_objs = data.get("view_to_objects", {})
meta = data.get("metadata", {})
if len(obj_to_views) == 0 or len(view_to_objs) == 0:
    raise SystemExit("visibility index looks empty")
print(f"[CHECK] visibility_index ok: objects={len(obj_to_views)}, views={len(view_to_objs)}, metadata_keys={list(meta.keys())}")
PY
}

info "Scene: ${SCENE_NAME}"
info "Replica root: ${REPLICA_ROOT}"
info "Scene path: ${SCENE_PATH}"
info "Log file: ${LOG_FILE}"

run_step "Clean scene outputs (preserve results/result + traj.txt)" clean_scene_outputs

run_step "1B - Extract 2D detections" \
    bash "${SCRIPT_DIR}/1b_extract_2d_segmentation_detect.sh" "${SCENE_NAME}"
assert_glob_nonempty "${SCENE_PATH}/gsa_detections_ram_withbg_allclasses/*.pkl.gz" "Step 1B output missing."

run_step "2B - Build 3D object map" \
    bash "${SCRIPT_DIR}/2b_build_3d_object_map_detect.sh" "${SCENE_NAME}"
assert_file_exists "${PCD_POST_FILE}" "Step 2B post map missing."

run_step "4B - Extract object captions" \
    bash "${SCRIPT_DIR}/4b_extract_object_captions_detect.sh" "${SCENE_NAME}"
assert_file_exists "${CAPTION_FILE}" "Step 4B caption file missing."
check_json_nonempty "${CAPTION_FILE}"

run_step "5B+ - Refine with affordance" \
    bash "${SCRIPT_DIR}/5b_refine_with_affordance.sh" "${SCENE_NAME}"
assert_file_exists "${AFFORDANCE_FILE}" "Step 5B+ affordance file missing."
check_json_nonempty "${AFFORDANCE_FILE}"

run_step "6B - Build visibility index" \
    bash "${SCRIPT_DIR}/6b_build_visibility_index.sh" "${SCENE_NAME}"
assert_file_exists "${INDEX_FILE}" "Step 6B visibility index missing."
check_visibility_index "${INDEX_FILE}"

echo ""
echo "============================================================"
echo "[DONE] Full detect pipeline to 6B completed for scene: ${SCENE_NAME}"
echo "============================================================"
echo "Key outputs:"
echo "  - ${PCD_POST_FILE}"
echo "  - ${CAPTION_FILE}"
echo "  - ${AFFORDANCE_FILE}"
echo "  - ${INDEX_FILE}"
echo "  - Log: ${LOG_FILE}"
