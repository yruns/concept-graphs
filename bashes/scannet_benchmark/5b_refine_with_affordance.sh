#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_NAME="${1:-}"
if [[ -z "${SCENE_NAME}" ]]; then
    echo "Usage: bash bashes/scannet_benchmark/5b_refine_with_affordance.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

export REPLICA_ROOT="${SCANNET_SCENE_ROOT:-${HOME}/Datasets/ScanNetReplicaLike}"
bash "${ROOT_DIR}/bashes/5b_refine_with_affordance.sh" "${SCENE_NAME}"
