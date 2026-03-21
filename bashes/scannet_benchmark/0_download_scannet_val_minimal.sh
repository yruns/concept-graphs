#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MANIFEST_PATH="${1:-${ROOT_DIR}/data/benchmark/manifests/scannet_scene_manifest.json}"
SET_NAME="${2:-scanrefer_val}"

bash "${SCRIPT_DIR}/0_download_scannet_raw_subset.sh" "${MANIFEST_PATH}" "${SET_NAME}"
