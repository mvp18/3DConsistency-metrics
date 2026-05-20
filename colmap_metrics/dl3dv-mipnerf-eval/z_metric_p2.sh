#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../batch_common.sh"

BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
MODELS_FILE="${MODELS_FILE:-$SCRIPT_DIR/models.txt}"
DL3DV_SCENES_FILE="${DL3DV_SCENES_FILE:-$SCRIPT_DIR/dl3dv_folders.txt}"
MIP_SCENES_FILE="${MIP_SCENES_FILE:-$SCRIPT_DIR/mipnerf360_scenes.txt}"
SPARSE_DIR_NAME="${SPARSE_DIR_NAME:-sparse}"
DENSE_DIR_NAME="${DENSE_DIR_NAME:-dense}"
SPARSE_MODEL_SUBDIR="${SPARSE_MODEL_SUBDIR:-0}"
RESULT_FOLDER="${RESULT_FOLDER:-$SCRIPT_DIR/results_all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
METRICS_SCRIPT="${METRICS_SCRIPT:-$SCRIPT_DIR/metrics.py}"
PHOTOMETRIC_KIND="${PHOTOMETRIC_KIND:-auto}"
RELATIVE_DEPTH_CLIP="${RELATIVE_DEPTH_CLIP:-0.2}"
COVERAGE_PLANE="${COVERAGE_PLANE:-pca}"

require_file "$METRICS_SCRIPT"
load_nonempty_lines "$MODELS_FILE" models
load_nonempty_lines "$DL3DV_SCENES_FILE" dl3dv_scenes
load_csv_entries "$MIP_SCENES_FILE" mip_scenes

echo "Models (${#models[@]}): ${models[*]}"
echo "DL3DV scenes (${#dl3dv_scenes[@]}): ${dl3dv_scenes[*]}"
echo "MipNeRF360 scenes (${#mip_scenes[@]}): ${mip_scenes[*]}"
echo "Results folder: $RESULT_FOLDER"

for m in "${models[@]}"; do
    echo "Computing metrics for model: $m"

    for scene in "${dl3dv_scenes[@]}"; do
        scene_root="$BASE_DIR/$m/dl3dv/colmap_metrics/$scene"
        sparse_root="$scene_root/$SPARSE_DIR_NAME"
        dense_dir="$scene_root/$DENSE_DIR_NAME"
        database_path="$scene_root/database.db"

        require_dir "$dense_dir"
        require_dir "$sparse_root"
        sparse_dir="$(resolve_sparse_model_dir "$sparse_root" "$SPARSE_MODEL_SUBDIR")" || {
            print_error "Could not find a sparse model directory under $sparse_root"
            exit 1
        }

        name="${m}_dl3dv_${scene}"
        echo "Processing DL3DV scene: $scene"
        metrics_args=(
            "$METRICS_SCRIPT"
            --sparse "$sparse_dir"
            --dense "$dense_dir"
            --name "$name"
            --outdir "$RESULT_FOLDER"
            --photometric-kind "$PHOTOMETRIC_KIND"
            --relative-depth-clip "$RELATIVE_DEPTH_CLIP"
            --coverage-plane "$COVERAGE_PLANE"
        )
        if [[ -f "$database_path" ]]; then
            metrics_args+=(--database "$database_path")
        fi
        "$PYTHON_BIN" "${metrics_args[@]}"
    done

    for scene in "${mip_scenes[@]}"; do
        scene_root="$BASE_DIR/$m/mip/colmap_metrics/$scene"
        sparse_root="$scene_root/$SPARSE_DIR_NAME"
        dense_dir="$scene_root/$DENSE_DIR_NAME"
        database_path="$scene_root/database.db"

        require_dir "$dense_dir"
        require_dir "$sparse_root"
        sparse_dir="$(resolve_sparse_model_dir "$sparse_root" "$SPARSE_MODEL_SUBDIR")" || {
            print_error "Could not find a sparse model directory under $sparse_root"
            exit 1
        }

        name="${m}_mip_${scene}"
        echo "Processing MipNeRF360 scene: $scene"
        metrics_args=(
            "$METRICS_SCRIPT"
            --sparse "$sparse_dir"
            --dense "$dense_dir"
            --name "$name"
            --outdir "$RESULT_FOLDER"
            --photometric-kind "$PHOTOMETRIC_KIND"
            --relative-depth-clip "$RELATIVE_DEPTH_CLIP"
            --coverage-plane "$COVERAGE_PLANE"
        )
        if [[ -f "$database_path" ]]; then
            metrics_args+=(--database "$database_path")
        fi
        "$PYTHON_BIN" "${metrics_args[@]}"
    done
done

echo "COLMAP metric calculation completed successfully."
