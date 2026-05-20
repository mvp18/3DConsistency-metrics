#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/batch_common.sh"

DATASET_NAME="${1:-${DATASET_NAME:-}}"
MODELS_FILE="${MODELS_FILE:-$SCRIPT_DIR/models.txt}"
SCENES_FILE="${SCENES_FILE:-$SCRIPT_DIR/scenes.txt}"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
IMAGES_DIR_NAME="${IMAGES_DIR_NAME:-images}"
SPARSE_DIR_NAME="${SPARSE_DIR_NAME:-sparse}"
DENSE_DIR_NAME="${DENSE_DIR_NAME:-dense}"
SPARSE_MODEL_SUBDIR="${SPARSE_MODEL_SUBDIR:-0}"
RESULT_FOLDER="${RESULT_FOLDER:-$SCRIPT_DIR/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
METRICS_SCRIPT="${METRICS_SCRIPT:-$SCRIPT_DIR/metrics.py}"
PHOTOMETRIC_KIND="${PHOTOMETRIC_KIND:-auto}"
RELATIVE_DEPTH_CLIP="${RELATIVE_DEPTH_CLIP:-0.2}"
COVERAGE_PLANE="${COVERAGE_PLANE:-pca}"

if [[ -z "$DATASET_NAME" ]]; then
    print_error "Dataset name is required. Usage: bash p2.sh <dataset_name>"
    exit 1
fi

require_file "$METRICS_SCRIPT"
load_nonempty_lines "$MODELS_FILE" models
load_nonempty_lines "$SCENES_FILE" scenes

echo "Dataset: $DATASET_NAME"
echo "Models (${#models[@]}): ${models[*]}"
echo "Scenes (${#scenes[@]}): ${scenes[*]}"
echo "Results folder: $RESULT_FOLDER"

for m in "${models[@]}"; do
    echo "Computing metrics for model: $m"
    for scene in "${scenes[@]}"; do
        scene_root="$BASE_DIR/$m/$DATASET_NAME/colmap_metrics/$scene"
        sparse_root="$scene_root/$SPARSE_DIR_NAME"
        dense_dir="$scene_root/$DENSE_DIR_NAME"
        database_path="$scene_root/database.db"

        require_dir "$scene_root"
        require_dir "$dense_dir"
        require_dir "$sparse_root"
        sparse_dir="$(resolve_sparse_model_dir "$sparse_root" "$SPARSE_MODEL_SUBDIR")" || {
            print_error "Could not find a sparse model directory under $sparse_root"
            exit 1
        }

        name="${m}_${DATASET_NAME}_${scene}"
        echo "Processing scene: $scene"
        echo "Sparse: $sparse_dir"
        echo "Dense: $dense_dir"

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
