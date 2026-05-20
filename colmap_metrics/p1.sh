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
COLMAP_RUN_SCRIPT="${COLMAP_RUN_SCRIPT:-$SCRIPT_DIR/colmap_run.sh}"

if [[ -z "$DATASET_NAME" ]]; then
    print_error "Dataset name is required. Usage: bash p1.sh <dataset_name>"
    exit 1
fi

require_file "$COLMAP_RUN_SCRIPT"
load_nonempty_lines "$MODELS_FILE" models
load_nonempty_lines "$SCENES_FILE" scenes

echo "Dataset: $DATASET_NAME"
echo "Models (${#models[@]}): ${models[*]}"
echo "Scenes (${#scenes[@]}): ${scenes[*]}"
echo "Image folder name: $IMAGES_DIR_NAME"

for m in "${models[@]}"; do
    echo "Running COLMAP for model: $m"
    for scene in "${scenes[@]}"; do
        scene_root="$BASE_DIR/$m/$DATASET_NAME/colmap_metrics/$scene"
        require_dir "$scene_root"
        require_dir "$scene_root/$IMAGES_DIR_NAME"

        echo "Processing scene: $scene"
        bash "$COLMAP_RUN_SCRIPT" "$scene_root" "$IMAGES_DIR_NAME" "$SPARSE_DIR_NAME" "$DENSE_DIR_NAME"
    done
done

echo "COLMAP reconstruction completed successfully."
