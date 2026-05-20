#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../batch_common.sh"

BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
MODELS_FILE="${MODELS_FILE:-$SCRIPT_DIR/models.txt}"
DL3DV_SCENES_FILE="${DL3DV_SCENES_FILE:-$SCRIPT_DIR/dl3dv_folders.txt}"
MIP_SCENES_FILE="${MIP_SCENES_FILE:-$SCRIPT_DIR/mipnerf360_scenes.txt}"
IMAGES_DIR_NAME="${IMAGES_DIR_NAME:-images}"
SPARSE_DIR_NAME="${SPARSE_DIR_NAME:-sparse}"
DENSE_DIR_NAME="${DENSE_DIR_NAME:-dense}"
COLMAP_RUN_SCRIPT="${COLMAP_RUN_SCRIPT:-$SCRIPT_DIR/colmap_run.sh}"

require_file "$COLMAP_RUN_SCRIPT"
load_nonempty_lines "$MODELS_FILE" models
load_nonempty_lines "$DL3DV_SCENES_FILE" dl3dv_scenes
load_csv_entries "$MIP_SCENES_FILE" mip_scenes

echo "Models (${#models[@]}): ${models[*]}"
echo "DL3DV scenes (${#dl3dv_scenes[@]}): ${dl3dv_scenes[*]}"
echo "MipNeRF360 scenes (${#mip_scenes[@]}): ${mip_scenes[*]}"

for m in "${models[@]}"; do
    echo "Running COLMAP for model: $m"
    for scene in "${dl3dv_scenes[@]}"; do
        scene_root="$BASE_DIR/$m/dl3dv/colmap_metrics/$scene"
        require_dir "$scene_root"
        require_dir "$scene_root/$IMAGES_DIR_NAME"

        echo "Processing DL3DV scene: $scene"
        bash "$COLMAP_RUN_SCRIPT" "$scene_root" "$IMAGES_DIR_NAME" "$SPARSE_DIR_NAME" "$DENSE_DIR_NAME"
    done

    for scene in "${mip_scenes[@]}"; do
        scene_root="$BASE_DIR/$m/mip/colmap_metrics/$scene"
        require_dir "$scene_root"
        require_dir "$scene_root/$IMAGES_DIR_NAME"

        echo "Processing MipNeRF360 scene: $scene"
        bash "$COLMAP_RUN_SCRIPT" "$scene_root" "$IMAGES_DIR_NAME" "$SPARSE_DIR_NAME" "$DENSE_DIR_NAME"
    done
done

echo "COLMAP reconstruction completed successfully."
