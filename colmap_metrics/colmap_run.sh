#!/usr/bin/env bash

set -euo pipefail

run_colmap() {
    if [[ "${COLMAP_USE_DOCKER:-1}" == "0" ]]; then
        if ! command -v colmap >/dev/null 2>&1; then
            echo "Error: native colmap binary not found on PATH." >&2
            exit 1
        fi
        command colmap "$@"
        return
    fi

    if ! command -v docker >/dev/null 2>&1; then
        echo "Error: docker not found. Install Docker or run with COLMAP_USE_DOCKER=0." >&2
        exit 1
    fi

    sudo docker run --rm --gpus "${COLMAP_DOCKER_GPUS:-all}" \
        -v "$(pwd)":"$(pwd)" \
        -w "$(pwd)" \
        "${COLMAP_IMAGE:-colmap/colmap:latest}" \
        colmap "$@"
}

detect_sparse_model_dir() {
    local sparse_root="$1"
    local preferred_subdir="${2:-0}"
    local candidate

    if [[ -d "$sparse_root/$preferred_subdir" ]] && [[ -n "$(find "$sparse_root/$preferred_subdir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
        printf '%s\n' "$sparse_root/$preferred_subdir"
        return 0
    fi

    while IFS= read -r candidate; do
        if [[ -n "$(find "$candidate" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$sparse_root" -mindepth 1 -maxdepth 1 -type d | sort)

    if [[ -d "$sparse_root/$preferred_subdir" ]]; then
        printf '%s\n' "$sparse_root/$preferred_subdir"
        return 0
    fi

    while IFS= read -r candidate; do
        printf '%s\n' "$candidate"
        return 0
    done < <(find "$sparse_root" -mindepth 1 -maxdepth 1 -type d | sort)

    return 1
}

run_patch_match() {
    local geom_consistency="$1"
    local gpu_index="$2"

    run_colmap patch_match_stereo \
        --workspace_path "$DENSE_DIR_NAME" \
        --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency "$geom_consistency" \
        --PatchMatchStereo.gpu_index "$gpu_index" \
        --PatchMatchStereo.cache_size "$COLMAP_PATCH_MATCH_CACHE_SIZE"
}

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_directory> [images_directory_name] [sparse_directory_name] [dense_directory_name]" >&2
    exit 1
fi

INPUT_DIR="$1"
IMAGES_DIR_NAME="${2:-images}"
SPARSE_DIR_NAME="${3:-sparse}"
DENSE_DIR_NAME="${4:-dense}"

COLMAP_CAMERA_MODEL="${COLMAP_CAMERA_MODEL:-OPENCV}"
COLMAP_SINGLE_CAMERA="${COLMAP_SINGLE_CAMERA:-0}"
COLMAP_MAPPER_INIT_MIN_TRI_ANGLE="${COLMAP_MAPPER_INIT_MIN_TRI_ANGLE:-4}"
COLMAP_MAPPER_FILTER_MIN_TRI_ANGLE="${COLMAP_MAPPER_FILTER_MIN_TRI_ANGLE:-1.5}"
COLMAP_MAPPER_MIN_NUM_MATCHES="${COLMAP_MAPPER_MIN_NUM_MATCHES:-10}"
COLMAP_PATCH_MATCH_NO_GEOM_GPU="${COLMAP_PATCH_MATCH_NO_GEOM_GPU:-0}"
COLMAP_PATCH_MATCH_GEOM_GPU="${COLMAP_PATCH_MATCH_GEOM_GPU:-1}"
COLMAP_PATCH_MATCH_CACHE_SIZE="${COLMAP_PATCH_MATCH_CACHE_SIZE:-50}"
COLMAP_PATCH_MATCH_PARALLEL="${COLMAP_PATCH_MATCH_PARALLEL:-1}"
COLMAP_SPARSE_MODEL_SUBDIR="${COLMAP_SPARSE_MODEL_SUBDIR:-0}"

if ! INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"; then
    echo "Error: failed to resolve input directory: $INPUT_DIR" >&2
    exit 1
fi

echo "Input directory: $INPUT_DIR"
echo "Images directory name: $IMAGES_DIR_NAME"
echo "Sparse directory name: $SPARSE_DIR_NAME"
echo "Dense directory name: $DENSE_DIR_NAME"

cd "$INPUT_DIR" || {
    echo "Error: failed to change directory to $INPUT_DIR" >&2
    exit 1
}

if [[ ! -d "$IMAGES_DIR_NAME" ]]; then
    echo "Error: images directory not found at $INPUT_DIR/$IMAGES_DIR_NAME" >&2
    exit 1
fi

echo "Running feature extraction..."
feature_args=(
    feature_extractor
    --database_path database.db
    --image_path "$IMAGES_DIR_NAME"
    --ImageReader.camera_model "$COLMAP_CAMERA_MODEL"
)
if [[ "$COLMAP_SINGLE_CAMERA" == "1" ]]; then
    feature_args+=(--ImageReader.single_camera 1)
fi
run_colmap "${feature_args[@]}"

echo "Running exhaustive matching..."
run_colmap exhaustive_matcher --database_path database.db

mkdir -p "$SPARSE_DIR_NAME"
echo "Running sparse reconstruction..."
run_colmap mapper \
    --database_path database.db \
    --image_path "$IMAGES_DIR_NAME" \
    --output_path "$SPARSE_DIR_NAME" \
    --Mapper.init_min_tri_angle "$COLMAP_MAPPER_INIT_MIN_TRI_ANGLE" \
    --Mapper.filter_min_tri_angle "$COLMAP_MAPPER_FILTER_MIN_TRI_ANGLE" \
    --Mapper.min_num_matches "$COLMAP_MAPPER_MIN_NUM_MATCHES"

SPARSE_MODEL_DIR="$(detect_sparse_model_dir "$SPARSE_DIR_NAME" "$COLMAP_SPARSE_MODEL_SUBDIR")" || {
    echo "Error: sparse reconstruction produced no model directories in $SPARSE_DIR_NAME" >&2
    exit 1
}
if [[ -z "$(find "$SPARSE_MODEL_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Error: sparse model directory is empty: $SPARSE_MODEL_DIR" >&2
    exit 1
fi
echo "Using sparse model directory: $SPARSE_MODEL_DIR"

echo "Running dense reconstruction..."
mkdir -p "$DENSE_DIR_NAME"
run_colmap image_undistorter \
    --image_path "$IMAGES_DIR_NAME" \
    --input_path "$SPARSE_MODEL_DIR" \
    --output_path "$DENSE_DIR_NAME" \
    --output_type COLMAP

if [[ "$COLMAP_PATCH_MATCH_PARALLEL" == "1" && "$COLMAP_PATCH_MATCH_NO_GEOM_GPU" != "$COLMAP_PATCH_MATCH_GEOM_GPU" ]]; then
    echo "Running Patch Match Stereo in parallel on GPU $COLMAP_PATCH_MATCH_NO_GEOM_GPU and GPU $COLMAP_PATCH_MATCH_GEOM_GPU"
    run_patch_match false "$COLMAP_PATCH_MATCH_NO_GEOM_GPU" &
    pid_no_geom=$!
    run_patch_match true "$COLMAP_PATCH_MATCH_GEOM_GPU" &
    pid_geom=$!

    status_no_geom=0
    status_geom=0
    wait "$pid_no_geom" || status_no_geom=$?
    wait "$pid_geom" || status_geom=$?
    if [[ $status_no_geom -ne 0 || $status_geom -ne 0 ]]; then
        echo "Error: dense reconstruction failed during Patch Match Stereo." >&2
        exit 1
    fi
else
    echo "Running Patch Match Stereo sequentially."
    run_patch_match false "$COLMAP_PATCH_MATCH_NO_GEOM_GPU"
    run_patch_match true "$COLMAP_PATCH_MATCH_GEOM_GPU"
fi

echo "COLMAP reconstruction completed successfully."
