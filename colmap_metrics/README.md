# COLMAP-Based 3D Consistency Metrics

This folder is the general COLMAP evaluation pipeline for any model and any dataset, as long as each scene has a directory of rendered images. The intended flow is:

1. Install Docker/NVIDIA COLMAP support with `colmap_install.sh`.
2. Create `models.txt` and `scenes.txt`.
3. Place images under `<base>/<model>/<dataset>/colmap_metrics/<scene>/images/`.
4. Run `p1.sh <dataset_name>` to build COLMAP sparse and dense outputs.
5. Run `p2.sh <dataset_name>` to compute JSON metrics for every model/scene pair.

The batch scripts now read `models.txt` and `scenes.txt` directly, so you do not need to edit hard-coded model arrays in the scripts.

## Files

- `colmap_install.sh`: sets up Docker/NVIDIA COLMAP support and installs a convenience alias for interactive use.
- `batch_common.sh`: shared shell helpers used by the batch wrappers.
- `p1.sh`: batch COLMAP reconstruction across all models and scenes for one dataset.
- `p2.sh`: batch metric computation across all models and scenes for one dataset.
- `colmap_run.sh`: single-scene COLMAP runner.
- `metrics.py`: metric computation for one sparse/dense reconstruction.
- `dl3dv-mipnerf-eval/`: paper-specific DL3DV + MipNeRF360 wrapper.

## Install COLMAP

From this directory:

```bash
cd colmap_metrics
bash colmap_install.sh
source ~/.zshrc
colmap help
```

Notes:

- The installer configures Docker for NVIDIA GPUs and pulls `colmap/colmap:latest`.
- The interactive alias is convenient for manual debugging, but the checked-in scripts no longer depend on shell alias expansion.
- On non-Arch systems, install Docker and `nvidia-container-toolkit` first, then rerun the installer.

## Required list files

Create these files in `colmap_metrics/`:

### `models.txt`

One model name per line. Blank lines and `#` comments are ignored.

```text
model_a
model_b
model_c
```

### `scenes.txt`

One scene name per line. Blank lines and `#` comments are ignored.

```text
scene_001
scene_002
scene_003
```

## Expected data layout

By default, `p1.sh` and `p2.sh` assume the directory containing `models.txt` also contains the model folders. If your model folders live elsewhere, set `BASE_DIR=/path/to/root`.

Expected layout for dataset `my_dataset`:

```text
<BASE_DIR>/
  model_a/
    my_dataset/
      colmap_metrics/
        scene_001/
          images/
            0001.png
            0002.png
        scene_002/
          images/
            ...
  model_b/
    my_dataset/
      colmap_metrics/
        scene_001/
          images/
            ...
```

After `p1.sh` runs, each scene directory contains:

```text
database.db
sparse/
dense/
```

## Batch usage

Minimal run:

```bash
cd colmap_metrics
bash p1.sh my_dataset
bash p2.sh my_dataset
```

Useful overrides:

```bash
BASE_DIR=/data/evals bash p1.sh my_dataset

BASE_DIR=/data/evals \
RESULT_FOLDER=/data/evals/results \
PHOTOMETRIC_KIND=confidence \
RELATIVE_DEPTH_CLIP=0.15 \
COVERAGE_PLANE=xz \
bash p2.sh my_dataset
```

## `p1.sh` and `p2.sh` wrapper defaults

These wrappers share the same scene-root convention:

```text
<BASE_DIR>/<model>/<dataset_name>/colmap_metrics/<scene>/
```

Shared wrapper variables:

| Variable | Default | Used by | Meaning |
| --- | --- | --- | --- |
| `MODELS_FILE` | `colmap_metrics/models.txt` | `p1.sh`, `p2.sh` | Model list file |
| `SCENES_FILE` | `colmap_metrics/scenes.txt` | `p1.sh`, `p2.sh` | Scene list file |
| `BASE_DIR` | `colmap_metrics/` | `p1.sh`, `p2.sh` | Root directory that contains the model folders |
| `IMAGES_DIR_NAME` | `images` | `p1.sh`, `p2.sh` | Name of the image folder inside each scene |
| `SPARSE_DIR_NAME` | `sparse` | `p1.sh`, `p2.sh` | Output sparse folder name |
| `DENSE_DIR_NAME` | `dense` | `p1.sh`, `p2.sh` | Output dense folder name |

`p2.sh` adds:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RESULT_FOLDER` | `colmap_metrics/results` | Directory for output JSON files |
| `SPARSE_MODEL_SUBDIR` | `0` | Preferred sparse model subdirectory; if missing, the script uses the first sparse subdirectory it finds |
| `PYTHON_BIN` | `python` | Python executable used to run `metrics.py` |
| `PHOTOMETRIC_KIND` | `auto` | Passed to `metrics.py --photometric-kind` |
| `RELATIVE_DEPTH_CLIP` | `0.2` | Passed to `metrics.py --relative-depth-clip` |
| `COVERAGE_PLANE` | `pca` | Passed to `metrics.py --coverage-plane` |

## `colmap_run.sh` usage and defaults

Single-scene usage:

```bash
bash colmap_run.sh /path/to/scene_root
```

Optional positional arguments:

```bash
bash colmap_run.sh <scene_root> [images_dir_name] [sparse_dir_name] [dense_dir_name]
```

The script runs:

1. `feature_extractor`
2. `exhaustive_matcher`
3. `mapper`
4. `image_undistorter`
5. `patch_match_stereo` without geometric consistency
6. `patch_match_stereo` with geometric consistency

Configurable defaults in `colmap_run.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `COLMAP_USE_DOCKER` | `1` | Use Dockerized COLMAP; set to `0` to use a native `colmap` binary from `PATH` |
| `COLMAP_DOCKER_GPUS` | `all` | Docker GPU selection passed to `docker run --gpus` |
| `COLMAP_IMAGE` | `colmap/colmap:latest` | Docker image |
| `COLMAP_CAMERA_MODEL` | `OPENCV` | `ImageReader.camera_model` |
| `COLMAP_SINGLE_CAMERA` | `0` | If `1`, adds `--ImageReader.single_camera 1` |
| `COLMAP_MAPPER_INIT_MIN_TRI_ANGLE` | `4` | `Mapper.init_min_tri_angle` |
| `COLMAP_MAPPER_FILTER_MIN_TRI_ANGLE` | `1.5` | `Mapper.filter_min_tri_angle` |
| `COLMAP_MAPPER_MIN_NUM_MATCHES` | `10` | `Mapper.min_num_matches` |
| `COLMAP_PATCH_MATCH_NO_GEOM_GPU` | `0` | GPU for non-geometric PatchMatch |
| `COLMAP_PATCH_MATCH_GEOM_GPU` | `1` | GPU for geometric-consistency PatchMatch |
| `COLMAP_PATCH_MATCH_CACHE_SIZE` | `50` | PatchMatch cache size |
| `COLMAP_PATCH_MATCH_PARALLEL` | `1` | If `1` and the two GPU IDs differ, PatchMatch runs in parallel; otherwise sequentially |
| `COLMAP_SPARSE_MODEL_SUBDIR` | `0` | Preferred sparse model subdirectory after mapping |

Common edits:

- Single GPU:

```bash
COLMAP_PATCH_MATCH_PARALLEL=0 \
COLMAP_PATCH_MATCH_NO_GEOM_GPU=0 \
COLMAP_PATCH_MATCH_GEOM_GPU=0 \
bash p1.sh my_dataset
```

- Shared intrinsics for all images:

```bash
COLMAP_SINGLE_CAMERA=1 bash p1.sh my_dataset
```

- Native COLMAP instead of Docker:

```bash
COLMAP_USE_DOCKER=0 bash colmap_run.sh /path/to/scene_root
```

## `metrics.py` defaults and editable arguments

Direct usage:

```bash
python metrics.py \
  --sparse /path/to/scene_root/sparse/0 \
  --dense /path/to/scene_root/dense \
  --name model_dataset_scene \
  --outdir results
```

Arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--dense` | `dense` | Dense COLMAP folder |
| `--sparse` | `sparse/0` | Sparse COLMAP model folder |
| `--photometric-kind` | `auto` | Interpret `*.photometric.bin` as `auto`, `error`, `confidence`, or `depth` |
| `--relative-depth-clip` | `0.2` | Relative depth mismatch threshold when photometric maps behave like depth |
| `--coverage-plane` | `pca` | Use `pca` or `xz` for angular coverage |
| `--name` | `results` | Output JSON stem |
| `--outdir` | `.` | Output directory |
| `--database` | unset | Explicit path to `database.db` for attempted-set metrics |

Important behavior:

- Output metrics are `gpc`, `density`, `consistency`, `weighted_gpc`, `icm`, `coverage`, `registration_rate`, `gpc_all`, and `icm_all`.
- `--photometric-kind auto` distinguishes depth-like vs non-depth photometric maps once, then uses that interpretation for the run.
- In `auto` mode, non-depth maps are treated as error maps. If your maps are confidence scores, set `--photometric-kind confidence`.
- If `--database` is omitted, the script first tries to locate `database.db` near the sparse/dense folders and then falls back to `./database.db`.
- The reader now fails fast on malformed COLMAP `.bin` files instead of hanging on a truncated header.

## Paper-specific two-dataset evaluation

For the DL3DV + MipNeRF360 paper setup, use:

- `dl3dv-mipnerf-eval/Readme.md`
