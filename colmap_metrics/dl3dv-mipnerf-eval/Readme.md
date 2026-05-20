# DL3DV + MipNeRF360 Paper Evaluation

This folder is the paper-specific wrapper for the two dataset experiments:

- DL3DV
- MipNeRF360

It uses its own copies of:

- `z_metric_p1.sh`
- `z_metric_p2.sh`
- `colmap_run.sh`
- `metrics.py`

The paper wrappers now read `models.txt`, `dl3dv_folders.txt`, and `mipnerf360_scenes.txt` directly. You do not need to edit hard-coded model arrays.

## Setup

From the parent folder:

```bash
cd colmap_metrics
bash colmap_install.sh
source ~/.zshrc
cd dl3dv-mipnerf-eval
```

The interactive alias is optional for the batch scripts, but the installer is still the recommended way to configure Docker/NVIDIA and pull the COLMAP image.

## Required files

- `models.txt`: one model name per line
- `dl3dv_folders.txt`: one DL3DV scene ID per line
- `mipnerf360_scenes.txt`: comma-separated MipNeRF360 scene names

Blank lines and `#` comments are ignored.

## Expected layout

By default, the script directory is also the base directory that contains the model folders. If your model folders live elsewhere, set `BASE_DIR=/path/to/root`.

Expected structure:

```text
<BASE_DIR>/
  <model_name>/
    dl3dv/
      colmap_metrics/
        <dl3dv_scene_id>/
          images/
            0001.png
            0002.png
            ...
    mip/
      colmap_metrics/
        <mip_scene_name>/
          images/
            0001.png
            0002.png
            ...
```

The dataset folder names are fixed by the scripts:

- `dl3dv`
- `mip`

After reconstruction, each scene directory contains:

```text
database.db
sparse/
dense/
```

## Run

From `colmap_metrics/dl3dv-mipnerf-eval/`:

```bash
bash z_metric_p1.sh
bash z_metric_p2.sh
```

Outputs from `z_metric_p2.sh` are written to:

```text
results_all/
```

## Paper wrapper defaults

`z_metric_p1.sh` and `z_metric_p2.sh` use these defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODELS_FILE` | `models.txt` | Model list |
| `DL3DV_SCENES_FILE` | `dl3dv_folders.txt` | DL3DV scene list |
| `MIP_SCENES_FILE` | `mipnerf360_scenes.txt` | MipNeRF360 scene list |
| `BASE_DIR` | current folder | Root containing model folders |
| `IMAGES_DIR_NAME` | `images` | Image folder name |
| `SPARSE_DIR_NAME` | `sparse` | Sparse output folder |
| `DENSE_DIR_NAME` | `dense` | Dense output folder |
| `RESULT_FOLDER` | `results_all` | Metric JSON output directory |
| `SPARSE_MODEL_SUBDIR` | `0` | Preferred sparse model subdirectory |
| `PYTHON_BIN` | `python` | Python executable |
| `PHOTOMETRIC_KIND` | `auto` | Passed to `metrics.py` |
| `RELATIVE_DEPTH_CLIP` | `0.2` | Passed to `metrics.py` |
| `COVERAGE_PLANE` | `pca` | Passed to `metrics.py` |

Examples:

```bash
BASE_DIR=/data/paper_eval bash z_metric_p1.sh

BASE_DIR=/data/paper_eval \
RESULT_FOLDER=/data/paper_eval/results_all \
PHOTOMETRIC_KIND=confidence \
RELATIVE_DEPTH_CLIP=0.15 \
COVERAGE_PLANE=xz \
bash z_metric_p2.sh
```

## `colmap_run.sh` and `metrics.py`

The local `colmap_run.sh` and `metrics.py` expose the same defaults as the general wrapper in `../README.md`, including:

- Docker-vs-native COLMAP selection through `COLMAP_USE_DOCKER`
- camera model, single-camera mode, mapper thresholds, and PatchMatch GPU IDs
- sparse-model auto-detection with preference for subdirectory `0`
- metric options for photometric interpretation, relative depth clipping, and coverage plane

See `../README.md` for the full list of COLMAP and metric parameters.
