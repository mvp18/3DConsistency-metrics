# Can These Views Be One Scene? Evaluating Multiview 3D Consistency when 3D Foundation Models Hallucinate

[**Paper (arXiv)**](https://arxiv.org/abs/2605.18754) &nbsp;&middot;&nbsp;
[**Project page**](https://mvp18.github.io/3d-consistency-metrics) &nbsp;&middot;&nbsp;
[**SysCON3D dataset**](https://huggingface.co/datasets/syscon3d-neurips26/syscon3d) &nbsp;&middot;&nbsp;
[**Human evaluation site**](https://mvp18.github.io/nvs-human-eval)

**Authors:** [Soumava Paul](https://mvp18.github.io)\*, [Prakhar Kaushik](https://toshi2k2.github.io)\*, [Alan Yuille](https://www.cs.jhu.edu/~ayuille1/) &nbsp;&mdash;&nbsp; CCVL, Johns Hopkins University. \*Equal contribution.

---

## Layout

- `demo_gradio_compare.py`: interactive MASt3R, DUSt3R, Fast3R, VGGT, and RobustVGGT comparison on uploaded images or SysCON3D samples.
- `run_syscon3d_project_assets.py`: batch asset generation for SysCON3D samples.
- `eval_image_dir_metrics.py`: neural/backbone metrics on arbitrary image sets.
- `eval_generated_nvs_metrics.py`: neural/backbone metrics on generated NVS output layouts.
- `colmap_metrics/`: COLMAP reconstruction and metric wrappers for arbitrary image sets.

Model checkpoints are not included. The code downloads public Hugging Face
checkpoints at runtime:

- MASt3R: `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`
- DUSt3R: `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`
- Fast3R: `jedyang97/Fast3R_ViT_Large_512`
- VGGT: `facebook/VGGT-1B`
- FeatUp: `mhamilton723/FeatUp`

---

## Hallucinated geometry on Gaussian noise

VGGT, MASt3R, DUSt3R, and Fast3R all return dense 3D structure when fed pure Gaussian noise as multi-view input. Metrics built on top (such as MEt3R) inherit this failure and report high 3D consistency on what is, in fact, no scene at all.

![Hallucinated reconstructions from Gaussian noise across four backbones](assets/gaussian_noise_hallucinations.gif)

In the [interactive gallery on the project page](https://mvp18.github.io/3d-consistency-metrics#interactive-syscon3d-demo), you can orbit these reconstructions and switch between scene types (Gaussian noise, cross-scene mixtures, one-outlier, patched corruptions, and clean baselines).

## Abstract

Multiview 3D evaluation assumes that the images being scored are observations of one static 3D scene. This assumption can fail in NVS and sparse-view reconstruction: inputs or generated outputs may contain artifacts, outlier frames, repeated views, or noise, yet still receive high 3D consistency scores. Existing reference-based metrics require ground truth, while ground-truth-free metrics such as MEt3R depend on learned reconstruction backbones whose failure modes are poorly characterized. We study this reliability problem by comparing neural reconstruction priors with classical geometric verification. We introduce SysCON3D, a controlled robustness benchmark for multiview 3D consistency, and a parametric family that decomposes neural metrics into backbone, residual, and aggregation components. This family recovers MEt3R and yields variants up to 3&times; more robust. Our analysis shows that VGGT, MASt3R, DUSt3R, and Fast3R can hallucinate dense geometry and cross-view support for unrelated scenes, repeated images, and random noise. We introduce COLMAP-based metrics that use matches, registration, dense support, and reconstruction failure as failure-aware consistency signals. On real NVS outputs and a structured human study, these metrics achieve up to 4&times; higher correlation with human judgments than MEt3R.

## SysCON3D Data

The scripts assume the SysCON3D release is available at:

```text
tmp/syscon3d_release/
  README.md
  mipnerf360_calibration_splits.json
  mipnerf360_impossible_splits.json
  archives/
    syscon3d_mipnerf360_000.tar
```

Download the files from the
[SysCON3D Hugging Face dataset](https://huggingface.co/datasets/syscon3d-neurips26/syscon3d)
into `tmp/syscon3d_release/`, then extract the archive payload:

```bash
mkdir -p tmp/syscon3d_release
for shard in tmp/syscon3d_release/archives/*.tar; do
  tar -xf "$shard" -C tmp/syscon3d_release
done
```

After extraction, `tmp/syscon3d_release/mipnerf360/` should exist. The paths
listed in `mipnerf360_impossible_splits.json` are relative to that directory.

The archive is an uncompressed tar file, so individual files can be inspected
or extracted without unpacking the full payload:

```bash
# List files in the shard.
tar -tf tmp/syscon3d_release/archives/syscon3d_mipnerf360_000.tar | less

# Extract one image into tmp/syscon3d_release/.
tar -xf tmp/syscon3d_release/archives/syscon3d_mipnerf360_000.tar \
  -C tmp/syscon3d_release \
  mipnerf360/syscon3d_scene_types/noise_gaussian/k09/noise_gaussian_k09_000/images_4/view_000.png
```

## Neural Metrics Environment

Use Python 3.10 or 3.11. Install PyTorch first with the CUDA build that matches
your machine, then install the neural metric requirements.

```bash
conda create -n robust3d python=3.11 -y
conda activate robust3d

# Choose the PyTorch command appropriate for your CUDA driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements-robust3d.txt
pip install -e met3r
pip install -e fast3r
pip install -e vggt
```

For installing pytorch3d, run ``pip install "git+https://github.com/facebookresearch/pytorch3d.git"`` once ``CUDA_HOME`` variable is set.

## Gradio Backbone Demo

Run on a GPU machine:

```bash
conda activate robust3d
python demo_gradio_compare.py \
  --server-name 127.0.0.1 \
  --server-port 7860
```

Then open `http://127.0.0.1:7860`. On a remote machine, forward the port:

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<host>
```

The demo has two input modes:

- `SysCON3D benchmark sample`: reads the default manifests from `tmp/syscon3d_release/`.
- `Upload images`: runs the selected backbones on any multi-view image set.

Generated point clouds and stats are written under `output/gradio_compare/`.

## SysCON3D Asset Generation

Generate GLB assets for a small SysCON3D subset:

```bash
conda activate robust3d
python run_syscon3d_project_assets.py \
  --backbones mast3r dust3r fast3r vggt \
  --subset-sizes 3 \
  --task-limit 8 \
  --gpus 0 \
  --launch \
  --wait
```

Useful options:

- `--backbones mast3r dust3r fast3r vggt robust_vggt`
- `--subset-sizes 3 6 9 12`
- `--write-videos` to also write MP4 turntables
- `--keep-ply` to retain intermediate PLY files

Outputs go to `output/syscon3d_project_assets/` by default.

## Neural Metrics on Arbitrary Images

Run metrics on one folder of images:

```bash
conda activate robust3d
python eval_image_dir_metrics.py \
  --image-dir /path/to/images \
  --metrics met3r mast3r-imq met3r_dust3r fast3r_pc vggt-robust \
  --out-csv output/image_metrics.csv
```

Run metrics on input views plus generated NVS frames by passing both folders:

```bash
python eval_image_dir_metrics.py \
  --image-dir /path/to/input_views /path/to/generated_views \
  --metrics met3r met3r_dust3r fast3r_pc vggt_pc \
  --out-csv output/nvs_image_set_metrics.csv
```

`--metrics all` runs the available MASt3R, DUSt3R, Fast3R, and VGGT metric
variants implemented in that script. Convenience aliases include
`mast3r-imq`, `fast3r-pc`, and `vggt-robust`.

## Neural Metrics on Paper NVS Layouts

`eval_generated_nvs_metrics.py` supports the generated-NVS directory conventions
used by the paper. Override the roots for your local outputs:

```bash
python eval_generated_nvs_metrics.py \
  --dataset mipnerf360 \
  --method mvsplat360 \
  --scene bicycle \
  --subset-size 9 \
  --stable-vc-mipnerf-root /path/to/stable_virtual_camera/mipnerf360 \
  --mvsplat360-root /path/to/mvsplat360_outputs \
  --metrics met3r mast3r-imq met3r_dust3r fast3r_pc vggt-robust \
  --out-csv output/generated_nvs/mvsplat360_bicycle_k9.csv
```

Supported `--method` values are `depthsplat`, `long-lrm`, `mvsplat360`,
`stable-virtual-camera`, `viewcrafter`, `difix3d`, `nvs_solver`, and
`mvgenmaster`.

## COLMAP Metrics

The COLMAP path can run in a smaller environment. Docker with NVIDIA runtime is
recommended for reconstruction; Python is only needed for metric computation.

```bash
conda create -n colmap-metrics python=3.11 -y
conda activate colmap-metrics
pip install -r requirements-colmap.txt
```

Install or configure COLMAP:

```bash
cd colmap_metrics
bash colmap_install.sh
```

For arbitrary image sets, create `models.txt` and `scenes.txt` in
`colmap_metrics/`, then arrange images as:

```text
<BASE_DIR>/<model>/<dataset_name>/colmap_metrics/<scene>/images/*.png
```

Run reconstruction and metrics:

```bash
cd colmap_metrics
BASE_DIR=/path/to/eval_root bash p1.sh my_dataset
BASE_DIR=/path/to/eval_root RESULT_FOLDER=/path/to/results bash p2.sh my_dataset
```

See `colmap_metrics/README.md` for all COLMAP wrapper options.

## Citation

```bibtex
@article{paul2026can,
  title={Can These Views Be One Scene? Evaluating Multiview 3D Consistency when 3D Foundation Models Hallucinate},
  author={Paul, Soumava and Kaushik, Prakhar and Yuille, Alan},
  journal={arXiv preprint arXiv:2605.18754},
  year={2026}
}
```
