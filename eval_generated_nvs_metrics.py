#!/usr/bin/env python3
"""
Evaluate multiview consistency metrics on generated novel views (NVS).

For a single (dataset, scene, K, method) combination this script:
  1) loads the K training/input views from stable-virtual-camera outputs;
  2) loads up to M novel frames from the requested method output directory;
  3) evaluates the selected metrics (subset of calibrate_recon_metrics.py) on the combined set.

Notes
-----
- NVS_Solver typically outputs 25 frames; other methods output 50.
- depthsplat can output >50 frames (e.g. 110); we randomly subsample M frames for it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Set, Tuple


DatasetName = Literal["dl3dv_benchmark", "mipnerf360"]
MethodName = Literal[
    "depthsplat",
    "long-lrm",
    "mvsplat360",
    "stable-virtual-camera",
    "viewcrafter",
    "difix3d",
    "nvs_solver",
    "mvgenmaster",
]


def _list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def _stable_vc_tag(dataset: DatasetName) -> str:
    if dataset == "mipnerf360":
        return "mipnerf360"
    return "dl3dv"


def _depthsplat_tag(dataset: DatasetName) -> str:
    if dataset == "mipnerf360":
        return "mipnerf360"
    return "dl3dv"


def _longlrm_tag(dataset: DatasetName) -> str:
    if dataset == "mipnerf360":
        return "mipnerf360"
    return "dl3dv-benchmark"


def _select_frame_subdir(frames_root: Path, subset_size: int) -> Optional[Path]:
    """
    Pick a `frame_*` subdirectory that matches `subset_size` indices if possible.
    """
    if not frames_root.is_dir():
        return None

    candidates: List[Path] = []
    candidates_matching_k: List[Path] = []
    for p in sorted(frames_root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("frame_"):
            continue
        candidates.append(p)
        parts = p.name.split("_")[1:]  # skip "frame"
        if len(parts) == int(subset_size):
            candidates_matching_k.append(p)

    if candidates_matching_k:
        return candidates_matching_k[0]
    if candidates:
        return candidates[0]
    return None


def resolve_stable_vc_input_dir(
    *,
    dataset: DatasetName,
    scene: str,
    subset_size: int,
    stable_vc_dl3dv_root: Path,
    stable_vc_mipnerf_root: Path,
) -> Path:
    base = stable_vc_mipnerf_root if dataset == "mipnerf360" else stable_vc_dl3dv_root
    tag = _stable_vc_tag(dataset)
    return base / f"{tag}-orbit-50out-{int(subset_size)}in" / scene / "input"


def resolve_method_novel_dir(
    *,
    method: MethodName,
    dataset: DatasetName,
    scene: str,
    subset_size: int,
    depthsplat_root: Path,
    longlrm_root: Path,
    mvsplat360_root: Path,
    stable_vc_dl3dv_root: Path,
    stable_vc_mipnerf_root: Path,
    viewcrafter_root: Path,
    mvgenmaster_root: Path,
    difix3d_root: Path,
    nvs_solver_root: Path,
) -> Optional[Path]:
    K = int(subset_size)

    if method == "stable-virtual-camera":
        base = stable_vc_mipnerf_root if dataset == "mipnerf360" else stable_vc_dl3dv_root
        tag = _stable_vc_tag(dataset)
        return base / f"{tag}-orbit-50out-{K}in" / scene / "samples-rgb"

    if method == "viewcrafter":
        return viewcrafter_root / dataset / scene / f"{K}view" / "diffusion_orbit_frames"

    if method == "mvgenmaster":
        return mvgenmaster_root / dataset / scene / f"{K}view" / "orbit50" / "frames"

    if method == "difix3d":
        return difix3d_root / dataset / scene / f"{K}view" / "orbit50" / "renders" / "novel" / "19999" / "Fixed"

    if method == "nvs_solver":
        return nvs_solver_root / dataset / scene / f"{K}view" / "renders"

    if method == "long-lrm":
        tag = _longlrm_tag(dataset)
        return longlrm_root / f"llrm-{tag}-orbit" / scene / f"{K}v" / f"in{K}_fixar_orbit50f" / "frames"

    if method == "depthsplat":
        tag = _depthsplat_tag(dataset)
        scene_root = depthsplat_root / f"depthsplat-{tag}-orbit-{K}v" / "frames" / scene
        return _select_frame_subdir(scene_root, subset_size=K)

    if method == "mvsplat360":
        base = mvsplat360_root / f"{dataset}_mvsplat360_orbit" / f"{K}v"
        scene_root = base / scene / "FramesRefined" / scene
        if scene_root.is_dir():
            return _select_frame_subdir(scene_root, subset_size=K)

        legacy_scene_root = base / "mipnerf360_mvsplat360_orbit" / "FramesRefined" / scene
        return _select_frame_subdir(legacy_scene_root, subset_size=K)

    raise ValueError(f"Unknown method: {method}")


def _select_novel_images(
    *,
    method: MethodName,
    images: Sequence[Path],
    num_novel_views: int,
    seed: int,
) -> List[Path]:
    if num_novel_views <= 0:
        return list(images)

    if len(images) <= num_novel_views:
        return list(images)

    if method == "depthsplat":
        rng = random.Random(int(seed))
        chosen = rng.sample(list(images), k=int(num_novel_views))
        return sorted(chosen)

    return list(images[: int(num_novel_views)])


def _select_metrics(raw: Sequence[str]) -> List[str]:
    aliases = {
        "mast3r-base": "met3r",
        "mast3r-energy": "met3r_energy",
        "mast3r-imq": "met3r_imq",
        "dust3r-base": "met3r_dust3r",
        "dust3r-energy": "met3r_dust3r_energy",
        "fast3r-pc": "fast3r_pc",
        "fast3r-pc-energy": "fast3r_pc_energy",
        "vggt-robust": "vggt_robust",
    }
    requested = [aliases.get(m.lower(), m.lower()) for m in raw]
    all_metrics = [
        "met3r",
        "met3r_mmd",
        "met3r_energy",
        "met3r_imq",
        "met3r_dust3r",
        "met3r_dust3r_energy",
        "fast3r",
        "fast3r_mmd",
        "fast3r_pc",
        "fast3r_pc_mmd",
        "fast3r_pc_energy",
        "vggt",
        "vggt_robust",
        "vggt_mmd",
        "vggt_pc",
        "vggt_pc_mmd",
    ]
    if "all" in requested:
        return all_metrics
    out: List[str] = []
    for m in requested:
        if m in all_metrics and m not in out:
            out.append(m)
    if not out:
        raise ValueError(f"No valid metrics selected from {raw}")
    return out


def _iterable_str(values: Iterable[str]) -> str:
    return ",".join(str(v) for v in values)


def _read_existing_metric_rows(out_csv: Path, fieldnames: Sequence[str]) -> Tuple[List[dict[str, str]], Set[str]]:
    if not out_csv.exists():
        return [], set()

    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = set(reader.fieldnames or [])
        expected = set(fieldnames)
        if not existing_fieldnames.issubset(expected) and existing_fieldnames != expected:
            raise ValueError(
                f"CSV fieldnames mismatch for append: {out_csv} has {sorted(existing_fieldnames)}, "
                f"expected subset of {sorted(expected)}"
            )

        rows_by_metric: dict[str, dict[str, str]] = {}
        for row in reader:
            metric = str(row.get("metric", "")).strip()
            if not metric:
                continue
            if metric in rows_by_metric:
                continue
            rows_by_metric[metric] = row
    return list(rows_by_metric.values()), set(rows_by_metric.keys())


def _write_metric_rows_atomic(out_csv: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=str(out_csv.parent),
        prefix=f"{out_csv.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        tmp_path = Path(f.name)
    tmp_path.replace(out_csv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Evaluate multiview metrics on generated NVS frames.")
    parser.add_argument("--dataset", type=str, required=True, choices=["dl3dv_benchmark", "mipnerf360"])
    parser.add_argument("--method", type=str, required=True, choices=list(MethodName.__args__))  
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--subset-size", type=int, required=True, choices=[3, 6, 9])
    parser.add_argument(
        "--num-novel-views",
        type=int,
        default=25,
        help="Maximum number of novel frames M to include (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for depthsplat subsampling (and any future stochastic selection).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["all"],
        help="Which metrics to run (default: all).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="CSV path to write (one row per metric).",
    )

    parser.add_argument(
        "--depthsplat-root",
        type=str,
        default="data/nvs_outputs/depthsplat",
    )
    parser.add_argument(
        "--longlrm-root",
        type=str,
        default="data/nvs_outputs/long-lrm",
    )
    parser.add_argument(
        "--mvsplat360-root",
        type=str,
        default="data/nvs_outputs/mvsplat360",
    )
    parser.add_argument(
        "--stable-vc-dl3dv-root",
        type=str,
        default="data/nvs_outputs/stable-virtual-camera/dl3dv",
    )
    parser.add_argument(
        "--stable-vc-mipnerf-root",
        type=str,
        default="data/nvs_outputs/stable-virtual-camera/mipnerf360",
    )
    parser.add_argument(
        "--viewcrafter-root",
        type=str,
        default="data/nvs_outputs/viewcrafter",
    )
    parser.add_argument(
        "--mvgenmaster-root",
        type=str,
        default="data/nvs_outputs/mvgenmaster",
    )
    parser.add_argument(
        "--difix3d-root",
        type=str,
        default="data/nvs_outputs/difix3d",
    )
    parser.add_argument(
        "--nvs-solver-root",
        type=str,
        default="data/nvs_outputs/nvs_solver",
    )

    parser.add_argument("--met3r-img-size", type=int, default=224)
    parser.add_argument("--met3r-batch-pairs", type=int, default=4)
    parser.add_argument("--met3r-per-pair-max-samples", type=int, default=128)
    parser.add_argument("--met3r-max-total-samples", type=int, default=4096)
    parser.add_argument("--fast3r-resize", type=int, default=224)
    parser.add_argument("--vggt-img-size", type=int, default=224)
    args = parser.parse_args()

    dataset: DatasetName = args.dataset  
    method: MethodName = args.method  
    scene = str(args.scene)
    K = int(args.subset_size)

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "method",
        "scene",
        "subset_size",
        "num_input",
        "num_novel",
        "num_total",
        "metric",
        "score",
        "n_pairs",
        "n_zero_overlap_pairs",
        "n_points_sampled",
        "n_points_multi_view",
        "n_retained_views",
        "n_rejected_views",
        "rejected_view_fraction",
    ]

    metric_names = _select_metrics(args.metrics)
    existing_rows, existing_metrics = _read_existing_metric_rows(out_path, fieldnames=fieldnames)
    metrics_to_run = [m for m in metric_names if m not in existing_metrics]
    if not metrics_to_run:
        logger.info("All requested metrics already present in %s; nothing to do.", out_path)
        return

    roots = {
        "depthsplat_root": Path(args.depthsplat_root).expanduser().resolve(),
        "longlrm_root": Path(args.longlrm_root).expanduser().resolve(),
        "mvsplat360_root": Path(args.mvsplat360_root).expanduser().resolve(),
        "stable_vc_dl3dv_root": Path(args.stable_vc_dl3dv_root).expanduser().resolve(),
        "stable_vc_mipnerf_root": Path(args.stable_vc_mipnerf_root).expanduser().resolve(),
        "viewcrafter_root": Path(args.viewcrafter_root).expanduser().resolve(),
        "mvgenmaster_root": Path(args.mvgenmaster_root).expanduser().resolve(),
        "difix3d_root": Path(args.difix3d_root).expanduser().resolve(),
        "nvs_solver_root": Path(args.nvs_solver_root).expanduser().resolve(),
    }

    input_dir = resolve_stable_vc_input_dir(
        dataset=dataset,
        scene=scene,
        subset_size=K,
        stable_vc_dl3dv_root=roots["stable_vc_dl3dv_root"],
        stable_vc_mipnerf_root=roots["stable_vc_mipnerf_root"],
    )
    input_images = _list_images(input_dir)
    if len(input_images) < K:
        logger.warning(
            "Skipping: missing input views (dataset=%s scene=%s K=%d) at %s (found %d images)",
            dataset,
            scene,
            K,
            input_dir,
            len(input_images),
        )
        return
    input_images = input_images[:K]

    novel_dir = resolve_method_novel_dir(
        method=method,
        dataset=dataset,
        scene=scene,
        subset_size=K,
        depthsplat_root=roots["depthsplat_root"],
        longlrm_root=roots["longlrm_root"],
        mvsplat360_root=roots["mvsplat360_root"],
        stable_vc_dl3dv_root=roots["stable_vc_dl3dv_root"],
        stable_vc_mipnerf_root=roots["stable_vc_mipnerf_root"],
        viewcrafter_root=roots["viewcrafter_root"],
        mvgenmaster_root=roots["mvgenmaster_root"],
        difix3d_root=roots["difix3d_root"],
        nvs_solver_root=roots["nvs_solver_root"],
    )
    if novel_dir is None or not novel_dir.is_dir():
        logger.warning(
            "Skipping: missing novel frames dir (dataset=%s method=%s scene=%s K=%d) at %s",
            dataset,
            method,
            scene,
            K,
            str(novel_dir) if novel_dir is not None else "<none>",
        )
        return
    all_novel = _list_images(novel_dir)
    if not all_novel:
        logger.warning(
            "Skipping: no novel frames found (dataset=%s method=%s scene=%s K=%d) at %s",
            dataset,
            method,
            scene,
            K,
            novel_dir,
        )
        return

    novel_images = _select_novel_images(
        method=method,
        images=all_novel,
        num_novel_views=int(args.num_novel_views),
        seed=int(args.seed),
    )
    combined = [*input_images, *novel_images]
    image_paths = [str(p) for p in combined]

    logger.info(
        "Prepared views: dataset=%s method=%s scene=%s K=%d + novel=%d (total=%d), metrics=%s",
        dataset,
        method,
        scene,
        K,
        len(novel_images),
        len(image_paths),
        _iterable_str(metrics_to_run),
    )

    import torch

    import calibrate_recon_metrics as crm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast3r_backbone = None

    sample_id = f"{dataset}_{scene}_{method}_K{K}_N{len(novel_images)}"
    sample = crm.MultiViewSample(
        sample_id=sample_id,
        kind="consistent",
        subset_size=len(image_paths),
        scene=scene,
        image_paths=image_paths,
    )

    new_rows: List[dict[str, object]] = []
    for metric_name in metrics_to_run:
        start = time.perf_counter()

        if metric_name == "met3r":
            metric = crm.MEt3R(
                img_size=args.met3r_img_size,
                use_norm=True,
                backbone="mast3r",
                feature_backbone="dinov2",
                feature_backbone_weights="mhamilton723/FeatUp",
                upsampler="featup",
                distance="cosine",
                freeze=True,
                rasterizer_kwargs={},
            )
            score = crm._evaluate_met3r(  
                metric,
                sample,
                img_size=args.met3r_img_size,
                batch_pairs=args.met3r_batch_pairs,
                device=device,
            )
        elif metric_name == "met3r_energy":
            per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
            max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
            metric = crm.MEt3R_Energy(
                img_size=args.met3r_img_size,
                use_norm=True,
                backbone="mast3r",
                feature_backbone="dinov2",
                feature_backbone_weights="mhamilton723/FeatUp",
                upsampler="featup",
                distance="cosine",
                freeze=True,
                per_pair_max_samples=per_pair_max,
                max_total_samples=max_total,
            )
            score = crm._evaluate_met3r_energy(  
                metric,
                sample,
                img_size=args.met3r_img_size,
                per_pair_max_samples=args.met3r_per_pair_max_samples,
                max_total_samples=args.met3r_max_total_samples,
                device=device,
            )
        elif metric_name == "met3r_dust3r":
            metric = crm.MEt3R(
                img_size=args.met3r_img_size,
                use_norm=True,
                backbone="dust3r",
                feature_backbone="dinov2",
                feature_backbone_weights="mhamilton723/FeatUp",
                upsampler="featup",
                distance="cosine",
                freeze=True,
                rasterizer_kwargs={},
            )
            score = crm._evaluate_met3r(  
                metric,
                sample,
                img_size=args.met3r_img_size,
                batch_pairs=args.met3r_batch_pairs,
                device=device,
            )
        elif metric_name == "met3r_dust3r_energy":
            per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
            max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
            metric = crm.MEt3R_Energy(
                img_size=args.met3r_img_size,
                use_norm=True,
                backbone="dust3r",
                feature_backbone="dinov2",
                feature_backbone_weights="mhamilton723/FeatUp",
                upsampler="featup",
                distance="cosine",
                freeze=True,
                per_pair_max_samples=per_pair_max,
                max_total_samples=max_total,
            )
            score = crm._evaluate_met3r_energy(  
                metric,
                sample,
                img_size=args.met3r_img_size,
                per_pair_max_samples=args.met3r_per_pair_max_samples,
                max_total_samples=args.met3r_max_total_samples,
                device=device,
            )
        elif metric_name == "met3r_mmd":
            per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
            max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
            metric = crm.MEt3R_MMD(
                img_size=args.met3r_img_size,
                use_norm=True,
                backbone="mast3r",
                feature_backbone="dinov2",
                feature_backbone_weights="mhamilton723/FeatUp",
                upsampler="featup",
                distance="cosine",
                freeze=True,
                per_pair_max_samples=per_pair_max,
                max_total_samples=max_total,
            )
            score = crm._evaluate_met3r_mmd(  
                metric,
                sample,
                img_size=args.met3r_img_size,
                per_pair_max_samples=args.met3r_per_pair_max_samples,
                max_total_samples=args.met3r_max_total_samples,
                device=device,
            )
        elif metric_name == "met3r_imq":
            per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
            max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
            metric = crm.MEt3R_IMQ(
                img_size=args.met3r_img_size,
                use_norm=True,
                backbone="mast3r",
                feature_backbone="dinov2",
                feature_backbone_weights="mhamilton723/FeatUp",
                upsampler="featup",
                distance="cosine",
                freeze=True,
                per_pair_max_samples=per_pair_max,
                max_total_samples=max_total,
            )
            score = crm._evaluate_met3r_imq(  
                metric,
                sample,
                img_size=args.met3r_img_size,
                per_pair_max_samples=args.met3r_per_pair_max_samples,
                max_total_samples=args.met3r_max_total_samples,
                device=device,
            )
        elif metric_name.startswith("fast3r"):
            from fast3r.models.fast3r import Fast3R

            if fast3r_backbone is None:
                fast3r_backbone = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512").to(str(device))

            if metric_name == "fast3r":
                metric = crm.MEt3R_Fast3R(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.0,
                    freeze=True,
                    rasterizer_kwargs={},
                    fast3r_weights=None,
                    fast3r_model=fast3r_backbone,
                    focal_length_estimation_method="first_view_from_global_head",
                    pnp_iterations=100,
                    default_focal_px=1.6,
                    min_points_per_view=50,
                    device=str(device),
                )
                score = crm._evaluate_fast3r(  
                    metric,
                    sample,
                    resize=args.fast3r_resize,
                    device=device,
                )
            elif metric_name == "fast3r_mmd":
                per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                metric = crm.MEt3R_Fast3R_MMD(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.0,
                    freeze=True,
                    rasterizer_kwargs={},
                    fast3r_weights=None,
                    fast3r_model=fast3r_backbone,
                    focal_length_estimation_method="first_view_from_global_head",
                    pnp_iterations=100,
                    default_focal_px=1.6,
                    min_points_per_view=50,
                    device=str(device),
                    per_pair_max_samples=per_pair_max,
                    max_total_samples=max_total,
                )
                score = crm._evaluate_fast3r_mmd(  
                    metric,
                    sample,
                    resize=args.fast3r_resize,
                    per_pair_max_samples=args.met3r_per_pair_max_samples,
                    max_total_samples=args.met3r_max_total_samples,
                    device=device,
                )
            elif metric_name == "fast3r_pc":
                metric = crm.MEt3R_Fast3R_PointConsistency(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.0,
                    freeze=True,
                    rasterizer_kwargs={},
                    fast3r_weights=None,
                    fast3r_model=fast3r_backbone,
                    focal_length_estimation_method="first_view_from_global_head",
                    pnp_iterations=100,
                    default_focal_px=1.6,
                    min_points_per_view=50,
                    device=str(device),
                )
                score = crm._evaluate_fast3r_pc(  
                    metric,
                    sample,
                    resize=args.fast3r_resize,
                    device=device,
                )
            elif metric_name == "fast3r_pc_mmd":
                max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                metric = crm.MEt3R_Fast3R_PointConsistency_MMD(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.0,
                    freeze=True,
                    rasterizer_kwargs={},
                    fast3r_weights=None,
                    fast3r_model=fast3r_backbone,
                    focal_length_estimation_method="first_view_from_global_head",
                    pnp_iterations=100,
                    default_focal_px=1.6,
                    min_points_per_view=50,
                    device=str(device),
                    max_total_samples=max_total,
                )
                score = crm._evaluate_fast3r_pc_mmd(  
                    metric,
                    sample,
                    resize=args.fast3r_resize,
                    device=device,
                )
            elif metric_name == "fast3r_pc_energy":
                max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                metric = crm.MEt3R_Fast3R_PointConsistency_Energy(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.0,
                    freeze=True,
                    rasterizer_kwargs={},
                    fast3r_weights=None,
                    fast3r_model=fast3r_backbone,
                    focal_length_estimation_method="first_view_from_global_head",
                    pnp_iterations=100,
                    default_focal_px=1.6,
                    min_points_per_view=50,
                    device=str(device),
                    max_total_samples=max_total,
                )
                score = crm._evaluate_fast3r_pc_energy(  
                    metric,
                    sample,
                    resize=args.fast3r_resize,
                    device=device,
                )
            else:
                raise ValueError(f"Unhandled metric_name={metric_name}")
        elif metric_name.startswith("vggt"):
            if metric_name == "vggt":
                metric = crm.MEt3R_VGGT(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.3,
                    freeze=True,
                    rasterizer_kwargs={},
                    vggt_weights="facebook/VGGT-1B",
                    device=str(device),
                )
                score = crm._evaluate_vggt(  
                    metric,
                    sample,
                    image_size=args.vggt_img_size,
                    device=device,
                )
            elif metric_name == "vggt_robust":
                metric = crm.MEt3R_VGGT_Robust(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.3,
                    freeze=True,
                    rasterizer_kwargs={},
                    vggt_weights="facebook/VGGT-1B",
                    device=str(device),
                )
                score = crm._evaluate_vggt(
                    metric,
                    sample,
                    image_size=args.vggt_img_size,
                    device=device,
                )
            elif metric_name == "vggt_mmd":
                per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                metric = crm.MEt3R_VGGT_MMD(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.3,
                    freeze=True,
                    rasterizer_kwargs={},
                    vggt_weights="facebook/VGGT-1B",
                    device=str(device),
                    per_pair_max_samples=per_pair_max,
                    max_total_samples=max_total,
                )
                score = crm._evaluate_vggt_mmd(  
                    metric,
                    sample,
                    image_size=args.vggt_img_size,
                    per_pair_max_samples=args.met3r_per_pair_max_samples,
                    max_total_samples=args.met3r_max_total_samples,
                    device=device,
                )
            elif metric_name == "vggt_pc":
                metric = crm.MEt3R_VGGT_PointConsistency(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.3,
                    freeze=True,
                    rasterizer_kwargs={},
                    vggt_weights="facebook/VGGT-1B",
                    device=str(device),
                )
                score = crm._evaluate_vggt_pc(  
                    metric,
                    sample,
                    image_size=args.vggt_img_size,
                    device=device,
                )
            elif metric_name == "vggt_pc_mmd":
                max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                metric = crm.MEt3R_VGGT_PointConsistency_MMD(
                    img_size=None,
                    distance="cosine",
                    feature_backbone="dinov2",
                    feature_backbone_weights="mhamilton723/FeatUp",
                    upsampler="featup",
                    use_norm=True,
                    confidence_threshold=0.3,
                    freeze=True,
                    rasterizer_kwargs={},
                    vggt_weights="facebook/VGGT-1B",
                    device=str(device),
                    max_total_samples=max_total,
                )
                score = crm._evaluate_vggt_pc_mmd(  
                    metric,
                    sample,
                    image_size=args.vggt_img_size,
                    device=device,
                )
            else:
                raise ValueError(f"Unhandled metric_name={metric_name}")
        else:
            raise ValueError(f"Unhandled metric_name={metric_name}")

        elapsed = time.perf_counter() - start
        logger.info(
            "metric=%s dataset=%s method=%s scene=%s K=%d score=%.6f elapsed=%.2fs",
            metric_name,
            dataset,
            method,
            scene,
            K,
            float(score),
            elapsed,
        )
        new_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "scene": scene,
                "subset_size": K,
                "num_input": len(input_images),
                "num_novel": len(novel_images),
                "num_total": len(image_paths),
                "metric": metric_name,
                "score": score,
                **crm._get_overlap_stats(metric),
            }
        )

    _write_metric_rows_atomic(out_path, fieldnames=fieldnames, rows=[*existing_rows, *new_rows])


if __name__ == "__main__":
    main()
