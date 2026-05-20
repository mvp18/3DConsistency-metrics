#!/usr/bin/env python3
"""
Evaluate backbone consistency metrics on one or more image directories.

This loads all images in each `--image-dir` in argument order, sorts each
directory by filename, and writes one row per metric to `--out-csv`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Iterable, List, Sequence


def _list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def _list_image_dirs(folders: Sequence[Path]) -> List[Path]:
    images: List[Path] = []
    for folder in folders:
        images.extend(_list_images(folder))
    return images


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
        "met3r_dust3r_mmd",
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Evaluate backbone consistency metrics on image directories.")
    parser.add_argument(
        "--image-dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing images. Multiple directories are concatenated in argument order.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="CSV path to write (defaults to <first-image-dir>/metrics.csv).",
    )
    parser.add_argument("--metrics", type=str, nargs="+", default=["all"])
    parser.add_argument("--met3r-img-size", type=int, default=224)
    parser.add_argument("--met3r-batch-pairs", type=int, default=4)
    parser.add_argument("--met3r-per-pair-max-samples", type=int, default=128)
    parser.add_argument("--met3r-max-total-samples", type=int, default=4096)
    parser.add_argument("--fast3r-resize", type=int, default=224)
    parser.add_argument("--vggt-img-size", type=int, default=224)
    args = parser.parse_args()

    image_dirs = [Path(p).expanduser().resolve() for p in args.image_dir]
    images = _list_image_dirs(image_dirs)
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images across {image_dirs}, found {len(images)}")

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else (image_dirs[0] / "metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    metric_names = _select_metrics(args.metrics)
    logger.info(
        "Evaluating %d images from %s with metrics=%s",
        len(images),
        _iterable_str(str(p) for p in image_dirs),
        _iterable_str(metric_names),
    )

    import torch

    import calibrate_recon_metrics as crm
    from fast3r.models.fast3r import Fast3R

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample = crm.MultiViewSample(
        sample_id="_".join(p.name for p in image_dirs),
        kind="consistent",
        subset_size=len(images),
        scene=None,
        image_paths=[str(p) for p in images],
    )

    fast3r_backbone = None

    fieldnames = ["image_dirs", "num_images", "metric", "score"]
    with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        f_csv.flush()

        for metric_name in metric_names:
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
                score = crm._evaluate_met3r(metric, sample, img_size=args.met3r_img_size, batch_pairs=args.met3r_batch_pairs, device=device)
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
            elif metric_name == "met3r_dust3r_mmd":
                per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                metric = crm.MEt3R_MMD(
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
                score = crm._evaluate_met3r_mmd(
                    metric,
                    sample,
                    img_size=args.met3r_img_size,
                    per_pair_max_samples=args.met3r_per_pair_max_samples,
                    max_total_samples=args.met3r_max_total_samples,
                    device=device,
                )
            elif metric_name.startswith("fast3r"):
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
                    score = crm._evaluate_fast3r(metric, sample, resize=args.fast3r_resize, device=device)
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
                    score = crm._evaluate_fast3r_pc(metric, sample, resize=args.fast3r_resize, device=device)
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
                    score = crm._evaluate_fast3r_pc_mmd(metric, sample, resize=args.fast3r_resize, device=device)
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
                    score = crm._evaluate_fast3r_pc_energy(metric, sample, resize=args.fast3r_resize, device=device)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
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
                    score = crm._evaluate_vggt(metric, sample, image_size=args.vggt_img_size, device=device)
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
                    score = crm._evaluate_vggt(metric, sample, image_size=args.vggt_img_size, device=device)
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
                    score = crm._evaluate_vggt_pc(metric, sample, image_size=args.vggt_img_size, device=device)
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
                    score = crm._evaluate_vggt_pc_mmd(metric, sample, image_size=args.vggt_img_size, device=device)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

            elapsed = time.perf_counter() - start
            logger.info("metric=%s score=%.6f elapsed=%.2fs", metric_name, float(score), elapsed)

            writer.writerow(
                {
                    "image_dirs": _iterable_str(str(p) for p in image_dirs),
                    "num_images": len(images),
                    "metric": metric_name,
                    "score": score,
                }
            )
            f_csv.flush()


if __name__ == "__main__":
    main()
