#!/usr/bin/env python3
import argparse
from typing import List

import torch

from fast3r.eval.met3r_fast3r import MEt3R_Fast3R
from fast3r.dust3r.utils.image import load_images


def _stack_images_to_batch(views: List[dict]) -> torch.Tensor:
    imgs = [view["img"] for view in views]
    stacked = torch.cat(imgs, dim=0)
    return (stacked + 1.0) * 0.5  # [-1,1] -> [0,1]


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    views = load_images(args.img_dir, size=args.resize, verbose=True)
    images_01 = _stack_images_to_batch(views).unsqueeze(0).to(device)

    metric = MEt3R_Fast3R(
        img_size=None,
        distance=args.distance,
        feature_backbone=args.feature_backbone,
        feature_backbone_weights=args.feature_backbone_weights,
        upsampler=args.upsampler,
        confidence_threshold=args.confidence_threshold,
        fast3r_weights=args.fast3r_weights,
        focal_length_estimation_method=args.focal_method,
        pnp_iterations=args.pnp_iterations,
        device=device,
    ).to(device).eval()

    score, = metric(images_01)
    print(f"MEt3R-Fast3R score: {score.mean().item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multi-view Fast3R self-consistency metric.")
    parser.add_argument("--img_dir", "-i", type=str, help="Directory containing input images.")
    parser.add_argument("--resize", type=int, default=224, help="Long-side resize applied before Fast3R (default: 512).")
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine", "mse", "rmse", "psnr", "lpips"], help="Distance used for the metric.")
    parser.add_argument("--feature-backbone", type=str, default="dinov2", help="Feature backbone for cosine distance.")
    parser.add_argument("--feature-backbone-weights", type=str, default="mhamilton723/FeatUp", help="torch.hub repo for the feature model.")
    parser.add_argument("--upsampler", type=str, default="featup", choices=["featup", "nearest", "bilinear", "bicubic"], help="Upsampling strategy for features.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Confidence threshold for filtering Fast3R points.")
    parser.add_argument("--fast3r-weights", type=str, default="jedyang97/Fast3R_ViT_Large_512", help="HuggingFace repo or local path for Fast3R weights.")
    parser.add_argument("--focal-method", type=str, default="first_view_from_global_head", help="Focal length estimation strategy.")
    parser.add_argument("--pnp-iterations", type=int, default=100, help="Number of iterations for the internal PnP solver.")
    args = parser.parse_args()

    main(args)
