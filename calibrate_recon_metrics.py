#!/usr/bin/env python3
"""
Run multiple 3D consistency metrics on Mip-NeRF360 calibration splits.

This script evaluates four metrics:
  - MEt3R (pairwise image consistency, MASt3R/DUSt3R-based)
  - MEt3R-MMD (global 1-sample MMD over per-pixel errors)
  - MEt3R-Fast3R (Fast3R-driven multiview consistency)
  - MEt3R-VGGT (VGGT-driven multiview consistency)

on:
  * Real, 3D-consistent subsets per scene (from `mipnerf360_calibration_splits.json`).
  * Cross-scene mixtures (views drawn from different scenes).
  * One-outlier mixtures (K-1 views from one scene + 1 outlier view).
  * Pure noise "scenes" with no geometric consistency (uniform + Gaussian).
  * Patched scenes (consistent views with Gaussian-noise patches).
  * Impossible-sample definitions are read from `tmp/syscon3d_release/mipnerf360_impossible_splits.json`
    so they are reproducible across runs.

Outputs:
  * A CSV with one row per (sample, metric) containing the raw score.
  * An optional JSON summary with per-metric statistics and simple
    threshold suggestions separating consistent from each type of
    impossible scene (mixed vs. pure noise).

This script is intended to be run once dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Import metric implementations from local checkouts of met3r / fast3r / vggt
# ---------------------------------------------------------------------------

import sys

ROOT = Path(__file__).resolve().parent

# Make local checkouts importable without requiring editable installs.
for subdir in ("met3r", "fast3r", "vggt"):
    candidate = ROOT / subdir
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from met3r import MEt3R, MEt3R_MMD
from met3r.met3r import MEt3R_Energy, MEt3R_IMQ
from images import load_images as met3r_load_images

from fast3r.eval.met3r_fast3r import (
    MEt3R_Fast3R,
    MEt3R_Fast3R_MMD,
    MEt3R_Fast3R_Energy,
    MEt3R_Fast3R_IMQ,
    MEt3R_Fast3R_PointConsistency,
    MEt3R_Fast3R_PointConsistency_MMD,
    MEt3R_Fast3R_PointConsistency_Energy,
    MEt3R_Fast3R_PointConsistency_IMQ,
)
from fast3r.dust3r.utils.image import load_images as fast3r_load_images

from met3r_vggt import (
    MEt3R_VGGT,
    MEt3R_VGGT_Robust,
    MEt3R_VGGT_MMD,
    MEt3R_VGGT_Energy,
    MEt3R_VGGT_IMQ,
    MEt3R_VGGT_PointConsistency,
    MEt3R_VGGT_PointConsistency_MMD,
    MEt3R_VGGT_PointConsistency_Energy,
    MEt3R_VGGT_PointConsistency_IMQ,
)
from vggt.utils.load_fn import load_and_preprocess_images  


SampleKind = Literal[
    "consistent",
    "mixed",
    "mixed_controlled",
    "mixed_one_outlier",
    "identical_images",
    "consistent_gaussian_epsilon",
    "full_mixture_distinct_scene",
    "noise",
    "noise_gaussian",
    "patched_gaussian",
]
MetricName = Literal[
    "met3r",
    "met3r_mmd",
    "met3r_energy",
    "met3r_imq",
    "met3r_dust3r",
    "met3r_dust3r_mmd",
    "met3r_dust3r_energy",
    "met3r_dust3r_imq",
    "fast3r",
    "fast3r_mmd",
    "fast3r_energy",
    "fast3r_imq",
    "fast3r_pc",
    "fast3r_pc_mmd",
    "fast3r_pc_energy",
    "fast3r_pc_imq",
    "vggt",
    "vggt_robust",
    "vggt_mmd",
    "vggt_energy",
    "vggt_imq",
    "vggt_pc",
    "vggt_pc_mmd",
    "vggt_pc_energy",
    "vggt_pc_imq",
    "prism_mmd",
    "sed",
    "tsed",
]


@dataclass
class MultiViewSample:
    """Definition of a multiview sample used for calibration."""

    sample_id: str
    kind: SampleKind
    subset_size: int
    scene: Optional[str]
    image_paths: Optional[List[str]]  # None for synthetic noise scenes
    noise_seed: Optional[int] = None
    noise_type: Optional[Literal["uniform", "gaussian"]] = None
    noise_gaussian_sigma: Optional[float] = None
    patch_seed: Optional[int] = None
    patch_ratio: Optional[float] = None
    patch_num_patches: Optional[int] = None
    patch_gaussian_sigma: Optional[float] = None
    materialized: bool = False


def _to_01(images_signed: torch.Tensor) -> torch.Tensor:
    return (images_signed + 1.0) * 0.5


def _to_signed(images_01: torch.Tensor) -> torch.Tensor:
    return images_01 * 2.0 - 1.0


def _generate_noise_images_01(
    *,
    num_images: int,
    size: int,
    seed: int,
    device: torch.device,
    noise_type: Literal["uniform", "gaussian"],
    gaussian_sigma: float,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    if noise_type == "uniform":
        return torch.rand(num_images, 3, size, size, generator=g, device=device)
    if noise_type == "gaussian":
        x = torch.randn(num_images, 3, size, size, generator=g, device=device) * float(gaussian_sigma) + 0.5
        return x.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def _add_gaussian_patches_01(
    images_01: torch.Tensor,
    *,
    patch_ratio: float,
    num_patches: int,
    seed: int,
    gaussian_sigma: float,
) -> torch.Tensor:
    if not (0.0 < patch_ratio <= 1.0):
        raise ValueError(f"patch_ratio must be in (0, 1], got {patch_ratio}")
    if num_patches <= 0:
        return images_01

    patched = images_01.clone()
    n, c, h, w = patched.shape
    patch_h = max(1, int(h * patch_ratio))
    patch_w = max(1, int(w * patch_ratio))

    g = torch.Generator(device=patched.device)
    g.manual_seed(int(seed))
    for i in range(n):
        for _ in range(int(num_patches)):
            y = int(torch.randint(0, h - patch_h + 1, (1,), generator=g, device=patched.device).item())
            x = int(torch.randint(0, w - patch_w + 1, (1,), generator=g, device=patched.device).item())
            patch = torch.randn(c, patch_h, patch_w, generator=g, device=patched.device) * float(gaussian_sigma) + 0.5
            patch = patch.clamp(0.0, 1.0)
            patched[i, :, y : y + patch_h, x : x + patch_w] = patch
    return patched


DEFAULT_GAUSSIAN_SIGMA = 0.2
DEFAULT_PATCH_RATIO = 0.25
DEFAULT_PATCH_NUM_PATCHES = 4
DEFAULT_DL3DV_BENCHMARK_ROOT = (
    "tmp/syscon3d_release/dl3dv_benchmark"
)


def _noise_params(sample: MultiViewSample) -> Tuple[int, Literal["uniform", "gaussian"], float]:
    seed = int(sample.noise_seed or 0)
    noise_type = sample.noise_type or "uniform"
    sigma = float(sample.noise_gaussian_sigma or DEFAULT_GAUSSIAN_SIGMA)
    return seed, noise_type, sigma


def _patch_params(sample: MultiViewSample) -> Tuple[int, float, int, float]:
    seed = int(sample.patch_seed or 0)
    ratio = float(sample.patch_ratio or DEFAULT_PATCH_RATIO)
    num_patches = int(sample.patch_num_patches or DEFAULT_PATCH_NUM_PATCHES)
    sigma = float(sample.patch_gaussian_sigma or DEFAULT_GAUSSIAN_SIGMA)
    return seed, ratio, num_patches, sigma


def _maybe_patch_images_01(images_01: torch.Tensor, sample: MultiViewSample) -> torch.Tensor:
    if sample.kind != "patched_gaussian" or sample.materialized:
        return images_01
    seed, ratio, num_patches, sigma = _patch_params(sample)
    return _add_gaussian_patches_01(
        images_01,
        patch_ratio=ratio,
        num_patches=num_patches,
        seed=seed,
        gaussian_sigma=sigma,
    )


def _load_images_met3r_signed(sample: MultiViewSample, *, img_size: int, device: torch.device) -> torch.Tensor:
    if sample.image_paths is None:
        seed, noise_type, sigma = _noise_params(sample)
        imgs_01 = _generate_noise_images_01(
            num_images=sample.subset_size,
            size=img_size,
            seed=seed,
            device=device,
            noise_type=noise_type,
            gaussian_sigma=sigma,
        )
        return _to_signed(imgs_01)

    img_list = met3r_load_images(sample.image_paths, size=img_size, square_ok=False, verbose=False)
    imgs = torch.cat(img_list, dim=0).to(device)
    imgs_01 = _to_01(imgs)
    imgs_01 = _maybe_patch_images_01(imgs_01, sample)
    return _to_signed(imgs_01)


def _load_images_fast3r_01_batch(sample: MultiViewSample, *, resize: int, device: torch.device) -> torch.Tensor:
    K = sample.subset_size
    if K < 2:
        raise ValueError("Need at least 2 views for Fast3R evaluation.")

    if sample.image_paths is None:
        seed, noise_type, sigma = _noise_params(sample)
        images_01 = _generate_noise_images_01(
            num_images=K,
            size=resize,
            seed=seed,
            device=device,
            noise_type=noise_type,
            gaussian_sigma=sigma,
        )
        return images_01.unsqueeze(0)

    views = fast3r_load_images(sample.image_paths, size=resize, square_ok=False, verbose=False)
    imgs = [v["img"] for v in views]
    stacked = torch.cat(imgs, dim=0).to(device)  # [-1, 1]
    images_01 = (stacked + 1.0) * 0.5  # [0, 1]
    images_01 = _maybe_patch_images_01(images_01, sample)
    return images_01.unsqueeze(0)


def _load_images_vggt_01(sample: MultiViewSample, *, image_size: int, device: torch.device) -> torch.Tensor:
    K = sample.subset_size
    if K < 2:
        raise ValueError("Need at least 2 views for VGGT evaluation.")

    if sample.image_paths is None:
        seed, noise_type, sigma = _noise_params(sample)
        images = _generate_noise_images_01(
            num_images=K,
            size=image_size,
            seed=seed,
            device=device,
            noise_type=noise_type,
            gaussian_sigma=sigma,
        )
        return images

    images = load_and_preprocess_images(sample.image_paths, target_size=image_size).to(device)
    return _maybe_patch_images_01(images, sample)


def _load_calibration_splits(
    splits_path: Path,
) -> Tuple[Path, Dict[str, Dict[int, List[int]]], List[int]]:
    """
    Load the calibration splits manifest produced by mipnerf360_prepare_calibration_splits.py.

    Returns:
        dataset_root: path to the Mip-NeRF360 dataset root.
        scenes_splits: mapping from scene name to {subset_size -> list of frame indices}.
        subset_sizes: sorted list of subset sizes available in the manifest.
    """
    with splits_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_root = Path(data["dataset_root"])
    subset_sizes_raw = data.get("subset_sizes", [])
    subset_sizes = sorted({int(k) for k in subset_sizes_raw})

    scenes_raw = data.get("scenes", {})
    scenes_splits: Dict[str, Dict[int, List[int]]] = {}
    for scene_name, per_size in scenes_raw.items():
        size_map: Dict[int, List[int]] = {}
        for k_str, indices in per_size.items():
            k_int = int(k_str)
            size_map[k_int] = [int(i) for i in indices]
        scenes_splits[scene_name] = size_map

    return dataset_root, scenes_splits, subset_sizes


def _load_transforms(
    dataset_root: Path,
    scene_names: Sequence[str],
) -> Dict[str, Dict]:
    """
    Load transforms.json for each scene into memory.
    """
    transforms: Dict[str, Dict] = {}
    for scene in scene_names:
        scene_dir = dataset_root / scene
        path = scene_dir / "transforms.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing transforms.json for scene {scene} at {path}")
        with path.open("r", encoding="utf-8") as f:
            transforms[scene] = json.load(f)
    return transforms


def _build_scene_image_paths(
    dataset_root: Path,
    transforms: Dict[str, Dict],
) -> Dict[str, List[str]]:
    """
    Precompute absolute image paths for every frame in each scene.
    """
    scene_paths: Dict[str, List[str]] = {}
    for scene, tf in transforms.items():
        frames = tf.get("frames", [])
        scene_dir = dataset_root / scene
        paths: List[str] = []
        for frame in frames:
            rel = frame["file_path"]
            # Handle leading "./" but keep relative structure.
            img_path = (scene_dir / rel).resolve()
            paths.append(str(img_path))
        scene_paths[scene] = paths
    return scene_paths


def _build_consistent_samples(
    dataset_root: Path,
    scenes_splits: Dict[str, Dict[int, List[int]]],
    transforms: Dict[str, Dict],
    subset_sizes: Sequence[int],
) -> List[MultiViewSample]:
    """
    Build MultiViewSample objects for real, 3D-consistent subsets.
    """
    scene_paths = _build_scene_image_paths(dataset_root, transforms)
    samples: List[MultiViewSample] = []
    for scene, per_size in scenes_splits.items():
        frame_paths = scene_paths[scene]
        for k in subset_sizes:
            indices = per_size.get(k)
            if not indices:
                continue
            # Guard against stale splits.
            valid_indices = [i for i in indices if 0 <= i < len(frame_paths)]
            if len(valid_indices) < 2:
                continue
            image_paths = [frame_paths[i] for i in valid_indices]
            sample_id = f"{scene}_k{k:02d}"
            samples.append(
                MultiViewSample(
                    sample_id=sample_id,
                    kind="consistent",
                    subset_size=k,
                    scene=scene,
                    image_paths=image_paths,
                )
            )
    return samples


def _build_mixed_samples(
    scene_paths: Dict[str, List[str]],
    subset_sizes: Sequence[int],
    num_mixed_per_size: int,
    rng: random.Random,
) -> List[MultiViewSample]:
    """
    Build cross-scene mixtures by sampling views from different scenes.
    """
    scenes = sorted(scene_paths.keys())
    samples: List[MultiViewSample] = []
    if not scenes or num_mixed_per_size <= 0:
        return samples

    for k in subset_sizes:
        if k <= 0:
            continue
        for mix_idx in range(num_mixed_per_size):
            image_paths: List[str] = []
            for view_idx in range(k):
                scene = rng.choice(scenes)
                candidates = scene_paths[scene]
                if not candidates:
                    continue
                img_path = rng.choice(candidates)
                image_paths.append(img_path)
            if len(image_paths) < 2:
                continue
            sample_id = f"mixed_k{k:02d}_{mix_idx:03d}"
            samples.append(
                MultiViewSample(
                    sample_id=sample_id,
                    kind="mixed",
                    subset_size=k,
                    scene=None,
                    image_paths=image_paths,
                )
            )
    return samples


def _build_noise_samples(
    subset_sizes: Sequence[int],
    num_noise_per_size: int,
    *,
    kind: SampleKind,
    noise_type: Literal["uniform", "gaussian"],
    gaussian_sigma: float,
    rng: random.Random,
) -> List[MultiViewSample]:
    """
    Build synthetic noise "scenes" without associated image paths.
    """
    samples: List[MultiViewSample] = []
    if num_noise_per_size <= 0:
        return samples
    for k in subset_sizes:
        if k <= 0:
            continue
        for noise_idx in range(num_noise_per_size):
            prefix = "noise" if kind == "noise" else str(kind)
            sample_id = f"{prefix}_k{k:02d}_{noise_idx:03d}"
            samples.append(
                MultiViewSample(
                    sample_id=sample_id,
                    kind=kind,
                    subset_size=k,
                    scene=None,
                    image_paths=None,
                    noise_seed=int(rng.randrange(2**31)),
                    noise_type=noise_type,
                    noise_gaussian_sigma=float(gaussian_sigma) if noise_type == "gaussian" else None,
                )
            )
    return samples


def _build_one_outlier_samples(
    scene_paths: Dict[str, List[str]],
    subset_sizes: Sequence[int],
    num_per_size: int,
    rng: random.Random,
) -> List[MultiViewSample]:
    scenes = sorted(scene_paths.keys())
    samples: List[MultiViewSample] = []
    if len(scenes) < 2 or num_per_size <= 0:
        return samples

    for k in subset_sizes:
        if k <= 1:
            continue
        for idx in range(num_per_size):
            base_scene = rng.choice(scenes)
            base_candidates = scene_paths.get(base_scene, [])
            if not base_candidates:
                continue
            image_paths = [rng.choice(base_candidates) for _ in range(k)]

            outlier_scenes = [s for s in scenes if s != base_scene]
            outlier_scene = rng.choice(outlier_scenes)
            outlier_candidates = scene_paths.get(outlier_scene, [])
            if not outlier_candidates:
                continue
            image_paths[-1] = rng.choice(outlier_candidates)
            samples.append(
                MultiViewSample(
                    sample_id=f"mixed_one_outlier_k{k:02d}_{idx:03d}",
                    kind="mixed_one_outlier",
                    subset_size=k,
                    scene=None,
                    image_paths=image_paths,
                )
            )
    return samples


# Pre-computed outlier counts per K for mixed_controlled: ~30% of views are foreign,
# giving roughly 50-60% cross-scene pairs (midpoint between one_outlier and fully mixed).
_CONTROLLED_OUTLIER_COUNTS: Dict[int, int] = {
    3: 2, 6: 2, 9: 3, 12: 4, 15: 4, 18: 5, 21: 6,
}


def _controlled_n_outlier(k: int) -> int:
    """Number of foreign views for mixed_controlled at subset size *k*."""
    return _CONTROLLED_OUTLIER_COUNTS.get(k, max(2, round(k * 0.3)))


def _build_controlled_mixed_samples(
    scene_paths: Dict[str, List[str]],
    subset_sizes: Sequence[int],
    num_per_size: int,
    rng: random.Random,
) -> List[MultiViewSample]:
    """Build mixed samples with a controlled number of foreign views (~30 %).

    For each sample, K - n_outlier views come from a single base scene and
    n_outlier views each come from a *different* foreign scene.  This yields
    roughly 50-60 % cross-scene pairs, halfway between mixed_one_outlier and
    the fully scrambled mixed variant.
    """
    scenes = sorted(scene_paths.keys())
    samples: List[MultiViewSample] = []
    if len(scenes) < 2 or num_per_size <= 0:
        return samples

    for k in subset_sizes:
        if k <= 1:
            continue
        n_outlier = _controlled_n_outlier(k)
        n_base = k - n_outlier
        for idx in range(num_per_size):
            base_scene = rng.choice(scenes)
            base_candidates = scene_paths.get(base_scene, [])
            if not base_candidates:
                continue
            image_paths = [rng.choice(base_candidates) for _ in range(n_base)]

            foreign_scenes = [s for s in scenes if s != base_scene]
            rng.shuffle(foreign_scenes)
            ok = True
            for fi in range(n_outlier):
                src = foreign_scenes[fi % len(foreign_scenes)]
                cands = scene_paths.get(src, [])
                if not cands:
                    ok = False
                    break
                image_paths.append(rng.choice(cands))
            if not ok or len(image_paths) < 2:
                continue

            rng.shuffle(image_paths)
            samples.append(
                MultiViewSample(
                    sample_id=f"mixed_controlled_k{k:02d}_{idx:03d}",
                    kind="mixed_controlled",
                    subset_size=k,
                    scene=None,
                    image_paths=image_paths,
                )
            )
    return samples


def _build_patched_samples(
    scene_paths: Dict[str, List[str]],
    subset_sizes: Sequence[int],
    num_per_size: int,
    *,
    patch_ratio: float,
    patch_num_patches: int,
    gaussian_sigma: float,
    rng: random.Random,
) -> List[MultiViewSample]:
    scenes = sorted(scene_paths.keys())
    samples: List[MultiViewSample] = []
    if not scenes or num_per_size <= 0:
        return samples

    for k in subset_sizes:
        if k <= 1:
            continue
        for idx in range(num_per_size):
            scene = rng.choice(scenes)
            candidates = scene_paths.get(scene, [])
            if not candidates:
                continue
            image_paths = [rng.choice(candidates) for _ in range(k)]
            samples.append(
                MultiViewSample(
                    sample_id=f"patched_gaussian_k{k:02d}_{idx:03d}",
                    kind="patched_gaussian",
                    subset_size=k,
                    scene=None,
                    image_paths=image_paths,
                    patch_seed=int(rng.randrange(2**31)),
                    patch_ratio=float(patch_ratio),
                    patch_num_patches=int(patch_num_patches),
                    patch_gaussian_sigma=float(gaussian_sigma),
                )
            )
    return samples


def _build_identical_image_samples(
    scene_paths: Dict[str, List[str]],
    subset_sizes: Sequence[int],
    num_per_size: int,
    rng: random.Random,
) -> List[MultiViewSample]:
    scenes = sorted(scene_paths.keys())
    samples: List[MultiViewSample] = []
    if not scenes or num_per_size <= 0:
        return samples

    for k in subset_sizes:
        if k <= 1:
            continue
        for idx in range(num_per_size):
            scene = rng.choice(scenes)
            candidates = scene_paths.get(scene, [])
            if not candidates:
                continue
            img_path = rng.choice(candidates)
            samples.append(
                MultiViewSample(
                    sample_id=f"identical_images_k{k:02d}_{idx:03d}",
                    kind="identical_images",
                    subset_size=k,
                    scene=None,
                    image_paths=[img_path] * k,
                )
            )
    return samples


def _collect_dl3dv_scene_image_paths(dl3dv_root: Path) -> Dict[str, List[str]]:
    if not dl3dv_root.exists():
        raise FileNotFoundError(f"DL3DV root does not exist: {dl3dv_root}")

    scene_paths: Dict[str, List[str]] = {}
    for scene_dir in sorted(dl3dv_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("."):
            continue
        image_dirs = [
            scene_dir / "gaussian_splat" / "images_4",
            scene_dir / "nerfstudio" / "images_4",
        ]
        for image_dir in image_dirs:
            if not image_dir.is_dir():
                continue
            image_paths = sorted(
                str(path.resolve())
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if image_paths:
                scene_paths[scene_dir.name] = image_paths
                break
    return scene_paths


def _build_full_mixture_distinct_scene_samples(
    mipnerf_scene_paths: Dict[str, List[str]],
    dl3dv_scene_paths: Dict[str, List[str]],
    subset_sizes: Sequence[int],
    num_per_size: int,
    rng: random.Random,
) -> List[MultiViewSample]:
    samples: List[MultiViewSample] = []
    if num_per_size <= 0:
        return samples

    for k in subset_sizes:
        if k <= 1:
            continue
        source_scene_paths = mipnerf_scene_paths if k <= 9 else dl3dv_scene_paths
        scenes = sorted(source_scene_paths.keys())
        if len(scenes) < k:
            raise ValueError(
                f"Need at least {k} distinct scenes for full_mixture_distinct_scene, "
                f"but only found {len(scenes)}."
            )
        for idx in range(num_per_size):
            selected_scenes = rng.sample(scenes, k)
            image_paths = [rng.choice(source_scene_paths[scene]) for scene in selected_scenes]
            rng.shuffle(image_paths)
            samples.append(
                MultiViewSample(
                    sample_id=f"full_mixture_distinct_scene_k{k:02d}_{idx:03d}",
                    kind="full_mixture_distinct_scene",
                    subset_size=k,
                    scene=None,
                    image_paths=image_paths,
                )
            )
    return samples


def _relative_to_dataset_root(path: str, dataset_root: Path) -> str:
    """
    Convert an absolute image path to a path relative to dataset_root when possible.
    """
    p = Path(path)
    try:
        return str(p.relative_to(dataset_root))
    except ValueError:
        return str(p)


def _load_manifest_paths(entry: dict, dataset_root: Path) -> List[str]:
    loaded_paths: List[str] = []
    for path_str in entry.get("image_rel_paths", []):
        path = Path(path_str)
        if path.is_absolute():
            loaded_paths.append(str(path))
            continue
        roots = [dataset_root]
        extra_root = os.environ.get("SYSCON3D_EXTRA_DATA_ROOT")
        if extra_root:
            roots.extend(Path(root).expanduser() for root in extra_root.split(os.pathsep) if root)
        roots.append(ROOT / "tmp" / "syscon3d_scene_types_source")
        for root in roots:
            candidate = root / path
            if candidate.exists():
                loaded_paths.append(str(candidate))
                break
        else:
            loaded_paths.append(str(dataset_root / path))
    return loaded_paths


def _load_or_build_impossible_samples(
    *,
    dataset_root: Path,
    scene_paths: Dict[str, List[str]],
    dl3dv_root: Path,
    subset_sizes: Sequence[int],
    num_mixed_per_size: int,
    num_noise_per_size: int,
    num_gaussian_noise_per_size: int,
    num_one_outlier_per_size: int,
    num_controlled_per_size: int,
    num_identical_per_size: int,
    num_full_distinct_per_size: int,
    num_patched_per_size: int,
    gaussian_sigma: float,
    patch_ratio: float,
    patch_num_patches: int,
    rng: random.Random,
    impossible_splits_path: Path,
    save_if_missing: bool,
) -> List[MultiViewSample]:
    """
    Load precomputed impossible samples if available, otherwise build them and optionally save a
    manifest to disk under data/.
    """
    if impossible_splits_path.exists():
        with impossible_splits_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_subset_sizes = {int(k) for k in data.get("subset_sizes", [])}
        required_keys = {"mixed", "mixed_one_outlier", "noise", "noise_gaussian", "patched_gaussian"}
        if (
            data.get("version") in (2, 3, 4, 5, 6)
            and loaded_subset_sizes.issuperset(set(subset_sizes))
            and required_keys.issubset(set(data.keys()))
        ):
            samples: List[MultiViewSample] = []

            for entry in data.get("mixed", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="mixed",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=_load_manifest_paths(entry, dataset_root),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )

            for entry in data.get("mixed_controlled", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="mixed_controlled",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=_load_manifest_paths(entry, dataset_root),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )

            for entry in data.get("mixed_one_outlier", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="mixed_one_outlier",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=_load_manifest_paths(entry, dataset_root),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )

            for entry in data.get("identical_images", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="identical_images",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=_load_manifest_paths(entry, dataset_root),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )
            for entry in data.get("consistent_gaussian_epsilon", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="consistent_gaussian_epsilon",
                        subset_size=subset_size,
                        scene=str(entry.get("source_scene", "")) or None,
                        image_paths=_load_manifest_paths(entry, dataset_root),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )
            for entry in data.get("full_mixture_distinct_scene", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="full_mixture_distinct_scene",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=_load_manifest_paths(entry, dataset_root),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )

            for entry in data.get("noise", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                image_paths = _load_manifest_paths(entry, dataset_root) if entry.get("image_rel_paths") else None
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="noise",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=image_paths,
                        noise_seed=int(entry["seed"]),
                        noise_type="uniform",
                        materialized=bool(entry.get("materialized", False)),
                    )
                )

            for entry in data.get("noise_gaussian", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                image_paths = _load_manifest_paths(entry, dataset_root) if entry.get("image_rel_paths") else None
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="noise_gaussian",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=image_paths,
                        noise_seed=int(entry["seed"]),
                        noise_type="gaussian",
                        noise_gaussian_sigma=float(entry["gaussian_sigma"]),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )

            for entry in data.get("patched_gaussian", []):
                subset_size = int(entry["subset_size"])
                if subset_size not in subset_sizes:
                    continue
                samples.append(
                    MultiViewSample(
                        sample_id=entry["sample_id"],
                        kind="patched_gaussian",
                        subset_size=subset_size,
                        scene=None,
                        image_paths=_load_manifest_paths(entry, dataset_root),
                        patch_seed=int(entry["patch_seed"]),
                        patch_ratio=float(entry["patch_ratio"]),
                        patch_num_patches=int(entry["patch_num_patches"]),
                        patch_gaussian_sigma=float(entry["gaussian_sigma"]),
                        materialized=bool(entry.get("materialized", False)),
                    )
                )

            updated = False

            if "mixed_controlled" not in data:
                controlled_samples = _build_controlled_mixed_samples(
                    scene_paths, subset_sizes, num_controlled_per_size, rng,
                )
                samples.extend(controlled_samples)
                data["mixed_controlled"] = [
                    {
                        "sample_id": s.sample_id,
                        "subset_size": s.subset_size,
                        "image_rel_paths": [
                            _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                        ],
                    }
                    for s in controlled_samples
                ]
                updated = True

            if "identical_images" not in data:
                identical_samples = _build_identical_image_samples(
                    scene_paths, subset_sizes, num_identical_per_size, rng,
                )
                samples.extend(identical_samples)
                data["identical_images"] = [
                    {
                        "sample_id": s.sample_id,
                        "subset_size": s.subset_size,
                        "image_rel_paths": [
                            _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                        ],
                    }
                    for s in identical_samples
                ]
                updated = True

            if "full_mixture_distinct_scene" not in data:
                dl3dv_scene_paths = _collect_dl3dv_scene_image_paths(dl3dv_root)
                full_distinct_samples = _build_full_mixture_distinct_scene_samples(
                    scene_paths,
                    dl3dv_scene_paths,
                    subset_sizes,
                    num_full_distinct_per_size,
                    rng,
                )
                samples.extend(full_distinct_samples)
                data["full_mixture_distinct_scene"] = [
                    {
                        "sample_id": s.sample_id,
                        "subset_size": s.subset_size,
                        "image_rel_paths": [
                            _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                        ],
                    }
                    for s in full_distinct_samples
                ]
                updated = True

            if updated:
                data["version"] = 4
                data["subset_sizes"] = sorted(loaded_subset_sizes | set(subset_sizes))
                if save_if_missing:
                    with impossible_splits_path.open("w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, sort_keys=True)
                    logging.getLogger(__name__).info(
                        "Migrated impossible splits to v4 with new impossible kinds.",
                    )

            return samples

    mixed_samples = _build_mixed_samples(scene_paths, subset_sizes, num_mixed_per_size, rng)
    one_outlier_samples = _build_one_outlier_samples(scene_paths, subset_sizes, num_one_outlier_per_size, rng)
    controlled_samples = _build_controlled_mixed_samples(scene_paths, subset_sizes, num_controlled_per_size, rng)
    identical_samples = _build_identical_image_samples(scene_paths, subset_sizes, num_identical_per_size, rng)
    dl3dv_scene_paths = _collect_dl3dv_scene_image_paths(dl3dv_root)
    full_distinct_samples = _build_full_mixture_distinct_scene_samples(
        scene_paths,
        dl3dv_scene_paths,
        subset_sizes,
        num_full_distinct_per_size,
        rng,
    )
    noise_samples = _build_noise_samples(
        subset_sizes,
        num_noise_per_size,
        kind="noise",
        noise_type="uniform",
        gaussian_sigma=gaussian_sigma,
        rng=rng,
    )
    noise_gaussian_samples = _build_noise_samples(
        subset_sizes,
        num_gaussian_noise_per_size,
        kind="noise_gaussian",
        noise_type="gaussian",
        gaussian_sigma=gaussian_sigma,
        rng=rng,
    )
    patched_samples = _build_patched_samples(
        scene_paths,
        subset_sizes,
        num_patched_per_size,
        patch_ratio=patch_ratio,
        patch_num_patches=patch_num_patches,
        gaussian_sigma=gaussian_sigma,
        rng=rng,
    )

    all_samples = (
        mixed_samples
        + controlled_samples
        + one_outlier_samples
        + identical_samples
        + full_distinct_samples
        + noise_samples
        + noise_gaussian_samples
        + patched_samples
    )

    if save_if_missing and all_samples:
        impossible_splits_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 4,
            "dataset_root": str(dataset_root),
            "subset_sizes": sorted({s.subset_size for s in all_samples}),
            "mixed": [
                {
                    "sample_id": s.sample_id,
                    "subset_size": s.subset_size,
                    "image_rel_paths": [
                        _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                    ],
                }
                for s in mixed_samples
            ],
            "mixed_controlled": [
                {
                    "sample_id": s.sample_id,
                    "subset_size": s.subset_size,
                    "image_rel_paths": [
                        _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                    ],
                }
                for s in controlled_samples
            ],
            "mixed_one_outlier": [
                {
                    "sample_id": s.sample_id,
                    "subset_size": s.subset_size,
                    "image_rel_paths": [
                        _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                    ],
                }
                for s in one_outlier_samples
            ],
            "identical_images": [
                {
                    "sample_id": s.sample_id,
                    "subset_size": s.subset_size,
                    "image_rel_paths": [
                        _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                    ],
                }
                for s in identical_samples
            ],
            "full_mixture_distinct_scene": [
                {
                    "sample_id": s.sample_id,
                    "subset_size": s.subset_size,
                    "image_rel_paths": [
                        _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                    ],
                }
                for s in full_distinct_samples
            ],
            "noise": [{"sample_id": s.sample_id, "subset_size": s.subset_size, "seed": s.noise_seed} for s in noise_samples],
            "noise_gaussian": [
                {
                    "sample_id": s.sample_id,
                    "subset_size": s.subset_size,
                    "seed": s.noise_seed,
                    "gaussian_sigma": s.noise_gaussian_sigma,
                }
                for s in noise_gaussian_samples
            ],
            "patched_gaussian": [
                {
                    "sample_id": s.sample_id,
                    "subset_size": s.subset_size,
                    "image_rel_paths": [
                        _relative_to_dataset_root(p, dataset_root) for p in (s.image_paths or [])
                    ],
                    "patch_seed": s.patch_seed,
                    "patch_ratio": s.patch_ratio,
                    "patch_num_patches": s.patch_num_patches,
                    "gaussian_sigma": s.patch_gaussian_sigma,
                }
                for s in patched_samples
            ],
        }
        with impossible_splits_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    return all_samples


# ---------------------------------------------------------------------------
# Metric evaluation helpers
# ---------------------------------------------------------------------------

MmdSigmaByMetric = Dict[str, Dict[int, float]]


MMD_METRICS: set[str] = {
    "met3r_mmd",
    "met3r_dust3r_mmd",
    "fast3r_mmd",
    "fast3r_pc_mmd",
    "vggt_mmd",
    "vggt_pc_mmd",
}

IMQ_METRICS: set[str] = {
    "met3r_imq",
    "met3r_dust3r_imq",
    "fast3r_imq",
    "fast3r_pc_imq",
    "vggt_imq",
    "vggt_pc_imq",
}

IMQ_TO_SIGMA_METRIC: dict[str, str] = {
    "met3r_imq": "met3r_mmd",
    "met3r_dust3r_imq": "met3r_dust3r_mmd",
    "fast3r_imq": "fast3r_mmd",
    "fast3r_pc_imq": "fast3r_pc_mmd",
    "vggt_imq": "vggt_mmd",
    "vggt_pc_imq": "vggt_pc_mmd",
}

AzimuthPlane = Literal["xy", "xz", "yz"]


def _stable_int_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _wrap_degrees(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _position_from_matrix(value: object) -> Tuple[float, float, float]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        rows = value
        if len(rows) == 3 and len(rows[0]) == 4:
            return float(rows[0][3]), float(rows[1][3]), float(rows[2][3])
        if len(rows) == 4 and len(rows[0]) == 4:
            return float(rows[0][3]), float(rows[1][3]), float(rows[2][3])
    if isinstance(value, list) and len(value) == 16:
        return float(value[3]), float(value[7]), float(value[11])
    raise ValueError("Unsupported transform matrix format.")


def _infer_azimuth_plane_from_positions(positions: Sequence[Tuple[float, float, float]]) -> AzimuthPlane:
    if len(positions) < 2:
        return "xy"
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    def _std(values: Sequence[float]) -> float:
        mean = float(sum(values) / len(values))
        return math.sqrt(float(sum((v - mean) ** 2 for v in values) / len(values)))

    stds = [_std(xs), _std(ys), _std(zs)]
    up_axis = int(min(range(3), key=lambda i: stds[i]))
    if up_axis == 0:
        return "yz"
    if up_axis == 1:
        return "xz"
    return "xy"


def _azimuth_deg_from_position(xyz: Tuple[float, float, float], plane: AzimuthPlane) -> float:
    x, y, z = xyz
    if plane == "xy":
        return math.degrees(math.atan2(y, x))
    if plane == "xz":
        return math.degrees(math.atan2(x, z))
    if plane == "yz":
        return math.degrees(math.atan2(z, y))
    raise ValueError(f"Unhandled plane={plane}")


def _build_azimuths_by_scene(transforms: Dict[str, Dict]) -> Tuple[Dict[str, AzimuthPlane], Dict[str, List[float]]]:
    plane_by_scene: Dict[str, AzimuthPlane] = {}
    azimuths_by_scene: Dict[str, List[float]] = {}
    for scene, tf in transforms.items():
        frames = tf.get("frames", [])
        positions: List[Tuple[float, float, float]] = []
        positions_by_frame: List[Optional[Tuple[float, float, float]]] = []
        for fr in frames:
            mat = fr.get("transform_matrix")
            if mat is None:
                positions_by_frame.append(None)
                continue
            pos = _position_from_matrix(mat)
            positions.append(pos)
            positions_by_frame.append(pos)
        plane = _infer_azimuth_plane_from_positions(positions)
        plane_by_scene[scene] = plane
        azimuths_by_scene[scene] = [
            _azimuth_deg_from_position(p, plane) if p is not None else 0.0 for p in positions_by_frame
        ]
    return plane_by_scene, azimuths_by_scene


def _build_azimuth_by_path(
    scene_paths: Dict[str, List[str]],
    azimuths_by_scene: Dict[str, List[float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for scene, paths in scene_paths.items():
        az = azimuths_by_scene.get(scene, [])
        for idx, path in enumerate(paths):
            if idx >= len(az):
                break
            out[path] = float(az[idx])
    return out


def _import_prism() -> Tuple[type[torch.nn.Module], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    import huggingface_hub

    if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
        setattr(huggingface_hub, "cached_download", huggingface_hub.hf_hub_download)

    from prism.embedders import PRISMEmbedder  
    from prism.mmd_distance import compute_mmd  

    return PRISMEmbedder, compute_mmd


@dataclass(frozen=True)
class _PrismView:
    path: Path
    azimuth_deg: float


@dataclass(frozen=True)
class _PrismPair:
    src: _PrismView
    tgt: _PrismView
    azimuth_diff_deg: float


def _load_prism_image_01(path: Path, *, size: int) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms.functional import resize

    img = Image.open(path).convert("RGB")
    out = T.ToTensor()(img)
    if int(size) > 0:
        out = resize(out, [int(size), int(size)])
    return out


@torch.no_grad()
def _embed_prism_pairs_from_paths(
    embedder: torch.nn.Module,
    pairs: Sequence[_PrismPair],
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
) -> torch.Tensor:
    if not pairs:
        raise ValueError("Need at least 1 PRISM pair.")
    batch_size = max(1, int(batch_size))

    out: List[torch.Tensor] = []
    cache: Dict[Path, torch.Tensor] = {}
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        src_imgs: List[torch.Tensor] = []
        tgt_imgs: List[torch.Tensor] = []
        for p in chunk:
            if p.src.path not in cache:
                cache[p.src.path] = _load_prism_image_01(p.src.path, size=int(image_size))
            if p.tgt.path not in cache:
                cache[p.tgt.path] = _load_prism_image_01(p.tgt.path, size=int(image_size))
            src_imgs.append(cache[p.src.path])
            tgt_imgs.append(cache[p.tgt.path])
        src = torch.stack(src_imgs, dim=0).to(device)
        tgt = torch.stack(tgt_imgs, dim=0).to(device)
        deltas = [float(p.azimuth_diff_deg) for p in chunk]
        emb = embedder((src, tgt, deltas))
        out.append(emb.detach().to("cpu"))
    return torch.cat(out, dim=0)


@torch.no_grad()
def _embed_prism_pairs_from_images(
    embedder: torch.nn.Module,
    images_01: torch.Tensor,
    pairs: Sequence[Tuple[int, int, float]],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if not pairs:
        raise ValueError("Need at least 1 PRISM pair.")
    batch_size = max(1, int(batch_size))

    out: List[torch.Tensor] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        src_idx = [int(p[0]) for p in chunk]
        tgt_idx = [int(p[1]) for p in chunk]
        deltas = [float(p[2]) for p in chunk]
        src = images_01[src_idx].to(device)
        tgt = images_01[tgt_idx].to(device)
        emb = embedder((src, tgt, deltas))
        out.append(emb.detach().to("cpu"))
    return torch.cat(out, dim=0)


def _build_prism_reference_embeddings_by_k(
    *,
    consistent_samples: Sequence[MultiViewSample],
    subset_sizes: Sequence[int],
    scene_paths: Dict[str, List[str]],
    azimuth_by_path: Dict[str, float],
    azimuths_by_scene: Dict[str, List[float]],
    ref_pool_size: int,
    embedder: torch.nn.Module,
    device: torch.device,
    image_size: int,
    embed_batch_size: int,
    seed: int,
    logger: logging.Logger,
) -> Dict[int, torch.Tensor]:
    ref_pool_size = int(ref_pool_size)
    if ref_pool_size < 2:
        raise ValueError(f"--prism-ref-pool-size must be >= 2, got {ref_pool_size}")

    ref_by_k: Dict[int, torch.Tensor] = {}
    consistent_by_k: Dict[int, List[MultiViewSample]] = {}
    for s in consistent_samples:
        if s.scene is None or not s.image_paths:
            continue
        consistent_by_k.setdefault(int(s.subset_size), []).append(s)

    for k in subset_sizes:
        samples_k = consistent_by_k.get(int(k), [])
        if not samples_k:
            continue

        rng = random.Random(int(seed) + 1009 * int(k))
        prism_pairs: List[_PrismPair] = []
        while len(prism_pairs) < ref_pool_size:
            sample = rng.choice(samples_k)
            scene = sample.scene
            assert scene is not None

            src_views = [
                _PrismView(Path(p), float(azimuth_by_path[p])) for p in (sample.image_paths or []) if p in azimuth_by_path
            ]
            if len(src_views) < 1:
                continue

            deltas: List[float] = []
            for i in range(len(src_views)):
                for j in range(len(src_views)):
                    if i == j:
                        continue
                    deltas.append(_wrap_degrees(src_views[j].azimuth_deg - src_views[i].azimuth_deg))
            if not deltas:
                continue

            target_paths = scene_paths.get(scene, [])
            target_az = azimuths_by_scene.get(scene, [])
            if len(target_paths) < 2 or len(target_az) < 2:
                continue
            tgt_views = [
                _PrismView(Path(p), float(target_az[idx]))
                for idx, p in enumerate(target_paths)
                if idx < len(target_az)
            ]

            delta = float(rng.choice(deltas))
            src = src_views[rng.randrange(len(src_views))]
            desired = _wrap_degrees(src.azimuth_deg + delta)

            best: Optional[_PrismView] = None
            best_dist = float("inf")
            for v in tgt_views:
                if v.path == src.path:
                    continue
                dist = abs(_wrap_degrees(v.azimuth_deg - desired))
                if dist < best_dist:
                    best_dist = dist
                    best = v
            if best is None:
                continue
            tgt = best
            actual = _wrap_degrees(tgt.azimuth_deg - src.azimuth_deg)
            prism_pairs.append(_PrismPair(src=src, tgt=tgt, azimuth_diff_deg=actual))

        logger.info("Building PRISM ref pool: K=%d pairs=%d", int(k), len(prism_pairs))
        emb = _embed_prism_pairs_from_paths(
            embedder,
            prism_pairs,
            device=device,
            image_size=int(image_size),
            batch_size=int(embed_batch_size),
        )
        ref_by_k[int(k)] = emb

    if not ref_by_k:
        raise ValueError("No PRISM reference embeddings built (no consistent samples found).")
    return ref_by_k


def _write_prism_reference_embeddings(path: Path, ref_by_k: Dict[int, torch.Tensor], *, ref_pool_size: int, seed: int) -> None:
    payload = {
        "ref_pool_size": int(ref_pool_size),
        "seed": int(seed),
        "embeddings_by_k": {str(k): v.to(dtype=torch.float16, device="cpu") for k, v in sorted(ref_by_k.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _load_prism_reference_embeddings(path: Path) -> Dict[int, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu")
    ref_raw = obj.get("embeddings_by_k", {})
    out: Dict[int, torch.Tensor] = {}
    for k_str, value in ref_raw.items():
        out[int(k_str)] = torch.as_tensor(value).to(dtype=torch.float32, device="cpu")
    return out


def _evaluate_prism_mmd(
    embedder: torch.nn.Module,
    compute_mmd: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sample: MultiViewSample,
    *,
    reference_embeddings: torch.Tensor,
    azimuth_by_path: Optional[Dict[str, float]],
    num_pairs: int,
    embed_batch_size: int,
    image_size: int,
    seed: int,
    device: torch.device,
) -> float:
    if reference_embeddings.shape[0] < 2:
        raise ValueError("PRISM reference embeddings must contain at least 2 samples.")

    K = int(sample.subset_size)
    if K < 2:
        return 0.0

    if sample.image_paths is None:
        seed_noise, noise_type, sigma = _noise_params(sample)
        images_01 = _generate_noise_images_01(
            num_images=K,
            size=int(image_size),
            seed=seed_noise,
            device=torch.device("cpu"),
            noise_type=noise_type,
            gaussian_sigma=float(sigma),
        )
        azimuths = [_wrap_degrees(-180.0 + 360.0 * i / float(K)) for i in range(K)]
    else:
        images_01 = torch.stack(
            [_load_prism_image_01(Path(p), size=int(image_size)) for p in sample.image_paths], dim=0
        )
        images_01 = _maybe_patch_images_01(images_01, sample)
        if azimuth_by_path is None:
            azimuths = [_wrap_degrees(-180.0 + 360.0 * i / float(K)) for i in range(K)]
        else:
            azimuths = [float(azimuth_by_path.get(p, 0.0)) for p in sample.image_paths]

    all_pairs: List[Tuple[int, int, float]] = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            all_pairs.append((i, j, _wrap_degrees(float(azimuths[j]) - float(azimuths[i]))))

    if len(all_pairs) < 2:
        raise ValueError(f"Need at least 2 PRISM pairs, got {len(all_pairs)} (K={K}).")

    num_pairs = int(num_pairs)
    if num_pairs > 0 and len(all_pairs) > num_pairs:
        rng = random.Random(int(seed) + _stable_int_hash(sample.sample_id))
        pairs = rng.sample(all_pairs, k=int(num_pairs))
    else:
        pairs = all_pairs
    if len(pairs) < 2:
        raise ValueError(f"--prism-num-pairs must be >= 2, got {len(pairs)} for K={K}.")

    gen_emb = _embed_prism_pairs_from_images(
        embedder,
        images_01,
        pairs,
        device=device,
        batch_size=int(embed_batch_size),
    )
    score = compute_mmd(gen_emb.to(device), reference_embeddings.to(device)).item()
    return float(score)


def _select_uniform_indices(num_total: int, num_select: int) -> List[int]:
    if num_total <= 0:
        return []
    if num_select <= 0 or num_select >= num_total:
        return list(range(num_total))
    if num_select == 1:
        return [0]
    raw = [int(round(i * (num_total - 1) / (num_select - 1))) for i in range(num_select)]
    out: List[int] = []
    used: set[int] = set()
    for idx in raw:
        idx = max(0, min(num_total - 1, idx))
        if idx not in used:
            out.append(idx)
            used.add(idx)
    if len(out) < num_select:
        for idx in range(num_total):
            if idx in used:
                continue
            out.append(idx)
            used.add(idx)
            if len(out) >= num_select:
                break
    return out


def _import_cv2():
    import cv2

    return cv2


def _symmetric_epipolar_distance(F, p1, p2) -> float:
    import numpy as np

    x1 = np.array([float(p1[0]), float(p1[1]), 1.0], dtype=np.float64)
    x2 = np.array([float(p2[0]), float(p2[1]), 1.0], dtype=np.float64)
    l2 = F @ x1
    l1 = F.T @ x2
    d2 = abs(float(x2.T @ l2)) / max(1e-12, float(np.hypot(l2[0], l2[1])))
    d1 = abs(float(x1.T @ l1)) / max(1e-12, float(np.hypot(l1[0], l1[1])))
    return 0.5 * (d1 + d2)


def _evaluate_sed_and_tsed(
    sample: MultiViewSample,
    *,
    azimuth_by_path: Optional[Dict[str, float]],
    noise_size: int,
    t_e: float,
    t_m: int,
    pair_step: int,
    max_frames: int,
    ratio_test: float,
    sift_nfeatures: int,
    max_dim: int,
) -> Tuple[float, float]:
    import numpy as np

    cv2 = _import_cv2()
    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create(nfeatures=int(sift_nfeatures))
    else:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(sift_nfeatures))

    K = int(sample.subset_size)
    if K < 2:
        return 0.0, 1.0

    images_gray: List[np.ndarray] = []
    if sample.image_paths is None:
        seed_noise, noise_type, sigma = _noise_params(sample)
        imgs_01 = _generate_noise_images_01(
            num_images=K,
            size=int(noise_size),
            seed=seed_noise,
            device=torch.device("cpu"),
            noise_type=noise_type,
            gaussian_sigma=float(sigma),
        )
        rgb_u8 = (imgs_01.mul(255.0).clamp(0.0, 255.0)).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        for i in range(K):
            images_gray.append(cv2.cvtColor(rgb_u8[i], cv2.COLOR_RGB2GRAY))
    elif sample.kind == "patched_gaussian" and not sample.materialized:
        patch_seed, patch_ratio, patch_num_patches, patch_sigma = _patch_params(sample)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(patch_seed))
        for p in (sample.image_paths or []):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(f"Failed to read image: {p}")
            im_01 = torch.from_numpy(im.astype(np.float32) / 255.0)
            h, w = int(im_01.shape[0]), int(im_01.shape[1])
            patch_h = max(1, int(h * float(patch_ratio)))
            patch_w = max(1, int(w * float(patch_ratio)))
            patched = im_01.clone()
            for _ in range(int(patch_num_patches)):
                y = int(torch.randint(0, h - patch_h + 1, (1,), generator=g).item())
                x = int(torch.randint(0, w - patch_w + 1, (1,), generator=g).item())
                patch = torch.randn(patch_h, patch_w, generator=g) * float(patch_sigma) + 0.5
                patch = patch.clamp(0.0, 1.0)
                patched[y : y + patch_h, x : x + patch_w] = patch
            images_gray.append((patched.mul(255.0).clamp(0.0, 255.0)).to(torch.uint8).numpy())
    else:
        for p in (sample.image_paths or []):
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(f"Failed to read image: {p}")
            images_gray.append(im)

    if len(images_gray) < 2:
        return float(t_e), 1.0

    if azimuth_by_path is not None and sample.image_paths is not None:
        scored: List[Tuple[float, int]] = []
        for idx, p in enumerate(sample.image_paths):
            scored.append((float(azimuth_by_path.get(p, 0.0)), idx))
        order = [idx for _, idx in sorted(scored, key=lambda t: t[0])]
        images_gray = [images_gray[i] for i in order]

    if int(max_frames) > 0 and len(images_gray) > int(max_frames):
        idxs = _select_uniform_indices(len(images_gray), int(max_frames))
        images_gray = [images_gray[i] for i in idxs]

    if int(max_dim) > 0:
        resized: List[np.ndarray] = []
        for im in images_gray:
            h, w = im.shape[:2]
            scale = float(max(h, w)) / float(max_dim)
            if scale <= 1.0:
                resized.append(im)
                continue
            new_w = max(1, int(round(w / scale)))
            new_h = max(1, int(round(h / scale)))
            resized.append(cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA))
        images_gray = resized

    h0 = int(images_gray[0].shape[0])
    t_e_px = float(t_e) * (float(h0) / 256.0)

    keypoints: List[Sequence[object]] = []
    descriptors: List[Optional[np.ndarray]] = []
    for im in images_gray:
        kp, desc = sift.detectAndCompute(im, None)
        keypoints.append(kp)
        descriptors.append(desc)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    per_pair_medians: List[float] = []
    total_pairs = 0
    for i in range(0, len(images_gray) - int(pair_step)):
        j = i + int(pair_step)
        total_pairs += 1
        desc1, desc2 = descriptors[i], descriptors[j]
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            per_pair_medians.append(float("nan"))
            continue

        knn = matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            a, b = pair
            if float(a.distance) < float(ratio_test) * float(b.distance):
                good.append(a)
        if len(good) < int(t_m):
            per_pair_medians.append(float("nan"))
            continue

        pts1 = np.array([keypoints[i][m.queryIdx].pt for m in good], dtype=np.float64)
        pts2 = np.array([keypoints[j][m.trainIdx].pt for m in good], dtype=np.float64)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99, maxIters=5000)
        if F is None or mask is None:
            per_pair_medians.append(float("nan"))
            continue

        mask = mask.reshape(-1).astype(bool)
        if int(mask.sum()) < int(t_m):
            per_pair_medians.append(float("nan"))
            continue

        seds: List[float] = []
        for a, b in zip(pts1[mask], pts2[mask]):
            seds.append(_symmetric_epipolar_distance(F, a, b))
        if len(seds) < int(t_m):
            per_pair_medians.append(float("nan"))
            continue
        per_pair_medians.append(float(np.median(seds)))

    finite = [float(v) for v in per_pair_medians if math.isfinite(v)]
    sed = float(np.mean(finite)) if finite else float(t_e_px * 10.0)
    n_consistent = sum(1 for v in per_pair_medians if math.isfinite(v) and float(v) < t_e_px)
    tsed_consistency = float(n_consistent) / float(total_pairs) if total_pairs > 0 else 0.0
    return sed, 1.0 - tsed_consistency


def _load_mmd_sigma_by_k(path: Path) -> MmdSigmaByMetric:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: MmdSigmaByMetric = {}
    for metric, per_k in data.items():
        out[str(metric)] = {int(k): float(v) for k, v in per_k.items()}
    return out


def _write_mmd_sigma_by_k(path: Path, sigmas: MmdSigmaByMetric) -> None:
    payload = {m: {str(k): float(v) for k, v in sorted(per_k.items())} for m, per_k in sorted(sigmas.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _get_overlap_stats(metric: object) -> dict[str, str]:
    """Extract overlap stats from metric._last_overlap_stats (set during forward)."""
    stats = getattr(metric, "_last_overlap_stats", None)
    if stats is None:
        return {
            "n_pairs": "",
            "n_zero_overlap_pairs": "",
            "n_points_sampled": "",
            "n_points_multi_view": "",
            "n_retained_views": "",
            "n_rejected_views": "",
            "rejected_view_fraction": "",
        }
    n_zero = stats.get("n_zero_overlap", "")
    if isinstance(n_zero, int) and n_zero < 0:
        n_zero = ""
    return {
        "n_pairs": str(stats.get("n_pairs", "")),
        "n_zero_overlap_pairs": str(n_zero),
        "n_points_sampled": str(stats.get("n_points_sampled", "")),
        "n_points_multi_view": str(stats.get("n_points_multi_view", "")),
        "n_retained_views": str(stats.get("n_retained_views", "")),
        "n_rejected_views": str(stats.get("n_rejected_views", "")),
        "rejected_view_fraction": str(stats.get("rejected_view_fraction", "")),
    }


def _evaluate_met3r(
    metric: MEt3R,
    sample: MultiViewSample,
    *,
    img_size: int,
    batch_pairs: int,
    device: torch.device,
) -> float:
    """
    Compute the mean pairwise MEt3R score over all image pairs in the sample.

    Sets ``metric._last_overlap_stats`` so the caller can emit overlap
    statistics even though the upstream MEt3R class does not track them.
    """
    imgs = _load_images_met3r_signed(sample, img_size=img_size, device=device)

    num_imgs = imgs.shape[0]
    if num_imgs < 2:
        metric._last_overlap_stats = {"n_pairs": 0, "n_zero_overlap": 0}
        return float("nan")

    pairs = list(itertools.combinations(range(num_imgs), 2))
    scores = torch.empty(len(pairs), device=device)
    n_zero_overlap = 0

    metric = metric.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_pairs):
            end = min(start + batch_pairs, len(pairs))
            idx_batch = pairs[start:end]
            batch = torch.stack([torch.stack((imgs[i], imgs[j])) for i, j in idx_batch], dim=0)
            outputs = metric(
                batch,
                return_overlap_mask=True,
                return_score_map=False,
                return_projections=False,
            )
            score_tensor = outputs[0]
            overlap_mask = outputs[1]  # (B, H, W)
            scores[start:end] = score_tensor.to(device).flatten()
            # Count pairs where the overlap mask is entirely zero.
            for b in range(overlap_mask.shape[0]):
                if overlap_mask[b].sum() == 0:
                    n_zero_overlap += 1

    metric._last_overlap_stats = {"n_pairs": len(pairs), "n_zero_overlap": n_zero_overlap}
    return float(scores.mean().item())


def _evaluate_met3r_mmd(
    metric: MEt3R_MMD,
    sample: MultiViewSample,
    *,
    img_size: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
    reference_sigma: Optional[float] = None,
    return_sigma: bool = False,
) -> tuple[float, Optional[float]]:
    """
    Compute the MEt3R-MMD score for the sample treated as a single multiview set.
    """
    imgs = _load_images_met3r_signed(sample, img_size=img_size, device=device)

    if imgs.shape[0] < 2:
        return float("nan"), None

    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples

    with torch.no_grad():
        outputs = metric(
            imgs.unsqueeze(0),
            return_sigma=return_sigma,
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
            reference_sigma=reference_sigma,
        )

    if isinstance(outputs, (tuple, list)):
        score_tensor = outputs[0]
        sigma_tensor = outputs[1] if return_sigma and len(outputs) > 1 else None
    else:
        score_tensor = outputs
        sigma_tensor = None
    score = float(score_tensor.squeeze().item())
    sigma_val = float(sigma_tensor.squeeze().item()) if sigma_tensor is not None else None
    return score, sigma_val


def _evaluate_met3r_energy(
    metric: MEt3R_Energy,
    sample: MultiViewSample,
    *,
    img_size: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
) -> float:
    """
    Compute the MEt3R-Energy score for the sample treated as a single multiview set.
    """
    imgs = _load_images_met3r_signed(sample, img_size=img_size, device=device)

    if imgs.shape[0] < 2:
        return float("nan")

    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples

    with torch.no_grad():
        outputs = metric(
            imgs.unsqueeze(0),
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
        )

    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_fast3r(
    metric: MEt3R_Fast3R,
    sample: MultiViewSample,
    *,
    resize: int,
    device: torch.device,
) -> float:
    """
    Compute the Fast3R-based multiview MEt3R score.
    """
    if sample.subset_size < 2:
        return float("nan")
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_01)
    # First element of the returned tuple is the batch score.
    if isinstance(outputs, (tuple, list)):
        score_tensor = outputs[0]
    else:
        score_tensor = outputs
    return float(score_tensor.mean().item())


def _evaluate_fast3r_mmd(
    metric: MEt3R_Fast3R_MMD,
    sample: MultiViewSample,
    *,
    resize: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
    reference_sigma: Optional[float] = None,
    return_sigma: bool = False,
) -> tuple[float, Optional[float]]:
    """
    Compute the Fast3R-based MMD score for the sample treated as a single multiview set.
    """
    if sample.subset_size < 2:
        return float("nan"), None
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples

    with torch.no_grad():
        outputs = metric(
            images_01,
            return_sigma=return_sigma,
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
            reference_sigma=reference_sigma,
        )

    if isinstance(outputs, (tuple, list)):
        score_tensor = outputs[0]
        sigma_tensor = outputs[1] if return_sigma and len(outputs) > 1 else None
    else:
        score_tensor = outputs
        sigma_tensor = None
    score = float(score_tensor.squeeze().item())
    sigma_val = float(sigma_tensor.squeeze().item()) if sigma_tensor is not None else None
    return score, sigma_val


def _evaluate_fast3r_energy(
    metric: MEt3R_Fast3R_Energy,
    sample: MultiViewSample,
    *,
    resize: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
) -> float:
    """
    Compute the Fast3R-based Energy distance score for the sample treated as a single multiview set.
    """
    if sample.subset_size < 2:
        return float("nan")
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples
    with torch.no_grad():
        outputs = metric(
            images_01,
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
        )
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_fast3r_pc(
    metric: MEt3R_Fast3R_PointConsistency,
    sample: MultiViewSample,
    *,
    resize: int,
    device: torch.device,
) -> float:
    """
    Compute the Fast3R point-consistency score (mean per-point dispersion).
    """
    if sample.subset_size < 2:
        return float("nan")
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_01)
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_fast3r_pc_mmd(
    metric: MEt3R_Fast3R_PointConsistency_MMD,
    sample: MultiViewSample,
    *,
    resize: int,
    device: torch.device,
    reference_sigma: Optional[float] = None,
    return_sigma: bool = False,
) -> tuple[float, Optional[float]]:
    """
    Compute the Fast3R point-consistency MMD² score over per-point dispersions.
    """
    if sample.subset_size < 2:
        return float("nan"), None
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_01, return_sigma=return_sigma, reference_sigma=reference_sigma)
    if isinstance(outputs, (tuple, list)):
        score_tensor = outputs[0]
        sigma_tensor = outputs[1] if return_sigma and len(outputs) > 1 else None
    else:
        score_tensor = outputs
        sigma_tensor = None
    score = float(score_tensor.squeeze().item())
    sigma_val = float(sigma_tensor.squeeze().item()) if sigma_tensor is not None else None
    return score, sigma_val


def _evaluate_fast3r_pc_energy(
    metric: MEt3R_Fast3R_PointConsistency_Energy,
    sample: MultiViewSample,
    *,
    resize: int,
    device: torch.device,
) -> float:
    """
    Compute the Fast3R point-consistency Energy distance score over per-point dispersions.
    """
    if sample.subset_size < 2:
        return float("nan")
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_01)
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_vggt(
    metric: MEt3R_VGGT,
    sample: MultiViewSample,
    *,
    image_size: int,
    device: torch.device,
) -> float:
    """
    Compute the VGGT-based multiview MEt3R score.
    """
    if sample.subset_size < 2:
        return float("nan")
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(
            images_batch,
            return_overlap_mask=False,
            return_score_map=False,
            return_projections=False,
        )

    if isinstance(outputs, (tuple, list)):
        score_tensor = outputs[0]
    else:
        score_tensor = outputs
    return float(score_tensor.mean().item())


def _evaluate_vggt_mmd(
    metric: MEt3R_VGGT_MMD,
    sample: MultiViewSample,
    *,
    image_size: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
    reference_sigma: Optional[float] = None,
    return_sigma: bool = False,
) -> tuple[float, Optional[float]]:
    """
    Compute the VGGT-based MMD score for the sample treated as a single multiview set.
    """
    if sample.subset_size < 2:
        return float("nan"), None
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples

    with torch.no_grad():
        outputs = metric(
            images_batch,
            return_sigma=return_sigma,
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
            reference_sigma=reference_sigma,
        )

    if isinstance(outputs, (tuple, list)):
        score_tensor = outputs[0]
        sigma_tensor = outputs[1] if return_sigma and len(outputs) > 1 else None
    else:
        score_tensor = outputs
        sigma_tensor = None
    score = float(score_tensor.squeeze().item())
    sigma_val = float(sigma_tensor.squeeze().item()) if sigma_tensor is not None else None
    return score, sigma_val


def _evaluate_vggt_energy(
    metric: MEt3R_VGGT_Energy,
    sample: MultiViewSample,
    *,
    image_size: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
) -> float:
    """
    Compute the VGGT-based Energy distance score for the sample treated as a single multiview set.
    """
    if sample.subset_size < 2:
        return float("nan")
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples

    with torch.no_grad():
        outputs = metric(
            images_batch,
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
        )

    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_vggt_pc(
    metric: MEt3R_VGGT_PointConsistency,
    sample: MultiViewSample,
    *,
    image_size: int,
    device: torch.device,
) -> float:
    """
    Compute the VGGT point-consistency score (mean per-point dispersion).
    """
    if sample.subset_size < 2:
        return float("nan")
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_batch)
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_vggt_pc_mmd(
    metric: MEt3R_VGGT_PointConsistency_MMD,
    sample: MultiViewSample,
    *,
    image_size: int,
    device: torch.device,
    reference_sigma: Optional[float] = None,
    return_sigma: bool = False,
) -> tuple[float, Optional[float]]:
    """
    Compute the VGGT point-consistency MMD² score over per-point dispersions.
    """
    if sample.subset_size < 2:
        return float("nan"), None
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_batch, return_sigma=return_sigma, reference_sigma=reference_sigma)
    if isinstance(outputs, (tuple, list)):
        score_tensor = outputs[0]
        sigma_tensor = outputs[1] if return_sigma and len(outputs) > 1 else None
    else:
        score_tensor = outputs
        sigma_tensor = None
    score = float(score_tensor.squeeze().item())
    sigma_val = float(sigma_tensor.squeeze().item()) if sigma_tensor is not None else None
    return score, sigma_val


def _evaluate_vggt_pc_energy(
    metric: MEt3R_VGGT_PointConsistency_Energy,
    sample: MultiViewSample,
    *,
    image_size: int,
    device: torch.device,
) -> float:
    """
    Compute the VGGT point-consistency Energy distance score over per-point dispersions.
    """
    if sample.subset_size < 2:
        return float("nan")
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_batch)
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_met3r_imq(
    metric: MEt3R_IMQ,
    sample: MultiViewSample,
    *,
    img_size: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
) -> float:
    """
    Compute the MEt3R-IMQ score for the sample treated as a single multiview set.
    """
    imgs = _load_images_met3r_signed(sample, img_size=img_size, device=device)

    if imgs.shape[0] < 2:
        return float("nan")

    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples

    with torch.no_grad():
        outputs = metric(
            imgs.unsqueeze(0),
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
        )

    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_fast3r_imq(
    metric: MEt3R_Fast3R_IMQ,
    sample: MultiViewSample,
    *,
    resize: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
) -> float:
    """
    Compute the Fast3R-based IMQ kernel MMD score for the sample.
    """
    if sample.subset_size < 2:
        return float("nan")
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples
    with torch.no_grad():
        outputs = metric(
            images_01,
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
        )
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_fast3r_pc_imq(
    metric: MEt3R_Fast3R_PointConsistency_IMQ,
    sample: MultiViewSample,
    *,
    resize: int,
    device: torch.device,
) -> float:
    """
    Compute the Fast3R point-consistency IMQ kernel MMD score.
    """
    if sample.subset_size < 2:
        return float("nan")
    images_01 = _load_images_fast3r_01_batch(sample, resize=resize, device=device)

    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_01)
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_vggt_imq(
    metric: MEt3R_VGGT_IMQ,
    sample: MultiViewSample,
    *,
    image_size: int,
    per_pair_max_samples: int,
    max_total_samples: int,
    device: torch.device,
) -> float:
    """
    Compute the VGGT-based IMQ kernel MMD score for the sample.
    """
    if sample.subset_size < 2:
        return float("nan")
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    per_pair_max = None if per_pair_max_samples <= 0 else per_pair_max_samples
    max_total = None if max_total_samples <= 0 else max_total_samples

    with torch.no_grad():
        outputs = metric(
            images_batch,
            per_pair_max_samples=per_pair_max,
            max_total_samples=max_total,
        )

    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _evaluate_vggt_pc_imq(
    metric: MEt3R_VGGT_PointConsistency_IMQ,
    sample: MultiViewSample,
    *,
    image_size: int,
    device: torch.device,
) -> float:
    """
    Compute the VGGT point-consistency IMQ kernel MMD score.
    """
    if sample.subset_size < 2:
        return float("nan")
    images = _load_images_vggt_01(sample, image_size=image_size, device=device)

    images_batch = images.unsqueeze(0)
    metric = metric.to(device).eval()
    with torch.no_grad():
        outputs = metric(images_batch)
    score_tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return float(score_tensor.squeeze().item())


def _select_metrics(raw: Sequence[str]) -> List[MetricName]:
    requested = [m.lower() for m in raw]
    base_metrics: List[MetricName] = [
        "met3r",
        "met3r_mmd",
        "met3r_energy",
        "met3r_imq",
        "met3r_dust3r",
        "met3r_dust3r_mmd",
        "met3r_dust3r_energy",
        "met3r_dust3r_imq",
        "fast3r",
        "fast3r_mmd",
        "fast3r_energy",
        "fast3r_imq",
        "fast3r_pc",
        "fast3r_pc_mmd",
        "fast3r_pc_energy",
        "fast3r_pc_imq",
        "vggt",
        "vggt_robust",
        "vggt_mmd",
        "vggt_energy",
        "vggt_imq",
        "vggt_pc",
        "vggt_pc_mmd",
        "vggt_pc_energy",
        "vggt_pc_imq",
    ]
    extra_metrics: List[MetricName] = [
        "prism_mmd",
        "sed",
        "tsed",
    ]
    if "all" in requested:
        out = list(base_metrics)
        for m in requested:
            if m in extra_metrics and m not in out:
                out.append(m)
        return out
    out: List[MetricName] = []
    for m in requested:
        if m in base_metrics and m not in out:
            out.append(m)
        if m in extra_metrics and m not in out:
            out.append(m)
    if not out:
        raise ValueError(f"No valid metrics selected from {raw}")
    return out


def _patch_torch_hub_load_to_pin_cached_refs(logger: logging.Logger) -> None:
    """
    Torch Hub probes GitHub to determine the default branch when no ref is given
    (e.g. `facebookresearch/dinov2` vs `facebookresearch/dinov2:main`). On Slurm
    clusters this can fail intermittently (502, RemoteDisconnected), crashing
    jobs even when the repo is already present in the local cache.

    If a cached copy exists for `main` or `master`, rewrite `owner/repo` to
    `owner/repo:<ref>` so torch.hub can load without any network access.
    """
    if getattr(torch.hub.load, "_met3r_cached_ref_patched", False):
        return

    original_load = torch.hub.load

    def load_patched(repo_or_dir: Any, model: str, *args: Any, **kwargs: Any) -> Any:
        source = str(kwargs.get("source", "github")).lower()
        if (
            source == "github"
            and isinstance(repo_or_dir, str)
            and ":" not in repo_or_dir
            and repo_or_dir.count("/") == 1
            and not Path(repo_or_dir).expanduser().exists()
        ):
            owner, repo = repo_or_dir.split("/", 1)
            hub_dir = Path(torch.hub.get_dir())
            for ref in ("main", "master"):
                if (hub_dir / f"{owner}_{repo}_{ref}").exists():
                    repo_or_dir = f"{owner}/{repo}:{ref}"
                    break
        return original_load(repo_or_dir, model, *args, **kwargs)

    setattr(load_patched, "_met3r_cached_ref_patched", True)
    torch.hub.load = load_patched
    logger.info("Pinned cached torch.hub repos to cached ref (avoids GitHub lookups).")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    _patch_torch_hub_load_to_pin_cached_refs(logger)
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MEt3R / MEt3R-MMD / Fast3R-based / VGGT-based consistency metrics "
            "on Mip-NeRF360 calibration splits and several impossible variants "
            "(mixed, one-outlier, noise, patched)."
        )
    )
    parser.add_argument(
        "--splits-json",
        type=str,
        default="tmp/syscon3d_release/mipnerf360_calibration_splits.json",
        help="Path to the calibration splits JSON produced by mipnerf360_prepare_calibration_splits.py.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Optional override for the dataset root. If omitted, the value from the splits JSON "
            "is used."
        ),
    )
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of view counts (e.g. 3 6 9 12); defaults to all sizes in the splits JSON.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=9,
        help="Maximum number of views per subset to include (default: 9, to avoid VGGT OOM).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["all"],
        help=(
            'Which metrics to run: any of ["met3r", "met3r_mmd", "met3r_energy", "met3r_dust3r", '
            '"met3r_dust3r_mmd", "met3r_dust3r_energy", "fast3r", "fast3r_mmd", "fast3r_energy", '
            '"fast3r_pc", "fast3r_pc_mmd", "fast3r_pc_energy", "vggt", "vggt_robust", "vggt_mmd", "vggt_energy", '
            '"vggt_pc", "vggt_pc_mmd", "vggt_pc_energy", "prism_mmd", "sed", "tsed", "all"].'
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "consistent-only", "impossible-only"],
        help=(
            "Which samples to evaluate: all types, only per-scene consistent splits, "
            "or only impossible (mixed/noise) splits."
        ),
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="If set, restrict evaluation to a single scene name from the splits JSON.",
    )
    parser.add_argument(
        "--only-kinds",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "consistent",
            "mixed",
            "mixed_controlled",
            "mixed_one_outlier",
            "identical_images",
            "consistent_gaussian_epsilon",
            "full_mixture_distinct_scene",
            "noise",
            "noise_gaussian",
            "patched_gaussian",
        ],
        help="If set, evaluate only samples of the given kinds (after loading splits/manifests).",
    )
    parser.add_argument(
        "--only-sample-ids",
        type=str,
        nargs="+",
        default=None,
        help="If set, evaluate only the given sample_id values (after loading splits/manifests).",
    )
    parser.add_argument(
        "--num-mixed-per-size",
        type=int,
        default=8,
        help="Number of cross-scene mixed samples per subset size.",
    )
    parser.add_argument(
        "--num-noise-per-size",
        type=int,
        default=4,
        help="Number of uniform-noise samples per subset size.",
    )
    parser.add_argument(
        "--num-gaussian-noise-per-size",
        type=int,
        default=4,
        help="Number of Gaussian-noise samples per subset size.",
    )
    parser.add_argument(
        "--num-one-outlier-per-size",
        type=int,
        default=8,
        help="Number of one-outlier mixed samples per subset size (K-1 views + 1 external view).",
    )
    parser.add_argument(
        "--num-controlled-per-size",
        type=int,
        default=8,
        help="Number of controlled-mix samples per subset size (~30%% foreign views).",
    )
    parser.add_argument(
        "--num-patched-per-size",
        type=int,
        default=4,
        help="Number of patched samples per subset size (Gaussian-noise patches on consistent views).",
    )
    parser.add_argument(
        "--num-identical-per-size",
        type=int,
        default=4,
        help="Number of identical-image samples per subset size.",
    )
    parser.add_argument(
        "--num-full-distinct-per-size",
        type=int,
        default=8,
        help="Number of fully mixed distinct-scene samples per subset size.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.2,
        help="Sigma for Gaussian noise (used for pure Gaussian noise and patched variants).",
    )
    parser.add_argument(
        "--patch-ratio",
        type=float,
        default=0.25,
        help="Patch side length as a fraction of image size (patched variant).",
    )
    parser.add_argument(
        "--patch-num-patches",
        type=int,
        default=4,
        help="Number of noise patches applied per image (patched variant).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for constructing mixed/noise samples.",
    )
    parser.add_argument(
        "--met3r-img-size",
        type=int,
        default=224,
        help="Long-side resize used for MEt3R / MEt3R-MMD image loading (matches met3r_eval.py).",
    )
    parser.add_argument(
        "--met3r-batch-pairs",
        type=int,
        default=4,
        help="Batch size over image pairs for MEt3R (reduce if you hit GPU OOM).",
    )
    parser.add_argument(
        "--met3r-per-pair-max-samples",
        type=int,
        default=128,
        help="Maximum number of pixels sampled per image pair for MEt3R-MMD (<=0 to disable).",
    )
    parser.add_argument(
        "--met3r-max-total-samples",
        type=int,
        default=4096,
        help="Maximum total pixel samples across all pairs for MEt3R-MMD (<=0 to disable).",
    )
    parser.add_argument(
        "--fast3r-resize",
        type=int,
        default=224,
        help="Long-side resize used for Fast3R image loading.",
    )
    parser.add_argument(
        "--vggt-noise-size",
        "--vggt-img-size",
        dest="vggt_img_size",
        type=int,
        default=518,
        help=(
            "Base spatial size used for VGGT image loading and synthetic noise scenes "
            "(alias: --vggt-noise-size)."
        ),
    )
    parser.add_argument(
        "--impossible-splits-json",
        type=str,
        default="tmp/syscon3d_release/mipnerf360_impossible_splits.json",
        help=(
            "Path to a JSON file describing mixed/noise (impossible) samples. "
            "If the file does not exist, mixed/noise samples are generated and "
            "written there for reuse."
        ),
    )
    parser.add_argument(
        "--dl3dv-root",
        type=str,
        default=DEFAULT_DL3DV_BENCHMARK_ROOT,
        help="DL3DV benchmark root used for full distinct-scene impossible splits when K > 9.",
    )
    parser.add_argument(
        "--no-save-impossible-splits",
        action="store_true",
        help=(
            "If set, do not write a manifest for mixed/noise samples even when "
            "it is missing; impossible samples will be generated on the fly only."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="metric_calibration_results.csv",
        help="CSV file to store per-sample metric scores (one row per sample and metric).",
    )
    parser.add_argument(
        "--mmd-sigma-in-json",
        type=str,
        default=None,
        help=(
            "If set, load per-metric per-K RBF sigma values for *_mmd metrics and pass them as "
            "reference_sigma during evaluation (disables adaptive sigma)."
        ),
    )
    parser.add_argument(
        "--mmd-sigma-out-json",
        type=str,
        default=None,
        help=(
            "If set, estimate per-metric per-K RBF sigma values from consistent samples by "
            "collecting the per-sample median-heuristic sigma returned by *_mmd metrics, then "
            "writing the median sigma per (metric,K) to this JSON file."
        ),
    )
    parser.add_argument(
        "--prism-ref-in-pt",
        type=str,
        default=None,
        help="If set, load PRISM reference embeddings per K from this torch .pt file.",
    )
    parser.add_argument(
        "--prism-ref-out-pt",
        type=str,
        default=None,
        help=(
            "If set, build PRISM reference embeddings per K from consistent samples and write "
            "them to this torch .pt file."
        ),
    )
    parser.add_argument(
        "--prism-ref-only",
        action="store_true",
        help="If set, only build PRISM reference embeddings (requires --prism-ref-out-pt) and exit.",
    )
    parser.add_argument(
        "--prism-ref-pool-size",
        type=int,
        default=2048,
        help="Number of PRISM reference-pool pair embeddings to store per K (>=2).",
    )
    parser.add_argument(
        "--prism-num-pairs",
        type=int,
        default=64,
        help="Number of within-sample PRISM pairs to embed per sample (<=0 to use all ordered pairs).",
    )
    parser.add_argument(
        "--prism-embed-batch-size",
        type=int,
        default=4,
        help="Mini-batch size for PRISM embedding (reduce if you hit GPU OOM).",
    )
    parser.add_argument(
        "--prism-repo-id",
        type=str,
        default="saar-st/prism-models",
        help="HuggingFace repo id used by PRISMEmbedder for automatic model download.",
    )
    parser.add_argument(
        "--prism-noise-size",
        "--prism-image-size",
        dest="prism_image_size",
        type=int,
        default=256,
        help="Spatial size used to resize images for PRISM (real + synthetic noise).",
    )
    parser.add_argument("--tsed-t-e", type=float, default=2.0, help="TSED SED threshold at 256px reference height.")
    parser.add_argument("--tsed-t-m", type=int, default=10, help="Minimum SIFT matches per neighbor pair.")
    parser.add_argument("--tsed-pair-step", type=int, default=1, help="Neighbor-pair step size (default: 1).")
    parser.add_argument("--tsed-max-frames", type=int, default=0, help="Optional number of frames to evaluate (0=all).")
    parser.add_argument("--tsed-ratio-test", type=float, default=0.75, help="SIFT Lowe ratio test threshold.")
    parser.add_argument("--tsed-sift-nfeatures", type=int, default=4096, help="SIFT nfeatures.")
    parser.add_argument(
        "--tsed-max-dim",
        type=int,
        default=0,
        help="Optional max image dimension for SIFT (0=use original resolution).",
    )
    parser.add_argument(
        "--free-gpu-between-metrics",
        action="store_true",
        help=(
            "If set (or when evaluating a single sample), release GPU memory after each metric by "
            "dropping metric modules and emptying the CUDA cache. Useful to avoid VGGT OOM when "
            "evaluating many metrics in one process."
        ),
    )
    args = parser.parse_args()
    if args.mmd_sigma_in_json is not None and args.mmd_sigma_out_json is not None:
        raise ValueError("Use at most one of --mmd-sigma-in-json and --mmd-sigma-out-json.")
    if args.prism_ref_in_pt is not None and args.prism_ref_out_pt is not None:
        raise ValueError("Use at most one of --prism-ref-in-pt and --prism-ref-out-pt.")
    if args.prism_ref_only and args.prism_ref_out_pt is None:
        raise ValueError("--prism-ref-only requires --prism-ref-out-pt.")

    splits_path = Path(args.splits_json).expanduser().resolve()
    dataset_root_from_splits, scenes_splits, subset_sizes_from_splits = _load_calibration_splits(splits_path)

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root is not None
        else dataset_root_from_splits
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    subset_sizes: List[int]
    if args.subset_sizes:
        subset_sizes = sorted({int(k) for k in args.subset_sizes})
    else:
        subset_sizes = subset_sizes_from_splits
    # Enforce a maximum number of views to avoid OOM in heavy backbones like VGGT.
    subset_sizes = [k for k in subset_sizes if k <= max(1, args.max_views)]

    metric_names = _select_metrics(args.metrics)
    rng = random.Random(args.seed)

    scene_names = sorted(scenes_splits.keys())
    if args.scene is not None:
        if args.scene not in scenes_splits:
            raise ValueError(f"Requested scene '{args.scene}' not found in splits JSON.")
        scene_names = [args.scene]
        # Restrict scenes_splits to the selected scene to avoid accidental use.
        scenes_splits = {args.scene: scenes_splits[args.scene]}
    transforms = _load_transforms(dataset_root, scene_names)
    scene_paths = _build_scene_image_paths(dataset_root, transforms)

    consistent_samples: List[MultiViewSample] = []
    impossible_samples: List[MultiViewSample] = []

    if args.mode in ("all", "consistent-only"):
        consistent_samples = _build_consistent_samples(dataset_root, scenes_splits, transforms, subset_sizes)

    if args.mode in ("all", "impossible-only"):
        impossible_splits_path = Path(args.impossible_splits_json).expanduser().resolve()
        dl3dv_root = Path(args.dl3dv_root).expanduser().resolve()
        impossible_samples = _load_or_build_impossible_samples(
            dataset_root=dataset_root,
            scene_paths=scene_paths,
            dl3dv_root=dl3dv_root,
            subset_sizes=subset_sizes,
            num_mixed_per_size=args.num_mixed_per_size,
            num_noise_per_size=args.num_noise_per_size,
            num_gaussian_noise_per_size=args.num_gaussian_noise_per_size,
            num_one_outlier_per_size=args.num_one_outlier_per_size,
            num_controlled_per_size=args.num_controlled_per_size,
            num_identical_per_size=args.num_identical_per_size,
            num_full_distinct_per_size=args.num_full_distinct_per_size,
            num_patched_per_size=args.num_patched_per_size,
            gaussian_sigma=args.gaussian_sigma,
            patch_ratio=args.patch_ratio,
            patch_num_patches=args.patch_num_patches,
            rng=rng,
            impossible_splits_path=impossible_splits_path,
            save_if_missing=not args.no_save_impossible_splits,
        )

    all_samples: List[MultiViewSample] = consistent_samples + impossible_samples

    if args.only_sample_ids:
        requested = {str(s) for s in args.only_sample_ids}
        available = {s.sample_id for s in all_samples}
        missing = requested - available
        if missing:
            raise ValueError(f"Requested sample_ids not found: {sorted(missing)}")
        all_samples = [s for s in all_samples if s.sample_id in requested]

    if args.only_kinds:
        kinds = {str(k) for k in args.only_kinds}
        all_samples = [s for s in all_samples if s.kind in kinds]

    logger.info(
        "Prepared %d samples (mode=%s, scenes=%d, subset_sizes=%s) with metrics=%s",
        len(all_samples),
        args.mode,
        len(scene_names),
        ",".join(str(k) for k in subset_sizes),
        ",".join(metric_names),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free_gpu_between_metrics = bool(args.free_gpu_between_metrics) or len(all_samples) == 1

    needs_azimuth = (
        "prism_mmd" in metric_names
        or "sed" in metric_names
        or "tsed" in metric_names
        or args.prism_ref_out_pt is not None
    )
    azimuth_by_path: Optional[Dict[str, float]] = None
    azimuths_by_scene: Dict[str, List[float]] = {}
    if needs_azimuth:
        _, azimuths_by_scene = _build_azimuths_by_scene(transforms)
        azimuth_by_path = _build_azimuth_by_path(scene_paths, azimuths_by_scene)

    prism_ref_by_k: Optional[Dict[int, torch.Tensor]] = None
    prism_embedder: Optional[torch.nn.Module] = None
    prism_compute_mmd: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    if args.prism_ref_out_pt is not None:
        if not consistent_samples:
            raise ValueError("--prism-ref-out-pt requires consistent samples (mode must include consistent-only).")
        PRISMEmbedder, compute_mmd = _import_prism()
        prism_compute_mmd = compute_mmd
        prism_embedder = PRISMEmbedder(device=str(device), repo_id=str(args.prism_repo_id))
        prism_embedder.eval()
        prism_ref_by_k = _build_prism_reference_embeddings_by_k(
            consistent_samples=consistent_samples,
            subset_sizes=subset_sizes,
            scene_paths=scene_paths,
            azimuth_by_path=azimuth_by_path or {},
            azimuths_by_scene=azimuths_by_scene,
            ref_pool_size=int(args.prism_ref_pool_size),
            embedder=prism_embedder,
            device=device,
            image_size=int(args.prism_image_size),
            embed_batch_size=int(args.prism_embed_batch_size),
            seed=int(args.seed),
            logger=logger,
        )
        out_pt = Path(args.prism_ref_out_pt).expanduser().resolve()
        _write_prism_reference_embeddings(
            out_pt,
            prism_ref_by_k,
            ref_pool_size=int(args.prism_ref_pool_size),
            seed=int(args.seed),
        )
        logger.info("Wrote PRISM reference embeddings: %s", out_pt)
        if args.prism_ref_only:
            return
    elif args.prism_ref_in_pt is not None:
        in_pt = Path(args.prism_ref_in_pt).expanduser().resolve()
        prism_ref_by_k = _load_prism_reference_embeddings(in_pt)
        logger.info("Loaded PRISM reference embeddings: %s", in_pt)

    if "prism_mmd" in metric_names and prism_ref_by_k is None:
        raise ValueError(
            "prism_mmd requested but no PRISM reference embeddings are available. "
            "Pass --prism-ref-in-pt, or run once with --prism-ref-out-pt to build them."
        )

    mmd_sigma_by_metric: Optional[MmdSigmaByMetric] = None
    if args.mmd_sigma_in_json is not None:
        sigma_path = Path(args.mmd_sigma_in_json).expanduser().resolve()
        mmd_sigma_by_metric = _load_mmd_sigma_by_k(sigma_path)

    sigma_observations: Dict[str, Dict[int, List[float]]] = {}

    # Instantiate metrics lazily to avoid unnecessary initialisation costs.
    met3r_metric: Optional[MEt3R] = None
    met3r_mmd_metric: Optional[MEt3R_MMD] = None
    met3r_energy_metric: Optional[MEt3R_Energy] = None
    met3r_imq_metric: Optional[MEt3R_IMQ] = None
    met3r_dust3r_metric: Optional[MEt3R] = None
    met3r_dust3r_mmd_metric: Optional[MEt3R_MMD] = None
    met3r_dust3r_energy_metric: Optional[MEt3R_Energy] = None
    met3r_dust3r_imq_metric: Optional[MEt3R_IMQ] = None
    fast3r_metric: Optional[MEt3R_Fast3R] = None
    fast3r_mmd_metric: Optional[MEt3R_Fast3R_MMD] = None
    fast3r_energy_metric: Optional[MEt3R_Fast3R_Energy] = None
    fast3r_imq_metric: Optional[MEt3R_Fast3R_IMQ] = None
    fast3r_pc_metric: Optional[MEt3R_Fast3R_PointConsistency] = None
    fast3r_pc_mmd_metric: Optional[MEt3R_Fast3R_PointConsistency_MMD] = None
    fast3r_pc_energy_metric: Optional[MEt3R_Fast3R_PointConsistency_Energy] = None
    fast3r_pc_imq_metric: Optional[MEt3R_Fast3R_PointConsistency_IMQ] = None
    vggt_metric: Optional[MEt3R_VGGT] = None
    vggt_robust_metric: Optional[MEt3R_VGGT_Robust] = None
    vggt_mmd_metric: Optional[MEt3R_VGGT_MMD] = None
    vggt_energy_metric: Optional[MEt3R_VGGT_Energy] = None
    vggt_imq_metric: Optional[MEt3R_VGGT_IMQ] = None
    vggt_pc_metric: Optional[MEt3R_VGGT_PointConsistency] = None
    vggt_pc_mmd_metric: Optional[MEt3R_VGGT_PointConsistency_MMD] = None
    vggt_pc_energy_metric: Optional[MEt3R_VGGT_PointConsistency_Energy] = None
    vggt_pc_imq_metric: Optional[MEt3R_VGGT_PointConsistency_IMQ] = None

    # For summary statistics and threshold suggestions.
    summary: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    prism_ref_device_cache: Dict[int, torch.Tensor] = {}
    sed_tsed_cache: Dict[str, Tuple[float, float]] = {}

    fieldnames = [
        "sample_id", "kind", "scene", "subset_size", "metric", "score",
        "n_pairs", "n_zero_overlap_pairs", "n_points_sampled", "n_points_multi_view",
        "n_retained_views", "n_rejected_views", "rejected_view_fraction",
    ]
    out_csv_path = Path(args.out_csv).expanduser()
    with out_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for sample in all_samples:
            for metric_name in metric_names:
                start_time = time.perf_counter()
                want_sigma = bool(args.mmd_sigma_out_json) and sample.kind == "consistent" and mmd_sigma_by_metric is None
                reference_sigma: Optional[float] = None
                if metric_name in MMD_METRICS and mmd_sigma_by_metric is not None:
                    try:
                        reference_sigma = mmd_sigma_by_metric[metric_name][int(sample.subset_size)]
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing sigma for metric={metric_name}, subset_size={sample.subset_size} "
                            f"in {args.mmd_sigma_in_json}."
                        ) from exc
                imq_c: Optional[float] = None
                if metric_name in IMQ_METRICS and mmd_sigma_by_metric is not None:
                    sigma_metric = IMQ_TO_SIGMA_METRIC[str(metric_name)]
                    try:
                        imq_c = mmd_sigma_by_metric[sigma_metric][int(sample.subset_size)]
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing IMQ c for metric={metric_name} (needs sigma for {sigma_metric}), "
                            f"subset_size={sample.subset_size} in {args.mmd_sigma_in_json}."
                        ) from exc
                logger.info(
                    "Evaluating metric=%s, scene=%s, kind=%s, subset_size=%d (sample_id=%s)",
                    metric_name,
                    sample.scene or "",
                    sample.kind,
                    sample.subset_size,
                    sample.sample_id,
                )
                if metric_name == "met3r":
                    if met3r_metric is None:
                        met3r_metric = MEt3R(
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
                    score_val = _evaluate_met3r(
                        met3r_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        batch_pairs=args.met3r_batch_pairs,
                        device=device,
                    )
                elif metric_name == "met3r_mmd":
                    if met3r_mmd_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        met3r_mmd_metric = MEt3R_MMD(
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
                    score_val = _evaluate_met3r_mmd(
                        met3r_mmd_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                        reference_sigma=reference_sigma,
                        return_sigma=want_sigma,
                    )
                    score_val, sigma_used = score_val
                    if want_sigma and sigma_used is not None:
                        sigma_observations.setdefault(metric_name, {}).setdefault(sample.subset_size, []).append(sigma_used)
                elif metric_name == "met3r_energy":
                    if met3r_energy_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        met3r_energy_metric = MEt3R_Energy(
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
                    score_val = _evaluate_met3r_energy(
                        met3r_energy_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "met3r_dust3r":
                    if met3r_dust3r_metric is None:
                        met3r_dust3r_metric = MEt3R(
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
                    score_val = _evaluate_met3r(
                        met3r_dust3r_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        batch_pairs=args.met3r_batch_pairs,
                        device=device,
                    )
                elif metric_name == "met3r_dust3r_mmd":
                    if met3r_dust3r_mmd_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        met3r_dust3r_mmd_metric = MEt3R_MMD(
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
                    score_val = _evaluate_met3r_mmd(
                        met3r_dust3r_mmd_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                        reference_sigma=reference_sigma,
                        return_sigma=want_sigma,
                    )
                    score_val, sigma_used = score_val
                    if want_sigma and sigma_used is not None:
                        sigma_observations.setdefault(metric_name, {}).setdefault(sample.subset_size, []).append(sigma_used)
                elif metric_name == "met3r_dust3r_energy":
                    if met3r_dust3r_energy_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        met3r_dust3r_energy_metric = MEt3R_Energy(
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
                    score_val = _evaluate_met3r_energy(
                        met3r_dust3r_energy_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "fast3r":
                    if fast3r_metric is None:
                        fast3r_metric = MEt3R_Fast3R(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                        )
                    score_val = _evaluate_fast3r(
                        fast3r_metric,
                        sample,
                        resize=args.fast3r_resize,
                        device=device,
                    )
                elif metric_name == "fast3r_mmd":
                    if fast3r_mmd_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        fast3r_mmd_metric = MEt3R_Fast3R_MMD(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                            per_pair_max_samples=per_pair_max,
                            max_total_samples=max_total,
                        )
                    score_val = _evaluate_fast3r_mmd(
                        fast3r_mmd_metric,
                        sample,
                        resize=args.fast3r_resize,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                        reference_sigma=reference_sigma,
                        return_sigma=want_sigma,
                    )
                    score_val, sigma_used = score_val
                    if want_sigma and sigma_used is not None:
                        sigma_observations.setdefault(metric_name, {}).setdefault(sample.subset_size, []).append(sigma_used)
                elif metric_name == "fast3r_energy":
                    if fast3r_energy_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        fast3r_energy_metric = MEt3R_Fast3R_Energy(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                            per_pair_max_samples=per_pair_max,
                            max_total_samples=max_total,
                        )
                    score_val = _evaluate_fast3r_energy(
                        fast3r_energy_metric,
                        sample,
                        resize=args.fast3r_resize,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "fast3r_pc":
                    if fast3r_pc_metric is None:
                        fast3r_pc_metric = MEt3R_Fast3R_PointConsistency(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                        )
                    score_val = _evaluate_fast3r_pc(
                        fast3r_pc_metric,
                        sample,
                        resize=args.fast3r_resize,
                        device=device,
                    )
                elif metric_name == "fast3r_pc_mmd":
                    if fast3r_pc_mmd_metric is None:
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        fast3r_pc_mmd_metric = MEt3R_Fast3R_PointConsistency_MMD(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                            max_total_samples=max_total,
                        )
                    score_val = _evaluate_fast3r_pc_mmd(
                        fast3r_pc_mmd_metric,
                        sample,
                        resize=args.fast3r_resize,
                        device=device,
                        reference_sigma=reference_sigma,
                        return_sigma=want_sigma,
                    )
                    score_val, sigma_used = score_val
                    if want_sigma and sigma_used is not None:
                        sigma_observations.setdefault(metric_name, {}).setdefault(sample.subset_size, []).append(sigma_used)
                elif metric_name == "fast3r_pc_energy":
                    if fast3r_pc_energy_metric is None:
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        fast3r_pc_energy_metric = MEt3R_Fast3R_PointConsistency_Energy(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                            max_total_samples=max_total,
                        )
                    score_val = _evaluate_fast3r_pc_energy(
                        fast3r_pc_energy_metric,
                        sample,
                        resize=args.fast3r_resize,
                        device=device,
                    )
                elif metric_name == "vggt":
                    if vggt_metric is None:
                        vggt_metric = MEt3R_VGGT(
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
                    score_val = _evaluate_vggt(
                        vggt_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        device=device,
                    )
                elif metric_name == "vggt_robust":
                    if vggt_robust_metric is None:
                        vggt_robust_metric = MEt3R_VGGT_Robust(
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
                    score_val = _evaluate_vggt(
                        vggt_robust_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        device=device,
                    )
                elif metric_name == "vggt_mmd":
                    if vggt_mmd_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        vggt_mmd_metric = MEt3R_VGGT_MMD(
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
                    score_val = _evaluate_vggt_mmd(
                        vggt_mmd_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                        reference_sigma=reference_sigma,
                        return_sigma=want_sigma,
                    )
                    score_val, sigma_used = score_val
                    if want_sigma and sigma_used is not None:
                        sigma_observations.setdefault(metric_name, {}).setdefault(sample.subset_size, []).append(sigma_used)
                elif metric_name == "vggt_energy":
                    if vggt_energy_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        vggt_energy_metric = MEt3R_VGGT_Energy(
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
                    score_val = _evaluate_vggt_energy(
                        vggt_energy_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "vggt_pc":
                    if vggt_pc_metric is None:
                        vggt_pc_metric = MEt3R_VGGT_PointConsistency(
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
                    score_val = _evaluate_vggt_pc(
                        vggt_pc_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        device=device,
                    )
                elif metric_name == "vggt_pc_mmd":
                    if vggt_pc_mmd_metric is None:
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        vggt_pc_mmd_metric = MEt3R_VGGT_PointConsistency_MMD(
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
                    score_val = _evaluate_vggt_pc_mmd(
                        vggt_pc_mmd_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        device=device,
                        reference_sigma=reference_sigma,
                        return_sigma=want_sigma,
                    )
                    score_val, sigma_used = score_val
                    if want_sigma and sigma_used is not None:
                        sigma_observations.setdefault(metric_name, {}).setdefault(sample.subset_size, []).append(sigma_used)
                elif metric_name == "vggt_pc_energy":
                    if vggt_pc_energy_metric is None:
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        vggt_pc_energy_metric = MEt3R_VGGT_PointConsistency_Energy(
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
                    score_val = _evaluate_vggt_pc_energy(
                        vggt_pc_energy_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        device=device,
                    )
                elif metric_name == "met3r_imq":
                    if met3r_imq_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        met3r_imq_metric = MEt3R_IMQ(
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
                    if imq_c is not None:
                        met3r_imq_metric.c = float(imq_c)
                    score_val = _evaluate_met3r_imq(
                        met3r_imq_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "met3r_dust3r_imq":
                    if met3r_dust3r_imq_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        met3r_dust3r_imq_metric = MEt3R_IMQ(
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
                    if imq_c is not None:
                        met3r_dust3r_imq_metric.c = float(imq_c)
                    score_val = _evaluate_met3r_imq(
                        met3r_dust3r_imq_metric,
                        sample,
                        img_size=args.met3r_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "fast3r_imq":
                    if fast3r_imq_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        fast3r_imq_metric = MEt3R_Fast3R_IMQ(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                            per_pair_max_samples=per_pair_max,
                            max_total_samples=max_total,
                        )
                    if imq_c is not None:
                        fast3r_imq_metric.c = float(imq_c)
                    score_val = _evaluate_fast3r_imq(
                        fast3r_imq_metric,
                        sample,
                        resize=args.fast3r_resize,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "fast3r_pc_imq":
                    if fast3r_pc_imq_metric is None:
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        fast3r_pc_imq_metric = MEt3R_Fast3R_PointConsistency_IMQ(
                            img_size=None,
                            distance="cosine",
                            feature_backbone="dinov2",
                            feature_backbone_weights="mhamilton723/FeatUp",
                            upsampler="featup",
                            use_norm=True,
                            confidence_threshold=0.0,
                            freeze=True,
                            rasterizer_kwargs={},
                            fast3r_weights="jedyang97/Fast3R_ViT_Large_512",
                            fast3r_model=None,
                            focal_length_estimation_method="first_view_from_global_head",
                            pnp_iterations=100,
                            default_focal_px=1.6,
                            min_points_per_view=50,
                            device=str(device),
                            max_total_samples=max_total,
                        )
                    if imq_c is not None:
                        fast3r_pc_imq_metric.c = float(imq_c)
                    score_val = _evaluate_fast3r_pc_imq(
                        fast3r_pc_imq_metric,
                        sample,
                        resize=args.fast3r_resize,
                        device=device,
                    )
                elif metric_name == "vggt_imq":
                    if vggt_imq_metric is None:
                        per_pair_max = None if args.met3r_per_pair_max_samples <= 0 else args.met3r_per_pair_max_samples
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        vggt_imq_metric = MEt3R_VGGT_IMQ(
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
                    if imq_c is not None:
                        vggt_imq_metric.c = float(imq_c)
                    score_val = _evaluate_vggt_imq(
                        vggt_imq_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        per_pair_max_samples=args.met3r_per_pair_max_samples,
                        max_total_samples=args.met3r_max_total_samples,
                        device=device,
                    )
                elif metric_name == "vggt_pc_imq":
                    if vggt_pc_imq_metric is None:
                        max_total = None if args.met3r_max_total_samples <= 0 else args.met3r_max_total_samples
                        vggt_pc_imq_metric = MEt3R_VGGT_PointConsistency_IMQ(
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
                    if imq_c is not None:
                        vggt_pc_imq_metric.c = float(imq_c)
                    score_val = _evaluate_vggt_pc_imq(
                        vggt_pc_imq_metric,
                        sample,
                        image_size=args.vggt_img_size,
                        device=device,
                    )
                elif metric_name == "prism_mmd":
                    if prism_ref_by_k is None:
                        raise ValueError("prism_mmd requested but PRISM reference embeddings are missing.")
                    if prism_embedder is None or prism_compute_mmd is None:
                        PRISMEmbedder, compute_mmd = _import_prism()
                        prism_compute_mmd = compute_mmd
                        prism_embedder = PRISMEmbedder(device=str(device), repo_id=str(args.prism_repo_id))
                        prism_embedder.eval()
                    ref = prism_ref_by_k.get(int(sample.subset_size))
                    if ref is None:
                        raise ValueError(f"Missing PRISM reference embeddings for subset_size={sample.subset_size}.")
                    ref_dev = ref
                    if device.type == "cuda":
                        cached_ref = prism_ref_device_cache.get(int(sample.subset_size))
                        if cached_ref is None:
                            cached_ref = ref.to(device)
                            prism_ref_device_cache[int(sample.subset_size)] = cached_ref
                        ref_dev = cached_ref
                    score_val = _evaluate_prism_mmd(
                        prism_embedder,
                        prism_compute_mmd,
                        sample,
                        reference_embeddings=ref_dev,
                        azimuth_by_path=azimuth_by_path,
                        num_pairs=args.prism_num_pairs,
                        embed_batch_size=args.prism_embed_batch_size,
                        image_size=args.prism_image_size,
                        seed=args.seed,
                        device=device,
                    )
                elif metric_name == "sed" or metric_name == "tsed":
                    cached = sed_tsed_cache.get(sample.sample_id)
                    if cached is None:
                        sed_score, tsed_score = _evaluate_sed_and_tsed(
                            sample,
                            azimuth_by_path=azimuth_by_path,
                            noise_size=args.met3r_img_size,
                            t_e=args.tsed_t_e,
                            t_m=args.tsed_t_m,
                            pair_step=args.tsed_pair_step,
                            max_frames=args.tsed_max_frames,
                            ratio_test=args.tsed_ratio_test,
                            sift_nfeatures=args.tsed_sift_nfeatures,
                            max_dim=args.tsed_max_dim,
                        )
                        cached = (float(sed_score), float(tsed_score))
                        sed_tsed_cache[sample.sample_id] = cached
                    sed_score, tsed_score = cached
                    score_val = float(sed_score) if metric_name == "sed" else float(tsed_score)
                else:
                    raise ValueError(f"Unknown metric name: {metric_name}")

                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Finished metric=%s, scene=%s, kind=%s, subset_size=%d in %.2fs (score=%.4f)",
                    metric_name,
                    sample.scene or "",
                    sample.kind,
                    sample.subset_size,
                    elapsed,
                    float(score_val),
                )

                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "kind": sample.kind,
                        "scene": sample.scene or "",
                        "subset_size": sample.subset_size,
                        "metric": metric_name,
                        "score": score_val,
                        **_get_overlap_stats({
                            "met3r": met3r_metric, "met3r_mmd": met3r_mmd_metric,
                            "met3r_energy": met3r_energy_metric, "met3r_imq": met3r_imq_metric,
                            "met3r_dust3r": met3r_dust3r_metric, "met3r_dust3r_mmd": met3r_dust3r_mmd_metric,
                            "met3r_dust3r_energy": met3r_dust3r_energy_metric, "met3r_dust3r_imq": met3r_dust3r_imq_metric,
                            "fast3r": fast3r_metric, "fast3r_mmd": fast3r_mmd_metric,
                            "fast3r_energy": fast3r_energy_metric, "fast3r_imq": fast3r_imq_metric,
                            "fast3r_pc": fast3r_pc_metric, "fast3r_pc_mmd": fast3r_pc_mmd_metric,
                            "fast3r_pc_energy": fast3r_pc_energy_metric, "fast3r_pc_imq": fast3r_pc_imq_metric,
                            "vggt": vggt_metric, "vggt_robust": vggt_robust_metric, "vggt_mmd": vggt_mmd_metric,
                            "vggt_energy": vggt_energy_metric, "vggt_imq": vggt_imq_metric,
                            "vggt_pc": vggt_pc_metric, "vggt_pc_mmd": vggt_pc_mmd_metric,
                            "vggt_pc_energy": vggt_pc_energy_metric, "vggt_pc_imq": vggt_pc_imq_metric,
                        }.get(metric_name)),
                    }
                )

                # Accumulate for summary statistics.
                metric_dict = summary.setdefault(metric_name, {})
                size_dict = metric_dict.setdefault(sample.subset_size, {})
                size_dict.setdefault(sample.kind, []).append(float(score_val))

                if free_gpu_between_metrics and device.type == "cuda":
                    if metric_name == "met3r":
                        met3r_metric = None
                    elif metric_name == "met3r_mmd":
                        met3r_mmd_metric = None
                    elif metric_name == "met3r_energy":
                        met3r_energy_metric = None
                    elif metric_name == "met3r_imq":
                        met3r_imq_metric = None
                    elif metric_name == "met3r_dust3r":
                        met3r_dust3r_metric = None
                    elif metric_name == "met3r_dust3r_mmd":
                        met3r_dust3r_mmd_metric = None
                    elif metric_name == "met3r_dust3r_energy":
                        met3r_dust3r_energy_metric = None
                    elif metric_name == "met3r_dust3r_imq":
                        met3r_dust3r_imq_metric = None
                    elif metric_name == "fast3r":
                        fast3r_metric = None
                    elif metric_name == "fast3r_mmd":
                        fast3r_mmd_metric = None
                    elif metric_name == "fast3r_energy":
                        fast3r_energy_metric = None
                    elif metric_name == "fast3r_imq":
                        fast3r_imq_metric = None
                    elif metric_name == "fast3r_pc":
                        fast3r_pc_metric = None
                    elif metric_name == "fast3r_pc_mmd":
                        fast3r_pc_mmd_metric = None
                    elif metric_name == "fast3r_pc_energy":
                        fast3r_pc_energy_metric = None
                    elif metric_name == "fast3r_pc_imq":
                        fast3r_pc_imq_metric = None
                    elif metric_name == "vggt":
                        vggt_metric = None
                    elif metric_name == "vggt_robust":
                        vggt_robust_metric = None
                    elif metric_name == "vggt_mmd":
                        vggt_mmd_metric = None
                    elif metric_name == "vggt_energy":
                        vggt_energy_metric = None
                    elif metric_name == "vggt_imq":
                        vggt_imq_metric = None
                    elif metric_name == "vggt_pc":
                        vggt_pc_metric = None
                    elif metric_name == "vggt_pc_mmd":
                        vggt_pc_mmd_metric = None
                    elif metric_name == "vggt_pc_energy":
                        vggt_pc_energy_metric = None
                    elif metric_name == "vggt_pc_imq":
                        vggt_pc_imq_metric = None
                    elif metric_name == "prism_mmd":
                        prism_embedder = None
                        prism_ref_device_cache.clear()
                    torch.cuda.empty_cache()

    if args.mmd_sigma_out_json is not None:
        sigma_path = Path(args.mmd_sigma_out_json).expanduser().resolve()
        sigmas_out: MmdSigmaByMetric = {}
        for metric, per_k in sigma_observations.items():
            for k, values in per_k.items():
                if not values:
                    continue
                median_sigma = float(statistics.median(values))
                sigmas_out.setdefault(metric, {})[int(k)] = median_sigma
                logger.info(
                    "Estimated MMD sigma: metric=%s, subset_size=%d, median=%.6f (n=%d)",
                    metric,
                    int(k),
                    median_sigma,
                    len(values),
                )
        if not sigmas_out:
            raise ValueError("No sigma observations collected; cannot write --mmd-sigma-out-json.")
        if sigma_path.exists():
            existing = _load_mmd_sigma_by_k(sigma_path)
            for metric, per_k in sigmas_out.items():
                existing.setdefault(metric, {}).update(per_k)
            sigmas_out = existing
        _write_mmd_sigma_by_k(sigma_path, sigmas_out)

if __name__ == "__main__":
    main()
