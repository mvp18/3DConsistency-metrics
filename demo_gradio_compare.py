#!/usr/bin/env python3
"""
Gradio demo to compare reconstruction outputs from four engines:
  - MASt3R / DUSt3R via the MEt3R wrapper
  - Fast3R
  - VGGT
  - RobustVGGT

The demo supports multiple input modes:
  - Upload images (multi-view)
  - SysCON3D benchmark samples

Run on a GPU node and SSH port-forward the chosen port to view locally.

Example (run on GPU node):
  python demo_gradio_compare.py --server-name 127.0.0.1 --server-port 7860

Then (on your laptop):
  ssh -L 7860:localhost:7860 <user>@<host-or-node>
  open http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
for subdir in ("met3r", "fast3r", "vggt"):
    candidate = ROOT / subdir
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FAST3R_SIZE = 224
MET3R_SIZE = 224
VGGT_SIZE = 224

EngineName = Literal["met3r", "fast3r", "vggt", "robust_vggt"]
Met3RReconBackbone = Literal["mast3r", "dust3r", "raft"]
NoiseType = Literal["Uniform", "Gaussian", "Salt & pepper", "Constant"]
BenchmarkSampleKind = Literal[
    "consistent",
    "identical_images",
    "mixed",
    "mixed_controlled",
    "mixed_one_outlier",
    "consistent_gaussian_epsilon",
    "full_mixture_distinct_scene",
    "noise",
    "noise_gaussian",
    "patched_gaussian",
]
InputMode = Literal[
    "Upload images",
    "SysCON3D benchmark sample",
]


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray  # (N, 3), float32
    rgb: Optional[np.ndarray]  # (N, 3), uint8


@dataclass(frozen=True)
class EngineArtifacts:
    engine: EngineName
    point_cloud: Optional[PointCloud]
    ply_path: Optional[str]
    depth_images: List[Tuple[np.ndarray, str]]
    conf_images: List[Tuple[np.ndarray, str]]
    stats: Dict[str, Any]
    wall_time_s: float


@dataclass(frozen=True)
class BenchmarkSampleSpec:
    kind: BenchmarkSampleKind
    sample_id: str
    subset_size: int
    image_paths: List[str]
    scene_names: List[str]
    noise_type: Optional[NoiseType] = None
    seed: Optional[int] = None
    gaussian_sigma: Optional[float] = None
    patch_ratio: Optional[float] = None
    patch_num_patches: Optional[int] = None
    patch_seed: Optional[int] = None
    materialized: bool = False


_FAST3R_MODEL: Any | None = None
_VGGT_MODEL: Any | None = None
_MET3R_CACHE: Dict[Tuple[Met3RReconBackbone, int], Any] = {}
_MIPNERF360_SCENE_CACHE: Dict[str, Dict[str, List[str]]] = {}
_MIPNERF360_IMPOSSIBLE_CACHE: Dict[str, Dict[str, Any]] = {}
_MIPNERF360_CALIBRATION_CACHE: Dict[str, Dict[str, Any]] = {}
BENCHMARK_SAMPLE_KINDS: Tuple[BenchmarkSampleKind, ...] = (
    "consistent",
    "identical_images",
    "mixed",
    "mixed_controlled",
    "mixed_one_outlier",
    "consistent_gaussian_epsilon",
    "full_mixture_distinct_scene",
    "noise",
    "noise_gaussian",
    "patched_gaussian",
)
BENCHMARK_SAMPLE_KIND_LABELS: Dict[BenchmarkSampleKind, str] = {
    "consistent": "Consistent (L0)",
    "identical_images": "Identical images",
    "mixed": "Mixed",
    "mixed_controlled": "Mixed-controlled",
    "mixed_one_outlier": "Mixed one-outlier",
    "consistent_gaussian_epsilon": "Consistent Gaussian epsilon",
    "full_mixture_distinct_scene": "Full mixture distinct-scene",
    "noise": "Noise",
    "noise_gaussian": "Noise Gaussian",
    "patched_gaussian": "Patched Gaussian",
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_tmpdir(tmp_root: Path) -> None:
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_root))
    os.environ.setdefault("TEMP", str(tmp_root))
    os.environ.setdefault("TMP", str(tmp_root))
    os.environ.setdefault("GRADIO_TEMP_DIR", str(tmp_root))
    tempfile.tempdir = str(tmp_root)


def _resolve_gradio_files(files: Optional[Sequence[Any]]) -> List[str]:
    if not files:
        return []
    paths: List[str] = []
    for f in files:
        if isinstance(f, dict) and "name" in f:
            paths.append(str(f["name"]))
        else:
            paths.append(str(f))
    return paths


def _normalize_dataset_root_override(dataset_root_override: Optional[Path | str]) -> Optional[Path]:
    if dataset_root_override in (None, ""):
        return None
    return Path(dataset_root_override).expanduser().resolve()


def _manifest_cache_key(manifest_path: Path, dataset_root_override: Optional[Path | str]) -> str:
    override = _normalize_dataset_root_override(dataset_root_override)
    return f"{manifest_path.resolve()}::{override if override is not None else ''}"


def _resolve_manifest_dataset_root(
    manifest: Dict[str, Any],
    manifest_path: Path,
    dataset_root_override: Optional[Path | str] = None,
) -> Path:
    override = _normalize_dataset_root_override(dataset_root_override)
    if override is not None:
        return override

    dataset_root = Path(manifest["dataset_root"]).expanduser()
    if dataset_root.is_absolute():
        return dataset_root.resolve()
    return (manifest_path.resolve().parent / dataset_root).resolve()


def _scene_names_from_rel_paths(rel_paths: Sequence[str]) -> List[str]:
    scene_names: List[str] = []
    for rel_path in rel_paths:
        parts = Path(rel_path).parts
        if len(parts) >= 3 and parts[0] == "syscon3d_scene_types":
            scene_names.append(parts[1] if parts[2].startswith("k") else parts[2])
        elif parts:
            scene_names.append(parts[0])
    return scene_names


def _display_rel_path(raw_path: str, dataset_root: Path) -> Path:
    path_value = Path(raw_path)
    if not path_value.is_absolute():
        return path_value

    for base in (dataset_root, dataset_root.parent):
        try:
            return path_value.relative_to(base)
        except ValueError:
            continue
    return Path(path_value.name)


def _resolve_manifest_image_path(raw_path: str, dataset_root: Path) -> Path:
    path_value = Path(raw_path)
    if path_value.is_absolute():
        return path_value
    candidate_roots = [dataset_root]
    extra_root = os.environ.get("SYSCON3D_EXTRA_DATA_ROOT")
    if extra_root:
        candidate_roots.extend(Path(root).expanduser() for root in extra_root.split(os.pathsep) if root)
    candidate_roots.append(Path(__file__).resolve().parent / "tmp" / "syscon3d_scene_types_source")
    for root in candidate_roots:
        candidate = root / path_value
        if candidate.exists():
            return candidate
    return dataset_root / path_value


def _configure_syscon3d_extra_data_root(extra_data_root: Optional[str]) -> None:
    if not extra_data_root:
        return
    root = Path(extra_data_root).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    os.environ["SYSCON3D_EXTRA_DATA_ROOT"] = str(root.resolve())


def _pretty_benchmark_kind(sample_kind: BenchmarkSampleKind) -> str:
    return BENCHMARK_SAMPLE_KIND_LABELS[sample_kind]


def _load_mipnerf360_scene_images(
    splits_json: Path,
    dataset_root_override: Optional[Path | str] = None,
) -> Dict[str, List[str]]:
    cache_key = _manifest_cache_key(splits_json, dataset_root_override)
    if cache_key in _MIPNERF360_SCENE_CACHE:
        return _MIPNERF360_SCENE_CACHE[cache_key]

    with splits_json.open("r") as f:
        data = json.load(f)
    dataset_root = _resolve_manifest_dataset_root(data, splits_json, dataset_root_override=dataset_root_override)
    scenes = sorted(data.get("scenes", {}).keys())

    scene_paths: Dict[str, List[str]] = {}
    for scene in scenes:
        transforms_path = dataset_root / scene / "transforms.json"
        with transforms_path.open("r") as f:
            transforms = json.load(f)
        frames = transforms.get("frames", [])

        paths: List[str] = []
        for fr in frames:
            rel_path = fr["file_path"]
            candidate = (dataset_root / scene / rel_path).resolve()
            if candidate.exists():
                paths.append(str(candidate))
        scene_paths[scene] = paths

    _MIPNERF360_SCENE_CACHE[cache_key] = scene_paths
    return scene_paths


def _load_mipnerf360_impossible_splits(impossible_splits_json: Path) -> Dict[str, Any]:
    cache_key = str(impossible_splits_json.resolve())
    if cache_key in _MIPNERF360_IMPOSSIBLE_CACHE:
        return _MIPNERF360_IMPOSSIBLE_CACHE[cache_key]

    with impossible_splits_json.open("r") as f:
        data = json.load(f)

    _MIPNERF360_IMPOSSIBLE_CACHE[cache_key] = data
    return data


def _load_mipnerf360_calibration_splits(splits_json: Path) -> Dict[str, Any]:
    cache_key = str(splits_json.resolve())
    if cache_key in _MIPNERF360_CALIBRATION_CACHE:
        return _MIPNERF360_CALIBRATION_CACHE[cache_key]

    with splits_json.open("r") as f:
        data = json.load(f)

    _MIPNERF360_CALIBRATION_CACHE[cache_key] = data
    return data


def _consistent_sample_ids(splits_json: Path, subset_size: int) -> List[str]:
    data = _load_mipnerf360_calibration_splits(splits_json)
    ids = [
        f"consistent_{scene}_k{int(subset_size):02d}"
        for scene, scene_data in data.get("scenes", {}).items()
        if str(int(subset_size)) in scene_data
    ]
    ids.sort()
    return ids


def _split_consistent_sample_id(sample_id: str) -> Tuple[str, int]:
    prefix = "consistent_"
    if not sample_id.startswith(prefix) or "_k" not in sample_id:
        raise ValueError(f"Invalid consistent sample_id: {sample_id}")
    scene, k_text = sample_id[len(prefix) :].rsplit("_k", 1)
    return scene, int(k_text)


def _benchmark_sample_ids(
    impossible_splits_json: Path,
    sample_kind: BenchmarkSampleKind,
    subset_size: int,
    calibration_splits_json: Optional[Path] = None,
) -> List[str]:
    if sample_kind == "consistent":
        splits_json = calibration_splits_json or Path("data/mipnerf360_calibration_splits.json")
        return _consistent_sample_ids(splits_json, subset_size=subset_size)

    data = _load_mipnerf360_impossible_splits(impossible_splits_json)
    entries = data.get(sample_kind, [])
    ids = [str(e["sample_id"]) for e in entries if int(e.get("subset_size", -1)) == int(subset_size)]
    ids.sort()
    return ids


def _load_benchmark_sample(
    impossible_splits_json: Path,
    sample_kind: BenchmarkSampleKind,
    sample_id: Optional[str],
    dataset_root_override: Optional[Path | str] = None,
    calibration_splits_json: Optional[Path] = None,
    subset_size: Optional[int] = None,
) -> BenchmarkSampleSpec:
    if subset_size is not None:
        ids = _benchmark_sample_ids(
            impossible_splits_json,
            sample_kind=sample_kind,
            subset_size=int(subset_size),
            calibration_splits_json=calibration_splits_json,
        )
        sample_text = "" if sample_id is None else str(sample_id)
        if (not sample_text or sample_text == "None" or sample_text not in ids) and ids:
            sample_id = ids[0]

    if sample_kind == "consistent":
        if calibration_splits_json is None:
            raise ValueError("calibration_splits_json is required for consistent samples.")
        sample_text = "" if sample_id is None else str(sample_id)
        scene, k = _split_consistent_sample_id(sample_text)
        paths = _load_consistent_sample(
            calibration_splits_json,
            scene=scene,
            subset_size=k,
            dataset_root_override=dataset_root_override,
        )
        return BenchmarkSampleSpec(
            kind=sample_kind,
            sample_id=sample_text,
            subset_size=k,
            image_paths=paths,
            scene_names=[scene] * len(paths),
            materialized=True,
        )

    data = _load_mipnerf360_impossible_splits(impossible_splits_json)
    dataset_root = _resolve_manifest_dataset_root(data, impossible_splits_json, dataset_root_override=dataset_root_override)
    entries = data.get(sample_kind, [])
    for entry in entries:
        if str(entry.get("sample_id", "")) != str(sample_id):
            continue

        rel_paths = [str(_display_rel_path(str(rel_path), dataset_root)) for rel_path in entry.get("image_rel_paths", [])]
        abs_paths = [str(_resolve_manifest_image_path(str(raw_path), dataset_root).resolve()) for raw_path in entry.get("image_rel_paths", [])]
        scene_names = [str(scene) for scene in entry.get("source_scenes", [])]
        if not scene_names:
            scene_names = _scene_names_from_rel_paths(rel_paths)
        materialized = bool(entry.get("materialized", False))

        if sample_kind == "noise":
            return BenchmarkSampleSpec(
                kind=sample_kind,
                sample_id=str(entry["sample_id"]),
                subset_size=int(entry["subset_size"]),
                image_paths=abs_paths,
                scene_names=scene_names,
                noise_type="Uniform",
                seed=int(entry["seed"]),
                materialized=materialized,
            )

        if sample_kind == "noise_gaussian":
            return BenchmarkSampleSpec(
                kind=sample_kind,
                sample_id=str(entry["sample_id"]),
                subset_size=int(entry["subset_size"]),
                image_paths=abs_paths,
                scene_names=scene_names,
                noise_type="Gaussian",
                seed=int(entry["seed"]),
                gaussian_sigma=float(entry["gaussian_sigma"]),
                materialized=materialized,
            )

        if sample_kind == "patched_gaussian":
            return BenchmarkSampleSpec(
                kind=sample_kind,
                sample_id=str(entry["sample_id"]),
                subset_size=int(entry["subset_size"]),
                image_paths=abs_paths,
                scene_names=scene_names,
                gaussian_sigma=float(entry["gaussian_sigma"]),
                patch_ratio=float(entry["patch_ratio"]),
                patch_num_patches=int(entry["patch_num_patches"]),
                patch_seed=int(entry["patch_seed"]),
                materialized=materialized,
            )

        return BenchmarkSampleSpec(
            kind=sample_kind,
            sample_id=str(entry["sample_id"]),
            subset_size=int(entry["subset_size"]),
            image_paths=abs_paths,
            scene_names=scene_names,
            gaussian_sigma=float(entry["gaussian_sigma"]) if "gaussian_sigma" in entry else None,
            materialized=materialized,
        )

    raise ValueError(f"Unknown {sample_kind} sample_id: {sample_id}")


def _mixed_controlled_sample_ids(impossible_splits_json: Path, subset_size: int) -> List[str]:
    return _benchmark_sample_ids(impossible_splits_json, sample_kind="mixed_controlled", subset_size=subset_size)


def _load_mixed_controlled_sample_paths(
    impossible_splits_json: Path,
    sample_id: str,
    dataset_root_override: Optional[Path | str] = None,
) -> Tuple[List[str], List[str], int]:
    sample = _load_benchmark_sample(
        impossible_splits_json,
        sample_kind="mixed_controlled",
        sample_id=sample_id,
        dataset_root_override=dataset_root_override,
    )
    return sample.image_paths, sample.scene_names, sample.subset_size


def _sample_mixed_scenes(scenes: Sequence[str], k: int, seed: int, unique_if_possible: bool) -> List[str]:
    rng = np.random.default_rng(seed)
    if unique_if_possible and k <= len(scenes):
        idx = rng.choice(len(scenes), size=k, replace=False)
        return [scenes[i] for i in idx.tolist()]
    idx = rng.choice(len(scenes), size=k, replace=True)
    return [scenes[i] for i in idx.tolist()]


def _sample_mixed_image_paths(
    scene_paths: Dict[str, List[str]],
    k: int,
    seed: int,
    unique_scenes_if_possible: bool,
) -> Tuple[List[str], List[str]]:
    scenes = sorted(scene for scene, paths in scene_paths.items() if paths)
    if not scenes:
        raise ValueError("No scenes found for mixed sampling.")

    chosen_scenes = _sample_mixed_scenes(scenes, k=k, seed=seed, unique_if_possible=unique_scenes_if_possible)

    rng = np.random.default_rng(seed + 1)
    paths: List[str] = []
    for scene in chosen_scenes:
        candidates = scene_paths.get(scene, [])
        if not candidates:
            raise ValueError(f"No images found for scene={scene}.")
        paths.append(str(candidates[int(rng.integers(0, len(candidates)))]))

    return paths, chosen_scenes


def _sample_outlier_image_path(
    scene_paths: Dict[str, List[str]],
    base_scene: str,
    seed: int,
    outlier_scene: str,
) -> Tuple[str, str]:
    scenes = sorted(scene for scene, paths in scene_paths.items() if paths)
    if not scenes:
        raise ValueError("No scenes found for outlier sampling.")

    chosen_scene = outlier_scene.strip()
    if chosen_scene:
        if chosen_scene not in scene_paths:
            raise ValueError(f"Unknown outlier scene: {chosen_scene}")
        if chosen_scene == base_scene:
            raise ValueError("Outlier scene must differ from the base scene.")
    else:
        candidates = [s for s in scenes if s != base_scene]
        if not candidates:
            raise ValueError("No alternative scenes available for outlier sampling.")
        rng = np.random.default_rng(seed)
        chosen_scene = str(candidates[int(rng.integers(0, len(candidates)))])

    candidates = scene_paths.get(chosen_scene, [])
    if not candidates:
        raise ValueError(f"No images found for outlier scene={chosen_scene}.")
    rng = np.random.default_rng(seed + 1)
    return str(candidates[int(rng.integers(0, len(candidates)))]), chosen_scene


def _load_consistent_sample(
    splits_json: Path,
    scene: str,
    subset_size: int,
    dataset_root_override: Optional[Path | str] = None,
) -> List[str]:
    with splits_json.open("r") as f:
        data = json.load(f)
    dataset_root = _resolve_manifest_dataset_root(data, splits_json, dataset_root_override=dataset_root_override)

    scene_data = data.get("scenes", {}).get(scene, {})
    indices = scene_data.get(str(subset_size), [])
    if not indices:
        raise ValueError(f"No indices found for scene={scene}, subset_size={subset_size} in {splits_json}")

    transforms_path = dataset_root / scene / "transforms.json"
    with transforms_path.open("r") as f:
        transforms = json.load(f)
    frames = transforms.get("frames", [])

    paths: List[str] = []
    for idx in indices:
        rel_path = frames[idx]["file_path"]
        full_path = (dataset_root / scene / rel_path).resolve()
        paths.append(str(full_path))
    return paths


def _load_images_square(paths: Sequence[str], target_size: int) -> torch.Tensor:
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([T.Resize((target_size, target_size)), T.ToTensor()])
    tensors: List[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)


def _generate_noise_images(
    num_images: int,
    size: int,
    seed: int,
    device: torch.device,
    noise_type: NoiseType = "Uniform",
    gaussian_sigma: float = 0.2,
    salt_pepper_prob: float = 0.05,
    constant_value: float = 0.5,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    if noise_type == "Uniform":
        return torch.rand(num_images, 3, size, size, generator=g, device=device)

    if noise_type == "Gaussian":
        x = torch.randn(num_images, 3, size, size, generator=g, device=device) * float(gaussian_sigma) + 0.5
        return x.clamp(0.0, 1.0)

    if noise_type == "Salt & pepper":
        p = float(salt_pepper_prob)
        base = torch.full((num_images, 3, size, size), fill_value=0.5, device=device)
        if p <= 0.0:
            return base
        mask = torch.rand(num_images, 3, size, size, generator=g, device=device) < p
        salt = torch.rand(num_images, 3, size, size, generator=g, device=device) < 0.5
        base[mask & salt] = 1.0
        base[mask & ~salt] = 0.0
        return base

    if noise_type == "Constant":
        v = float(np.clip(constant_value, 0.0, 1.0))
        return torch.full((num_images, 3, size, size), fill_value=v, device=device)

    raise ValueError(f"Unsupported noise_type: {noise_type}")


def _add_gaussian_patches(
    images: torch.Tensor,
    patch_ratio: float,
    num_patches: int,
    seed: int,
    gaussian_sigma: float = 0.2,
) -> torch.Tensor:
    g = torch.Generator(device=images.device)
    g.manual_seed(seed)
    patched = images.clone()
    _, c, h, w = images.shape
    patch_h = max(1, int(h * patch_ratio))
    patch_w = max(1, int(w * patch_ratio))
    for i in range(images.shape[0]):
        for _ in range(num_patches):
            y = int(torch.randint(0, h - patch_h + 1, (1,), generator=g, device=images.device).item())
            x = int(torch.randint(0, w - patch_w + 1, (1,), generator=g, device=images.device).item())
            patch = torch.randn(c, patch_h, patch_w, generator=g, device=images.device) * float(gaussian_sigma) + 0.5
            patch = patch.clamp(0.0, 1.0)
            patched[i, :, y : y + patch_h, x : x + patch_w] = patch
    return patched


def _depth_from_points(points_hw3: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(points_hw3, dim=-1)


def _normalize_to_uint8(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)
    lo = np.percentile(x[finite], p_lo)
    hi = np.percentile(x[finite], p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    y = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return (255.0 * y).astype(np.uint8)


def _subsample_points(xyz: np.ndarray, rgb: Optional[np.ndarray], max_points: int, seed: int) -> PointCloud:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return PointCloud(xyz=xyz, rgb=rgb)

    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    idx.sort()
    return PointCloud(
        xyz=xyz[idx].astype(np.float32, copy=False),
        rgb=None if rgb is None else rgb[idx].astype(np.uint8, copy=False),
    )


def _write_ply(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray]) -> None:
    if rgb is not None and rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 colors for PLY export.")

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if rgb is not None:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")

    with path.open("w") as f:
        f.write("\n".join(header) + "\n")
        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _rgb_to_hex(rgb: np.ndarray) -> List[str]:
    rgb = rgb.astype(np.uint8, copy=False)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb.tolist()]


def _plot_point_cloud(pc: PointCloud, title: str, color_mode: Literal["rgb", "solid"], solid_color: str) -> Any:
    import plotly.graph_objects as go

    xyz = pc.xyz
    if xyz.size == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    if color_mode == "rgb" and pc.rgb is not None:
        marker_color = _rgb_to_hex(pc.rgb)
    else:
        marker_color = solid_color

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=1, color=marker_color, opacity=0.9),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _plot_compare(artifacts: Dict[EngineName, EngineArtifacts]) -> Any:
    import plotly.graph_objects as go

    engine_colors = {
        "met3r": "#ef4444",
        "fast3r": "#22c55e",
        "vggt": "#3b82f6",
        "robust_vggt": "#f59e0b",
    }

    fig = go.Figure()
    for engine, art in artifacts.items():
        if art.point_cloud is None:
            continue
        xyz = art.point_cloud.xyz
        if xyz.size == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name=engine,
                marker=dict(size=1, color=engine_colors[engine], opacity=0.8),
            )
        )
    fig.update_layout(
        title="Point Cloud Comparison (engine-colored)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h"),
    )
    return fig


def _stats_markdown(artifacts: Dict[EngineName, EngineArtifacts]) -> str:
    lines = [
        "### Summary",
        "",
        "| engine | points | bbox_vol | depth_mean | conf_mean | time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for engine in ("met3r", "fast3r", "vggt", "robust_vggt"):
        art = artifacts.get(engine)
        if art is None:
            continue
        stats = art.stats
        pc = stats.get("point_cloud", {})
        depth = stats.get("depth", {})
        conf = stats.get("confidence", {})
        lines.append(
            f"| {engine} | {pc.get('num_points', '—')} | {pc.get('bbox_vol', '—')} |"
            f" {depth.get('mean', '—')} | {conf.get('mean', '—')} | {art.wall_time_s:.2f} |"
        )
    return "\n".join(lines)


def _json_pretty(x: Any) -> str:
    return json.dumps(x, indent=2, sort_keys=True)


def _pc_stats(xyz: np.ndarray) -> Dict[str, Any]:
    if xyz.size == 0:
        return {"num_points": 0}
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    extent = maxs - mins
    bbox_vol = float(np.prod(extent))
    return {
        "num_points": int(xyz.shape[0]),
        "mean_xyz": [float(x) for x in xyz.mean(axis=0)],
        "std_xyz": [float(x) for x in xyz.std(axis=0)],
        "min_xyz": [float(x) for x in mins],
        "max_xyz": [float(x) for x in maxs],
        "bbox_vol": bbox_vol,
    }


def _scalar_stats(x: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(x)
    if not np.any(finite):
        return {"num_valid": 0, "num_total": int(x.size)}
    xf = x[finite]
    return {
        "mean": float(xf.mean()),
        "std": float(xf.std()),
        "min": float(xf.min()),
        "max": float(xf.max()),
        "num_valid": int(xf.size),
        "num_total": int(x.size),
    }


def _mast3r_repo_candidates() -> List[Path]:
    candidates = []
    env_path = os.environ.get("MAST3R_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            ROOT / "mast3r",
            ROOT.parent / "mast3r",
            Path.cwd() / "mast3r",
            Path.cwd().parent / "mast3r",
            ROOT / "met3r" / "mast3r",
        ]
    )
    return candidates


def _has_met3r_recon_backends() -> bool:
    for mast3r_repo in _mast3r_repo_candidates():
        if (mast3r_repo / "mast3r").is_dir() and (mast3r_repo / "dust3r" / "dust3r").is_dir():
            return True
    return False


def _get_fast3r_model(device: torch.device) -> Any:
    global _FAST3R_MODEL
    if _FAST3R_MODEL is None:
        from fast3r.models.fast3r import Fast3R

        logger.info("Loading Fast3R model...")
        _FAST3R_MODEL = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
        _FAST3R_MODEL.eval()
    return _FAST3R_MODEL.to(device)


def _get_vggt_model(device: torch.device) -> Any:
    global _VGGT_MODEL
    if _VGGT_MODEL is None:
        from vggt.models.vggt import VGGT

        logger.info("Loading VGGT model...")
        _VGGT_MODEL = VGGT.from_pretrained("facebook/VGGT-1B")
        _VGGT_MODEL.eval()
    return _VGGT_MODEL.to(device)


def _load_vggt_input(
    image_paths: Optional[Sequence[str]],
    images_01: Optional[torch.Tensor],
) -> torch.Tensor:
    if image_paths is not None:
        from vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images(list(image_paths), target_size=VGGT_SIZE)
    if images_01 is None:
        raise ValueError("VGGT requires either image_paths or images_01.")
    return images_01.detach().cpu()


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.to(torch.float32)
    denom = scores.max() - scores.min()
    if float(denom.abs().item()) < 1e-6:
        return torch.ones_like(scores)
    return (scores - scores.min()) / denom


def _robust_vggt_view_selection(
    model: Any,
    images: torch.Tensor,
    device: torch.device,
    *,
    robust_layer_idx: int,
    attention_weight: float,
    cosine_weight: float,
    rejection_threshold: float,
    anchor_index: int,
) -> Dict[str, Any]:
    if images.ndim != 4:
        raise ValueError(f"Expected VGGT images with shape (K,3,H,W), got {tuple(images.shape)}")
    if not 0 <= anchor_index < images.shape[0]:
        raise ValueError(f"anchor_index={anchor_index} is out of bounds for K={images.shape[0]}")

    q_cache: List[torch.Tensor] = []
    k_cache: List[torch.Tensor] = []

    def _store_q(_module: Any, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        q_cache.append(output.detach())

    def _store_k(_module: Any, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        k_cache.append(output.detach())

    attn = model.aggregator.global_blocks[robust_layer_idx].attn
    handles = [
        attn.q_norm.register_forward_hook(_store_q),
        attn.k_norm.register_forward_hook(_store_k),
    ]
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    images_batch = images.unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
                aggregated_tokens_list, patch_start_idx = model.aggregator(images_batch)
    finally:
        for handle in handles:
            handle.remove()

    if not q_cache or not k_cache:
        raise RuntimeError("Failed to capture VGGT q/k tensors for RobustVGGT view scoring.")

    layer_tokens = aggregated_tokens_list[robust_layer_idx]
    patch_tokens = layer_tokens[..., patch_start_idx:, layer_tokens.shape[-1] // 2 :]
    features = F.normalize(patch_tokens[0].to(torch.float32), p=2, dim=-1)
    ref_features = features[anchor_index]
    cosine_scores = torch.matmul(features, ref_features.transpose(0, 1)).mean(dim=(1, 2))

    q_tokens = q_cache[0]
    k_tokens = k_cache[0]
    patch_size = int(model.aggregator.patch_size)
    image_h, image_w = int(images.shape[-2]), int(images.shape[-1])
    num_patch_tokens = (image_h // patch_size) * (image_w // patch_size)
    tokens_per_image = int(patch_start_idx) + num_patch_tokens
    query_start = anchor_index * tokens_per_image + int(patch_start_idx)
    query_end = query_start + num_patch_tokens
    total_tokens = int(images.shape[0]) * tokens_per_image

    q_anchor = q_tokens[:, :, query_start:query_end, :].to(torch.float32)
    k_all = k_tokens[:, :, :total_tokens, :].to(torch.float32)
    attn_logits = torch.einsum("bhqd,bhtd->bhqt", q_anchor, k_all) / math.sqrt(float(q_anchor.shape[-1]))
    attn_probs = torch.softmax(attn_logits, dim=-1).mean(dim=1).mean(dim=1)[0]
    attention_scores = []
    for view_idx in range(int(images.shape[0])):
        start = view_idx * tokens_per_image + int(patch_start_idx)
        end = start + num_patch_tokens
        attention_scores.append(attn_probs[start:end].mean())
    attention_scores_t = torch.stack(attention_scores)

    combined_scores = (
        float(attention_weight) * _normalize_scores(attention_scores_t)
        + float(cosine_weight) * _normalize_scores(cosine_scores)
    )
    keep_mask = combined_scores >= float(rejection_threshold)
    keep_mask[anchor_index] = True
    kept_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).detach().cpu().tolist()
    rejected_indices = torch.nonzero(~keep_mask, as_tuple=False).squeeze(1).detach().cpu().tolist()

    return {
        "kept_indices": [int(idx) for idx in kept_indices],
        "rejected_indices": [int(idx) for idx in rejected_indices],
        "combined_scores": [float(x) for x in combined_scores.detach().cpu()],
        "attention_scores": [float(x) for x in attention_scores_t.detach().cpu()],
        "cosine_scores": [float(x) for x in cosine_scores.detach().cpu()],
        "robust_layer_idx": int(robust_layer_idx),
        "attention_weight": float(attention_weight),
        "cosine_weight": float(cosine_weight),
        "rejection_threshold": float(rejection_threshold),
        "anchor_index": int(anchor_index),
    }


def _get_met3r_model(
    device: torch.device,
    recon_backbone: Met3RReconBackbone,
    img_size: int,
) -> Any:
    key = (recon_backbone, img_size)
    if key not in _MET3R_CACHE:
        from met3r import MEt3R

        feature_backbone = "dinov2"
        logger.info(f"Loading MEt3R model (backbone={recon_backbone}, feat={feature_backbone})...")
        metric = MEt3R(
            img_size=img_size,
            use_norm=True,
            backbone=recon_backbone,
            feature_backbone=feature_backbone,
            feature_backbone_weights="mhamilton723/FeatUp",
            upsampler="featup",
            distance="mse",
            freeze=True,
        )
        metric.eval()
        _MET3R_CACHE[key] = metric
    return _MET3R_CACHE[key].to(device)


def _run_fast3r(
    images_01: torch.Tensor,
    device: torch.device,
    max_points: int,
    seed: int,
    conf_threshold: float,
    out_dir: Path,
    colorize: bool,
) -> EngineArtifacts:
    t0 = time.time()
    model = _get_fast3r_model(device)

    k, _, h, w = images_01.shape
    images_norm = (images_01 * 2.0) - 1.0

    views: List[Dict[str, torch.Tensor]] = []
    for idx in range(k):
        views.append(
            {
                "img": images_norm[idx : idx + 1].to(device, dtype=torch.float32),
                "true_shape": torch.tensor([[h, w]], device=device, dtype=torch.long),
                "dataset": torch.tensor([0], device=device),
                "label": torch.tensor([0], device=device),
                "instance": torch.tensor([idx], device=device),
            }
        )

    with torch.no_grad():
        preds = model(views)

    points_list: List[np.ndarray] = []
    colors_list: List[np.ndarray] = []
    depth_imgs: List[Tuple[np.ndarray, str]] = []
    conf_imgs: List[Tuple[np.ndarray, str]] = []
    depth_vals: List[np.ndarray] = []
    conf_vals: List[np.ndarray] = []

    images_rgb = (images_01.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()  # (K,3,H,W)

    for idx, pred in enumerate(preds):
        pts3d = pred.get("pts3d_in_other_view")
        conf = pred.get("conf")
        if pts3d is None:
            continue
        pts = pts3d[0].detach().cpu()  # (H,W,3)
        conf_map = None
        if isinstance(conf, torch.Tensor):
            conf_map = conf[0].detach().cpu()

        depth = _depth_from_points(pts).numpy()
        depth_vals.append(depth.reshape(-1))
        depth_imgs.append((_normalize_to_uint8(depth), f"depth view {idx}"))

        if conf_map is not None:
            conf_np = conf_map.numpy()
            conf_vals.append(conf_np.reshape(-1))
            conf_imgs.append((_normalize_to_uint8(conf_np), f"conf view {idx}"))

        pts_np = pts.reshape(-1, 3).numpy()
        if conf_map is not None:
            mask = (conf_map.reshape(-1) >= conf_threshold).numpy()
            pts_np = pts_np[mask]
            rgb_np = images_rgb[idx].transpose(1, 2, 0).reshape(-1, 3)[mask]
        else:
            rgb_np = images_rgb[idx].transpose(1, 2, 0).reshape(-1, 3)

        finite = np.isfinite(pts_np).all(axis=1)
        pts_np = pts_np[finite]
        rgb_np = rgb_np[finite]
        points_list.append(pts_np.astype(np.float32, copy=False))
        colors_list.append(rgb_np.astype(np.uint8, copy=False))

    if not points_list:
        wall = time.time() - t0
        return EngineArtifacts(
            engine="fast3r",
            point_cloud=None,
            ply_path=None,
            depth_images=[],
            conf_images=[],
            stats={"error": "Fast3R produced no 3D points."},
            wall_time_s=wall,
        )

    xyz = np.concatenate(points_list, axis=0)
    rgb = np.concatenate(colors_list, axis=0) if colorize else None
    pc_full = PointCloud(xyz=xyz, rgb=rgb)
    pc = _subsample_points(pc_full.xyz, pc_full.rgb, max_points=max_points, seed=seed)

    ply_path = out_dir / "fast3r_points.ply"
    _write_ply(ply_path, pc.xyz, pc.rgb)

    depth_all = np.concatenate(depth_vals, axis=0) if depth_vals else np.array([], dtype=np.float32)
    conf_all = np.concatenate(conf_vals, axis=0) if conf_vals else np.array([], dtype=np.float32)
    stats = {
        "point_cloud": _pc_stats(xyz),
        "depth": _scalar_stats(depth_all),
        "confidence": _scalar_stats(conf_all) if conf_vals else {"note": "no conf map returned"},
        "conf_threshold": conf_threshold,
        "image_size": [h, w],
    }

    wall = time.time() - t0
    return EngineArtifacts(
        engine="fast3r",
        point_cloud=pc,
        ply_path=str(ply_path),
        depth_images=depth_imgs,
        conf_images=conf_imgs,
        stats=stats,
        wall_time_s=wall,
    )


def _vggt_predictions_to_artifacts(
    *,
    engine: EngineName,
    preds: Dict[str, Any],
    images: torch.Tensor,
    max_points: int,
    seed: int,
    conf_threshold: float,
    out_dir: Path,
    colorize: bool,
    ply_name: str,
    wall_time_s: float,
    view_indices: Optional[Sequence[int]] = None,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> EngineArtifacts:
    world_pts = preds.get("world_points")
    world_conf = preds.get("world_points_conf")
    depth = preds.get("depth")
    depth_conf = preds.get("depth_conf")

    if world_pts is None:
        return EngineArtifacts(
            engine=engine,
            point_cloud=None,
            ply_path=None,
            depth_images=[],
            conf_images=[],
            stats={"error": f"{engine} produced no world_points."},
            wall_time_s=wall_time_s,
        )

    world_pts_k = world_pts[0].detach().cpu()  # (K,H,W,3)
    k, h, w, _ = world_pts_k.shape
    images_rgb = (images.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()  # (K,3,H,W)
    if view_indices is None:
        view_indices = list(range(k))

    xyz_list: List[np.ndarray] = []
    rgb_list: List[np.ndarray] = []
    depth_imgs: List[Tuple[np.ndarray, str]] = []
    conf_imgs: List[Tuple[np.ndarray, str]] = []
    depth_vals: List[np.ndarray] = []
    conf_vals: List[np.ndarray] = []

    conf_map_k = None
    if world_conf is not None:
        conf_map_k = world_conf[0].detach().cpu()
        if conf_map_k.ndim == 4 and conf_map_k.shape[-1] == 1:
            conf_map_k = conf_map_k[..., 0]

    for idx in range(k):
        pts = world_pts_k[idx]  # (H,W,3)
        depth_np = _depth_from_points(pts).numpy()
        depth_vals.append(depth_np.reshape(-1))
        source_idx = int(view_indices[idx]) if idx < len(view_indices) else idx
        depth_imgs.append((_normalize_to_uint8(depth_np), f"depth view {source_idx}"))

        conf_np = None
        if conf_map_k is not None:
            conf_np = conf_map_k[idx].numpy()
            conf_vals.append(conf_np.reshape(-1))
            conf_imgs.append((_normalize_to_uint8(conf_np), f"conf view {source_idx}"))

        pts_np = pts.reshape(-1, 3).numpy()
        rgb_np = images_rgb[idx].transpose(1, 2, 0).reshape(-1, 3)

        if conf_np is not None:
            mask = conf_np.reshape(-1) >= conf_threshold
            pts_np = pts_np[mask]
            rgb_np = rgb_np[mask]

        finite = np.isfinite(pts_np).all(axis=1)
        pts_np = pts_np[finite]
        rgb_np = rgb_np[finite]
        xyz_list.append(pts_np.astype(np.float32, copy=False))
        rgb_list.append(rgb_np.astype(np.uint8, copy=False))

    xyz = np.concatenate(xyz_list, axis=0)
    rgb = np.concatenate(rgb_list, axis=0) if colorize else None
    pc_full = PointCloud(xyz=xyz, rgb=rgb)
    pc = _subsample_points(pc_full.xyz, pc_full.rgb, max_points=max_points, seed=seed)

    ply_path = out_dir / ply_name
    _write_ply(ply_path, pc.xyz, pc.rgb)

    depth_all = np.concatenate(depth_vals, axis=0) if depth_vals else np.array([], dtype=np.float32)
    conf_all = np.concatenate(conf_vals, axis=0) if conf_vals else np.array([], dtype=np.float32)
    stats = {
        "point_cloud": _pc_stats(xyz),
        "depth": _scalar_stats(depth_all),
        "confidence": _scalar_stats(conf_all) if conf_vals else {"note": "no conf map returned"},
        "conf_threshold": conf_threshold,
        "image_size": [h, w],
        "has_depth": depth is not None,
        "has_depth_conf": depth_conf is not None,
    }
    if extra_stats:
        stats.update(extra_stats)

    return EngineArtifacts(
        engine=engine,
        point_cloud=pc,
        ply_path=str(ply_path),
        depth_images=depth_imgs,
        conf_images=conf_imgs,
        stats=stats,
        wall_time_s=wall_time_s,
    )


def _run_vggt(
    image_paths: Optional[Sequence[str]],
    images_01: Optional[torch.Tensor],
    device: torch.device,
    max_points: int,
    seed: int,
    conf_threshold: float,
    out_dir: Path,
    colorize: bool,
) -> EngineArtifacts:
    t0 = time.time()
    model = _get_vggt_model(device)
    images = _load_vggt_input(image_paths, images_01).to(device)

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
            preds = model(images)

    return _vggt_predictions_to_artifacts(
        engine="vggt",
        preds=preds,
        images=images,
        max_points=max_points,
        seed=seed,
        conf_threshold=conf_threshold,
        out_dir=out_dir,
        colorize=colorize,
        ply_name="vggt_points.ply",
        wall_time_s=time.time() - t0,
    )


def _run_robust_vggt(
    image_paths: Optional[Sequence[str]],
    images_01: Optional[torch.Tensor],
    device: torch.device,
    max_points: int,
    seed: int,
    conf_threshold: float,
    out_dir: Path,
    colorize: bool,
    rejection_threshold: float,
) -> EngineArtifacts:
    t0 = time.time()
    model = _get_vggt_model(device)
    images = _load_vggt_input(image_paths, images_01)
    view_selection = _robust_vggt_view_selection(
        model,
        images,
        device,
        robust_layer_idx=23,
        attention_weight=0.5,
        cosine_weight=0.5,
        rejection_threshold=rejection_threshold,
        anchor_index=0,
    )
    kept_indices = view_selection["kept_indices"]
    robust_images = images[kept_indices].to(device)

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
            preds = model(robust_images)

    return _vggt_predictions_to_artifacts(
        engine="robust_vggt",
        preds=preds,
        images=robust_images,
        max_points=max_points,
        seed=seed,
        conf_threshold=conf_threshold,
        out_dir=out_dir,
        colorize=colorize,
        ply_name="robust_vggt_points.ply",
        wall_time_s=time.time() - t0,
        view_indices=kept_indices,
        extra_stats={"view_selection": view_selection},
    )


def _run_met3r(
    images_01: torch.Tensor,
    device: torch.device,
    recon_backbone: Met3RReconBackbone,
    max_points: int,
    seed: int,
    conf_threshold: float,
    out_dir: Path,
    colorize: bool,
) -> EngineArtifacts:
    t0 = time.time()
    feature_backbone = "dinov2"
    metric = _get_met3r_model(device, recon_backbone=recon_backbone, img_size=MET3R_SIZE)

    k, _, h, w = images_01.shape
    images_norm = (images_01 * 2.0) - 1.0

    if recon_backbone == "raft":
        wall = time.time() - t0
        return EngineArtifacts(
            engine="met3r",
            point_cloud=None,
            ply_path=None,
            depth_images=[],
            conf_images=[],
            stats={"note": "RAFT backbone does not output a point cloud in this demo."},
            wall_time_s=wall,
        )

    with torch.no_grad():
        # Pair every view with view0 to place points in view0 coordinates
        view0 = images_norm[0:1].to(device)
        xyz_list: List[np.ndarray] = []
        rgb_list: List[np.ndarray] = []
        depth_imgs: List[Tuple[np.ndarray, str]] = []
        conf_imgs: List[Tuple[np.ndarray, str]] = []
        depth_vals: List[np.ndarray] = []
        conf_vals: List[np.ndarray] = []

        images_rgb = (images_01.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()  # (K,3,H,W)

        for idx in range(1, k):
            view_i = images_norm[idx : idx + 1].to(device)
            pred0, pred_i = metric.backbone_model({"img": view0, "instance": [""]}, {"img": view_i, "instance": [""]})

            pts0 = pred0["pts3d"][0].detach().cpu()
            pts_i = pred_i["pts3d_in_other_view"][0].detach().cpu()
            conf0 = pred0["conf"][0].detach().cpu()
            conf_i = pred_i["conf"][0].detach().cpu()

            if idx == 1:
                depth0 = _depth_from_points(pts0).numpy()
                depth_vals.append(depth0.reshape(-1))
                depth_imgs.append((_normalize_to_uint8(depth0), "depth view 0"))
                conf0_np = conf0.numpy()
                conf_vals.append(conf0_np.reshape(-1))
                conf_imgs.append((_normalize_to_uint8(conf0_np), "conf view 0"))

                pts0_np = pts0.reshape(-1, 3).numpy()
                rgb0_np = images_rgb[0].transpose(1, 2, 0).reshape(-1, 3)
                mask0 = conf0_np.reshape(-1) >= conf_threshold
                pts0_np = pts0_np[mask0]
                rgb0_np = rgb0_np[mask0]
                finite0 = np.isfinite(pts0_np).all(axis=1)
                xyz_list.append(pts0_np[finite0].astype(np.float32, copy=False))
                rgb_list.append(rgb0_np[finite0].astype(np.uint8, copy=False))

            depth_i = _depth_from_points(pts_i).numpy()
            depth_vals.append(depth_i.reshape(-1))
            depth_imgs.append((_normalize_to_uint8(depth_i), f"depth view {idx}"))
            conf_i_np = conf_i.numpy()
            conf_vals.append(conf_i_np.reshape(-1))
            conf_imgs.append((_normalize_to_uint8(conf_i_np), f"conf view {idx}"))

            pts_i_np = pts_i.reshape(-1, 3).numpy()
            rgb_i_np = images_rgb[idx].transpose(1, 2, 0).reshape(-1, 3)
            mask_i = conf_i_np.reshape(-1) >= conf_threshold
            pts_i_np = pts_i_np[mask_i]
            rgb_i_np = rgb_i_np[mask_i]
            finite_i = np.isfinite(pts_i_np).all(axis=1)
            xyz_list.append(pts_i_np[finite_i].astype(np.float32, copy=False))
            rgb_list.append(rgb_i_np[finite_i].astype(np.uint8, copy=False))

    if not xyz_list:
        wall = time.time() - t0
        return EngineArtifacts(
            engine="met3r",
            point_cloud=None,
            ply_path=None,
            depth_images=[],
            conf_images=[],
            stats={"error": f"MEt3R({recon_backbone}) produced no 3D points."},
            wall_time_s=wall,
        )

    xyz = np.concatenate(xyz_list, axis=0)
    rgb = np.concatenate(rgb_list, axis=0) if colorize else None
    pc_full = PointCloud(xyz=xyz, rgb=rgb)
    pc = _subsample_points(pc_full.xyz, pc_full.rgb, max_points=max_points, seed=seed)

    ply_path = out_dir / f"met3r_{recon_backbone}_{feature_backbone}_points.ply"
    _write_ply(ply_path, pc.xyz, pc.rgb)

    depth_all = np.concatenate(depth_vals, axis=0) if depth_vals else np.array([], dtype=np.float32)
    conf_all = np.concatenate(conf_vals, axis=0) if conf_vals else np.array([], dtype=np.float32)
    stats = {
        "recon_backbone": recon_backbone,
        "feature_backbone": "dinov2",
        "point_cloud": _pc_stats(xyz),
        "depth": _scalar_stats(depth_all),
        "confidence": _scalar_stats(conf_all),
        "conf_threshold": conf_threshold,
        "image_size": [h, w],
        "note": "Points are expressed in view0 coordinates via pairwise alignment (0,i).",
    }

    wall = time.time() - t0
    return EngineArtifacts(
        engine="met3r",
        point_cloud=pc,
        ply_path=str(ply_path),
        depth_images=depth_imgs,
        conf_images=conf_imgs,
        stats=stats,
        wall_time_s=wall,
    )


def build_demo(args: argparse.Namespace) -> Any:
    import gradio as gr

    device = _device()
    logger.info(f"Using device: {device}")
    if device.type != "cuda":
        logger.warning("CUDA is not available; running the demo on CPU.")
    dataset_root_override = _normalize_dataset_root_override(getattr(args, "dataset_root_override", None))
    met3r_available = _has_met3r_recon_backends()
    if not met3r_available:
        logger.warning("MASt3R/DUSt3R reconstruction will stay disabled until a MASt3R checkout is available.")

    out_root = ROOT / "output" / "gradio_compare"
    out_root.mkdir(parents=True, exist_ok=True)

    def _run(
        input_mode: InputMode,
        upload_files: Sequence[Any],
        subset_size: int,
        seed: int,
        benchmark_sample_kind: BenchmarkSampleKind,
        benchmark_sample_id: str,
        met3r_recon_backbone: Met3RReconBackbone,
        run_met3r: bool,
        run_fast3r: bool,
        run_vggt: bool,
        run_robust_vggt: bool,
        robust_vggt_rejection_threshold: float,
        max_points: int,
        conf_threshold: float,
        colorize: bool,
    ):
        t0 = time.time()
        subset_size = int(subset_size)
        seed = int(seed)
        max_points = int(max_points)
        data_noise_type: NoiseType = "Uniform"
        data_gaussian_sigma = 0.2
        data_salt_pepper_prob = 0.05
        data_constant_value = 0.5
        data_patch_ratio = 0.25
        data_num_patches = 4
        data_patch_seed = seed
        data_noise_seed = seed

        run_dir = out_root / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        paths: Optional[List[str]] = None
        display_images: List[Tuple[np.ndarray, str]] = []
        view_labels: List[str] = []
        benchmark_sample: Optional[BenchmarkSampleSpec] = None

        if input_mode == "Upload images":
            paths = _resolve_gradio_files(upload_files)
            if len(paths) > subset_size:
                paths = paths[:subset_size]
            subset_size = len(paths)
        elif input_mode == "SysCON3D benchmark sample":
            benchmark_sample = _load_benchmark_sample(
                Path(args.impossible_splits_json),
                sample_kind=benchmark_sample_kind,
                sample_id=benchmark_sample_id,
                dataset_root_override=dataset_root_override,
                calibration_splits_json=Path(args.splits_json),
                subset_size=subset_size,
            )
            if benchmark_sample.subset_size != subset_size:
                raise ValueError(
                    f"{benchmark_sample.kind} sample_id={benchmark_sample.sample_id} has subset_size="
                    f"{benchmark_sample.subset_size}, but UI subset_size={subset_size}."
                )
            if benchmark_sample.image_paths:
                paths = list(benchmark_sample.image_paths)
                view_labels = [f"view {i} — {scene_name}" for i, scene_name in enumerate(benchmark_sample.scene_names)]
            if benchmark_sample.noise_type is not None and not benchmark_sample.materialized:
                paths = None
                data_noise_type = benchmark_sample.noise_type
                if benchmark_sample.seed is not None:
                    data_noise_seed = benchmark_sample.seed
            if benchmark_sample.gaussian_sigma is not None:
                data_gaussian_sigma = benchmark_sample.gaussian_sigma
            if benchmark_sample.patch_ratio is not None:
                data_patch_ratio = benchmark_sample.patch_ratio
            if benchmark_sample.patch_num_patches is not None:
                data_num_patches = benchmark_sample.patch_num_patches
            if benchmark_sample.patch_seed is not None:
                data_patch_seed = benchmark_sample.patch_seed
        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        if paths is not None and len(paths) < 2:
            raise ValueError("Need at least 2 views (images) for reconstruction.")
        use_noise_inputs = (
            benchmark_sample is not None
            and benchmark_sample.noise_type is not None
            and not benchmark_sample.materialized
        )
        use_patched_inputs = (
            benchmark_sample is not None
            and benchmark_sample.patch_ratio is not None
            and not benchmark_sample.materialized
        )

        if paths is not None:
            disp = _load_images_square(paths, target_size=256)
            if use_patched_inputs:
                disp = _add_gaussian_patches(
                    disp,
                    patch_ratio=data_patch_ratio,
                    num_patches=data_num_patches,
                    seed=data_patch_seed,
                    gaussian_sigma=data_gaussian_sigma,
                )
            for idx, img in enumerate((disp.clamp(0, 1) * 255.0).to(torch.uint8).numpy()):
                label = view_labels[idx] if view_labels else f"view {idx}"
                display_images.append((img.transpose(1, 2, 0), label))
        else:
            disp = _generate_noise_images(
                num_images=subset_size,
                size=256,
                seed=data_noise_seed,
                device=torch.device("cpu"),
                noise_type=data_noise_type,
                gaussian_sigma=data_gaussian_sigma,
                salt_pepper_prob=data_salt_pepper_prob,
                constant_value=data_constant_value,
            )
            for idx, img in enumerate((disp.clamp(0, 1) * 255.0).to(torch.uint8).numpy()):
                display_images.append((img.transpose(1, 2, 0), f"view {idx}"))

        artifacts: Dict[EngineName, EngineArtifacts] = {}

        if run_met3r:
            if use_noise_inputs:
                met3r_imgs = _generate_noise_images(
                    subset_size,
                    MET3R_SIZE,
                    seed=data_noise_seed,
                    device=device,
                    noise_type=data_noise_type,
                    gaussian_sigma=data_gaussian_sigma,
                    salt_pepper_prob=data_salt_pepper_prob,
                    constant_value=data_constant_value,
                )
            elif use_patched_inputs:
                if paths is None:
                    raise ValueError("Patched inputs require image paths.")
                met3r_imgs = _load_images_square(paths, target_size=MET3R_SIZE).to(device)
                met3r_imgs = _add_gaussian_patches(
                    met3r_imgs,
                    patch_ratio=data_patch_ratio,
                    num_patches=data_num_patches,
                    seed=data_patch_seed,
                    gaussian_sigma=data_gaussian_sigma,
                )
            else:
                if paths is None:
                    raise ValueError("Image-backed inputs require paths.")
                met3r_imgs = _load_images_square(paths, target_size=MET3R_SIZE).to(device)
            artifacts["met3r"] = _run_met3r(
                images_01=met3r_imgs,
                device=device,
                recon_backbone=met3r_recon_backbone,
                max_points=max_points,
                seed=seed,
                conf_threshold=conf_threshold,
                out_dir=run_dir,
                colorize=colorize,
            )

        if run_fast3r:
            if use_noise_inputs:
                fast3r_imgs = _generate_noise_images(
                    subset_size,
                    FAST3R_SIZE,
                    seed=data_noise_seed,
                    device=device,
                    noise_type=data_noise_type,
                    gaussian_sigma=data_gaussian_sigma,
                    salt_pepper_prob=data_salt_pepper_prob,
                    constant_value=data_constant_value,
                )
            elif use_patched_inputs:
                if paths is None:
                    raise ValueError("Patched inputs require image paths.")
                fast3r_imgs = _load_images_square(paths, target_size=FAST3R_SIZE).to(device)
                fast3r_imgs = _add_gaussian_patches(
                    fast3r_imgs,
                    patch_ratio=data_patch_ratio,
                    num_patches=data_num_patches,
                    seed=data_patch_seed,
                    gaussian_sigma=data_gaussian_sigma,
                )
            else:
                if paths is None:
                    raise ValueError("Image-backed inputs require paths.")
                fast3r_imgs = _load_images_square(paths, target_size=FAST3R_SIZE).to(device)
            artifacts["fast3r"] = _run_fast3r(
                images_01=fast3r_imgs,
                device=device,
                max_points=max_points,
                seed=seed,
                conf_threshold=conf_threshold,
                out_dir=run_dir,
                colorize=colorize,
            )

        def _run_vggt_family(runner: Any) -> EngineArtifacts:
            if use_noise_inputs:
                vggt_imgs = _generate_noise_images(
                    subset_size,
                    VGGT_SIZE,
                    seed=data_noise_seed,
                    device=device,
                    noise_type=data_noise_type,
                    gaussian_sigma=data_gaussian_sigma,
                    salt_pepper_prob=data_salt_pepper_prob,
                    constant_value=data_constant_value,
                )
                return runner(
                    image_paths=None,
                    images_01=vggt_imgs,
                    device=device,
                    max_points=max_points,
                    seed=seed,
                    conf_threshold=conf_threshold,
                    out_dir=run_dir,
                    colorize=colorize,
                )
            if use_patched_inputs:
                if paths is None:
                    raise ValueError("Patched inputs require image paths.")
                vggt_imgs = _load_images_square(paths, target_size=VGGT_SIZE).to(device)
                vggt_imgs = _add_gaussian_patches(
                    vggt_imgs,
                    patch_ratio=data_patch_ratio,
                    num_patches=data_num_patches,
                    seed=data_patch_seed,
                    gaussian_sigma=data_gaussian_sigma,
                )
                return runner(
                    image_paths=None,
                    images_01=vggt_imgs,
                    device=device,
                    max_points=max_points,
                    seed=seed,
                    conf_threshold=conf_threshold,
                    out_dir=run_dir,
                    colorize=colorize,
                )
            if paths is None:
                raise ValueError("Image-backed inputs require paths.")
            return runner(
                image_paths=paths,
                images_01=None,
                device=device,
                max_points=max_points,
                seed=seed,
                conf_threshold=conf_threshold,
                out_dir=run_dir,
                colorize=colorize,
            )

        if run_vggt:
            artifacts["vggt"] = _run_vggt_family(_run_vggt)

        if run_robust_vggt:
            artifacts["robust_vggt"] = _run_vggt_family(
                lambda **kwargs: _run_robust_vggt(
                    **kwargs,
                    rejection_threshold=robust_vggt_rejection_threshold,
                )
            )

        compare_plot = _plot_compare(artifacts)
        input_md = f"### Input\n- mode: {input_mode}\n- K: {subset_size}\n- run seed: {seed}"
        if input_mode == "SysCON3D benchmark sample" and benchmark_sample is not None:
            input_md += f"\n- scene type: {_pretty_benchmark_kind(benchmark_sample.kind)}"
            input_md += f"\n- sample_id: {benchmark_sample.sample_id}"
            if benchmark_sample.scene_names:
                counts = {
                    scene_name: benchmark_sample.scene_names.count(scene_name)
                    for scene_name in sorted(set(benchmark_sample.scene_names))
                }
                input_md += f"\n- scene counts: {counts}"
                outlier_candidates = [scene_name for scene_name, count in counts.items() if count == 1]
                if benchmark_sample.kind == "mixed_one_outlier" and len(outlier_candidates) == 1:
                    input_md += f"\n- inferred outlier scene: {outlier_candidates[0]}"
            if benchmark_sample.noise_type is not None:
                input_md += f"\n- synthetic seed: {data_noise_seed}"
                input_md += f"\n- noise type: {data_noise_type}"
                if data_noise_type == "Gaussian":
                    input_md += f"\n- gaussian sigma: {data_gaussian_sigma:g}"
            elif benchmark_sample.gaussian_sigma is not None:
                input_md += f"\n- gaussian sigma: {benchmark_sample.gaussian_sigma:g}"
            if benchmark_sample.patch_ratio is not None:
                input_md += f"\n- patch ratio: {data_patch_ratio:g}"
                input_md += f"\n- patches per image: {data_num_patches}"
                input_md += f"\n- patch seed: {data_patch_seed}"
        if run_robust_vggt:
            input_md += f"\n- RobustVGGT rejection threshold: {robust_vggt_rejection_threshold:g}"

        summary_md = input_md + "\n\n" + _stats_markdown(artifacts)

        def _engine_tab(engine: EngineName, title: str, solid_color: str):
            art = artifacts.get(engine)
            if art is None or art.point_cloud is None:
                return None, None, _json_pretty({"note": f"{engine} not run or produced no point cloud."}), [], []
            fig = _plot_point_cloud(
                art.point_cloud,
                title=title,
                color_mode="rgb" if colorize else "solid",
                solid_color=solid_color,
            )
            return fig, art.ply_path, _json_pretty(art.stats), art.depth_images, art.conf_images

        met3r_fig, met3r_ply, met3r_stats, met3r_depth, met3r_conf = _engine_tab(
            "met3r", "MASt3R/DUSt3R Point Cloud (via MEt3R)", "#ef4444"
        )
        fast3r_fig, fast3r_ply, fast3r_stats, fast3r_depth, fast3r_conf = _engine_tab(
            "fast3r", "Fast3R Point Cloud (RGB)", "#22c55e"
        )
        vggt_fig, vggt_ply, vggt_stats, vggt_depth, vggt_conf = _engine_tab(
            "vggt", "VGGT Point Cloud (RGB)", "#3b82f6"
        )
        robust_vggt_fig, robust_vggt_ply, robust_vggt_stats, robust_vggt_depth, robust_vggt_conf = _engine_tab(
            "robust_vggt", "RobustVGGT Point Cloud (RGB)", "#f59e0b"
        )

        elapsed = time.time() - t0
        run_note = f"Run saved in `{run_dir}` (elapsed: {elapsed:.2f}s)."
        return (
            display_images,
            compare_plot,
            summary_md,
            run_note,
            met3r_fig,
            met3r_ply,
            met3r_stats,
            met3r_depth,
            met3r_conf,
            fast3r_fig,
            fast3r_ply,
            fast3r_stats,
            fast3r_depth,
            fast3r_conf,
            vggt_fig,
            vggt_ply,
            vggt_stats,
            vggt_depth,
            vggt_conf,
            robust_vggt_fig,
            robust_vggt_ply,
            robust_vggt_stats,
            robust_vggt_depth,
            robust_vggt_conf,
        )

    def _update_benchmark_sample_dropdown(sample_kind: BenchmarkSampleKind, k: int):
        ids = _benchmark_sample_ids(
            Path(args.impossible_splits_json),
            sample_kind=sample_kind,
            subset_size=int(k),
            calibration_splits_json=Path(args.splits_json),
        )
        return gr.update(choices=ids, value=(ids[0] if ids else None))

    theme = gr.themes.Ocean()
    demo_title = "Robustness Analysis of 3D Recon Backbones"
    with gr.Blocks(theme=theme, title=demo_title) as demo:
        gr.Markdown(
            f"""
            # {demo_title}

            Compare reconstruction artifacts from **MASt3R/DUSt3R / Fast3R / VGGT / RobustVGGT** on the same multi-view input.
            The point clouds are interactive (Plotly) and can be exported as `.ply` from `output/gradio_compare/<timestamp>/`.

            **Tip (remote):**
            ```bash
            # on a GPU node
            python demo_gradio_compare.py \
              --server-name 127.0.0.1 \
              --server-port 7860 \
              --splits-json tmp/syscon3d_release/mipnerf360_calibration_splits.json \
              --impossible-splits-json tmp/syscon3d_release/mipnerf360_impossible_splits.json

            # on your laptop (use your cluster login or a ProxyJump if needed)
            ssh -L 7860:127.0.0.1:7860 <user>@<host-or-node>
            ```
            Then open `http://127.0.0.1:7860` in your browser.
            """
        )
        if not met3r_available:
            gr.Markdown(
                "MASt3R/DUSt3R reconstruction is disabled because no MASt3R checkout was found. "
                "Fast3R, VGGT, and RobustVGGT still run normally."
            )

        with gr.Row():
            with gr.Column(scale=2):
                input_mode = gr.Radio(
                    choices=[
                        "Upload images",
                        "SysCON3D benchmark sample",
                    ],
                    value="SysCON3D benchmark sample",
                    label="Input mode",
                )
                upload_files = gr.File(file_count="multiple", label="Upload images (multi-view)")
                subset_size = gr.Slider(minimum=3, maximum=21, step=3, value=9, label="Number of views (K)")
                seed = gr.Number(value=42, precision=0, label="Seed")
                benchmark_sample_kind = gr.Dropdown(
                    choices=list(BENCHMARK_SAMPLE_KINDS),
                    value="consistent",
                    label="SysCON3D scene type",
                )
                benchmark_ids = _benchmark_sample_ids(
                    Path(args.impossible_splits_json),
                    sample_kind="consistent",
                    subset_size=9,
                    calibration_splits_json=Path(args.splits_json),
                )
                benchmark_sample_id = gr.Dropdown(
                    choices=benchmark_ids,
                    value=(benchmark_ids[0] if benchmark_ids else None),
                    label="SysCON3D sample id",
                    allow_custom_value=False,
                )
                gr.Markdown(
                    "`sample id` is populated from the selected SysCON3D scene type and K. "
                    "If it is blank or stale at run time, the demo uses the first valid sample."
                )

                subset_size.change(
                    fn=_update_benchmark_sample_dropdown,
                    inputs=[benchmark_sample_kind, subset_size],
                    outputs=[benchmark_sample_id],
                )
                benchmark_sample_kind.change(
                    fn=_update_benchmark_sample_dropdown,
                    inputs=[benchmark_sample_kind, subset_size],
                    outputs=[benchmark_sample_id],
                )

                gr.Markdown("### MASt3R/DUSt3R options")
                met3r_recon_backbone = gr.Dropdown(
                    choices=["mast3r", "dust3r", "raft"],
                    value="mast3r",
                    label="Recon backbone (via MEt3R)",
                    interactive=met3r_available,
                )
                gr.Markdown("Feature backbone is fixed to `dinov2` for point cloud reconstruction in this demo.")

                gr.Markdown("### Run options")
                with gr.Row():
                    run_met3r = gr.Checkbox(
                        value=met3r_available,
                        label="Run MASt3R/DUSt3R",
                        interactive=met3r_available,
                    )
                    run_fast3r = gr.Checkbox(value=True, label="Run Fast3R")
                    run_vggt = gr.Checkbox(value=True, label="Run VGGT")
                    run_robust_vggt = gr.Checkbox(value=False, label="Run RobustVGGT")
                robust_vggt_rejection_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.4,
                    label="RobustVGGT rejection threshold",
                )
                max_points = gr.Slider(
                    minimum=1000,
                    maximum=300000,
                    step=1000,
                    value=100000,
                    label="Max points (plot/PLY subsample; per engine)",
                )
                gr.Markdown(
                    "Note: `max points` is a visualization/export subsample to keep the browser responsive; "
                    "increase if you want denser plots (millions of points may freeze the tab)."
                )
                conf_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Confidence threshold")
                colorize = gr.Checkbox(value=True, label="Colorize points by RGB")

                run_btn = gr.Button("Run reconstruction", variant="primary")

            with gr.Column(scale=3):
                input_gallery = gr.Gallery(label="Input views", columns=4, height=260, preview=True, object_fit="contain")

                with gr.Tab("Compare"):
                    compare_plot = gr.Plot(label="Compare point clouds")
                    compare_stats = gr.Markdown()
                    run_note = gr.Markdown()

                with gr.Tab("MASt3R/DUSt3R"):
                    met3r_plot = gr.Plot()
                    met3r_ply = gr.File(label="Point cloud (.ply)")
                    met3r_stats = gr.Code(label="Stats (json)", language="json")
                    met3r_depth = gr.Gallery(label="Depth (proxy)", columns=4, height=220, preview=True)
                    met3r_conf = gr.Gallery(label="Confidence", columns=4, height=220, preview=True)

                with gr.Tab("Fast3R"):
                    fast3r_plot = gr.Plot()
                    fast3r_ply = gr.File(label="Point cloud (.ply)")
                    fast3r_stats = gr.Code(label="Stats (json)", language="json")
                    fast3r_depth = gr.Gallery(label="Depth (proxy)", columns=4, height=220, preview=True)
                    fast3r_conf = gr.Gallery(label="Confidence", columns=4, height=220, preview=True)

                with gr.Tab("VGGT"):
                    vggt_plot = gr.Plot()
                    vggt_ply = gr.File(label="Point cloud (.ply)")
                    vggt_stats = gr.Code(label="Stats (json)", language="json")
                    vggt_depth = gr.Gallery(label="Depth (proxy)", columns=4, height=220, preview=True)
                    vggt_conf = gr.Gallery(label="Confidence", columns=4, height=220, preview=True)

                with gr.Tab("RobustVGGT"):
                    robust_vggt_plot = gr.Plot()
                    robust_vggt_ply = gr.File(label="Point cloud (.ply)")
                    robust_vggt_stats = gr.Code(label="Stats (json)", language="json")
                    robust_vggt_depth = gr.Gallery(label="Depth (proxy)", columns=4, height=220, preview=True)
                    robust_vggt_conf = gr.Gallery(label="Confidence", columns=4, height=220, preview=True)

        run_btn.click(
            fn=_run,
            inputs=[
                input_mode,
                upload_files,
                subset_size,
                seed,
                benchmark_sample_kind,
                benchmark_sample_id,
                met3r_recon_backbone,
                run_met3r,
                run_fast3r,
                run_vggt,
                run_robust_vggt,
                robust_vggt_rejection_threshold,
                max_points,
                conf_threshold,
                colorize,
            ],
            outputs=[
                input_gallery,
                compare_plot,
                compare_stats,
                run_note,
                met3r_plot,
                met3r_ply,
                met3r_stats,
                met3r_depth,
                met3r_conf,
                fast3r_plot,
                fast3r_ply,
                fast3r_stats,
                fast3r_depth,
                fast3r_conf,
                vggt_plot,
                vggt_ply,
                vggt_stats,
                vggt_depth,
                vggt_conf,
                robust_vggt_plot,
                robust_vggt_ply,
                robust_vggt_stats,
                robust_vggt_depth,
                robust_vggt_conf,
            ],
            api_name=False,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio demo to compare MASt3R/DUSt3R / Fast3R / VGGT / RobustVGGT recon outputs"
    )
    parser.add_argument("--splits-json", type=str, default="tmp/syscon3d_release/mipnerf360_calibration_splits.json")
    parser.add_argument(
        "--impossible-splits-json",
        type=str,
        default="tmp/syscon3d_release/mipnerf360_impossible_splits.json",
    )
    parser.add_argument(
        "--dataset-root-override",
        type=str,
        default=None,
        help="Optional dataset root that overrides `dataset_root` inside both split manifests.",
    )
    parser.add_argument(
        "--syscon3d-extra-data-root",
        type=str,
        default="",
        help="Additional root for deterministic SysCON3D scene-type files referenced by the impossible manifest.",
    )
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="tmp/gradio_compare",
        help="Writable temp dir used by Gradio.",
    )
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    if not tmp_dir.is_absolute():
        tmp_dir = ROOT / tmp_dir
    _configure_tmpdir(tmp_dir)
    _configure_syscon3d_extra_data_root(args.syscon3d_extra_data_root)

    demo = build_demo(args)
    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
