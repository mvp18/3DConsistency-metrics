#!/usr/bin/env python3
"""Generate prerecorded SysCON3D reconstruction assets for the project page."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "tmp" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "tmp" / "torch"))

import demo_gradio_compare as demo

logger = logging.getLogger(__name__)

BACKBONES = ("mast3r", "dust3r", "fast3r", "vggt", "robust_vggt")
DEFAULT_SUBSET_SIZES = (3, 6, 9, 12)


@dataclass(frozen=True)
class AssetTask:
    index: int
    scene_type: str
    sample_id: str
    subset_size: int
    backbone: str

    @property
    def run_id(self) -> str:
        return f"{self.scene_type}/k{self.subset_size:02d}/{self.sample_id}/{self.backbone}"


def _as_root_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_tasks(
    *,
    impossible_splits_json: Path,
    calibration_splits_json: Path,
    subset_sizes: Sequence[int],
    backbones: Sequence[str],
    task_limit: int,
) -> list[AssetTask]:
    sample_tasks: list[AssetTask] = []
    for scene_type in demo.BENCHMARK_SAMPLE_KINDS:
        for subset_size in subset_sizes:
            sample_ids = demo._benchmark_sample_ids(
                impossible_splits_json,
                sample_kind=scene_type,
                subset_size=int(subset_size),
                calibration_splits_json=calibration_splits_json,
            )
            for sample_id in sample_ids:
                for backbone in backbones:
                    sample_tasks.append(
                        AssetTask(
                            index=len(sample_tasks),
                            scene_type=scene_type,
                            sample_id=sample_id,
                            subset_size=int(subset_size),
                            backbone=backbone,
                        )
                    )
    if task_limit > 0:
        tasks = sample_tasks[:task_limit]
    else:
        backbone_order = {backbone: idx for idx, backbone in enumerate(backbones)}
        tasks = sorted(
            sample_tasks,
            key=lambda task: (
                backbone_order[task.backbone],
                task.scene_type,
                task.subset_size,
                task.sample_id,
            ),
        )
    return [
        AssetTask(
            index=idx,
            scene_type=task.scene_type,
            sample_id=task.sample_id,
            subset_size=task.subset_size,
            backbone=task.backbone,
        )
        for idx, task in enumerate(tasks)
    ]


def _task_output_dir(output_root: Path, task: AssetTask) -> Path:
    return output_root / "runs" / task.scene_type / f"k{task.subset_size:02d}" / task.sample_id / task.backbone


def _save_input_thumbnails(paths: Sequence[str], scene_names: Sequence[str], output_dir: Path) -> None:
    from PIL import Image

    thumb_dir = output_dir / "inputs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, path_text in enumerate(paths):
        src = Path(path_text)
        image = Image.open(src).convert("RGB")
        image.thumbnail((224, 224))
        out_path = thumb_dir / f"view_{idx:02d}.jpg"
        image.save(out_path, quality=92)
        rows.append(
            {
                "index": idx,
                "scene": scene_names[idx] if idx < len(scene_names) else "",
                "source_path": str(src),
                "thumbnail": str(out_path.relative_to(output_dir)),
            }
        )
    _safe_write_json(output_dir / "input_manifest.json", rows)


def _colors_rgba(rgb: Optional[np.ndarray], n: int) -> np.ndarray:
    if rgb is None:
        colors = np.full((n, 4), 255, dtype=np.uint8)
        colors[:, :3] = np.array([31, 58, 95], dtype=np.uint8)
        return colors
    if rgb.shape[1] == 4:
        return rgb.astype(np.uint8, copy=False)
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb.astype(np.uint8, copy=False), alpha], axis=1)


def _write_glb(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray]) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = _colors_rgba(rgb, xyz.shape[0])
    cloud = trimesh.points.PointCloud(vertices=xyz.astype(np.float32, copy=False), colors=colors)
    cloud.export(path)


def _sample_points(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
    idx.sort()
    return xyz[idx], None if rgb is None else rgb[idx]


def _axis_limits(xyz: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    finite = np.isfinite(xyz).all(axis=1)
    pts = xyz[finite]
    if pts.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    center = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    radius = span / 2.0
    return tuple((float(c - radius), float(c + radius)) for c in center)  # type: ignore[return-value]


def _write_turntable_video(
    path: Path,
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    *,
    fps: int,
    duration_s: float,
    image_size: int,
    max_points: int,
    seed: int,
    background_color: str,
    point_size: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, cols = _sample_points(xyz, rgb, max_points=max_points, seed=seed)
    colors = None if cols is None else cols.astype(np.float32) / 255.0
    xlim, ylim, zlim = _axis_limits(pts)
    frames = max(1, int(round(float(duration_s) * int(fps))))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors if colors is not None else "#1f3a5f",
        s=float(point_size),
        alpha=1.0,
        depthshade=False,
        linewidths=0,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    with imageio.get_writer(
        path,
        fps=int(fps),
        codec="libx264",
        output_params=["-crf", "23", "-preset", "veryfast"],
        ffmpeg_log_level="error",
    ) as writer:
        for frame_idx in range(frames):
            azim = 360.0 * frame_idx / frames
            ax.view_init(elev=18.0, azim=azim)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(frame)
    plt.close(fig)


def _offload_models_to_cpu() -> None:
    if demo._FAST3R_MODEL is not None:
        demo._FAST3R_MODEL = demo._FAST3R_MODEL.to(torch.device("cpu"))
    if demo._VGGT_MODEL is not None:
        demo._VGGT_MODEL = demo._VGGT_MODEL.to(torch.device("cpu"))
    for key, model in list(demo._MET3R_CACHE.items()):
        demo._MET3R_CACHE[key] = model.to(torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_runtime_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_family(backbone: str) -> str:
    if backbone in {"vggt", "robust_vggt"}:
        return "vggt"
    return backbone


def _run_backbone(
    *,
    task: AssetTask,
    image_paths: Sequence[str],
    device: torch.device,
    output_dir: Path,
    max_points: int,
    conf_threshold: float,
    colorize: bool,
    robust_vggt_rejection_threshold: float,
) -> demo.EngineArtifacts:
    if task.backbone == "mast3r":
        images = demo._load_images_square(image_paths, target_size=demo.MET3R_SIZE).to(device)
        return demo._run_met3r(
            images_01=images,
            device=device,
            recon_backbone="mast3r",
            max_points=max_points,
            seed=task.index,
            conf_threshold=conf_threshold,
            out_dir=output_dir,
            colorize=colorize,
        )
    if task.backbone == "dust3r":
        images = demo._load_images_square(image_paths, target_size=demo.MET3R_SIZE).to(device)
        return demo._run_met3r(
            images_01=images,
            device=device,
            recon_backbone="dust3r",
            max_points=max_points,
            seed=task.index,
            conf_threshold=conf_threshold,
            out_dir=output_dir,
            colorize=colorize,
        )
    if task.backbone == "fast3r":
        images = demo._load_images_square(image_paths, target_size=demo.FAST3R_SIZE).to(device)
        return demo._run_fast3r(
            images_01=images,
            device=device,
            max_points=max_points,
            seed=task.index,
            conf_threshold=conf_threshold,
            out_dir=output_dir,
            colorize=colorize,
        )
    if task.backbone == "vggt":
        return demo._run_vggt(
            image_paths=image_paths,
            images_01=None,
            device=device,
            max_points=max_points,
            seed=task.index,
            conf_threshold=conf_threshold,
            out_dir=output_dir,
            colorize=colorize,
        )
    if task.backbone == "robust_vggt":
        return demo._run_robust_vggt(
            image_paths=image_paths,
            images_01=None,
            device=device,
            max_points=max_points,
            seed=task.index,
            conf_threshold=conf_threshold,
            out_dir=output_dir,
            colorize=colorize,
            rejection_threshold=robust_vggt_rejection_threshold,
        )
    raise ValueError(f"Unknown backbone: {task.backbone}")


def _run_task(args: argparse.Namespace, task: AssetTask, device: torch.device) -> bool:
    output_root = _as_root_path(args.output_root)
    run_dir = _task_output_dir(output_root, task)
    done_path = run_dir / "done.json"
    error_path = run_dir / "error.json"
    if args.skip_existing and done_path.exists():
        logger.info("Skipping existing task %s", task.run_id)
        return True

    if run_dir.exists() and not done_path.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.write_videos:
        for stale_video in run_dir.glob("*_turntable.mp4"):
            stale_video.unlink()
    if not args.keep_ply:
        for stale_ply in run_dir.glob("*.ply"):
            stale_ply.unlink()

    sample = demo._load_benchmark_sample(
        _as_root_path(args.impossible_splits_json),
        sample_kind=task.scene_type,
        sample_id=task.sample_id,
        dataset_root_override=args.dataset_root_override or None,
        calibration_splits_json=_as_root_path(args.splits_json),
        subset_size=task.subset_size,
    )
    if not sample.image_paths:
        raise ValueError(f"Task {task.run_id} did not resolve image-backed inputs.")

    image_paths = list(sample.image_paths)
    scene_names = list(sample.scene_names)
    _safe_write_json(run_dir / "task.json", asdict(task))
    _save_input_thumbnails(image_paths, scene_names, run_dir)

    t0 = time.time()
    try:
        artifact = _run_backbone(
            task=task,
            image_paths=image_paths,
            device=device,
            output_dir=run_dir,
            max_points=int(args.max_points),
            conf_threshold=float(args.conf_threshold),
            colorize=bool(args.colorize),
            robust_vggt_rejection_threshold=float(args.robust_vggt_rejection_threshold),
        )
        ply_path = Path(artifact.ply_path) if artifact.ply_path else None
        if ply_path is not None and not args.keep_ply:
            ply_path.unlink(missing_ok=True)

        stats = {
            "task": asdict(task),
            "scene_names": scene_names,
            "stats": artifact.stats,
            "wall_time_s": float(artifact.wall_time_s),
            "total_time_s": float(time.time() - t0),
            "ply_path": str(ply_path) if ply_path is not None and args.keep_ply else None,
        }

        if artifact.point_cloud is not None:
            glb_path = run_dir / f"{task.backbone}.glb"
            _write_glb(glb_path, artifact.point_cloud.xyz, artifact.point_cloud.rgb)
            stats["glb_path"] = str(glb_path)
            if args.write_videos:
                video_path = run_dir / f"{task.backbone}_turntable.mp4"
                _write_turntable_video(
                    video_path,
                    artifact.point_cloud.xyz,
                    artifact.point_cloud.rgb,
                    fps=int(args.video_fps),
                    duration_s=float(args.video_duration_s),
                    image_size=int(args.video_size),
                    max_points=int(args.video_max_points),
                    seed=task.index,
                    background_color=str(args.video_background),
                    point_size=float(args.video_point_size),
                )
                stats["video_path"] = str(video_path)

        _safe_write_json(run_dir / "stats.json", stats)
        _safe_write_json(
            done_path,
            {
                "task": asdict(task),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ok" if artifact.point_cloud is not None else "no_point_cloud",
            },
        )
        if error_path.exists():
            error_path.unlink()
        logger.info("Completed %s", task.run_id)
        return True
    except Exception as exc:
        logger.exception("Failed %s", task.run_id)
        _safe_write_json(
            error_path,
            {
                "task": asdict(task),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": repr(exc),
            },
        )
        if args.fail_fast:
            raise
        return False
    finally:
        _clear_runtime_cache()


def _write_manifest(output_root: Path, tasks: Sequence[AssetTask]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), sort_keys=True) + "\n")

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.scene_type] = counts.get(task.scene_type, 0) + 1
    _safe_write_json(
        output_root / "manifest_summary.json",
        {
            "num_tasks": len(tasks),
            "num_reconstruction_samples": len({(t.scene_type, t.sample_id) for t in tasks}),
            "backbones": sorted(set(t.backbone for t in tasks)),
            "subset_sizes": sorted(set(t.subset_size for t in tasks)),
            "tasks_by_scene_type": counts,
        },
    )


def _iter_worker_tasks(tasks: Sequence[AssetTask], worker_index: int, num_workers: int) -> Iterable[AssetTask]:
    for task in tasks:
        if task.index % num_workers == worker_index:
            yield task


def _worker(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | worker=%(process)d | %(message)s",
    )
    demo._configure_tmpdir(_as_root_path(args.tmp_dir))
    demo._configure_syscon3d_extra_data_root(args.syscon3d_extra_data_root)

    tasks = _load_tasks(
        impossible_splits_json=_as_root_path(args.impossible_splits_json),
        calibration_splits_json=_as_root_path(args.splits_json),
        subset_sizes=args.subset_sizes,
        backbones=args.backbones,
        task_limit=int(args.task_limit),
    )
    output_root = _as_root_path(args.output_root)
    if args.worker_index == 0:
        _write_manifest(output_root, tasks)

    device = demo._device()
    logger.info(
        "Starting worker %d/%d on device=%s with %d total tasks",
        args.worker_index,
        args.num_workers,
        device,
        len(tasks),
    )
    ok = True
    active_family: Optional[str] = None
    for task in _iter_worker_tasks(tasks, int(args.worker_index), int(args.num_workers)):
        task_family = _model_family(task.backbone)
        if active_family is not None and task_family != active_family:
            logger.info("Switching model family %s -> %s; offloading cached models to CPU", active_family, task_family)
            _offload_models_to_cpu()
        active_family = task_family
        ok = _run_task(args, task, device) and ok
    _offload_models_to_cpu()
    return 0 if ok else 1


def _launch(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    output_root = _as_root_path(args.output_root)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks(
        impossible_splits_json=_as_root_path(args.impossible_splits_json),
        calibration_splits_json=_as_root_path(args.splits_json),
        subset_sizes=args.subset_sizes,
        backbones=args.backbones,
        task_limit=int(args.task_limit),
    )
    _write_manifest(output_root, tasks)

    script_path = Path(__file__).resolve()
    base_cmd = [
        sys.executable,
        str(script_path),
        "--splits-json",
        args.splits_json,
        "--impossible-splits-json",
        args.impossible_splits_json,
        "--output-root",
        args.output_root,
        "--tmp-dir",
        args.tmp_dir,
        "--syscon3d-extra-data-root",
        args.syscon3d_extra_data_root,
        "--subset-sizes",
        *[str(k) for k in args.subset_sizes],
        "--backbones",
        *args.backbones,
        "--max-points",
        str(args.max_points),
        "--video-max-points",
        str(args.video_max_points),
        "--video-duration-s",
        str(args.video_duration_s),
        "--video-fps",
        str(args.video_fps),
        "--video-size",
        str(args.video_size),
        "--video-background",
        str(args.video_background),
        "--video-point-size",
        str(args.video_point_size),
        "--conf-threshold",
        str(args.conf_threshold),
        "--robust-vggt-rejection-threshold",
        str(args.robust_vggt_rejection_threshold),
        "--task-limit",
        str(args.task_limit),
        "--num-workers",
        str(len(args.gpus)),
    ]
    if args.dataset_root_override:
        base_cmd.extend(["--dataset-root-override", args.dataset_root_override])
    if args.no_skip_existing:
        base_cmd.append("--no-skip-existing")
    if args.no_colorize:
        base_cmd.append("--no-colorize")
    if args.write_videos:
        base_cmd.append("--write-videos")
    if args.keep_ply:
        base_cmd.append("--keep-ply")
    if args.fail_fast:
        base_cmd.append("--fail-fast")

    processes = []
    for worker_index, gpu in enumerate(args.gpus):
        cmd = [*base_cmd, "--worker-index", str(worker_index)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = log_dir / f"worker_{worker_index}_gpu_{gpu}.log"
        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append({"worker_index": worker_index, "gpu": gpu, "pid": proc.pid, "log": str(log_path), "process": proc})
        logger.info("Started worker=%d gpu=%s pid=%d log=%s", worker_index, gpu, proc.pid, log_path)

    _safe_write_json(
        output_root / "pids.json",
        [{k: v for k, v in row.items() if k != "process"} for row in processes],
    )

    if not args.wait:
        return 0

    codes = []
    for row in processes:
        code = row["process"].wait()
        codes.append(int(code))
        logger.info("Worker=%d exited with code=%d", row["worker_index"], code)
    return max(codes) if codes else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-json", default="tmp/syscon3d_release/mipnerf360_calibration_splits.json")
    parser.add_argument("--impossible-splits-json", default="tmp/syscon3d_release/mipnerf360_impossible_splits.json")
    parser.add_argument("--dataset-root-override", default="")
    parser.add_argument("--syscon3d-extra-data-root", default="")
    parser.add_argument("--output-root", default="output/syscon3d_project_assets")
    parser.add_argument("--tmp-dir", default="tmp/syscon3d_project_assets_tmp")
    parser.add_argument("--subset-sizes", nargs="+", type=int, default=list(DEFAULT_SUBSET_SIZES))
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=150_000)
    parser.add_argument("--video-max-points", type=int, default=30_000)
    parser.add_argument("--video-duration-s", type=float, default=3.0)
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--video-size", type=int, default=720)
    parser.add_argument("--video-background", default="#f7f3e8")
    parser.add_argument("--video-point-size", type=float, default=0.55)
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--keep-ply", action="store_true")
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--robust-vggt-rejection-threshold", type=float, default=0.4)
    parser.add_argument("--colorize", dest="colorize", action="store_true", default=True)
    parser.add_argument("--no-colorize", dest="no_colorize", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="no_skip_existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    args.colorize = not bool(args.no_colorize)
    args.skip_existing = not bool(args.no_skip_existing)
    return args


def main() -> None:
    args = _parse_args()
    if args.launch:
        raise SystemExit(_launch(args))
    raise SystemExit(_worker(args))


if __name__ == "__main__":
    main()
