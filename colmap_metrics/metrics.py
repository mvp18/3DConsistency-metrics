import os
import struct
import logging
import numpy as np
import collections
import argparse
import json
import sqlite3
from typing import Dict, Optional, Tuple

# ==============================================================================
# PART 1: COLMAP BINARY READERS
# ==============================================================================

def read_array(path):
    """ Reads COLMAP dense binary arrays (.bin) into NumPy. """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        while num_delimiter < 3:
            byte = fid.read(1)
            if not byte:
                raise ValueError(f"Invalid COLMAP array header in {path}: expected 3 '&' delimiters.")
            if byte == b"&":
                num_delimiter += 1
        array = np.fromfile(fid, np.float32)

    expected_size = width * height * channels
    if array.size != expected_size:
        raise ValueError(
            f"Unexpected data size in {path}: expected {expected_size} float32 values, found {array.size}."
        )

    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
AttemptedImage = collections.namedtuple("AttemptedImage", ["name", "width", "height"])

logger = logging.getLogger(__name__)

def qvec2rotmat(qvec):
    """ Converts quaternion to rotation matrix. """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            # FIXED: Format <IdddddddI = Little-endian, Int(4), 7 Doubles(56), Int(4) = 64 bytes
            binary_image_properties = struct.unpack("<IdddddddI", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = fid.read(1)
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = fid.read(1)
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.read(num_points2D * 24)
            images[image_id] = BaseImage(id=image_id, qvec=qvec, tvec=tvec,
                                         camera_id=camera_id, name=image_name,
                                         xys=None, point3D_ids=None)
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            # FIXED: Format <QdddBBBd = Little-endian, ULong(8), 3 Doubles(24), 3 UChars(3), Double(8) = 43 bytes
            binary_point_line_properties = struct.unpack("<QdddBBBd", fid.read(43))
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_elems = struct.unpack("<" + "II" * track_length, fid.read(8 * track_length))
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                           error=error, image_ids=image_ids,
                                           point2D_idxs=point2D_idxs)
    return points3D


def normalize_image_name(name: str) -> str:
    return os.path.normpath(name).replace("\\", "/")


def read_attempted_images_database(path_to_database: str) -> Dict[str, AttemptedImage]:
    attempted_images = {}
    with sqlite3.connect(path_to_database) as conn:
        rows = conn.execute(
            """
            SELECT images.name, cameras.width, cameras.height
            FROM images
            JOIN cameras ON images.camera_id = cameras.camera_id
            ORDER BY images.name
            """
        )
        for name, width, height in rows:
            normalized_name = normalize_image_name(name)
            attempted_images[normalized_name] = AttemptedImage(
                name=normalized_name,
                width=int(width),
                height=int(height),
            )
    return attempted_images

# ==============================================================================
# PART 2: THE METRICS
# ==============================================================================

def infer_photometric_kind(
    depth: np.ndarray,
    photo: np.ndarray,
    rel_diff_threshold: float,
) -> Tuple[str, Dict[str, float]]:
    """Heuristically classify photometric map as depth or error/confidence."""
    valid_mask = (depth > 1e-5) & np.isfinite(depth) & np.isfinite(photo)
    if not np.any(valid_mask):
        return "error", {"valid_ratio": 0.0}

    depth_vals = depth[valid_mask]
    photo_vals = photo[valid_mask]

    photo_max = float(np.nanmax(photo_vals))
    photo_median = float(np.nanmedian(photo_vals))
    depth_median = float(np.nanmedian(depth_vals))

    rel_diff = np.abs(photo_vals - depth_vals) / np.maximum(depth_vals, 1e-6)
    median_rel_diff = float(np.nanmedian(rel_diff))
    ratio = photo_median / depth_median if depth_median > 0 else float("inf")

    ambiguous_small_range = photo_max <= 1.05 and depth_median <= 1.05

    if photo_max <= 1.05:
        kind = "error"
    elif median_rel_diff <= rel_diff_threshold and 0.25 <= ratio <= 4.0:
        kind = "depth"
    else:
        kind = "error"

    stats = {
        "photo_max": photo_max,
        "photo_median": photo_median,
        "depth_median": depth_median,
        "median_rel_diff": median_rel_diff,
        "ambiguous_small_range": float(ambiguous_small_range),
        "valid_ratio": float(np.count_nonzero(valid_mask) / valid_mask.size),
    }
    return kind, stats


def normalize_photometric_values(
    photo: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Normalize an error/confidence map to [0, 1] using a simple max-based heuristic."""
    values = photo[valid_mask] if np.any(valid_mask) else photo
    max_val = float(np.nanmax(values)) if values.size else 0.0
    scaled = photo.astype(np.float32, copy=False)
    if max_val > 1.05:
        scaled = scaled / 100.0
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled, max_val


def maybe_infer_effective_kind(
    depth: np.ndarray,
    photo: np.ndarray,
    photometric_kind: str,
    effective_kind: str,
    kind_inferred: bool,
    rel_diff_clip: float,
) -> Tuple[str, bool]:
    if kind_inferred:
        return effective_kind, True

    inferred_kind, stats = infer_photometric_kind(depth, photo, rel_diff_clip)
    if photometric_kind == "auto":
        effective_kind = inferred_kind
        logger.info(
            "Photometric map inferred as %s (median_rel_diff=%.4f, photo_max=%.3f).",
            inferred_kind,
            stats.get("median_rel_diff", float("nan")),
            stats.get("photo_max", float("nan")),
        )
        if stats.get("ambiguous_small_range", 0.0):
            logger.warning(
                "Photometric values and depths are both <= 1.0; auto inference may be ambiguous. "
                "Set --photometric-kind explicitly if needed."
            )
        if inferred_kind == "error":
            logger.info(
                "Auto mode treats photometric values as error (lower is better); "
                "use --photometric-kind confidence if needed."
            )
    else:
        # Heuristic can reliably flag depth-vs-non-depth confusion; it cannot tell error vs confidence.
        if inferred_kind == "depth" and photometric_kind in {"error", "confidence"}:
            logger.warning(
                "Photometric map looks like depth, but --photometric-kind=%s; metrics may be invalid.",
                photometric_kind,
            )
        if inferred_kind == "error" and photometric_kind == "depth":
            logger.warning(
                "Photometric map looks like error/confidence, but --photometric-kind=depth; "
                "metrics may be invalid."
            )
    return effective_kind, True


def compute_quality(
    depth: np.ndarray,
    photo: np.ndarray,
    valid_mask: np.ndarray,
    photometric_kind: str,
    rel_diff_clip: float,
) -> np.ndarray:
    """Return per-pixel quality in [0, 1] for valid pixels; invalid pixels are 0."""
    rel_diff_clip = max(rel_diff_clip, 1e-6)
    if photometric_kind == "depth":
        rel_diff = np.abs(photo - depth) / np.maximum(depth, 1e-6)
        quality = 1.0 - np.clip(rel_diff / rel_diff_clip, 0.0, 1.0)
    else:
        scaled, _ = normalize_photometric_values(photo, valid_mask)
        if photometric_kind == "confidence":
            quality = scaled
        else:
            quality = 1.0 - scaled
    return np.where(valid_mask, quality, 0.0)


def compute_pca_axes(points: np.ndarray) -> np.ndarray:
    """Compute the top-2 PCA axes for a set of 3D points."""
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")
    centered = points - np.mean(points, axis=0)
    _, svals, vh = np.linalg.svd(centered, full_matrices=False)
    if svals[1] < 1e-6:
        raise ValueError("Degenerate plane fit.")
    return vh[:2].T


def calculate_gpc_icm(
    dense_folder: str,
    photometric_kind: str = "auto",
    rel_diff_clip: float = 0.2,
) -> Tuple[float, float, float, float]:
    """Calculates GPC (Quality) and ICM (Mass)."""
    depth_folder = os.path.join(dense_folder, "stereo/depth_maps")
    if not os.path.exists(depth_folder):
        logger.error("Depth maps not found at %s", depth_folder)
        return 0, 0, 0, 0

    logger.info("Scanning %s...", depth_folder)

    gpc_scores = []
    density_scores = []
    consistency_scores = []
    # Accumulate quality "mass" and pixel counts so ICM can be normalized across resolutions.
    total_icm_mass = 0.0
    total_pixels = 0

    # Sort for deterministic inference/logging.
    files = sorted(f for f in os.listdir(depth_folder) if f.endswith(".geometric.bin"))
    logger.info("Found %d geometric depth maps.", len(files))

    effective_kind = photometric_kind
    kind_inferred = False

    for i, filename in enumerate(files):
        # Paths
        geom_path = os.path.join(depth_folder, filename)
        photo_path = os.path.join(depth_folder, filename.replace(".geometric.bin", ".photometric.bin"))

        # Read depth and photometric map.
        try:
            depth = read_array(geom_path)
            photo = read_array(photo_path)
        except Exception as e:
            logger.warning("Skipping %s due to read error: %s", filename, e)
            continue

        if depth.shape != photo.shape:
            logger.warning("Skipping %s due to shape mismatch: %s vs %s", filename, depth.shape, photo.shape)
            continue

        valid_mask = (depth > 1e-5) & np.isfinite(depth) & np.isfinite(photo)
        if not np.any(valid_mask):
            logger.warning("Skipping %s due to empty valid mask.", filename)
            continue

        effective_kind, kind_inferred = maybe_infer_effective_kind(
            depth,
            photo,
            photometric_kind,
            effective_kind,
            kind_inferred,
            rel_diff_clip,
        )

        if effective_kind in {"error", "confidence"}:
            _, photo_max = normalize_photometric_values(photo, valid_mask)
            if photo_max > 100.0:
                logger.warning(
                    "Photometric max %.2f suggests non-[0,1]/[0,100] scale; adjust normalization if needed.",
                    photo_max,
                )

        quality = compute_quality(depth, photo, valid_mask, effective_kind, rel_diff_clip)

        # Counts
        n_total = valid_mask.size
        n_valid = int(np.count_nonzero(valid_mask))
        total_pixels += n_total

        # GPC Components
        if n_valid == 0:
            dens = 0.0
            cons = 0.0
        else:
            dens = n_valid / n_total
            cons = float(np.mean(quality[valid_mask]))

        gpc_scores.append(dens * cons)
        density_scores.append(dens)
        consistency_scores.append(cons)

        # ICM Calculation (Integrated Consistency Mass).
        # Mass is the sum of QUALITY on valid surfaces; final reporting normalizes by total pixels
        # across all views for cross-resolution comparability.
        total_icm_mass += float(np.sum(quality[valid_mask]))

        if i % 20 == 0:
            logger.info("Processed %d/%d views...", i + 1, len(files))

    if not gpc_scores:
        return 0, 0, 0, 0

    # Cross-resolution comparable ICM: average quality mass per pixel across all views.
    avg_icm = (total_icm_mass / total_pixels) if total_pixels else 0.0

    return np.mean(gpc_scores), np.mean(density_scores), np.mean(consistency_scores), avg_icm


def resolve_database_path(
    database_path: Optional[str],
    sparse_folder: str,
    dense_folder: str,
) -> str:
    if database_path:
        return database_path

    try:
        common_root = os.path.commonpath([os.path.abspath(sparse_folder), os.path.abspath(dense_folder)])
    except ValueError:
        return "database.db"
    candidate = os.path.join(common_root, "database.db")
    if os.path.exists(candidate):
        return candidate
    return "database.db"


def calculate_gpc_icm_all(
    dense_folder: str,
    sparse_folder: str,
    database_path: Optional[str] = None,
    photometric_kind: str = "auto",
    rel_diff_clip: float = 0.2,
) -> Tuple[float, float, float]:
    """Calculates attempted-set GPC/ICM and sparse registration rate."""
    resolved_database_path = resolve_database_path(database_path, sparse_folder, dense_folder)
    if not os.path.exists(resolved_database_path):
        logger.error("COLMAP database not found at %s", resolved_database_path)
        return 0.0, 0.0, 0.0

    try:
        attempted_images = read_attempted_images_database(resolved_database_path)
    except sqlite3.Error as exc:
        logger.error("Failed to read attempted images from %s: %s", resolved_database_path, exc)
        return 0.0, 0.0, 0.0

    if not attempted_images:
        logger.warning("No attempted images found in %s.", resolved_database_path)
        return 0.0, 0.0, 0.0

    images_bin = os.path.join(sparse_folder, "images.bin")
    registered_names = set()
    if os.path.exists(images_bin):
        registered_images = read_images_binary(images_bin)
        registered_names = {normalize_image_name(img.name) for img in registered_images.values()}
    else:
        logger.warning("Sparse images.bin not found at %s; registration rate will be 0.", images_bin)

    attempted_count = len(attempted_images)
    registered_count = sum(1 for name in attempted_images if name in registered_names)
    registration_rate = registered_count / attempted_count if attempted_count else 0.0

    depth_folder = os.path.join(dense_folder, "stereo/depth_maps")
    if not os.path.exists(depth_folder):
        logger.error("Depth maps not found at %s", depth_folder)
        return 0.0, 0.0, registration_rate

    total_pixels = sum(img.width * img.height for img in attempted_images.values())
    if total_pixels <= 0:
        logger.warning("Attempted images have non-positive total pixel count; attempted-set metrics are 0.")
        return 0.0, 0.0, registration_rate

    logger.info(
        "Computing attempted-set metrics over %d images (%d registered, %.2f%% registration).",
        attempted_count,
        registered_count,
        registration_rate * 100.0,
    )

    gpc_scores = []
    total_icm_mass = 0.0
    effective_kind = photometric_kind
    kind_inferred = False

    for i, (image_name, attempted_image) in enumerate(sorted(attempted_images.items())):
        n_total = attempted_image.width * attempted_image.height
        if n_total <= 0:
            logger.warning("Counting %s as zero: non-positive attempted resolution %dx%d.", image_name, attempted_image.width, attempted_image.height)
            gpc_scores.append(0.0)
            continue

        if image_name not in registered_names:
            gpc_scores.append(0.0)
            continue

        dense_name = image_name.replace("/", os.sep)
        geom_path = os.path.join(depth_folder, f"{dense_name}.geometric.bin")
        photo_path = os.path.join(depth_folder, f"{dense_name}.photometric.bin")

        if not os.path.exists(geom_path) or not os.path.exists(photo_path):
            logger.warning("Counting %s as zero for attempted-set metrics: missing dense map(s).", image_name)
            gpc_scores.append(0.0)
            continue

        try:
            depth = read_array(geom_path)
            photo = read_array(photo_path)
        except Exception as exc:
            logger.warning("Counting %s as zero for attempted-set metrics due to read error: %s", image_name, exc)
            gpc_scores.append(0.0)
            continue

        if depth.shape != photo.shape:
            logger.warning(
                "Counting %s as zero for attempted-set metrics due to shape mismatch: %s vs %s",
                image_name,
                depth.shape,
                photo.shape,
            )
            gpc_scores.append(0.0)
            continue

        valid_mask = (depth > 1e-5) & np.isfinite(depth) & np.isfinite(photo)
        if not np.any(valid_mask):
            logger.warning("Counting %s as zero for attempted-set metrics due to empty valid mask.", image_name)
            gpc_scores.append(0.0)
            continue

        effective_kind, kind_inferred = maybe_infer_effective_kind(
            depth,
            photo,
            photometric_kind,
            effective_kind,
            kind_inferred,
            rel_diff_clip,
        )

        if effective_kind in {"error", "confidence"}:
            _, photo_max = normalize_photometric_values(photo, valid_mask)
            if photo_max > 100.0:
                logger.warning(
                    "Photometric max %.2f suggests non-[0,1]/[0,100] scale; adjust normalization if needed.",
                    photo_max,
                )

        dense_pixels = int(np.prod(depth.shape[:2]))
        if dense_pixels != n_total:
            logger.warning(
                "Attempted resolution for %s is %dx%d (%d px), but dense map has %d px; "
                "attempted-set metrics use the attempted resolution in the denominator.",
                image_name,
                attempted_image.width,
                attempted_image.height,
                n_total,
                dense_pixels,
            )

        quality = compute_quality(depth, photo, valid_mask, effective_kind, rel_diff_clip)
        n_valid = int(np.count_nonzero(valid_mask))
        dens = n_valid / n_total
        cons = float(np.mean(quality[valid_mask]))

        gpc_scores.append(dens * cons)
        total_icm_mass += float(np.sum(quality[valid_mask]))

        if i % 50 == 0:
            logger.info("Processed attempted-set metrics for %d/%d images...", i + 1, attempted_count)

    gpc_all = float(np.mean(gpc_scores)) if gpc_scores else 0.0
    icm_all = (total_icm_mass / total_pixels) if total_pixels else 0.0
    return gpc_all, icm_all, registration_rate

def calculate_coverage(sparse_folder: str, coverage_plane: str = "pca") -> float:
    """Calculates angular coverage relative to object center."""
    images_bin = os.path.join(sparse_folder, "images.bin")
    points_bin = os.path.join(sparse_folder, "points3D.bin")

    if not os.path.exists(images_bin) or not os.path.exists(points_bin):
        logger.error("Sparse model files missing in %s", sparse_folder)
        return 0.0

    logger.info("Reading Sparse Reconstruction...")
    images = read_images_binary(images_bin)
    points3D = read_points3D_binary(points_bin)

    # 1. Calculate Object Center (Centroid of sparse points)
    all_xyz = [p.xyz for p in points3D.values()]
    if not all_xyz:
        logger.warning("Sparse point cloud is empty; coverage is 0.")
        return 0.0
    # obj_center = np.mean(all_xyz, axis=0)
    obj_center = np.median(all_xyz, axis=0)
    logger.info("Object Center estimated at: %s", obj_center)

    # 2. Calculate Camera Centers
    centers = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        center = -R.T @ t
        centers.append(center)

    centers = np.array(centers)
    if centers.shape[0] < 2:
        logger.warning("Need at least 2 camera centers for coverage; got %d.", centers.shape[0])
        return 0.0

    # 3. Calculate Azimuths
    azimuths = []
    if coverage_plane == "pca":
        try:
            axes = compute_pca_axes(centers)
            coords = (centers - obj_center) @ axes
            for coord in coords:
                angle = np.degrees(np.arctan2(coord[0], coord[1]))
                if angle < 0:
                    angle += 360
                azimuths.append(angle)
        except ValueError as exc:
            logger.warning("PCA plane failed (%s); falling back to X-Z plane.", exc)
            coverage_plane = "xz"

    if coverage_plane != "pca":
        for cam_center in centers:
            vec = cam_center - obj_center
            angle = np.degrees(np.arctan2(vec[0], vec[2]))
            if angle < 0:
                angle += 360
            azimuths.append(angle)

    # 4. Find Max Gap
    azimuths.sort()
    gaps = []
    for i in range(len(azimuths) - 1):
        gaps.append(azimuths[i + 1] - azimuths[i])

    # Loop closure gap
    if azimuths:
        gaps.append((360 - azimuths[-1]) + azimuths[0])

    return 360 - max(gaps) if gaps else 0.0

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", default="dense", help="Path to dense folder")
    parser.add_argument("--sparse", default="sparse/0", help="Path to sparse/0 folder")
    parser.add_argument(
        "--photometric-kind",
        choices=["auto", "error", "confidence", "depth"],
        default="auto",
        help="How to interpret *.photometric.bin (auto|error|confidence|depth).",
    )
    parser.add_argument(
        "--relative-depth-clip",
        type=float,
        default=0.2,
        help="Relative depth diff clip when photometric map is depth (e.g., 0.2 = 20%%).",
    )
    parser.add_argument(
        "--coverage-plane",
        choices=["pca", "xz"],
        default="pca",
        help="Plane for angular coverage: PCA of cameras (pca) or world X-Z (xz).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="results",
        help="Name prefix for output files.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for result JSON.",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to COLMAP database.db used to determine attempted images and resolutions.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("==========================================")
    logger.info("      COLMAP GOLD STANDARD METRICS")
    logger.info("==========================================")

    # 1. GPC & ICM
    gpc, dens, cons, icm = calculate_gpc_icm(
        args.dense,
        photometric_kind=args.photometric_kind,
        rel_diff_clip=args.relative_depth_clip,
    )
    gpc_all, icm_all, registration_rate = calculate_gpc_icm_all(
        args.dense,
        args.sparse,
        database_path=args.database,
        photometric_kind=args.photometric_kind,
        rel_diff_clip=args.relative_depth_clip,
    )
    
    # 2. Coverage
    cov = calculate_coverage(args.sparse, coverage_plane=args.coverage_plane)

    # 3. Weighted Score (Penalize low coverage)
    # If coverage is < 360, score is reduced linearly.
    weighted_gpc = gpc * (cov / 360.0)

    logger.info("========================================")
    logger.info(" FINAL RESULTS (Normalized)")
    logger.info("========================================")
    logger.info("1. GPC Score (Quality):      %.5f (0-1)", gpc)
    logger.info("   - Avg Density:            %.5f", dens)
    logger.info("   - Avg Consistency:        %.5f", cons)
    logger.info("   - Weighted GPC:           %.5f (Quality * Coverage)", weighted_gpc)
    logger.info("2. ICM Score (Mass):         %.5f (Avg mass/pixel; cross-resolution comparable)", icm)
    logger.info("3. Angular Coverage:         %.1f degrees", cov)
    logger.info("4. Registration Rate:        %.5f", registration_rate)
    logger.info("5. GPC All (Attempted):      %.5f", gpc_all)
    logger.info("6. ICM All (Attempted):      %.5f", icm_all)
    logger.info("========================================")

    # 3. Save results to JSON
    results = {
        "gpc": gpc,
        "density": dens,
        "consistency": cons,
        "weighted_gpc": weighted_gpc,
        "icm": icm,
        "coverage": cov,
        "registration_rate": registration_rate,
        "gpc_all": gpc_all,
        "icm_all": icm_all,
    }
    filepath = os.path.join(args.outdir, f"{args.name}.json")
    os.makedirs(args.outdir, exist_ok=True)
    # with open(f"{args.name}.json", "w") as f:
    #     # Convert numpy types to native python floats for JSON serialization
    #     json.dump({k: float(v) for k, v in results.items()}, f, indent=4)
    with open(filepath, "w") as f:
        # Convert numpy types to native python floats for JSON serialization
        json.dump({k: float(v) for k, v in results.items()}, f, indent=4)

    logger.info("Results saved to %s", filepath)
