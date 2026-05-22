# Align predicted depth maps to COLMAP metric scale via least-squares scale factor.

import os
import numpy as np

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_next_bytes,
)


def read_points3d_id_to_xyz(sparse_dir: str) -> dict:
    """Return COLMAP point3D id -> world XYZ."""
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")

    if os.path.isfile(bin_path):
        points = {}
        with open(bin_path, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                props = read_next_bytes(fid, 43, "QdddBBBd")
                point_id = int(props[0])
                xyz = np.array(props[1:4], dtype=np.float64)
                track_length = read_next_bytes(fid, 8, "Q")[0]
                if track_length > 0:
                    read_next_bytes(
                        fid,
                        8 * track_length,
                        "ii" * track_length,
                    )
                points[point_id] = xyz
        return points

    if os.path.isfile(txt_path):
        points = {}
        with open(txt_path, "r") as fid:
            for line in fid:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                elems = line.split()
                point_id = int(elems[0])
                xyz = np.array(list(map(float, elems[1:4])), dtype=np.float64)
                points[point_id] = xyz
        return points

    raise FileNotFoundError(
        f"No COLMAP points3D file found in {sparse_dir}"
    )


def load_colmap_sparse(source_path: str):
    """Load COLMAP sparse/0 extrinsics, intrinsics, and point3D positions."""
    sparse_dir = os.path.join(source_path, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        return None

    try:
        extrinsics = read_extrinsics_binary(
            os.path.join(sparse_dir, "images.bin")
        )
        intrinsics = read_intrinsics_binary(
            os.path.join(sparse_dir, "cameras.bin")
        )
    except Exception:
        extrinsics = read_extrinsics_text(
            os.path.join(sparse_dir, "images.txt")
        )
        intrinsics = read_intrinsics_text(
            os.path.join(sparse_dir, "cameras.txt")
        )

    points3d = read_points3d_id_to_xyz(sparse_dir)
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "points3d": points3d,
    }


def find_colmap_image(extrinsics: dict, image_stem: str):
    """Match a camera image stem to a COLMAP Image entry."""
    for extr in extrinsics.values():
        stem = os.path.basename(extr.name).split(".")[0]
        if stem == image_stem:
            return extr
    return None


def colmap_depth_in_camera(X_world: np.ndarray, R_stored: np.ndarray, T: np.ndarray) -> float:
    """z in COLMAP camera frame; R_stored/T are the 3DGS-stored pose."""
    R_colmap = R_stored.T
    X_cam = R_colmap @ X_world + T
    return float(X_cam[2])


def estimate_depth_scale(
    depth_map: np.ndarray,
    R_stored: np.ndarray,
    T: np.ndarray,
    colmap_image,
    colmap_intr,
    points3d: dict,
    train_width: int,
    train_height: int,
    min_pairs: int = 10,
    min_depth: float = 1e-6,
) -> float:
    """
    Solve s* = sum(d_pred * z_colmap) / sum(d_pred^2) for COLMAP-visible points.

    depth_map: (H, W) at training resolution.
    COLMAP 2D observations are scaled from intrinsics resolution to train resolution.
    """
    colmap_w = float(colmap_intr.width)
    colmap_h = float(colmap_intr.height)
    sx = train_width / colmap_w
    sy = train_height / colmap_h

    d_preds = []
    z_colmaps = []

    for (u, v), point_id in zip(colmap_image.xys, colmap_image.point3D_ids):
        if point_id < 0:
            continue
        if point_id not in points3d:
            continue

        X = points3d[point_id]
        z = colmap_depth_in_camera(X, R_stored, T)
        if z <= min_depth or not np.isfinite(z):
            continue

        u_t = int(round(u * sx))
        v_t = int(round(v * sy))
        if u_t < 0 or v_t < 0 or u_t >= train_width or v_t >= train_height:
            continue

        d_pred = depth_map[v_t, u_t]
        if not np.isfinite(d_pred) or d_pred <= min_depth:
            continue

        d_preds.append(d_pred)
        z_colmaps.append(z)

    if len(d_preds) < min_pairs:
        return 1.0

    d = np.asarray(d_preds, dtype=np.float64)
    z = np.asarray(z_colmaps, dtype=np.float64)
    denom = np.dot(d, d)
    if denom <= min_depth:
        return 1.0
    return float(np.dot(d, z) / denom)


def align_depth_to_colmap(
    depth_map_hw: np.ndarray,
    cam_info,
    colmap_data: dict,
    train_width: int,
    train_height: int,
    min_pairs: int = 10,
) -> tuple:
    """
    Compute scale and return scaled depth map (H, W).
    Returns (scaled_depth, scale_factor).
    """
    if colmap_data is None:
        return depth_map_hw, 1.0

    extr = find_colmap_image(colmap_data["extrinsics"], cam_info.image_name)
    if extr is None:
        return depth_map_hw, 1.0

    intr = colmap_data["intrinsics"][extr.camera_id]
    scale = estimate_depth_scale(
        depth_map_hw,
        cam_info.R,
        cam_info.T,
        extr,
        intr,
        colmap_data["points3d"],
        train_width,
        train_height,
        min_pairs=min_pairs,
    )
    return depth_map_hw * scale, scale
