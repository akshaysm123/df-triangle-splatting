# Tools for loading and resizing DepthAnything depth / confidence maps.

import os
import cv2
import numpy as np
import torch


def _as_2d_float32(arr: np.ndarray, key: str, path: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D '{key}' in {path}, got shape {arr.shape}")
    return arr


def load_depth_confidence_maps(scene_path: str, image_stem: str):
    """
    Load depth and confidence from a single .npz per image.

    Layout: {scene_path}/frame_data/{stem}.npz with arrays "depth" and "confidence".
    Returns (depth, confidence) as float32 HxW arrays.
    """
    maps_dir = os.path.join(scene_path, "frame_data")
    if not os.path.isdir(maps_dir):
        raise FileNotFoundError(f"Expected depth/confidence directory not found: {maps_dir}")

    path = os.path.join(maps_dir, f"{image_stem}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected depth/confidence file not found: {path}")

    with np.load(path) as data:
        if "depth" not in data or "confidence" not in data:
            raise KeyError(
                f"Expected 'depth' and 'confidence' in {path}, found keys: {list(data.files)}"
            )
        depth = _as_2d_float32(data["depth"], "depth", path)
        confidence = _as_2d_float32(data["confidence"], "confidence", path)

    return depth, confidence


def resize_map_cv2(map_np: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize HxW float map to (width, height) with nearest-neighbor interpolation."""
    if map_np.shape[0] == height and map_np.shape[1] == width:
        return map_np.astype(np.float32, copy=False)
    return cv2.resize(
        map_np.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def prepare_map_for_camera(map_np: np.ndarray, orig_size, train_resolution):
    """
    Resize a depth/confidence map to match the RGB image pipeline:
      1) native map size -> original image size (orig_size)
      2) original image size -> training resolution (train_resolution)

    orig_size / train_resolution: (width, height) tuples, same as PIL image resize.
    Returns (1, H, W) float tensor or None.
    """
    if map_np is None:
        return None

    orig_w, orig_h = orig_size
    train_w, train_h = train_resolution

    map_np = resize_map_cv2(map_np, orig_w, orig_h)
    map_np = resize_map_cv2(map_np, train_w, train_h)
    return torch.from_numpy(map_np).unsqueeze(0)
