# Tools for loading and resizing DepthAnything depth / confidence maps.

import os
import cv2
import numpy as np
import torch

_NPZ_ARRAY_KEYS = ("depth", "confidence", "data", "arr_0")


def load_npz_float32(path: str) -> np.ndarray:
    """Load a single float32 array from an .npz file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected map file not found: {path}")

    with np.load(path) as data:
        if len(data.files) == 1:
            arr = data[data.files[0]]
        else:
            arr = None
            for key in _NPZ_ARRAY_KEYS:
                if key in data:
                    arr = data[key]
                    break
            if arr is None:
                raise KeyError(
                    f"Could not find array in {path}. Keys: {list(data.files)}. "
                    f"Expected one array or one of {_NPZ_ARRAY_KEYS}."
                )

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D map in {path}, got shape {arr.shape}")
    return arr


def load_depth_confidence_maps(scene_path: str, image_stem: str):
    """
    Load depth and confidence .npz maps for an image stem (no extension).

    Layout: {scene_path}/depths/{stem}.npz and {scene_path}/confidence/{stem}.npz
    Returns (depth, confidence) as float32 HxW arrays, or (None, None) if depths/ is absent.
    """
    depth_dir = os.path.join(scene_path, "depths")
    if not os.path.isdir(depth_dir):
        return None, None

    depth_path = os.path.join(depth_dir, f"{image_stem}.npz")
    conf_path = os.path.join(scene_path, "confidence", f"{image_stem}.npz")
    depth = load_npz_float32(depth_path)
    confidence = load_npz_float32(conf_path)
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
