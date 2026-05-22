# Estimate camera-space surface normals from a depth map (PyTorch).

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _smooth_depth_bilateral(depth_hw: torch.Tensor) -> torch.Tensor:
    """Apply OpenCV bilateral filtering on CPU, then return tensor on original device."""
    device = depth_hw.device
    dtype = depth_hw.dtype
    d_np = depth_hw.detach().cpu().numpy().astype(np.float32)
    d_smooth = cv2.bilateralFilter(d_np, d=9, sigmaColor=0.1, sigmaSpace=5)
    return torch.from_numpy(d_smooth).to(device=device, dtype=dtype)


def depth_map_to_normals(
    depth: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float = 1e-6,
    fd_radius: int = 3,
    min_unnormalized_norm: float = 1e-4,
    smooth_depth: bool = True,
) -> torch.Tensor:
    """
    Backproject depth to a camera-space point grid and estimate normals via
    central differences and cross products.

    Args:
        depth: (1, H, W) or (H, W) depth in camera coordinates (z = depth).
        fx, fy, cx, cy: pinhole intrinsics in pixels.
        fd_radius: central-difference radius r (uses u±r and v±r neighbors).
        min_unnormalized_norm: |n| before normalize below this → NaN.
        smooth_depth: apply bilateral filter before backprojection.

    Returns:
        (3, H, W) unit normals in camera space. Border, invalid depth, and
        low-confidence estimates are NaN.
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    device = depth.device
    dtype = depth.dtype
    H, W = depth.shape
    r = fd_radius
    span = 2 * r

    output = torch.full((H, W, 3), float("nan"), device=device, dtype=dtype)

    if H < span + 1 or W < span + 1:
        return output.permute(2, 0, 1)

    depth = _smooth_depth_bilateral(depth) if smooth_depth else depth

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    d = depth
    x = (u_grid - cx) * d / fx
    y = (v_grid - cy) * d / fy
    z = d
    points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # t_x(u,v) = P(u+r,v) - P(u-r,v)  -> (H, W-span, 3)
    # t_y(u,v) = P(u,v+r) - P(u,v-r)  -> (H-span, W, 3)
    t_x = points[:, span:, :] - points[:, :-span, :]
    t_y = points[span:, :, :] - points[:-span, :, :]

    # Interior (H-span, W-span): center pixels with u,v in [r, H-r-1] x [r, W-r-1]
    t_x_mid = t_x[r:-r, :, :]
    t_y_mid = t_y[:, r:-r, :]

    normals_raw = torch.cross(t_x_mid, t_y_mid, dim=-1)
    norm_mag = normals_raw.norm(dim=-1)

    normals = normals_raw / norm_mag.clamp(min=1e-8).unsqueeze(-1)

    view_dirs = points[r:-r, r:-r, :]
    facing = (normals * view_dirs).sum(dim=-1, keepdim=True)
    normals = torch.where(facing > 0, -normals, normals)

    # interior_valid = norm_mag >= min_unnormalized_norm
    # normals_to_write = torch.where(
    #     interior_valid.unsqueeze(-1), normals, torch.full_like(normals, float("nan"))
    # )

    output[r:-r, r:-r, :] = normals

    valid_depth = torch.isfinite(depth) & (depth > min_depth)
    output[~valid_depth] = float("nan")

    return output.permute(2, 0, 1)
