# Estimate camera-space surface normals from a depth map (PyTorch).

import torch
import torch.nn.functional as F


def depth_map_to_normals(
    depth: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float = 1e-6,
) -> torch.Tensor:
    """
    Backproject depth to a camera-space point grid and estimate normals via
    central differences and cross products.

    Args:
        depth: (1, H, W) or (H, W) depth in camera coordinates (z = depth).
        fx, fy, cx, cy: pinhole intrinsics in pixels.

    Returns:
        (3, H, W) unit normals in camera space. Border pixels and invalid depth
        are NaN (use torch.isfinite() to build masks for training losses).
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    device = depth.device
    dtype = depth.dtype
    H, W = depth.shape

    output = torch.full((H, W, 3), float("nan"), device=device, dtype=dtype)

    if H < 3 or W < 3:
        return output.permute(2, 0, 1)

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    d = depth
    x = (u_grid - cx) * d / fx
    y = (v_grid - cy) * d / fy
    z = d
    points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # t_x: (H, W-2, 3) along u;  t_y: (H-2, W, 3) along v
    t_x = points[:, 2:, :] - points[:, :-2, :]
    t_y = points[2:, :, :] - points[:-2, :, :]

    # Interior overlap (H-2, W-2): rows v=1..H-2, cols u=1..W-2
    t_x_mid = t_x[1:-1, :, :]
    t_y_mid = t_y[:, 1:-1, :]

    normals = torch.cross(t_x_mid, t_y_mid, dim=-1)
    normals = F.normalize(normals, dim=-1, eps=1e-8)

    view_dirs = points[1:-1, 1:-1, :]
    facing = (normals * view_dirs).sum(dim=-1, keepdim=True)
    normals = torch.where(facing > 0, -normals, normals)

    output[1:-1, 1:-1, :] = normals

    valid_depth = torch.isfinite(depth) & (depth > min_depth)
    invalid = ~valid_depth
    output[invalid] = float("nan")

    return output.permute(2, 0, 1)
