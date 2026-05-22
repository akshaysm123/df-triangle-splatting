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
        (3, H, W) unit normals in camera space; border and invalid pixels are zero.
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    device = depth.device
    dtype = depth.dtype
    H, W = depth.shape

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    d = depth
    x = (u_grid - cx) * d / fx
    y = (v_grid - cy) * d / fy
    z = d
    points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # t_x = P(u+1, v) - P(u-1, v),  t_y = P(u, v+1) - P(u, v-1)
    t_x = points[:, 2:, :] - points[:, :-2, :]
    t_y = points[2:, :, :] - points[:-2, :, :]

    normals = torch.cross(t_x, t_y, dim=-1)
    normals = F.normalize(normals, dim=-1, eps=1e-8)

    # Interior patch of view directions for facing check
    view_dirs = points[1:-1, 1:-1, :]
    facing = (normals * view_dirs).sum(dim=-1, keepdim=True)
    normals = torch.where(facing > 0, -normals, normals)

    output = torch.zeros((H, W, 3), device=device, dtype=dtype)
    output[1:-1, 1:-1, :] = normals

    valid = (
        torch.isfinite(depth)
        & (depth > min_depth)
        & torch.isfinite(output).all(dim=-1)
    )
    output[~valid] = 0.0

    return output.permute(2, 0, 1)
