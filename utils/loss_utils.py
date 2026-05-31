#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp



def equilateral_regularizer(triangles):

    nan_mask = torch.isnan(triangles).any(dim=(1, 2))
    if nan_mask.any():
        print("NaN detected in triangle(s):")

    v0 = triangles[:, 1, :] - triangles[:, 0, :]
    v1 = triangles[:, 2, :] - triangles[:, 0, :]
    cross = torch.cross(v0, v1, dim=1)
    area = 0.5 * torch.norm(cross, dim=1)

    return area


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def log_l1_loss(network_output, gt, eps=1e-6, normalize=False):
    """L1 loss in log space: mean(|log(pred) - log(gt)|). Inputs must be positive."""
    pred = network_output.clamp(min=eps)
    target = gt.clamp(min=eps)
    return torch.abs(torch.log(pred) - torch.log(target)).mean()


def _pearson_loss_batched(pred, target, weight=None, eps=1e-8):
    """
    1 minus weighted Pearson r per row.
    pred, target: (N, P). Optional weight: (N, P), zero excludes a pixel.
    """
    if weight is None:
        weight = torch.ones_like(pred)
    w = weight.clamp(min=0)
    w_sum = w.sum(dim=-1, keepdim=True).clamp(min=eps)

    pred_mean = (pred * w).sum(dim=-1, keepdim=True) / w_sum
    target_mean = (target * w).sum(dim=-1, keepdim=True) / w_sum
    pred_c = pred - pred_mean
    target_c = target - target_mean

    cov = (w * pred_c * target_c).sum(dim=-1)
    var_pred = (w * pred_c.pow(2)).sum(dim=-1)
    var_target = (w * target_c.pow(2)).sum(dim=-1)
    denom = torch.sqrt(var_pred) * torch.sqrt(var_target) + eps
    return 1.0 - cov / denom


def _pad_to_patch_grid(t, patch_size):
    h, w = t.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h))
    return t


def _patches_from_padded_hw(t, patch_size):
    hp, wp = t.shape
    nh, nw = hp // patch_size, wp // patch_size
    return (
        t.view(nh, patch_size, nw, patch_size)
        .permute(0, 2, 1, 3)
        .reshape(nh * nw, patch_size * patch_size)
    )


def _extract_patches_hw(t, patch_size):
    """Split (H, W) into non-overlapping patches -> (num_patches, patch_size**2)."""
    return _patches_from_padded_hw(_pad_to_patch_grid(t, patch_size), patch_size)


def pearson_correlation_loss(network_output, gt, eps=1e-8):
    """
    1 minus Pearson correlation between flattened pred and gt.
    Zero when perfectly linearly correlated; higher when less correlated.
    """
    pred = network_output.reshape(1, -1)
    target = gt.reshape(1, -1)
    return _pearson_loss_batched(pred, target, eps=eps).squeeze()


def pearson_correlation_loss_patches(
    network_output,
    gt,
    confidence=None,
    patch_size=32,
    min_depth=1e-6,
    min_valid_ratio=0.5,
    eps=1e-8,
):
    """
    Mean patch-wise Pearson loss (1 - r), with optional confidence weighting.
    pred, gt, confidence: (1, H, W) or (H, W). Invalid / low-confidence pixels
    are excluded per patch; patches with too few valid pixels are skipped.
    """
    pred = _squeeze_hw(network_output)
    target = _squeeze_hw(gt)

    valid = torch.isfinite(target) & torch.isfinite(pred) & (target > min_depth)
    if confidence is not None:
        confidence = _squeeze_hw(confidence)
        valid = valid & (confidence > 0)
        pixel_weight = confidence
    else:
        pixel_weight = torch.ones_like(pred)

    pred = torch.where(valid, pred, torch.zeros_like(pred))
    target = torch.where(valid, target, torch.zeros_like(target))
    pixel_weight = torch.where(valid, pixel_weight, torch.zeros_like(pixel_weight))

    pred = _pad_to_patch_grid(pred, patch_size)
    target = _pad_to_patch_grid(target, patch_size)
    pixel_weight = _pad_to_patch_grid(pixel_weight, patch_size)
    valid = _pad_to_patch_grid(valid.float(), patch_size) > 0.5

    pred_patches = _patches_from_padded_hw(pred, patch_size)
    target_patches = _patches_from_padded_hw(target, patch_size)
    weight_patches = _patches_from_padded_hw(pixel_weight, patch_size)
    valid_patches = _patches_from_padded_hw(valid.float(), patch_size) > 0.5

    patch_losses = _pearson_loss_batched(
        pred_patches, target_patches, weight=weight_patches, eps=eps
    )
    valid_ratio = valid_patches.float().mean(dim=-1)
    patch_valid = valid_ratio >= min_valid_ratio

    valid_count = valid_patches.float().sum(dim=-1).clamp(min=eps)
    patch_weight = (weight_patches * valid_patches.float()).sum(dim=-1) / valid_count

    return confidence_weighted_mean(patch_losses, patch_weight, patch_valid, eps=eps)


def depth_combined_loss(
    pred,
    gt,
    log_l1_weight=0.9,
    pearson_weight=0.1,
    patch_size=32,
    min_depth=1e-6,
    eps=1e-6,
):
    """
    Depth supervision: log_l1_weight * log-L1 + pearson_weight * patch Pearson.
    pred, gt: (1, H, W) or (H, W). No confidence weighting.
    """
    pred = _squeeze_hw(pred)
    gt = _squeeze_hw(gt)

    valid = (
        torch.isfinite(gt)
        & torch.isfinite(pred)
        & (gt > min_depth)
        & (pred > min_depth)
    )
    log_l1 = pred.new_zeros(())
    if valid.any():
        log_err = torch.abs(
            torch.log(pred.clamp(min=eps)) - torch.log(gt.clamp(min=eps))
        )
        log_l1 = log_err[valid].mean()

    pearson = pearson_correlation_loss_patches(
        pred, gt, confidence=None, patch_size=patch_size, min_depth=min_depth
    )
    return log_l1_weight * log_l1 + pearson_weight * pearson


def _squeeze_hw(t):
    if t is None:
        return None
    return t.squeeze(0) if t.dim() == 3 else t


def confidence_weighted_mean(values, weight, mask, eps=1e-8):
    """Mean of `values` over pixels where `mask` is true, weighted by `weight`."""
    m = mask.to(values.dtype)
    w = weight * m
    denom = w.sum()
    if denom <= eps:
        return values.new_zeros(())
    num = (values * w).sum()
    if not torch.isfinite(num):
        return values.new_zeros(())
    return num / denom


def depth_supervision_loss(pred, gt, confidence, min_depth=1e-6):
    """
    L1 depth error weighted by confidence and 1/gt (more weight on nearby geometry).
    pred, gt, confidence: (1, H, W) or (H, W).
    """
    pred = _squeeze_hw(pred)
    gt = _squeeze_hw(gt)
    confidence = _squeeze_hw(confidence)

    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > min_depth) & (confidence > 0)
    err = torch.where(valid, (pred - gt).abs(), torch.zeros_like(pred)) # L1
    inv_depth_weight = 1.0 / gt.clamp(min=min_depth) # inverse depth importance weighing
    pixel_weight = confidence * inv_depth_weight # confidence in [0,1]
    return confidence_weighted_mean(err, pixel_weight, valid)


def normal_supervision_loss(pred, gt, confidence):
    """
    Angular normal error (1 - dot product) for unit normals, confidence-weighted.
    pred, gt: (3, H, W). Invalid gt pixels (NaN) are excluded.
    """
    confidence = _squeeze_hw(confidence)
    valid = (
        torch.isfinite(gt).all(dim=0) # all(dim=0) for (H x W)
        & torch.isfinite(pred).all(dim=0)
        & (confidence > 0)
    )
    pred_u = F.normalize(pred, dim=0, eps=1e-8)
    gt_u = F.normalize(torch.nan_to_num(gt, nan=0.0), dim=0, eps=1e-8)
    dot = (pred_u * gt_u).sum(dim=0).clamp(-1.0, 1.0)
    err = torch.where(valid, 1.0 - dot, torch.zeros_like(dot))
    return confidence_weighted_mean(err, confidence, valid)


def camera_normals_to_world(normal_map, world_view_transform):
    """Map camera-space normals (3, H, W) to world space (same convention as renderer)."""
    return (
        normal_map.permute(1, 2, 0) @ world_view_transform[:3, :3].T
    ).permute(2, 0, 1)

def lp_loss(pred, target, p=0.7, eps=1e-6):
    """
    Computes Lp loss with 0 < p < 1.
    Args:
        pred: (N, C, H, W) predicted image
        target: (N, C, H, W) groundtruth image
        p: norm degree < 1
        eps: small constant for numerical stability
    """
    diff = torch.abs(pred - target) + eps
    loss = torch.pow(diff, p).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

