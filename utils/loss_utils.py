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

