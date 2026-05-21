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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from triangle_renderer import render
import torchvision
from matplotlib import cm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel


def apply_depth_colormap(depth, alpha, cmap_name="turbo"):
    """Map a single-channel depth map [1,H,W] to RGB using a matplotlib colormap."""
    d = depth[0].detach().float().cpu()
    a = alpha[0].detach().float().cpu()
    mask = a > 1e-3
    if mask.any():
        lo = torch.quantile(d[mask], 0.02)
        hi = torch.quantile(d[mask], 0.98)
    else:
        lo, hi = d.min(), d.max()
    if hi <= lo:
        hi = lo + 1e-6
    d_norm = ((d - lo) / (hi - lo)).clamp(0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(d_norm.numpy())
    rgb = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).float()
    rgb[:, ~mask] = 0.0
    return rgb


def render_set(model_path, name, iteration, views, triangles, pipeline, background, quick):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if quick:
        views = views[:10]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, triangles, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_set_extended(model_path, name, iteration, views, triangles, pipeline, background, quick):
    """Render RGB, ground truth, depth colormap, and per-triangle random-color views."""
    base = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(base, "renders")
    gts_path = os.path.join(base, "gt")
    depth_path = os.path.join(base, "depth")
    random_color_path = os.path.join(base, "random_color")

    for path in (render_path, gts_path, depth_path, random_color_path):
        makedirs(path, exist_ok=True)

    if quick:
        views = views[:10]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pkg = render(view, triangles, pipeline, background)
        rendering = pkg["render"]
        depth_vis = apply_depth_colormap(pkg["surf_depth"], pkg["rend_alpha"])
        random_color = pkg["rend_random_color"]
        gt = view.original_image[0:3, :, :]

        stem = '{0:05d}'.format(idx) + ".png"
        torchvision.utils.save_image(rendering, os.path.join(render_path, stem))
        torchvision.utils.save_image(gt, os.path.join(gts_path, stem))
        torchvision.utils.save_image(depth_vis, os.path.join(depth_path, stem))
        torchvision.utils.save_image(random_color, os.path.join(random_color_path, stem))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, quick : bool):
    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  init_size=None,
                  nb_points=None,
                  set_sigma=None,
                  no_dome=False,
                  load_iteration=args.iteration,
                  shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set_extended(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), triangles, pipeline, background, quick)

        if not skip_test:
             render_set_extended(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), triangles, pipeline, background, quick)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.quick)